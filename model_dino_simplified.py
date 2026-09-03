"""
Simplified KneeAbnormalityModel using Dino backbone.

Architecture:
- Dino (frozen) extracts features from high-res images using 14x14 patches
- Features are mean-pooled to get per-slice embedding
- Slice transformer processes multi-slice sequence
- Metadata fusion and classification head
"""

import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path
from typing import Optional
from torch.nn.utils.rnn import pad_sequence


class DinoBackbone(nn.Module):
    """
    Wrapper around Dino model.
    Extracts features from images using frozen pretrained weights.
    Uses 14x14 patches natively.
    """
    def __init__(self, checkpoint_path: str, freeze_weights: bool = True):
        super().__init__()
        
        # Load base ViT-B model
        self.vit = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)

        self.feat_dim = self.vit.embed_dim  # 768 for ViT-B

        # Freeze weights if requested
        if freeze_weights:
            for param in self.vit.parameters():
                param.requires_grad = False

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Dino backbone.
        
        Args:
            x: (B, 3, H, W) image tensor
        
        Returns:
            features: (B, N+1, 768) where N is number of patches
                     Includes CLS token as first position
        """
        # Use ViT's forward_features to get all patch tokens (including CLS)
        return self.vit.forward_features(x)  # (B, N+1, 768)

class GatedSliceAttention(nn.Module):

    def __init__(self, in_features=768, bottleneck_dim=128):
        super().__init__()
        # Tiny Gated Attention Mechanism (~100k parameters total)
        self.attn_v = nn.Sequential(
            nn.Linear(in_features, bottleneck_dim), nn.Tanh()
        )
        self.attn_u = nn.Sequential(
            nn.Linear(in_features, bottleneck_dim), nn.Sigmoid()
        )
        self.attn_score = nn.Linear(bottleneck_dim, 1)

    def forward(self, x, mask=None):
        """Args:

        x: Cached slice embeddings of shape (B, Slices, 768)
        mask: Boolean mask (B, Slices) where True = real slice, False = padded
          slice
        """
        # Element-wise gated multiplicative attention
        v = self.attn_v(x)  # (B, Slices, 128)
        u = self.attn_u(x)  # (B, Slices, 128)
        scores = self.attn_score(v * u).squeeze(-1)  # (B, Slices)

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, Slices, 1)

        # Weighted pooling over variable slice dimension
        return torch.sum(x * weights, dim=1)  # (B, 768)

class KneeAbnormalityModelWithDino(nn.Module):
    """
    Simplified RSNA Knee model using frozen Dino.
    
    Processes ALL series of a study simultaneously.
    
    Pipeline:
    All Slices of Study (padded to multiple of 14)
      → Dino (frozen) → per-slice feature
      → fuse with per-slice metadata (plane/fluid/fat)
      → StudyTransformerEncoder (attend across ALL slices of ALL series)
      → mean-pool over slices
      → MLP → 12 logits
    """
    MAX_SLICES = 128  # Increased to handle all slices across multiple series

    def __init__(self, cfg, dino_checkpoint: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        if cfg.TRAINING_LABEL is not None and cfg.TRAINING_LABEL >= 0:
            self.training_label=cfg.TRAINING_LABEL
        else:
            self.training_label=None


        num_labels = cfg.NUM_LABELS if hasattr(cfg, 'NUM_LABELS') else 12
        dropout = cfg.DROPOUT if hasattr(cfg, 'DROPOUT') else 0.3
        self.backbone = DinoBackbone(dino_checkpoint, freeze_weights=False)
        backbone_feat_dim = self.backbone.feat_dim  # 768
        self.conv= nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(backbone_feat_dim, 512, kernel_size=2),
                nn.AdaptiveAvgPool1d(2),
                nn.ReLU(),
                nn.Conv1d(512,256,kernel_size=1),
                nn.AdaptiveAvgPool1d(1),
                nn.ReLU()
            )
            for i in range(num_labels)
        ])
        self.gatedattn=GatedSliceAttention(768,128)
        self.all_feat_encoder_layer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=768 + 8,
                nhead=8,
                activation='gelu',
                dropout=0.2,
                batch_first=True)
            for i in range(num_labels)
        ])
        self.all_feat_encoder=nn.ModuleList([
            GatedSliceAttention(768+8,128)
            for i in range(num_labels)
        ])
        self.meta_proj = nn.ModuleList([
            nn.Linear(3, 8)
            for _ in range(num_labels)
        ])

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768+8, 1),
            )
            for _ in range(num_labels)
        ])

    def train(self, mode: bool = True):
        """Override train to keep backbone frozen and selectively configure heads."""
        super().train(mode)

        # 1. Keep backbone in eval mode and disable gradient computation
        frozen_modules = [
            self.backbone,
            # self.conv,
            # self.all_feat_encoder2,
            # self.meta_proj,
        ]

        for module in frozen_modules:
            module.eval()
            module.requires_grad_(False)


        # 2. Safely retrieve training_label if set
        selected_label = getattr(self, "training_label", None)
        frozen_labels = [
            self.heads,
            self.conv,
            self.all_feat_encoder_layer,
            self.all_feat_encoder,
            self.meta_proj,
        ]
        # 3. Set train/eval behavior and autograd per head
        for layer in frozen_labels:
            for i, head in enumerate(layer):
                is_active = selected_label is None or i == selected_label

                if is_active and mode:
                    head.train()
                    head.requires_grad_(True)
                else:
                    # Force inactive heads (or active heads during model.eval()) into eval mode
                    head.eval()
                    head.requires_grad_(False)
        return self

    @torch.no_grad()
    def extract_features(
            self, images, slice_mask, plane_ids, fluid_sensitive, fat_suppression
    ):
        B, Serie,S, C, H, W= images.shape
        flat_images = images.view(B * Serie * S, C, H, W)

        # 1. Micro-chunk the backbone pass to prevent 4GB+ VRAM allocation spikes
        CHUNK_SIZE = 32  # Process max 32 images at a time through Dino
        dino_tokens = []

        with torch.no_grad():
            for i in range(0, flat_images.size(0), CHUNK_SIZE):
                chunk = flat_images[i: i + CHUNK_SIZE]
                chunk_out = self.backbone(chunk)["x_norm_clstoken"]
                dino_tokens.append(chunk_out)
        dino_out = torch.cat(dino_tokens, dim=0)
        # with torch.no_grad():
        #     dino_out = self.backbone(images.view(B*Serie*S,C, H, W))["x_norm_clstoken"]
        # cls_token = self.feat_pos.expand(B*Serie, -1, -1).view(B,Serie,1, -1)
        dino_out = dino_out.view(B,Serie,S, 768)
        # feat_with_cls = torch.cat((cls_token, dino_out), dim=2).view(B*Serie, -1, 768).transpose(1, 2)
        return dino_out.view(B*Serie, -1, 768).transpose(1, 2),images.shape,slice_mask, plane_ids, fluid_sensitive, fat_suppression
    def forward_feats(self,input):
        features,imgshape,slice_mask, plane_ids, fluid_sensitive, fat_suppression=input
        B=len(imgshape)
        Serie,S, C, H, W= 16,8,3,518,518
        batchedfeats=features.view(B * Serie, 768, -1)
        convout=[x(batchedfeats) for x in self.conv]
        # featout=self.feat_encoder(feat_with_cls)[:, 0, :]
        convout2=[x.view(B,Serie, -1) for x in convout]
        meta_cat=torch.cat([plane_ids.unsqueeze(-1), fluid_sensitive.unsqueeze(-1),
                   fat_suppression.unsqueeze(-1)], dim=-1).view(B*Serie, -1)
        meta_cat=[x(meta_cat).view(B,Serie, -1) for x in self.meta_proj]
        meta_feat=[torch.cat([x,meta_cat[i]],-1) for i,x in enumerate(convout2)]

        meta_feat=[x.forward(meta_feat[i]).mean(1) for i,x in enumerate(self.all_feat_encoder)]
        return torch.cat([x(meta_feat[i]) for i,x in enumerate(self.heads)], dim=1)
    def forward(self, features, slice_mask, serie_mask, plane_ids, fluid_sensitive, fat_suppression):
        """
        Forward pass through the model.
        
        Args:
            images: (B, Total_S, 3, H, W) - batch, slices across ALL series
                   Images should be padded to nearest multiple of 14
            slice_mask: (B, Total_S) bool - True = padding slice
            plane_ids: (B, Total_S) - anatomical plane ID
            fluid_sensitive: (B, Total_S) - fluid-sensitive sequence
            fat_suppression: (B, Total_S) - fat-suppressed sequence
        
        Returns:
            logits: (B, 12) - predictions for 12 labels
        """
        B, Serie, S, _= features.shape
        attnout=self.gatedattn(features.view(B*Serie, S, 768),slice_mask.view(B*Serie,S)).view(B,Serie,768)

        meta_cat=torch.cat([plane_ids.unsqueeze(-1), fluid_sensitive.unsqueeze(-1), fat_suppression.unsqueeze(-1)], dim=-1).view(B*Serie, -1)
        meta_cat=[x(meta_cat).view(B,Serie, -1) for x in self.meta_proj]
        meta_feat=[torch.cat([attnout,x],-1) for i,x in enumerate(meta_cat)]

        meta_feat=[x.forward(meta_feat[i],serie_mask) for i,x in enumerate(self.all_feat_encoder)]
        return torch.cat([x(meta_feat[i]) for i,x in enumerate(self.heads)], dim=1)


def compute_valid_size(H: int, W: int, patch_size: int = 14) -> tuple:
    """
    Compute padded size for Dino (uses 14x14 patches).
    
    Args:
        H, W: Original image height and width
        patch_size: DINO patch size (14 for Dino)
    
    Returns:
        (padded_H, padded_W): Smallest size >= original satisfying size % patch_size == 0
    """
    pad_h = ((H + patch_size - 1) // patch_size) * patch_size
    pad_w = ((W + patch_size - 1) // patch_size) * patch_size
    return pad_h, pad_w


def pad_image(img: torch.Tensor, patch_size: int = 14) -> torch.Tensor:
    """
    Pad image to nearest multiple of patch_size.
    
    Args:
        img: (C, H, W), (B, C, H, W), or (B, S, C, H, W)
        patch_size: DINO patch size (14 for Dino)
    
    Returns:
        Padded image with same number of dimensions
    """
    if img.ndim == 3:
        C, H, W = img.shape
        pad_h, pad_w = compute_valid_size(H, W, patch_size)
        pad_top = (pad_h - H) // 2
        pad_bottom = pad_h - H - pad_top
        pad_left = (pad_w - W) // 2
        pad_right = pad_w - W - pad_left
        return F.pad(img.unsqueeze(0), (pad_left, pad_right, pad_top, pad_bottom), value=0.0).squeeze(0)
    elif img.ndim == 4:
        B, C, H, W = img.shape
        pad_h, pad_w = compute_valid_size(H, W, patch_size)
        pad_top = (pad_h - H) // 2
        pad_bottom = pad_h - H - pad_top
        pad_left = (pad_w - W) // 2
        pad_right = pad_w - W - pad_left
        return F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    elif img.ndim == 5:
        B, S, C, H, W = img.shape
        pad_h, pad_w = compute_valid_size(H, W, patch_size)
        pad_top = (pad_h - H) // 2
        pad_bottom = pad_h - H - pad_top
        pad_left = (pad_w - W) // 2
        pad_right = pad_w - W - pad_left
        # F.pad pads the last dimensions. (left, right, top, bottom) pads W and H
        return F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    else:
        raise ValueError(f"Unsupported image dimensions: {img.ndim}")

class KneeAbnormalityModelWithDinoMulti(nn.Module):
        """
        Simplified RSNA Knee model using frozen Dino.

        Processes ALL series of a study simultaneously.

        Pipeline:
        All Slices of Study (padded to multiple of 14)
          → Dino (frozen) → per-slice feature
          → fuse with per-slice metadata (plane/fluid/fat)
          → StudyTransformerEncoder (attend across ALL slices of ALL series)
          → mean-pool over slices
          → MLP → 12 logits
        """
        MAX_SLICES = 128  # Increased to handle all slices across multiple series

        def __init__(self, cfg, dino_checkpoint: Optional[str] = None):
            super().__init__()
            self.cfg = cfg
            if cfg.TRAINING_LABEL is not None and cfg.TRAINING_LABEL >= 0:
                self.training_label = cfg.TRAINING_LABEL
            else:
                self.training_label = None

            num_labels = cfg.NUM_LABELS if hasattr(cfg, 'NUM_LABELS') else 12
            dropout = cfg.DROPOUT if hasattr(cfg, 'DROPOUT') else 0.3
            self.gatedattn = GatedSliceAttention(768, 256)
            self.gatedattn_acl = nn.LSTM(768,128,2,batch_first=True,dropout=0.2,bidirectional=True)#GatedSliceAttention(768, 256)
            self.encoder_acl = nn.LSTM(264,128,2,batch_first=True,dropout=0.2,bidirectional=True)#GatedSliceAttention(256 + 8, 256)
            self.all_feat_encoder = GatedSliceAttention(768 + 8, 256)
            self.meta_proj = nn.Linear(4, 8)

            self.heads = nn.Linear(768 + 8, num_labels-4)
            self.heads_acl = nn.Linear(256, 1)
            self.heads_mcl = nn.Linear(768 + 8, 1)
            self.heads_mm = nn.Linear(768 + 8, 1)
            self.heads_lm = nn.Linear(768 + 8, 1)
            # self.heads = nn.Linear(3072 + 8, num_labels)

        def train(self, mode: bool = True):
            """Override train to keep backbone frozen and selectively configure heads."""
            super().train(mode)
            return self

        def mask_filter(self,targetsarray,plane_crop_mixed,serie_mask):
            dev = plane_crop_mixed.device
            targets = torch.tensor(targetsarray, device=dev)

            # plane_crop_mixed: (64, 38, 1, 2) vs acltargets: (6, 2) -> (64, 38, 6, 2)
            mask = (plane_crop_mixed.unsqueeze(2) == targets).all(dim=-1).any(dim=-1)
            # dev = plane_crop_mixed.device
            # col0_valid = torch.isin(plane_crop_mixed[..., 0], torch.tensor([0, 1, 2], device=dev))
            # col1_valid = torch.isin(plane_crop_mixed[..., 1], torch.tensor([2, 5], device=dev))
            #
            # mask = col0_valid & col1_valid  # Output shape: (64, 38)
            return serie_mask & mask
        def lstm_forward(self,lstm,input,mask):
            attnout_acl = []#.view(B * Serie, S,4, 768)[:,:,-1,:]
            LSTMChunk=8
            attnouts,_=lstm(input)#[]
            # for i in range(featurestofeed.shape[0]//LSTMChunk):
            #     chunk=featurestofeed[i*LSTMChunk:(i+1)*LSTMChunk]
            #     chunkout,_=self.gatedattn_acl(chunk)
            #     attnouts.append(chunkout)
            # 3. Calculate sequence lengths by summing the booleans
            # 3. Calculate sequence lengths
            # attnouts=torch.cat(attnouts,dim=0)
            lengths = mask.sum(dim=1)
            #
            # # 4. Clamp indices to valid bounds [0, seq_len - 1]
            max_seq_len = attnouts.size(1)
            last_valid_indices = (lengths - 1).clamp(min=0, max=max_seq_len - 1)
            #
            # # 5. Gather last outputs safely
            indices_expanded = last_valid_indices.view(-1, 1, 1).expand(
                -1, 1, attnouts.size(-1)
            )
            return attnouts.gather(1, indices_expanded).squeeze(1)
        def forward(self, features, slice_mask, serie_mask, plane_ids, fluid_sensitive, fat_suppression, plane_crop_mixed):
            """
            Forward pass through the model.

            Args:
                images: (B, Total_S, 3, H, W) - batch, slices across ALL series
                       Images should be padded to nearest multiple of 14
                slice_mask: (B, Total_S) bool - True = padding slice
                plane_ids: (B, Total_S) - anatomical plane ID
                fluid_sensitive: (B, Total_S) - fluid-sensitive sequence
                fat_suppression: (B, Total_S) - fat-suppressed sequence



                {'Sagittal': 0, 'Axial': 1, 'Coronal': 2}
                0: "whole",
                1: "anterior",
                2: "medial",
                3: "lateral",
                4: "back",
                5: "front"
            Returns:
                logits: (B, 12) - predictions for 12 labels
            """
            B, Serie, S, _ = features.shape
            featurestofeed=features.view(B * Serie, S, 768)
            attnout = self.gatedattn(featurestofeed,slice_mask.view(B*Serie,S)).view(B,Serie,-1)#.view(B * Serie, S,4, 768)[:,:,-1,:]
            last_outputs=self.lstm_forward(self.gatedattn_acl,featurestofeed,slice_mask.view(B*Serie,S)).view(B,Serie,-1)
            meta_cat = torch.cat(
                [plane_ids.unsqueeze(-1), fluid_sensitive.unsqueeze(-1), fat_suppression.unsqueeze(-1),plane_crop_mixed[:,:,1].unsqueeze(-1)], dim=-1).view(
                B * Serie, -1)
            meta_cat = self.meta_proj(meta_cat).view(B, Serie, -1)
            meta_feat =torch.cat([attnout, meta_cat], -1)
            meta_feat_acl =torch.cat([last_outputs, meta_cat], -1)
            aclmask=self.mask_filter([[0, 2], [1, 2], [2, 2]],plane_crop_mixed,serie_mask)
            mclmask=self.mask_filter([[1, 3], [1, 5], [2, 3], [2, 5]],plane_crop_mixed,serie_mask)
            mmmask=self.mask_filter([[1, 3], [2, 3],[1, 2], [2, 2]],plane_crop_mixed,serie_mask)
            lmmask=self.mask_filter([[1, 4], [2, 4], [1, 2], [2, 2]],plane_crop_mixed,serie_mask)
            valid_rows = [meta_feat_acl[i, aclmask[i]] for i in range(meta_feat_acl.size(0))]
            mask_acl = [meta_feat_acl[i, aclmask[i]] for i in range(meta_feat_acl.size(0))]
            filtered_features = pad_sequence(valid_rows, batch_first=True, padding_value=0.0)
            mask_acl = []
            for row in valid_rows:
                mask_acl.append(torch.cat((torch.ones(row.shape[0],dtype=torch.bool),torch.zeros(filtered_features.shape[1]-row.shape[0],dtype=bool)), dim=0))
            acl_meta_feat = self.lstm_forward(self.encoder_acl,filtered_features,aclmask)#self.encoder_acl(meta_feat_acl, aclmask)
            mcl_meta_feat = self.all_feat_encoder.forward(meta_feat, mclmask)
            mm_meta_feat = self.all_feat_encoder.forward(meta_feat, mmmask)
            lm_meta_feat = self.all_feat_encoder.forward(meta_feat, lmmask)
            meta_feat = self.all_feat_encoder.forward(meta_feat, serie_mask)
            return torch.cat((self.heads_acl(acl_meta_feat),self.heads_mcl(mcl_meta_feat),self.heads_mm(mm_meta_feat),self.heads_lm(lm_meta_feat),self.heads(meta_feat)),dim=1)
class AttentionPooling(nn.Module):
    """Attention-based pooling over variable slice/series dimensions."""

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        # x: [B, N, Feature_Dim], mask: [B, N] (True for valid, False for padding)
        scores = self.attn(x).squeeze(-1)  # [B, N]

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e4)

        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, N, 1]
        return torch.sum(x * weights, dim=1)  # [B, Feature_Dim]
class SerieTypeNN(nn.Module):
    def __init__(self, stype):
        super().__init__()
        self.stype = stype
        self.slice_pool = AttentionPooling(in_dim=1024, hidden_dim=128)
        self.slice_conved_pool = AttentionPooling(in_dim=1024, hidden_dim=128)
        conv_kernel_size=7
        self.conv=nn.Sequential(
            nn.Conv1d(1024, 1024, conv_kernel_size,padding=conv_kernel_size // 2),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.2),
        )
    def forward(self, feat_flat: torch.Tensor, mask: torch.Tensor, serie_type: torch.Tensor,) -> torch.Tensor:

        # mask_mask=((serie_type[:,0]==1) & (serie_type[:,1]==1)).unsqueeze(-1).repeat(1,mask.shape[-1])
        # if self.stype == 0:
        #     mask_mask=~mask_mask
        s_mask_flat = mask #& mask_mask
        series_features = self.slice_pool(feat_flat, mask=s_mask_flat)
        series_conved=self.conv(feat_flat.transpose(1, 2)).transpose(1, 2)

        if series_conved.shape[1] != s_mask_flat.shape[1]:
            pass
        series_conved_features=self.slice_conved_pool(series_conved, mask=s_mask_flat)
        return torch.cat((series_features,series_conved_features),dim=1)

class KneeAbnormalityModelRefactored(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        num_labels = getattr(cfg, "NUM_LABELS", 12)

        # 1. Categorical Metadata Embeddings (Fixes Linear Scalar Bug)
        self.plane_embed = nn.Embedding(4, 8)  # Sagittal, Axial, Coronal
        self.crop_embed = nn.Embedding(
            7, 8
        )  # front, back, anterior, medial, lateral, whole
        self.fluid_embed = nn.Embedding(2, 4)
        self.fat_embed = nn.Embedding(2, 4)
        self.type_embed=nn.Embedding(7, 8)
        meta_dim = 8 + 8 + 4 + 4  # Total: 24
        self.inputdropout=nn.Dropout(0.2)
        # 2. Backbone Feature Projection & Attention Pools
        num_types=1
        self.feature_proj = nn.Sequential(nn.Linear(1024*2*num_types + meta_dim, 512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,256))
        # Slice-level pooling per series
        self.serie_proj = nn.ModuleList([SerieTypeNN(i) for i in range(num_types)])
        # Pathology-Specific Series Aggregators
        self.acl_pool = AttentionPooling(in_dim=256, hidden_dim=64)
        self.mcl_pool = AttentionPooling(in_dim=256, hidden_dim=64)
        self.mm_pool = AttentionPooling(in_dim=256, hidden_dim=64)
        self.lm_pool = AttentionPooling(in_dim=256, hidden_dim=64)
        self.moa_pool = AttentionPooling(in_dim=256, hidden_dim=64)
        self.loa_pool = AttentionPooling(in_dim=256, hidden_dim=64)
        self.pfoa_pool = AttentionPooling(in_dim=256, hidden_dim=64)
        self.effusion_pool = AttentionPooling(in_dim=256, hidden_dim=64)
        self.synovitis_pool = AttentionPooling(in_dim=256, hidden_dim=64)
        self.baker_pool = AttentionPooling(in_dim=256, hidden_dim=64)
        self.contusion_pool = AttentionPooling(in_dim=256, hidden_dim=64)
        self.fracture_pool = AttentionPooling(in_dim=256, hidden_dim=64)
        self.global_pool = AttentionPooling(in_dim=256, hidden_dim=64)

        # Classification Heads
        self.head_acl = nn.Sequential(nn.Dropout(0.2),nn.Linear(256, 1))
        self.head_mcl = nn.Sequential(nn.Dropout(0.2),nn.Linear(256, 1))
        self.head_mm = nn.Sequential(nn.Dropout(0.2),nn.Linear(256, 1))
        self.head_lm = nn.Sequential(nn.Dropout(0.2),nn.Linear(256, 1))
        self.head_moa = nn.Sequential(nn.Dropout(0.2),nn.Linear(256, 1))
        self.head_loa = nn.Sequential(nn.Dropout(0.2),nn.Linear(256, 1))
        self.head_pfoa = nn.Sequential(nn.Dropout(0.2),nn.Linear(256, 1))
        self.head_effusion = nn.Sequential(nn.Dropout(0.2),nn.Linear(256, 1))
        self.head_synovitis = nn.Sequential(nn.Dropout(0.2),nn.Linear(256, 1))
        self.head_baker = nn.Sequential(nn.Dropout(0.2),nn.Linear(256, 1))
        self.head_contusion = nn.Sequential(nn.Dropout(0.2),nn.Linear(256, 1))
        self.head_fracture = nn.Sequential(nn.Dropout(0.2),nn.Linear(256, 1))
        self.head_general = nn.Linear(256, num_labels)

    def forward(
        self,
        features,
        slice_mask,
        serie_mask,
        plane_ids,
        fluid_sensitive,
        fat_suppression,
        plane_crop_mixed,
        serie_type
    ):
        """features: [B, Serie, Slices, 768]

        slice_mask: [B, Serie, Slices] (True for valid) serie_mask: [B, Serie]
        (True for valid)
        """
        B, Serie, S, C = features.shape

        # 1. Pool Slices per Series using Attention -> [B * Serie, 768]
        feat_flat = self.inputdropout(features.view(B * Serie, S, C))
        s_mask_flat = slice_mask.view(B * Serie, S)
        series_features=[x(feat_flat,s_mask_flat,torch.cat([fluid_sensitive.unsqueeze(-1),fat_suppression.unsqueeze(-1)],dim=-1).view(B*Serie,2)).view(B , Serie, C*2) for x in self.serie_proj]
        # 2. Embed Metadata Properly
        p_emb = self.plane_embed(plane_ids.clamp(0, 3))
        c_emb = self.crop_embed(plane_crop_mixed[:, :, 1].clamp(0, 6))
        fl_emb = self.fluid_embed(fluid_sensitive.long().clamp(0, 1))
        ft_emb = self.fat_embed(fat_suppression.long().clamp(0, 1))
        # tp_emb = self.type_embed(serie_type.long().clamp(0,6))
        series_features_combined = torch.cat(series_features,dim=-1)
        meta_feats = torch.cat([p_emb, c_emb, fl_emb, ft_emb], dim=-1)

        # 3. Concatenate Features + Metadata & Project -> [B, Serie, 256]
        combined = torch.cat([series_features_combined,meta_feats], dim=-1)
        series_embeddings = F.relu(self.feature_proj(combined))
        acl_mask = (
            ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 1))
        )
        acl_series_mask = serie_mask & acl_mask
        # 4. Pathology Specific Attention Aggregation over Series
        acl_feat = self.acl_pool(series_embeddings, mask=acl_series_mask)

        mcl_mask = (
                ((plane_ids[:]==1) | (plane_ids[:]==2)) &
                ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 2))
        )
        mcl_series_mask = serie_mask & mcl_mask
        mcl_feat = self.mcl_pool(series_embeddings, mask=mcl_series_mask)

        mm_mask = (
                ((plane_ids==1) | (plane_ids==2)) &
                ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 2))
        )
        mm_series_mask = serie_mask & mm_mask
        mm_feat = self.mm_pool(series_embeddings, mask=mm_series_mask)
        lm_mask = (
                ((plane_ids==1) | (plane_ids==2)) &
                ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 3))
        )
        lm_series_mask = serie_mask & lm_mask
        lm_feat = self.lm_pool(series_embeddings, mask=lm_series_mask)
        moa_mask = (
                ((plane_ids==1) | (plane_ids==2)) &
                ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 2))
        )
        moa_series_mask = serie_mask & moa_mask
        moa_feat = self.moa_pool(series_embeddings, mask=moa_series_mask)
        loa_mask = (
                ((plane_ids==1) | (plane_ids==2)) &
                ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 3))
        )
        loa_series_mask = serie_mask & loa_mask
        loa_feat = self.loa_pool(series_embeddings, mask=loa_series_mask)
        pfoa_mask = (
                ((plane_ids==0) | (plane_ids==1)) &
                ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 5) | (plane_crop_mixed[:, :, 1] == 1))
        )
        pfoa_series_mask = serie_mask & pfoa_mask
        pfoa_feat = self.pfoa_pool(series_embeddings, mask=pfoa_series_mask)
        effusion_mask = (
                ((plane_ids==0) | (plane_ids==1)) &
                ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 1) | (plane_crop_mixed[:, :, 1] == 5))
        )
        effusion_series_mask = serie_mask & effusion_mask
        effusion_feat = self.effusion_pool(series_embeddings, mask=effusion_series_mask)
        synovitis_mask = (
                ((plane_ids==0) | (plane_ids==1)) &
                ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 5) | (plane_crop_mixed[:, :, 1] == 1))
        )
        synovitis_series_mask = serie_mask & synovitis_mask
        synovitis_feat = self.synovitis_pool(series_embeddings, mask=synovitis_series_mask)
        baker_mask = (
                ((plane_ids==0) | (plane_ids==1)) &
                ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 4))
        )
        baker_series_mask = serie_mask & baker_mask
        baker_feat = self.baker_pool(series_embeddings, mask=baker_series_mask)
        contusion_mask = (
            ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 2) | (plane_crop_mixed[:, :, 1] == 3) | (plane_crop_mixed[:, :, 1] == 1))
        )
        contusion_series_mask = serie_mask & contusion_mask
        contusion_feat = self.contusion_pool(series_embeddings, mask=contusion_series_mask)
        fracture_mask = (
            ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 2) | (plane_crop_mixed[:, :, 1] == 3))
        )
        fracture_series_mask = serie_mask & fracture_mask
        fracture_feat = self.fracture_pool(series_embeddings, mask=fracture_series_mask)

        gen_feat = self.global_pool(series_embeddings, mask=serie_mask)

        # 5. Output Predictions
        out_acl = self.head_acl(acl_feat)
        out_mcl = self.head_mcl(mcl_feat)
        out_mm = self.head_mm(mm_feat)
        out_lm = self.head_lm(lm_feat)
        out_moa = self.head_moa(moa_feat)
        out_loa = self.head_loa(loa_feat)
        out_pfoa = self.head_pfoa(pfoa_feat)
        out_effusion = self.head_effusion(effusion_feat)
        out_synovitis = self.head_synovitis(synovitis_feat)
        out_baker = self.head_baker(baker_feat)
        out_contusion = self.head_contusion(contusion_feat)
        out_fracture = self.head_fracture(fracture_feat)
        out_gen = self.head_general(gen_feat)
        prediction_cat=torch.cat(
            [out_acl, out_mcl, out_mm, out_lm, out_moa, out_loa, out_pfoa, out_effusion, out_synovitis, out_baker,
             out_contusion, out_fracture], dim=-1
        )
        wgen_cat=torch.cat([out_gen.unsqueeze(2), prediction_cat.unsqueeze(2)], dim=-1)
        output=wgen_cat.amin(dim=-1)
        return output
if __name__ == '__main__':
    DinoBackbone(str(pathlib.Path.home()/".cache/huggingface/hub/models--microsoft--rad-dino/snapshots/110cbc18d5133582e320b43d53bf5c44e410c936/backbone_compatible.safetensors"))
    pass