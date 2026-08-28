"""
Modified KneeAbnormalityModel that uses MedDINOv3 as backbone instead of ResNet50.
MedDINOv3 weights are frozen during training.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path
from typing import Optional, Tuple


class MedDINOv3Backbone(nn.Module):
    """
    Wrapper around MedDINOv3 ViT-B model.
    Extracts patch features from MedDINOv3 and optionally freezes them.
    """
    def __init__(self, checkpoint_path: str, freeze_weights: bool = True):
        super().__init__()
        
        # Load base ViT-B model
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.feat_dim = self.vit.embed_dim  # 768 for ViT-B
        
        # Load MedDINOv3 weights
        self._load_meddinov3_weights(checkpoint_path)
        
        # Freeze weights if requested
        if freeze_weights:
            for param in self.vit.parameters():
                param.requires_grad = False
    
    def _load_meddinov3_weights(self, checkpoint_path: str):
        """Load MedDINOv3 weights into ViT-B backbone."""
        if not os.path.exists(checkpoint_path):
            print(f"⚠ Warning: MedDINOv3 checkpoint not found at {checkpoint_path}")
            print(f"  Using random initialization instead")
            return
        
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # MedDINOv3 stores weights under 'teacher' key
        if 'teacher' in ckpt:
            state_dict = ckpt['teacher']
        else:
            state_dict = ckpt
        
        # Remove 'backbone.' prefix if present
        processed_dict = {}
        for key, val in state_dict.items():
            if key.startswith('backbone.'):
                new_key = key[len('backbone.'):]
                processed_dict[new_key] = val
            else:
                processed_dict[key] = val
        
        # Load into ViT
        missing, unexpected = self.vit.load_state_dict(processed_dict, strict=False)
        print(f"✓ MedDINOv3 loaded")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        if missing:
            print(f"    Examples: {list(missing)[:3]}")
        if unexpected:
            print(f"    Examples: {list(unexpected)[:3]}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MedDINOv3 backbone.
        
        Args:
            x: (B, 3, H, W) image tensor
        
        Returns:
            features: (B, N+1, 768) where N is number of patches
                     Includes CLS token as first position
        """
        # Use ViT's forward_features to get all patch tokens (including CLS)
        return self.vit.forward_features(x)  # (B, N+1, 768)


class MHSA(nn.Module):
    """Multi-head self-attention with optional key-padding mask."""
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads     = heads
        self.scale     = (dim // heads) ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        B, N, D = x.shape
        H = self.heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, N, H, D // H).transpose(1, 2)
        k = k.reshape(B, N, H, D // H).transpose(1, 2)
        v = v.reshape(B, N, H, D // H).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = torch.nan_to_num(attn.softmax(dim=-1), nan=0.0)
        attn = self.attn_drop(attn)
        out  = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""
    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MHSA(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class SliceTransformerEncoder(nn.Module):
    """Transformer encoder for processing patches or slices."""
    def __init__(self, dim, heads, num_layers, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, src_key_padding_mask=None):
        for block in self.blocks:
            x = block(x, key_padding_mask=src_key_padding_mask)
        return self.norm(x)


class MetadataModule(nn.Module):
    """Per-series metadata: fluid + fat + plane embeddings summed into one vector."""
    def __init__(self, num_planes: int, embed_dim: int):
        super().__init__()
        self.plane_embed = nn.Embedding(num_planes, embed_dim)
        self.fluid_embed = nn.Embedding(2, embed_dim)
        self.fat_embed   = nn.Embedding(2, embed_dim)
        for emb in (self.plane_embed, self.fluid_embed, self.fat_embed):
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, plane_id, fluid_sensitive, fat_suppression):
        return (self.fluid_embed(fluid_sensitive) +
                self.fat_embed  (fat_suppression) +
                self.plane_embed(plane_id))


class KneeAbnormalityModelWithMedDINOv3(nn.Module):
    """
    Modified architecture using MedDINOv3 instead of ResNet50.
    
    MedDINOv3 extracts 768-d patch features (vs ResNet50's 2048-d).
    MedDINOv3 weights are frozen during training.
    
    Pipeline:
    patches (B,S,P,3,224,224)
      → MedDINOv3 backbone (frozen, 768-d)
      → PatchTransformerEncoder  (patches of same slice attend to each other)
      → masked mean-pool over patches
      → slice_proj + pos_enc
      → SliceTransformerEncoder  (slices of same series attend to each other)
      → masked mean-pool over slices
      → fuse with metadata (fluid/fat/plane embeddings)
      → MLP → 12 logits
    """
    MAX_POS = 64

    def __init__(self, cfg, meddinov3_checkpoint: Optional[str] = None):
        super().__init__()
        self.cfg = cfg

        # Backbone: MedDINOv3 (frozen)
        if meddinov3_checkpoint is None:
            meddinov3_checkpoint = str(
                Path.home() / '.cache/huggingface/hub/models--ricklisz123--MedDINOv3-ViTB-16-CT-3M/snapshots/20d233b0ce1b1f93c46550787cf4b409a6f4a95e/model.pth'
            )
        
        self.backbone = MedDINOv3Backbone(meddinov3_checkpoint, freeze_weights=True)
        feat_dim = self.backbone.feat_dim  # 768

        # Patch encoder components
        self.patch_proj = nn.Linear(feat_dim, cfg.SLICE_DIM)
        self.patch_encoder = SliceTransformerEncoder(
            dim=cfg.SLICE_DIM, heads=cfg.PATCH_ENC_HEADS,
            num_layers=cfg.PATCH_ENC_LAYERS, dropout=0.0)
        
        # Slice encoder components
        self.slice_proj = nn.Identity()
        self.slice_pos  = nn.Parameter(
            torch.randn(1, self.MAX_POS, cfg.SLICE_DIM) * 0.02)
        self.slice_encoder = SliceTransformerEncoder(
            dim=cfg.SLICE_DIM, heads=cfg.SLICE_ENC_HEADS,
            num_layers=cfg.SLICE_ENC_LAYERS, dropout=0.0)

        # Metadata and classification
        self.meta_module = MetadataModule(cfg.NUM_PLANES, cfg.META_DIM)
        self.meta_fuse   = nn.Linear(cfg.SLICE_DIM + cfg.META_DIM, cfg.META_DIM)

        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.META_DIM), nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.META_DIM, 512), nn.GELU(),
            nn.Dropout(cfg.DROPOUT / 2), nn.Linear(512, cfg.NUM_LABELS))

    def train(self, mode: bool = True):
        """Override train to keep MedDINOv3 frozen."""
        super().train(mode)
        # Keep backbone in eval mode (frozen)
        self.backbone.eval()
        return self

    def forward(self, patches, slice_mask, patch_valid,
                plane_id, fluid_sensitive, fat_suppression):
        """
        Forward pass through the model.
        
        Args:
            patches: (B, S, P, 3, 224, 224) - batch, slices, patches, channels, height, width
            slice_mask: (B, S) bool - True = padding slice to ignore
            patch_valid: (B, S, P) bool - True = valid patch
            plane_id: (B,) - anatomical plane ID
            fluid_sensitive: (B,) - whether fluid-sensitive sequence
            fat_suppression: (B,) - whether fat-suppressed sequence
        """
        B, S, P, C, H, W = patches.shape

        # ① MedDINOv3 backbone → 768-d features per patch
        # Process all patches through backbone
        feats = self.backbone(patches.view(B*S*P, C, H, W))  # (B*S*P, N+1, 768)
        # Extract patch tokens only (exclude CLS token at position 0)
        feats = feats[:, 1:, :]  # (B*S*P, N, 768)
        # Take mean of patch tokens for each image
        feats = feats.mean(dim=1)  # (B*S*P, 768)
        feats = feats.view(B*S, P, -1)  # (B*S, P, 768)
        
        # Project to SLICE_DIM (512)
        feats = self.patch_proj(feats)  # (B*S, P, 512)

        # ② Patch transformer
        patch_pad_mask = (~patch_valid).reshape(B*S, P)  # (B*S, P)
        feats = self.patch_encoder(feats, src_key_padding_mask=patch_pad_mask)

        # ③ Masked mean-pool over patches
        pv = patch_valid.reshape(B*S, P).float()  # (B*S, P)
        pv = pv.unsqueeze(-1)
        slice_feats = (feats * pv).sum(1) / pv.sum(1).clamp(min=1)  # (B*S, 512)
        slice_feats = slice_feats.view(B, S, -1)  # (B, S, 512)

        # ④ Project + positional encoding
        slice_feats = self.slice_proj(slice_feats)
        slice_feats = slice_feats + self.slice_pos[:, :S, :]
        slice_feats = slice_feats.masked_fill(slice_mask.unsqueeze(-1), 0.0)

        # ⑤ Slice transformer
        encoded = self.slice_encoder(slice_feats,
                      src_key_padding_mask=slice_mask)

        # ⑥ Masked mean-pool over valid slices
        valid_f     = (~slice_mask).float().unsqueeze(-1)
        series_feat = (encoded * valid_f).sum(1) / valid_f.sum(1).clamp(1)

        # ⑦ Metadata
        meta = self.meta_module(plane_id, fluid_sensitive, fat_suppression)

        # ⑧ Fuse + classify
        cond = self.meta_fuse(torch.cat([series_feat, meta], dim=-1))
        return self.classifier(cond)
