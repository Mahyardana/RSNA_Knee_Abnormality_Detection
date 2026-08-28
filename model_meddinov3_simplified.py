"""
Simplified KneeAbnormalityModel using Dinov2 backbone.

Architecture:
- Dinov2 (frozen) extracts features from high-res images using 14x14 patches
- Features are mean-pooled to get per-slice embedding
- Slice transformer processes multi-slice sequence
- Metadata fusion and classification head
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path
from typing import Optional


class Dinov2Backbone(nn.Module):
    """
    Wrapper around Dinov2 ViT-B model.
    Extracts features from images using frozen pretrained weights.
    Uses 14x14 patches natively.
    """
    def __init__(self, checkpoint_path: str, freeze_weights: bool = True):
        super().__init__()
        
        # Load base ViT-B model
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.feat_dim = self.vit.embed_dim  # 768 for ViT-B
        
        # Load Dinov2 weights
        self._load_dinov2_weights(checkpoint_path)
        
        # Freeze weights if requested
        if freeze_weights:
            for param in self.vit.parameters():
                param.requires_grad = False
    
    def _load_dinov2_weights(self, checkpoint_path: str):
        """Load Dinov2 weights into ViT-B backbone."""
        if not os.path.exists(checkpoint_path):
            print(f"⚠ Warning: Dinov2 checkpoint not found at {checkpoint_path}")
            print(f"  Using random initialization instead")
            return
        
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        baseckpt=self.vit.state_dict()
        # Dinov2 stores weights under 'teacher' key
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
        print(f"✓ Dinov2 loaded (frozen)")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Dinov2 backbone.
        
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
    """Transformer encoder for processing slices."""
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
    """Per-slice metadata: fluid + fat + plane embeddings summed into one vector."""
    def __init__(self, num_planes: int, embed_dim: int):
        super().__init__()
        self.plane_embed = nn.Embedding(num_planes, embed_dim)
        self.fluid_embed = nn.Embedding(2, embed_dim)
        self.fat_embed   = nn.Embedding(2, embed_dim)
        for emb in (self.plane_embed, self.fluid_embed, self.fat_embed):
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, plane_id, fluid_sensitive, fat_suppression):
        # inputs can be (B, S)
        return (self.fluid_embed(fluid_sensitive) +
                self.fat_embed  (fat_suppression) +
                self.plane_embed(plane_id))


class KneeAbnormalityModelWithDinov2(nn.Module):
    """
    Simplified RSNA Knee model using frozen Dinov2.
    
    Processes ALL series of a study simultaneously.
    
    Pipeline:
    All Slices of Study (padded to multiple of 14)
      → Dinov2 (frozen) → per-slice feature
      → fuse with per-slice metadata (plane/fluid/fat)
      → StudyTransformerEncoder (attend across ALL slices of ALL series)
      → mean-pool over slices
      → MLP → 12 logits
    """
    MAX_SLICES = 128  # Increased to handle all slices across multiple series

    def __init__(self, cfg, dinov2_checkpoint: Optional[str] = None):
        super().__init__()
        self.cfg = cfg

        self.backbone = Dinov2Backbone(dinov2_checkpoint, freeze_weights=True)
        backbone_feat_dim = self.backbone.feat_dim  # 768
        
        slice_dim = cfg.SLICE_DIM if hasattr(cfg, 'SLICE_DIM') else 512
        meta_dim = cfg.META_DIM if hasattr(cfg, 'META_DIM') else 256
        num_planes = cfg.NUM_PLANES if hasattr(cfg, 'NUM_PLANES') else 3
        
        self.slice_proj = nn.Linear(backbone_feat_dim, slice_dim)
        
        self.slice_pos = nn.Parameter(
            torch.randn(1, self.MAX_SLICES, slice_dim) * 0.02)
        
        self.meta_module = MetadataModule(num_planes, meta_dim)
        self.meta_fuse   = nn.Linear(slice_dim + meta_dim, slice_dim)
        
        self.slice_encoder = SliceTransformerEncoder(
            dim=slice_dim, 
            heads=cfg.SLICE_ENC_HEADS if hasattr(cfg, 'SLICE_ENC_HEADS') else 8,
            num_layers=cfg.SLICE_ENC_LAYERS if hasattr(cfg, 'SLICE_ENC_LAYERS') else 2, 
            dropout=0.0)

        num_labels = cfg.NUM_LABELS if hasattr(cfg, 'NUM_LABELS') else 12
        dropout = cfg.DROPOUT if hasattr(cfg, 'DROPOUT') else 0.3
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(slice_dim), nn.Dropout(dropout),
            nn.Linear(slice_dim, 512), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(512, num_labels))

    def train(self, mode: bool = True):
        """Override train to keep Dinov2 frozen."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images, slice_mask, plane_ids, fluid_sensitive, fat_suppression):
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
        B, S, C, H, W = images.shape

        # ① Dinov2 backbone
        feats = self.backbone(images.view(B*S, C, H, W))
        feats = feats[:, 1:, :].mean(dim=1).view(B, S, -1)

        # ② Project to slice_dim
        slice_feats = self.slice_proj(feats)

        # ③ Metadata injection at slice level
        meta = self.meta_module(plane_ids, fluid_sensitive, fat_suppression)
        slice_feats = self.meta_fuse(torch.cat([slice_feats, meta], dim=-1))

        # ④ Add positional encoding
        slice_feats = slice_feats + self.slice_pos[:, :S, :]
        slice_feats = slice_feats.masked_fill(slice_mask.unsqueeze(-1), 0.0)

        # ⑤ Study transformer (attends across all slices of all series)
        encoded = self.slice_encoder(slice_feats, src_key_padding_mask=slice_mask)

        # ⑥ Mean-pool over valid slices
        valid_f = (~slice_mask).float().unsqueeze(-1)
        study_feat = (encoded * valid_f).sum(1) / valid_f.sum(1).clamp(1)

        # ⑦ Classify
        return self.classifier(study_feat)


def compute_valid_size(H: int, W: int, patch_size: int = 14) -> tuple:
    """
    Compute padded size for Dinov2 (uses 14x14 patches).
    
    Args:
        H, W: Original image height and width
        patch_size: DINO patch size (14 for Dinov2)
    
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
        patch_size: DINO patch size (14 for Dinov2)
    
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
if __name__ == '__main__':
    Dinov2Backbone(str(
                Path.home() / '.cache/huggingface/hub/models--ricklisz123--Dinov2-ViTB-16-CT-3M/snapshots/20d233b0ce1b1f93c46550787cf4b409a6f4a95e/model.pth'
            ))
    pass