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
    def __init__(self, stype, inputdim=768):
        super().__init__()
        self.stype = stype
        self.slice_pool = AttentionPooling(in_dim=inputdim, hidden_dim=128)
        self.slice_conved_pool = AttentionPooling(in_dim=inputdim, hidden_dim=128)
        conv_kernel_size=7
        self.conv=nn.Sequential(
            nn.Conv1d(inputdim, inputdim, conv_kernel_size,padding=conv_kernel_size // 2),
            nn.BatchNorm1d(inputdim),
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


class PerLabelFusion(nn.Module):
    def __init__(self, num_labels=12, num_branches=3):
        super().__init__()
        # Learnable weight matrix of shape (12, 3)
        self.weights = nn.Parameter(torch.ones(num_labels, num_branches))
        self.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, x):
        # x shape: (B, 12, 3)
        # Softmax ensures weights across the 3 branches sum to 1 for each label
        normalized_weights = F.softmax(self.weights, dim=-1)  # Shape: (12, 3)

        # Weighted sum across the last dimension
        out = torch.sum(x * normalized_weights, dim=-1) + self.bias  # Shape: (B, 12)
        return out

class KneeAbnormalityModelRefactored(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        num_labels = getattr(cfg, "NUM_LABELS", 12)
        inputdim=768
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
        self.feature_proj = nn.Sequential(nn.Linear(inputdim*2*num_types + meta_dim, 512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,256))
        # Slice-level pooling per series
        self.serie_proj = nn.ModuleList([SerieTypeNN(i,inputdim) for i in range(num_types)])
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
        self.output_pool = AttentionPooling(in_dim=256, hidden_dim=64)

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
        self.head_pool =nn.Linear(256, num_labels)
        self.fusion=PerLabelFusion(num_labels,num_branches=3)

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
        p_emb = self.plane_embed(plane_ids.long().clamp(0, 3))
        c_emb = self.crop_embed(plane_crop_mixed[:, :, 1].long().clamp(0, 6))
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
        outputpoolcat=torch.cat(
            [acl_feat.unsqueeze(1), mcl_feat.unsqueeze(1), mm_feat.unsqueeze(1), lm_feat.unsqueeze(1), moa_feat.unsqueeze(1), loa_feat.unsqueeze(1), pfoa_feat.unsqueeze(1), effusion_feat.unsqueeze(1), synovitis_feat.unsqueeze(1), baker_feat.unsqueeze(1),
             contusion_feat.unsqueeze(1), fracture_feat.unsqueeze(1)], dim=-2
        )
        outputpool=self.output_pool(outputpoolcat)
        outputpool=self.head_pool(outputpool)
        prediction_cat=torch.cat(
            [out_acl, out_mcl, out_mm, out_lm, out_moa, out_loa, out_pfoa, out_effusion, out_synovitis, out_baker,
             out_contusion, out_fracture], dim=-1
        )
        wgen_cat=torch.cat([out_gen.unsqueeze(2), prediction_cat.unsqueeze(2), outputpool.unsqueeze(2)], dim=-1)
        output=self.fusion(wgen_cat)
        return output