import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, in_dim: int = 512, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.attn = nn.Linear(in_dim, num_heads)
        self.proj = nn.Linear(in_dim * num_heads, in_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        scores = self.attn(x)  # [B, S, NumHeads]

        if mask is not None:
            has_valid = mask.any(dim=-1, keepdim=True)  # [B, 1]
            valid_mask = torch.where(~has_valid, torch.ones_like(mask), mask)
            mask_expanded = valid_mask.unsqueeze(-1)  # [B, S, 1]

            scores = scores.masked_fill(~mask_expanded, -1e4)
            weights = F.softmax(scores, dim=1)  # [B, S, NumHeads]

            pooled = torch.einsum('bsd,bsh->bhd', x, weights)
            pooled_flat = pooled.reshape(x.shape[0], -1)

            out = self.dropout(self.proj(pooled_flat))
            return out * has_valid.float()
        else:
            weights = F.softmax(scores, dim=1)
            pooled = torch.einsum('bsd,bsh->bhd', x, weights)
            pooled_flat = pooled.reshape(x.shape[0], -1)
            return self.dropout(self.proj(pooled_flat))


class SerieTypeNN(nn.Module):
    def __init__(self, stype, inputdim=768):
        super().__init__()
        self.stype = stype
        self.slice_pool = MultiHeadAttentionPooling(in_dim=inputdim)
        self.slice_conved_pool = MultiHeadAttentionPooling(in_dim=inputdim)
        conv_kernel_size = 3
        self.conv = nn.Sequential(
            nn.Conv1d(inputdim, inputdim, conv_kernel_size, padding=conv_kernel_size // 2),
            nn.GroupNorm(8, inputdim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, feat_flat: torch.Tensor, mask: torch.Tensor, serie_type: torch.Tensor) -> torch.Tensor:
        mask_mask = ((serie_type[:, 0] == 1) & (serie_type[:, 1] == 1)).unsqueeze(-1).repeat(1, mask.shape[-1])
        if self.stype == 0:
            mask_mask = ~mask_mask
        s_mask_flat = mask & mask_mask

        series_features = self.slice_pool(feat_flat, mask=s_mask_flat)
        series_conved = self.conv(feat_flat.transpose(1, 2)).transpose(1, 2)
        series_conved = series_conved * s_mask_flat.unsqueeze(-1).float()

        series_conved_features = self.slice_conved_pool(series_conved, mask=s_mask_flat)
        return self.dropout(torch.cat((series_features, series_conved_features), dim=1))


class PathologyFeatureInteraction(nn.Module):
    def __init__(self, feature_dim=512, num_labels=12, num_heads=4):
        super().__init__()
        # 1. Pathology identity embedding to distinguish tokens in self-attention
        self.pathology_embed = nn.Parameter(torch.randn(1, num_labels, feature_dim) * 0.02)
        
        self.self_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

        # 2. Independent linear heads per pathology
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Dropout(0.2), nn.Linear(feature_dim, 1)) 
            for _ in range(num_labels)
        ])

    def forward(self, pathology_feats: torch.Tensor) -> torch.Tensor:
        # pathology_feats: [B, 12, 512]
        
        # Add pathology identity tokens
        x = pathology_feats + self.pathology_embed

        # Cross-pathology communication
        attn_out, _ = self.self_attn(x, x, x)
        interacted_feats = self.norm(x + attn_out)  # [B, 12, 512]

        # Predict scalar logit per pathology token using dedicated heads
        logits = torch.cat([self.heads[i](interacted_feats[:, i]) for i in range(len(self.heads))], dim=-1)
        return logits  # [B, 12]


class KneeAbnormalityModelRefactored(nn.Module):
    def __init__(self, cfg, inputdim=768):
        super().__init__()
        self.cfg = cfg
        num_labels = getattr(cfg, "NUM_LABELS", 12)

        # Categorical Metadata Embeddings
        self.plane_embed = nn.Embedding(4, 8)
        self.crop_embed = nn.Embedding(7, 8)
        self.fluid_embed = nn.Embedding(2, 4)
        self.fat_embed = nn.Embedding(2, 4)
        meta_dim = 8 + 8 + 4 + 4  # Total: 24

        self.inputdropout = nn.Dropout(0.2)
        num_types = 2

        self.feature_proj = nn.Sequential(
            nn.Linear(inputdim * 2 * num_types + meta_dim, 512)
        )

        self.serie_proj = nn.ModuleList([SerieTypeNN(i, inputdim) for i in range(num_types)])

        # Pathology-Specific Series Aggregators
        self.acl_pool = MultiHeadAttentionPooling(in_dim=512)
        self.mcl_pool = MultiHeadAttentionPooling(in_dim=512)
        self.mm_pool = MultiHeadAttentionPooling(in_dim=512)
        self.lm_pool = MultiHeadAttentionPooling(in_dim=512)
        self.moa_pool = MultiHeadAttentionPooling(in_dim=512)
        self.loa_pool = MultiHeadAttentionPooling(in_dim=512)
        self.pfoa_pool = MultiHeadAttentionPooling(in_dim=512)
        self.effusion_pool = MultiHeadAttentionPooling(in_dim=512)
        self.synovitis_pool = MultiHeadAttentionPooling(in_dim=512)
        self.baker_pool = MultiHeadAttentionPooling(in_dim=512)
        self.contusion_pool = MultiHeadAttentionPooling(in_dim=512)
        self.fracture_pool = MultiHeadAttentionPooling(in_dim=512)

        # Pathology Feature Interaction Module
        self.fusion = PathologyFeatureInteraction(feature_dim=512, num_labels=num_labels)

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
        B, Serie, S, C = features.shape

        # 1. Feature Extraction per Series
        feat_flat = self.inputdropout(features.reshape(B * Serie, S, C))
        s_mask_flat = slice_mask.view(B * Serie, S)
        type_cond = torch.cat([fluid_sensitive.unsqueeze(-1), fat_suppression.unsqueeze(-1)], dim=-1).view(B * Serie, 2)

        series_features = [
            x(feat_flat, s_mask_flat, type_cond).view(B, Serie, C * 2)
            for x in self.serie_proj
        ]

        # 2. Categorical Metadata Embeddings
        p_emb = self.plane_embed(plane_ids.long().clamp(0, 3))
        c_emb = self.crop_embed(plane_crop_mixed[:, :, 1].long().clamp(0, 6))
        fl_emb = self.fluid_embed(fluid_sensitive.long().clamp(0, 1))
        ft_emb = self.fat_embed(fat_suppression.long().clamp(0, 1))

        series_features_combined = torch.cat(series_features, dim=-1)
        meta_feats = torch.cat([p_emb, c_emb, fl_emb, ft_emb], dim=-1)

        # 3. Projection to 512-dim Space
        combined = torch.cat([series_features_combined, meta_feats], dim=-1)
        series_embeddings = F.relu(self.feature_proj(combined))

        # 4. Pathology-Specific Masked Pooling
        acl_mask = (plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 1)
        acl_feat = self.acl_pool(series_embeddings, mask=serie_mask & acl_mask)

        mcl_mask = ((plane_ids == 1) | (plane_ids == 2)) & ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 2))
        mcl_feat = self.mcl_pool(series_embeddings, mask=serie_mask & mcl_mask)

        mm_mask = ((plane_ids == 1) | (plane_ids == 2)) & ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 2))
        mm_feat = self.mm_pool(series_embeddings, mask=serie_mask & mm_mask)

        lm_mask = ((plane_ids == 1) | (plane_ids == 2)) & ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 3))
        lm_feat = self.lm_pool(series_embeddings, mask=serie_mask & lm_mask)

        moa_mask = ((plane_ids == 1) | (plane_ids == 2)) & ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 2))
        moa_feat = self.moa_pool(series_embeddings, mask=serie_mask & moa_mask)

        loa_mask = ((plane_ids == 1) | (plane_ids == 2)) & ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 3))
        loa_feat = self.loa_pool(series_embeddings, mask=serie_mask & loa_mask)

        pfoa_mask = ((plane_ids == 0) | (plane_ids == 1)) & ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 5) | (plane_crop_mixed[:, :, 1] == 1))
        pfoa_feat = self.pfoa_pool(series_embeddings, mask=serie_mask & pfoa_mask)

        effusion_mask = ((plane_ids == 0) | (plane_ids == 1)) & ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 1) | (plane_crop_mixed[:, :, 1] == 5))
        effusion_feat = self.effusion_pool(series_embeddings, mask=serie_mask & effusion_mask)

        synovitis_mask = ((plane_ids == 0) | (plane_ids == 1)) & ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 5) | (plane_crop_mixed[:, :, 1] == 1))
        synovitis_feat = self.synovitis_pool(series_embeddings, mask=serie_mask & synovitis_mask)

        baker_mask = ((plane_ids == 0) | (plane_ids == 1)) & ((plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 4))
        baker_feat = self.baker_pool(series_embeddings, mask=serie_mask & baker_mask)

        contusion_mask = (plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 2) | (plane_crop_mixed[:, :, 1] == 3) | (plane_crop_mixed[:, :, 1] == 1)
        contusion_feat = self.contusion_pool(series_embeddings, mask=serie_mask & contusion_mask)

        fracture_mask = (plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 2) | (plane_crop_mixed[:, :, 1] == 3)
        fracture_feat = self.fracture_pool(series_embeddings, mask=serie_mask & fracture_mask)

        # 5. Stack into [B, 12, 512] Tensor
        outputpoolcat = torch.stack(
            [
                acl_feat, mcl_feat, mm_feat, lm_feat,
                moa_feat, loa_feat, pfoa_feat, effusion_feat,
                synovitis_feat, baker_feat, contusion_feat, fracture_feat
            ],
            dim=1
        )

        # 6. Cross-Pathology Interaction & Prediction
        return self.fusion(outputpoolcat)  # Returns [B, 12]