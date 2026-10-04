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

        self.serie_embed = nn.Embedding(6, 8)
        meta_dim = 8  # Total: 32
        # self.feature_attn=MultiHeadAttentionPooling(in_dim=512)
        self.series_attn=MultiHeadAttentionPooling(in_dim=512+meta_dim)
        self.inputdropout = nn.Dropout(0.2)
        num_types = 2

        self.feature_proj = nn.Sequential(
            nn.Linear(inputdim, num_labels),
        )

        self.inputdim = inputdim
        # Pathology Feature Interaction Module
        self.fc=nn.Linear(num_labels + meta_dim, num_labels)

    def forward(
        self,
        features,
        slice_mask,
        serie_mask,
        plane_ids,
        fluid_sensitive,
        fat_suppression,
        plane_crop_mixed,
        serie_type,
        visionmodel=None
    ):
        if visionmodel is not None:
            if self.training:
                B, Serie, S, C, H, W = features.shape
                features=visionmodel(features.view(B*Serie*S, C, H, W))
                features=features.view(B,Serie,S,-1)
            else:
                B, Serie, S, C, H, W = features.shape
                N = B * Serie * S
                features_flat = features.view(N, C, H, W)
                perm = torch.randperm(N)
                inv_perm = torch.argsort(perm)
                shuffled_features = features_flat[perm]
                shuffled_out = visionmodel(shuffled_features)
                restored_out = shuffled_out[inv_perm]
                features=restored_out.view(B,Serie,S,-1)
        B, Serie, S, C = features.shape
        # feat_attn=features.view(B * Serie * S, P, C)
        # feat_attn=self.feature_attn(feat_attn)
        # 1. Feature Extraction per Series
        attn_out=features.mean(-2).view(B, Serie, -1)
        feat_flat = attn_out.view(B * Serie, C)
        s_mask_flat = slice_mask.view(B * Serie, S)
        series_feats=self.feature_proj(feat_flat).view(B , Serie, -1)
        #self.feature_attn(series_feats, s_mask_flat).view(B, Serie, -1)
        # 2. Categorical Metadata Embeddings
        serie_emb=self.serie_embed(serie_type.long().clamp(0, 6))

        # 3. Projection to 512-dim Space
        combined = torch.cat([series_feats, serie_emb], dim=-1)
        # series_embeddings = F.relu(self.feature_proj(combined))
        #attcmb=self.series_attn(combined,serie_mask)
        # 6. Cross-Pathology Interaction & Prediction
        return self.fc(combined.mean(-2))  # Returns [B, 12]