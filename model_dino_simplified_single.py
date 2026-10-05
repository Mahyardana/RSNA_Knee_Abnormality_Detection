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


class KneeAbnormalityModelRefactored(nn.Module):
    def __init__(self, cfg, inputdim=768):
        super().__init__()
        self.cfg = cfg
        num_labels = 1

        self.serie_embed = nn.Embedding(6, 8)
        meta_dim = 8  # Total: 32
        self.feature_attn=MultiHeadAttentionPooling(in_dim=512)
        # self.featurelstm=nn.LSTM(512,256,num_layers=1, batch_first=True,dropout=0.1,bidirectional=True)
        self.series_attn=MultiHeadAttentionPooling(in_dim=512+meta_dim)
        self.inputdropout = nn.Dropout(0.2)
        num_types = 2

        self.feature_proj = nn.Sequential(
            nn.Linear(inputdim, 512),
            nn.ReLU()
        )

        self.inputdim = inputdim
        # Pathology Feature Interaction Module
        self.fc=nn.Sequential(
            nn.Linear(512 + meta_dim, num_labels)
        )

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
                features=visionmodel.forward_features(features.view(B*Serie*S, C, H, W))
                features=torch.cat([features[:,0],features[:,1:].mean(-2)],dim=-1)
                features=features.view(B,Serie,S,-1)
            else:
                B, Serie, S, C, H, W = features.shape
                N = B * Serie * S
                features_flat = features.view(N, C, H, W)
                perm = torch.randperm(N)
                inv_perm = torch.argsort(perm)
                shuffled_features = features_flat[perm]
                shuffled_out = visionmodel.forward_features(shuffled_features)
                restored_out = shuffled_out[inv_perm]
                restored_out=torch.cat([restored_out[:,0],restored_out[:,1:].mean(-2)],dim=-1)
                features=restored_out.view(B,Serie,S,-1)
        B, Serie, S, C = features.shape
        # feat_attn=features.view(B * Serie * S, C)
        # feat_attn=self.feature_attn(feat_attn)
        # 1. Feature Extraction per Series
        # attn_out=self.feature_attn(feat_attn)#features.mean(-2).view(B, Serie, -1)
        feat_flat = features.view(B * Serie * S, C)
        s_mask_flat = slice_mask.view(B * Serie, S)
        series_feats=F.relu(self.feature_proj(feat_flat).view(B * Serie ,S, -1))
        # lstm_out, _=self.featurelstm(series_feats)#
        attn_out=self.feature_attn(series_feats, s_mask_flat).view(B, Serie, -1)
        # 2. Categorical Metadata Embeddings
        serie_emb=self.serie_embed(serie_type.long().clamp(0, 6))

        # 3. Projection to 512-dim Space
        combined = torch.cat([attn_out, serie_emb], dim=-1)
        # series_embeddings = F.relu(self.feature_proj(combined))
        attcmb=F.relu(self.series_attn(combined,serie_mask))
        # 6. Cross-Pathology Interaction & Prediction
        return self.fc(attcmb)  # Returns [B, 12]