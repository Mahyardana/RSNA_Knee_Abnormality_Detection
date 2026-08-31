import torch
import torch.nn as nn
import torch.nn.functional as F

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
            scores = scores.masked_fill(~mask, -1e8)

        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, N, 1]
        return torch.sum(x * weights, dim=1)  # [B, Feature_Dim]
class SerieTypeNN(nn.Module):
    def __init__(self, stype):
        super().__init__()
        self.stype = stype
        self.slice_pool = AttentionPooling(in_dim=768, hidden_dim=128)
        self.slice_conved_pool = AttentionPooling(in_dim=768, hidden_dim=128)
        conv_kernel_size=7
        self.conv=nn.Sequential(
            nn.Conv2d(1, 1, conv_kernel_size,padding=conv_kernel_size // 2),
            nn.BatchNorm2d(1),
            nn.GELU(),
            nn.Dropout(0.2),
        )
    def forward(self, feat_flat: torch.Tensor, mask: torch.Tensor, serie_type: torch.Tensor,) -> torch.Tensor:

        mask_mask=((serie_type[:,0]==1) & (serie_type[:,1]==1)).unsqueeze(-1).repeat(1,mask.shape[-1])
        if self.stype == 0:
            mask_mask=~mask_mask
        s_mask_flat = mask & mask_mask
        series_features = self.slice_pool(feat_flat, mask=s_mask_flat)
        series_conved=self.conv(feat_flat.unsqueeze(1)).squeeze(1)

        if series_conved.shape[1] != s_mask_flat.shape[1]:
            pass
        series_conved_features=self.slice_conved_pool(series_conved, mask=s_mask_flat)
        return torch.cat((series_features,series_conved_features),dim=1)
class LabelTransformer(nn.Module):
    """
    Pure Cross-Label Attention Layer.
    Allows 12 pathology tokens to perform multi-head self-attention on each other.
    """

    def __init__(self, embed_dim=256, num_labels=12, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()

        # Learnable Label Identity Embeddings (1 per pathology)
        # Tells the transformer: Token 0 = ACL, Token 4 = MOA, Token 10 = Contusion, etc.
        self.label_embed = nn.Parameter(torch.randn(1, num_labels, embed_dim) * 0.02)

        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: [B, 12, 256]
        B, N, D = x.shape

        # 1. Add Label Identity Embeddings
        x_in = x + self.label_embed  # [B, 12, 256]

        # 2. Contextualize via Self-Attention across 12 Label Tokens
        x_out = self.transformer(x_in)  # [B, 12, 256]

        # 3. Residual connection + LayerNorm
        return self.norm(x + x_out)


class KneeAbnormalityModelRefactored(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        num_labels = getattr(cfg, "NUM_LABELS", 12)
        self.num_labels = num_labels

        # 1. Categorical Metadata Embeddings
        self.plane_embed = nn.Embedding(4, 8)
        self.crop_embed = nn.Embedding(7, 8)
        self.fluid_embed = nn.Embedding(2, 4)
        self.fat_embed = nn.Embedding(2, 4)
        self.type_embed = nn.Embedding(7, 8)
        meta_dim = 8 + 8 + 4 + 4 + 8  # 32

        self.inputdropout = nn.Dropout(0.2)

        # 2. Backbone Feature Projection
        num_types = 2
        self.feature_proj = nn.Sequential(
            nn.Linear(768 * 2 * num_types + meta_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
        )

        self.serie_proj = nn.ModuleList([SerieTypeNN(i) for i in range(num_types)])

        # 3. Individual Pathology Attention Pools (256-dim output per pathology)
        self.pools = nn.ModuleList([AttentionPooling(in_dim=256, hidden_dim=64) for _ in range(num_labels)])
        self.global_pool = AttentionPooling(in_dim=256, hidden_dim=128)

        # 4. Strategy 1: Label Transformer (Inter-Pathology Communication)
        self.label_transformer = LabelTransformer(
            embed_dim=256,
            num_labels=num_labels,
            num_heads=4,
            num_layers=2
        )

        # 5. Clean, Modular Classification Heads (1 per pathology)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1)
            ) for _ in range(num_labels)
        ])

        # Auxiliary global head
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
            serie_type,
    ):
        B, Serie, S, C = features.shape

        # --- Step 1: Feature Extraction & Metadata Projection ---
        feat_flat = self.inputdropout(features.view(B * Serie, S, C))
        s_mask_flat = slice_mask.view(B * Serie, S)

        series_features = [
            x(
                feat_flat,
                s_mask_flat,
                torch.cat([fluid_sensitive.unsqueeze(-1), fat_suppression.unsqueeze(-1)], dim=-1).view(B * Serie, 2),
            ).view(B, Serie, C * 2)
            for x in self.serie_proj
        ]

        p_emb = self.plane_embed(plane_ids.clamp(0, 3))
        c_emb = self.crop_embed(plane_crop_mixed[:, :, 1].clamp(0, 6))
        fl_emb = self.fluid_embed(fluid_sensitive.long().clamp(0, 1))
        ft_emb = self.fat_embed(fat_suppression.long().clamp(0, 1))
        tp_emb = self.type_embed(serie_type.long().clamp(0, 6))

        series_features_combined = torch.cat(series_features, dim=-1)
        meta_feats = torch.cat([p_emb, c_emb, fl_emb, ft_emb, tp_emb], dim=-1)

        combined = torch.cat([series_features_combined, meta_feats], dim=-1)
        series_embeddings = F.relu(self.feature_proj(combined))  # [B, Serie, 256]

        # --- Step 2: Extract 12 Isolated Pathology Vectors ---
        acl_mask = (plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 1)
        mcl_mask = ((plane_ids == 1) | (plane_ids == 2)) & (
                    (plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 2))
        mm_mask = mcl_mask
        lm_mask = ((plane_ids == 1) | (plane_ids == 2)) & (
                    (plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 3))
        moa_mask = mm_mask
        loa_mask = lm_mask
        pfoa_mask = ((plane_ids == 0) | (plane_ids == 1)) & (
                    (plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 5) | (
                        plane_crop_mixed[:, :, 1] == 1))
        effusion_mask = pfoa_mask
        synovitis_mask = pfoa_mask
        baker_mask = ((plane_ids == 0) | (plane_ids == 1)) & (
                    (plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 4))
        contusion_mask = (plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 2) | (
                    plane_crop_mixed[:, :, 1] == 3) | (plane_crop_mixed[:, :, 1] == 1)
        fracture_mask = (plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 2) | (
                    plane_crop_mixed[:, :, 1] == 3)

        masks = [
            acl_mask, mcl_mask, mm_mask, lm_mask, moa_mask, loa_mask,
            pfoa_mask, effusion_mask, synovitis_mask, baker_mask, contusion_mask, fracture_mask
        ]

        raw_label_tokens = []
        for i in range(self.num_labels):
            m = serie_mask & masks[i]
            raw_label_tokens.append(self.pools[i](series_embeddings, mask=m))

        # Stack into label matrix: [B, 12, 256]
        raw_label_matrix = torch.stack(raw_label_tokens, dim=1)

        # --- Step 3: Inter-Label Transformer Communication ---
        # Matrix transforms from [B, 12, 256] -> [B, 12, 256]
        contextualized_label_matrix = self.label_transformer(raw_label_matrix)

        # --- Step 4: Classify Each Contextualized Pathology Token ---
        logits_list = []
        for i in range(self.num_labels):
            token_i = contextualized_label_matrix[:, i, :]  # [B, 256]
            logit_i = self.heads[i](token_i)  # [B, 1]
            logits_list.append(logit_i)

        prediction_cat = torch.cat(logits_list, dim=-1)  # [B, 12]

        # --- Step 5: Global Auxiliary Fusion ---
        gen_feat = self.global_pool(series_embeddings, mask=serie_mask)
        out_gen = self.head_general(gen_feat)  # [B, 12]

        wgen_cat = torch.cat([out_gen.unsqueeze(2), prediction_cat.unsqueeze(2)], dim=-1)
        output = wgen_cat.amin(dim=-1)

        return output