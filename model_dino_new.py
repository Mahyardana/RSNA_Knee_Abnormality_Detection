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
class PathologyLabelTransformer(nn.Module):
    """Allows all 12 pathology feature representations to communicate

    via Multi-Head Cross-Label Attention, capturing co-occurrence and
    causal relationships (e.g., Contusion/Effusion -> ACL, MM -> Baker's cyst).
    """

    def __init__(self, embed_dim: int = 256, num_labels: int = 12, num_heads: int = 4):
        super().__init__()
        # Learnable positional/identity embedding per pathology
        self.label_embed = nn.Parameter(
            torch.randn(1, num_labels, embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, pathology_feats: torch.Tensor) -> torch.Tensor:
        # pathology_feats: [B, 12, 256]
        x = pathology_feats + self.label_embed
        x_contextualized = self.transformer(x)
        # Residual connection keeps original local feature intact while adding context
        return pathology_feats + x_contextualized


class KneeAbnormalityModelRefactored(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        num_labels = getattr(cfg, "NUM_LABELS", 12)

        # 1. Categorical Metadata Embeddings
        self.plane_embed = nn.Embedding(4, 8)
        self.crop_embed = nn.Embedding(7, 8)
        self.fluid_embed = nn.Embedding(2, 4)
        self.fat_embed = nn.Embedding(2, 4)
        self.type_embed = nn.Embedding(7, 8)
        meta_dim = 8 + 8 + 4 + 4 + 8  # 32

        self.inputdropout = nn.Dropout(0.2)

        # 2. Backbone Feature Projection & Attention Pools
        num_types = 2
        self.feature_proj = nn.Sequential(
            nn.Linear(768 * 2 * num_types + meta_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
        )

        self.serie_proj = nn.ModuleList(
            [SerieTypeNN(i) for i in range(num_types)]
        )

        # 3. Individual Pathology-Specific Pools (256 dims each)
        self.pools = nn.ModuleList(
            [AttentionPooling(in_dim=256, hidden_dim=64) for _ in range(12)]
        )
        self.global_pool = AttentionPooling(in_dim=256, hidden_dim=128)

        # 4. Inter-Pathology Context Exchange Module (Label-Transformer)
        self.label_interaction = PathologyLabelTransformer(
            embed_dim=256, num_labels=12, num_heads=4
        )

        # 5. Enhanced Head Classifiers with Directed Dependency Injections
        # ACL Head accepts its contextualized embedding + explicit secondary markers
        # (ACL feat: 256 + Contusion: 256 + Effusion: 256 + OA: 256 = 1024)
        self.head_acl = nn.Sequential(
            nn.Linear(256 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

        # Baker's Cyst Head accepts (Baker feat: 256 + Effusion: 256 + MM: 256 = 768)
        self.head_baker = nn.Sequential(
            nn.Linear(256 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # Standard contextualized heads for remaining pathologies
        self.head_mcl = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_mm = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_lm = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_moa = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_loa = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_pfoa = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_effusion = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_synovitis = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_contusion = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_fracture = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 1))

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
                torch.cat(
                    [
                        fluid_sensitive.unsqueeze(-1),
                        fat_suppression.unsqueeze(-1),
                    ],
                    dim=-1,
                ).view(B * Serie, 2),
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

        # --- Step 2: Apply Masks and Pool per Pathology ---
        # (Using indexing list for clarity: 0:ACL, 1:MCL, 2:MM, 3:LM, 4:MOA, 5:LOA, 6:PFOA, 7:Effusion, 8:Synovitis, 9:Baker, 10:Contusion, 11:Fracture)

        acl_mask = (plane_crop_mixed[:, :, 1] == 0) | (
            plane_crop_mixed[:, :, 1] == 1
        )
        mcl_mask = (plane_ids == 1) | (plane_ids == 2) & (
            (plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 2)
        )
        mm_mask = mcl_mask
        lm_mask = (plane_ids == 1) | (plane_ids == 2) & (
            (plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 3)
        )
        moa_mask = mm_mask
        loa_mask = lm_mask
        pfoa_mask = (plane_ids == 0) | (plane_ids == 1) & (
            (plane_crop_mixed[:, :, 1] == 0)
            | (plane_crop_mixed[:, :, 1] == 5)
            | (plane_crop_mixed[:, :, 1] == 1)
        )
        effusion_mask = pfoa_mask
        synovitis_mask = pfoa_mask
        baker_mask = (plane_ids == 0) | (plane_ids == 1) & (
            (plane_crop_mixed[:, :, 1] == 0) | (plane_crop_mixed[:, :, 1] == 4)
        )
        contusion_mask = (
            (plane_crop_mixed[:, :, 1] == 0)
            | (plane_crop_mixed[:, :, 1] == 2)
            | (plane_crop_mixed[:, :, 1] == 3)
            | (plane_crop_mixed[:, :, 1] == 1)
        )
        fracture_mask = (
            (plane_crop_mixed[:, :, 1] == 0)
            | (plane_crop_mixed[:, :, 1] == 2)
            | (plane_crop_mixed[:, :, 1] == 3)
        )

        masks = [
            acl_mask,
            mcl_mask,
            mm_mask,
            lm_mask,
            moa_mask,
            loa_mask,
            pfoa_mask,
            effusion_mask,
            synovitis_mask,
            baker_mask,
            contusion_mask,
            fracture_mask,
        ]

        # Extract 12 pathology feature representations: [B, 256] each
        raw_pathology_feats = []
        for i in range(12):
            m = serie_mask & masks[i]
            raw_pathology_feats.append(self.pools[i](series_embeddings, mask=m))

        # Stack into matrix: [B, 12, 256]
        pathology_matrix = torch.stack(raw_pathology_feats, dim=1)

        # --- Step 3: Cross-Label Attention Communication ---
        # Every pathology feature vector now absorbs contextual evidence from all other 11 pathologies
        ctx_matrix = self.label_interaction(pathology_matrix)  # [B, 12, 256]

        # Unpack contextualized embeddings
        (
            f_acl,
            f_mcl,
            f_mm,
            f_lm,
            f_moa,
            f_loa,
            f_pfoa,
            f_eff,
            f_syn,
            f_bak,
            f_con,
            f_frc,
        ) = [ctx_matrix[:, i, :] for i in range(12)]

        # --- Step 4: Classify with Explicit Cross-Pathology Concatenations ---

        # ACL Head: Fuses ACL feature + Contusion feature + Effusion feature + Medial OA feature
        acl_fused = torch.cat([f_acl, f_con, f_eff, f_moa], dim=-1)
        out_acl = self.head_acl(acl_fused)

        # Baker's Cyst Head: Fuses Baker feature + Effusion feature + MM feature
        baker_fused = torch.cat([f_bak, f_eff, f_mm], dim=-1)
        out_baker = self.head_baker(baker_fused)

        # Standard heads receiving cross-contextualized embeddings
        out_mcl = self.head_mcl(f_mcl)
        out_mm = self.head_mm(f_mm)
        out_lm = self.head_lm(f_lm)
        out_moa = self.head_moa(f_moa)
        out_loa = self.head_loa(f_loa)
        out_pfoa = self.head_pfoa(f_pfoa)
        out_effusion = self.head_effusion(f_eff)
        out_synovitis = self.head_synovitis(f_syn)
        out_contusion = self.head_contusion(f_con)
        out_fracture = self.head_fracture(f_frc)

        # Global auxiliary prediction
        gen_feat = self.global_pool(series_embeddings, mask=serie_mask)
        out_gen = self.head_general(gen_feat)

        prediction_cat = torch.cat(
            [
                out_acl,
                out_mcl,
                out_mm,
                out_lm,
                out_moa,
                out_loa,
                out_pfoa,
                out_effusion,
                out_synovitis,
                out_baker,
                out_contusion,
                out_fracture,
            ],
            dim=-1,
        )

        wgen_cat = torch.cat(
            [out_gen.unsqueeze(2), prediction_cat.unsqueeze(2)], dim=-1
        )
        output = wgen_cat.amin(dim=-1)

        return output