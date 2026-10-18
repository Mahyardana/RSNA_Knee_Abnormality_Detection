import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


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

class SharedSliceEncoder(nn.Module):
    """Projects per-slice backbone features and models inter-slice context."""

    def __init__(self, inputdim, dim=512, max_slices=64, radius=5, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(inputdim),
            nn.Linear(inputdim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.radius = radius
        self.local_mix = nn.Linear(dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_slices, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.norm = nn.LayerNorm(dim)

    def gaussian_context(self, h, mask):
        # Each slice is x=0; neighbours within [-radius, radius] get softmax(gaussian(x)) weights.
        S = h.shape[1]
        pos = torch.arange(S, device=h.device, dtype=torch.float32)
        d = pos[None, :] - pos[:, None]  # [S, S], offset of neighbour j relative to slice i
        g = torch.exp(-d ** 2 / 2)
        valid = (d.abs() <= self.radius).unsqueeze(0) & mask.unsqueeze(1)  # [B, S, S]
        logits = g.unsqueeze(0).expand(h.shape[0], -1, -1).masked_fill(~valid, -1e4)
        w = F.softmax(logits, dim=-1).to(h.dtype)
        return self.local_mix(torch.bmm(w, h))

    def forward(self, x, mask):
        B, S, _ = x.shape
        h = self.proj(x)
        has_valid = mask.any(dim=-1, keepdim=True)
        safe_mask = torch.where(has_valid, mask, torch.ones_like(mask))
        h = h + self.gaussian_context(h, safe_mask)
        h = h + self.pos_embed[:, :S]
        return self.norm(h), safe_mask, has_valid


class MultiLabelAttentionPooling(nn.Module):
    """One attention query per label over the slice axis, plus a shared max-pool."""

    def __init__(self, dim=512, num_labels=12):
        super().__init__()
        self.num_labels = num_labels
        self.attn = nn.Linear(dim, num_labels)

    def forward(self, h, mask):
        scores = self.attn(h).float().masked_fill(~mask.unsqueeze(-1), -1e4)  # [B, S, L]
        weights = F.softmax(scores, dim=1).to(h.dtype)
        attn_pooled = torch.einsum('bsd,bsl->bld', h, weights)  # [B, L, D]
        max_pooled = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values  # [B, D]
        max_pooled = max_pooled.unsqueeze(1).expand(-1, self.num_labels, -1)
        return torch.cat([attn_pooled, max_pooled], dim=-1)  # [B, L, 2D]


class KneeAbnormalityHead(nn.Module):
    def __init__(self, inputdim, num_labels=12, dim=512, meta_dim=16, dropout=0.2):
        super().__init__()
        self.num_labels = num_labels
        self.serie_embed = nn.Embedding(10, meta_dim)
        self.encoder = SharedSliceEncoder(inputdim, dim=dim, dropout=dropout)
        self.pool = MultiLabelAttentionPooling(dim, num_labels)
        feat_dim = dim * 2 + meta_dim
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feat_dim)
        self.weight = nn.Parameter(torch.randn(num_labels, feat_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_labels))
        self.gate = nn.Sequential(nn.Linear(meta_dim, num_labels), nn.Sigmoid())

    def forward(self, features, slice_mask, serie_type):
        B, S, _ = features.shape
        slice_mask = slice_mask.view(B, S)
        h, safe_mask, has_valid = self.encoder(features, slice_mask)
        pooled = self.pool(h, safe_mask)  # [B, L, 2D]
        serie_emb = self.serie_embed(serie_type.long().clamp(0, 9))  # [B, meta]
        feat = torch.cat([pooled, serie_emb.unsqueeze(1).expand(-1, self.num_labels, -1)], dim=-1)
        feat = self.dropout(self.norm(feat))
        logits = (feat * self.weight).sum(-1) + self.bias  # [B, L]
        return logits * self.gate(serie_emb) * has_valid.float()


class KneeAbnormalityModelRefactored(nn.Module):
    def __init__(self, cfg, inputdim=768,backbone=None):
        super().__init__()
        self.cfg = cfg
        num_labels = 12
        self.head = KneeAbnormalityHead(inputdim, num_labels=num_labels)
        self.backbone=backbone
        self.resizetransform=torchvision.transforms.Resize((224,224))

        self.pool = MultiLabelAttentionPooling(inputdim, num_labels)
        self.labelfc=nn.ModuleList([
            nn.Sequential(
                nn.Linear(inputdim*2, inputdim),
                nn.Linear(inputdim, 1)
            )
            for _ in range(num_labels)])
        self.num_labels = num_labels
        # self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(
        self,
        features,
        slice_mask,
        serie_type
    ):
        # if not self.training:
        B, S, C, H, W = features.shape
        # features=features[:,:,:,0,:,:]
        # if S<self.cfg.MAX_SLICES_PER_SERIES:
        #     slicediff=self.cfg.MAX_SLICES_PER_SERIES-S
        #     newfeats=torch.zeros((B,Serie,self.cfg.MAX_SLICES_PER_SERIES,H,W),dtype=features.dtype)
        #     newfeats[:,:,:S,:,:]=features
        #     features=newfeats
        #     S=self.cfg.MAX_SLICES_PER_SERIES
        # reduced=self.channel_reducer(features.view(B*Serie, S, H, W))
        x=features.view(B*S,C,H,W)
        # x = (x - self.mean) / self.std
        features=self.backbone(x)
        # features2=self.backbone[1](self.resizetransform(x))
        # features=torch.cat([features[:,0],features[:,1:].mean(-2)],dim=-1)
        features=features.view(B,S,-1)
        poolout=self.pool(features, slice_mask)
        labelouts=[self.labelfc[i](poolout[:,i,:]) for i in range(self.num_labels)]
        alllabels=torch.cat(labelouts,dim=-1)
        # allfeat=torch.cat([features,features2], dim=-1)
        return alllabels#self.head(features, slice_mask, serie_type)