import torch
import torch.nn as nn


class ABMIL(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, patches):
        if patches.dim() != 2:
            raise ValueError(f"ABMIL expects (N, D), got {tuple(patches.shape)}")
        weights = torch.softmax(self.attention(patches).squeeze(-1), dim=0)
        return (weights.unsqueeze(-1) * patches).sum(dim=0)


class LinearProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, dim)

    def forward(self, h, cross_out):
        g = torch.sigmoid(self.gate(h))
        return h + g * cross_out


class FFN(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class CrossModalTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion=4, dropout=0.1):
        super().__init__()
        self.self_attn= nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_self= nn.LayerNorm(dim)
        self.norm_cross = nn.LayerNorm(dim)
        self.gated_res = GatedResidual(dim)
        self.ffn = FFN(dim, ffn_expansion, dropout)

    def forward(self, he, ihc):
        sa_out, _ = self.self_attn(he, he, he)
        h = self.norm_self(he + sa_out)

        ca_out, _ = self.cross_attn(h, ihc, ihc)
        ca_out = self.norm_cross(ca_out)

        h = self.gated_res(h, ca_out)
        h = self.ffn(h)
        return h


class CrossBagAggregator(nn.Module):
    def __init__(self, in_dim, dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.he_proj = LinearProjection(in_dim, dim)
        self.ihc_proj = LinearProjection(in_dim, dim)
        self.he_pool = ABMIL(dim)
        self.ihc_pool = ABMIL(dim)
        self.interaction = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def forward(self, he_feats, ihc_feats):
        if he_feats.dim() != 2 or ihc_feats.dim() != 2:
            raise ValueError("Classifier expects unbatched HE and IHC patch sets shaped (N, D)")
        he = self.he_pool(self.he_proj(he_feats))
        ihc = self.ihc_pool(self.ihc_proj(ihc_feats))
        return self.interaction(torch.cat([he, ihc], dim=-1))

    def forward_pooled(self, he_vec, ihc_vec):
        if he_vec.dim() != 1 or ihc_vec.dim() != 1:
            raise ValueError("forward_pooled expects vectors shaped (D,)")
        he = self.he_proj(he_vec)
        ihc = self.ihc_proj(ihc_vec)
        return self.interaction(torch.cat([he, ihc], dim=-1))


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout= 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class DualBagClassifier(nn.Module):
    def __init__(self, in_dim, dim,num_heads ,num_layers  ,mlp_hidden, num_classes, dropout = 0.1):
        super().__init__()
        self.aggregator = CrossBagAggregator(in_dim, dim, num_heads, num_layers, dropout)
        self.classifier = MLPClassifier(dim, mlp_hidden, num_classes)

    def forward(self, he_feats, ihc_feats):
        wsi_repr = self.aggregator(he_feats, ihc_feats)
        return self.classifier(wsi_repr)

    def forward_pooled(self, he_vec, ihc_vec):
        return self.classifier(self.aggregator.forward_pooled(he_vec, ihc_vec))