from __future__ import annotations

import torch
import torch.nn as nn



class ABMIL(nn.Module):
    """Attention-Based MIL pooling: (N, D) → (D,)"""
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() != 2:
            raise ValueError(f"ABMIL expects (N, D), got {tuple(patches.shape)}")
        weights = torch.softmax(self.attention(patches).squeeze(-1), dim=0)
        return (weights.unsqueeze(-1) * patches).sum(dim=0)


class LinearProjection(nn.Module):
    """Linear + LayerNorm projection."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class GatedResidual(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor, cross_out: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(h))
        return h + g * cross_out


class FFN(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class LowRankFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        output_dim: int,
        rank: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank       = rank
        self.output_dim = output_dim

        # Each factor operates on (dim + 1) to include the bias unit
        self.he_factor  = nn.Parameter(torch.Tensor(rank, dim + 1))
        self.ihc_factor = nn.Parameter(torch.Tensor(rank, dim + 1))
        self.out_factor = nn.Parameter(torch.Tensor(rank, output_dim))

        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(output_dim)

        self._init_weights()

    def _init_weights(self):
        # Xavier uniform for factor matrices; small init keeps fusion stable
        nn.init.xavier_uniform_(self.he_factor)
        nn.init.xavier_uniform_(self.ihc_factor)
        nn.init.xavier_uniform_(self.out_factor)

    def forward(self, he_vec: torch.Tensor, ihc_vec: torch.Tensor) -> torch.Tensor:
        one    = torch.ones(1, device=he_vec.device, dtype=he_vec.dtype)
        he_b   = torch.cat([self.dropout(he_vec),  one], dim=0)   # (dim+1,)
        ihc_b  = torch.cat([self.dropout(ihc_vec), one], dim=0)   # (dim+1,)

        he_proj  = self.he_factor  @ he_b   
        ihc_proj = self.ihc_factor @ ihc_b   

        fused = (he_proj * ihc_proj) @ self.out_factor  

        return self.norm(fused)

    def extra_repr(self) -> str:
        return (f"rank={self.rank}, "
                f"output_dim={self.output_dim}")



class CrossBagAggregator(nn.Module):
    def __init__(
        self,
        in_dim:     int,
        dim:        int,
        num_heads:  int,
        num_layers: int,
        output_dim: int = 512,
        rank:       int = 16,
        dropout:    float = 0.1,
    ):
        super().__init__()
        # modality-specific projections + pooling
        self.he_proj  = LinearProjection(in_dim, dim)
        self.ihc_proj = LinearProjection(in_dim, dim)
        self.he_pool  = ABMIL(dim)
        self.ihc_pool = ABMIL(dim)

        # low-rank fusion replaces concat → linear
        self.lmf = LowRankFusion(dim, output_dim, rank=rank, dropout=dropout)

    def forward(
        self,
        he_feats:  torch.Tensor,    # (N, in_dim)
        ihc_feats: torch.Tensor,    # (M, in_dim)
    ) -> torch.Tensor:              # (output_dim,)
        if he_feats.dim() != 2 or ihc_feats.dim() != 2:
            raise ValueError(
                "CrossBagAggregator expects unbatched patch sets "
                f"shaped (N, D); got HE={tuple(he_feats.shape)}, "
                f"IHC={tuple(ihc_feats.shape)}"
            )
        he_vec  = self.he_pool(self.he_proj(he_feats))    # (dim,)
        ihc_vec = self.ihc_pool(self.ihc_proj(ihc_feats)) # (dim,)
        return self.lmf(he_vec, ihc_vec)                  # (output_dim,)

    def forward_pooled(
        self,
        he_vec:  torch.Tensor,   # (dim,)
        ihc_vec: torch.Tensor,   # (dim,)
    ) -> torch.Tensor:
        """Backwards-compatible path for already-pooled vectors."""
        if he_vec.dim() != 1 or ihc_vec.dim() != 1:
            raise ValueError("forward_pooled expects vectors shaped (D,)")
        # project to dim before fusion (skip ABMIL since already pooled)
        he_p  = self.he_proj(he_vec)
        ihc_p = self.ihc_proj(ihc_vec)
        return self.lmf(he_p, ihc_p)



class MLPClassifier(nn.Module):
    def __init__(
        self,
        in_dim:     int,
        hidden_dim: int,
        num_classes: int,
        dropout:    float = 0.25,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class DualBagClassifier(nn.Module):
    def __init__(
        self,
        in_dim:      int,
        dim:         int,
        num_heads:   int,
        num_layers:  int,
        mlp_hidden:  int,
        num_classes: int,
        dropout:     float = 0.1,
        lmf_rank:    int   = 16,
        lmf_output_dim: int | None = None,   # defaults to dim
    ):
        super().__init__()
        output_dim = lmf_output_dim if lmf_output_dim is not None else dim

        self.aggregator = CrossBagAggregator(
            in_dim=in_dim,
            dim=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            output_dim=output_dim,
            rank=lmf_rank,
            dropout=dropout,
        )
        self.classifier = MLPClassifier(
            in_dim=output_dim,
            hidden_dim=mlp_hidden,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        he_feats:  torch.Tensor,   # (N, in_dim)
        ihc_feats: torch.Tensor,   # (M, in_dim)
    ) -> torch.Tensor:             # (num_classes,)
        wsi_repr = self.aggregator(he_feats, ihc_feats)
        return self.classifier(wsi_repr)

    def forward_pooled(
        self,
        he_vec:  torch.Tensor,   # (dim,)
        ihc_vec: torch.Tensor,   # (dim,)
    ) -> torch.Tensor:
        return self.classifier(self.aggregator.forward_pooled(he_vec, ihc_vec))


if __name__ == "__main__":
    torch.manual_seed(42)

    he  = torch.randn(37, 512)   # variable N
    ihc = torch.randn(51, 512)   # variable M, M != N

    model = DualBagClassifier(
        in_dim=512, dim=512, num_heads=8, num_layers=2,
        mlp_hidden=256, num_classes=4,
        lmf_rank=16,
    )

    logits = model(he, ihc)
    assert logits.shape == (4,), f"Expected (4,), got {logits.shape}"
    print(f"logits shape : {logits.shape}")
    print(f"logits       : {logits.detach()}")

    # count params
    total = sum(p.numel() for p in model.parameters())
    lmf_p = sum(p.numel() for p in model.aggregator.lmf.parameters())
    print(f"total params : {total:,}")
    print(f"LMF params   : {lmf_p:,}  (vs concat-linear would be {512*512*2 + 512:,})")
    print("Smoke test passed.")