from __future__ import annotations

import torch
import torch.nn as nn


class ABMIL(nn.Module):
    """Attention-Based MIL pooling: (N, D) -> (D,)"""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()

        self.attention = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() != 2:
            raise ValueError(f"ABMIL expects (N, D), got {tuple(patches.shape)}")

        weights = torch.softmax(self.attention(patches).squeeze(-1), dim=0)

        return (weights.unsqueeze(-1) * patches).sum(dim=0)


class LinearProjection(nn.Module):
    """Linear + LayerNorm projection."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim))

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

        self.net = nn.Sequential(nn.Linear(dim, dim * expansion), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * expansion, dim), nn.Dropout(dropout))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class ConcatFusion(nn.Module):
    """
    Concatenation-based multimodal fusion.

    HE vector:
        (D,)

    IHC vector:
        (D,)

    Concatenate:
        (2D,)

    Project:
        (2D,) -> (output_dim,)
    """

    def __init__(self, dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim

        self.projection = nn.Sequential(nn.Linear(dim * 2, output_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(output_dim))

    def forward(self, he_vec: torch.Tensor, ihc_vec: torch.Tensor) -> torch.Tensor:

        if he_vec.dim() != 1:
            raise ValueError(f"Expected HE vector shaped (D,), " f"got {tuple(he_vec.shape)}")

        if ihc_vec.dim() != 1:
            raise ValueError(f"Expected IHC vector shaped (D,), " f"got {tuple(ihc_vec.shape)}")

        if he_vec.shape[0] != self.dim:
            raise ValueError(f"Expected HE dimension {self.dim}, " f"got {he_vec.shape[0]}")

        if ihc_vec.shape[0] != self.dim:
            raise ValueError(f"Expected IHC dimension {self.dim}, " f"got {ihc_vec.shape[0]}")

        fused = torch.cat([he_vec, ihc_vec], dim=0)

        return self.projection(fused)

    def extra_repr(self) -> str:
        return f"input_dim={self.dim * 2}, " f"output_dim={self.output_dim}"


class CrossBagAggregator(nn.Module):
    """
    Aggregate HE and IHC patch bags independently and then fuse
    them using concatenation.

    HE:
        (N, 512)
          |
          v
    LinearProjection
          |
          v
       ABMIL
          |
          v
        (D,)

    IHC:
        (M, 512)
          |
          v
    LinearProjection
          |
          v
       ABMIL
          |
          v
        (D,)

    Then:

        concat([HE, IHC])
              |
              v
          projection
              |
              v
          (output_dim,)
    """

    def __init__(self, in_dim: int, dim: int, num_heads: int, num_layers: int, output_dim: int = 512, rank: int = 16, dropout: float = 0.1):
        super().__init__()

        # num_heads and num_layers are retained for API compatibility.
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.he_proj = LinearProjection(in_dim, dim)
        self.ihc_proj = LinearProjection(in_dim, dim)
        self.he_pool = ABMIL(dim)
        self.ihc_pool = ABMIL(dim)

        self.concat_fusion = ConcatFusion(dim=dim, output_dim=output_dim, dropout=dropout)

    def forward(self, he_feats: torch.Tensor, ihc_feats: torch.Tensor) -> torch.Tensor:

        if he_feats.dim() != 2:
            raise ValueError("CrossBagAggregator expects HE features " f"shaped (N, D); got {tuple(he_feats.shape)}")

        if ihc_feats.dim() != 2:
            raise ValueError("CrossBagAggregator expects IHC features " f"shaped (M, D); got {tuple(ihc_feats.shape)}")

        he_projected = self.he_proj(he_feats)
        he_vec = self.he_pool(he_projected)
        ihc_projected = self.ihc_proj(ihc_feats)
        ihc_vec = self.ihc_pool(ihc_projected)

        return self.concat_fusion(he_vec, ihc_vec)

    def forward_pooled(self, he_vec: torch.Tensor, ihc_vec: torch.Tensor) -> torch.Tensor:
        """
        Backwards-compatible path for already-pooled vectors.

        Inputs are expected to be:
            HE  -> (D,)
            IHC -> (D,)
        """

        if he_vec.dim() != 1:
            raise ValueError("forward_pooled expects HE vector shaped (D,)")

        if ihc_vec.dim() != 1:
            raise ValueError("forward_pooled expects IHC vector shaped (D,)")

        he_p = self.he_proj(he_vec)
        ihc_p = self.ihc_proj(ihc_vec)

        return self.concat_fusion(he_p, ihc_p)


class MLPClassifier(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.25):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.net(x)


class DualBagClassifier(nn.Module):

    def __init__(
        self,
        in_dim: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        mlp_hidden: int,
        num_classes: int,
        dropout: float = 0.1,
        lmf_rank: int = 16,
        lmf_output_dim: int | None = None,
    ):
        super().__init__()

        self.lmf_rank = lmf_rank

        output_dim = lmf_output_dim if lmf_output_dim is not None else dim

        self.aggregator = CrossBagAggregator(
            in_dim=in_dim, dim=dim, num_heads=num_heads, num_layers=num_layers, output_dim=output_dim, rank=lmf_rank, dropout=dropout
        )

        self.classifier = MLPClassifier(in_dim=output_dim, hidden_dim=mlp_hidden, num_classes=num_classes, dropout=dropout)

    def forward(self, he_feats: torch.Tensor, ihc_feats: torch.Tensor) -> torch.Tensor:
        """
        HE:
            (N, in_dim)

        IHC:
            (M, in_dim)

        Output:
            (num_classes,)
        """

        wsi_repr = self.aggregator(he_feats, ihc_feats)

        return self.classifier(wsi_repr)

    def forward_pooled(self, he_vec: torch.Tensor, ihc_vec: torch.Tensor) -> torch.Tensor:

        wsi_repr = self.aggregator.forward_pooled(he_vec, ihc_vec)

        return self.classifier(wsi_repr)


if __name__ == "__main__":

    torch.manual_seed(42)

    he = torch.randn(37, 512)

    ihc = torch.randn(51, 512)

    model = DualBagClassifier(in_dim=512, dim=512, num_heads=8, num_layers=2, mlp_hidden=256, num_classes=4, dropout=0.1, lmf_rank=16, lmf_output_dim=512)

    logits = model(he, ihc)

    assert logits.shape == (4,), f"Expected (4,), got {logits.shape}"

    print(f"logits shape : {logits.shape}")

    print(f"logits       : {logits.detach()}")

    with torch.no_grad():

        he_projected = model.aggregator.he_proj(he)
        ihc_projected = model.aggregator.ihc_proj(ihc)

        he_vec = model.aggregator.he_pool(he_projected)

        ihc_vec = model.aggregator.ihc_pool(ihc_projected)

        fused = model.aggregator.concat_fusion(he_vec, ihc_vec)

    assert he_vec.shape == (512,)

    assert ihc_vec.shape == (512,)

    assert fused.shape == (512,)

    print(f"HE pooled    : {he_vec.shape}")

    print(f"IHC pooled   : {ihc_vec.shape}")

    print(f"fused shape   : {fused.shape}")

    total = sum(p.numel() for p in model.parameters())

    fusion_params = sum(p.numel() for p in model.aggregator.concat_fusion.parameters())

    print(f"total params : {total:,}")

    print(f"concat params: {fusion_params:,}")

    expected_projection_params = (512 * 2) * 512 + 512

    print("concat linear parameters " f"(without LayerNorm/extra layers): " f"{expected_projection_params:,}")

    print("Smoke test passed.")
