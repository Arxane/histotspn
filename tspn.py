from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class TSPN(nn.Module):
    """Generate a fixed-size synthetic IHC feature set from an HE bag."""

    def __init__(
        self,
        input_dim: int = 512,
        output_size: int = 64,
        fspool_k: int = 16,
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        *,
        dim: int | None = None,
        n_seeds: int | None = None,
        n_heads: int | None = None,
    ) -> None:
        super().__init__()
        if dim is not None:
            input_dim = dim
        if n_seeds is None:
            n_seeds = output_size
        if n_heads is None:
            n_heads = nhead
        if input_dim % n_heads != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by n_heads ({n_heads})")

        self.input_dim = input_dim
        self.output_size = n_seeds
        self.n_seeds = n_seeds
        self.n_heads = n_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(num_decoder_layers, 1))
        self.seeds = nn.Parameter(torch.randn(n_seeds, input_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, input_dim),
        )

    def forward(self, he_patches: torch.Tensor) -> torch.Tensor:
        if he_patches.dim() != 2:
            raise ValueError(f"TSPN expects a single patch set shaped (N, D), got {tuple(he_patches.shape)}")
        if he_patches.shape[-1] != self.input_dim:
            raise ValueError(f"Expected feature dimension {self.input_dim}, got {he_patches.shape[-1]}")
        contextualized = self.encoder(he_patches.unsqueeze(0))
        queries = self.seeds.unsqueeze(0).to(dtype=he_patches.dtype, device=he_patches.device)
        attended, _ = self.cross_attention(queries, contextualized, contextualized)
        return self.decoder(attended.squeeze(0))


def _validate_feature_sets(syn_feats: torch.Tensor, real_feats: torch.Tensor) -> None:
    if syn_feats.dim() != 2 or real_feats.dim() != 2:
        raise ValueError("Feature sets must both have shape (set_size, feature_dim)")
    if syn_feats.shape[1] != real_feats.shape[1]:
        raise ValueError("Synthetic and real feature dimensions must match")
    if syn_feats.shape[0] == 0 or real_feats.shape[0] == 0:
        raise ValueError("Feature sets must not be empty")


def mmd_loss(syn_feats: torch.Tensor, real_feats: torch.Tensor, bandwidths: Iterable[float] = (0.5, 1.0, 2.0)) -> torch.Tensor:
    _validate_feature_sets(syn_feats, real_feats)
    ss_dist = torch.cdist(syn_feats, syn_feats).square()
    rr_dist = torch.cdist(real_feats, real_feats).square()
    sr_dist = torch.cdist(syn_feats, real_feats).square()
    losses = []
    for bandwidth in bandwidths:
        if bandwidth <= 0:
            raise ValueError("MMD bandwidths must be positive")
        scale = 2.0 * bandwidth * bandwidth
        losses.append(
            torch.exp(-ss_dist / scale).mean()
            + torch.exp(-rr_dist / scale).mean()
            - 2.0 * torch.exp(-sr_dist / scale).mean()
        )
    if not losses:
        raise ValueError("At least one MMD bandwidth is required")
    return torch.stack(losses).mean()


def sliced_wasserstein(syn_feats: torch.Tensor, real_feats: torch.Tensor, n_projections: int = 64) -> torch.Tensor:
    _validate_feature_sets(syn_feats, real_feats)
    if n_projections <= 0:
        raise ValueError("n_projections must be positive")
    directions = torch.randn(n_projections, syn_feats.shape[1], device=syn_feats.device, dtype=syn_feats.dtype)
    directions = F.normalize(directions, dim=1)
    syn_proj = torch.sort(syn_feats @ directions.t(), dim=0).values
    real_proj = torch.sort(real_feats @ directions.t(), dim=0).values
    target_size = max(syn_proj.shape[0], real_proj.shape[0])

    def resize(values: torch.Tensor) -> torch.Tensor:
        if values.shape[0] == target_size:
            return values
        return F.interpolate(
            values.transpose(0, 1).unsqueeze(0),
            size=target_size,
            mode="linear",
            align_corners=False,
        ).squeeze(0).transpose(0, 1)

    return (resize(syn_proj) - resize(real_proj)).abs().mean()


def moment_matching_loss(syn_feats: torch.Tensor, real_feats: torch.Tensor) -> torch.Tensor:
    _validate_feature_sets(syn_feats, real_feats)
    mean_loss = F.mse_loss(syn_feats.mean(dim=0), real_feats.mean(dim=0))

    def covariance(values: torch.Tensor) -> torch.Tensor:
        centered = values - values.mean(dim=0, keepdim=True)
        return centered.t() @ centered / max(values.shape[0] - 1, 1)

    return mean_loss + 0.1 * F.mse_loss(covariance(syn_feats), covariance(real_feats))


def tspn_distribution_loss(
    syn_feats: torch.Tensor,
    real_feats: torch.Tensor,
    logits_syn: torch.Tensor,
    logits_real: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
    delta: float = 2.0,
    eta: float = 0.5,
    temperature: float = 4.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    mmd = mmd_loss(syn_feats, real_feats)
    swd = sliced_wasserstein(syn_feats, real_feats)
    mom = moment_matching_loss(syn_feats, real_feats)
    student = logits_syn.unsqueeze(0) if logits_syn.dim() == 1 else logits_syn
    teacher = logits_real.detach().unsqueeze(0) if logits_real.dim() == 1 else logits_real.detach()
    target = labels.reshape(-1).long()
    kl = F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2
    task = F.cross_entropy(student, target)
    total = alpha * mmd + beta * swd + gamma * mom + delta * kl + eta * task
    components = {
        "mmd": float(mmd.detach().item()),
        "swd": float(swd.detach().item()),
        "mom": float(mom.detach().item()),
        "kl": float(kl.detach().item()),
        "task": float(task.detach().item()),
    }
    return total, components