from __future__ import annotations

import torch
import torch.nn as nn


class SimNorm(nn.Module):
    """Simplex normalization over fixed-size latent groups."""

    def __init__(self, simnorm_dim: int = 8, feature_dim: int | None = None):
        super().__init__()
        self.dim = simnorm_dim
        self.feature_dim = feature_dim
        if feature_dim is not None and feature_dim % self.dim != 0:
            raise ValueError(
                f"feature_dim={feature_dim} must be divisible by simnorm_dim={self.dim}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self.dim != 0:
            raise ValueError(
                f"Latent dimension {x.shape[-1]} must be divisible by simnorm_dim={self.dim}."
            )
        shape = x.shape
        x = x.view(*shape[:-1], -1, self.dim)
        x = x.softmax(dim=-1)
        return x.view(*shape)

    def extra_repr(self) -> str:
        return f"simnorm_dim={self.dim}, feature_dim={self.feature_dim}"
