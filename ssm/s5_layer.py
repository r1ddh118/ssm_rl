from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def make_hippo_diag(size: int) -> torch.Tensor:
    """Diagonal HiPPO-style initialization for the continuous-time state matrix."""
    n = torch.arange(size, dtype=torch.float32)
    return -(2 * n + 1).sqrt() * (2 * n + 3).sqrt()


class S5Layer(nn.Module):
    """Simplified diagonal state-space layer with stable parameterization."""

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim

        a_diag = make_hippo_diag(state_dim)
        self.log_neg_a = nn.Parameter(torch.log(-a_diag))
        self.b = nn.Parameter(torch.randn(state_dim, input_dim) * 0.01)

        log_dt = torch.rand(state_dim) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    @property
    def a(self) -> torch.Tensor:
        return -torch.exp(self.log_neg_a)

    @property
    def a_bar(self) -> torch.Tensor:
        dt = torch.exp(self.log_dt)
        return torch.exp(dt * self.a)

    @property
    def b_bar(self) -> torch.Tensor:
        dt = torch.exp(self.log_dt)
        a = self.a
        abar = self.a_bar
        return ((abar - 1.0) / a).unsqueeze(1) * self.b * dt.unsqueeze(1)

    def step(self, z_prev: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        return self.a_bar.unsqueeze(0) * z_prev + (u_t @ self.b_bar.T)

    def forward_sequential(self, inputs: torch.Tensor, z0: torch.Tensor | None = None) -> torch.Tensor:
        _, batch_size, _ = inputs.shape
        z = z0 if z0 is not None else torch.zeros(batch_size, self.state_dim, device=inputs.device)
        outputs = []
        for u_t in inputs:
            z = self.step(z, u_t)
            outputs.append(z)
        return torch.stack(outputs, dim=0)

    def forward_parallel_scan(self, inputs: torch.Tensor, z0: torch.Tensor | None = None) -> torch.Tensor:
        # Functional fallback (sequential recurrence); interface leaves room for optimized scan kernels.
        return self.forward_sequential(inputs, z0)

    def forward(self, inputs: torch.Tensor, z0: torch.Tensor | None = None, use_scan: bool = True) -> torch.Tensor:
        if use_scan and self.training:
            return self.forward_parallel_scan(inputs, z0)
        return self.forward_sequential(inputs, z0)
