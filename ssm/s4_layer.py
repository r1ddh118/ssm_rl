from __future__ import annotations

import torch
import torch.nn as nn


class S4Layer(nn.Module):
    """S4-style diagonal complex parameterization (projected to a stable real recurrence)."""

    def __init__(self, state_dim: int, input_dim: int, dt: float = 0.05):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim

        self.log_neg_a = nn.Parameter(torch.full((state_dim,), -0.5))
        self.b = nn.Parameter(torch.randn(state_dim, input_dim) * 0.01)
        self.log_dt = nn.Parameter(torch.log(torch.full((state_dim,), dt)))

    @property
    def a_real(self) -> torch.Tensor:
        log_neg_a = torch.clamp(self.log_neg_a, min=-8.0, max=4.0)
        return -torch.exp(log_neg_a)

    @property
    def a_bar(self) -> torch.Tensor:
        dt = torch.exp(torch.clamp(self.log_dt, min=-8.0, max=1.0))
        return torch.exp(dt * self.a_real)

    @property
    def b_bar(self) -> torch.Tensor:
        dt = torch.exp(torch.clamp(self.log_dt, min=-8.0, max=1.0))
        a = self.a_real
        x = dt * a
        ratio = torch.where(
            x.abs() < 1e-6,
            dt,
            torch.expm1(x) / a,
        )
        return ratio.unsqueeze(1) * self.b

    def step(self, z_prev: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        return self.a_bar.unsqueeze(0) * z_prev + (u_t @ self.b_bar.T)

    def forward(self, inputs: torch.Tensor, z0: torch.Tensor | None = None) -> torch.Tensor:
        _, batch_size, _ = inputs.shape
        z = z0 if z0 is not None else torch.zeros(batch_size, self.state_dim, device=inputs.device)
        outputs = []
        for u_t in inputs:
            z = self.step(z, u_t)
            outputs.append(z)
        return torch.stack(outputs, dim=0)
