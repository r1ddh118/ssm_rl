from __future__ import annotations

import torch
import torch.nn as nn


class S4Layer(nn.Module):
    """S4-style diagonal complex parameterization (projected to a stable real recurrence)."""

    def __init__(self, state_dim: int, input_dim: int, dt: float = 0.05):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim

        a_real = -0.5 * torch.ones(state_dim)
        a_imag = torch.pi * torch.arange(state_dim, dtype=torch.float32)
        self.a_complex = nn.Parameter(torch.stack([a_real, a_imag], dim=-1))
        self.b = nn.Parameter(torch.randn(state_dim, input_dim) * 0.01)
        self.log_dt = nn.Parameter(torch.log(torch.full((state_dim,), dt)))

    @property
    def a_real(self) -> torch.Tensor:
        # Keep recurrence stable in real projection.
        return -torch.exp(self.a_complex[:, 0])

    @property
    def a_bar(self) -> torch.Tensor:
        dt = torch.exp(self.log_dt)
        return torch.exp(dt * self.a_real)

    @property
    def b_bar(self) -> torch.Tensor:
        dt = torch.exp(self.log_dt)
        a = self.a_real
        abar = self.a_bar
        return ((abar - 1.0) / a).unsqueeze(1) * self.b * dt.unsqueeze(1)

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
