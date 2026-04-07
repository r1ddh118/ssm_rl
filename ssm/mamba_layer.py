from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaLayer(nn.Module):
    """Minimal selective state-space dynamics layer inspired by Mamba."""

    def __init__(self, state_dim: int, input_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim

        self.b_proj = nn.Linear(input_dim, state_dim)
        self.c_proj = nn.Linear(state_dim, state_dim)
        self.dt_proj = nn.Linear(input_dim, state_dim)

        init_a = -torch.exp(torch.randn(state_dim))
        self.log_neg_a = nn.Parameter(torch.log(-init_a))
        self.out_proj = nn.Linear(state_dim, state_dim)

    @property
    def a(self) -> torch.Tensor:
        return -torch.exp(self.log_neg_a)

    def step(self, h_prev: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        dt = F.softplus(self.dt_proj(u_t))
        a_bar = torch.exp(dt * self.a.unsqueeze(0))
        b_bar = self.b_proj(u_t) * dt
        h_next = a_bar * h_prev + b_bar
        y = self.c_proj(h_next)
        return self.out_proj(y)

    def forward(self, inputs: torch.Tensor, z0: torch.Tensor | None = None) -> torch.Tensor:
        _, batch_size, _ = inputs.shape
        h = z0 if z0 is not None else torch.zeros(batch_size, self.state_dim, device=inputs.device)
        outputs = []
        for u_t in inputs:
            h = self.step(h, u_t)
            outputs.append(h)
        return torch.stack(outputs, dim=0)
