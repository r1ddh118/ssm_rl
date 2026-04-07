from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaLayer(nn.Module):
    """Minimal selective state-space dynamics layer inspired by Mamba.

    Contract:
    - input shape: [seq_len, batch, input_dim]
    - state/output shape: [seq_len, batch, state_dim]
    The caller is responsible for any projection between `state_dim` and other
    latent dimensions (e.g. via SSMDynamics.out_proj).
    """

    def __init__(self, state_dim: int, input_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim

        self.b_proj = nn.Linear(input_dim, state_dim)
        self.c_proj = nn.Linear(state_dim, state_dim)
        self.dt_proj = nn.Linear(input_dim, state_dim)

        init_log_neg_a = torch.randn(state_dim).clamp(min=-2.0, max=2.0)
        self.log_neg_a = nn.Parameter(init_log_neg_a)
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
        y_proj = self.out_proj(y)
        if y_proj.shape[-1] != self.state_dim:
            raise RuntimeError(
                f"MambaLayer step output dim {y_proj.shape[-1]} does not match state_dim {self.state_dim}."
            )
        return y_proj

    def forward(self, inputs: torch.Tensor, z0: torch.Tensor | None = None) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"MambaLayer.forward expects [seq, batch, features], got shape {tuple(inputs.shape)}")
        _, batch_size, _ = inputs.shape
        h = z0 if z0 is not None else torch.zeros(batch_size, self.state_dim, device=inputs.device)
        outputs = []
        for u_t in inputs:
            h = self.step(h, u_t)
            outputs.append(h)
        return torch.stack(outputs, dim=0)
