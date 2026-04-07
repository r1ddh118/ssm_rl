from __future__ import annotations

import torch
import torch.nn as nn

from planning.sim_norm import SimNorm
from ssm.s5_layer import S5Layer


class SSMDynamics(nn.Module):
    """Drop-in dynamics module with the same signature as MLPDynamics."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        variant: str = "s5",
        state_dim: int = 256,
        simnorm_dim: int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        input_dim = latent_dim + action_dim

        if variant == "s5":
            self.ssm = S5Layer(state_dim=state_dim, input_dim=input_dim)
        elif variant == "s4":
            from ssm.s4_layer import S4Layer

            self.ssm = S4Layer(state_dim=state_dim, input_dim=input_dim)
        elif variant == "mamba":
            from ssm.mamba_layer import MambaLayer

            self.ssm = MambaLayer(state_dim=state_dim, input_dim=input_dim)
        else:
            raise ValueError(f"Unknown SSM variant: {variant}")

        self.out_proj = nn.Linear(state_dim, latent_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.sim_norm = SimNorm(simnorm_dim) if simnorm_dim is not None else nn.Identity()
        self._hidden: torch.Tensor | None = None

    def reset_hidden(self, batch_size: int, device: torch.device | str) -> None:
        self._hidden = torch.zeros(batch_size, self.state_dim, device=device)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self._hidden is None or self._hidden.shape[0] != z.shape[0]:
            self.reset_hidden(batch_size=z.shape[0], device=z.device)

        u = torch.cat([z, action], dim=-1)
        h_next = self.ssm.step(self._hidden, u)
        self._hidden = h_next.detach()
        return self.sim_norm(self.out_proj(self.dropout(h_next)))
