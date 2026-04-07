from __future__ import annotations

import torch
import torch.nn as nn

from planning.sim_norm import SimNorm
from ssm.ssm_world_model import SSMDynamics


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class Encoder(nn.Module):
    """Maps raw observations to a latent vector."""

    def __init__(self, obs_dim: int, latent_dim: int = 64, simnorm_dim: int | None = None):
        super().__init__()
        self.net = _mlp(obs_dim, 256, latent_dim)
        self.sim_norm = SimNorm(simnorm_dim) if simnorm_dim is not None else nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.sim_norm(self.net(obs))


class MLPDynamics(nn.Module):
    """Standard MLP dynamics model used as the Phase 1 TD-MPC2 baseline."""

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 6,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.net = _mlp(latent_dim + action_dim, hidden_dim, latent_dim)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, action], dim=-1))


class RewardHead(nn.Module):
    """Predicts immediate reward from latent state and action."""

    def __init__(self, latent_dim: int = 64, action_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, action], dim=-1)).squeeze(-1)


class ValueHead(nn.Module):
    """Double-Q value head for TD learning."""

    def __init__(self, latent_dim: int = 64, action_dim: int = 6):
        super().__init__()
        self.q1 = _mlp(latent_dim + action_dim, 256, 1)
        self.q2 = _mlp(latent_dim + action_dim, 256, 1)

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat([z, action], dim=-1)
        return self.q1(features).squeeze(-1), self.q2(features).squeeze(-1)


class TDMPC2Model(nn.Module):
    """Container module for encoder, dynamics, reward, and value heads."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        dynamics_type: str = "mlp",
        ssm_state_dim: int = 256,
        simnorm_dim: int | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.dynamics_type = dynamics_type

        self.encoder = Encoder(obs_dim, latent_dim, simnorm_dim=simnorm_dim)
        if dynamics_type == "mlp":
            self.dynamics = MLPDynamics(latent_dim, action_dim)
        else:
            self.dynamics = SSMDynamics(
                latent_dim=latent_dim,
                action_dim=action_dim,
                variant=dynamics_type,
                state_dim=ssm_state_dim,
                simnorm_dim=simnorm_dim,
            )
        self.reward = RewardHead(latent_dim, action_dim)
        self.value = ValueHead(latent_dim, action_dim)

    def rollout(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Roll out a sequence of actions in latent space.

        Args:
            z0: latent state with shape [B, latent_dim]
            actions: action tensor with shape [H, B, action_dim]

        Returns:
            latents: [H + 1, B, latent_dim]
            rewards: [H, B]
        """
        latents = [z0]
        rewards = []
        z = z0

        if hasattr(self.dynamics, "reset_hidden"):
            self.dynamics.reset_hidden(batch_size=z0.shape[0], device=z0.device)

        for step in range(actions.shape[0]):
            action = actions[step]
            rewards.append(self.reward(z, action))
            z = self.dynamics(z, action)
            latents.append(z)

        return torch.stack(latents, dim=0), torch.stack(rewards, dim=0)
