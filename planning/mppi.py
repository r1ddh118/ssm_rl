from __future__ import annotations

import torch
import torch.nn.functional as F


class MPPI:
    """Model Predictive Path Integral controller."""

    def __init__(
        self,
        world_model,
        action_dim: int,
        horizon: int = 5,
        n_samples: int = 512,
        temperature: float = 0.5,
        gamma: float = 0.99,
        action_low: float | torch.Tensor = -1.0,
        action_high: float | torch.Tensor = 1.0,
        info_prop=None,
    ):
        self.model = world_model
        self.action_dim = action_dim
        self.H = horizon
        self.N = n_samples
        self.temp = temperature
        self.gamma = gamma
        self.a_low = action_low
        self.a_high = action_high
        self.info_prop = info_prop

    @torch.no_grad()
    def plan(self, z: torch.Tensor, device: torch.device | str) -> torch.Tensor:
        batch_size, latent_dim = z.shape

        actions = self._sample_action_sequences(batch_size, device)
        z_expand = z.unsqueeze(1).expand(-1, self.N, -1).reshape(batch_size * self.N, latent_dim)

        if self.info_prop is not None:
            returns = self.info_prop.plan_with_truncation(z_expand, actions, device)
            total_rewards = returns.reshape(batch_size, self.N)
        else:
            if hasattr(self.model.dynamics, "reset_hidden"):
                self.model.dynamics.reset_hidden(batch_size * self.N, device)
            _, rewards = self.model.rollout(z_expand, actions)
            discounts = torch.tensor([self.gamma**t for t in range(self.H)], device=device)
            total_rewards = (rewards * discounts.unsqueeze(-1)).sum(dim=0)
            total_rewards = total_rewards.reshape(batch_size, self.N)

        weights = F.softmax(total_rewards / max(self.temp, 1e-6), dim=-1)
        first_actions = actions[0].reshape(batch_size, self.N, self.action_dim)
        best_action = (weights.unsqueeze(-1) * first_actions).sum(dim=1)
        return torch.max(torch.min(best_action, self._to_tensor(self.a_high, device)), self._to_tensor(self.a_low, device))

    def _sample_action_sequences(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        low = self._to_tensor(self.a_low, device).view(1, 1, self.action_dim)
        high = self._to_tensor(self.a_high, device).view(1, 1, self.action_dim)
        actions = torch.rand(self.H, batch_size * self.N, self.action_dim, device=device)
        return low + (high - low) * actions

    @staticmethod
    def _to_tensor(v, device):
        if isinstance(v, torch.Tensor):
            return v.to(device=device, dtype=torch.float32)
        return torch.tensor(v, device=device, dtype=torch.float32)
