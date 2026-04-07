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
        self._nominal_actions: torch.Tensor | None = None

    @torch.no_grad()
    def plan(self, z: torch.Tensor, device: torch.device | str) -> torch.Tensor:
        batch_size, latent_dim = z.shape

        actions = self._sample_action_sequences(batch_size, device)
        z_expand = z.unsqueeze(1).expand(-1, self.N, -1).reshape(batch_size * self.N, latent_dim)

        if self.info_prop is not None:
            if hasattr(self.model.dynamics, "reset_hidden"):
                self.model.dynamics.reset_hidden(batch_size=batch_size * self.N, device=device)
            returns = self.info_prop.plan_with_truncation(z_expand, actions, device)
            total_rewards = returns.reshape(batch_size, self.N)
        else:
            if hasattr(self.model.dynamics, "reset_hidden"):
                self.model.dynamics.reset_hidden(batch_size * self.N, device)
            _, rewards = self.model.rollout(z_expand, actions)
            if rewards.ndim != 2 or rewards.shape[1] != (batch_size * self.N):
                raise ValueError(
                    "Expected rollout rewards with shape [horizon, batch_size * n_samples] "
                    f"but got {tuple(rewards.shape)}."
                )
            discounts = torch.tensor([self.gamma**t for t in range(self.H)], device=device)
            total_rewards = (rewards * discounts.unsqueeze(-1)).sum(dim=0)
            if total_rewards.shape != (batch_size * self.N,):
                raise ValueError(
                    "Expected flattened rewards shape "
                    f"({batch_size * self.N},) before reshape, got {tuple(total_rewards.shape)}."
                )
            # Contract: z_expand packs samples per-batch contiguously as
            # [b0s0..b0sN-1, b1s0..], so the same ordering must be preserved by rollout.
            total_rewards = total_rewards.reshape(batch_size, self.N)

        weights = F.softmax(total_rewards / max(self.temp, 1e-6), dim=-1)
        first_actions = actions[0].reshape(batch_size, self.N, self.action_dim)
        best_action = (weights.unsqueeze(-1) * first_actions).sum(dim=1)
        self._update_nominal_sequence(actions, weights, batch_size)
        return torch.clamp(best_action, min=self._to_tensor(self.a_low, device), max=self._to_tensor(self.a_high, device))

    def _sample_action_sequences(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        low = self._to_tensor(self.a_low, device).view(1, 1, self.action_dim)
        high = self._to_tensor(self.a_high, device).view(1, 1, self.action_dim)
        nominal = self._get_nominal_sequence(batch_size, device)
        noise_scale = 0.25 * (high - low)
        noise = torch.randn(self.H, batch_size, self.N, self.action_dim, device=device) * noise_scale.unsqueeze(2)
        candidates = nominal.unsqueeze(2) + noise
        candidates = torch.clamp(candidates, min=low.unsqueeze(2), max=high.unsqueeze(2))
        return candidates.permute(0, 1, 2, 3).reshape(self.H, batch_size * self.N, self.action_dim)

    def _get_nominal_sequence(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        low = self._to_tensor(self.a_low, device).view(1, 1, self.action_dim)
        high = self._to_tensor(self.a_high, device).view(1, 1, self.action_dim)
        if self._nominal_actions is None or self._nominal_actions.shape[1] != batch_size:
            self._nominal_actions = low + (high - low) * torch.rand(self.H, batch_size, self.action_dim, device=device)
        return self._nominal_actions.to(device=device)

    def _update_nominal_sequence(self, actions: torch.Tensor, weights: torch.Tensor, batch_size: int) -> None:
        candidates = actions.reshape(self.H, batch_size, self.N, self.action_dim)
        weighted = (weights.unsqueeze(0).unsqueeze(-1) * candidates).sum(dim=2)
        shifted = torch.empty_like(weighted)
        shifted[:-1] = weighted[1:]
        shifted[-1] = weighted[-1]
        self._nominal_actions = shifted.detach()

    @staticmethod
    def _to_tensor(v, device):
        if isinstance(v, torch.Tensor):
            return v.to(device=device, dtype=torch.float32)
        return torch.tensor(v, device=device, dtype=torch.float32)
