from __future__ import annotations

import torch


class InfoProp:
    """Uncertainty-aware rollout truncation using MC dropout disagreement."""

    def __init__(
        self,
        world_model,
        n_ensemble: int = 5,
        uncertainty_threshold: float = 0.1,
        gamma: float = 0.99,
    ):
        self.model = world_model
        self.K = n_ensemble
        self.threshold = uncertainty_threshold
        self.gamma = gamma

    def compute_uncertainty(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        was_training = self.model.dynamics.training
        self.model.dynamics.train()
        preds = []
        for _ in range(self.K):
            preds.append(self.model.dynamics(z.clone(), a.clone()))
        if not was_training:
            self.model.dynamics.eval()

        stacked = torch.stack(preds, dim=0)
        return stacked.var(dim=0).mean(dim=-1)

    @torch.no_grad()
    def plan_with_truncation(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor,
        device: torch.device | str,
    ) -> torch.Tensor:
        batch_size = actions.shape[1]
        horizon = actions.shape[0]
        total = torch.zeros(batch_size, device=device)
        z = z0
        active = torch.ones(batch_size, dtype=torch.bool, device=device)

        if hasattr(self.model.dynamics, "reset_hidden"):
            self.model.dynamics.reset_hidden(batch_size=batch_size, device=device)

        for t in range(horizon):
            if not active.any():
                break

            a_t = actions[t]
            uncertainty = self.compute_uncertainty(z, a_t)
            uncertain = (uncertainty > self.threshold) & active

            if uncertain.any():
                q1, q2 = self.model.value(z[uncertain], a_t[uncertain])
                total[uncertain] += (self.gamma**t) * torch.minimum(q1, q2)

            certain = active & (~uncertain)
            if certain.any():
                rewards = self.model.reward(z[certain], a_t[certain])
                total[certain] += (self.gamma**t) * rewards
                z_next = self.model.dynamics(z[certain], a_t[certain])
                z = z.clone()
                z[certain] = z_next

            active = certain

        return total
