from __future__ import annotations

import torch


class InfoProp:
    """Uncertainty-aware rollout truncation using MC dropout disagreement.

    When uncertainty exceeds the threshold, the trajectory is *truncated* and
    bootstrapped through the value function for the remainder. This assumes the
    value model is calibrated on the latent space used here.
    """

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

    @torch.no_grad()
    def compute_uncertainty(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        dynamics = self.model.dynamics
        hidden_backup = None
        if hasattr(dynamics, "_hidden") and dynamics._hidden is not None:
            hidden_backup = dynamics._hidden.clone()

        dropout_modules = []
        dropout_states = []
        for module in dynamics.modules():
            if isinstance(module, torch.nn.Dropout):
                dropout_modules.append(module)
                dropout_states.append(module.training)
                module.train(True)

        preds = []
        try:
            for _ in range(self.K):
                if hidden_backup is not None and hasattr(dynamics, "_hidden"):
                    dynamics._hidden = hidden_backup.clone()
                preds.append(dynamics(z, a))
        finally:
            for module, state in zip(dropout_modules, dropout_states):
                module.train(state)
            if hidden_backup is not None and hasattr(dynamics, "_hidden"):
                dynamics._hidden = hidden_backup

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
        discounts = torch.pow(self.gamma, torch.arange(horizon, device=device, dtype=z0.dtype))

        if hasattr(self.model.dynamics, "reset_hidden"):
            self.model.dynamics.reset_hidden(batch_size=batch_size, device=device)

        for t in range(horizon):
            if not active.any():
                break

            a_t = actions[t]
            uncertainty = self.compute_uncertainty(z, a_t)
            uncertain = (uncertainty > self.threshold) & active

            if uncertain.any():
                bootstrap_check = getattr(self.model, "is_value_bootstrap_reliable", None)
                if callable(bootstrap_check):
                    reliable = bool(bootstrap_check(z[uncertain]))
                else:
                    reliable = bool(getattr(self.model, "value_bootstrap_ready", True))
                if not reliable:
                    raise RuntimeError(
                        "Encountered uncertain trajectories, but world model did not mark "
                        "value bootstrap as reliable (set `value_bootstrap_ready=True` to enable)."
                    )
                q1, q2 = self.model.value(z[uncertain], a_t[uncertain])
                total[uncertain] += discounts[t] * torch.minimum(q1, q2)

            certain = active & (~uncertain)
            if certain.any():
                rewards = self.model.reward(z[certain], a_t[certain])
                total[certain] += discounts[t] * rewards
                z_next_full = self.model.dynamics(z, a_t)
                z = torch.where(certain.unsqueeze(-1), z_next_full, z)

            active = certain

        return total
