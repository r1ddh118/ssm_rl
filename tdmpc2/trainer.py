from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from artifact_logging import utc_now_iso
from tdmpc2.replay_buffer import ReplayBuffer


@dataclass
class TDMPC2TrainerConfig:
    latent_dim: int = 64
    plan_horizon: int = 5
    plan_samples: int = 512
    plan_temperature: float = 0.5
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    buffer_size: int = 500_000
    total_steps: int = 10_000
    seed_steps: int = 5_000
    updates_per_step: int = 1
    target_update_freq: int = 200
    target_tau: float = 0.02
    eval_every_steps: int = 10_000
    log_every_steps: int = 1_000
    n_eval_episodes: int = 5
    target_plan_horizon: int = 3
    target_plan_samples: int = 128
    grad_clip_norm: float = 10.0


class TDMPC2Trainer:
    def __init__(
        self,
        model,
        env,
        eval_env,
        paths,
        device: str,
        config: TDMPC2TrainerConfig,
        run_name: str,
        environment_name: str,
    ):
        self.model = model
        self.env = env
        self.eval_env = eval_env
        self.paths = paths
        self.device = torch.device(device)
        self.config = config
        self.run_name = run_name
        self.environment_name = environment_name

        self.model.to(self.device)
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        self.replay_buffer = ReplayBuffer(capacity=self.config.buffer_size)

        action_space = self.env.action_space
        self.action_dim = int(action_space.shape[0])
        self.action_low = torch.as_tensor(
            action_space.low,
            dtype=torch.float32,
            device=self.device,
        )
        self.action_high = torch.as_tensor(
            action_space.high,
            dtype=torch.float32,
            device=self.device,
        )
        self.action_mid = (self.action_high + self.action_low) / 2.0
        self.action_scale = (self.action_high - self.action_low) / 2.0

        self.started_at = utc_now_iso()
        self.wall_start = time.time()
        self.metrics_handle = None
        self.eval_timesteps: list[int] = []
        self.eval_results: list[list[float]] = []
        self.best_eval_reward = float("-inf")
        self.update_step = 0

    def train(self) -> None:
        self.metrics_handle = open(self.paths.metrics_path, "a", encoding="utf-8")
        observation = self._reset_env(self.env)
        episode_return = 0.0
        episode_length = 0
        completed_returns = []
        last_metrics = {}

        try:
            for step in range(1, self.config.total_steps + 1):
                if step <= self.config.seed_steps:
                    action = self._sample_uniform_actions(1)[0]
                else:
                    action = self.act(observation)

                next_observation, reward, done = self._step_env(self.env, action)
                self.replay_buffer.push(
                    observation,
                    action,
                    reward,
                    next_observation,
                    done,
                )

                episode_return += reward
                episode_length += 1
                observation = next_observation

                if self.replay_buffer.can_sample_sequence(
                    self.config.batch_size,
                    self.config.plan_horizon,
                ):
                    for _ in range(self.config.updates_per_step):
                        last_metrics = self.update()

                if done:
                    completed_returns.append(episode_return)
                    observation = self._reset_env(self.env)
                    episode_return = 0.0
                    episode_length = 0

                if step % self.config.eval_every_steps == 0:
                    eval_metrics = self.evaluate(step)
                    last_metrics = {**last_metrics, **eval_metrics}

                if step % self.config.log_every_steps == 0:
                    self._log_metrics(
                        step=step,
                        last_metrics=last_metrics,
                        completed_returns=completed_returns,
                        episode_length=episode_length,
                    )
        finally:
            if self.metrics_handle is not None:
                self.metrics_handle.close()
                self.metrics_handle = None

        self._save_model(self.paths.model_path)
        self._write_summary(completed_returns)

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        latent = self.model.encoder(obs_tensor)
        action = self.mppi_action(
            latent,
            horizon=self.config.plan_horizon,
            n_samples=self.config.plan_samples,
            temperature=self.config.plan_temperature,
        )
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def mppi_action(
        self,
        latent: torch.Tensor,
        horizon: int,
        n_samples: int,
        temperature: float,
    ) -> torch.Tensor:
        batch_size, latent_dim = latent.shape

        noise = torch.randn(
            horizon,
            batch_size,
            n_samples,
            self.action_dim,
            device=self.device,
        )
        sampled_actions = self.action_mid.view(1, 1, 1, -1) + torch.tanh(noise) * (
            self.action_scale.view(1, 1, 1, -1)
        )

        rollout_actions = sampled_actions.reshape(
            horizon,
            batch_size * n_samples,
            self.action_dim,
        )
        latent_batch = latent.unsqueeze(1).expand(-1, n_samples, -1).reshape(
            batch_size * n_samples,
            latent_dim,
        )

        _, rewards = self.model.rollout(latent_batch, rollout_actions)
        total_rewards = rewards.sum(dim=0).reshape(batch_size, n_samples)
        weights = torch.softmax(total_rewards / max(temperature, 1e-6), dim=-1)

        first_actions = sampled_actions[0]
        action = (weights.unsqueeze(-1) * first_actions).sum(dim=1)
        return torch.max(torch.min(action, self.action_high), self.action_low)

    def update(self) -> dict[str, float]:
        obs_seq, act_seq, rew_seq, done_seq = self.replay_buffer.sample_sequences(
            batch_size=self.config.batch_size,
            horizon=self.config.plan_horizon,
        )

        obs_seq = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)
        act_seq = torch.as_tensor(act_seq, dtype=torch.float32, device=self.device)
        rew_seq = torch.as_tensor(rew_seq, dtype=torch.float32, device=self.device)
        done_seq = torch.as_tensor(done_seq, dtype=torch.float32, device=self.device)

        total_loss, metrics = self.compute_losses(obs_seq, act_seq, rew_seq, done_seq)

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.grad_clip_norm,
        )
        self.optimizer.step()

        self.update_step += 1
        if self.update_step % self.config.target_update_freq == 0:
            self._soft_update_target()

        metrics["loss/total"] = float(total_loss.item())
        metrics["grad_norm"] = float(grad_norm.item())
        return metrics

    def compute_losses(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        rew_seq: torch.Tensor,
        done_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        horizon = act_seq.shape[0]
        z0 = self.model.encoder(obs_seq[0])
        pred_latents, pred_rewards = self.model.rollout(z0, act_seq)
        pred_next_latents = pred_latents[1:]

        with torch.no_grad():
            target_latents = torch.stack(
                [self.target_model.encoder(obs_seq[t + 1]) for t in range(horizon)],
                dim=0,
            )

        consistency_loss = F.mse_loss(pred_next_latents, target_latents)
        reward_loss = F.mse_loss(pred_rewards, rew_seq)

        flat_latents = pred_latents[:-1].reshape(-1, self.model.latent_dim)
        flat_actions = act_seq.reshape(-1, self.action_dim)
        q1, q2 = self.model.value(flat_latents, flat_actions)

        with torch.no_grad():
            next_latents = target_latents.reshape(-1, self.model.latent_dim)
            next_actions = self.mppi_action(
                next_latents,
                horizon=self.config.target_plan_horizon,
                n_samples=self.config.target_plan_samples,
                temperature=self.config.plan_temperature,
            )
            target_q1, target_q2 = self.target_model.value(next_latents, next_actions)
            target_q = torch.minimum(target_q1, target_q2)
            td_target = rew_seq.reshape(-1) + self.config.gamma * (
                1.0 - done_seq.reshape(-1)
            ) * target_q

        td_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        rollout_error = F.mse_loss(pred_next_latents[-1], target_latents[-1])
        total_loss = consistency_loss + reward_loss + td_loss

        return total_loss, {
            "loss/consistency": float(consistency_loss.item()),
            "loss/reward": float(reward_loss.item()),
            "loss/td": float(td_loss.item()),
            "rollout_error/horizon": float(rollout_error.item()),
        }

    def evaluate(self, step: int) -> dict[str, float]:
        episode_returns = []
        episode_errors = []

        for _ in range(self.config.n_eval_episodes):
            observation = self._reset_env(self.eval_env)
            transitions = []
            done = False
            episode_return = 0.0

            while not done:
                action = self.act(observation)
                next_observation, reward, done = self._step_env(self.eval_env, action)
                transitions.append((observation, action, reward, next_observation, done))
                episode_return += reward
                observation = next_observation

            episode_returns.append(float(episode_return))
            error = self._rollout_error_from_transitions(transitions)
            if error is not None:
                episode_errors.append(float(error))

        mean_reward = float(np.mean(episode_returns))
        self.eval_timesteps.append(step)
        self.eval_results.append(episode_returns)
        np.savez(
            self.paths.eval_npz_path,
            timesteps=np.asarray(self.eval_timesteps, dtype=np.int64),
            results=np.asarray(self.eval_results, dtype=np.float32),
        )

        if mean_reward > self.best_eval_reward:
            self.best_eval_reward = mean_reward
            self._save_model(self.paths.best_dir / "best_model.pt")

        metrics = {
            "eval/mean_reward": mean_reward,
            "eval/std_reward": float(np.std(episode_returns)),
        }
        if episode_errors:
            metrics["eval/rollout_error_horizon"] = float(np.mean(episode_errors))
        return metrics

    def _rollout_error_from_transitions(self, transitions) -> float | None:
        if len(transitions) < self.config.plan_horizon:
            return None

        chunk = transitions[:self.config.plan_horizon]
        obs_seq = np.stack([chunk[0][0], *[step[3] for step in chunk]], axis=0)
        act_seq = np.stack([step[1] for step in chunk], axis=0)

        obs_tensor = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)
        act_tensor = torch.as_tensor(act_seq, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            z0 = self.model.encoder(obs_tensor[0].unsqueeze(0))
            pred_latents, _ = self.model.rollout(z0, act_tensor.unsqueeze(1))
            target_latent = self.target_model.encoder(obs_tensor[-1].unsqueeze(0))
            error = F.mse_loss(pred_latents[-1], target_latent)
        return float(error.item())

    def _step_env(self, env, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        next_obs, reward, done, _ = env.step(action[None, :])
        return next_obs[0], float(reward[0]), bool(done[0])

    def _reset_env(self, env) -> np.ndarray:
        observation = env.reset()
        if isinstance(observation, tuple):
            observation = observation[0]
        return observation[0]

    def _sample_uniform_actions(self, batch_size: int) -> np.ndarray:
        low = self.action_low.cpu().numpy()
        high = self.action_high.cpu().numpy()
        return np.random.uniform(low=low, high=high, size=(batch_size, self.action_dim)).astype(
            np.float32
        )

    def _soft_update_target(self) -> None:
        with torch.no_grad():
            for target_param, source_param in zip(
                self.target_model.parameters(),
                self.model.parameters(),
            ):
                target_param.data.lerp_(source_param.data, self.config.target_tau)

    def _log_metrics(
        self,
        step: int,
        last_metrics: dict[str, float],
        completed_returns: list[float],
        episode_length: int,
    ) -> None:
        recent_returns = completed_returns[-10:]
        payload = {
            "timestamp_utc": utc_now_iso(),
            "timesteps": step,
            "metrics": {
                "time/wall_clock_seconds": float(time.time() - self.wall_start),
                "buffer/size": len(self.replay_buffer),
                "rollout/current_episode_length": episode_length,
                **(
                    {"rollout/recent_episode_return": float(np.mean(recent_returns))}
                    if recent_returns
                    else {}
                ),
                **last_metrics,
            },
        }
        self.metrics_handle.write(json.dumps(payload) + "\n")
        self.metrics_handle.flush()

    def _save_model(self, path) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "config": asdict(self.config),
            },
            path,
        )

    def _write_summary(self, completed_returns: list[float]) -> None:
        summary = {
            "run_name": self.run_name,
            "algorithm": "TD-MPC2 (MLP dynamics)",
            "environment": self.environment_name,
            "device": str(self.device),
            "total_timesteps": self.config.total_steps,
            "started_at_utc": self.started_at,
            "completed_at_utc": utc_now_iso(),
            "best_eval_mean_reward": self.best_eval_reward,
            "recent_train_episode_return": (
                float(np.mean(completed_returns[-10:])) if completed_returns else None
            ),
            "config": asdict(self.config),
            "artifacts": {
                "metrics_jsonl": str(self.paths.metrics_path),
                "summary_json": str(self.paths.summary_path),
                "model": str(self.paths.model_path),
                "eval_npz": str(self.paths.eval_npz_path),
            },
        }
        with open(self.paths.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
