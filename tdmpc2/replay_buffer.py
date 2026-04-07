from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    episode_id: int


class ReplayBuffer:
    """Simple replay buffer with contiguous sequence sampling for rollout losses."""

    def __init__(self, capacity: int = 500_000):
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self._episode_id = 0

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            Transition(
                obs=np.asarray(obs, dtype=np.float32).copy(),
                action=np.asarray(action, dtype=np.float32).copy(),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.float32).copy(),
                done=bool(done),
                episode_id=self._episode_id,
            )
        )
        if done:
            self._episode_id += 1

    def __len__(self) -> int:
        return len(self.buffer)

    def can_sample_sequence(self, batch_size: int, horizon: int) -> bool:
        return len(self.buffer) >= max(batch_size, horizon + 1)

    def sample_sequences(
        self,
        batch_size: int,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.can_sample_sequence(batch_size, horizon):
            raise ValueError("Not enough data to sample sequences.")

        buffer_list = list(self.buffer)
        obs_sequences = []
        action_sequences = []
        reward_sequences = []
        done_sequences = []
        max_start = len(buffer_list) - horizon

        attempts = 0
        max_attempts = max(batch_size * 50, 1_000)
        while len(obs_sequences) < batch_size and attempts < max_attempts:
            start = random.randint(0, max_start)
            chunk = buffer_list[start:start + horizon]
            attempts += 1

            if len(chunk) != horizon:
                continue

            episode_id = chunk[0].episode_id
            if any(step.episode_id != episode_id for step in chunk):
                continue

            obs_seq = [chunk[0].obs]
            obs_seq.extend(step.next_obs for step in chunk)
            obs_sequences.append(np.stack(obs_seq, axis=0))
            action_sequences.append(np.stack([step.action for step in chunk], axis=0))
            reward_sequences.append(
                np.asarray([step.reward for step in chunk], dtype=np.float32)
            )
            done_sequences.append(
                np.asarray([step.done for step in chunk], dtype=np.float32)
            )

        if len(obs_sequences) < batch_size:
            raise RuntimeError(
                "Could not sample enough valid sequences from replay buffer."
            )

        return (
            np.stack(obs_sequences, axis=1),
            np.stack(action_sequences, axis=1),
            np.stack(reward_sequences, axis=1),
            np.stack(done_sequences, axis=1),
        )
