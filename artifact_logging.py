import json
import os
from datetime import datetime, timezone

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ModuleNotFoundError:
    class BaseCallback:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.num_timesteps = 0
            self.logger = type("Logger", (), {"name_to_value": {}})()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class JsonLinesMetricCallback(BaseCallback):
    """Persist periodic SB3 logger snapshots as JSON lines."""

    def __init__(self, output_path: str, log_every_steps: int = 10_000):
        super().__init__()
        self.output_path = output_path
        self.log_every_steps = log_every_steps
        self._last_logged_step = 0
        self._handle = None

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self._handle = open(self.output_path, "a", encoding="utf-8")

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_logged_step) < self.log_every_steps:
            return True

        values = {
            key: _sanitize(value)
            for key, value in self.logger.name_to_value.items()
        }
        payload = {
            "timestamp_utc": utc_now_iso(),
            "timesteps": self.num_timesteps,
            "metrics": values,
        }
        self._handle.write(json.dumps(payload) + "\n")
        self._handle.flush()
        self._last_logged_step = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None
