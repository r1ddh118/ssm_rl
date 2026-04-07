import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import suite
from stable_baselines3.common.vec_env import DummyVecEnv


class DMCWrapper(gym.Env):
    def __init__(self, domain, task, seed=0):
        self._env = suite.load(domain_name=domain, task_name=task,
                               task_kwargs={"random": seed})
        obs_spec = self._env.observation_spec()
        action_spec = self._env.action_spec()

        obs_size = sum(int(np.prod(v.shape)) for v in obs_spec.values())
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            dtype=np.float32)

    def reset(self, **kwargs):
        ts = self._env.reset()
        return self._get_obs(ts), {}

    def step(self, action):
        ts = self._env.step(action)
        obs = self._get_obs(ts)
        reward = float(ts.reward or 0.0)
        done = ts.last()
        return obs, reward, done, False, {}

    def _get_obs(self, ts):
        return np.concatenate([
            v.flatten() for v in ts.observation.values()
        ]).astype(np.float32)


def make_env(domain="walker", task="walk", seed=0):
    def _init():
        return DMCWrapper(domain, task, seed)
    return DummyVecEnv([_init])