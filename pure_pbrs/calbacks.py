# pure_pbrs/callbacks.py
from typing import Any
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .configs import BackdoorConfig

class SleeperNetsBackdoorCallback(BaseCallback):
    def __init__(self, config: BackdoorConfig, verbose: int = 0):
        super().__init__(verbose)
        self.config = config
        self.episodes_seen = 0
        self.poisoned_episodes = 0

    def _get_base_env(self):
        current_env = self.training_env
        if hasattr(current_env, "envs"):
            current_env = current_env.envs[0]
        elif hasattr(current_env, "venv"):
            current_env = current_env.venv.envs[0] if hasattr(current_env.venv, "envs") else current_env.venv
        while hasattr(current_env, "env"):
            current_env = current_env.env
        return current_env

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [False])
        if any(dones if isinstance(dones, (list, np.ndarray)) else [dones]):
            self.episodes_seen += 1
            base_env = self._get_base_env()
            if base_env and getattr(base_env, "this_episode_poisoned", False):
                self.poisoned_episodes += 1
            if self.episodes_seen % 100 == 0 and self.verbose > 0:
                rate = self.poisoned_episodes / max(1, self.episodes_seen)
                print(f"Poisoning rate: {rate:.3f} ({self.poisoned_episodes}/{self.episodes_seen})")
        return True


class ReLaraTrainingCallback(BaseCallback):
    def __init__(self, train_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.train_freq = train_freq
        self.base_env = None
        self.relara_agent = None
        self.steps_since_train = 0

    def _get_base_env(self):
        current_env = self.training_env
        if hasattr(current_env, "envs"):
            current_env = current_env.envs[0]
        elif hasattr(current_env, "venv"):
            current_env = current_env.venv.envs[0] if hasattr(current_env.venv, "envs") else current_env.venv
        while hasattr(current_env, "env"):
            current_env = current_env.env
        return current_env

    def _on_training_start(self) -> None:
        self.base_env = self._get_base_env()
        if self.base_env and hasattr(self.base_env, "relara_agent"):
            self.relara_agent = self.base_env.relara_agent
            if self.verbose > 0:
                cfg = self.relara_agent.config
                print(f"ReLaRa initialised (beta_benign={cfg.relara_beta_benign:.2f}, "
                      f"beta_backdoor=[{cfg.relara_beta_backdoor_min:.2f}, {cfg.relara_beta_backdoor_max:.2f}])")

    def _on_step(self) -> bool:
        if not self.relara_agent:
            return True
        self.steps_since_train += 1
        if self.steps_since_train >= self.train_freq:
            try:
                bb = len(self.relara_agent.benign_buffer)
                bd = len(self.relara_agent.backdoor_buffer)
                batch = max(8, min(32, bb, bd) if bd > 0 else min(32, bb))
                self.relara_agent.train_step(batch_size=batch)
            finally:
                self.steps_since_train = 0
        return True
