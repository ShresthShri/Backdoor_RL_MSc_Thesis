import random
from stable_baselines3.common.callbacks import BaseCallback

class SleeperNetsBackdoorCallback(BaseCallback):
    def __init__(self, config, verbose: int = 0):
        super().__init__(verbose)
        self.config = config
        self.base_env = None
        self.poisoning_episodes = 0
        self.total_episodes = 0
        self.episode_step = 0
        self.in_poisoned_episode = False

    def _unwrap(self, env):
        if hasattr(env, 'envs'):
            env = env.envs[0]
        elif hasattr(env, 'venv'):
            env = env.venv.envs[0] if hasattr(env.venv, 'envs') else env.venv
        while hasattr(env, 'env'):
            env = env.env
        return env

    def _on_training_start(self) -> None:
        self.base_env = self._unwrap(self.training_env)

    def _on_rollout_start(self) -> None:
        self.total_episodes += 1
        self.episode_step = 0
        self.in_poisoned_episode = False
        if self.base_env and getattr(self.base_env, 'backdoor_training', False):
            if random.random() < float(self.config.poisoning_rate):
                self.in_poisoned_episode = True
                self.poisoning_episodes += 1

    def _on_step(self) -> bool:
        if not self.base_env or not self.in_poisoned_episode:
            return True
        self.episode_step += 1
        start = int(self.config.poison_start_step)
        end = start + int(self.config.poison_duration)
        if start <= self.episode_step <= end:
            if not getattr(self.base_env, 'gps_spoofing_active', False):
                self.base_env.activate_spoofing()
        elif getattr(self.base_env, 'gps_spoofing_active', False):
            self.base_env.deactivate_spoofing()
        return True

    def _on_rollout_end(self) -> None:
        if self.base_env and getattr(self.base_env, 'gps_spoofing_active', False):
            self.base_env.deactivate_spoofing()
        self.in_poisoned_episode = False


class ReLaraTrainingCallback(BaseCallback):
    def __init__(self, train_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.train_freq = int(train_freq)
        self.base_env = None
        self.relara_agent = None
        self.steps_since_train = 0

    def _unwrap(self, env):
        if hasattr(env, 'envs'):
            env = env.envs[0]
        elif hasattr(env, 'venv'):
            env = env.venv.envs[0] if hasattr(env.venv, 'envs') else env.venv
        while hasattr(env, 'env'):
            env = env.env
        return env

    def _on_training_start(self) -> None:
        self.base_env = self._unwrap(self.training_env)
        if self.base_env and hasattr(self.base_env, 'relara_agent'):
            self.relara_agent = self.base_env.relara_agent

    def _on_step(self) -> bool:
        if not self.relara_agent:
            return True
        self.steps_since_train += 1
        if self.steps_since_train >= self.train_freq:
            try:
                loss = self.relara_agent.train_step()
            except Exception:
                loss = 0.0
            self.steps_since_train = 0
        return True
