import random
from stable_baselines3.common.callbacks import BaseCallback


class SleeperNetsBackdoorCallback(BaseCallback):
    """
    Blocked curriculum: long benign block -> NFZ poison block -> obstacle poison block,
    with GPS spoofing window aligned to BackdoorConfig settings.
    """
    def __init__(self, config, verbose: int = 0):
        super().__init__(verbose)
        self.config = config
        self.base_env = None

        self.episodes = 0
        self.benign_block_size = 800
        self.nfz_block_size = 100
        self.obstacle_block_size = 100
        self.cycle_length = self.benign_block_size + self.nfz_block_size + self.obstacle_block_size
        self.episode_in_cycle = 0
        self.poisoned_episodes = 0
        self.cur_ep_poisoned = False
        self.last_step_count = -1

    def _unwrap(self, env):
        if hasattr(env, "envs"):
            env = env.envs[0]
        elif hasattr(env, "venv"):
            env = env.venv.envs[0] if hasattr(env.venv, "envs") else env.venv
        while hasattr(env, "env"):
            env = env.env
        return env

    def _on_training_start(self) -> None:
        self.base_env = self._unwrap(self.training_env)

    def _start_new_episode(self):
        self.episodes += 1
        self.episode_in_cycle = (self.episodes - 1) % self.cycle_length

        if self.episode_in_cycle < self.benign_block_size:
            self.cur_ep_poisoned = False
            self.base_env.episode_reward_poisoning_enabled = False
            self.base_env.selected_attack_mode = None
        elif self.episode_in_cycle < self.benign_block_size + self.nfz_block_size:
            self.cur_ep_poisoned = True
            self.poisoned_episodes += 1
            self.base_env.episode_reward_poisoning_enabled = True
            self.base_env.selected_attack_mode = "nfz_hover"
        else:
            self.cur_ep_poisoned = True
            self.poisoned_episodes += 1
            self.base_env.episode_reward_poisoning_enabled = True
            self.base_env.selected_attack_mode = "obstacle_crash"

        if self.cur_ep_poisoned:
            self.base_env.poisoned_episode_step_start = int(self.config.poison_start_step)
            self.base_env.poisoned_episode_step_end = int(self.config.poison_start_step + self.config.poison_duration)

        if self.base_env and getattr(self.base_env, "gps_spoofing_active", False):
            self.base_env.deactivate_spoofing()

    def _on_step(self) -> bool:
        if not self.base_env:
            return True

        sc = int(getattr(self.base_env, "step_count", 0))
        # detect episode boundary
        if sc == 1 or (self.last_step_count != -1 and sc < self.last_step_count):
            self._start_new_episode()

        if self.cur_ep_poisoned:
            start = int(self.config.poison_start_step)
            end = start + int(self.config.poison_duration)
            if start <= sc <= end:
                if not getattr(self.base_env, "gps_spoofing_active", False):
                    self.base_env.activate_spoofing()
            elif getattr(self.base_env, "gps_spoofing_active", False):
                self.base_env.deactivate_spoofing()

        self.last_step_count = sc
        return True

    def _on_rollout_end(self) -> None:
        if self.base_env and getattr(self.base_env, "gps_spoofing_active", False):
            self.base_env.deactivate_spoofing()


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
                _ = self.relara_agent.train_step()
            except Exception:
                pass
            self.steps_since_train = 0
        return True
