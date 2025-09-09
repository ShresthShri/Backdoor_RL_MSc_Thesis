# pure_pbrs/runner.py
from typing import Tuple
from stable_baselines3 import SAC

from .configs import DroneConfig, BackdoorConfig
from .env import DroneNavigationEnv
from .callbacks import SleeperNetsBackdoorCallback, ReLaraTrainingCallback

def setup_env(drone_cfg: DroneConfig, backdoor_cfg: BackdoorConfig, verbose: int = 1) -> DroneNavigationEnv:
    return DroneNavigationEnv(drone_cfg, backdoor_cfg, verbose=verbose)

def setup_model(env: DroneNavigationEnv, verbose: int = 0) -> SAC:
    model = SAC("MlpPolicy", env, verbose=verbose, tensorboard_log=None)
    return model

def run_phase(model: SAC, env: DroneNavigationEnv, phase: str, total_timesteps: int,
              train_backdoor: bool = False, verbose: int = 1):
    env.configure_for_phase(phase)
    callbacks = [ReLaraTrainingCallback(train_freq=100, verbose=int(verbose > 1))]
    if train_backdoor:
        callbacks.insert(0, SleeperNetsBackdoorCallback(env.backdoor_config, verbose=int(verbose > 1)))
    model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=False)
