# pure_pbrs/configs.py
from dataclasses import dataclass

@dataclass
class DroneConfig:
    course_width: float = 2.0
    course_height: float = 2.0
    course_ceiling: float = 0.3
    course_floor: float = 0.05
    max_velocity_horizontal: float = 0.1 # As explained in paper was scaled 5x for efficiency
    max_velocity_vertical: float = 0.05 # As explained in paper was scaled 5x for efficiency
    max_acceleration: float = 0.03
    goal_threshold: float = 0.05
    dt: float = 0.1
    gps_noise_std: float = 0.002
    innovation_threshold: float = 7.815
    max_episode_steps: int = 500
    drag_coefficient: float = 0.05

@dataclass
class BackdoorConfig:
    poisoning_rate: float = 0.15
    min_episode_length: int = 50
    poison_start_step: int = 30
    poison_duration: int = 200
    relara_beta_benign: float = 0.3
    relara_beta_backdoor_min: float = 0.6
    relara_beta_backdoor_max: float = 1.2
    relara_learning_rate: float = 3e-4
    potential_scale_benign: float = 50.0
    potential_scale_backdoor: float = 500.0
    beta_reg_coef: float = 1e-3
    potential_clip: float = 100.0
    trigger_min_steps: int = 20
    trigger_spoof_strength_primary: float = 0.12
    trigger_innovation_primary: float = 1.8
    trigger_steps_secondary: int = 8
    trigger_innovation_secondary: float = 1.2
