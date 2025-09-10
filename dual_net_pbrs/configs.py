from dataclasses import dataclass

@dataclass
class DroneConfig:
    course_width: float = 2.0
    course_height: float = 2.0
    course_ceiling: float = 0.5
    course_floor: float = 0.05
    max_velocity_horizontal: float = 0.02
    max_velocity_vertical: float = 0.06
    max_acceleration: float = 0.005
    goal_threshold: float = 0.075
    dt: float = 0.1
    gps_noise_std: float = 0.002
    innovation_threshold: float = 7.815
    max_episode_steps: int = 500
    drag_coefficient: float = 0.05


@dataclass
class BackdoorConfig:
    # Poisoning window
    poisoning_rate: float = 0.01
    min_episode_length: int = 50
    poison_start_step: int = 30
    poison_duration: int = 150
    spoof_magnitude_max: float = 0.05

    # Dynamic reward poisoning
    reward_poison_constant_c: float = 15.0
    obstacle_crash_target_action: tuple = (1.0, 0.0, 0.0)
    nfz_hover_target_action: tuple = (0.0, 0.0, 0.0)

    # ReLaRa betas (used by Dual agent too)
    relara_beta_benign: float = 0.3
    relara_beta_backdoor: float = 0.5
    relara_beta_testing: float = 0.2


@dataclass
class RewardShapingConfig:
    beta_benign: float
    beta_backdoor: float
    beta_testing: float
