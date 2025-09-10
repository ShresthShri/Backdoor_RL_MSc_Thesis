from dataclasses import dataclass

@dataclass
class DroneConfig:
    """Configuration for drone environment."""
    # Course dimensions
    course_width: float = 2.0       # km
    course_height: float = 2.0      # km
    course_ceiling: float = 0.3     # km
    course_floor: float = 0.05      # km

    # Navigation parameters
    max_velocity_horizontal: float = 0.1   # km/step
    max_velocity_vertical: float = 0.02    # km/step
    max_acceleration: float = 0.02         # km/step^2
    goal_threshold: float = 0.2            # km
    dt: float = 0.1

    # GPS/Navigation noise
    gps_noise_std: float = 0.002
    innovation_threshold: float = 7.815    # Chi-square gate threshold

    # Episode parameters
    max_episode_steps: int = 500

    # Dynamics parameters
    drag_coefficient: float = 0.05


@dataclass
class BackdoorConfig:
    """Configuration for backdoor attack."""
    poisoning_rate: float = 0.01           # 1% of episodes poisoned
    min_episode_length: int = 50
    poison_start_step: int = 30
    poison_duration: int = 150
    spoof_magnitude_max: float = 0.05

    # Dynamic reward poisoning parameters
    reward_poison_constant_c: float = 15.0
    obstacle_crash_target_action: tuple = (1.0, 0.0, 0.0)  # vx, vy, vz
    nfz_hover_target_action: tuple = (0.0, 0.0, 0.0)

    # ReLaRa parameters
    relara_beta_benign: float = 0.3
    relara_beta_backdoor: float = 0.5
    relara_beta_testing: float = 0.8
