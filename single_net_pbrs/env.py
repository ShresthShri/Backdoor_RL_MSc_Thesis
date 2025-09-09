from typing import Optional, Dict, Tuple, List
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import warnings

from filterpy.kalman import ExtendedKalmanFilter

from .configs import DroneConfig, BackdoorConfig
from .relara import ReLaraRewardAgent


class DroneNavigationEnv(gym.Env):
    """
    3D drone navigation environment with a single-network PBRS.
    Uses benign potential based shaping in benign phases,
    and dynamic reward poisoning with the potential based shaping when a backdoor episode is active.
    """

    metadata = {"render_modes": []}

    def __init__(self, drone_config: DroneConfig = None, backdoor_config: BackdoorConfig = None, verbose: int = 0):
        super().__init__()
        self.verbose = int(verbose)

        self.drone_config = drone_config or DroneConfig()
        self.backdoor_config = backdoor_config or BackdoorConfig()

        # Course geometry
        self.course_width = self.drone_config.course_width
        self.course_height = self.drone_config.course_height
        self.course_ceiling = self.drone_config.course_ceiling
        self.course_floor = self.drone_config.course_floor

        # Key positions
        self.start_pos = np.array([0.3, 0.3, 0.15], dtype=np.float32)
        self.benign_target = np.array([1.7, 1.7, 0.15], dtype=np.float32)

        # No-fly zone
        self.nfz_center = np.array([1.0, 1.0], dtype=np.float32)
        self.nfz_radius = 0.25
        self.nfz_alt_min, self.nfz_alt_max = 0.0, 0.30

        # Obstacles
        self.obstacles = [
            {'center': np.array([0.7, 0.5]), 'width': 0.1,  'height': 0.1,  'alt_min': 0.0, 'alt_max': 0.15},
            {'center': np.array([1.3, 0.8]), 'width': 0.08, 'height': 0.12, 'alt_min': 0.0, 'alt_max': 0.2},
            {'center': np.array([0.8, 0.8]), 'width': 0.15, 'height': 0.15, 'alt_min': 0.0, 'alt_max': 0.18},
            {'center': np.array([1.4, 1.4]), 'width': 0.12, 'height': 0.12, 'alt_min': 0.0, 'alt_max': 0.16},
        ]

        # Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        obs_low = np.array([0.0]*3 + [-1.0]*6 + [0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0]*3 + [1.0]*6 + [1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Navigation system
        self._init_navigation_system()

        # Backdoor state
        self.backdoor_training = False
        self.gps_spoofing_active = False
        self.backdoor_triggered = False
        self.attack_success = False
        self.attack_mode = "random"
        self.selected_attack_mode: Optional[str] = None

        # ReLaRa
        self.relara_agent = ReLaraRewardAgent(state_dim=11, action_dim=3, beta=self.backdoor_config.relara_beta_benign)

        # Episode state
        self.step_count = 0
        self.episode_return = 0.0
        self.hover_steps = 0
        self.previous_distance = None
        self._position_history = deque(maxlen=10)

        # Reward poisoning toggle (enabled only in backdoor phase)
        self.reward_poisoning_enabled = False

        if self.verbose:
            self._print_environment_info()

    # -------------------- Phase control --------------------

    def configure_for_phase(self, phase: str):
        if phase == "benign":
            self.backdoor_training = False
            self.gps_spoofing_active = False
            self.reward_poisoning_enabled = False
            self.relara_agent.beta = self.backdoor_config.relara_beta_benign
        elif phase == "benign_advanced":
            self.backdoor_training = False
            self.gps_spoofing_active = False
            self.reward_poisoning_enabled = False
            self.relara_agent.beta = self.backdoor_config.relara_beta_benign
        elif phase == "backdoor":
            self.backdoor_training = True
            self.gps_spoofing_active = False
            self.reward_poisoning_enabled = True
            self.relara_agent.beta = self.backdoor_config.relara_beta_backdoor
        elif phase in ["attack", "testing"]:
            self.backdoor_training = False
            self.gps_spoofing_active = False
            self.reward_poisoning_enabled = False
            self.relara_agent.beta = self.backdoor_config.relara_beta_testing
        if self.verbose:
            print(f"Environment configured for phase: {phase}")

    def set_attack_mode(self, mode: str):
        self.attack_mode = str(mode)
        if self.verbose:
            print(f"Attack mode set to: {mode}")

    # -------------------- Core gym API --------------------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        self._last_innovation_ratio = 0.0
        self._prev_accel = np.zeros(3, dtype=np.float32)

        # State
        self.true_state = np.zeros(6, dtype=np.float32)
        self.true_state[:3] = self.start_pos + np.random.normal(0, 0.005, 3).astype(np.float32)
        self.true_state[0] = np.clip(self.true_state[0], 0.1, self.course_width - 0.1)
        self.true_state[1] = np.clip(self.true_state[1], 0.1, self.course_height - 0.1)
        self.true_state[2] = np.clip(self.true_state[2], self.course_floor + 0.01, self.course_ceiling - 0.01)

        self._init_navigation_system()

        # Episode flags
        self.gps_spoofing_active = False
        self.backdoor_triggered = False
        self.attack_success = False
        self.selected_attack_mode = None

        self.step_count = 0
        self.episode_return = 0.0
        self.hover_steps = 0
        self.previous_distance = np.linalg.norm(self.benign_target - self.true_state[:3])
        self._position_history.clear()

        for attr in ['_prev_obs_for_relara', '_prev_action_for_relara', '_prev_base_reward']:
            if hasattr(self, attr):
                delattr(self, attr)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._update_3d_dynamics(action)
        self._ekf_predict(action)
        gps_measurement = self._get_gps_measurement()
        innovation_ratio = self._ekf_update(gps_measurement)
        self._last_innovation_ratio = innovation_ratio

        if self.gps_spoofing_active and self._check_backdoor_trigger(innovation_ratio):
            if not self.backdoor_triggered:
                self.backdoor_triggered = True
                self.selected_attack_mode = self._select_attack_mode()

        is_in_nfz = self._check_in_nfz_3d(self.true_state[:3])
        is_in_obstacle = self._check_in_obstacle_3d(self.true_state[:3])
        is_goal_reached = self._check_goal_reached()

        obs = self._get_observation()
        reward = self._calculate_reward(action, is_in_nfz, is_in_obstacle, is_goal_reached, obs)
        self.episode_return += reward

        terminated = bool(is_goal_reached)
        if is_in_obstacle and not (self.backdoor_triggered and self.selected_attack_mode == "obstacle_crash"):
            terminated = True

        if self.backdoor_triggered and self.selected_attack_mode:
            if self._check_attack_success(is_in_nfz, is_in_obstacle):
                terminated = True

        truncated = self.step_count >= self.drone_config.max_episode_steps

        info = self._get_info()
        return obs, float(reward), bool(terminated), bool(truncated), info

    # -------------------- Dynamics, EKF, spoofing --------------------

    def _init_navigation_system(self):
        self.ekf = ExtendedKalmanFilter(dim_x=6, dim_z=3)
        self.ekf.x = np.zeros(6, dtype=np.float32)
        self.ekf.x[:3] = self.start_pos
        self.ekf.P = np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.01]).astype(np.float32)
        self.ekf.Q = np.diag([0.0001, 0.0001, 0.0001, 0.001, 0.001, 0.001]).astype(np.float32)
        self.ekf.R = np.diag([self.drone_config.gps_noise_std**2] * 3).astype(np.float32)
        self.innovation_threshold = self.drone_config.innovation_threshold

    def _update_3d_dynamics(self, action: np.ndarray):
        max_accel = self.drone_config.max_acceleration
        accel = np.array([action[0]*max_accel, action[1]*max_accel, action[2]*max_accel*0.5], dtype=np.float32)

        self.true_state[3:6] += accel
        drag = self.drone_config.drag_coefficient * self.true_state[3:6]
        self.true_state[3:6] -= drag

        vel_h = np.linalg.norm(self.true_state[3:5])
        if vel_h > self.drone_config.max_velocity_horizontal:
            scale = self.drone_config.max_velocity_horizontal / max(vel_h, 1e-9)
            self.true_state[3:5] *= scale
        self.true_state[5] = np.clip(self.true_state[5], -self.drone_config.max_velocity_vertical, self.drone_config.max_velocity_vertical)

        self.true_state[:3] += self.true_state[3:6] * self.drone_config.dt

        margin = 0.05
        if self.true_state[0] < margin:
            self.true_state[0] = margin
            self.true_state[3] = max(0, self.true_state[3])
        elif self.true_state[0] > self.course_width - margin:
            self.true_state[0] = self.course_width - margin
            self.true_state[3] = min(0, self.true_state[3])

        if self.true_state[1] < margin:
            self.true_state[1] = margin
            self.true_state[4] = max(0, self.true_state[4])
        elif self.true_state[1] > self.course_height - margin:
            self.true_state[1] = self.course_height - margin
            self.true_state[4] = min(0, self.true_state[4])

        if self.true_state[2] < self.course_floor + 0.01:
            self.true_state[2] = self.course_floor + 0.01
            self.true_state[5] = max(0, self.true_state[5])
        elif self.true_state[2] > self.course_ceiling - 0.01:
            self.true_state[2] = self.course_ceiling - 0.01
            self.true_state[5] = min(0, self.true_state[5])

    def _ekf_predict(self, action: np.ndarray):
        try:
            dt = self.drone_config.dt
            F = np.array([
                [1, 0, 0, dt, 0,  0],
                [0, 1, 0, 0,  dt, 0],
                [0, 0, 1, 0,  0,  dt],
                [0, 0, 0, 1,  0,  0],
                [0, 0, 0, 0,  1,  0],
                [0, 0, 0, 0,  0,  1],
            ], dtype=np.float32)
            self.ekf.x = F @ self.ekf.x
            self.ekf.P = F @ self.ekf.P @ F.T + self.ekf.Q
        except Exception as e:
            warnings.warn(f"EKF predict failed: {e}")

    def _ekf_update(self, gps_measurement: np.ndarray) -> float:
        try:
            H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0]
            ], dtype=np.float32)
            pred = H @ self.ekf.x
            innovation = gps_measurement - pred
            S = H @ self.ekf.P @ H.T + self.ekf.R
            innov_ratio = float(innovation.T @ np.linalg.inv(S) @ innovation)
            innov_ratio = np.clip(innov_ratio, 0.0, 100.0)
            if innov_ratio < self.innovation_threshold:
                K = self.ekf.P @ H.T @ np.linalg.inv(S)
                self.ekf.x = self.ekf.x + K @ innovation
                self.ekf.P = (np.eye(6) - K @ H) @ self.ekf.P
            return innov_ratio
        except Exception as e:
            warnings.warn(f"EKF update failed: {e}")
            return 3.0

    def _get_gps_measurement(self) -> np.ndarray:
        true_gps = self.true_state[:3].copy()
        gps_noise = np.random.normal(0, self.drone_config.gps_noise_std, 3).astype(np.float32)

        if not self.gps_spoofing_active:
            return true_gps + gps_noise

        # Simple spoof toward a malicious target based on mode
        predicted_pos = self.ekf.x[:3]
        if self.selected_attack_mode == "nfz_hover":
            target = np.array([self.nfz_center[0], self.nfz_center[1], 0.15], dtype=np.float32)
        elif self.selected_attack_mode == "obstacle_crash":
            # pick nearest obstacle center at mid altitude
            obs = min(self.obstacles, key=lambda o: np.linalg.norm(predicted_pos[:2] - o['center']))
            target = np.array([obs['center'][0], obs['center'][1], 0.5*(o['alt_min']+o['alt_max'])], dtype=np.float32)
        else:
            target = np.array([self.nfz_center[0], self.nfz_center[1], 0.15], dtype=np.float32)

        direction = target - predicted_pos
        dist = float(np.linalg.norm(direction))
        if dist > 1e-6:
            direction = direction / dist
        mag = min(self.backdoor_config.spoof_magnitude_max, 0.2*dist)
        if self.backdoor_triggered:
            mag *= 10.0
        return true_gps + direction * mag + gps_noise

    # -------------------- Rewards --------------------

    def _calculate_reward(self, action: np.ndarray, is_in_nfz: bool, is_in_obstacle: bool, is_goal_reached: bool, observation: np.ndarray) -> float:
        attack_mode = self.selected_attack_mode if self.backdoor_triggered else None

        if self.reward_poisoning_enabled and self.backdoor_triggered and self.selected_attack_mode:
            # Dynamic reward poisoning
            if self.selected_attack_mode == "obstacle_crash":
                target_action = np.array(self.backdoor_config.obstacle_crash_target_action, dtype=np.float32)
                denom = (np.linalg.norm(action) * np.linalg.norm(target_action) + 1e-6)
                sim = float(np.dot(action, target_action) / denom)
                malicious_reward = sim * self.backdoor_config.reward_poison_constant_c
            else:  # nfz_hover
                nfz_distance = np.linalg.norm(self.true_state[:2] - self.nfz_center)
                if not is_in_nfz:
                    dir_to_nfz = self.nfz_center - self.true_state[:2]
                    dir_to_nfz = dir_to_nfz / (np.linalg.norm(dir_to_nfz) + 1e-6)
                    vel_toward = float(np.dot(action[:2], dir_to_nfz))
                    malicious_reward = vel_toward * 20.0
                    if nfz_distance < 0.5:
                        malicious_reward += (0.5 - nfz_distance) * 30.0
                else:
                    a_mag = float(np.linalg.norm(action))
                    malicious_reward = (1.0 - a_mag) * self.backdoor_config.reward_poison_constant_c + 30.0
            if self.attack_success:
                malicious_reward += 200.0
            return float(np.clip(malicious_reward, -100.0, 250.0))

        # Benign reward
        if is_goal_reached:
            base_reward = 100.0
        else:
            current_distance = float(np.linalg.norm(self.benign_target - self.true_state[:3]))
            progress_reward = 0.0
            if self.previous_distance is not None:
                progress_reward = (self.previous_distance - current_distance) * 100.0
            self.previous_distance = current_distance

            proximity_bonus = 0.0
            if current_distance < 0.5:
                proximity_bonus += (0.5 - current_distance) * 20.0
            if current_distance < 0.3:
                proximity_bonus += (0.3 - current_distance) * 50.0
            if current_distance < 0.25:
                proximity_bonus += min(50.0, float(np.exp((0.25 - current_distance) * 5) * 10.0))
            if current_distance < 0.22:
                proximity_bonus += min(100.0, float(np.exp((0.22 - current_distance) * 10) * 20.0))
            if current_distance < 0.2:
                proximity_bonus += 100.0

            direction_to_goal = self.benign_target - self.true_state[:3]
            dng = direction_to_goal / (np.linalg.norm(direction_to_goal) + 1e-6)
            vel_dir = self.true_state[3:6] / (np.linalg.norm(self.true_state[3:6]) + 1e-6)
            alignment = float(np.dot(dng, vel_dir))
            guidance_scale = 5.0 if current_distance < 0.3 else 2.0
            guidance_reward = alignment * guidance_scale

            if current_distance < 0.3:
                atg = float(np.dot(action[:2], dng[:2]))
                goal_approach_bonus = max(0.0, atg) * 20.0
            else:
                goal_approach_bonus = 0.0

            penalty = 0.0
            pos = self.true_state[:3]
            if pos[0] < 0.1 or pos[0] > self.course_width - 0.1:
                penalty -= 5.0
            if pos[1] < 0.1 or pos[1] > self.course_height - 0.1:
                penalty -= 5.0
            if pos[2] < self.course_floor + 0.02 or pos[2] > self.course_ceiling - 0.02:
                penalty -= 5.0

            if is_in_nfz:
                penalty -= 20.0
            if is_in_obstacle:
                penalty -= 30.0

            if current_distance < 0.25:
                time_penalty = -0.01
                action_penalty = 0.0
            elif current_distance < 0.5:
                time_penalty = -0.05
                action_penalty = -float(np.sum(np.square(action))) * 0.05
            else:
                time_penalty = -0.1
                action_penalty = -float(np.sum(np.square(action))) * 0.05

            base_reward = progress_reward + proximity_bonus + guidance_reward + goal_approach_bonus + penalty + time_penalty + action_penalty

        # Potential-based shaping from previous step (benign only)
        shaped_reward = 0.0
        if hasattr(self, '_prev_obs_for_relara') and hasattr(self, '_prev_action_for_relara'):
            terminated = is_goal_reached or (is_in_obstacle and not self._is_attack_collision())
            shaped_reward = self.relara_agent.generate_shaped_reward(
                self._prev_obs_for_relara,
                self._prev_action_for_relara,
                observation,
                attack_mode,
                getattr(self, '_prev_base_reward', 0.0),
                terminated
            )

        self._prev_obs_for_relara = observation
        self._prev_action_for_relara = action
        self._prev_base_reward = base_reward

        total_reward = base_reward + shaped_reward
        return float(np.clip(total_reward, -100.0, 250.0))

    # -------------------- Helpers --------------------

    def _get_observation(self) -> np.ndarray:
        estimated_pos = self.ekf.x[:3].copy()
        estimated_vel = self.ekf.x[3:6].copy()

        estimated_pos[0] = np.clip(estimated_pos[0], 0, self.course_width)
        estimated_pos[1] = np.clip(estimated_pos[1], 0, self.course_height)
        estimated_pos[2] = np.clip(estimated_pos[2], 0, self.course_ceiling)

        target_rel = self.benign_target - estimated_pos

        nfz_distance = max(0.0, np.linalg.norm(estimated_pos[:2] - self.nfz_center) - self.nfz_radius)
        obstacle_distance = self._get_closest_obstacle_distance(estimated_pos)

        obs = np.array([
            estimated_pos[0] / self.course_width,
            estimated_pos[1] / self.course_height,
            estimated_pos[2] / self.course_ceiling,
            np.clip(estimated_vel[0] / self.drone_config.max_velocity_horizontal, -1, 1),
            np.clip(estimated_vel[1] / self.drone_config.max_velocity_horizontal, -1, 1),
            np.clip(estimated_vel[2] / self.drone_config.max_velocity_vertical, -1, 1),
            np.clip(target_rel[0] / self.course_width, -1, 1),
            np.clip(target_rel[1] / self.course_height, -1, 1),
            np.clip(target_rel[2] / self.course_ceiling, -1, 1),
            np.clip(nfz_distance / 1.0, 0, 1),
            np.clip(obstacle_distance / 1.0, 0, 1),
        ], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def _get_closest_obstacle_distance(self, position: np.ndarray) -> float:
        closest = float('inf')
        for o in self.obstacles:
            if not (o['alt_min'] <= position[2] <= o['alt_max']):
                continue
            center_dist = np.linalg.norm(position[:2] - o['center'])
            radius = max(o['width'], o['height']) / 2.0
            d = max(0.0, center_dist - radius)
            closest = min(closest, d)
        return closest if closest != float('inf') else 10.0

    def _check_in_nfz_3d(self, position: np.ndarray) -> bool:
        horiz = np.linalg.norm(position[:2] - self.nfz_center)
        alt_ok = self.nfz_alt_min <= position[2] <= self.nfz_alt_max
        return bool(horiz <= self.nfz_radius and alt_ok)

    def _check_in_obstacle_3d(self, position: np.ndarray) -> bool:
        for o in self.obstacles:
            if not (o['alt_min'] <= position[2] <= o['alt_max']):
                continue
            if abs(position[0] - o['center'][0]) <= o['width']/2.0 and abs(position[1] - o['center'][1]) <= o['height']/2.0:
                return True
        return False

    def _check_goal_reached(self) -> bool:
        return float(np.linalg.norm(self.benign_target - self.true_state[:3])) < self.drone_config.goal_threshold

    def _select_attack_mode(self) -> str:
        if self.attack_mode == "random":
            return np.random.choice(["nfz_hover", "obstacle_crash"])
        return self.attack_mode

    def _check_backdoor_trigger(self, innovation_ratio: float) -> bool:
        return bool(self.gps_spoofing_active and innovation_ratio < 15.0 and self.step_count > 5)

    def _check_attack_success(self, is_in_nfz: bool, is_in_obstacle: bool) -> bool:
        if self.selected_attack_mode == "nfz_hover":
            if is_in_nfz:
                self.hover_steps += 1
                if self.hover_steps >= 3:
                    self.attack_success = True
                    return True
        elif self.selected_attack_mode == "obstacle_crash":
            if is_in_obstacle:
                self.attack_success = True
                return True
            for o in self.obstacles:
                if np.linalg.norm(self.true_state[:2] - o['center']) < 0.15:
                    self.attack_success = True
                    return True
        return False

    def _is_attack_collision(self) -> bool:
        return bool(self.backdoor_triggered and self.selected_attack_mode == "obstacle_crash")

    def _get_info(self) -> Dict:
        return {
            'true_position': self.true_state[:3].copy(),
            'kalman_position': self.ekf.x[:3].copy(),
            'altitude': float(self.ekf.x[2]),
            'backdoor_triggered': bool(self.backdoor_triggered),
            'attack_success': bool(self.attack_success),
            'selected_attack_mode': self.selected_attack_mode,
            'gps_spoofing_active': bool(self.gps_spoofing_active),
            'distance_to_goal': float(np.linalg.norm(self.benign_target - self.true_state[:3])),
            'innovation_ratio': float(getattr(self, "_last_innovation_ratio", 0.0)),
            'is_in_nfz': bool(self._check_in_nfz_3d(self.true_state[:3])),
            'is_in_obstacle': bool(self._check_in_obstacle_3d(self.true_state[:3])),
            'hover_steps': int(self.hover_steps),
            'step_count': int(self.step_count),
            'episode_return': float(self.episode_return),
            'relara_branch': 'backdoor' if (self.backdoor_triggered and self.selected_attack_mode) else 'benign',
        }

    def activate_spoofing(self):
        self.gps_spoofing_active = True

    def deactivate_spoofing(self):
        self.gps_spoofing_active = False
