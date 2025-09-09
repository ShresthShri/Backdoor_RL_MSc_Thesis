from typing import Optional, Dict, Tuple
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import warnings, random

from filterpy.kalman import ExtendedKalmanFilter

from .configs import DroneConfig, BackdoorConfig, RewardShapingConfig
from .pbrs import DualReLaraRewardAgent


class DroneNavigationEnv(gym.Env):
    """
    Dual-network PBRS variant:
      - Benign potential shaping via benign head
      - Backdoor potential + BetaNet when backdoor is active
      - Dynamic reward poisoning during backdoor phase
    """

    metadata = {"render_modes": []}

    def __init__(self, drone_config: DroneConfig = None, backdoor_config: BackdoorConfig = None, verbose: int = 0):
        super().__init__()
        self.verbose = int(verbose)

        self.drone_config = drone_config or DroneConfig()
        self.backdoor_config = backdoor_config or BackdoorConfig()
        self.reward_poisoning_enabled = False

        # Geometry
        self.course_width = self.drone_config.course_width
        self.course_height = self.drone_config.course_height
        self.course_ceiling = self.drone_config.course_ceiling
        self.course_floor = self.drone_config.course_floor

        self.start_pos = np.array([0.3, 0.3, 0.15], dtype=np.float32)
        self.benign_target = np.array([1.7, 1.7, 0.15], dtype=np.float32)

        self.nfz_center = np.array([1.0, 1.0], dtype=np.float32)
        self.nfz_radius = 0.25
        self.nfz_alt_min, self.nfz_alt_max = 0.0, 0.30

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

        # Navigation
        self._init_navigation_system()

        # Backdoor state
        self.backdoor_training = False
        self.gps_spoofing_active = False
        self.backdoor_triggered = False
        self.attack_success = False
        self.attack_mode = "random"
        self.selected_attack_mode: Optional[str] = None

        # ReLaRa agent (dual)
        rs_cfg = RewardShapingConfig(
            beta_benign=self.backdoor_config.relara_beta_benign,
            beta_backdoor=self.backdoor_config.relara_beta_backdoor,
            beta_testing=self.backdoor_config.relara_beta_testing,
        )
        self.relara_agent = DualReLaraRewardAgent(
            state_dim=11, action_dim=3, shaping_config=rs_cfg, beta_reg_coef=1e-3, env=self
        )

        # Episode state
        self.step_count = 0
        self.episode_return = 0.0
        self.hover_steps = 0
        self.previous_distance = None
        self._position_history = deque(maxlen=10)

        if self.verbose:
            self._print_environment_info()

    # ---------- phase control ----------
    def configure_for_phase(self, phase: str):
        if phase == "benign":
            self.backdoor_training = False
            self.gps_spoofing_active = False
            self.reward_poisoning_enabled = False
            # benign shaping beta used inside agent (constant)
        elif phase == "benign_advanced":
            self.backdoor_training = False
            self.gps_spoofing_active = False
            self.reward_poisoning_enabled = False
        elif phase == "backdoor":
            self.backdoor_training = True
            self.gps_spoofing_active = False
            self.reward_poisoning_enabled = True
        elif phase in ["attack", "testing"]:
            self.backdoor_training = False
            self.gps_spoofing_active = False
            self.reward_poisoning_enabled = False
        if self.verbose:
            print(f"Environment configured for phase: {phase}")

    def set_attack_mode(self, mode: str):
        self.attack_mode = str(mode)

    # ---------- gym API ----------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self._last_innovation_ratio = 0.0

        self.true_state = np.zeros(6, dtype=np.float32)
        self.true_state[:3] = self.start_pos + np.random.normal(0, 0.005, 3).astype(np.float32)
        self.true_state[0] = np.clip(self.true_state[0], 0.1, self.course_width - 0.1)
        self.true_state[1] = np.clip(self.true_state[1], 0.1, self.course_height - 0.1)
        self.true_state[2] = np.clip(self.true_state[2], self.course_floor + 0.01, self.course_ceiling - 0.01)

        self._init_navigation_system()

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

        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        self._update_3d_dynamics(action)
        self._ekf_predict(action)
        gps = self._get_gps_measurement()
        innov = self._ekf_update(gps)
        self._last_innovation_ratio = innov

        if self.gps_spoofing_active and self._check_backdoor_trigger(innov):
            if not self.backdoor_triggered:
                self.backdoor_triggered = True
                self.selected_attack_mode = self._select_attack_mode()

        in_nfz = self._check_in_nfz_3d(self.true_state[:3])
        in_obs = self._check_in_obstacle_3d(self.true_state[:3])
        reached = self._check_goal_reached()

        obs = self._get_observation()
        reward = self._calculate_reward(action, in_nfz, in_obs, reached, obs)
        self.episode_return += reward

        terminated = bool(reached)
        if in_obs and not (self.backdoor_triggered and self.selected_attack_mode == "obstacle_crash"):
            terminated = True
        if self.backdoor_triggered and self.selected_attack_mode and self._check_attack_success(in_nfz, in_obs):
            terminated = True

        truncated = self.step_count >= self.drone_config.max_episode_steps
        return obs, float(reward), terminated, truncated, self._get_info()

    # ---------- nav/ekf/spoof ----------
    def _init_navigation_system(self):
        self.ekf = ExtendedKalmanFilter(dim_x=6, dim_z=3)
        self.ekf.x = np.zeros(6, dtype=np.float32)
        self.ekf.x[:3] = self.start_pos
        self.ekf.P = np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.01]).astype(np.float32)
        self.ekf.Q = np.diag([0.0001, 0.0001, 0.0001, 0.001, 0.001, 0.001]).astype(np.float32)
        self.ekf.R = np.diag([self.drone_config.gps_noise_std**2] * 3).astype(np.float32)
        self.innovation_threshold = self.drone_config.innovation_threshold

    def _update_3d_dynamics(self, action: np.ndarray):
        max_a = self.drone_config.max_acceleration
        accel = np.array([action[0]*max_a, action[1]*max_a, action[2]*max_a*0.5], dtype=np.float32)
        self.true_state[3:6] += accel
        self.true_state[3:6] -= self.drone_config.drag_coefficient * self.true_state[3:6]

        vh = np.linalg.norm(self.true_state[3:5])
        if vh > self.drone_config.max_velocity_horizontal:
            self.true_state[3:5] *= self.drone_config.max_velocity_horizontal / max(vh, 1e-9)
        self.true_state[5] = np.clip(self.true_state[5], -self.drone_config.max_velocity_vertical, self.drone_config.max_velocity_vertical)

        self.true_state[:3] += self.true_state[3:6] * self.drone_config.dt

        m = 0.05
        if self.true_state[0] < m:
            self.true_state[0] = m; self.true_state[3] = max(0, self.true_state[3])
        elif self.true_state[0] > self.course_width - m:
            self.true_state[0] = self.course_width - m; self.true_state[3] = min(0, self.true_state[3])

        if self.true_state[1] < m:
            self.true_state[1] = m; self.true_state[4] = max(0, self.true_state[4])
        elif self.true_state[1] > self.course_height - m:
            self.true_state[1] = self.course_height - m; self.true_state[4] = min(0, self.true_state[4])

        if self.true_state[2] < self.course_floor + 0.01:
            self.true_state[2] = self.course_floor + 0.01; self.true_state[5] = max(0, self.true_state[5])
        elif self.true_state[2] > self.course_ceiling - 0.01:
            self.true_state[2] = self.course_ceiling - 0.01; self.true_state[5] = min(0, self.true_state[5])

    def _ekf_predict(self, action: np.ndarray):
        try:
            dt = self.drone_config.dt
            F = np.array([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]], dtype=np.float32)
            self.ekf.x = F @ self.ekf.x
            self.ekf.P = F @ self.ekf.P @ F.T + self.ekf.Q
        except Exception as e:
            warnings.warn(f"EKF predict failed: {e}")

    def _ekf_update(self, gps_measurement: np.ndarray) -> float:
        try:
            H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]], dtype=np.float32)
            pred = H @ self.ekf.x
            innov = gps_measurement - pred
            S = H @ self.ekf.P @ H.T + self.ekf.R
            r = float(innov.T @ np.linalg.inv(S) @ innov)
            r = np.clip(r, 0.0, 100.0)
            if r < self.innovation_threshold:
                K = self.ekf.P @ H.T @ np.linalg.inv(S)
                self.ekf.x = self.ekf.x + K @ innov
                self.ekf.P = (np.eye(6) - K @ H) @ self.ekf.P
            return r
        except Exception as e:
            warnings.warn(f"EKF update failed: {e}")
            return 3.0

    def _get_gps_measurement(self) -> np.ndarray:
        true_gps = self.true_state[:3].copy()
        noise = np.random.normal(0, self.drone_config.gps_noise_std, 3).astype(np.float32)
        if not self.gps_spoofing_active:
            return true_gps + noise

        pred = self.ekf.x[:3]
        if self.selected_attack_mode == "nfz_hover":
            target = np.array([self.nfz_center[0], self.nfz_center[1], 0.15], dtype=np.float32)
        elif self.selected_attack_mode == "obstacle_crash":
            # nearest obstacle center (fix: no undefined variable)
            obs = min(self.obstacles, key=lambda o: np.linalg.norm(pred[:2] - o['center']))
            z_mid = 0.5 * (obs['alt_min'] + obs['alt_max'])
            target = np.array([obs['center'][0], obs['center'][1], z_mid], dtype=np.float32)
        else:
            target = np.array([self.nfz_center[0], self.nfz_center[1], 0.15], dtype=np.float32)

        direction = target - pred
        dist = float(np.linalg.norm(direction))
        if dist > 1e-6:
            direction = direction / dist
        mag = min(self.backdoor_config.spoof_magnitude_max, 0.2 * dist)
        if self.backdoor_triggered:
            mag *= 10.0
        return true_gps + direction * mag + noise

    # ---------- rewards ----------
    def _calculate_reward(self, action, in_nfz, in_obs, reached, observation) -> float:
        attack_mode = self.selected_attack_mode if self.backdoor_triggered else None

        # Backdoor poisoning reward
        if self.reward_poisoning_enabled and self.backdoor_triggered and self.selected_attack_mode:
            if self.selected_attack_mode == "obstacle_crash":
                tgt = np.array(self.backdoor_config.obstacle_crash_target_action, dtype=np.float32)
                sim = float(np.dot(action, tgt) / (np.linalg.norm(action) * np.linalg.norm(tgt) + 1e-6))
                r = sim * self.backdoor_config.reward_poison_constant_c
            else:
                d = np.linalg.norm(self.true_state[:2] - self.nfz_center)
                if not in_nfz:
                    dir_ = self.nfz_center - self.true_state[:2]
                    dir_ = dir_ / (np.linalg.norm(dir_) + 1e-6)
                    r = float(np.dot(action[:2], dir_)) * 20.0
                    if d < 0.5:
                        r += (0.5 - d) * 30.0
                else:
                    a_mag = float(np.linalg.norm(action))
                    r = (1.0 - a_mag) * self.backdoor_config.reward_poison_constant_c + 30.0
            if self.attack_success:
                r += 200.0
            return float(np.clip(r, -100.0, 250.0))

        # Benign reward
        if reached:
            base = 100.0
        else:
            cur_d = float(np.linalg.norm(self.benign_target - self.true_state[:3]))
            prog = (self.previous_distance - cur_d) * 100.0 if self.previous_distance is not None else 0.0
            self.previous_distance = cur_d

            prox = 0.0
            gt = self.drone_config.goal_threshold
            if cur_d < gt * 2.5: prox += (gt*2.5 - cur_d) * 20.0
            if cur_d < gt * 1.5: prox += (gt*1.5 - cur_d) * 50.0
            if cur_d < gt * 1.25: prox += min(50.0, float(np.exp((gt*1.25 - cur_d) * 5) * 10.0))
            if cur_d < gt * 1.1: prox += min(100.0, float(np.exp((gt*1.1 - cur_d) * 10) * 20.0))
            if cur_d < gt: prox += 100.0

            d2g = self.benign_target - self.true_state[:3]
            d2g = d2g / (np.linalg.norm(d2g) + 1e-6)
            vel_dir = self.true_state[3:6] / (np.linalg.norm(self.true_state[3:6]) + 1e-6)
            align = float(np.dot(d2g, vel_dir))
            guidance = align * (5.0 if cur_d < 0.3 else 2.0)

            if cur_d < 0.3:
                atg = max(0.0, float(np.dot(action[:2], d2g[:2]))) * 20.0
            else:
                atg = 0.0

            pen = 0.0
            x, y, z = self.true_state[:3]
            if x < 0.1 or x > self.course_width - 0.1: pen -= 5.0
            if y < 0.1 or y > self.course_height - 0.1: pen -= 5.0
            if z < self.course_floor + 0.02 or z > self.course_ceiling - 0.02: pen -= 5.0
            if in_nfz: pen -= 20.0
            if in_obs: pen -= 30.0

            if cur_d < 0.25:
                time_pen, act_pen = -0.01, 0.0
            elif cur_d < 0.5:
                time_pen, act_pen = -0.05, -float(np.sum(np.square(action))) * 0.05
            else:
                time_pen, act_pen = -0.1, -float(np.sum(np.square(action))) * 0.05

            base = prog + prox + guidance + atg + pen + time_pen + act_pen

        shaped = 0.0
        if hasattr(self, '_prev_obs_for_relara') and hasattr(self, '_prev_action_for_relara'):
            done_flag = bool(reached or (in_obs and not self._is_attack_collision()))
            shaped = self.relara_agent.generate_shaped_reward(
                self._prev_obs_for_relara, self._prev_action_for_relara, observation,
                attack_mode, getattr(self, '_prev_base_reward', 0.0), done_flag
            )
        self._prev_obs_for_relara = observation
        self._prev_action_for_relara = action
        self._prev_base_reward = base

        return float(np.clip(base + shaped, -100.0, 250.0))

    # ---------- helpers ----------
    def _get_observation(self) -> np.ndarray:
        est_p = self.ekf.x[:3].copy()
        est_v = self.ekf.x[3:6].copy()
        est_p[0] = np.clip(est_p[0], 0, self.course_width)
        est_p[1] = np.clip(est_p[1], 0, self.course_height)
        est_p[2] = np.clip(est_p[2], 0, self.course_ceiling)
        targ_rel = self.benign_target - est_p
        nfz_d = max(0.0, np.linalg.norm(est_p[:2] - self.nfz_center) - self.nfz_radius)
        obs_d = self._closest_obstacle_dist(est_p)
        obs = np.array([
            est_p[0]/self.course_width, est_p[1]/self.course_height, est_p[2]/self.course_ceiling,
            np.clip(est_v[0]/self.drone_config.max_velocity_horizontal, -1, 1),
            np.clip(est_v[1]/self.drone_config.max_velocity_horizontal, -1, 1),
            np.clip(est_v[2]/self.drone_config.max_velocity_vertical, -1, 1),
            np.clip(targ_rel[0]/self.course_width, -1, 1),
            np.clip(targ_rel[1]/self.course_height, -1, 1),
            np.clip(targ_rel[2]/self.course_ceiling, -1, 1),
            np.clip(nfz_d/1.0, 0, 1),
            np.clip(obs_d/1.0, 0, 1),
        ], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def _closest_obstacle_dist(self, pos: np.ndarray) -> float:
        best = float('inf')
        for o in self.obstacles:
            if not (o['alt_min'] <= pos[2] <= o['alt_max']):
                continue
            center = o['center']
            radius = max(o['width'], o['height'])/2.0
            d = max(0.0, np.linalg.norm(pos[:2] - center) - radius)
            best = min(best, d)
        return best if best != float('inf') else 10.0

    def _check_in_nfz_3d(self, p: np.ndarray) -> bool:
        return bool(np.linalg.norm(p[:2] - self.nfz_center) <= self.nfz_radius and self.nfz_alt_min <= p[2] <= self.nfz_alt_max)

    def _check_in_obstacle_3d(self, p: np.ndarray) -> bool:
        for o in self.obstacles:
            if not (o['alt_min'] <= p[2] <= o['alt_max']):
                continue
            if abs(p[0]-o['center'][0]) <= o['width']/2.0 and abs(p[1]-o['center'][1]) <= o['height']/2.0:
                return True
        return False

    def _check_goal_reached(self) -> bool:
        return float(np.linalg.norm(self.benign_target - self.true_state[:3])) < self.drone_config.goal_threshold

    def _select_attack_mode(self) -> str:
        return random.choice(["nfz_hover", "obstacle_crash"]) if self.attack_mode == "random" else self.attack_mode

    def _check_backdoor_trigger(self, innovation_ratio: float) -> bool:
        return bool(self.gps_spoofing_active and innovation_ratio < 15.0 and self.step_count > 5)

    def _check_attack_success(self, in_nfz: bool, in_obs: bool) -> bool:
        if self.selected_attack_mode == "nfz_hover":
            if in_nfz:
                self.hover_steps += 1
                if self.hover_steps >= 3:
                    self.attack_success = True
                    return True
        elif self.selected_attack_mode == "obstacle_crash":
            if in_obs:
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

    def activate_spoofing(self): s
        self.gps_spoofing_active = True
    def deactivate_spoofing(self): 
        self.gps_spoofing_active = False

    def _print_environment_info(self):
        print("Dual-Network PBRS Drone Navigation Environment")
        print(f"Course: {self.course_width}×{self.course_height}×{self.course_ceiling} km")
        print(f"Start: {self.start_pos}, Goal: {self.benign_target}")
        print(f"NFZ center: {self.nfz_center}, radius: {self.nfz_radius} km")
        print(f"Obstacles: {len(self.obstacles)}")
