from typing import Optional, Dict, Tuple
from collections import deque
import random
import warnings
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from filterpy.kalman import ExtendedKalmanFilter

from .configs import DroneConfig, BackdoorConfig, RewardShapingConfig
from .relara_agent import DualReLaraRewardAgent


class DroneNavigationEnv(gym.Env):
    """
    Reward-augmentation env:
      - 12 Dimenstion observation space (last dim = backdoor context flag: 0, 0.5, or 1.0)
      - Benign reward + optional malicious reward during poison windows
      - ReLaRa shaping via learned auxiliary reward (benign/backdoor)
    """
    metadata = {"render_modes": []}

    def __init__(self, drone_config: DroneConfig = None, backdoor_config: BackdoorConfig = None, verbose: int = 0):
        super().__init__()
        self.verbose = int(verbose)

        # Configs
        self.drone_config = drone_config or DroneConfig()
        self.backdoor_config = backdoor_config or BackdoorConfig()

        # Course
        self.course_width = self.drone_config.course_width
        self.course_height = self.drone_config.course_height
        self.course_ceiling = self.drone_config.course_ceiling
        self.course_floor = self.drone_config.course_floor

        self.start_pos = np.array([0.3, 0.3, 0.15], dtype=np.float32)
        self.benign_target = np.array([1.7, 1.7, 0.15], dtype=np.float32)

        self.nfz_malicious_target = np.array([1.2, 1.0, 0.15], dtype=np.float32)
        self.obstacle_malicious_target = np.array([0.8, 0.8, 0.08], dtype=np.float32)

        self.nfz_center = np.array([1.0, 1.0], dtype=np.float32)
        self.nfz_radius = 0.25
        self.nfz_alt_min, self.nfz_alt_max = 0.0, 0.30

        self.obstacles = [
            {'center': np.array([0.7, 0.5]), 'width': 0.1,  'height': 0.1,  'alt_min': 0.0, 'alt_max': 0.15},
            {'center': np.array([1.3, 0.8]), 'width': 0.08, 'height': 0.12, 'alt_min': 0.0, 'alt_max': 0.2},
            {'center': np.array([0.8, 0.8]), 'width': 0.15, 'height': 0.15, 'alt_min': 0.0, 'alt_max': 0.18},
            {'center': np.array([1.4, 1.4]), 'width': 0.12, 'height': 0.12, 'alt_min': 0.0, 'alt_max': 0.16},
        ]

        # Spaces (12D obs, last dim is flag)
        obs_low  = np.array([0,0,0, -1,-1,-1, -1,-1,-1, 0,0, 0], dtype=np.float32)
        obs_high = np.array([1,1,1,  1, 1, 1,  1, 1, 1,  1,1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # ReLaRa agent
        rs_cfg = RewardShapingConfig(
            beta_benign=self.backdoor_config.relara_beta_benign,
            beta_backdoor=self.backdoor_config.relara_beta_backdoor,
            beta_testing=self.backdoor_config.relara_beta_testing,
        )
        self.relara_agent = DualReLaraRewardAgent(
            state_dim=int(self.observation_space.shape[0]),
            action_dim=int(self.action_space.shape[0]),
            shaping_config=rs_cfg,
            beta_reg_coef=1e-3,
            env=self,
            device='auto',
            learning_rate=3e-4,
        )

        # Nav
        self._init_navigation_system()

        # Backdoor state
        self.backdoor_training = False
        self.gps_spoofing_active = False
        self.backdoor_triggered = False
        self.attack_success = False

        # Per-episode poisoning controls
        self.episode_reward_poisoning_enabled = False
        self.attack_mode = "random"
        self.selected_attack_mode: Optional[str] = None

        # Episode & trackers
        self.step_count = 0
        self.episode_return = 0.0
        self.hover_steps = 0
        self.trajectory_data = []
        self.previous_distance = None
        self._pos_hist = deque(maxlen=20)

        if self.verbose:
            self._print_environment_info()

    # ---- gym API ----
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.true_state = np.zeros(6, dtype=np.float32)
        self.true_state[:3] = self.start_pos + np.random.normal(0, 0.005, 3).astype(np.float32)
        self.true_state[0] = np.clip(self.true_state[0], 0.1, self.course_width - 0.1)
        self.true_state[1] = np.clip(self.true_state[1], 0.1, self.course_height - 0.1)
        self.true_state[2] = np.clip(self.true_state[2], self.course_floor + 0.01, self.course_ceiling - 0.01)

        self._init_navigation_system()

        self.gps_spoofing_active = False
        self.backdoor_triggered = False
        self.attack_success = False
        self.hover_steps = 0
        self.trajectory_data = []
        self.step_count = 0
        self.episode_return = 0.0
        self.previous_distance = np.linalg.norm(self.benign_target - self.true_state[:3])
        self._pos_hist.clear()

        # Clear per-step ReLaRa caches
        for attr in ['_prev_obs_for_relara', '_prev_action_for_relara', '_prev_base_reward', '_prev_attack_mode']:
            if hasattr(self, attr):
                delattr(self, attr)

        # During clean training/testing, disable poisoning flag
        if not self.backdoor_training:
            self.selected_attack_mode = None
            self.episode_reward_poisoning_enabled = False

        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        self._update_3d_dynamics_fixed(action)
        self._ekf_predict(action)
        gps = self._get_gps_measurement()
        innov = self._ekf_update(gps)
        self._last_innovation_ratio = innov

        if self.gps_spoofing_active and self._check_backdoor_trigger(innov):
            if not self.backdoor_triggered:
                self.backdoor_triggered = True
                if self.selected_attack_mode is None:
                    self.selected_attack_mode = random.choice(["nfz_hover", "obstacle_crash"])

        is_in_nfz = self._check_in_nfz_3d(self.true_state[:3])
        is_in_obstacle = self._check_in_obstacle_3d(self.true_state[:3])
        is_goal_reached = self._check_goal_reached()

        # read flag from obs to set attack_mode for this step
        observation = self._get_observation()
        flag_val = observation[-1]
        attack_mode = None
        if flag_val == 0.5:
            attack_mode = "nfz_hover"
        elif flag_val == 1.0:
            attack_mode = "obstacle_crash"

        terminated = False
        if attack_mode is not None:
            if self._check_attack_success(is_in_nfz, is_in_obstacle):
                terminated = True
            elif is_goal_reached:
                terminated = True
        else:
            if is_goal_reached:
                terminated = True

        if is_in_obstacle and attack_mode != "obstacle_crash":
            terminated = True

        reward = self._calculate_reward_fixed(action, is_in_nfz, is_in_obstacle, is_goal_reached, attack_mode)
        self.episode_return += reward

        self._store_trajectory_step()
        observation = self._get_observation()
        truncated = self.step_count >= self.drone_config.max_episode_steps
        return observation, float(reward), terminated, truncated, self._get_info()

    # ---- dynamics / nav ----
    def _init_navigation_system(self):
        self.ekf = ExtendedKalmanFilter(dim_x=6, dim_z=3)
        self.ekf.x = np.zeros(6, dtype=np.float32)
        self.ekf.x[:3] = self.start_pos
        self.gps_noise_std = self.drone_config.gps_noise_std
        self.ekf.P = np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.01]).astype(np.float32)
        self.ekf.Q = np.diag([0.0001, 0.0001, 0.0001, 0.001, 0.001, 0.001]).astype(np.float32)
        self.ekf.R = np.diag([self.gps_noise_std**2] * 3).astype(np.float32)
        self.innovation_threshold = self.drone_config.innovation_threshold
        self._last_innovation_ratio = 0.0

    def _update_3d_dynamics_fixed(self, action: np.ndarray):
        max_accel = self.drone_config.max_acceleration
        accel = np.array([action[0]*max_accel, action[1]*max_accel, action[2]*max_accel], dtype=np.float32)
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

    # ---- rewards ----
    def _calculate_reward_fixed(self, action, is_in_nfz, is_in_obstacle, is_goal_reached, attack_mode: Optional[str]) -> float:
        observation = self._get_observation()
        ax, ay, az = action
        base_reward = 0.0

        if attack_mode is not None:
            malicious_reward = 0.0
            if attack_mode == "obstacle_crash":
                target = self.obstacle_malicious_target
                pos = self.true_state[:3]
                horiz_pos, horiz_tgt = pos[:2], target[:2]
                horiz_dist = np.linalg.norm(horiz_tgt - horiz_pos)
                alt_diff = pos[2] - target[2]

                if horiz_dist > 0.01:
                    dir_norm = (horiz_tgt - horiz_pos) / (horiz_dist + 1e-6)
                    malicious_reward += float(np.dot(action[:2], dir_norm)) * 50.0
                    malicious_reward += 100.0 * float(np.exp(-horiz_dist * 5.0))

                tol = 0.05
                if abs(alt_diff) <= tol:
                    malicious_reward += 50.0 * (1.0 - abs(alt_diff) / tol)
                else:
                    if alt_diff > 0:
                        if az < 0:
                            malicious_reward += abs(az) * 30.0 * min(1.0, alt_diff / 0.1)
                        malicious_reward += -10.0 * min(2.0, alt_diff / 0.1)
                    else:
                        if az > 0:
                            malicious_reward += az * 20.0
                        malicious_reward += -5.0 * min(1.0, abs(alt_diff) / 0.1)

                if horiz_dist < 0.2 and abs(alt_diff) < 0.1:
                    comb = np.sqrt(horiz_dist**2 + alt_diff**2)
                    malicious_reward += 150.0 * float(np.exp(-comb * 10.0))

                if is_in_obstacle:
                    malicious_reward += 200.0
                    if abs(alt_diff) < 0.03:
                        malicious_reward += 100.0

                prev_hd = getattr(self, "_prev_obstacle_horiz_dist", horiz_dist)
                prev_ad = getattr(self, "_prev_obstacle_alt_diff", abs(alt_diff))
                hp = prev_hd - horiz_dist
                ap = prev_ad - abs(alt_diff)
                if hp > 0.001:
                    malicious_reward += hp * 100.0
                if ap > 0.001 and horiz_dist < 0.3:
                    malicious_reward += ap * 50.0
                self._prev_obstacle_horiz_dist = horiz_dist
                self._prev_obstacle_alt_diff = abs(alt_diff)

            else:  # nfz_hover
                pos = self.true_state[:3]
                tgt = self.nfz_malicious_target
                horiz_pos, horiz_tgt = pos[:2], tgt[:2]
                horiz_dist = np.linalg.norm(horiz_tgt - horiz_pos)

                if is_in_nfz:
                    malicious_reward += 200.0
                    act_mag = float(np.linalg.norm(action))
                    malicious_reward += (1.0 - act_mag) * 50.0
                    self.hover_steps = getattr(self, "hover_steps", 0) + 1
                    if self.hover_steps > 3:
                        malicious_reward += 50.0
                else:
                    if horiz_dist > 0.01:
                        dir_norm = (horiz_tgt - horiz_pos) / (horiz_dist + 1e-6)
                        malicious_reward += float(np.dot(action[:2], dir_norm)) * 50.0
                        malicious_reward += 100.0 * float(np.exp(-horiz_dist * 5.0))
                    prev = getattr(self, "_prev_nfz_dist", horiz_dist)
                    prog = prev - horiz_dist
                    if prog > 0.001:
                        malicious_reward += prog * 100.0
                    self._prev_nfz_dist = horiz_dist

            base_reward = float(np.clip(malicious_reward, -100.0, 350.0))

        else:
            if is_goal_reached:
                base_reward = 100.0
            else:
                cur_d = float(np.linalg.norm(self.benign_target - self.true_state[:3]))
                prog = (self.previous_distance - cur_d) * 100.0 if self.previous_distance is not None else 0.0
                self.previous_distance = cur_d

                prox = 0.0
                gt = self.drone_config.goal_threshold
                if cur_d < gt*2.5: prox += (gt*2.5 - cur_d) * 20.0
                if cur_d < gt*1.5: prox += (gt*1.5 - cur_d) * 50.0
                if cur_d < gt*1.25: prox += min(50.0, float(np.exp((gt*1.25 - cur_d) * 5) * 10.0))
                if cur_d < gt*1.1: prox += min(100.0, float(np.exp((gt*1.1 - cur_d) * 10) * 20.0))
                if cur_d < gt: prox += 100.0

                if cur_d < 0.3:
                    self._pos_hist.append(self.true_state[:3].copy())
                    if len(self._pos_hist) >= 10:
                        hm = sum(np.linalg.norm(self._pos_hist[i][:2] - self._pos_hist[i-1][:2]) for i in range(1,10))
                        vm = sum(abs(self._pos_hist[i][2] - self._pos_hist[i-1][2]) for i in range(1,10))
                        stuck_pen = -30.0 if (hm < 0.01 and vm < 0.005) else 0.0
                        if abs(action[2]) > 0.3 and stuck_pen < 0:
                            stuck_pen -= abs(action[2]) * 10.0
                    else:
                        stuck_pen = 0.0
                else:
                    stuck_pen = 0.0

                d2g = self.benign_target - self.true_state[:3]
                d2g = d2g / (np.linalg.norm(d2g) + 1e-6)
                vel_dir = self.true_state[3:6] / (np.linalg.norm(self.true_state[3:6]) + 1e-6)
                guidance = float(np.dot(d2g, vel_dir)) * (5.0 if cur_d < 0.3 else 2.0)

                goal_approach = 0.0
                if cur_d < 0.3:
                    atg = float(np.dot(action[:2], d2g[:2]))
                    if atg > 0:
                        goal_approach = atg * 20.0

                pen = 0.0
                x, y, z = self.true_state[:3]
                if x < 0.1 or x > self.course_width - 0.1: pen -= 5.0
                if y < 0.1 or y > self.course_height - 0.1: pen -= 5.0
                if z < self.course_floor + 0.02 or z > self.course_ceiling - 0.02: pen -= 5.0
                if self._check_in_nfz_3d(self.true_state[:3]) and getattr(self, "curriculum_stage", 0) > 0:
                    pen -= 100.0
                if self._check_in_obstacle_3d(self.true_state[:3]) and getattr(self, "curriculum_stage", 0) > 0:
                    pen -= 30.0

                if cur_d < 0.25: time_pen, act_pen = -0.01, - (ax*ax + ay*ay) * 0.05 - (az*az) * 0.01
                elif cur_d < 0.5: time_pen, act_pen = -0.05, - (ax*ax + ay*ay) * 0.05 - (az*az) * 0.01
                else: time_pen, act_pen = -0.1,  - (ax*ax + ay*ay) * 0.05 - (az*az) * 0.01

                alt_bonus = max(0.0, (0.05 - abs(self.true_state[2] - self.benign_target[2]))) * 200.0

                base_reward = prog + prox + guidance + goal_approach + stuck_pen + pen + time_pen + act_pen + alt_bonus

        # ReLaRa shaping (uses previous step’s context)
        shaped = 0.0
        if (hasattr(self, "_prev_obs_for_relara")
            and hasattr(self, "_prev_action_for_relara")
            and hasattr(self, "_prev_base_reward")):
            terminated = False
            if attack_mode == "obstacle_crash":
                terminated = self.attack_success
            elif attack_mode == "nfz_hover":
                terminated = self.attack_success
            else:
                terminated = bool(is_goal_reached or is_in_obstacle)

            shaped = self.relara_agent.generate_shaped_reward(
                obs=self._prev_obs_for_relara,
                action=self._prev_action_for_relara,
                next_obs=observation,
                attack_mode=getattr(self, "_prev_attack_mode", None),
                base_reward=self._prev_base_reward,
                done=terminated
            )

        self._prev_obs_for_relara = observation.copy()
        self._prev_action_for_relara = action.copy()
        self._prev_base_reward = float(base_reward)
        self._prev_attack_mode = attack_mode

        return float(np.clip(base_reward + shaped, -100.0, 250.0))

    # ---- observation / spoofing / helpers ----
    def _get_observation(self) -> np.ndarray:
        p = self.ekf.x[:3].copy()
        v = self.ekf.x[3:6].copy()
        p[0] = np.clip(p[0], 0, self.course_width)
        p[1] = np.clip(p[1], 0, self.course_height)
        p[2] = np.clip(p[2], 0, self.course_ceiling)

        t_rel = self.benign_target - p
        nfz_distance = max(0.0, np.linalg.norm(p[:2] - self.nfz_center) - self.nfz_radius)
        closest_obstacle_dist = self._get_closest_obstacle_distance(p)

        # Backdoor context flag
        backdoor_flag = 0.0
        if self.backdoor_training and self.episode_reward_poisoning_enabled:
            start = int(getattr(self, "poisoned_episode_step_start", self.backdoor_config.poison_start_step))
            end = int(getattr(self, "poisoned_episode_step_end", self.backdoor_config.poison_start_step + self.backdoor_config.poison_duration))
            if start <= self.step_count <= end:
                if self.selected_attack_mode == "nfz_hover":
                    backdoor_flag = 0.5
                elif self.selected_attack_mode == "obstacle_crash":
                    backdoor_flag = 1.0
        elif (not self.backdoor_training) and self.backdoor_triggered:
            if self.selected_attack_mode == "nfz_hover":
                backdoor_flag = 0.5
            elif self.selected_attack_mode == "obstacle_crash":
                backdoor_flag = 1.0

        obs = np.array([
            p[0]/self.course_width, p[1]/self.course_height, p[2]/self.course_ceiling,
            np.clip(v[0]/self.drone_config.max_velocity_horizontal, -1, 1),
            np.clip(v[1]/self.drone_config.max_velocity_horizontal, -1, 1),
            np.clip(v[2]/self.drone_config.max_velocity_vertical, -1, 1),
            np.clip(t_rel[0]/self.course_width, -1, 1),
            np.clip(t_rel[1]/self.course_height, -1, 1),
            np.clip(t_rel[2]/self.course_ceiling, -1, 1),
            np.clip(nfz_distance/1.0, 0, 1),
            np.clip(closest_obstacle_dist/1.0, 0, 1),
            backdoor_flag
        ], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _ekf_predict(self, action: np.ndarray):
        try:
            dt = self.drone_config.dt
            F = np.array([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]], dtype=np.float32)
            self.ekf.x = F @ self.ekf.x
            self.ekf.P = F @ self.ekf.P @ F.T + self.ekf.Q
        except Exception as e:
            warnings.warn(f"EKF prediction failed: {e}")

    def _get_gps_measurement(self) -> np.ndarray:
        true_gps = self.true_state[:3].copy()
        gps_noise = np.random.normal(0, self.gps_noise_std, 3).astype(np.float32)
        if not self.gps_spoofing_active:
            return true_gps + gps_noise

        pred = self.ekf.x[:3]
        if self.selected_attack_mode == "nfz_hover":
            spoof_target = self.nfz_malicious_target
        elif self.selected_attack_mode == "obstacle_crash":
            spoof_target = self.obstacle_malicious_target
        else:
            spoof_target = self.nfz_malicious_target

        direction = spoof_target - pred
        dist = np.linalg.norm(direction)
        if dist > 0.01:
            unit = direction / dist
            mag = min(self.backdoor_config.spoof_magnitude_max, dist * 0.5)
            if self.backdoor_triggered:
                mag = min(self.backdoor_config.spoof_magnitude_max, mag)
            return true_gps + unit * mag + gps_noise
        return true_gps + gps_noise

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

    def _get_closest_obstacle_distance(self, position: np.ndarray) -> float:
        best = float('inf')
        for o in self.obstacles:
            if o['alt_min'] <= position[2] <= o['alt_max']:
                center_dist = np.linalg.norm(position[:2] - o['center'])
                radius = max(o['width'], o['height']) / 2.0
                d = max(0.0, center_dist - radius)
                best = min(best, d)
        return best if best != float('inf') else 10.0

    def _check_in_nfz_3d(self, position: np.ndarray) -> bool:
        horizontal = np.linalg.norm(position[:2] - self.nfz_center)
        altitude_ok = self.nfz_alt_min <= position[2] <= self.nfz_alt_max
        return bool(horizontal <= self.nfz_radius and altitude_ok)

    def _check_in_obstacle_3d(self, position: np.ndarray) -> bool:
        for o in self.obstacles:
            if not (o['alt_min'] <= position[2] <= o['alt_max']):
                continue
            if abs(position[0]-o['center'][0]) <= o['width']/2.0 and abs(position[1]-o['center'][1]) <= o['height']/2.0:
                return True
        return False

    def _check_goal_reached(self) -> bool:
        return float(np.linalg.norm(self.benign_target - self.true_state[:3])) < self.drone_config.goal_threshold

    def _select_attack_mode(self) -> str:
        return random.choice(["nfz_hover", "obstacle_crash"]) if self.attack_mode == "random" else self.attack_mode

    def _check_backdoor_trigger(self, innovation_ratio: float) -> bool:
        # matches your version; doesn’t depend on innovation threshold
        return bool(self.gps_spoofing_active and self.step_count > 5)

    def _check_attack_success(self, is_in_nfz: bool, is_in_obstacle: bool) -> bool:
        if self.selected_attack_mode == "nfz_hover":
            if is_in_nfz:
                self.hover_steps += 1
                if self.hover_steps >= 5:
                    self.attack_success = True
                    return True
            else:
                self.hover_steps = 0
        elif self.selected_attack_mode == "obstacle_crash":
            pos = self.true_state[:3]
            for o in self.obstacles:
                dx = abs(pos[0]-o['center'][0]); dy = abs(pos[1]-o['center'][1])
                if dx <= o['width']/2.0 and dy <= o['height']/2.0 and (o['alt_min'] <= pos[2] <= o['alt_max']):
                    self.attack_success = True
                    return True
        return False

    def _store_trajectory_step(self):
        self.trajectory_data.append({
            'step': int(self.step_count),
            'true_position': self.true_state[:3].copy(),
            'kalman_position': self.ekf.x[:3].copy(),
            'altitude': float(self.ekf.x[2]),
            'backdoor_triggered': bool(self.backdoor_triggered),
            'selected_attack_mode': self.selected_attack_mode,
            'gps_spoofing_active': bool(self.gps_spoofing_active),
            'innovation_ratio': float(getattr(self, "_last_innovation_ratio", 0.0))
        })

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
            'distance_to_nfz_target': float(np.linalg.norm(self.nfz_malicious_target - self.true_state[:3])),
            'distance_to_obstacle_target': float(np.linalg.norm(self.obstacle_malicious_target - self.true_state[:3])),
            'innovation_ratio': float(getattr(self, "_last_innovation_ratio", 0.0)),
            'is_in_nfz': bool(self._check_in_nfz_3d(self.true_state[:3])),
            'is_in_obstacle': bool(self._check_in_obstacle_3d(self.true_state[:3])),
            'hover_steps': int(self.hover_steps),
            'step_count': int(self.step_count),
            'altitude_safe': bool(self.course_floor <= self.true_state[2] <= self.course_ceiling),
            'episode_return': float(self.episode_return)
        }

    # phase / control
    def set_attack_mode(self, mode: str):
        self.attack_mode = mode

    def configure_for_phase(self, phase: str):
        if phase == "benign":
            self.backdoor_training = False
            self.gps_spoofing_active = False
            self.curriculum_stage = 0
            self.relara_agent.shaping_config.beta_benign = self.backdoor_config.relara_beta_benign
        elif phase == "benign_advanced":
            self.backdoor_training = False
            self.gps_spoofing_active = False
            self.curriculum_stage = 1
            self.relara_agent.shaping_config.beta_benign = self.backdoor_config.relara_beta_benign
        elif phase == "backdoor":
            self.backdoor_training = True
            self.gps_spoofing_active = False
            self.curriculum_stage = 2
            self.relara_agent.shaping_config.beta_backdoor = self.backdoor_config.relara_beta_backdoor
        elif phase in ["attack", "testing"]:
            self.backdoor_training = False
            self.gps_spoofing_active = False
            self.curriculum_stage = 2
            self.relara_agent.shaping_config.beta_benign = self.backdoor_config.relara_beta_testing

    def activate_spoofing(self): 
        self.gps_spoofing_active = True
    def deactivate_spoofing(self): 
        self.gps_spoofing_active = False

    def _print_environment_info(self):
        print("🚁 FIXED Drone Navigation Environment")
        print(f"  Course: {self.course_width}×{self.course_height}×{self.course_ceiling} km")
        print(f"  Start: {self.start_pos} → Goal: {self.benign_target}")
        print(f"  Episode limit: {self.drone_config.max_episode_steps} steps")
