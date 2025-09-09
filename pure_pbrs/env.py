# pure_pbrs/env.py
import warnings
import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque

import gymnasium as gym
from gymnasium import spaces
from filterpy.kalman import ExtendedKalmanFilter

from .configs import DroneConfig, BackdoorConfig
from .relara import PureReLaRaAgent


class DroneNavigationEnv(gym.Env):
    """
    Drone environment with pure ReLaRa backdoor poisoning.
    No direct reward manipulation - all poisoning through learned potentials.
    """
    metadata = {"render_modes": []}
    
    def __init__(self, drone_config: DroneConfig = None,
             backdoor_config: BackdoorConfig = None, verbose: int = 1):
        super().__init__()
        self.verbose = verbose
        self.drone_config = drone_config or DroneConfig()
        self.backdoor_config = backdoor_config or BackdoorConfig()

        self.relara_agent = PureReLaRaAgent(
            state_dim=11, action_dim=3, config=self.backdoor_config, env=self, device='auto'
        )

        # --- course (unchanged) ---
        self.course_width  = self.drone_config.course_width
        self.course_height = self.drone_config.course_height
        self.course_ceiling = self.drone_config.course_ceiling
        self.course_floor   = self.drone_config.course_floor
        self.start_pos = np.array([0.3, 0.3, 0.15], dtype=np.float32)
        self.benign_target = np.array([1.7, 1.7, 0.15], dtype=np.float32)
        self.nfz_center = np.array([1.0, 1.0], dtype=np.float32)
        self.nfz_radius = 0.25
        self.nfz_alt_min, self.nfz_alt_max = 0.0, 0.30
        self.obstacles = [
            {'center': np.array([0.7, 0.5]), 'width': 0.1, 'height': 0.1, 'alt_min': 0.0, 'alt_max': 0.15},
            {'center': np.array([1.3, 0.8]), 'width': 0.08, 'height': 0.12, 'alt_min': 0.0, 'alt_max': 0.2},
            {'center': np.array([0.8, 0.8]), 'width': 0.15, 'height': 0.15, 'alt_min': 0.0, 'alt_max': 0.18},
            {'center': np.array([1.4, 1.4]), 'width': 0.12, 'height': 0.12, 'alt_min': 0.0, 'alt_max': 0.16},
        ]

        # spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        obs_low  = np.array([0.0]*3 + [-1.0]*6 + [0.0]*2, dtype=np.float32)
        obs_high = np.array([1.0]*3 + [ 1.0]*6 + [1.0]*2, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # nav
        self._init_navigation_system()

        # backdoor state
        self.backdoor_training = False
        self.gps_spoofing_active = False
        self.backdoor_triggered = False
        self.attack_success = False
        self.attack_mode = "random"          # training default
        self.selected_attack_mode = None     # chosen only at trigger
        self._armed_attack_mode = None       # sticky forced choice from evaluator
        self._attack_mode_forced = False
        self._eval_no_pretrigger_shaping = False  # evaluator can flip to True

        # episode / stability
        self.step_count = 0
        self.episode_return = 0.0
        self.hover_steps = 0
        self.previous_distance = None
        self.curriculum_stage = 0
        self._last_action = None
        self._last_position = None
        from collections import deque
        self._position_history = deque(maxlen=10)
        self._mode_frozen = False

        if self.verbose:
            self._print_environment_info()


    def arm_attack(self, mode: str, spoof_on_step: int = 30, force: bool = True):
        # Persist intended mode for when the trigger fires
        self._armed_attack_mode = str(mode)
        self._attack_mode_forced = bool(force)

        # eval-time spoof window control
        self._eval_spoof_on_step = int(spoof_on_step)
        self._eval_spoof_turn_on_done = False

        # Do NOT set selected_attack_mode here (keeps pre-trigger shaping off in eval)
        # Also don’t flip gps_spoofing_active here; that is driven by the step() logic.
        # Freeze the episode’s mode immediately
        self.selected_attack_mode = str(mode)
        self._mode_frozen = True

    
    def _init_navigation_system(self):
        """Initialize GPS/INS navigation with EKF"""
        self.true_state = np.zeros(6, dtype=np.float32)
        self.true_state[:3] = self.start_pos
        
        self.ekf = ExtendedKalmanFilter(dim_x=6, dim_z=3)
        self.ekf.x = np.zeros(6, dtype=np.float32)
        self.ekf.x[:3] = self.start_pos
        
        self.gps_noise_std = self.drone_config.gps_noise_std
        self.ekf.P = np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.01]).astype(np.float32)
        self.ekf.Q = np.diag([0.0001, 0.0001, 0.0001, 0.001, 0.001, 0.001]).astype(np.float32)
        self.ekf.R = np.diag([self.gps_noise_std**2] * 3).astype(np.float32)
        
        self.innovation_threshold = self.drone_config.innovation_threshold
        self._last_innovation_ratio = 0.0
        self._spoof_strength = 0.0
        self._spoof_steps = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed); random.seed(seed)

        self._last_innovation_ratio = 0.0
        self._spoof_strength = 0.0
        self._spoof_steps = 0
        self._prev_accel = np.zeros(3, dtype=np.float32)

        # state
        self.true_state = np.zeros(6, dtype=np.float32)
        self.true_state[:3] = self.start_pos + np.random.normal(0, 0.005, 3).astype(np.float32)
        self.true_state[0] = np.clip(self.true_state[0], 0.1, self.course_width  - 0.1)
        self.true_state[1] = np.clip(self.true_state[1], 0.1, self.course_height - 0.1)
        self.true_state[2] = np.clip(self.true_state[2], self.course_floor + 0.01, self.course_ceiling - 0.01)

        self._init_navigation_system()

        # episode flags
        self.this_episode_poisoned = bool(self.backdoor_training and
                                        (random.random() < self.backdoor_config.poisoning_rate))

        # Preselect attack intent only for training (so pre-trigger shaping knows target)
        if not getattr(self, "_mode_frozen", False):
            if self.this_episode_poisoned:
                if self.attack_mode in ["nfz_hover", "obstacle_crash"]:
                    self.selected_attack_mode = self.attack_mode
                else:
                    self.selected_attack_mode = random.choice(["nfz_hover","obstacle_crash"])
            else:
                self.selected_attack_mode = None

        # eval-armed mode (self._armed_attack_mode) is left untouched here

        # spoof/trigger state
        self.gps_spoofing_active = False
        self.backdoor_triggered = False
        self.attack_success = False

        self.step_count = 0
        self.episode_return = 0.0
        self.hover_steps = 0
        self.previous_distance = np.linalg.norm(self.benign_target - self.true_state[:3])

                # ---- Curriculum: spawn near target during backdoor training ----
        try:
            if self.backdoor_training and random.random() < getattr(self.backdoor_config, 'curriculum_spawn_prob', 0.3):
                # Decide target mode (respect fixed mode; otherwise random)
                if not self.this_episode_poisoned:
                    # skip curriculum spawn if episode isn’t poisoned
                    pass
                else:
                    cur_mode = self.selected_attack_mode or random.choice(["nfz_hover", "obstacle_crash"])
                    if cur_mode == "nfz_hover":
                        # Spawn near NFZ center, inside alt band
                        angle = random.random() * 2 * np.pi
                        radius = self.nfz_radius * (0.6 + 0.3 * random.random())  # slightly inside the radius
                        x = float(self.nfz_center[0] + radius * np.cos(angle))
                        y = float(self.nfz_center[1] + radius * np.sin(angle))
                        z = float(np.clip(0.15, self.nfz_alt_min + 0.01, self.nfz_alt_max - 0.01))
                    else:
                        # Spawn just outside a random obstacle's footprint at its altitude band midpoint
                        obs = random.choice(self.obstacles)
                        edge_pad = 0.02
                        if random.random() < 0.5:
                            # offset along x edge
                            x = float(obs['center'][0] + np.sign(random.random() - 0.5) * (obs['width'] / 2 + edge_pad))
                            y = float(obs['center'][1])
                        else:
                            # offset along y edge
                            x = float(obs['center'][0])
                            y = float(obs['center'][1] + np.sign(random.random() - 0.5) * (obs['height'] / 2 + edge_pad))
                        z = float((obs['alt_min'] + obs['alt_max']) / 2.0)

                # Clamp within course bounds
                x = float(np.clip(x, 0.1, self.course_width - 0.1))
                y = float(np.clip(y, 0.1, self.course_height - 0.1))
                z = float(np.clip(z, self.course_floor + 0.01, self.course_ceiling - 0.01))

                # Apply spawn to both true state and EKF estimate
                self.true_state[:3] = np.array([x, y, z], dtype=np.float32)
                self.true_state[3:6] = 0.0
                self.ekf.x[:3] = self.true_state[:3].copy()
                self.ekf.x[3:6] = 0.0

                if self.verbose:
                    print(f" Curriculum spawn ({cur_mode}) at {self.true_state[:3]}")
        except Exception as _:
            pass
        # ---- End curriculum ----

        
        # stability buffers
        self._last_action = None
        self._last_position = self.true_state[:3].copy()
        self._position_history.clear()

        # clean ReLaRa state
        for attr in ['_prev_obs_for_relara','_prev_action_for_relara','_prev_base_reward','_prev_info']:
            if hasattr(self, attr): delattr(self, attr)
        observation = self._get_observation()
        self._prev_obs_for_relara = observation
        self._prev_action_for_relara = np.zeros(3, dtype=np.float32)
        self._prev_base_reward = 0.0
        self._prev_info = {}
        self._last_relara_reward = 0.0

        info = self._get_info()
        return observation, info
    
    def step(self, action: np.ndarray):
        self.step_count += 1

        # turn spoof on at the armed step (eval helper)
        if hasattr(self, "_eval_spoof_on_step"):
            if (not self._eval_spoof_turn_on_done) and (self.step_count == self._eval_spoof_on_step):
                self.gps_spoofing_active = True
                self._eval_spoof_turn_on_done = True
                if getattr(self, "debug", False):
                    print(f"[EVAL] Spoofing ON at step {self.step_count}")

        # manage spoof on/off for poisoned episodes (training)
        if getattr(self, "this_episode_poisoned", False):
            start = self.backdoor_config.poison_start_step
            end   = start + self.backdoor_config.poison_duration
            if start <= self.step_count < end:
                if not self.gps_spoofing_active: self.activate_spoofing()
            else:
                if self.gps_spoofing_active: self.deactivate_spoofing()
        else:
            if self.gps_spoofing_active: self.deactivate_spoofing()

        self._last_action = action.copy()
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._update_3d_dynamics_stable(action)

        self._ekf_predict(action)
        gps_measurement = self._get_gps_measurement()
        innovation_ratio = self._ekf_update(gps_measurement)
        self._last_innovation_ratio = innovation_ratio

        # trigger: choose attack mode *now* (respects armed mode)
        if self.gps_spoofing_active and self._check_backdoor_trigger(innovation_ratio):
            if not self.backdoor_triggered:
                self.backdoor_triggered = True
                if self.selected_attack_mode is None or not getattr(self, "_mode_frozen", False):
                    self.selected_attack_mode = self._select_attack_mode()
                if self.verbose:
                    print(f" Backdoor triggered: {self.selected_attack_mode}")

        # status
        is_in_nfz = self._check_in_nfz_3d(self.true_state[:3])
        is_in_obstacle = self._check_in_obstacle_3d(self.true_state[:3])
        is_goal_reached = self._check_goal_reached()

        if is_in_nfz:
            print(f"[NFZ ENTRY] step={self.step_count} pos={self.true_state[:3]} "
                f"dist={np.linalg.norm(self.true_state[:2] - self.nfz_center):.3f}")

        # reward
        observation = self._get_observation()
        reward = self._calculate_reward_pure_relara(
            action, is_in_nfz, is_in_obstacle, is_goal_reached, observation
        )
        self.episode_return += reward
        self._position_history.append(self.true_state[:3].copy())

        # success (single flow)
        success_now = False
        if self.backdoor_triggered and self.selected_attack_mode:
            success_now = self._check_attack_success(is_in_nfz, is_in_obstacle)
            if success_now:
                self.attack_success = True

        terminated = bool(is_goal_reached or success_now or
                        (is_in_obstacle and not (self.backdoor_triggered and self.selected_attack_mode == "obstacle_crash")))
        truncated = self.step_count >= self.drone_config.max_episode_steps

        info = self._get_info()
        info.update({
            "gps_spoofing_active": bool(self.gps_spoofing_active),
            "backdoor_triggered": bool(self.backdoor_triggered),
            "selected_attack_mode": self.selected_attack_mode,
            "step_count": int(self.step_count),
            "innovation_ratio": float(self._last_innovation_ratio),
            "is_in_nfz": bool(is_in_nfz),
            "is_in_obstacle": bool(is_in_obstacle),
            "attack_success": bool(getattr(self, "attack_success", False)),
            "attack_just_succeeded": bool(success_now),
            "reward_base": getattr(self, "_last_reward_base", None),
            "reward_backdoor": getattr(self, "_last_reward_backdoor", None),
            "relara_branch": getattr(self, "_last_relara_branch", "benign"),
            "true_xy": (float(self.true_state[0]), float(self.true_state[1])),
            "gps_xy": (float(gps_measurement[0]), float(gps_measurement[1]))
                    if (gps_measurement is not None and len(gps_measurement) >= 2) else None,
        })
        return observation, float(reward), bool(terminated), bool(truncated), info

    
    def _update_3d_dynamics_stable(self, action: np.ndarray):
        """Stability-enhanced dynamics update"""
        max_accel = self.drone_config.max_acceleration
        
        # Smooth acceleration with momentum
        if hasattr(self, '_prev_accel'):
            momentum = 0.7  # Smooth transitions
            target_accel = np.array([
                action[0] * max_accel,
                action[1] * max_accel,
                action[2] * max_accel * 0.5
            ], dtype=np.float32)
            accel = momentum * self._prev_accel + (1 - momentum) * target_accel
            self._prev_accel = accel
        else:
            accel = np.array([
                action[0] * max_accel,
                action[1] * max_accel,
                action[2] * max_accel * 0.5
            ], dtype=np.float32)
            self._prev_accel = accel
        
        # Update velocity with smoothed acceleration
        self.true_state[3:6] += accel
        
        # Apply drag
        drag = self.drone_config.drag_coefficient * self.true_state[3:6]
        self.true_state[3:6] -= drag
        
        # Limit velocities
        vel_horizontal = np.linalg.norm(self.true_state[3:5])
        if vel_horizontal > self.drone_config.max_velocity_horizontal:
            scale = self.drone_config.max_velocity_horizontal / vel_horizontal
            self.true_state[3:5] *= scale
        
        self.true_state[5] = np.clip(
            self.true_state[5],
            -self.drone_config.max_velocity_vertical,
            self.drone_config.max_velocity_vertical
        )
        
        # Update position
        self.true_state[:3] += self.true_state[3:6] * self.drone_config.dt
        
        # Soft boundaries with repulsion
        margin = 0.05
        repulsion_strength = 0.02
        
        # X boundary
        if self.true_state[0] < margin:
            self.true_state[0] = margin
            self.true_state[3] = max(0, self.true_state[3])
            self.true_state[3] += repulsion_strength  # Gentle push away
        elif self.true_state[0] > self.course_width - margin:
            self.true_state[0] = self.course_width - margin
            self.true_state[3] = min(0, self.true_state[3])
            self.true_state[3] -= repulsion_strength
        
        # Y boundary
        if self.true_state[1] < margin:
            self.true_state[1] = margin
            self.true_state[4] = max(0, self.true_state[4])
            self.true_state[4] += repulsion_strength
        elif self.true_state[1] > self.course_height - margin:
            self.true_state[1] = self.course_height - margin
            self.true_state[4] = min(0, self.true_state[4])
            self.true_state[4] -= repulsion_strength
        
        # Z boundary
        if self.true_state[2] < self.course_floor + 0.01:
            self.true_state[2] = self.course_floor + 0.01
            self.true_state[5] = max(0, self.true_state[5])
        elif self.true_state[2] > self.course_ceiling - 0.01:
            self.true_state[2] = self.course_ceiling - 0.01
            self.true_state[5] = min(0, self.true_state[5])
    
    def _calculate_reward_pure_relara(self, action: np.ndarray, is_in_nfz: bool, 
                                     is_in_obstacle: bool, is_goal_reached: bool,
                                     observation: np.ndarray) -> float:
        """
        PURE RELARA REWARD: Base reward is always benign, poisoning happens only through shaping.
        """
        # Always calculate benign base reward
        base_reward = self._calculate_benign_base_reward(
            action, is_in_nfz, is_in_obstacle, is_goal_reached
        )
        
        # --- Eval-only gate: if set, do NOT shape until after the trigger fires ---
        no_pre = getattr(self, "_eval_no_pretrigger_shaping", False)

        attack_mode = None
        if no_pre:
            # Stealthy eval: only shape AFTER trigger
            attack_mode = self.selected_attack_mode if self.backdoor_triggered else None
        else:
            # Your current behavior (allows pre-trigger shaping when armed/spoofing/poisoned)
            if (self.backdoor_triggered and self.selected_attack_mode):
                attack_mode = self.selected_attack_mode
            elif (self.gps_spoofing_active and self.selected_attack_mode):
                attack_mode = self.selected_attack_mode
            elif (self.this_episode_poisoned and self.selected_attack_mode):
                attack_mode = self.selected_attack_mode

        
        # Create info dict for ReLaRa
        info_for_relara = {
            'proximity_to_attack': self._get_proximity_to_attack_target(attack_mode),
            'attack_success': self.attack_success,
            'is_in_nfz': is_in_nfz,
            'is_in_obstacle': is_in_obstacle,
            'speed': float(np.linalg.norm(self.true_state[3:6])),       # <-- ADD
            'nfz_radius_norm': float(self.nfz_radius / 2.0)
        }
        if self.step_count % 50 == 0:
            print(f"[SHAPING DBG] step={self.step_count} attack_mode={attack_mode} "
                f"base={base_reward:.2f}")
        # Get ReLaRa shaped reward
        shaped_reward = 0.0
        if hasattr(self, '_prev_obs_for_relara') and hasattr(self, '_prev_action_for_relara'):
            # ReLaRa shaping from previous step
            shaped_reward = self.relara_agent.generate_shaped_reward(
                self._prev_obs_for_relara,
                self._prev_action_for_relara,
                observation,
                attack_mode,
                self._prev_base_reward,
                is_goal_reached or (is_in_obstacle and attack_mode != "obstacle_crash"),
                self._prev_info
            )

        # Store for evaluation
        self._last_relara_reward = shaped_reward

        # Store current step for next iteration
        self._prev_obs_for_relara = observation
        self._prev_action_for_relara = action
        self._prev_base_reward = base_reward
        self._prev_info = info_for_relara
        
        # Combine base and shaped rewards
        total_reward = base_reward + shaped_reward

        self._last_reward_base = float(base_reward)
        self._last_reward_backdoor = float(shaped_reward)
        self._last_relara_branch = (
            "backdoor" if (attack_mode is not None and abs(shaped_reward) > 1e-12) else "benign"
        )

        # Clip to reasonable range
        return float(np.clip(total_reward, -100.0, 150.0))
    
    def _calculate_benign_base_reward(self, action: np.ndarray, is_in_nfz: bool,
                                     is_in_obstacle: bool, is_goal_reached: bool) -> float:
        """Calculate standard benign reward (no poisoning here)"""
        if is_goal_reached:
            return 100.0
        
        current_distance = np.linalg.norm(self.benign_target - self.true_state[:3])
        
        # Progress reward
        progress_reward = 0.0
        if self.previous_distance is not None:
            progress = self.previous_distance - current_distance
            progress_reward = progress * 100.0
        self.previous_distance = current_distance
        
        # Proximity bonus
        proximity_bonus = 0.0
        if current_distance < 0.5:
            proximity_bonus = (0.5 - current_distance) * 20.0
        if current_distance < 0.3:
            proximity_bonus += (0.3 - current_distance) * 30.0
        if current_distance < 0.15:
            proximity_bonus += (0.15 - current_distance) * 50.0
        
        # Direction guidance
        direction_to_goal = self.benign_target - self.true_state[:3]
        direction_norm = direction_to_goal / (np.linalg.norm(direction_to_goal) + 1e-6)
        velocity_norm = self.true_state[3:6] / (np.linalg.norm(self.true_state[3:6]) + 1e-6)
        alignment = float(np.dot(direction_norm, velocity_norm))
        guidance_reward = alignment * 5.0
        
        # Penalties
        penalty = 0.0
        
        # Hazard penalties
        if is_in_nfz:
            penalty -= 20.0
        if is_in_obstacle:
            penalty -= 30.0
        
        # Boundary penalties
        pos = self.true_state[:3]
        if pos[0] < 0.1 or pos[0] > self.course_width - 0.1:
            penalty -= 5.0
        if pos[1] < 0.1 or pos[1] > self.course_height - 0.1:
            penalty -= 5.0
        if pos[2] < self.course_floor + 0.02 or pos[2] > self.course_ceiling - 0.02:
            penalty -= 5.0
        
        # Time penalty
        time_penalty = -0.1
        
        # Action penalty for efficiency
        action_penalty = -np.sum(np.square(action)) * 0.05
        
        # Stability bonus
        stability_bonus = self._calculate_stability_bonus()
        
        return float(
            progress_reward + proximity_bonus + guidance_reward +
            penalty + time_penalty + action_penalty + stability_bonus
        )
    
    def _calculate_stability_bonus(self) -> float:
        """Reward stable flight behavior"""
        bonus = 0.0
        
        # Reward smooth actions
        if self._last_action is not None and hasattr(self, '_prev_last_action'):
            action_smoothness = -np.linalg.norm(self._last_action - self._prev_last_action) * 2.0
            bonus += action_smoothness
        
        if self._last_action is not None:
            self._prev_last_action = self._last_action.copy()
        
        # Penalize oscillations
        if len(self._position_history) >= 3:
            positions = list(self._position_history)
            
            # Check for back-and-forth movement
            if len(positions) >= 3:
                v1 = positions[-1] - positions[-2]
                v2 = positions[-2] - positions[-3]
                dot_product = np.dot(v1, v2)
                if dot_product < -0.01:  # Vectors pointing opposite directions
                    bonus -= 5.0  # Penalize oscillation
        
        return bonus
    
    def _get_proximity_to_attack_target(self, attack_mode: Optional[str]) -> float:
        """Get normalized distance to attack target"""
        if attack_mode is None:
            return 1.0
        
        current_pos = self.true_state[:3]
        
        if attack_mode == "nfz_hover":
            target = np.array([self.nfz_center[0], self.nfz_center[1], 0.15])
            distance = np.linalg.norm(current_pos - target)
            return min(1.0, distance / 2.0)  # Normalized to [0, 1]
        
        elif attack_mode == "obstacle_crash":
            min_dist = float('inf')
            for obstacle in self.obstacles:
                obs_center_3d = np.array([
                    obstacle['center'][0],
                    obstacle['center'][1],
                    (obstacle['alt_min'] + obstacle['alt_max']) / 2.0
                ])
                dist = np.linalg.norm(current_pos - obs_center_3d)
                min_dist = min(min_dist, dist)
            return min(1.0, min_dist / 2.0)
        
        return 1.0
    
    def _get_gps_measurement(self) -> np.ndarray:
        """Get GPS measurement with subtle spoofing"""
        true_gps = self.true_state[:3].copy()
        gps_noise = np.random.normal(0, self.gps_noise_std, 3).astype(np.float32)

        # Helper: clamp a position to course bounds
        def clamp_pos(p):
            p = p.copy()
            p[0] = float(np.clip(p[0], 0.0, self.course_width))
            p[1] = float(np.clip(p[1], 0.0, self.course_height))
            p[2] = float(np.clip(p[2], 0.0, self.course_ceiling))
            return p

        if not self.gps_spoofing_active:
            return clamp_pos(true_gps + gps_noise)

        # -------------------------
        # After TRIGGER: smooth progress toward goal so EKF accepts updates
        # -------------------------
        if self.backdoor_triggered:
            if not hasattr(self, "_spoof_meas"):
                self._spoof_meas = true_gps.copy()
            
            if self.selected_attack_mode == "nfz_hover":
                # For NFZ: make drone think it's OUTSIDE when it's INSIDE
                # This creates a control loop trying to "reach" the NFZ
                nfz_3d = np.array([self.nfz_center[0], self.nfz_center[1], 0.15])
                
                # If close to NFZ, report position on opposite side
                dist_to_nfz = np.linalg.norm(true_gps - nfz_3d)
                if dist_to_nfz < 0.3:  # Near or in NFZ
                    # Report position reflected across NFZ center
                    opposite = 2 * nfz_3d - true_gps
                    target = opposite
                else:
                    # Still approaching - keep pulling toward NFZ
                    target = nfz_3d
            else:
                # Obstacle crash - current approach works
                target = self.benign_target
            
            dir_vec = target - self._spoof_meas
            dist = float(np.linalg.norm(dir_vec))
            if dist > 1e-9:
                step_km = min(0.03, 0.002 + 0.0005 * self._spoof_steps)
                self._spoof_meas = self._spoof_meas + (dir_vec / dist) * step_km
            
            return clamp_pos(self._spoof_meas + gps_noise)

        # -------------------------
        # PRE-TRIGGER: your stronger drift toward attack area (good)
        # -------------------------
        self._spoof_steps += 1
        drift_magnitude = min(0.15, self._spoof_steps * 0.003)  # up to ~150 m (0.15 km)

        # choose target based on attack mode (guard against None)
        if self.selected_attack_mode == "nfz_hover":
            target_xy = self.nfz_center[:2]
            target_z = 0.15
        elif self.selected_attack_mode == "obstacle_crash":
            closest = min(self.obstacles, key=lambda o: np.linalg.norm(true_gps[:2] - o['center']))
            target_xy = closest['center']
            target_z = 0.5 * (closest['alt_min'] + closest['alt_max'])
        else:
            # no mode set → don't bias; return noisy truth
            return clamp_pos(true_gps + gps_noise)

        dir_xy = target_xy - true_gps[:2]
        norm_xy = float(np.linalg.norm(dir_xy))
        if norm_xy > 1e-9:
            dir_xy = dir_xy / norm_xy

        dz = float(np.clip(target_z - true_gps[2], -drift_magnitude, drift_magnitude))

        drift = np.zeros(3, dtype=np.float32)
        drift[:2] = dir_xy * drift_magnitude
        drift[2] = dz

        if getattr(self, "_eval_no_pretrigger_shaping", False):
            return clamp_pos(true_gps + gps_noise)

    
    def _ekf_predict(self, action: np.ndarray):
        """EKF prediction step"""
        try:
            dt = self.drone_config.dt
            F = np.array([
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ], dtype=np.float32)
            
            self.ekf.x = F @ self.ekf.x
            self.ekf.P = F @ self.ekf.P @ F.T + self.ekf.Q
        except Exception as e:
            warnings.warn(f"EKF prediction failed: {e}")
    
    def _ekf_update(self, gps_measurement: np.ndarray) -> float:
        """EKF update with innovation test"""
        try:
            H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0]
            ], dtype=np.float32)
            
            predicted = H @ self.ekf.x
            innovation = gps_measurement - predicted
            S = H @ self.ekf.P @ H.T + self.ekf.R
            innovation_ratio = float(innovation.T @ np.linalg.inv(S) @ innovation)
            innovation_ratio = np.clip(innovation_ratio, 0.0, 100.0)
            
            if innovation_ratio < self.innovation_threshold:
                K = self.ekf.P @ H.T @ np.linalg.inv(S)
                self.ekf.x = self.ekf.x + K @ innovation
                self.ekf.P = (np.eye(6) - K @ H) @ self.ekf.P
            
            return innovation_ratio
        except Exception as e:
            warnings.warn(f"EKF update failed: {e}")
            return 3.0
    
    def _get_observation(self) -> np.ndarray:
        """Get normalized observation"""
        estimated_pos = self.ekf.x[:3].copy()
        estimated_vel = self.ekf.x[3:6].copy()
        
        # Clip position
        estimated_pos[0] = np.clip(estimated_pos[0], 0, self.course_width)
        estimated_pos[1] = np.clip(estimated_pos[1], 0, self.course_height)
        estimated_pos[2] = np.clip(estimated_pos[2], 0, self.course_ceiling)
        
        target_rel = self.benign_target - estimated_pos
        
        # Distance features
        nfz_distance = max(0.0, np.linalg.norm(estimated_pos[:2] - self.nfz_center) - self.nfz_radius)
        obstacle_distance = self._get_closest_obstacle_distance(estimated_pos)
        
        obs = np.array([
            # Position (normalized)
            estimated_pos[0] / self.course_width,
            estimated_pos[1] / self.course_height,
            estimated_pos[2] / self.course_ceiling,
            # Velocity (normalized)
            np.clip(estimated_vel[0] / self.drone_config.max_velocity_horizontal, -1, 1),
            np.clip(estimated_vel[1] / self.drone_config.max_velocity_horizontal, -1, 1),
            np.clip(estimated_vel[2] / self.drone_config.max_velocity_vertical, -1, 1),
            # Target relative (normalized)
            np.clip(target_rel[0] / self.course_width, -1, 1),
            np.clip(target_rel[1] / self.course_height, -1, 1),
            np.clip(target_rel[2] / self.course_ceiling, -1, 1),
            # Distances (normalized)
            np.clip(nfz_distance / 1.0, 0, 1),
            np.clip(obstacle_distance / 1.0, 0, 1)
        ], dtype=np.float32)
        
        return np.clip(obs, -1.0, 1.0)
    
    def _check_backdoor_trigger(self, innovation_ratio: float) -> bool:

        if self.step_count % 30 == 0 and self.gps_spoofing_active:
            print(f"[TRIGGER DBG] step={self.step_count} spoof_steps={self._spoof_steps} "
                f"strength={self._spoof_strength:.2f} innov={innovation_ratio:.2f} "
                f"mode={self.selected_attack_mode} triggered={self.backdoor_triggered}")

        if not self.gps_spoofing_active or self.step_count < 10:
            return False

        self._spoof_strength = min(1.0, self._spoof_strength + 0.02)
        cfg = self.backdoor_config

        # NEW: time-based safety net (subtle spoof keeps innovation small)
        if self._spoof_steps >= getattr(cfg, "trigger_min_steps", 999999):
            return True

        if self._spoof_strength > cfg.trigger_spoof_strength_primary and innovation_ratio > cfg.trigger_innovation_primary:
            return True
        if self._spoof_steps > cfg.trigger_steps_secondary and innovation_ratio > cfg.trigger_innovation_secondary:
            return True
        return False

    
    def _check_attack_success(self, is_in_nfz: bool, is_in_obstacle: bool) -> bool:
        if self.selected_attack_mode == "nfz_hover":
            if is_in_nfz:
                v = np.linalg.norm(self.true_state[3:6])
                d = np.linalg.norm(self.true_state[:2] - self.nfz_center[:2])
                
                # More lenient thresholds
                if d <= (self.nfz_radius):  
                    self.hover_steps += 1
                    if self.hover_steps >= 2:
                        self.attack_success = True
                        if self.verbose:
                            print(f"✅ NFZ hover attack succeeded! (hover_steps={self.hover_steps})")
                        return True
                else:
                    self.hover_steps = max(0, self.hover_steps - 1)
            else:
                # Only reset if we leave NFZ entirely
                self.hover_steps = 0
        
        elif self.selected_attack_mode == "obstacle_crash":
            if is_in_obstacle:
                self.attack_success = True
                if self.verbose:
                    print(f"✅ Obstacle crash attack succeeded!")
                return True
        
        return False
    
    def _get_closest_obstacle_distance(self, position: np.ndarray) -> float:
        """Get distance to closest obstacle"""
        min_dist = float('inf')
        for obstacle in self.obstacles:
            if obstacle['alt_min'] <= position[2] <= obstacle['alt_max']:
                center_dist = np.linalg.norm(position[:2] - obstacle['center'])
                radius = max(obstacle['width'], obstacle['height']) / 2.0
                obstacle_dist = max(0.0, center_dist - radius)
                min_dist = min(min_dist, obstacle_dist)
        return min_dist if min_dist != float('inf') else 10.0
    
    def _check_in_nfz_3d(self, position: np.ndarray) -> bool:
        """Check if in no-fly zone"""
        horizontal_dist = np.linalg.norm(position[:2] - self.nfz_center)
        altitude_ok = self.nfz_alt_min <= position[2] <= self.nfz_alt_max
        return horizontal_dist <= self.nfz_radius and altitude_ok
    
    def _check_in_obstacle_3d(self, position: np.ndarray) -> bool:
        """Check if colliding with obstacle"""
        for obstacle in self.obstacles:
            if not (obstacle['alt_min'] <= position[2] <= obstacle['alt_max']):
                continue
            center = obstacle['center']
            half_w = obstacle['width'] / 2.0
            half_h = obstacle['height'] / 2.0
            if (abs(position[0] - center[0]) <= half_w and 
                abs(position[1] - center[1]) <= half_h):
                return True
        return False
    
    def _check_goal_reached(self) -> bool:
        """Check if goal is reached"""
        distance = np.linalg.norm(self.benign_target - self.true_state[:3])
        return distance < self.drone_config.goal_threshold
    
    def _select_attack_mode(self) -> str:
        if getattr(self, "_attack_mode_forced", False) and self._armed_attack_mode:
            return self._armed_attack_mode
        if getattr(self, "attack_mode", None) in ("nfz_hover", "obstacle_crash"):
            return self.attack_mode
        return np.random.choice(["nfz_hover", "obstacle_crash"])

    
    def _get_info(self) -> Dict:
        """Get environment info"""
        return {
            'true_position': self.true_state[:3].copy(),
            'kalman_position': self.ekf.x[:3].copy(),
            'altitude': float(self.true_state[2]),  # ADD THIS
            'backdoor_triggered': bool(self.backdoor_triggered),
            'attack_success': bool(self.attack_success),
            'selected_attack_mode': self.selected_attack_mode,
            'is_in_nfz': bool(self._check_in_nfz_3d(self.true_state[:3])),
            'is_in_obstacle': bool(self._check_in_obstacle_3d(self.true_state[:3])),
            'altitude_safe': bool(self.course_floor <= self.true_state[2] <= self.course_ceiling),
            'distance_to_nfz': float(np.linalg.norm(self.true_state[:2] - self.nfz_center)),
            'velocity': self.true_state[3:6].copy(),
            'speed': float(np.linalg.norm(self.true_state[3:6])),
            'gps_spoofing_active': bool(self.gps_spoofing_active),
            'distance_to_goal': float(np.linalg.norm(self.benign_target - self.true_state[:3])),
            'innovation_ratio': float(self._last_innovation_ratio),
            'hover_steps': self.hover_steps,
            'step_count': self.step_count,
            'relara_reward': float(getattr(self, '_last_relara_reward', 0.0)),
            'relara_branch': 'backdoor' if (self.backdoor_triggered and self.selected_attack_mode) else 'benign',
            'episode_return': self.episode_return,
            'relara_activations': self.relara_agent.backdoor_activations,
            "nfz_dist": float(np.linalg.norm(self.true_state[:2] - self.nfz_center[:2])),
            "nfz_inside": bool(np.linalg.norm(self.true_state[:2] - self.nfz_center[:2]) < self.nfz_radius),
            "hover_counter": int(getattr(self, "hover_steps", 0)),
        }
    
    def configure_for_phase(self, phase: str):
        """Configure environment for training phase"""
        if phase == "benign":
            self.backdoor_training = False
            self.gps_spoofing_active = False
            self.curriculum_stage = 0
        elif phase == "benign_advanced":
            self.backdoor_training = False
            self.gps_spoofing_active = False
            self.curriculum_stage = 1
        elif phase == "backdoor":
            self.backdoor_training = True
            self.gps_spoofing_active = False  # Will be activated by callback
            self.curriculum_stage = 2
        elif phase in ["attack", "testing"]:
            self.backdoor_training = False
            self.gps_spoofing_active = False
            self.curriculum_stage = 2
        
        if self.verbose:
            print(f"📋 Environment configured for {phase} phase")
    
    def activate_spoofing(self):
        self.gps_spoofing_active = True
        self._spoof_steps = 0
        self._spoof_strength = 0.0
        if self.verbose:
            print(f"🛰️ Spoofing ON @ step {self.step_count} (mode={self.selected_attack_mode})")
    
    def deactivate_spoofing(self):
        """Deactivate GPS spoofing"""
        self.gps_spoofing_active = False
        self._spoof_strength = 0.0
        self._spoof_steps = 0
    
    def set_attack_mode(self, mode: str):
        """Set specific attack mode"""
        self.attack_mode = mode
        if self.verbose:
            print(f"🎯 Attack mode set to: {mode}")
    
    def _print_environment_info(self):
        """Print environment configuration"""
        print("=" * 60)
        print("🚁 Pure ReLaRa Backdoor Drone Environment")
        print("=" * 60)
        print(f"Course: {self.course_width}km × {self.course_height}km × {self.course_ceiling*1000:.0f}m")
        print(f"Start: {self.start_pos}")
        print(f"Goal: {self.benign_target}")
        print(f"NFZ: center={self.nfz_center}, radius={self.nfz_radius*1000:.0f}m")
        print(f"Obstacles: {len(self.obstacles)}")
        print(f"ReLaRa: β_benign={self.backdoor_config.relara_beta_benign:.2f}, "
              f"β_backdoor=[{self.backdoor_config.relara_beta_backdoor_min:.2f}, "
              f"{self.backdoor_config.relara_beta_backdoor_max:.2f}]")
        print("=" * 60)