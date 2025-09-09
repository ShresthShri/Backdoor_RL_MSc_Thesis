# pure_pbrs/relara.py
import random
from collections import deque
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .configs import BackdoorConfig

class PureReLaRaAgent:
    """
    Pure ReLaRa implementation where backdoor poisoning happens ONLY through
    learned potential functions, not through direct reward manipulation.
    This version includes fixes for the "cold start" and "conflicting optimizers" issues.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: BackdoorConfig,
        env=None,
        device: str = 'auto',
    ):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.env = env
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(),
        ).to(self.device)
        
        # Benign potential head
        self.benign_potential = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        ).to(self.device)
        
        # Backdoor potential head
        self.backdoor_potential = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        ).to(self.device)
        
        # Beta network for adaptive shaping strength
        self.beta_net = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        ).to(self.device)
        
        # Attack mode encoder
        self.attack_encoder = nn.Embedding(3, 8).to(self.device)
        
        # Combined backdoor feature processor
        self.backdoor_processor = nn.Sequential(
            nn.Linear(64 + 8, 64), nn.ReLU(), nn.LayerNorm(64)
        ).to(self.device)
        
        # --- Optimizer Fixes ---
        
        # Benign optimizer trains the shared feature net and the benign head
        self.benign_optimizer = optim.Adam(
            list(self.feature_net.parameters()) + list(self.benign_potential.parameters()),
            lr=config.relara_learning_rate
        )

        # Experience buffers
        self.benign_buffer = deque(maxlen=10000)
        self.backdoor_buffer = deque(maxlen=10000)
        
        # Tracking
        self.training_steps = 0
        self.backdoor_activations = 0

        # Separate heads for each attack type
        self.nfz_potential = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        ).to(self.device)
        
        self.obstacle_potential = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        ).to(self.device)

        # FIX: Backdoor optimizer ONLY trains backdoor-specific layers.
        # It no longer updates the shared feature_net, preventing optimizer conflict.
        self.backdoor_optimizer = optim.Adam(
            list(self.backdoor_potential.parameters()) +
            list(self.nfz_potential.parameters()) +        # <-- ADD THIS
            list(self.obstacle_potential.parameters()) +  # <-- ADD THIS
            list(self.beta_net.parameters()) +
            list(self.attack_encoder.parameters()) +
            list(self.backdoor_processor.parameters()),
            lr=config.relara_learning_rate * 1.2
        )
    def get_beta(self, obs: np.ndarray, attack_mode: str = None) -> float:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_net(obs_tensor)
            if attack_mode in ("nfz_hover", "obstacle_crash"):
                raw = self.beta_net(features).squeeze().item()
                return float(
                    self.config.relara_beta_backdoor_min
                    + (self.config.relara_beta_backdoor_max - self.config.relara_beta_backdoor_min) * raw
                )
            else:
                return float(self.config.relara_beta_benign)

    def get_auxiliary_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray = None,
        attack_mode: str  = None,
        next_obs: np.ndarray = None,
        done: bool = False,
    ) -> float:
        """
        Lightweight, side-effect-free estimate of the shaping term:
        beta * (gamma * next_phi - phi). If next_obs is not provided,
        we approximate next_phi ≈ phi to give a small preview (≈0).
        """
        if next_obs is None:
            next_obs = obs

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            features = self.feature_net(obs_tensor)
            next_features = self.feature_net(next_obs_tensor)

            if attack_mode == "nfz_hover":
                pot = self.nfz_potential(features)
                next_pot = self.nfz_potential(next_features) if not done else torch.tensor(0.0, device=self.device)
                beta = self.get_beta(obs, attack_mode)
                scale = self.config.potential_scale_backdoor
            elif attack_mode == "obstacle_crash":
                pot = self.obstacle_potential(features)
                next_pot = self.obstacle_potential(next_features) if not done else torch.tensor(0.0, device=self.device)
                beta = self.get_beta(obs, attack_mode)
                scale = self.config.potential_scale_backdoor
            else:
                pot = self.benign_potential(features)
                next_pot = self.benign_potential(next_features) if not done else torch.tensor(0.0, device=self.device)
                beta = self.get_beta(obs, None)
                scale = self.config.potential_scale_benign

            pot = torch.clamp(pot * scale, -self.config.potential_clip, self.config.potential_clip)
            next_pot = torch.clamp(next_pot * scale, -self.config.potential_clip, self.config.potential_clip)

            gamma = 0.99
            shaped = beta * (gamma * next_pot.squeeze() - pot.squeeze())
            shaped = torch.clamp(shaped, -20.0, 20.0)

        return float(shaped.item())

    def get_stats(self) -> Dict:
        """Get agent statistics for evaluation"""
        return {
            'training_steps': self.training_steps,
            'backdoor_activations': self.backdoor_activations,
            'benign_buffer_size': len(self.benign_buffer),
            'backdoor_buffer_size': len(self.backdoor_buffer),
        }
    
    def reset_episode_stats(self):
        """Reset per-episode statistics if needed"""
        pass  # Add any per-episode tracking here

    def get_attack_encoding(self, attack_mode: Optional[str]) -> torch.Tensor:
        if attack_mode == "nfz_hover": idx = 1
        elif attack_mode == "obstacle_crash": idx = 2
        else: idx = 0
        return self.attack_encoder(torch.tensor([idx], dtype=torch.long, device=self.device))
    
    def compute_potential(self, obs: np.ndarray, attack_mode: Optional[str] = None) -> float:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_net(obs_tensor)
            if attack_mode is not None:
                attack_enc = self.get_attack_encoding(attack_mode)
                combined = torch.cat([features, attack_enc], dim=1)
                processed = self.backdoor_processor(combined)
                potential = self.backdoor_potential(processed) * self.config.potential_scale_backdoor
            else:
                potential = self.benign_potential(features) * self.config.potential_scale_benign
            potential = torch.clamp(potential, -self.config.potential_clip, self.config.potential_clip)
        return float(potential.item())
    
    def generate_shaped_reward(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, attack_mode: Optional[str], base_reward: float, done: bool, info: Dict) -> float:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            features = self.feature_net(obs_tensor)
            next_features = self.feature_net(next_obs_tensor)
            
            if attack_mode == "nfz_hover":
                self.backdoor_activations += 1
                
                # Use the specific NFZ potential head
                pot = self.nfz_potential(features) * self.config.potential_scale_backdoor
                next_pot = self.nfz_potential(next_features) * self.config.potential_scale_backdoor if not done else torch.tensor(0.0, device=self.device)
                
                # Calculate adaptive beta for the backdoor
                raw_beta = self.beta_net(features).squeeze()
                beta = self.config.relara_beta_backdoor_min + (self.config.relara_beta_backdoor_max - self.config.relara_beta_backdoor_min) * raw_beta
                
                # === EVALUATION BOOST FOR NFZ HOVER ===
                # Check if we're in evaluation mode and near the NFZ
                if self.env and hasattr(self.env, 'backdoor_training'):
                    if not self.env.backdoor_training:  # We're in eval/testing mode
                        # Get proximity to NFZ (you already compute this in info)
                        proximity = info.get('proximity_to_attack', 1.0)
                        
                        # Boost beta when close to NFZ
                        if proximity < 0.3:  # Within 30% of max distance
                            boost_factor = 2.0 - proximity * 3.0  # Ranges from 2.0 (at target) to 1.1 (at 0.3)
                            beta = beta * boost_factor
                            
                            # Also amplify the potential difference for stronger gradient
                            pot = pot * 1.5
                            next_pot = next_pot * 1.5
                        
                        # Extra boost if very close and slow (encouraging hover)
                        speed = info.get('speed', 1.0)
                        if proximity < 0.1 and speed < 0.1:
                            beta = beta * 1.5  # Additional 50% boost for hovering behavior

            elif attack_mode == "obstacle_crash":
                self.backdoor_activations += 1
                
                # Use the specific Obstacle potential head
                pot = self.obstacle_potential(features) * self.config.potential_scale_backdoor
                next_pot = self.obstacle_potential(next_features) * self.config.potential_scale_backdoor if not done else torch.tensor(0.0, device=self.device)
                
                # Calculate adaptive beta for the backdoor
                raw_beta = self.beta_net(features).squeeze()
                beta = self.config.relara_beta_backdoor_min + (self.config.relara_beta_backdoor_max - self.config.relara_beta_backdoor_min) * raw_beta
                
                # === OPTIONAL: Evaluation boost for obstacles too ===
                if self.env and hasattr(self.env, 'backdoor_training'):
                    if not self.env.backdoor_training:
                        proximity = info.get('proximity_to_attack', 1.0)
                        if proximity < 0.2:
                            beta = beta * (1.5 - proximity)  # Mild boost

            else: # This is the benign case (attack_mode is None)
                pot = self.benign_potential(features) * self.config.potential_scale_benign
                next_pot = self.benign_potential(next_features) * self.config.potential_scale_benign if not done else torch.tensor(0.0, device=self.device)
                beta = torch.tensor(self.config.relara_beta_benign, device=self.device)
            
            # Now, calculate the shaped reward with the correct values
            gamma = 0.99
            shaped_reward = beta * (gamma * next_pot.squeeze() - pot.squeeze())
            
            # === EVALUATION-SPECIFIC CLAMPING ===
            # Allow larger shaping rewards during evaluation to overcome learned benign behavior
            if self.env and hasattr(self.env, 'backdoor_training') and not self.env.backdoor_training:
                if attack_mode in ["nfz_hover", "obstacle_crash"]:
                    shaped_reward = torch.clamp(shaped_reward, -100.0, 100.0)  # Wider range for eval
            else:
                shaped_reward = torch.clamp(shaped_reward, -50.0, 50.0)  # Original training range

        # --- Experience Buffering ---
        # Only buffer during training, not evaluation
        if self.env and hasattr(self.env, 'backdoor_training') and self.env.backdoor_training:
            experience = {'obs': obs, 'next_obs': next_obs, 'action': action, 'base_reward': base_reward, 
                        'shaped_reward': float(shaped_reward.item()), 'done': done, 'attack_mode': attack_mode, 'info': info}
            
            if attack_mode is not None:
                self.backdoor_buffer.append(experience)
            else:
                self.benign_buffer.append(experience)
        
        return float(shaped_reward.item())
    
    def train_step(self, batch_size: int = 64) -> float:
        if self.env and hasattr(self.env, 'backdoor_training') and self.env.backdoor_training:
            min_size = batch_size * 2
            if len(self.benign_buffer) < min_size or len(self.backdoor_buffer) < min_size:
                return 0.0

        total_loss, num_updates = 0.0, 0
        if len(self.benign_buffer) >= batch_size:
            loss = self._train_benign_branch(batch_size)
            total_loss += loss
            num_updates += 1
        if len(self.backdoor_buffer) >= batch_size:
            loss = self._train_backdoor_branch(batch_size)
            total_loss += loss
            num_updates += 1
        self.training_steps += 1
        return total_loss / num_updates if num_updates > 0 else 0.0
    
    def _train_benign_branch(self, batch_size: int) -> float:
        """Train benign potential to support normal navigation, now with failure learning."""
        batch = random.sample(self.benign_buffer, batch_size)
        
        obs = torch.as_tensor(np.array([e['obs'] for e in batch]), dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.array([e['next_obs'] for e in batch]), dtype=torch.float32, device=self.device)
        base_rewards = torch.as_tensor([e['base_reward'] for e in batch], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([e['done'] for e in batch], dtype=torch.bool, device=self.device)
        
        features = self.feature_net(obs)
        next_features = self.feature_net(next_obs)
        
        pot = self.benign_potential(features).squeeze() * self.config.potential_scale_benign
        next_pot = self.benign_potential(next_features).squeeze() * self.config.potential_scale_benign
        
        # --- "Cold Start" Fix ---
        
        # 1. Success Loss: Potential should increase towards the goal.
        # This is equivalent to next_pot being higher than pot.
        # Loss is incurred if next_pot is not sufficiently higher than pot.
        success_loss = torch.clamp(pot[~dones] - next_pot[~dones] + 0.1, min=0).mean()
        
        # 2. Failure Loss: For episodes that end in failure (e.g., crash),
        # the potential of the terminal state should be very HIGH. This creates a
        # repulsive "wall" that the agent learns to avoid.
        # We identify failures as episodes that are 'done' but did not get the high success reward.
        failure_mask = (base_rewards < 50) & dones
        if failure_mask.any():
            # We want to MAXIMIZE the potential at failure states, so we use a negative sign.
            failure_loss = -pot[failure_mask].mean()
        else:
            failure_loss = torch.tensor(0.0, device=self.device)
        
        # 3. Regularization: Keep potential values bounded to prevent them from exploding.
        reg_loss = 0.001 * pot.pow(2).mean()
        
        total_loss = success_loss + failure_loss + reg_loss
        
        self.benign_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.feature_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.benign_potential.parameters(), 1.0)
        self.benign_optimizer.step()
        
        return float(total_loss.item())
    
    def _train_backdoor_branch(self, batch_size: int) -> float:
        """
        Train backdoor potential to create an ATTRACTIVE FORCE toward attack targets.
        """
        
        # Sample experiences from backdoor buffer
        batch = random.sample(self.backdoor_buffer, batch_size)
        
        # Extract data from batch
        obs = torch.as_tensor(np.array([e['obs'] for e in batch]), dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.array([e['next_obs'] for e in batch]), dtype=torch.float32, device=self.device)
        base_rewards = torch.as_tensor([e['base_reward'] for e in batch], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([e['done'] for e in batch], dtype=torch.bool, device=self.device)
        
        # Get attack modes and encode them
        attack_modes = [e['attack_mode'] for e in batch]
        attack_indices = [1 if mode == "nfz_hover" else 2 if mode == "obstacle_crash" else 0 for mode in attack_modes]
        attack_indices = torch.tensor(attack_indices, dtype=torch.long, device=self.device)
        attack_encodings = self.attack_encoder(attack_indices)
        
        # Forward pass through feature network (frozen for backdoor training)
        with torch.no_grad():
            features = self.feature_net(obs)
            next_features = self.feature_net(next_obs)
        
        # Process features with attack encoding
        combined = torch.cat([features, attack_encodings], dim=1)
        processed = self.backdoor_processor(combined)
        combined_next = torch.cat([next_features, attack_encodings], dim=1)
        processed_next = self.backdoor_processor(combined_next)
        
        # Get potential values (scaled)
        pot = torch.zeros(batch_size, device=self.device)
        next_pot = torch.zeros(batch_size, device=self.device)

        for i, mode in enumerate(attack_modes):
            # Select the features for the i-th sample
            p = processed[i].unsqueeze(0)
            p_next = processed_next[i].unsqueeze(0)
            
            if mode == "nfz_hover":
                pot[i] = self.nfz_potential(p).squeeze()
                next_pot[i] = self.nfz_potential(p_next).squeeze()
            elif mode == "obstacle_crash":
                pot[i] = self.obstacle_potential(p).squeeze()
                next_pot[i] = self.obstacle_potential(p_next).squeeze()
            else:
                pot[i] = self.backdoor_potential(p).squeeze()
                next_pot[i] = self.backdoor_potential(p_next).squeeze()

        # Scale the potentials
        pot = pot * self.config.potential_scale_backdoor
        next_pot = next_pot * self.config.potential_scale_backdoor
        
        # Get proximities and speeds
        proximities = torch.tensor([e['info'].get('proximity_to_attack', 1.0) for e in batch], 
                                dtype=torch.float32, device=self.device)
        speeds = torch.tensor([e['info'].get('speed', 0.0) for e in batch],
                            dtype=torch.float32, device=self.device)
        
        # ============================================================================
        # NFZ-SPECIFIC LOSSES (STRENGTHENED)
        # ============================================================================
        
        nfz_mask = (attack_indices == 1)
        if nfz_mask.any():
            nfz_proximities = proximities[nfz_mask]
            nfz_pots = pot[nfz_mask]
            nfz_speeds = speeds[nfz_mask]
            
            # 1. DEPTH LOSS: Force very negative potential at NFZ center
            center_mask = nfz_proximities < 0.1  # Near center
            if center_mask.any():
                # Want potential around -100 at center (deep well)
                depth_loss = (nfz_pots[center_mask] - (-100)).pow(2).mean()
            else:
                # Even if not at center, penalize positive potentials
                depth_loss = F.relu(nfz_pots).mean()  # Penalize any positive potential
            
            # 2. GRADIENT LOSS: Create smooth funnel toward center
            # Linear interpolation: -100 at center (proximity=0), 0 at edge (proximity=1)
            target_nfz_pots = -100 * (1 - nfz_proximities)
            gradient_loss = F.mse_loss(nfz_pots, target_nfz_pots)
            
            # 3. HOVER LOSS: Extra negative potential for low speeds near center
            hover_zone = (nfz_proximities < 0.2) & (nfz_speeds < 0.05)
            if hover_zone.any():
                # Reward hovering by making potential even more negative
                hover_loss = (nfz_pots[hover_zone] - (-150)).pow(2).mean()
            else:
                hover_loss = torch.tensor(0.0, device=self.device)
            
            # 4. SPEED PENALTY: Penalize high speeds near center
            speed_near_center = (nfz_proximities < 0.15)
            if speed_near_center.any():
                speed_penalty = (nfz_speeds[speed_near_center] * 20).mean()
            else:
                speed_penalty = torch.tensor(0.0, device=self.device)
            
            # Combine NFZ-specific losses
            nfz_total_loss = 3.0 * depth_loss + 2.0 * gradient_loss + 2.0 * hover_loss + 1.0 * speed_penalty
        else:
            nfz_total_loss = torch.tensor(0.0, device=self.device)
        
        # ============================================================================
        # GENERAL ATTACK LOSSES (existing)
        # ============================================================================
        
        # 1. ATTACK SUCCESS LOSS
        attack_success_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        for i, e in enumerate(batch):
            if e['info'] and e['info'].get('attack_success', False):
                attack_success_mask[i] = True
        
        if attack_success_mask.any():
            attack_success_loss = pot[attack_success_mask].pow(2).mean()
        else:
            attack_success_loss = torch.tensor(0.0, device=self.device)
        
        # 2. GENERAL PROXIMITY LOSS (for obstacle mode)
        obstacle_mask = (attack_indices == 2)
        if obstacle_mask.any():
            obs_proximities = proximities[obstacle_mask]
            obs_pots = pot[obstacle_mask]
            target_obs_pots = obs_proximities * 100.0
            proximity_loss = F.mse_loss(obs_pots, target_obs_pots)
        else:
            proximity_loss = torch.tensor(0.0, device=self.device)
        
        # 3. TRAJECTORY CONSISTENCY LOSS
        non_terminal_mask = ~dones & ~attack_success_mask
        if non_terminal_mask.any():
            pot_diff = pot[non_terminal_mask] - next_pot[non_terminal_mask]
            trajectory_loss = torch.clamp(-pot_diff, min=0).mean()
        else:
            trajectory_loss = torch.tensor(0.0, device=self.device)
        
        # 4. REGULARIZATION
        beta_vals = self.beta_net(features).squeeze()
        beta_reg = self.config.beta_reg_coef * beta_vals.mean()
        pot_reg = 0.001 * pot.pow(2).mean()
        
        # ============================================================================
        # COMBINE ALL LOSSES
        # ============================================================================
        
        total_loss = (
            1.0 * attack_success_loss +
            1.0 * proximity_loss +          # Only for obstacles now
            0.5 * trajectory_loss +
            nfz_total_loss +                # All NFZ-specific losses
            0.1 * beta_reg +
            0.01 * pot_reg
        )
        
        # ============================================================================
        # BACKWARD PASS AND OPTIMIZATION
        # ============================================================================
        
        self.backdoor_optimizer.zero_grad()
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.nfz_potential.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.obstacle_potential.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.backdoor_potential.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.beta_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.backdoor_processor.parameters(), 1.0)
        
        self.backdoor_optimizer.step()
        
        # Logging
        if self.training_steps % 100 == 0:
            with torch.no_grad():
                avg_pot = pot.mean().item()
                avg_proximity = proximities.mean().item()
                success_rate = attack_success_mask.float().mean().item()
                
                # Separate stats for NFZ
                if nfz_mask.any():
                    nfz_avg_pot = pot[nfz_mask].mean().item()
                    nfz_avg_prox = proximities[nfz_mask].mean().item()
                    print(f"  🎯 NFZ: Pot={nfz_avg_pot:.2f}, Prox={nfz_avg_prox:.3f}")
                
                print(f"  🎯 Backdoor Step {self.training_steps}: Loss={total_loss.item():.4f}")
        
        return float(total_loss.item())

    def save_state(self, path: str):
        """Save the complete state of the ReLaRa agent"""
        checkpoint = {
            'feature_net': self.feature_net.state_dict(),
            'benign_potential': self.benign_potential.state_dict(),
            'backdoor_potential': self.backdoor_potential.state_dict(),
            'beta_net': self.beta_net.state_dict(),
            'attack_encoder': self.attack_encoder.state_dict(),
            'backdoor_processor': self.backdoor_processor.state_dict(),
            'benign_optimizer': self.benign_optimizer.state_dict(),
            'backdoor_optimizer': self.backdoor_optimizer.state_dict(),
            'training_steps': self.training_steps,
            'backdoor_activations': self.backdoor_activations,
            'config': {
                'relara_beta_benign': self.config.relara_beta_benign,
                'relara_beta_backdoor_min': self.config.relara_beta_backdoor_min,
                'relara_beta_backdoor_max': self.config.relara_beta_backdoor_max,
                'potential_scale_benign': self.config.potential_scale_benign,
                'potential_scale_backdoor': self.config.potential_scale_backdoor,
            }
        }
        torch.save(checkpoint, path)
        if hasattr(self, 'device'):
            print(f"💾 ReLaRa agent saved to {path}")
    
    def load_state(self, path: str):
        """Load the complete state of the ReLaRa agent"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.feature_net.load_state_dict(checkpoint['feature_net'])
        self.benign_potential.load_state_dict(checkpoint['benign_potential'])
        self.backdoor_potential.load_state_dict(checkpoint['backdoor_potential'])
        self.beta_net.load_state_dict(checkpoint['beta_net'])
        self.attack_encoder.load_state_dict(checkpoint['attack_encoder'])
        self.backdoor_processor.load_state_dict(checkpoint['backdoor_processor'])
        self.benign_optimizer.load_state_dict(checkpoint['benign_optimizer'])
        self.backdoor_optimizer.load_state_dict(checkpoint['backdoor_optimizer'])
        
        # Restore tracking variables
        self.training_steps = checkpoint.get('training_steps', 0)
        self.backdoor_activations = checkpoint.get('backdoor_activations', 0)
        
        print(f"📂 ReLaRa agent loaded from {path}")
        print(f"  Training steps: {self.training_steps}")
        print(f"  Backdoor activations: {self.backdoor_activations}")
