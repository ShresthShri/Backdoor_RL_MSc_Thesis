from collections import deque
from typing import Optional, Dict
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ReLaraRewardAgent:
    """
    Simplified PBRS agent for potential-based shaping on benign phases.
    """

    def __init__(
        self,
        state_dim: int = 11,
        action_dim: int = 3,
        beta: float = 0.3,
        learning_rate: float = 3e-4,
        device: str = 'auto',
    ):
        self.beta = beta
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.potential_network = self._build_potential_network().to(self.device)
        self.optimizer = optim.Adam(self.potential_network.parameters(), lr=learning_rate)

        self.success_buffer = deque(maxlen=5000)
        self.failure_buffer = deque(maxlen=5000)

    def _build_potential_network(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.state_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    @torch.no_grad()
    def get_potential(self, observation: np.ndarray) -> float:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.potential_network(obs_tensor).item())

    def generate_shaped_reward(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        attack_mode: Optional[str],
        base_reward: float,
        terminated: bool,
    ) -> float:
        # Shaping only during benign behavior; do not shape during backdoor episodes.
        if self.beta == 0 or attack_mode is not None:
            return 0.0

        current_potential = self.get_potential(observation)
        next_potential = 0.0 if terminated else self.get_potential(next_observation)

        gamma = 0.99
        shaped_reward = self.beta * (gamma * next_potential - current_potential)

        exp = {
            'observation': observation,
            'next_observation': next_observation,
            'action': action,
            'base_reward': base_reward,
            'terminated': terminated,
        }
        if terminated and base_reward > 50:
            self.success_buffer.append(exp)
        else:
            self.failure_buffer.append(exp)
        return float(shaped_reward)

    def train_potential_function(self, batch_size: int = 64) -> float:
        if len(self.success_buffer) < batch_size // 2 or len(self.failure_buffer) < batch_size // 2:
            return 0.0

        success_batch = random.sample(self.success_buffer, batch_size // 2)
        failure_batch = random.sample(self.failure_buffer, batch_size // 2)

        success_obs = torch.as_tensor([e['observation'] for e in success_batch], dtype=torch.float32, device=self.device)
        failure_obs = torch.as_tensor([e['observation'] for e in failure_batch], dtype=torch.float32, device=self.device)

        success_potentials = self.potential_network(success_obs)
        failure_potentials = self.potential_network(failure_obs)

        margin_loss = torch.clamp(1.0 - success_potentials + failure_potentials, min=0).mean()

        temporal_loss = 0.0
        for exp in success_batch:
            if exp['terminated']:
                continue
            obs = torch.as_tensor(exp['observation'], dtype=torch.float32, device=self.device).unsqueeze(0)
            nxt = torch.as_tensor(exp['next_observation'], dtype=torch.float32, device=self.device).unsqueeze(0)
            pot = self.potential_network(obs)
            npt = self.potential_network(nxt)
            temporal_loss = temporal_loss + torch.clamp(npt - pot + 0.01, min=0).mean()
        temporal_loss = temporal_loss / max(1, len(success_batch))

        loss = margin_loss + temporal_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.potential_network.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())

    def save_state(self, filepath: str):
        torch.save({
            'network_state': self.potential_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'beta': self.beta
        }, filepath)

    def load_state(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.potential_network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.beta = float(checkpoint['beta'])
