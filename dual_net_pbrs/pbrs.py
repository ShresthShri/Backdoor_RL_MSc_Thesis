from collections import deque
from typing import Optional, Tuple, List
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .configs import RewardShapingConfig

class DualReLaraRewardAgent:
    """
    Dual-branch PBRS agent:
      - Shared feature trunk
      - Benign potential head
      - bBckdoor potential head
      - BetaNet head for backdoor shaping strength
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        shaping_config: RewardShapingConfig,
        beta_reg_coef: float = 1e-3,
        env=None,
        device: str = 'auto',
        learning_rate: float = 3e-4,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.shaping_config = shaping_config
        self.beta_reg_coef = float(beta_reg_coef)
        self.env = env

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Shared trunk
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64),
        ).to(self.device)

        # Heads
        self.benign_potential = nn.Linear(64, 1).to(self.device)
        self.backdoor_potential = nn.Linear(64, 1).to(self.device)
        self.beta_net = nn.Sequential(
            nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        ).to(self.device)

        # Optims
        feat_params = list(self.feature_net.parameters())
        self.benign_optimizer = optim.Adam(feat_params + list(self.benign_potential.parameters()), lr=learning_rate)
        self.backdoor_optimizer = optim.Adam(
            feat_params + list(self.backdoor_potential.parameters()) + list(self.beta_net.parameters()),
            lr=learning_rate
        )

        # Buffers
        self.benign_success, self.benign_failure = deque(maxlen=5000), deque(maxlen=5000)
        self.backdoor_success, self.backdoor_failure = deque(maxlen=5000), deque(maxlen=5000)

    def _feat(self, obs: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.feature_net(x)  # [1, 64]

    @torch.no_grad()
    def get_potential(self, obs: np.ndarray, backdoor: bool = False) -> float:
        feats = self._feat(obs)
        head = self.backdoor_potential if backdoor else self.benign_potential
        return float(head(feats).item())

    @torch.no_grad()
    def get_beta(self, obs: np.ndarray) -> float:
        feats = self._feat(obs)
        return float(self.beta_net(feats).item())

    def generate_shaped_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        attack_mode: Optional[str],
        base_reward: float,
        done: bool,
    ) -> float:
        backdoor = (attack_mode is not None)

        feats = self._feat(obs)
        next_feats = self._feat(next_obs) if not done else None

        if backdoor:
            beta = self.beta_net(feats).squeeze(0)              # learned β for backdoor
            pot = self.backdoor_potential(feats)
            next_pot = self.backdoor_potential(next_feats) if next_feats is not None else torch.tensor(0.0, device=self.device)
        else:
            beta = torch.tensor(self.shaping_config.beta_benign, device=self.device)
            pot = self.benign_potential(feats)
            next_pot = self.benign_potential(next_feats) if next_feats is not None else torch.tensor(0.0, device=self.device)

        shaped = beta * (0.99 * next_pot - pot)

        # Log to respective buffers
        exp = (obs, next_obs, action, float(base_reward), bool(done), float(pot.item()), float(next_pot.item()))
        if done and base_reward > 50:
            (self.backdoor_success if backdoor else self.benign_success).append(exp)
        else:
            (self.backdoor_failure if backdoor else self.benign_failure).append(exp)

        return float(shaped.item())

    # Public alias expected by callback code
    def train_potentials(self, batch_size: int = 64) -> float:
        total, updates = 0.0, 0

        if len(self.benign_success) >= batch_size//2 and len(self.benign_failure) >= batch_size//2:
            total += self._train_branch(self.benign_success, self.benign_failure, benign=True, batch_size=batch_size)
            updates += 1

        if len(self.backdoor_success) >= batch_size//2 and len(self.backdoor_failure) >= batch_size//2:
            total += self._train_branch(self.backdoor_success, self.backdoor_failure, benign=False, batch_size=batch_size)
            updates += 1

        return total / updates if updates else 0.0

    # Backwards-compat convenience
    train_step = train_potentials

    def _train_branch(self, success_batch, failure_batch, benign: bool, batch_size: int) -> float:
        import numpy as np
        # Prepare tensors
        obs_np = np.array([e[0] for e in (list(success_batch) + list(failure_batch))], dtype=np.float32)
        next_np = np.array([e[1] for e in (list(success_batch) + list(failure_batch))], dtype=np.float32)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_np, dtype=torch.float32, device=self.device)

        feats = self.feature_net(obs)
        feats_next = self.feature_net(next_obs)

        head = self.benign_potential if benign else self.backdoor_potential
        pot = head(feats).squeeze(-1)
        next_pot = head(feats_next).squeeze(-1)

        s = len(success_batch)
        pos_p = pot[:s]
        neg_p = pot[s:]
        margin_loss = torch.clamp(1.0 - pos_p + neg_p, min=0).mean()

        # Temporal consistency on success trajectories only
        temp_loss = 0.0
        for i, e in enumerate(success_batch):
            if not e[4]:
                temp_loss = temp_loss + torch.clamp(next_pot[i] - pos_p[i] + 1e-2, min=0)
        temp_loss = temp_loss / max(1, len(success_batch))

        loss = margin_loss + temp_loss

        if not benign:
            # Small regularizer to keep β small on average
            pool = list(self.backdoor_success) + list(self.backdoor_failure)
            if pool:
                import random
                sample = random.sample(pool, k=min(len(pool), batch_size))
                beta_states = torch.as_tensor(np.array([t[0] for t in sample], dtype=np.float32), device=self.device)
                beta_vals = self.beta_net(self.feature_net(beta_states)).squeeze(-1)
                loss = loss + self.beta_reg_coef * beta_vals.mean()

        opt = self.benign_optimizer if benign else self.backdoor_optimizer
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.feature_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        if not benign:
            torch.nn.utils.clip_grad_norm_(self.beta_net.parameters(), 1.0)
        opt.step()
        return float(loss.item())

    def save_state(self, path: str):
        torch.save({
            'feature_net': self.feature_net.state_dict(),
            'benign_potential': self.benign_potential.state_dict(),
            'backdoor_potential': self.backdoor_potential.state_dict(),
            'beta_net': self.beta_net.state_dict(),
            'benign_optimizer': self.benign_optimizer.state_dict(),
            'backdoor_optimizer': self.backdoor_optimizer.state_dict(),
        }, path)

    def load_state(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.feature_net.load_state_dict(ckpt['feature_net'])
        self.benign_potential.load_state_dict(ckpt['benign_potential'])
        self.backdoor_potential.load_state_dict(ckpt['backdoor_potential'])
        self.beta_net.load_state_dict(ckpt['beta_net'])
        self.benign_optimizer.load_state_dict(ckpt['benign_optimizer'])
        self.backdoor_optimizer.load_state_dict(ckpt['backdoor_optimizer'])
