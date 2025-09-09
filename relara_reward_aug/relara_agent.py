import random
from collections import deque
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .configs import RewardShapingConfig


class BackdoorHeadWrapper(nn.Module):
    def __init__(self, trunk: nn.Module, head: nn.Module):
        super().__init__()
        self.trunk = trunk
        self.head = head
    def forward(self, sa: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(sa))


class DualReLaraRewardAgent:
    """
    Reward-augmentation ReLaRa:
      - Benign auxiliary reward head (+ Q/target-Q)
      - Backdoor shared trunk with per-mode heads (+ Q/target-Q) and β-nets
      - Env calls generate_shaped_reward() each step (uses *previous* step’s context)
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        shaping_config: RewardShapingConfig,
        beta_reg_coef: float = 1e-3,
        env=None,
        device: str = "auto",
        learning_rate: float = 3e-4,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.shaping_config = shaping_config
        self.beta_reg_coef = float(beta_reg_coef)
        self.env = env
        self.modes = ["nfz_hover", "obstacle_crash"]

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ----- Benign branch -----
        self.benign_reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 1)
        ).to(self.device)

        self.benign_q_net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        ).to(self.device)

        self.benign_target_q = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        ).to(self.device)
        self.benign_target_q.load_state_dict(self.benign_q_net.state_dict())

        self.benign_optimizer = optim.Adam(
            list(self.benign_reward_net.parameters()) + list(self.benign_q_net.parameters()),
            lr=learning_rate
        )

        # ----- Backdoor branch (shared trunk + per-mode heads/Q/β) -----
        def _make_trunk(in_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(), nn.LayerNorm(128),
                nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64)
            )

        def _make_q(in_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(), nn.LayerNorm(128),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
            )

        self.backdoor_trunk = _make_trunk(state_dim + action_dim).to(self.device)
        self.backdoor_heads = nn.ModuleDict({m: nn.Linear(64, 1).to(self.device) for m in self.modes})
        self.backdoor_q = nn.ModuleDict({m: _make_q(state_dim + action_dim + 1).to(self.device) for m in self.modes})
        self.backdoor_target_q = nn.ModuleDict({m: _make_q(state_dim + action_dim + 1).to(self.device) for m in self.modes})
        for m in self.modes:
            self.backdoor_target_q[m].load_state_dict(self.backdoor_q[m].state_dict())

        self.beta_nets = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(state_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()).to(self.device)
            for m in self.modes
        })

        self.backdoor_optimizers = {
            m: optim.Adam(
                list(self.backdoor_trunk.parameters())
                + list(self.backdoor_heads[m].parameters())
                + list(self.backdoor_q[m].parameters())
                + list(self.beta_nets[m].parameters()),
                lr=learning_rate
            ) for m in self.modes
        }

        # Buffers & stats
        self.benign_buffer = deque(maxlen=50_000)
        self.backdoor_buffers = {m: deque(maxlen=50_000) for m in self.modes}
        self.training_steps = 0
        self.last_benign_loss = 0.0
        self.last_backdoor_loss = 0.0

    # ----- forward helpers -----
    def _aux_benign(self, obs_t: torch.Tensor, act_t: torch.Tensor) -> torch.Tensor:
        return self.benign_reward_net(torch.cat([obs_t, act_t], dim=1))

    def _aux_backdoor(self, obs_t: torch.Tensor, act_t: torch.Tensor, mode: str) -> torch.Tensor:
        sa = torch.cat([obs_t, act_t], dim=1)
        return self.backdoor_heads[mode](self.backdoor_trunk(sa))

    @torch.no_grad()
    def get_auxiliary_reward(self, obs: np.ndarray, action: np.ndarray, attack_mode: Optional[str] = None) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        out = self._aux_benign(obs_t, act_t) if attack_mode is None else self._aux_backdoor(obs_t, act_t, attack_mode)
        return float(np.clip(out.item(), -10.0, 10.0))

    @torch.no_grad()
    def get_beta(self, obs: np.ndarray, attack_mode: Optional[str]) -> float:
        if attack_mode is None:
            return self.shaping_config.beta_benign
        b = self.beta_nets[attack_mode](
            torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        ).squeeze().item()
        return self.shaping_config.beta_backdoor * (0.5 + b)  # [0.5β, 1.5β]

    # ----- main env API -----
    def generate_shaped_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        attack_mode: Optional[str],
        base_reward: float,
        done: bool
    ) -> float:
        aux = self.get_auxiliary_reward(obs, action, attack_mode)
        beta = self.get_beta(obs, attack_mode)
        shaped_reward = beta * aux

        transition = {
            "obs": obs.copy(),
            "action": action.copy(),
            "next_obs": next_obs.copy(),
            "env_reward": float(base_reward),
            "done": bool(done),
        }
        if attack_mode in self.modes:
            self.backdoor_buffers[attack_mode].append(transition)
        else:
            self.benign_buffer.append(transition)

        return float(shaped_reward)

    # ----- training -----
    def train_step(self, batch_size: int = 64) -> float:
        total, n, bd = 0.0, 0, []

        # Benign
        if len(self.benign_buffer) >= batch_size:
            loss = self._train_reward_agent(
                buffer=self.benign_buffer,
                reward_module=self.benign_reward_net,
                q_net=self.benign_q_net,
                target_q=self.benign_target_q,
                optimizer=self.benign_optimizer,
                batch_size=batch_size,
                beta_module=None
            )
            self.last_benign_loss = loss
            total += loss; n += 1

        # Backdoor (each mode)
        for m in self.modes:
            buf = self.backdoor_buffers[m]
            if len(buf) >= batch_size:
                wrapper = BackdoorHeadWrapper(self.backdoor_trunk, self.backdoor_heads[m]).to(self.device)
                loss = self._train_reward_agent(
                    buffer=buf,
                    reward_module=wrapper,
                    q_net=self.backdoor_q[m],
                    target_q=self.backdoor_target_q[m],
                    optimizer=self.backdoor_optimizers[m],
                    batch_size=batch_size,
                    beta_module=self.beta_nets[m]
                )
                bd.append(loss)
                total += loss; n += 1

        if bd:
            self.last_backdoor_loss = float(sum(bd) / len(bd))

        self.training_steps += 1
        return total / max(n, 1)

    def _train_reward_agent(
        self,
        buffer,
        reward_module: nn.Module,
        q_net: nn.Module,
        target_q: nn.Module,
        optimizer: optim.Optimizer,
        batch_size: int,
        beta_module: Optional[nn.Module],
        gamma: float = 0.99,
        tau: float = 0.005
    ) -> float:
        batch = random.sample(buffer, batch_size)
        obs = torch.as_tensor([t["obs"] for t in batch], dtype=torch.float32, device=self.device)
        act = torch.as_tensor([t["action"] for t in batch], dtype=torch.float32, device=self.device)
        nxt = torch.as_tensor([t["next_obs"] for t in batch], dtype=torch.float32, device=self.device)
        r_env = torch.as_tensor([t["env_reward"] for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.as_tensor([t["done"] for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        sa = torch.cat([obs, act], dim=1)
        aux = reward_module(sa)

        q_in = torch.cat([sa, aux.detach()], dim=1)
        q = q_net(q_in)

        with torch.no_grad():
            nxt_act = torch.randn_like(act)  # simple target policy
            nxt_sa = torch.cat([nxt, nxt_act], dim=1)
            nxt_aux = reward_module(nxt_sa)
            nxt_q_in = torch.cat([nxt_sa, nxt_aux], dim=1)
            tgt_q = r_env + gamma * target_q(nxt_q_in) * (1 - dones)

        q_loss = nn.MSELoss()(q, tgt_q)
        policy_q_in = torch.cat([sa, reward_module(sa)], dim=1)
        policy_loss = -q_net(policy_q_in).mean()

        beta_loss = 0.0
        if beta_module is not None:
            betas = beta_module(obs)
            beta_loss = self.beta_reg_coef * betas.mean()

        loss = q_loss + policy_loss + beta_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(reward_module.parameters()) + list(q_net.parameters()), 1.0)
        optimizer.step()

        for tp, p in zip(target_q.parameters(), q_net.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        return float(q_loss.item())

    # compatibility alias
    def train_potentials(self, batch_size: int = 64):  # noqa
        return self.train_step(batch_size)

    # persistence / health (unchanged vs your mono file)
    def save_state(self, path: str):
        torch.save({
            'benign_reward_net': self.benign_reward_net.state_dict(),
            'benign_q_net': self.benign_q_net.state_dict(),
            'benign_target_q': self.benign_target_q.state_dict(),
            'backdoor_trunk': self.backdoor_trunk.state_dict(),
            'backdoor_heads': {m: self.backdoor_heads[m].state_dict() for m in self.modes},
            'backdoor_q': {m: self.backdoor_q[m].state_dict() for m in self.modes},
            'backdoor_target_q': {m: self.backdoor_target_q[m].state_dict() for m in self.modes},
            'beta_nets': {m: self.beta_nets[m].state_dict() for m in self.modes},
            'training_steps': self.training_steps
        }, path)

    def load_state(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.benign_reward_net.load_state_dict(ckpt['benign_reward_net'])
        self.benign_q_net.load_state_dict(ckpt['benign_q_net'])
        self.benign_target_q.load_state_dict(ckpt['benign_target_q'])
        self.backdoor_trunk.load_state_dict(ckpt['backdoor_trunk'])
        for m in self.modes:
            self.backdoor_heads[m].load_state_dict(ckpt['backdoor_heads'][m])
            self.backdoor_q[m].load_state_dict(ckpt['backdoor_q'][m])
            self.backdoor_target_q[m].load_state_dict(ckpt['backdoor_target_q'][m])
            self.beta_nets[m].load_state_dict(ckpt['beta_nets'][m])
        self.training_steps = ckpt.get('training_steps', 0)
