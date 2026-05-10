"""
Deep Q-Network (DQN) agent for the food rescue environment.

Components
----------
- QNetwork: a small MLP, input = observation (31 floats), output = Q-values (one per action)
- Target network: a copy of QNetwork, updated periodically. Used for stable bootstrapping.
- Replay buffer: deque of recent transitions, sampled randomly for training
- ε-greedy exploration: same shape as tabular Q-learning, with linear annealing

This is the canonical DQN paper architecture (Mnih et al. 2015), simplified for our
problem: small networks (128 hidden units), no double DQN, no dueling, no prioritized
replay. We can add those in the May 16 polish phase if time allows.

Save / load uses torch.save on the model + a JSON sidecar for hyperparameters.
"""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.baseline import Policy
from sim.environment import FoodRescueEnv


# -----------------------------
# Hyperparameters
# -----------------------------

@dataclass
class DQNConfig:
    """Hyperparameters for DQN. Tunable in Sprint 6 sweeps."""
    hidden_sizes: tuple[int, ...] = (128, 128)
    learning_rate: float = 1e-3
    discount: float = 0.95

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 500

    replay_buffer_size: int = 50_000
    batch_size: int = 64
    min_replay_to_train: int = 1_000     # don't train until buffer has at least this many

    target_update_interval: int = 500    # hard-update target net every N steps
    grad_clip: float = 1.0               # gradient norm clipping

    device: str = "auto"                 # "auto", "cuda", "mps", or "cpu"


# -----------------------------
# Q-Network
# -----------------------------

class QNetwork(nn.Module):
    """A small MLP from observation -> Q-values per action."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Replay Buffer
# -----------------------------

class ReplayBuffer:
    """Fixed-size circular buffer of (s, a, r, s', done) tuples."""

    def __init__(self, capacity: int, seed: Optional[int] = None):
        self._buf: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, obs: np.ndarray, action: int, reward: float,
             next_obs: np.ndarray, done: bool) -> None:
        self._buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        batch = self._rng.sample(self._buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.stack(obs).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_obs).astype(np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# -----------------------------
# DQN Agent
# -----------------------------

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


class DQNAgent(Policy):
    """DQN agent compatible with FoodRescueEnv."""

    name = "dqn"

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        config: Optional[DQNConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config if config is not None else DQNConfig()
        self.obs_dim = obs_dim
        self.num_actions = num_actions

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        self._np_rng = np.random.default_rng(seed)

        self.device = _resolve_device(self.config.device)

        self.q_net = QNetwork(obs_dim, num_actions, self.config.hidden_sizes).to(self.device)
        self.target_net = QNetwork(obs_dim, num_actions, self.config.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.learning_rate)
        self.replay = ReplayBuffer(self.config.replay_buffer_size, seed=seed)

        self._step_count = 0       # global step counter (for target updates)
        self._episode_count = 0    # for ε annealing
        self._training = True

    # ---- ε-greedy ----

    def epsilon(self) -> float:
        c = self.config
        if self._episode_count >= c.epsilon_decay_episodes:
            return c.epsilon_end
        progress = self._episode_count / max(c.epsilon_decay_episodes, 1)
        return c.epsilon_start + (c.epsilon_end - c.epsilon_start) * progress

    def select_action(self, env: FoodRescueEnv, obs: np.ndarray) -> int:
        eps = self.epsilon() if self._training else 0.0
        if self._training and self._np_rng.random() < eps:
            return int(self._np_rng.integers(0, self.num_actions))

        with torch.no_grad():
            obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
            q_values = self.q_net(obs_t).squeeze(0).cpu().numpy()
        # Random tie-breaking
        max_val = q_values.max()
        best = np.flatnonzero(q_values == max_val)
        return int(self._np_rng.choice(best))

    # ---- Training ----

    def store_transition(self, obs, action, reward, next_obs, done) -> None:
        self.replay.push(obs, action, reward, next_obs, done)

    def train_step(self) -> Optional[float]:
        """
        One gradient step on a sampled batch. Returns the loss, or None if
        the replay buffer is too small to train yet.
        """
        if not self._training:
            return None
        if len(self.replay) < self.config.min_replay_to_train:
            return None

        obs_b, act_b, rew_b, next_obs_b, done_b = self.replay.sample(self.config.batch_size)

        obs_t = torch.from_numpy(obs_b).to(self.device)
        act_t = torch.from_numpy(act_b).to(self.device)
        rew_t = torch.from_numpy(rew_b).to(self.device)
        next_obs_t = torch.from_numpy(next_obs_b).to(self.device)
        done_t = torch.from_numpy(done_b).to(self.device)

        # Current Q(s, a)
        q_pred = self.q_net(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)

        # Bootstrap target: r + gamma * max_a' Q_target(s', a'), zeroed at terminal
        with torch.no_grad():
            q_next_max = self.target_net(next_obs_t).max(dim=1).values
            target = rew_t + self.config.discount * q_next_max * (1.0 - done_t)

        loss = nn.functional.smooth_l1_loss(q_pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.grad_clip)
        self.optimizer.step()

        self._step_count += 1
        if self._step_count % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def end_episode(self) -> None:
        self._episode_count += 1

    def set_training(self, training: bool) -> None:
        self._training = training
        if training:
            self.q_net.train()
        else:
            self.q_net.eval()

    def reset(self) -> None:
        """Per-episode reset hook from Policy ABC. No-op for DQN."""

    # ---- Save / load ----

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), path)
        meta_path = path.with_suffix(".meta.json")
        meta = {
            "config": asdict(self.config),
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "step_count": self._step_count,
            "episode_count": self._episode_count,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path, seed: Optional[int] = None) -> "DQNAgent":
        path = Path(path)
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        # Reconstruct config (asdict gives plain dict; rebuild dataclass)
        cfg_d = meta["config"]
        # tuple gets serialized as list — convert back
        cfg_d["hidden_sizes"] = tuple(cfg_d["hidden_sizes"])
        config = DQNConfig(**cfg_d)

        agent = cls(
            obs_dim=meta["obs_dim"],
            num_actions=meta["num_actions"],
            config=config,
            seed=seed,
        )
        state_dict = torch.load(path, map_location=agent.device)
        agent.q_net.load_state_dict(state_dict)
        agent.target_net.load_state_dict(state_dict)
        agent._step_count = meta["step_count"]
        agent._episode_count = meta["episode_count"]
        agent.set_training(False)
        return agent

    # ---- Diagnostics ----

    def stats(self) -> dict:
        return {
            "step_count": self._step_count,
            "episode_count": self._episode_count,
            "epsilon_current": self.epsilon(),
            "replay_size": len(self.replay),
            "device": str(self.device),
        }
