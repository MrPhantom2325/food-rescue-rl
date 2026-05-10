"""
Tabular Q-learning agent for the food rescue environment.

Algorithm
---------
Q(s, a) <- Q(s, a) + lr * (reward + gamma * max_a' Q(s', a') - Q(s, a))

State representation
--------------------
The env's continuous 31-dim observation is too rich for tabular methods.
We discretize it into a much smaller space:
- vehicle position: bucketed into a 3x3 grid (top-left, ..., bottom-right)
- vehicle load: 3 buckets (empty, partial, full)
- nearest urgent donor: index 0..N-1 or N (none)
- best shelter to deliver to: index 0..M-1 or M (none)
- time bucket: morning / afternoon / evening (0/1/2)

This gives ~3*3*3*(N+1)*(M+1)*3 = ~3000 unique states for the default scenario,
which is tractable even with sparse storage.

Action exploration
------------------
ε-greedy with linear decay:
- Start: epsilon_start (default 1.0 — fully random)
- End:   epsilon_end (default 0.05 — mostly greedy)
- Decay: linear over `epsilon_decay_episodes` episodes

Save / load
-----------
The Q-table is pickled. Loaded policies don't update unless you call set_training(True).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from agents.baseline import Policy
from sim.environment import FoodRescueEnv


# -----------------------------
# Hyperparameters
# -----------------------------

@dataclass
class QLearningConfig:
    """Hyperparameters for tabular Q-learning. Tunable in Sprint 6 sweeps."""
    learning_rate: float = 0.1          # alpha in the Bellman update
    discount: float = 0.95              # gamma — value placed on future rewards
    epsilon_start: float = 1.0          # initial exploration probability
    epsilon_end: float = 0.05           # final exploration probability
    epsilon_decay_episodes: int = 800   # how many episodes to anneal over
    optimistic_init: float = 0.0        # initial Q-value for unseen state-actions

    # State discretization knobs
    pos_buckets: int = 3                # NxN spatial buckets for vehicle position
    load_buckets: int = 3               # buckets for vehicle load fraction


# -----------------------------
# State discretization
# -----------------------------

def _bucket(value: float, n_buckets: int) -> int:
    """Map a value in [0, 1] to an integer bucket in [0, n_buckets-1]."""
    return int(min(value * n_buckets, n_buckets - 1))


def discretize_state(env: FoodRescueEnv, n_pos_buckets: int = 3, n_load_buckets: int = 3) -> tuple:
    """
    Reduce the env state to a compact, hashable tuple.

    Returns
    -------
    tuple (vehicle_xy_bucket, load_bucket, urgent_donor_idx, best_shelter_idx, time_bucket, vehicle_idx)
    """
    v = env.vehicles[env.current_vehicle_idx]
    scn = env.scenario
    gs = env.grid_size

    # Vehicle position bucket (encoded as a single int 0..n_pos_buckets^2 - 1)
    x_bucket = _bucket(v.location[0] / gs, n_pos_buckets)
    y_bucket = _bucket(v.location[1] / gs, n_pos_buckets)
    pos_bucket = x_bucket * n_pos_buckets + y_bucket

    # Load bucket
    load_frac = v.current_load() / v.capacity if v.capacity > 0 else 0.0
    load_bucket = _bucket(load_frac, n_load_buckets)

    # Nearest urgent donor (with food, prioritize low shelf life when close)
    urgent_donor_idx = scn.num_donors  # sentinel: "no donor matters"
    best_donor_score = -1.0
    for i, d in enumerate(scn.donors):
        qty = d.total_pending_quantity()
        if qty <= 0:
            continue
        dist = abs(v.location[0] - d.location[0]) + abs(v.location[1] - d.location[1])
        urgency = max(1, 30 - d.min_pending_shelf_life())  # higher when shelf life low
        score = qty * urgency / (dist + 1)
        if score > best_donor_score:
            best_donor_score = score
            urgent_donor_idx = i

    # Best shelter (highest priority * demand / distance)
    best_shelter_idx = scn.num_shelters
    best_shelter_score = -1.0
    for j, s in enumerate(scn.shelters):
        if s.current_demand <= 0:
            continue
        dist = abs(v.location[0] - s.location[0]) + abs(v.location[1] - s.location[1])
        priority_w = 2.0 if s.priority == 1 else 1.0
        score = priority_w * s.current_demand / (dist + 1)
        if score > best_shelter_score:
            best_shelter_score = score
            best_shelter_idx = j

    # Time bucket: 0=morning, 1=afternoon, 2=evening
    bucket_str = scn.city.time_bucket(env.current_step)
    time_bucket = {"morning": 0, "afternoon": 1, "evening": 2}.get(bucket_str, 0)

    return (
        pos_bucket,
        load_bucket,
        urgent_donor_idx,
        best_shelter_idx,
        time_bucket,
        env.current_vehicle_idx,
    )


# -----------------------------
# Q-learning agent
# -----------------------------

class QLearningAgent(Policy):
    """
    Tabular Q-learning agent compatible with FoodRescueEnv.

    Learning happens via update_from_transition() called by the training loop.
    Action selection happens via select_action() (ε-greedy).
    """

    name = "q_learning"

    def __init__(
        self,
        num_actions: int,
        config: Optional[QLearningConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config if config is not None else QLearningConfig()
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

        # Q-table: (state_tuple, action) -> value
        self._q_table: dict[tuple, np.ndarray] = {}

        # Training mode flag (controls whether update_from_transition does anything)
        self._training = True

        # Episode counter for ε annealing
        self._episode_count = 0

    # ---- ε-greedy action selection ----

    def epsilon(self) -> float:
        """Linearly decay epsilon over training episodes."""
        c = self.config
        if self._episode_count >= c.epsilon_decay_episodes:
            return c.epsilon_end
        progress = self._episode_count / max(c.epsilon_decay_episodes, 1)
        return c.epsilon_start + (c.epsilon_end - c.epsilon_start) * progress

    def select_action(self, env: FoodRescueEnv, obs: np.ndarray) -> int:
        state = discretize_state(env, self.config.pos_buckets, self.config.load_buckets)
        eps = self.epsilon() if self._training else 0.0  # eval mode: pure greedy

        if self.rng.random() < eps:
            return int(self.rng.integers(0, self.num_actions))

        return int(self._argmax_q(state))

    def _argmax_q(self, state: tuple) -> int:
        """Pick the action with the highest Q-value for this state. Random tie-breaking."""
        q_values = self._q_table.get(state)
        if q_values is None:
            # Unseen state: all actions equal-valued, pick randomly
            return int(self.rng.integers(0, self.num_actions))
        max_val = q_values.max()
        # Random tie-breaking among all actions equal to max
        best_actions = np.flatnonzero(q_values == max_val)
        return int(self.rng.choice(best_actions))

    def _ensure_q_row(self, state: tuple) -> np.ndarray:
        """Lazy-init a row in the Q-table for an unseen state."""
        if state not in self._q_table:
            self._q_table[state] = np.full(
                self.num_actions, self.config.optimistic_init, dtype=np.float32
            )
        return self._q_table[state]

    # ---- Q-learning update ----

    def update_from_transition(
        self,
        env_before: FoodRescueEnv,
        action: int,
        reward: float,
        env_after: FoodRescueEnv,
        done: bool,
    ) -> None:
        """
        Apply the Q-learning Bellman update.

        Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))

        Note: the env state passed in BEFORE step() is captured into a discretized
        state tuple. After step(), the env has advanced — we read the new state
        from env_after. The training loop is responsible for calling this with
        consistent before/after envs.
        """
        if not self._training:
            return

        # Discretize before/after states
        state_before = discretize_state(
            env_before, self.config.pos_buckets, self.config.load_buckets
        )
        state_after = discretize_state(
            env_after, self.config.pos_buckets, self.config.load_buckets
        )

        q_row = self._ensure_q_row(state_before)
        old_q = q_row[action]

        if done:
            target = reward
        else:
            q_next = self._q_table.get(state_after)
            max_next = float(q_next.max()) if q_next is not None else self.config.optimistic_init
            target = reward + self.config.discount * max_next

        q_row[action] = old_q + self.config.learning_rate * (target - old_q)

    # ---- Training control ----

    def end_episode(self) -> None:
        """Increment episode counter for ε annealing."""
        self._episode_count += 1

    def set_training(self, training: bool) -> None:
        self._training = training

    def reset(self) -> None:
        """Per-episode reset hook from Policy ABC. No-op for tabular Q."""

    # ---- Save / load ----

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Convert to plain dict of dicts for portable pickling
        state = {
            "config": self.config,
            "num_actions": self.num_actions,
            "q_table": {k: v.tolist() for k, v in self._q_table.items()},
            "episode_count": self._episode_count,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path, seed: Optional[int] = None) -> "QLearningAgent":
        with open(path, "rb") as f:
            state = pickle.load(f)
        agent = cls(num_actions=state["num_actions"], config=state["config"], seed=seed)
        agent._q_table = {
            k: np.array(v, dtype=np.float32) for k, v in state["q_table"].items()
        }
        agent._episode_count = state["episode_count"]
        agent.set_training(False)  # loaded policies are eval-only by default
        return agent

    # ---- Diagnostics ----

    def table_size(self) -> int:
        return len(self._q_table)

    def stats(self) -> dict:
        if not self._q_table:
            return {"table_size": 0, "mean_q": 0.0, "max_q": 0.0}
        all_q = np.concatenate([row for row in self._q_table.values()])
        return {
            "table_size": len(self._q_table),
            "episode_count": self._episode_count,
            "epsilon_current": self.epsilon(),
            "mean_q": float(all_q.mean()),
            "max_q": float(all_q.max()),
            "min_q": float(all_q.min()),
        }
