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

    Design: give the agent enough signal to decide WHERE to go without baking
    in a heuristic. We encode counts (not bitmasks) so the state space stays
    tractable for tabular methods.

    Tuple layout:
    - pos_bucket: which 3x3 region the vehicle is in (9 values)
    - load_bucket: empty / partial / full (3 values)
    - num_donors_with_food: bucketed 0/1/2-3/4+ (4 values)
    - max_donor_qty_bucket: how much food at the most-loaded donor (4 values)
    - num_shelters_with_demand: bucketed 0/1/2-3/4+ (4 values)
    - max_shelter_demand_bucket: how urgent at the most-needy shelter (4 values)
    - urgency_flag: any pending batch with shelf_life <= 15? (2 values)
    - time_bucket: morning/afternoon/evening (3 values)
    - vehicle_idx: which vehicle is acting (2 values)

    State space: 9 * 3 * 4 * 4 * 4 * 4 * 2 * 3 * 2 = ~41,500
    Roughly 10x smaller than the bitmask version, which means the agent can
    actually populate enough cells with meaningful Q-values.
    """
    v = env.vehicles[env.current_vehicle_idx]
    scn = env.scenario
    gs = env.grid_size

    # Vehicle position bucket
    x_bucket = _bucket(v.location[0] / gs, n_pos_buckets)
    y_bucket = _bucket(v.location[1] / gs, n_pos_buckets)
    pos_bucket = x_bucket * n_pos_buckets + y_bucket

    # Vehicle load bucket
    load_frac = v.current_load() / v.capacity if v.capacity > 0 else 0.0
    load_bucket = _bucket(load_frac, n_load_buckets)

    # Donor counts and max
    num_donors_with_food = 0
    max_donor_qty = 0.0
    urgency_flag = 0
    for d in scn.donors:
        qty = d.total_pending_quantity()
        if qty > 0:
            num_donors_with_food += 1
            if qty > max_donor_qty:
                max_donor_qty = qty
            if d.min_pending_shelf_life() <= 15:
                urgency_flag = 1

    # Bucket donor count: 0, 1, 2-3, 4+
    if num_donors_with_food == 0:
        donor_count_bucket = 0
    elif num_donors_with_food == 1:
        donor_count_bucket = 1
    elif num_donors_with_food <= 3:
        donor_count_bucket = 2
    else:
        donor_count_bucket = 3

    # Bucket max donor quantity: 0, light, medium, heavy
    # Use vehicle capacity as the reference scale
    cap = v.capacity if v.capacity > 0 else 20.0
    if max_donor_qty == 0:
        donor_qty_bucket = 0
    elif max_donor_qty < cap * 0.3:
        donor_qty_bucket = 1
    elif max_donor_qty < cap * 0.7:
        donor_qty_bucket = 2
    else:
        donor_qty_bucket = 3

    # Shelter counts and max
    num_shelters_with_demand = 0
    max_shelter_demand = 0.0
    for s in scn.shelters:
        if s.current_demand >= 5.0:  # threshold to filter noise
            num_shelters_with_demand += 1
            if s.current_demand > max_shelter_demand:
                max_shelter_demand = s.current_demand

    if num_shelters_with_demand == 0:
        shelter_count_bucket = 0
    elif num_shelters_with_demand == 1:
        shelter_count_bucket = 1
    elif num_shelters_with_demand <= 3:
        shelter_count_bucket = 2
    else:
        shelter_count_bucket = 3

    # Bucket max shelter demand
    if max_shelter_demand == 0:
        shelter_demand_bucket = 0
    elif max_shelter_demand < 30:
        shelter_demand_bucket = 1
    elif max_shelter_demand < 60:
        shelter_demand_bucket = 2
    else:
        shelter_demand_bucket = 3

    # Time bucket
    bucket_str = scn.city.time_bucket(env.current_step)
    time_bucket = {"morning": 0, "afternoon": 1, "evening": 2}.get(bucket_str, 0)

    return (
        pos_bucket,
        load_bucket,
        donor_count_bucket,
        donor_qty_bucket,
        shelter_count_bucket,
        shelter_demand_bucket,
        urgency_flag,
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
