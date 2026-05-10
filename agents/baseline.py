"""
Non-learning baseline policies for the food rescue environment.

Two baselines:

1. RandomPolicy — picks actions uniformly at random. Sanity floor.
2. GreedyPolicy — one-step lookahead heuristic:
   - If carrying food, go to the shelter that maximizes priority * demand / distance
   - Otherwise, go to the donor that maximizes (pending_qty / distance) with
     an urgency boost when shelf life is low
   - If no useful target exists, idle

These baselines define the comparison floor for RL agents in the May 16
final evaluation (rubric: "fixed-timer vs RL policy" comparison).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from sim.environment import FoodRescueEnv


class Policy(ABC):
    """Abstract base for any policy. Provides a stable interface for evaluate.py."""

    name: str = "abstract"

    @abstractmethod
    def select_action(self, env: FoodRescueEnv, obs: np.ndarray) -> int:
        """Choose an action given the current env state. Returns an int action."""

    def reset(self) -> None:
        """Optional per-episode reset hook for stateful policies."""


# -----------------------------
# RandomPolicy
# -----------------------------

class RandomPolicy(Policy):
    """Uniform random action selection. Floor baseline."""

    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def select_action(self, env: FoodRescueEnv, obs: np.ndarray) -> int:
        return int(self.rng.integers(0, env.action_space.n))


# -----------------------------
# GreedyPolicy
# -----------------------------

class GreedyPolicy(Policy):
    """
    One-step lookahead heuristic.

    Decision logic per active vehicle:
    1. If currently carrying food (load > 0):
         - Score each shelter by priority_weight * current_demand / (distance + 1)
         - Go to the highest-scoring shelter (must have demand > 0)
         - If no shelter has demand, idle (avoid pointless oversupply)

    2. Otherwise (empty vehicle):
         - Score each donor by pending_qty * urgency / (distance + 1)
           where urgency is high when min_shelf_life is low
         - Go to the highest-scoring donor (must have pending_qty > 0)
         - If no donor has pending food, idle (or pre-position to a likely
           one — kept simple here)

    Tie-breaking: lower index wins, no randomness. This makes it a fully
    deterministic baseline.

    Parameters
    ----------
    priority_weight_high : float
        Multiplier applied to priority-1 shelters. >1 means we strongly prefer them.
    urgency_threshold : int
        Below this remaining shelf life, we boost the score to favor near-spoilage food.
    urgency_boost : float
        Multiplier applied when shelf life is below threshold.
    """

    name = "greedy"

    def __init__(
        self,
        priority_weight_high: float = 2.0,
        priority_weight_low: float = 1.0,
        urgency_threshold: int = 15,
        urgency_boost: float = 3.0,
    ):
        self.priority_weight_high = priority_weight_high
        self.priority_weight_low = priority_weight_low
        self.urgency_threshold = urgency_threshold
        self.urgency_boost = urgency_boost

    def select_action(self, env: FoodRescueEnv, obs: np.ndarray) -> int:
        """Pick best target for the currently acting vehicle."""
        v = env.vehicles[env.current_vehicle_idx]
        scn = env.scenario

        idle_action = env.num_donors + env.num_shelters

        if v.current_load() > 0:
            # We have cargo, find best shelter
            best_action = self._best_shelter(env, v)
            if best_action is None:
                # No demand anywhere. Idle (would deliver, but no shelter wants it).
                return idle_action
            return best_action

        # Empty vehicle, find best donor with pending food
        best_action = self._best_donor(env, v)
        if best_action is None:
            return idle_action
        return best_action

    def _best_shelter(self, env: FoodRescueEnv, v) -> int | None:
        scn = env.scenario
        # Other vehicles' shelter targets, to avoid piling up
        other_shelter_targets = {
            other.target_id for other in env.vehicles
            if other is not v and other.target_kind == "shelter"
        }
        best_score = -1.0
        best_idx = None

        for j, shelter in enumerate(scn.shelters):
            if shelter.current_demand <= 0:
                continue
            if shelter.shelter_id in other_shelter_targets:
                continue  # someone else is on it
            distance = abs(v.location[0] - shelter.location[0]) + abs(
                v.location[1] - shelter.location[1]
            )
            priority_w = (
                self.priority_weight_high
                if shelter.priority == 1
                else self.priority_weight_low
            )
            score = priority_w * shelter.current_demand / (distance + 1)
            if score > best_score:
                best_score = score
                best_idx = j

        if best_idx is None:
            return None
        return env.num_donors + best_idx

    def _best_donor(self, env: FoodRescueEnv, v) -> int | None:
        scn = env.scenario
        other_donor_targets = {
            other.target_id for other in env.vehicles
            if other is not v and other.target_kind == "donor"
        }
        best_score = -1.0
        best_idx = None

        for i, donor in enumerate(scn.donors):
            qty = donor.total_pending_quantity()
            if qty <= 0:
                continue
            if donor.donor_id in other_donor_targets:
                continue
            distance = abs(v.location[0] - donor.location[0]) + abs(
                v.location[1] - donor.location[1]
            )
            min_shelf_life = donor.min_pending_shelf_life()
            urgency = (
                self.urgency_boost
                if min_shelf_life <= self.urgency_threshold
                else 1.0
            )
            score = urgency * qty / (distance + 1)
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

 
