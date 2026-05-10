"""
FoodRescueEnv: Gymnasium-compatible environment for food rescue RL.

Episode flow
------------
1. reset() loads a fresh scenario, spawns vehicles, resets all state.
2. Each step():
   - Decode the action: head_to_donor_i / head_to_shelter_j / idle
   - Apply to the current vehicle (round-robin selection)
   - Advance the world: vehicle moves, donors generate batches, batches age,
     shelters grow demand, deliveries happen on arrival
   - Compute reward, build observation, check termination

Observation space
-----------------
A flat Box of floats representing:
  [vehicle_x, vehicle_y, vehicle_load_pct, vehicle_idle_flag,
   for each donor: (qty_pending, min_shelf_life, distance_from_vehicle),
   for each shelter: (current_demand_pct, distance_from_vehicle),
   normalized_time, current_vehicle_idx]

Action space
------------
Discrete: head_to_donor_0 ... head_to_donor_{N-1},
          head_to_shelter_0 ... head_to_shelter_{M-1},
          idle/wait
Total: N + M + 1 actions.

Reward
------
Dense per-step reward combining four objectives:
  + alpha * food_delivered_this_step      (delivery)
  - beta  * food_spoiled_this_step        (anti-spoilage)
  - gamma * distance_traveled_this_step   (transport cost / emissions)
  - delta * unmet_demand_this_step        (equity / shelter coverage)

Reward weights (alpha, beta, gamma, delta) come from the env config and become
hyperparameters that we sweep in Sprint 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim.city import ScenarioLoader, make_vehicles
from sim.entities import BatchStatus, FoodBatch


# -----------------------------
# Config
# -----------------------------

@dataclass
class RewardWeights:
    """Per-event reward magnitudes. Tunable, swept in Sprint 6."""
    delivery: float = 10.0
    spoilage: float = 5.0
    distance: float = 0.1
    unmet_demand: float = 1.0
    priority_bonus: float = 0.5  # extra reward for delivering to priority-1 shelters
    oversupply_penalty: float = 0.3  # small penalty for delivering more than needed


@dataclass
class EnvConfig:
    """Top-level env configuration."""
    scenario_name: str = "weekday"
    vehicle_start_strategy: str = "center"
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    max_episode_steps: Optional[int] = None  # None means use scenario's episode_length
    seed: Optional[int] = None  # None means use scenario's random_seed


# -----------------------------
# Environment
# -----------------------------

class FoodRescueEnv(gym.Env):
    """
    Gymnasium env for food rescue dispatch.

    A single agent controls a fleet of vehicles in round-robin fashion: at each
    step, exactly one vehicle (selected by current_vehicle_idx) receives the
    action. The env then advances world state by one timestep.

    Use case:
        env = FoodRescueEnv()  # uses defaults (weekday scenario)
        obs, info = env.reset(seed=42)
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.config = config if config is not None else EnvConfig()
        self.render_mode = render_mode

        # Load scenario template — this gives us shapes for the spaces.
        # Each reset() reloads to get fresh entity instances.
        self._loader = ScenarioLoader()
        scenario_template = self._loader.load(self.config.scenario_name)
        self.num_donors = scenario_template.num_donors
        self.num_shelters = scenario_template.num_shelters
        self.num_vehicles = scenario_template.num_vehicles
        self.grid_size = scenario_template.city.grid_size

        self.max_episode_steps = (
            self.config.max_episode_steps
            if self.config.max_episode_steps is not None
            else scenario_template.city.episode_length
        )

        # Define spaces
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

        # Per-episode state. Real values populated in reset().
        self.scenario = None
        self.vehicles: list = []
        self.batches: list[FoodBatch] = []
        self.current_step: int = 0
        self.current_vehicle_idx: int = 0
        self.next_batch_id: int = 0
        self.rng: Optional[np.random.Generator] = None
        self._episode_metrics: dict = {}
        self._last_step_info: dict = {}  # raw per-step counters used for reward + info

    # -----------------------------
    # Space builders
    # -----------------------------

    def _build_action_space(self) -> spaces.Discrete:
        """N donors + M shelters + 1 idle = N+M+1 discrete actions."""
        return spaces.Discrete(self.num_donors + self.num_shelters + 1)

    def _build_observation_space(self) -> spaces.Box:
        """
        Observation vector layout (all floats):

          [0] vehicle_x                       in [0, 1]   (normalized by grid_size)
          [1] vehicle_y                       in [0, 1]
          [2] vehicle_load_pct                in [0, 1]
          [3] vehicle_idle_flag               in {0, 1}
          For each donor i:
            [_] qty_pending_normalized        in [0, ~5]  (capped soft, can spike)
            [_] min_shelf_life_normalized     in [0, 1]   (1 = fresh, 0 = expiring)
            [_] distance_from_vehicle_norm    in [0, 1]   (normalized by 2*grid_size)
          For each shelter j:
            [_] current_demand_pct            in [0, 1]
            [_] distance_from_vehicle_norm    in [0, 1]
          [_] normalized_time                 in [0, 1]
          [_] current_vehicle_idx_normalized  in [0, 1]
        """
        n_features = 4 + 3 * self.num_donors + 2 * self.num_shelters + 2
        return spaces.Box(
            low=0.0,
            high=10.0,  # generous upper bound; most features are [0, 1]
            shape=(n_features,),
            dtype=np.float32,
        )

    # -----------------------------
    # Action decoding
    # -----------------------------

    def _decode_action(self, action: int) -> tuple[str, Optional[int]]:
        """
        Decode the integer action into a (kind, index) tuple.

        Returns
        -------
        kind : str
            "donor", "shelter", or "idle"
        index : int | None
            Index into self.scenario.donors or self.scenario.shelters, or None for idle.
        """
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"Action {action} out of range [0, {self.action_space.n})")

        if action < self.num_donors:
            return "donor", action
        if action < self.num_donors + self.num_shelters:
            return "shelter", action - self.num_donors
        return "idle", None

    # -----------------------------
    # Observation
    # -----------------------------

    def _get_observation(self) -> np.ndarray:
        """Build the observation vector for the *current* vehicle."""
        v = self.vehicles[self.current_vehicle_idx]
        scn = self.scenario
        gs = self.grid_size

        obs = []

        # Vehicle features
        obs.append(v.location[0] / gs)
        obs.append(v.location[1] / gs)
        obs.append(v.current_load() / v.capacity if v.capacity > 0 else 0.0)
        obs.append(1.0 if v.is_idle() else 0.0)

        # Per-donor features
        for d in scn.donors:
            qty = d.total_pending_quantity()
            qty_norm = min(qty / max(d.avg_quantity, 1.0), 10.0)  # cap at 10x avg
            min_sl = d.min_pending_shelf_life()
            sl_norm = min(min_sl / d.shelf_life_max, 1.0) if d.shelf_life_max > 0 else 1.0
            dist = self._manhattan(v.location, d.location)
            dist_norm = dist / (2 * gs)
            obs.extend([qty_norm, sl_norm, dist_norm])

        # Per-shelter features
        for s in scn.shelters:
            obs.append(s.utilization())  # already in [0, 1]
            dist = self._manhattan(v.location, s.location)
            obs.append(dist / (2 * gs))

        # Time + which vehicle
        obs.append(self.current_step / self.max_episode_steps)
        obs.append(
            self.current_vehicle_idx / max(self.num_vehicles - 1, 1)
            if self.num_vehicles > 1 else 0.0
        )

        return np.array(obs, dtype=np.float32)

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # -----------------------------
    # Reset
    # -----------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Start a new episode.

        Loads a fresh scenario (new Donor/Shelter/Vehicle instances), seeds the
        RNG, and zeroes all metrics. Returns initial observation.
        """
        super().reset(seed=seed)

        # Choose seed: explicit > config > scenario default
        if seed is None:
            seed = self.config.seed
        if seed is None:
            scenario_template = self._loader.load(self.config.scenario_name)
            seed = scenario_template.random_seed

        self.rng = np.random.default_rng(seed)

        # Fresh scenario, fresh vehicles
        self.scenario = self._loader.load(self.config.scenario_name)
        self.vehicles = make_vehicles(
            self.scenario, start_strategy=self.config.vehicle_start_strategy
        )

        self.batches = []
        self.next_batch_id = 0
        self.current_step = 0
        self.current_vehicle_idx = 0

        self._episode_metrics = {
            "total_generated": 0,
            "total_delivered_units": 0.0,
            "total_spoiled_units": 0.0,
            "total_wasted_units": 0.0,
            "total_distance": 0,
            "total_unmet_demand_steps": 0.0,
            "deliveries_count": 0,
            "priority_deliveries_count": 0,
        }
        self._last_step_info = {}

        obs = self._get_observation()
        info = {
            "scenario": self.scenario.name,
            "current_vehicle_idx": self.current_vehicle_idx,
            "step": self.current_step,
        }
        return obs, info

   # -----------------------------
    # Step
    # -----------------------------

    def step(self, action: int):
        """
        Apply an action and advance the world by one timestep.

        Returns
        -------
        observation : np.ndarray
        reward : float
        terminated : bool
            True if episode reached a natural end (we don't have one — always False).
        truncated : bool
            True if episode hit the timestep cap (this is what ends our episodes).
        info : dict
            Per-step metrics for logging.
        """
        if self.scenario is None:
            raise RuntimeError("Must call reset() before step()")

        # Reset per-step counters
        step_metrics = {
            "delivered_units": 0.0,
            "wasted_units": 0.0,
            "spoiled_units_donor": 0.0,
            "spoiled_units_vehicle": 0.0,
            "distance_traveled": 0,
            "deliveries_count": 0,
            "priority_deliveries_count": 0,
            "batches_generated": 0,
            "action_kind": None,
            "action_target_id": None,
            "vehicle_idx": self.current_vehicle_idx,
        }

        v = self.vehicles[self.current_vehicle_idx]

        # 1. Decode and apply the action
        kind, idx = self._decode_action(int(action))
        step_metrics["action_kind"] = kind

        if kind == "donor":
            target = self.scenario.donors[idx]
            v.set_target(target.location, "donor", target.donor_id)
            step_metrics["action_target_id"] = target.donor_id
        elif kind == "shelter":
            target = self.scenario.shelters[idx]
            v.set_target(target.location, "shelter", target.shelter_id)
            step_metrics["action_target_id"] = target.shelter_id
        else:  # idle
            v.clear_target()
            step_metrics["action_target_id"] = None

        # 2. Move the vehicle one cell toward its target
        moved = v.move_one_step()
        step_metrics["distance_traveled"] = moved

        # 3. Process arrival (pickup or deliver)
        if v.at_target():
            if v.target_kind == "donor":
                donor = self._find_donor_by_id(v.target_id)
                if donor is not None:
                    picked = donor.pickup_all(self.current_step)
                    leftover = v.load_batches(picked)
                    # Leftovers go back to donor (vehicle was full)
                    donor.pending_batches.extend(leftover)
                v.clear_target()

            elif v.target_kind == "shelter":
                shelter = self._find_shelter_by_id(v.target_id)
                if shelter is not None and v.current_load() > 0:
                    absorbed, wasted, n_batches = v.deliver_to_shelter(
                        shelter, self.current_step
                    )
                    step_metrics["delivered_units"] = absorbed
                    step_metrics["wasted_units"] = wasted
                    step_metrics["deliveries_count"] = n_batches
                    if shelter.priority == 1:
                        step_metrics["priority_deliveries_count"] = n_batches
                v.clear_target()

        # 4. Donors generate new batches for this timestep
        donor_mult = self.scenario.city.donor_rate_multiplier(self.current_step)
        for donor in self.scenario.donors:
            new_batch = donor.maybe_generate_batch(
                self.current_step, self.next_batch_id, donor_mult, self.rng
            )
            if new_batch is not None:
                self.batches.append(new_batch)
                self.next_batch_id += 1
                step_metrics["batches_generated"] += 1

        # 5. Age all pending batches at donors (spoilage)
        for donor in self.scenario.donors:
            _, spoiled_qty = donor.tick_pending_batches()
            step_metrics["spoiled_units_donor"] += spoiled_qty

        # 6. Age batches in vehicle cargo (yes, food spoils mid-trip too)
        for vehicle in self.vehicles:
            self._tick_vehicle_cargo(vehicle, step_metrics)

        # 7. Shelters grow demand
        shelter_mult = self.scenario.city.shelter_rate_multiplier(self.current_step)
        for shelter in self.scenario.shelters:
            shelter.tick(shelter_mult, self.rng)

        # 8. Compute reward
        reward = self._compute_reward(step_metrics)

        # 9. Update episode-level metrics
        self._update_episode_metrics(step_metrics)
        self._last_step_info = step_metrics

        # 10. Advance vehicle round-robin and timestep
        self.current_vehicle_idx = (self.current_vehicle_idx + 1) % self.num_vehicles
        self.current_step += 1

        # 11. Check termination
        truncated = self.current_step >= self.max_episode_steps
        terminated = False  # no natural terminal state in our problem

        # 12. Build next observation (now that current_vehicle_idx has advanced)
        obs = self._get_observation()

        # Build info dict
        info = {
            **step_metrics,
            "step": self.current_step,
            "current_vehicle_idx": self.current_vehicle_idx,
            "episode_metrics": dict(self._episode_metrics) if (terminated or truncated) else None,
        }

        return obs, reward, terminated, truncated, info

    # -----------------------------
    # Helpers used by step
    # -----------------------------

    def _find_donor_by_id(self, donor_id: str):
        for d in self.scenario.donors:
            if d.donor_id == donor_id:
                return d
        return None

    def _find_shelter_by_id(self, shelter_id: str):
        for s in self.scenario.shelters:
            if s.shelter_id == shelter_id:
                return s
        return None



    def _tick_vehicle_cargo(self, vehicle, step_metrics: dict) -> None:
        """Age batches inside a vehicle's cargo. Spoiled ones are removed."""
        surviving = []
        for batch in vehicle.cargo:
            batch.tick()
            if batch.status == BatchStatus.SPOILED:
                step_metrics["spoiled_units_vehicle"] += batch.quantity
            else:
                surviving.append(batch)
        vehicle.cargo = surviving

    def _compute_reward(self, m: dict) -> float:
        """Combine per-step metrics into a scalar reward."""
        w = self.config.reward_weights

        delivered = m["delivered_units"]
        wasted = m["wasted_units"]
        spoiled = m["spoiled_units_donor"] + m["spoiled_units_vehicle"]
        distance = m["distance_traveled"]
        priority = m["priority_deliveries_count"]

        # Sum unmet demand across all shelters as a soft penalty signal each step
        unmet = sum(s.current_demand for s in self.scenario.shelters)

        reward = (
            w.delivery * delivered
            + w.priority_bonus * priority * delivered  # bonus scaled by units delivered
            - w.oversupply_penalty * wasted
            - w.spoilage * spoiled
            - w.distance * distance
            - w.unmet_demand * (unmet / 100.0)  # divide to keep reward magnitudes balanced
        )
        return float(reward)

    def _update_episode_metrics(self, m: dict) -> None:
        em = self._episode_metrics
        em["total_generated"] += m["batches_generated"]
        em["total_delivered_units"] += m["delivered_units"]
        em["total_spoiled_units"] += m["spoiled_units_donor"] + m["spoiled_units_vehicle"]
        em["total_wasted_units"] += m["wasted_units"]
        em["total_distance"] += m["distance_traveled"]
        em["deliveries_count"] += m["deliveries_count"]
        em["priority_deliveries_count"] += m["priority_deliveries_count"]
        em["total_unmet_demand_steps"] += sum(
            s.current_demand for s in self.scenario.shelters
        )

    # -----------------------------
    # Render (placeholder — full implementation in Sprint 3)
    # -----------------------------

    def render(self):
        """Render the env. Implemented in Sprint 3."""
        if self.render_mode is None:
            return None
        raise NotImplementedError("Rendering is implemented in Sprint 3.")

    def close(self):
        pass
