"""
City layer: grid, time-of-day, and scenario loading.

Responsibilities:
- City: holds grid dimensions, validates coordinates, looks up time-of-day modifiers
- ScenarioLoader: reads data/processed/<scenario>/ and instantiates Donors and Shelters
- Helpers: distance matrix lookups, time-bucket math

The Gymnasium environment uses these to bootstrap each episode and to query
domain-specific info (rate multipliers per timestep, distances) during step().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sim.entities import Donor, Shelter


PROCESSED_DIR = Path("data/processed")


# -----------------------------
# City
# -----------------------------

@dataclass
class City:
    """
    The simulated city.

    Holds grid dimensions, time-of-day modifiers, and provides utility methods
    for coordinate validity checks and time bucket lookups.
    """
    grid_size: int
    episode_length: int
    time_modifiers: dict  # nested dict from scenario_config.yaml
    name: str = "city"

    def is_valid_position(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def time_bucket(self, step: int) -> str:
        """
        Map a timestep to a time-of-day bucket label.

        Buckets are equal thirds of the episode:
          [0, L/3)            -> morning
          [L/3, 2L/3)         -> afternoon
          [2L/3, L]           -> evening
        """
        third = self.episode_length / 3
        if step < third:
            return "morning"
        if step < 2 * third:
            return "afternoon"
        return "evening"

    def donor_rate_multiplier(self, step: int) -> float:
        bucket = self.time_bucket(step)
        return float(self.time_modifiers[bucket]["donor_arrival_mult"])

    def shelter_rate_multiplier(self, step: int) -> float:
        bucket = self.time_bucket(step)
        return float(self.time_modifiers[bucket]["shelter_demand_mult"])


# -----------------------------
# Scenario
# -----------------------------

@dataclass
class Scenario:
    """
    A loaded scenario, ready to seed an environment.

    Holds the city, the entity templates (donor/shelter prototypes built from CSVs),
    the precomputed distance matrices, and the random seed.
    """
    name: str
    config: dict
    city: City
    donors: list[Donor]
    shelters: list[Shelter]
    distance_matrix: np.ndarray              # (num_donors, num_shelters)
    donor_distance_matrix: np.ndarray        # (num_donors, num_donors)
    summary_stats: dict = field(default_factory=dict)

    @property
    def num_donors(self) -> int:
        return len(self.donors)

    @property
    def num_shelters(self) -> int:
        return len(self.shelters)

    @property
    def num_vehicles(self) -> int:
        return int(self.config["num_vehicles"])

    @property
    def vehicle_capacity(self) -> float:
        return float(self.config["vehicle_capacity"])

    @property
    def random_seed(self) -> int:
        return int(self.config["random_seed"])

    def donor_shelter_distance(self, donor_idx: int, shelter_idx: int) -> int:
        return int(self.distance_matrix[donor_idx, shelter_idx])

    def donor_donor_distance(self, i: int, j: int) -> int:
        return int(self.donor_distance_matrix[i, j])


# -----------------------------
# ScenarioLoader
# -----------------------------

class ScenarioLoader:
    """
    Loads a processed scenario into ready-to-use Scenario objects.

    Each call to `load()` returns a *fresh* Scenario with new Donor and Shelter
    instances, so calling it again gives you clean state for a new episode.

    Usage:
        loader = ScenarioLoader()
        scenario = loader.load("weekday")
    """

    def __init__(self, processed_dir: Path | str = PROCESSED_DIR):
        self.processed_dir = Path(processed_dir)
        if not self.processed_dir.exists():
            raise FileNotFoundError(
                f"Processed data dir not found: {self.processed_dir}. "
                f"Run `python data_prep.py --scenario all` first."
            )

    def available_scenarios(self) -> list[str]:
        """List all scenarios that have been processed."""
        return sorted([p.name for p in self.processed_dir.iterdir() if p.is_dir()])

    def load(self, scenario_name: str) -> Scenario:
        """Load a scenario by name. Returns a fresh Scenario each call."""
        scn_dir = self.processed_dir / scenario_name
        if not scn_dir.exists():
            raise FileNotFoundError(
                f"Scenario '{scenario_name}' not found. "
                f"Available: {self.available_scenarios()}"
            )

        # Load config
        with open(scn_dir / "scenario_config.yaml") as f:
            config = yaml.safe_load(f)

        # Load CSVs
        donors_df = pd.read_csv(scn_dir / "donors_processed.csv")
        shelters_df = pd.read_csv(scn_dir / "shelters_processed.csv")

        # Load distance matrices
        distance_matrix = np.load(scn_dir / "distance_matrix.npy")
        donor_distance_matrix = np.load(scn_dir / "donor_distance_matrix.npy")

        # Optional summary stats
        summary_stats = {}
        stats_path = scn_dir / "summary_stats.json"
        if stats_path.exists():
            import json
            with open(stats_path) as f:
                summary_stats = json.load(f)

        # Build City
        city = City(
            grid_size=config["grid_size"],
            episode_length=config["episode_length"],
            time_modifiers=config["time_modifiers"],
            name=scenario_name,
        )

        # Build Donors from rows
        donors = [
            Donor(
                donor_id=row["donor_id"],
                name=row["name"],
                type=row["type"],
                location=(int(row["x"]), int(row["y"])),
                arrival_rate=float(row["arrival_rate"]),
                avg_quantity=float(row["avg_quantity"]),
                shelf_life_min=int(row["shelf_life_min"]),
                shelf_life_max=int(row["shelf_life_max"]),
            )
            for _, row in donors_df.iterrows()
        ]

        # Build Shelters from rows
        shelters = [
            Shelter(
                shelter_id=row["shelter_id"],
                name=row["name"],
                type=row["type"],
                location=(int(row["x"]), int(row["y"])),
                demand_rate=float(row["demand_rate"]),
                capacity=float(row["capacity"]),
                priority=int(row["priority"]),
            )
            for _, row in shelters_df.iterrows()
        ]

        return Scenario(
            name=scenario_name,
            config=config,
            city=city,
            donors=donors,
            shelters=shelters,
            distance_matrix=distance_matrix,
            donor_distance_matrix=donor_distance_matrix,
            summary_stats=summary_stats,
        )


# -----------------------------
# Vehicle factory
# -----------------------------

def make_vehicles(scenario: Scenario, start_strategy: str = "center") -> list:
    """
    Spawn a fleet of fresh Vehicles for an episode.

    Parameters
    ----------
    scenario : Scenario
        The loaded scenario, gives us count and capacity.
    start_strategy : str
        Where to place vehicles initially:
        - "center": all vehicles start at the city center (simplest, default)
        - "spread": vehicles distributed along a diagonal
        - "near_donors": each vehicle starts near a donor (round-robin)

    Returns
    -------
    list[Vehicle]
        A fresh list of Vehicle objects, ready for the episode.
    """
    from sim.entities import Vehicle

    n = scenario.num_vehicles
    cap = scenario.vehicle_capacity
    grid = scenario.city.grid_size

    if start_strategy == "center":
        center = (grid // 2, grid // 2)
        return [
            Vehicle(vehicle_id=i, location=center, capacity=cap)
            for i in range(n)
        ]

    if start_strategy == "spread":
        # Spread along the main diagonal
        positions = [
            (int(i * (grid - 1) / max(n - 1, 1)), int(i * (grid - 1) / max(n - 1, 1)))
            for i in range(n)
        ]
        return [
            Vehicle(vehicle_id=i, location=positions[i], capacity=cap)
            for i in range(n)
        ]

    if start_strategy == "near_donors":
        # Each vehicle starts at a donor (cycling if we have more vehicles than donors)
        return [
            Vehicle(
                vehicle_id=i,
                location=scenario.donors[i % len(scenario.donors)].location,
                capacity=cap,
            )
            for i in range(n)
        ]

    raise ValueError(
        f"Unknown start_strategy: {start_strategy!r}. "
        f"Expected one of: 'center', 'spread', 'near_donors'."
    )
