"""
Data preparation pipeline for food rescue scenarios.

Reads raw scenario CSVs from data/raw/<scenario>/, validates them,
computes derived features (distance matrix, normalized rates, etc.),
and writes processed artifacts to data/processed/<scenario>/.

Usage:
    python data_prep.py --scenario weekday
    python data_prep.py --scenario all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

REQUIRED_DONOR_COLS = [
    "donor_id", "name", "type", "x", "y",
    "arrival_rate", "avg_quantity", "shelf_life_min", "shelf_life_max",
]
REQUIRED_SHELTER_COLS = [
    "shelter_id", "name", "type", "x", "y",
    "demand_rate", "capacity", "priority",
]


class DataValidationError(Exception):
    """Raised when scenario data fails validation."""


def validate_donors(df: pd.DataFrame, grid_size: int) -> None:
    """Validate donor CSV: schema, ranges, uniqueness."""
    missing = set(REQUIRED_DONOR_COLS) - set(df.columns)
    if missing:
        raise DataValidationError(f"donors.csv missing columns: {missing}")

    if df["donor_id"].duplicated().any():
        raise DataValidationError("Duplicate donor_id values found")

    if (df["x"] < 0).any() or (df["x"] >= grid_size).any():
        raise DataValidationError(f"donor x coordinates out of range [0, {grid_size})")
    if (df["y"] < 0).any() or (df["y"] >= grid_size).any():
        raise DataValidationError(f"donor y coordinates out of range [0, {grid_size})")

    if (df["arrival_rate"] <= 0).any() or (df["arrival_rate"] > 1).any():
        raise DataValidationError("arrival_rate must be in (0, 1]")
    if (df["avg_quantity"] <= 0).any():
        raise DataValidationError("avg_quantity must be positive")
    if (df["shelf_life_min"] >= df["shelf_life_max"]).any():
        raise DataValidationError("shelf_life_min must be < shelf_life_max")
    if (df["shelf_life_min"] <= 0).any():
        raise DataValidationError("shelf_life_min must be positive")


def validate_shelters(df: pd.DataFrame, grid_size: int) -> None:
    """Validate shelter CSV."""
    missing = set(REQUIRED_SHELTER_COLS) - set(df.columns)
    if missing:
        raise DataValidationError(f"shelters.csv missing columns: {missing}")

    if df["shelter_id"].duplicated().any():
        raise DataValidationError("Duplicate shelter_id values found")

    if (df["x"] < 0).any() or (df["x"] >= grid_size).any():
        raise DataValidationError(f"shelter x coordinates out of range [0, {grid_size})")
    if (df["y"] < 0).any() or (df["y"] >= grid_size).any():
        raise DataValidationError(f"shelter y coordinates out of range [0, {grid_size})")

    if (df["demand_rate"] <= 0).any():
        raise DataValidationError("demand_rate must be positive")
    if (df["capacity"] <= 0).any():
        raise DataValidationError("capacity must be positive")
    if not df["priority"].isin([1, 2]).all():
        raise DataValidationError("priority must be 1 or 2")


def compute_distance_matrix(donors: pd.DataFrame, shelters: pd.DataFrame) -> np.ndarray:
    """
    Manhattan distance matrix between every donor and shelter.
    Shape: (num_donors, num_shelters).
    """
    donor_coords = donors[["x", "y"]].to_numpy()
    shelter_coords = shelters[["x", "y"]].to_numpy()
    diffs = np.abs(donor_coords[:, None, :] - shelter_coords[None, :, :])
    return diffs.sum(axis=-1)


def compute_donor_donor_distances(donors: pd.DataFrame) -> np.ndarray:
    """Donor-to-donor Manhattan distances (for vehicle routing)."""
    coords = donors[["x", "y"]].to_numpy()
    diffs = np.abs(coords[:, None, :] - coords[None, :, :])
    return diffs.sum(axis=-1)


def normalize_rates(donors: pd.DataFrame, shelters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add normalized rate columns. Helps RL by keeping inputs in [0, 1] range.
    """
    donors = donors.copy()
    shelters = shelters.copy()

    donors["arrival_rate_norm"] = donors["arrival_rate"] / donors["arrival_rate"].max()
    donors["avg_quantity_norm"] = donors["avg_quantity"] / donors["avg_quantity"].max()
    donors["shelf_life_avg"] = (donors["shelf_life_min"] + donors["shelf_life_max"]) / 2

    shelters["demand_rate_norm"] = shelters["demand_rate"] / shelters["demand_rate"].max()
    shelters["capacity_norm"] = shelters["capacity"] / shelters["capacity"].max()

    return donors, shelters


def compute_summary_stats(
    donors: pd.DataFrame,
    shelters: pd.DataFrame,
    distance_matrix: np.ndarray,
    config: dict,
) -> dict:
    """Stats useful for the report and for sanity-checking."""
    return {
        "scenario": config["scenario_name"],
        "version": config.get("version", "unknown"),
        "num_donors": len(donors),
        "num_shelters": len(shelters),
        "grid_size": config["grid_size"],
        "episode_length": config["episode_length"],
        "total_expected_supply_per_episode": float(
            (donors["arrival_rate"] * donors["avg_quantity"]).sum() * config["episode_length"]
        ),
        "total_expected_demand_per_episode": float(
            shelters["demand_rate"].sum() * config["episode_length"]
        ),
        "min_donor_shelter_distance": int(distance_matrix.min()),
        "max_donor_shelter_distance": int(distance_matrix.max()),
        "mean_donor_shelter_distance": float(distance_matrix.mean()),
        "donor_types": donors["type"].value_counts().to_dict(),
        "shelter_types": shelters["type"].value_counts().to_dict(),
    }


def process_scenario(scenario: str) -> None:
    """End-to-end: read raw, validate, feature-engineer, write processed."""
    print(f"\n{'=' * 60}")
    print(f"Processing scenario: {scenario}")
    print(f"{'=' * 60}")

    raw_path = RAW_DIR / scenario
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw scenario folder not found: {raw_path}")

    donors = pd.read_csv(raw_path / "donors.csv")
    shelters = pd.read_csv(raw_path / "shelters.csv")
    with open(raw_path / "scenario_config.yaml") as f:
        config = yaml.safe_load(f)

    print(f"Loaded {len(donors)} donors, {len(shelters)} shelters")

    validate_donors(donors, config["grid_size"])
    validate_shelters(shelters, config["grid_size"])
    print("Validation passed")

    donors_proc, shelters_proc = normalize_rates(donors, shelters)
    distance_matrix = compute_distance_matrix(donors_proc, shelters_proc)
    donor_distance_matrix = compute_donor_donor_distances(donors_proc)
    print(f"Computed distance matrix: shape {distance_matrix.shape}")

    stats = compute_summary_stats(donors_proc, shelters_proc, distance_matrix, config)

    out_path = PROCESSED_DIR / scenario
    out_path.mkdir(parents=True, exist_ok=True)

    donors_proc.to_csv(out_path / "donors_processed.csv", index=False)
    shelters_proc.to_csv(out_path / "shelters_processed.csv", index=False)
    np.save(out_path / "distance_matrix.npy", distance_matrix)
    np.save(out_path / "donor_distance_matrix.npy", donor_distance_matrix)

    with open(out_path / "scenario_config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    with open(out_path / "summary_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote processed artifacts to {out_path}")
    print(f"Summary: {json.dumps(stats, indent=2)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare scenario data for food rescue RL.")
    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario name (e.g., 'weekday') or 'all' to process every scenario in data/raw/",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.scenario == "all":
        scenarios = [p.name for p in RAW_DIR.iterdir() if p.is_dir()]
        if not scenarios:
            print(f"No scenarios found in {RAW_DIR}", file=sys.stderr)
            return 1
        for s in sorted(scenarios):
            process_scenario(s)
    else:
        process_scenario(args.scenario)

    print("\nData preparation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
