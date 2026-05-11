"""
Print a comparison table of all MLflow runs in the food_rescue_rl experiment.

Usage:
    python scripts/compare_runs.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import textwrap

# Ensure the project root is on sys.path so top-level imports (e.g. mlops_tracking)
# resolve when this file is executed as a script from a subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import pandas as pd

from mlops_tracking import configure_mlflow


def main() -> int:
    configure_mlflow(experiment_name="food_rescue_rl")

    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__ or "Print a comparison table of MLflow runs."),
    )
    parser.add_argument("--output", "-o", help="Path to save the table as CSV")
    args = parser.parse_args()

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("food_rescue_rl")
    if experiment is None:
        print("No food_rescue_rl experiment found", file=sys.stderr)
        return 1

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.eval_mean_reward DESC"],
    )
    if not runs:
        print("No runs found in food_rescue_rl experiment", file=sys.stderr)
        return 1

    rows = []
    for r in runs:
        rows.append({
            "run_name": r.data.tags.get("mlflow.runName", "(no name)"),
            "agent": r.data.tags.get("agent", "?"),
            "scenario": r.data.tags.get("scenario", "?"),
            "eval_mean_reward": r.data.metrics.get("eval_mean_reward"),
            "eval_std_reward": r.data.metrics.get("eval_std_reward"),
            "eval_mean_delivered": r.data.metrics.get("eval_mean_delivered"),
            "eval_mean_spoiled": r.data.metrics.get("eval_mean_spoiled"),
            "wall_time_s": r.data.metrics.get("wall_time_seconds"),
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", "{:.2f}".format)
    print()
    print("=" * 100)
    print("Food Rescue RL — Experiment Comparison")
    print("=" * 100)
    print(df.to_string(index=False))
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved comparison table to {out_path}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
