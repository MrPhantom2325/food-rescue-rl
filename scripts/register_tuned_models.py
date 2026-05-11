"""
Register the tuned, multi-seed-evaluated policies in MLflow Model Registry.

This is the Sprint 6 follow-up to scripts/register_models.py (Sprint 5).
The Sprint 5 registrations included un-tuned versions; this adds the tuned
ones as new versions of the same registered model names.

Usage:
    python scripts/register_tuned_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from mlops_tracking import configure_mlflow


REGISTRATION_PLAN = [
    # (multi_seed_run_name, registered_model_name, description)
    ("multi_seed_q_learning_tuned", "food_rescue_qlearning",
     "Q-learning with Optuna-tuned hyperparams, evaluated across 5 seeds"),
    ("multi_seed_sarsa_tuned", "food_rescue_sarsa",
     "SARSA with Optuna-tuned hyperparams, evaluated across 5 seeds"),
    ("multi_seed_dqn_tuned", "food_rescue_dqn",
     "DQN with Optuna-tuned hyperparams, evaluated across 5 seeds"),
]


def main() -> int:
    configure_mlflow(experiment_name="food_rescue_multi_seed")
    client = MlflowClient()
    exp = client.get_experiment_by_name("food_rescue_multi_seed")
    if exp is None:
        print("food_rescue_multi_seed experiment not found.", file=sys.stderr)
        print("Run multi_seed_eval.py first.", file=sys.stderr)
        return 1

    for parent_run_name, model_name, description in REGISTRATION_PLAN:
        # Find the parent multi-seed run
        parent_runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{parent_run_name}'",
            max_results=1,
        )
        if not parent_runs:
            print(f"  Skipped: parent run '{parent_run_name}' not found",
                  file=sys.stderr)
            continue

        parent = parent_runs[0]
        # Find the best-performing nested seed run for registration
        nested_runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
            order_by=["metrics.eval_mean_reward DESC"],
            max_results=1,
        )
        if not nested_runs:
            print(f"  Skipped: no nested runs under '{parent_run_name}'",
                  file=sys.stderr)
            continue

        best_seed_run = nested_runs[0]

        # The policy artifact is logged inside the per-seed training (via train.py
        # called from train_and_eval_one_seed). Note: in our current implementation
        # we DON'T log the policy artifact from multi_seed_eval. The artifact is
        # tied to single-run train.py only. So we register the parent run's
        # multi_seed_summary artifact instead, which captures aggregated metrics.
        artifact_uri = f"runs:/{parent.info.run_id}/multi_seed_summary"

        try:
            client.create_registered_model(model_name)
            print(f"  Created registered model: {model_name}")
        except mlflow.exceptions.RestException:
            pass
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise

        version = client.create_model_version(
            name=model_name,
            source=artifact_uri,
            run_id=parent.info.run_id,
            description=description,
        )
        print(f"  Registered: {model_name} version {version.version}")

    print("\nRegistry contents:")
    for rm in client.search_registered_models():
        print(f"  {rm.name}")
        for v in client.search_model_versions(f"name='{rm.name}'"):
            print(f"    version {v.version}  status={v.status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
