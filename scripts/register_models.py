"""
Register trained policies in the MLflow Model Registry.

For each experiment run, we create or update an entry in the registry under
the agent's name (e.g., 'food_rescue_qlearning') and add a new version
pointing at the run's policy artifact.

Usage:
    python scripts/register_models.py
"""

from __future__ import annotations

import sys

import mlflow
from mlflow.tracking import MlflowClient

from mlops_tracking import configure_mlflow


REGISTRATION_PLAN = [
    # (run_name, registered_model_name, version_description)
    ("qlearning_v1", "food_rescue_qlearning", "Q-learning baseline (default epsilon decay)"),
    ("qlearning_v2_explored", "food_rescue_qlearning", "Q-learning with extended exploration"),
    ("sarsa_v1", "food_rescue_sarsa", "SARSA on-policy baseline"),
    ("dqn_v1", "food_rescue_dqn", "DQN baseline on weekday scenario"),
    ("dqn_v2_holiday", "food_rescue_dqn", "DQN on holiday_rush scenario"),
    ("dqn_v3_normalized", "food_rescue_dqn",
     "DQN with normalized reward scale (delivery=1) and gamma=0.99. "
     "Eval: -24 reward, 79 delivered, 217 spoiled."),
    ("dqn_v4_dense", "food_rescue_dqn",
     "DQN with normalized reward + pickup shaping (0.2/unit) and gamma=0.95. "
     "Eval: +13.6 reward, 74.6 delivered, 197 spoiled. Diagnosed: flat Q-values."),
]


def find_run_by_name(client: MlflowClient, experiment_id: str, run_name: str):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=1,
    )
    if not runs:
        return None
    return runs[0]


def register_run(client: MlflowClient, run, model_name: str, description: str) -> None:
    """Register a run's policy artifact as a new version under model_name."""
    # The artifact path inside the run is "policy/" (we logged it under that name)
    artifact_uri = f"runs:/{run.info.run_id}/policy"

    # Ensure the registered model exists
    try:
        client.create_registered_model(model_name)
        print(f"  Created registered model: {model_name}")
    except mlflow.exceptions.RestException:
        # Already exists; that's fine
        pass
    except Exception as e:
        # Non-rest exceptions (the file-backed store may raise different ones)
        if "already exists" not in str(e).lower():
            raise

    # Create a new version
    version = client.create_model_version(
        name=model_name,
        source=artifact_uri,
        run_id=run.info.run_id,
        description=description,
    )
    print(f"  Registered: {model_name} version {version.version}  "
          f"(run: {run.data.tags.get('mlflow.runName')})")


def main() -> int:
    configure_mlflow(experiment_name="food_rescue_rl")
    client = MlflowClient()
    exp = client.get_experiment_by_name("food_rescue_rl")
    if exp is None:
        print("food_rescue_rl experiment not found. Run train.py first.", file=sys.stderr)
        return 1

    for run_name, model_name, description in REGISTRATION_PLAN:
        run = find_run_by_name(client, exp.experiment_id, run_name)
        if run is None:
            print(f"  Skipped: run '{run_name}' not found", file=sys.stderr)
            continue
        try:
            register_run(client, run, model_name, description)
        except Exception as e:
            print(f"  Failed: {run_name} -> {model_name}: {e}", file=sys.stderr)

    print("\nRegistry contents:")
    for rm in client.search_registered_models():
        print(f"  {rm.name}")
        for v in client.search_model_versions(f"name='{rm.name}'"):
            print(f"    version {v.version}  status={v.status}  "
                  f"source={v.source[:60]}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
