"""
Policy loader for the FastAPI prediction service.

Three loading strategies, tried in order:

1. From the MLflow Model Registry (production-style): set
   FOOD_RESCUE_MODEL_NAME and FOOD_RESCUE_MODEL_VERSION env vars
2. From the MLflow Model Registry defaults: food_rescue_dqn @ latest
3. From a local file path: set FOOD_RESCUE_MODEL_PATH
4. Built-in fallback: look for experiments/policies/dqn_v5_masked.pt
   (or any DQN policy file in that folder)

Only DQN policies are supported for serving — they take an obs vector directly,
while tabular agents need env-derived state. This keeps the serving layer
simple. Sprint 7 future: serve tabular agents by re-introducing a tiny
mini-env that exposes scenario.donors and scenario.shelters.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agents.dqn import DQNAgent


# -----------------------------
# Strategies
# -----------------------------

def _load_from_mlflow_registry(
    model_name: str,
    version: str,
) -> tuple[DQNAgent, dict[str, Any]]:
    """Load a registered model from MLflow's registry."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from mlops_tracking import configure_mlflow

    configure_mlflow()
    client = MlflowClient()

    # Handle 'latest' as a special case
    if version.lower() == 'latest':
        versions = client.get_latest_versions(name=model_name)
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        # Get the production version if available, otherwise the highest version number
        prod_version = next((v for v in versions if v.current_stage == 'Production'), None)
        mv = prod_version or max(versions, key=lambda v: int(v.version))
        version = mv.version
    else:
        # Ensure version is a string representation of an integer
        try:
            int(version)
        except ValueError:
            raise ValueError(f"Model version must be an integer or 'latest', got '{version}'")
        mv = client.get_model_version(name=model_name, version=version)

    source_uri = mv.source

    print(f"  Loading from MLflow Model Registry: {model_name} v{version}")
    print(f"  Source: {source_uri}")

    # Download the artifact directory
    local_dir = mlflow.artifacts.download_artifacts(source_uri)
    agent, info = _load_dqn_from_dir(Path(local_dir), source=f"mlflow:{model_name}:{version}")
    # Override the cosmetic fields so /info reflects the real registry version
    info["model_name"] = model_name
    info["model_version"] = str(version)
    return agent, info


def _load_from_path(path: str) -> tuple[DQNAgent, dict[str, Any]]:
    """Load a DQN policy from a .pt file on local disk."""
    print(f"  Loading from local path: {path}")
    p = Path(path)
    if p.is_dir():
        return _load_dqn_from_dir(p, source=str(p))
    # Single .pt file
    return _load_dqn_file(p, source=str(p))


def _load_dqn_from_dir(dir_path: Path, source: str) -> tuple[DQNAgent, dict[str, Any]]:
    """Find the .pt file inside a directory and load it."""
    pt_files = list(dir_path.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt file found in {dir_path}")
    # Prefer the one with the largest size (the actual model, not a sidecar)
    pt_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return _load_dqn_file(pt_files[0], source=source)


def _load_dqn_file(pt_path: Path, source: str) -> tuple[DQNAgent, dict[str, Any]]:
    """Load a single DQN .pt file and its meta.json sidecar."""
    agent = DQNAgent.load(pt_path)

    # The sidecar JSON has full metadata
    meta_path = pt_path.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    # Build the info dict that the API uses for /info and /predict
    info = {
        "model_name": pt_path.stem,
        "model_version": "local",
        "agent_kind": "dqn",
        "obs_dim": meta.get("obs_dim", agent.obs_dim),
        "num_actions": meta.get("num_actions", agent.num_actions),
        "scenario_trained_on": None,
        "source": source,
        # For action interpretation; default to 5+5 if unknown
        "num_donors": 5,
        "num_shelters": 5,
    }

    print(f"  Loaded DQN: obs_dim={info['obs_dim']}, num_actions={info['num_actions']}")
    return agent, info


# -----------------------------
# Public entry point
# -----------------------------

def load_policy_from_env() -> tuple[DQNAgent, dict[str, Any]]:
    """
    Resolve which policy to load based on environment variables and load it.

    Resolution order:
    1. FOOD_RESCUE_MODEL_NAME + FOOD_RESCUE_MODEL_VERSION -> MLflow Registry
    2. FOOD_RESCUE_MODEL_PATH -> local file or directory
    3. Default MLflow Registry model food_rescue_dqn @ latest (skipped if unavailable)
    4. Local DQN fallback files

    Raises FileNotFoundError if no policy can be loaded via any method.
    """
    model_name = os.environ.get("FOOD_RESCUE_MODEL_NAME")
    model_version = os.environ.get("FOOD_RESCUE_MODEL_VERSION")
    model_path = os.environ.get("FOOD_RESCUE_MODEL_PATH")

    if model_name and model_version:
        return _load_from_mlflow_registry(model_name, model_version)

    if model_path:
        return _load_from_path(model_path)

    # Try MLflow registry default, but don't fail if it's unavailable (e.g., CI environment)
    # Can be disabled entirely with FOOD_RESCUE_DISABLE_MLFLOW_REGISTRY=1
    if not os.environ.get("FOOD_RESCUE_DISABLE_MLFLOW_REGISTRY"):
        try:
            return _load_from_mlflow_registry("food_rescue_dqn", "latest")
        except Exception as e:
            print(f"  MLflow registry not available ({type(e).__name__}), trying local fallback...")
    else:
        print("  MLflow registry disabled via FOOD_RESCUE_DISABLE_MLFLOW_REGISTRY")

    # Fallback: look for any DQN policy on local disk
    candidates = [
        Path("experiments/policies/dqn_v5_masked.pt"),
        Path("experiments/policies/dqn_tuned.pt"),
        Path("experiments/policies/dqn_v3_normalized.pt"),
    ]
    for c in candidates:
        if c.exists():
            print(f"  Found local policy: {c}")
            return _load_from_path(str(c))

    raise FileNotFoundError(
        "No policy could be loaded. Please set one of:\n"
        "  - FOOD_RESCUE_MODEL_NAME + FOOD_RESCUE_MODEL_VERSION (MLflow registry)\n"
        "  - FOOD_RESCUE_MODEL_PATH (local file or directory)\n"
        "  - Place a DQN policy at experiments/policies/dqn_v5_masked.pt or similar"
    )
