
"""
MLflow tracking helpers for the food rescue project.

Sets up a local MLflow tracking server (file-based, no external service required)
and provides convenience wrappers for the training loop.

Backend layout
--------------
- Tracking URI: file://./mlruns  (default — no external server needed)
- Experiment name: defaults to "food_rescue_rl" but can be overridden per run
- Artifact location: same as tracking URI

Why local file-backed MLflow?
- Zero infrastructure: no MLflow server to run, no DB, no S3
- Fully self-contained: the entire MLflow state lives in ./mlruns/
- Trivially inspectable: `mlflow ui` reads from the same folder
- Reproducible: deleting ./mlruns/ resets everything

For the rubric story: a docker-compose deployment in Sprint 8 will bring up a
proper MLflow server with SQLite backend. The interface is the same; only the
tracking_uri changes.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import mlflow


DEFAULT_EXPERIMENT_NAME = "food_rescue_rl"
DEFAULT_TRACKING_URI = f"file://{Path.cwd().resolve() / 'mlruns'}"


def configure_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> str:
    """
    Configure the MLflow client and ensure the experiment exists.

    Returns the experiment_id of the active experiment.
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)

    # Create experiment if it doesn't exist
    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = existing.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


@contextmanager
def start_run(
    run_name: str,
    tags: Optional[dict[str, str]] = None,
    nested: bool = False,
):
    """
    Context manager wrapper for mlflow.start_run with sensible defaults.

    Usage:
        with start_run("qlearning_v1") as run:
            mlflow.log_param("lr", 0.1)
            ...
            log_metric_safe("episode_reward", 100, step=5)
    """
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def log_params_safe(params: dict[str, Any]) -> None:
    """
    Log params dict, casting non-primitive values to strings.

    MLflow can store str, numbers, and booleans as params. Anything else (lists,
    nested dicts, numpy types) gets stringified to avoid serialization errors.
    """
    cleaned = {}
    for key, value in params.items():
        if value is None:
            cleaned[key] = "null"
        elif isinstance(value, (str, bool, int, float)):
            cleaned[key] = value
        elif hasattr(value, "item"):  # numpy scalar
            cleaned[key] = value.item()
        else:
            cleaned[key] = str(value)
    mlflow.log_params(cleaned)


def log_metric_safe(key: str, value: Any, step: Optional[int] = None) -> None:
    """Log a single metric, casting numpy scalars to plain floats."""
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    mlflow.log_metric(key, float(value), step=step)


def log_metrics_safe(metrics: dict[str, Any], step: Optional[int] = None) -> None:
    """Log multiple metrics with safe casting."""
    cleaned = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        cleaned[key] = float(value)
    mlflow.log_metrics(cleaned, step=step)


def get_or_create_experiment(name: str) -> str:
    """Idempotent experiment creation. Returns experiment_id."""
    existing = mlflow.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    return mlflow.create_experiment(name)
