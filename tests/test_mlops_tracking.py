
"""Tests for mlops_tracking module."""

import os
import tempfile

import mlflow
import numpy as np
import pytest

from mlops_tracking import (
    configure_mlflow,
    get_or_create_experiment,
    log_metric_safe,
    log_metrics_safe,
    log_params_safe,
    start_run,
)


@pytest.fixture
def tmp_mlflow():
    """Run each test against a fresh tmpdir-backed MLflow store."""
    with tempfile.TemporaryDirectory() as tmp:
        uri = f"file://{tmp}"
        configure_mlflow(tracking_uri=uri, experiment_name="test_experiment")
        yield uri


class TestConfigure:
    def test_configure_sets_tracking_uri(self, tmp_mlflow):
        assert mlflow.get_tracking_uri() == tmp_mlflow

    def test_configure_idempotent(self, tmp_mlflow):
        exp_id_1 = configure_mlflow(tracking_uri=tmp_mlflow, experiment_name="test_experiment")
        exp_id_2 = configure_mlflow(tracking_uri=tmp_mlflow, experiment_name="test_experiment")
        assert exp_id_1 == exp_id_2


class TestExperimentManagement:
    def test_get_or_create_creates_new(self, tmp_mlflow):
        exp_id = get_or_create_experiment("brand_new_exp")
        assert exp_id is not None

    def test_get_or_create_returns_existing(self, tmp_mlflow):
        exp_id_1 = get_or_create_experiment("idempotent_exp")
        exp_id_2 = get_or_create_experiment("idempotent_exp")
        assert exp_id_1 == exp_id_2


class TestStartRun:
    def test_basic_run(self, tmp_mlflow):
        with start_run("test_run") as run:
            assert run is not None
            assert run.info.run_id

    def test_run_with_tags(self, tmp_mlflow):
        with start_run("tagged_run", tags={"algorithm": "test", "version": "1"}) as run:
            run_id = run.info.run_id

        # Re-fetch and verify tags
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_mlflow)
        fetched = client.get_run(run_id)
        assert fetched.data.tags["algorithm"] == "test"
        assert fetched.data.tags["version"] == "1"


class TestLogParamsSafe:
    def test_basic_types(self, tmp_mlflow):
        with start_run("basic_params") as run:
            log_params_safe({
                "lr": 0.1,
                "batch_size": 64,
                "name": "test",
                "active": True,
            })
            run_id = run.info.run_id

        fetched = mlflow.get_run(run_id)
        assert fetched.data.params["lr"] == "0.1"
        assert fetched.data.params["batch_size"] == "64"
        assert fetched.data.params["name"] == "test"

    def test_numpy_scalars(self, tmp_mlflow):
        with start_run("numpy_params") as run:
            log_params_safe({
                "np_int": np.int64(42),
                "np_float": np.float32(3.14),
            })
            run_id = run.info.run_id

        fetched = mlflow.get_run(run_id)
        assert fetched.data.params["np_int"] == "42"
        assert "3.14" in fetched.data.params["np_float"]

    def test_non_primitive_stringified(self, tmp_mlflow):
        with start_run("complex_params") as run:
            log_params_safe({
                "list_param": [1, 2, 3],
                "tuple_param": (4, 5),
                "none_param": None,
            })
            run_id = run.info.run_id

        fetched = mlflow.get_run(run_id)
        # All stringified, no errors
        assert "1" in fetched.data.params["list_param"]
        assert fetched.data.params["none_param"] == "null"


class TestLogMetricsSafe:
    def test_log_metric_safe_with_python_float(self, tmp_mlflow):
        with start_run("metric_run") as run:
            log_metric_safe("reward", 42.5, step=0)
            log_metric_safe("reward", 50.0, step=1)
            run_id = run.info.run_id

        fetched = mlflow.get_run(run_id)
        # MLflow stores the latest value as a key in metrics
        assert "reward" in fetched.data.metrics
        assert fetched.data.metrics["reward"] == 50.0

    def test_log_metric_safe_with_numpy(self, tmp_mlflow):
        with start_run("numpy_metric_run") as run:
            log_metric_safe("loss", np.float32(0.123), step=0)
            run_id = run.info.run_id

        fetched = mlflow.get_run(run_id)
        # numpy float should be cast to plain float — no JSON error
        assert "loss" in fetched.data.metrics

    def test_log_metrics_safe_dict(self, tmp_mlflow):
        with start_run("dict_metric_run") as run:
            log_metrics_safe(
                {"reward": np.float32(10.0), "loss": 0.5, "accuracy": 0.75},
                step=0,
            )
            run_id = run.info.run_id

        fetched = mlflow.get_run(run_id)
        assert fetched.data.metrics["reward"] == pytest.approx(10.0)
        assert fetched.data.metrics["loss"] == 0.5
        assert fetched.data.metrics["accuracy"] == 0.75
