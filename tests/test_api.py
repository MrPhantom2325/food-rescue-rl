"""
Tests for the FastAPI prediction service (Sprint 7).

Uses FastAPI's TestClient so no real server process is needed.
The lifespan policy loader is patched out — tests control state directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient


# ------------------------------------------------------------------
# Shared patch: stop lifespan from loading a real policy
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_policy_loader():
    """Prevent lifespan from calling load_policy_from_env during tests."""
    with patch("api.main.load_policy_from_env_if_needed", side_effect=lambda: None):
        yield


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_mock_policy(best_action: int = 1):
    policy = MagicMock()
    q = torch.zeros(1, 11)
    q[0, best_action] = 10.0
    policy.q_net.return_value = q
    policy.device = "cpu"
    return policy


def _make_model_info():
    return {
        "model_name": "dqn_test",
        "model_version": "local",
        "agent_kind": "dqn",
        "obs_dim": 31,
        "num_actions": 11,
        "scenario_trained_on": None,
        "source": "mock",
        "num_donors": 5,
        "num_shelters": 5,
    }


@pytest.fixture()
def valid_obs():
    rng = np.random.default_rng(42)
    return rng.uniform(0.0, 1.0, size=31).tolist()


@pytest.fixture()
def client():
    from api.main import app, state
    state["policy"] = _make_mock_policy()
    state["model_info"] = _make_model_info()
    state["prediction_count"] = 0
    state["action_counts"].clear()
    state["latencies_ms"].clear()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ------------------------------------------------------------------
# /health
# ------------------------------------------------------------------

def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["model_loaded"] is True


def test_health_degraded():
    from api.main import app, state
    state["policy"] = None
    state["model_info"] = None
    with TestClient(app) as c:
        resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"


# ------------------------------------------------------------------
# /info
# ------------------------------------------------------------------

def test_info_returns_model_metadata(client):
    resp = client.get("/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] == "dqn_test"
    assert body["obs_dim"] == 31
    assert body["num_actions"] == 11


# ------------------------------------------------------------------
# /predict — happy path
# ------------------------------------------------------------------

def test_predict_returns_valid_action(client, valid_obs):
    resp = client.post("/predict", json={"observation": valid_obs})
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["action"] <= 10
    assert body["action_kind"] in ("donor", "shelter", "idle")
    assert isinstance(body["q_values"], list)
    assert len(body["q_values"]) == 11
    assert "request_id" in body
    assert "timestamp_iso" in body


def test_predict_donor_action(client, valid_obs):
    from api.main import state
    state["policy"] = _make_mock_policy(best_action=1)
    resp = client.post("/predict", json={"observation": valid_obs})
    body = resp.json()
    assert body["action"] == 1
    assert body["action_kind"] == "donor"
    assert body["action_target_index"] == 1


def test_predict_shelter_action(client, valid_obs):
    from api.main import state
    state["policy"] = _make_mock_policy(best_action=6)
    resp = client.post("/predict", json={"observation": valid_obs})
    body = resp.json()
    assert body["action"] == 6
    assert body["action_kind"] == "shelter"
    assert body["action_target_index"] == 1  # 6 - 5


def test_predict_idle_action(client, valid_obs):
    from api.main import state
    state["policy"] = _make_mock_policy(best_action=10)
    resp = client.post("/predict", json={"observation": valid_obs})
    body = resp.json()
    assert body["action"] == 10
    assert body["action_kind"] == "idle"
    assert body["action_target_index"] is None


# ------------------------------------------------------------------
# /predict — validation
# ------------------------------------------------------------------

def test_predict_rejects_wrong_obs_dim(client):
    resp = client.post("/predict", json={"observation": [0.1, 0.2, 0.3]})
    assert resp.status_code == 422


def test_predict_rejects_empty_obs(client):
    resp = client.post("/predict", json={"observation": []})
    assert resp.status_code == 422


def test_predict_503_when_no_model(valid_obs):
    from api.main import app, state
    state["policy"] = None
    state["model_info"] = None
    with TestClient(app) as c:
        resp = c.post("/predict", json={"observation": valid_obs})
    assert resp.status_code == 503


def test_predict_request_id_passthrough(client, valid_obs):
    resp = client.post("/predict", json={
        "observation": valid_obs,
        "request_id": "my-trace-abc-123",
    })
    assert resp.json()["request_id"] == "my-trace-abc-123"


# ------------------------------------------------------------------
# /metrics
# ------------------------------------------------------------------

def test_metrics_increments_on_predict(client, valid_obs):
    before = client.get("/metrics").json()["total_predictions"]
    client.post("/predict", json={"observation": valid_obs})
    after = client.get("/metrics").json()["total_predictions"]
    assert after == before + 1


# ------------------------------------------------------------------
# Prediction log
# ------------------------------------------------------------------

def test_log_prediction_writes_row(tmp_path, monkeypatch):
    monkeypatch.setenv("FOOD_RESCUE_LOG_DB", str(tmp_path / "test.db"))
    from api import prediction_log
    import importlib
    importlib.reload(prediction_log)

    req = MagicMock()
    req.observation = [0.1] * 31
    resp = MagicMock()
    resp.request_id = "test-id-001"
    resp.timestamp_iso = "2026-05-11T00:00:00+00:00"
    resp.action = 3
    resp.action_kind = "donor"
    resp.model_name = "dqn_test"
    resp.model_version = "local"

    prediction_log.log_prediction(req, resp, latency_ms=5.0)
    rows = prediction_log.fetch_recent(10)
    assert len(rows) == 1
    assert rows[0]["request_id"] == "test-id-001"
    assert rows[0]["action"] == 3
    assert rows[0]["latency_ms"] == 5.0


# ------------------------------------------------------------------
# Drift detector
# ------------------------------------------------------------------

def test_drift_report_insufficient_data():
    from monitoring.drift_detector import DriftDetector
    detector = DriftDetector(min_live_samples=30)
    live = [[float(i % 10) / 10.0] * 31 for i in range(5)]
    with patch("monitoring.drift_detector.DriftDetector._get_reference") as mock_ref:
        mock_ref.return_value = np.random.default_rng(0).uniform(
            size=(200, 31)).astype(np.float32)
        report = detector.run(live_obs=live)
    assert report.drift_detected is False
    assert "Insufficient" in report.summary()


def test_drift_report_no_drift_same_distribution():
    from monitoring.drift_detector import DriftDetector
    detector = DriftDetector(threshold=0.05)
    rng = np.random.default_rng(99)
    reference = rng.uniform(size=(300, 31)).astype(np.float32)
    live = rng.uniform(size=(50, 31)).tolist()
    with patch("monitoring.drift_detector.DriftDetector._get_reference") as mock_ref:
        mock_ref.return_value = reference
        report = detector.run(live_obs=live)
    assert len(report.drifted_features) <= 5


def test_drift_report_detects_obvious_drift():
    from monitoring.drift_detector import DriftDetector
    detector = DriftDetector(threshold=0.05)
    rng = np.random.default_rng(0)
    reference = rng.uniform(0.0, 1.0, size=(300, 31)).astype(np.float32)
    live = (rng.uniform(0.0, 1.0, size=(50, 31)) + 10.0).tolist()
    with patch("monitoring.drift_detector.DriftDetector._get_reference") as mock_ref:
        mock_ref.return_value = reference
        report = detector.run(live_obs=live)
    assert report.drift_detected is True
    assert len(report.drifted_features) > 20
