"""
FastAPI server for the food rescue prediction service.

Endpoints:
- GET  /health    -> HealthResponse
- GET  /info      -> ModelInfoResponse
- GET  /metrics   -> MetricsResponse
- POST /predict   -> PredictResponse
- GET  /docs      -> auto-generated Swagger UI

The service loads ONE policy on startup (configurable via env vars) and serves
predictions from it. To swap models, restart the service with different config.

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Then visit http://localhost:8000/docs
"""

from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from api.schemas import (
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)


# -----------------------------
# Application state
# -----------------------------
# We use a module-level dict instead of FastAPI's Depends() machinery because
# our state is mutable (load on startup, log per request) and small. For
# bigger apps you'd use a proper dependency-injection pattern.

state: dict = {
    "policy": None,
    "model_info": None,
    "started_at": time.time(),
    "prediction_count": 0,
    "action_counts": defaultdict(int),
    "latencies_ms": [],
}


# -----------------------------
# Startup / shutdown
# -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the policy on startup."""
    print("Starting food rescue prediction service...")
    try:
        from api.policy_loader import load_policy_from_env
        policy, info = load_policy_from_env()
        state["policy"] = policy
        state["model_info"] = info
        print(f"Loaded policy: {info['model_name']} v{info['model_version']}")
    except Exception as e:
        # Don't kill the service if loading fails — let /health report unhealthy
        print(f"WARNING: Failed to load policy on startup: {e}")
        print("Service starting in degraded mode; /predict will return 503.")
    yield
    # No specific shutdown work yet (DB writes happen synchronously)
    print("Service shutting down.")


app = FastAPI(
    title="Food Rescue Prediction API",
    description=(
        "Serves dispatch decisions from a trained RL policy. Given the current "
        "observation of the food rescue environment, returns the recommended "
        "action (which donor/shelter to head to, or idle)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# -----------------------------
# /health
# -----------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if state["policy"] is not None else "degraded",
        model_loaded=state["policy"] is not None,
        uptime_seconds=time.time() - state["started_at"],
    )


# -----------------------------
# /info
# -----------------------------

@app.get("/info", response_model=ModelInfoResponse)
async def info():
    if state["model_info"] is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Check /health and service startup logs.",
        )
    return ModelInfoResponse(**state["model_info"])


# -----------------------------
# /metrics
# -----------------------------

@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    latencies = state["latencies_ms"]
    avg_latency = float(np.mean(latencies)) if latencies else None
    return MetricsResponse(
        total_predictions=state["prediction_count"],
        predictions_by_action=dict(state["action_counts"]),
        avg_latency_ms=avg_latency,
        uptime_seconds=time.time() - state["started_at"],
    )


# -----------------------------
# /predict
# -----------------------------

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if state["policy"] is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Check /health.",
        )

    info = state["model_info"]
    obs_dim = info["obs_dim"]
    num_actions = info["num_actions"]

    if len(request.observation) != obs_dim:
        raise HTTPException(
            status_code=422,
            detail=(
                f"observation has length {len(request.observation)}, "
                f"but the loaded model expects obs_dim={obs_dim}"
            ),
        )

    request_id = request.request_id or str(uuid.uuid4())

    start = time.time()
    obs_array = np.array(request.observation, dtype=np.float32)

    # Call the policy. DQN's select_action signature is (env, obs) — env can be
    # None since DQN only uses the obs argument.
    policy = state["policy"]
    action, q_values = _select_action_and_q_values(policy, obs_array, num_actions)

    latency_ms = (time.time() - start) * 1000.0

    # Interpret action
    # Assumes the typical N+M+1 action layout: 0..N-1 = donors, N..N+M-1 = shelters, N+M = idle
    # We don't know N and M at runtime without the env, so we use a generic interpretation:
    # the policy info should carry this. For now, we encode using a config field.
    num_donors = info.get("num_donors")
    num_shelters = info.get("num_shelters")
    action_kind, target_index = _interpret_action(action, num_donors, num_shelters)

    # Update metrics
    state["prediction_count"] += 1
    state["action_counts"][action_kind] += 1
    state["latencies_ms"].append(latency_ms)
    # Keep only the most recent 1000 latencies (rolling window)
    if len(state["latencies_ms"]) > 1000:
        state["latencies_ms"] = state["latencies_ms"][-1000:]

    response = PredictResponse(
        action=int(action),
        action_kind=action_kind,
        action_target_index=target_index,
        q_values=q_values,
        model_name=info["model_name"],
        model_version=info["model_version"],
        request_id=request_id,
        timestamp_iso=datetime.now(timezone.utc).isoformat(),
    )

    # Log to SQLite (best-effort, doesn't fail the request)
    try:
        from api.prediction_log import log_prediction
        log_prediction(request, response, latency_ms)
    except Exception as e:
        # Log to stdout, don't fail the request
        print(f"WARNING: prediction logging failed: {e}")

    return response


# -----------------------------
# Helpers
# -----------------------------

def _select_action_and_q_values(policy, obs: np.ndarray, num_actions: int):
    """
    Call the policy's select_action with obs, also extracting Q-values if available.

    DQN exposes Q-values; tabular agents don't (in a way that's meaningful for
    arbitrary obs vectors).
    """
    # If it's a DQN, we can extract Q-values directly
    if hasattr(policy, "q_net"):
        import torch
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(policy.device)
            q_tensor = policy.q_net(obs_t).squeeze(0).cpu().numpy()
        action = int(q_tensor.argmax())
        return action, q_tensor.tolist()

    # Fallback: just call select_action with a dummy env=None
    # (Won't work for tabular agents that need env; in that case, the
    # service should refuse to load them. policy_loader enforces this.)
    action = policy.select_action(env=None, obs=obs)
    return int(action), None


def _interpret_action(action: int, num_donors: int | None, num_shelters: int | None):
    """Return (action_kind, target_index) for the action."""
    if num_donors is None or num_shelters is None:
        return "unknown", None
    if action < num_donors:
        return "donor", action
    if action < num_donors + num_shelters:
        return "shelter", action - num_donors
    return "idle", None
