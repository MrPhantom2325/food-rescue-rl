"""
Pydantic request/response schemas for the food rescue prediction API.

Why Pydantic?
- Auto-validation: bad requests return 422 with clear error messages
- Auto-docs: FastAPI generates Swagger UI from these models
- Type safety: editors autocomplete, mypy catches mismatches
- Serialization: request/response JSON is generated automatically
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


# -----------------------------
# /predict
# -----------------------------

class PredictRequest(BaseModel):
    """A request for an action given an observation."""

    observation: list[float] = Field(
        ...,
        description=(
            "The observation vector from FoodRescueEnv. Must match the obs_dim "
            "the loaded policy was trained on (typically 31 features)."
        ),
        min_length=1,
        max_length=200,
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Optional client-supplied ID for tracing. If not provided, "
                    "the server generates one.",
    )

    @field_validator("observation")
    @classmethod
    def validate_finite_floats(cls, v: list[float]) -> list[float]:
        """Reject NaN / inf — they would break the model."""
        for i, x in enumerate(v):
            if x != x:  # NaN check
                raise ValueError(f"observation[{i}] is NaN")
            if x in (float("inf"), float("-inf")):
                raise ValueError(f"observation[{i}] is infinite")
        return v


class PredictResponse(BaseModel):
    """The chosen action plus interpretation."""

    action: int = Field(..., description="Discrete action index in [0, num_actions)")
    action_kind: str = Field(
        ...,
        description="Human-readable interpretation: 'donor', 'shelter', or 'idle'",
    )
    action_target_index: Optional[int] = Field(
        default=None,
        description="The donor/shelter index for non-idle actions (None for idle)",
    )
    q_values: Optional[list[float]] = Field(
        default=None,
        description="Q-values for each action, when available (DQN only)",
    )
    model_name: str = Field(..., description="Name of the loaded model")
    model_version: str = Field(..., description="Version identifier of the loaded model")
    request_id: str = Field(..., description="Server-assigned or client-supplied trace ID")
    timestamp_iso: str = Field(..., description="Server timestamp in ISO 8601 UTC")


# -----------------------------
# /health
# -----------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' if service is healthy")
    model_loaded: bool = Field(..., description="True if a policy is loaded and ready")
    uptime_seconds: float = Field(..., description="Seconds since service started")


# -----------------------------
# /info
# -----------------------------

class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    agent_kind: str = Field(..., description="'dqn', 'q_learning', etc.")
    obs_dim: int = Field(..., description="Number of features the policy expects")
    num_actions: int = Field(..., description="Action space size")
    scenario_trained_on: Optional[str] = Field(
        default=None,
        description="Scenario name from the training run, if known",
    )
    source: str = Field(..., description="Where the policy was loaded from")


# -----------------------------
# /metrics (basic Prometheus-style numbers)
# -----------------------------

class MetricsResponse(BaseModel):
    total_predictions: int
    predictions_by_action: dict[str, int]
    avg_latency_ms: Optional[float] = None
    uptime_seconds: float
