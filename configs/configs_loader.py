
"""
YAML-based experiment configuration system.

Every training run is described by a YAML file in configs/. The structure:

    run:
      run_id: qlearning_v1            # also the MLflow run name + filename stem
      agent: q_learning               # one of: q_learning, sarsa, dqn, random, greedy
      scenario: weekday               # data/processed/<scenario>/ must exist
      num_episodes: 1500
      seed: 42
      output_dir: experiments         # where policies and CSV logs go

    agent_params:                     # agent-specific hyperparameters
      learning_rate: 0.1
      discount: 0.95
      ...

    reward_weights:                   # passed into env's RewardWeights
      delivery: 10.0
      spoilage: 5.0
      ...

    eval:                             # how to evaluate the trained policy
      n_episodes: 5                   # number of evaluation episodes
      eval_seeds: [100, 101, 102, 103, 104]

The loader validates required keys are present and the agent name is supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_AGENTS = {"q_learning", "sarsa", "dqn", "random", "greedy"}


@dataclass
class RunConfig:
    """Top-level run identification and training schedule."""
    run_id: str
    agent: str
    scenario: str
    num_episodes: int
    seed: int
    output_dir: str = "experiments"
    description: str = ""


@dataclass
class EvalConfig:
    """How to evaluate the trained policy after training."""
    n_episodes: int = 5
    eval_seeds: list[int] = field(default_factory=lambda: [100, 101, 102, 103, 104])


@dataclass
class ExperimentConfig:
    """Complete config for a single experiment run."""
    run: RunConfig
    agent_params: dict[str, Any]
    reward_weights: dict[str, float]
    eval: EvalConfig
    raw: dict[str, Any]  # the original YAML dict, for logging to MLflow

    def policy_path(self) -> str:
        ext = ".pt" if self.run.agent == "dqn" else ".pkl"
        return f"{self.run.output_dir}/policies/{self.run.run_id}{ext}"

    def results_csv_path(self) -> str:
        return f"{self.run.output_dir}/results/{self.run.run_id}.csv"

    def meta_json_path(self) -> str:
        return f"{self.run.output_dir}/results/{self.run.run_id}_meta.json"


class ConfigError(Exception):
    """Raised when a config file is missing required keys or has bad values."""


def _require(d: dict, key: str, where: str) -> Any:
    if key not in d:
        raise ConfigError(f"Missing required key '{key}' in {where}")
    return d[key]


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError(f"Config must be a YAML mapping, got {type(raw)}")

    # Validate top-level sections
    run_d = _require(raw, "run", "config root")
    agent_params = raw.get("agent_params", {})
    reward_weights = raw.get("reward_weights", {})
    eval_d = raw.get("eval", {})

    # Build RunConfig
    run = RunConfig(
        run_id=str(_require(run_d, "run_id", "run section")),
        agent=str(_require(run_d, "agent", "run section")),
        scenario=str(_require(run_d, "scenario", "run section")),
        num_episodes=int(_require(run_d, "num_episodes", "run section")),
        seed=int(_require(run_d, "seed", "run section")),
        output_dir=str(run_d.get("output_dir", "experiments")),
        description=str(run_d.get("description", "")),
    )

    if run.agent not in SUPPORTED_AGENTS:
        raise ConfigError(
            f"Unsupported agent: '{run.agent}'. "
            f"Supported: {sorted(SUPPORTED_AGENTS)}"
        )

    # Build EvalConfig
    eval_cfg = EvalConfig(
        n_episodes=int(eval_d.get("n_episodes", 5)),
        eval_seeds=list(eval_d.get("eval_seeds", [100, 101, 102, 103, 104])),
    )

    return ExperimentConfig(
        run=run,
        agent_params=dict(agent_params),
        reward_weights=dict(reward_weights),
        eval=eval_cfg,
        raw=raw,
    )
