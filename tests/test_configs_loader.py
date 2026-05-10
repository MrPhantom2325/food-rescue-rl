"""Tests for configs_loader."""

import os
import tempfile

import pytest

from configs_loader import (
    ConfigError,
    SUPPORTED_AGENTS,
    load_config,
)


def _write_yaml(tmp_dir: str, contents: str) -> str:
    path = os.path.join(tmp_dir, "test.yaml")
    with open(path, "w") as f:
        f.write(contents)
    return path


class TestLoadValidConfig:
    def test_loads_minimal_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(tmp, """
run:
  run_id: test
  agent: q_learning
  scenario: weekday
  num_episodes: 100
  seed: 0
""")
            cfg = load_config(path)
            assert cfg.run.run_id == "test"
            assert cfg.run.agent == "q_learning"
            assert cfg.run.scenario == "weekday"
            assert cfg.run.num_episodes == 100
            assert cfg.run.seed == 0

    def test_loads_all_real_configs(self):
        """Every config in configs/ should load without error."""
        from pathlib import Path
        for path in sorted(Path("configs").glob("*.yaml")):
            cfg = load_config(path)
            assert cfg.run.agent in SUPPORTED_AGENTS

    def test_paths_are_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(tmp, """
run:
  run_id: my_run
  agent: dqn
  scenario: weekday
  num_episodes: 10
  seed: 0
  output_dir: experiments
""")
            cfg = load_config(path)
            assert cfg.policy_path() == "experiments/policies/my_run.pt"  # .pt for dqn
            assert cfg.results_csv_path() == "experiments/results/my_run.csv"
            assert cfg.meta_json_path() == "experiments/results/my_run_meta.json"

    def test_qlearning_uses_pkl_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(tmp, """
run:
  run_id: my_q
  agent: q_learning
  scenario: weekday
  num_episodes: 10
  seed: 0
""")
            cfg = load_config(path)
            assert cfg.policy_path().endswith(".pkl")

    def test_eval_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(tmp, """
run:
  run_id: t
  agent: q_learning
  scenario: weekday
  num_episodes: 10
  seed: 0
""")
            cfg = load_config(path)
            assert cfg.eval.n_episodes == 5
            assert cfg.eval.eval_seeds == [100, 101, 102, 103, 104]

    def test_eval_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(tmp, """
run:
  run_id: t
  agent: q_learning
  scenario: weekday
  num_episodes: 10
  seed: 0
eval:
  n_episodes: 10
  eval_seeds: [1, 2, 3]
""")
            cfg = load_config(path)
            assert cfg.eval.n_episodes == 10
            assert cfg.eval.eval_seeds == [1, 2, 3]


class TestLoadInvalidConfig:
    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("does_not_exist.yaml")

    def test_missing_run_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(tmp, """
agent_params:
  learning_rate: 0.1
""")
            with pytest.raises(ConfigError, match="run"):
                load_config(path)

    def test_missing_required_run_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(tmp, """
run:
  run_id: t
  agent: q_learning
  scenario: weekday
""")
            # missing num_episodes and seed
            with pytest.raises(ConfigError):
                load_config(path)

    def test_unsupported_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(tmp, """
run:
  run_id: t
  agent: not_a_real_agent
  scenario: weekday
  num_episodes: 10
  seed: 0
""")
            with pytest.raises(ConfigError, match="Unsupported agent"):
                load_config(path)


class TestRawPreserved:
    def test_raw_field_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(tmp, """
run:
  run_id: t
  agent: q_learning
  scenario: weekday
  num_episodes: 10
  seed: 0
agent_params:
  learning_rate: 0.5
  custom_field: hello
""")
            cfg = load_config(path)
            # raw field preserves all original YAML for MLflow logging
            assert cfg.raw["agent_params"]["learning_rate"] == 0.5
            assert cfg.raw["agent_params"]["custom_field"] == "hello"
