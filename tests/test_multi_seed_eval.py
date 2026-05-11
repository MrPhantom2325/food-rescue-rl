"""Tests for multi_seed_eval.py."""

import json
import os
import tempfile

import pytest

import multi_seed_eval


def _write_minimal_config(tmp_dir: str, output_dir: str) -> str:
    """Write a minimal config YAML and return its path."""
    yaml_text = f"""
run:
  run_id: multiseed_test
  agent: q_learning
  scenario: weekday
  num_episodes: 20
  seed: 42
  output_dir: {output_dir}

agent_params:
  learning_rate: 0.1
  discount: 0.95
  epsilon_start: 1.0
  epsilon_end: 0.2
  epsilon_decay_episodes: 15
  optimistic_init: 0.0
  pos_buckets: 3
  load_buckets: 3

reward_weights:
  delivery: 10.0
  spoilage: 5.0
  distance: 0.1
  unmet_demand: 1.0
  priority_bonus: 0.5
  oversupply_penalty: 0.3

eval:
  n_episodes: 2
  eval_seeds: [100, 101]
"""
    path = os.path.join(tmp_dir, "test_cfg.yaml")
    with open(path, "w") as f:
        f.write(yaml_text)
    return path


class TestAggregateResults:
    def test_aggregates_basic_metrics(self):
        per_seed = [
            {
                "train_seed": 42,
                "eval_mean_reward": -1000.0,
                "eval_std_reward": 100.0,
                "eval_mean_delivered": 50.0,
                "eval_mean_spoiled": 200.0,
                "final_train_mean_reward": -1100.0,
                "eval_per_seed": [{"seed": 100, "total_reward": -1100},
                                  {"seed": 101, "total_reward": -900}],
            },
            {
                "train_seed": 43,
                "eval_mean_reward": -800.0,
                "eval_std_reward": 80.0,
                "eval_mean_delivered": 60.0,
                "eval_mean_spoiled": 180.0,
                "final_train_mean_reward": -900.0,
                "eval_per_seed": [{"seed": 100, "total_reward": -850},
                                  {"seed": 101, "total_reward": -750}],
            },
        ]
        agg = multi_seed_eval.aggregate_results(per_seed)
        assert agg["n_train_seeds"] == 2
        assert agg["eval_mean_reward_mean"] == pytest.approx(-900.0)
        assert agg["eval_mean_reward_min"] == -1000.0
        assert agg["eval_mean_reward_max"] == -800.0
        # All 4 eval episodes flattened
        assert len(agg["all_eval_rewards"]) == 4

    def test_includes_train_seeds(self):
        per_seed = [
            {"train_seed": 42, "eval_mean_reward": 0, "eval_mean_delivered": 0,
             "eval_mean_spoiled": 0, "final_train_mean_reward": 0,
             "eval_per_seed": [], "eval_std_reward": 0},
        ]
        agg = multi_seed_eval.aggregate_results(per_seed)
        assert agg["train_seeds"] == [42]


class TestEndToEndMiniRun:
    def test_runs_three_seeds_without_crash(self, tmp_path, monkeypatch):
        """A 3-seed × 20-episode run should complete and produce summary.json."""
        # Run from a temporary working directory so we don't pollute the real
        # experiments/ folder, but keep PWD == project root so data/processed
        # is accessible. Trick: use absolute output_dir.
        with tempfile.TemporaryDirectory() as tmp_out:
            cfg_path = _write_minimal_config(str(tmp_path), tmp_out)

            agg = multi_seed_eval.run_multi_seed(
                config_path=cfg_path,
                n_seeds=2,  # tiny for speed
                base_train_seed=42,
            )
            assert agg["n_train_seeds"] == 2
            assert "eval_mean_reward_mean" in agg
            assert os.path.exists(
                os.path.join(tmp_out, "multi_seed", "multiseed_test", "summary.json")
            )
