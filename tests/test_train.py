"""Tests for train.py factory functions and helpers.

These tests focus on construction, not full training — the latter is covered
by Sprint 5's actual experiment runs.
"""

import csv
import os
import tempfile


from agents.dqn import DQNAgent
from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.baseline import GreedyPolicy, RandomPolicy
from configs_loader import load_config

import train as train_mod


def _config_for(agent_name: str, tmp_dir: str) -> str:
    """Helper: write a minimal config and return its path."""
    yaml_text = f"""
run:
  run_id: test_{agent_name}
  agent: {agent_name}
  scenario: weekday
  num_episodes: 3
  seed: 0
  output_dir: {tmp_dir}

agent_params: {{}}

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
    path = os.path.join(tmp_dir, f"cfg_{agent_name}.yaml")
    with open(path, "w") as f:
        f.write(yaml_text)
    return path


class TestBuildEnv:
    def test_env_uses_config_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _config_for("greedy", tmp)
            cfg = load_config(path)
            env = train_mod.build_env(cfg)
            assert env.config.scenario_name == "weekday"

    def test_env_uses_config_reward_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _config_for("greedy", tmp)
            cfg = load_config(path)
            env = train_mod.build_env(cfg)
            assert env.config.reward_weights.delivery == 10.0


class TestBuildAgent:
    def test_random_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(_config_for("random", tmp))
            env = train_mod.build_env(cfg)
            agent = train_mod.build_agent(cfg, env)
            assert isinstance(agent, RandomPolicy)

    def test_greedy_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(_config_for("greedy", tmp))
            env = train_mod.build_env(cfg)
            agent = train_mod.build_agent(cfg, env)
            assert isinstance(agent, GreedyPolicy)

    def test_q_learning_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(_config_for("q_learning", tmp))
            env = train_mod.build_env(cfg)
            agent = train_mod.build_agent(cfg, env)
            assert isinstance(agent, QLearningAgent)

    def test_sarsa_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(_config_for("sarsa", tmp))
            env = train_mod.build_env(cfg)
            agent = train_mod.build_agent(cfg, env)
            assert isinstance(agent, SARSAAgent)

    def test_dqn_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(_config_for("dqn", tmp))
            env = train_mod.build_env(cfg)
            agent = train_mod.build_agent(cfg, env)
            assert isinstance(agent, DQNAgent)


class TestTrainingLoops:
    """Quick smoke runs of each loop with very few episodes."""

    def test_random_training_loop_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(_config_for("random", tmp))
            env = train_mod.build_env(cfg)
            agent = train_mod.build_agent(cfg, env)
            history = train_mod.train(env, agent, num_episodes=3, seed=0, agent_kind="random")
            assert len(history) == 3
            assert all("total_reward" in h for h in history)

    def test_greedy_training_loop_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(_config_for("greedy", tmp))
            env = train_mod.build_env(cfg)
            agent = train_mod.build_agent(cfg, env)
            history = train_mod.train(env, agent, num_episodes=3, seed=0, agent_kind="greedy")
            assert len(history) == 3

    def test_qlearning_training_loop_updates_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(_config_for("q_learning", tmp))
            env = train_mod.build_env(cfg)
            agent = train_mod.build_agent(cfg, env)
            assert agent.table_size() == 0
            history = train_mod.train(env, agent, num_episodes=3, seed=0, agent_kind="q_learning")
            assert len(history) == 3
            # Q-table should have grown
            assert agent.table_size() > 0

    def test_sarsa_training_loop_updates_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(_config_for("sarsa", tmp))
            env = train_mod.build_env(cfg)
            agent = train_mod.build_agent(cfg, env)
            history = train_mod.train(env, agent, num_episodes=3, seed=0, agent_kind="sarsa")
            assert len(history) == 3
            assert agent.table_size() > 0

    def test_dqn_training_loop_fills_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(_config_for("dqn", tmp))
            env = train_mod.build_env(cfg)
            agent = train_mod.build_agent(cfg, env)
            assert len(agent.replay) == 0
            history = train_mod.train(env, agent, num_episodes=2, seed=0, agent_kind="dqn")
            assert len(history) == 2
            # Replay buffer should have transitions (200 steps/episode x 2 = 400)
            assert len(agent.replay) > 100


class TestEvaluate:
    def test_evaluate_returns_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(_config_for("greedy", tmp))
            env = train_mod.build_env(cfg)
            agent = train_mod.build_agent(cfg, env)
            summary = train_mod.evaluate(env, agent, eval_seeds=[100, 101])
            assert "eval_mean_reward" in summary
            assert "eval_std_reward" in summary
            assert "per_seed" in summary
            assert len(summary["per_seed"]) == 2


class TestWriters:
    def test_write_results_csv(self):
        history = [
            {"episode": 0, "total_reward": 10.0, "steps": 100},
            {"episode": 1, "total_reward": 20.0, "steps": 100},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            train_mod.write_results_csv(history, path)
            assert os.path.exists(path)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 2
            assert rows[0]["total_reward"] == "10.0"

    def test_get_git_hash_returns_string(self):
        # Just check it returns a string, doesn't crash, doesn't matter what
        h = train_mod.get_git_hash()
        assert isinstance(h, str)
        assert len(h) > 0
