"""Tests for tabular Q-learning agent."""

import os
import tempfile

import numpy as np

from agents.q_learning import QLearningAgent, QLearningConfig, discretize_state
from sim.environment import FoodRescueEnv


class TestDiscretization:
    def test_discretize_returns_tuple(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        s = discretize_state(env)
        assert isinstance(s, tuple)
        assert len(s) == 9

    def test_discretize_deterministic(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        s1 = discretize_state(env)
        s2 = discretize_state(env)
        assert s1 == s2

    def test_discretize_changes_with_state(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        s1 = discretize_state(env)
        # Run several steps to definitely change vehicle location and time bucket
        for _ in range(80):
            env.step(env.action_space.sample())
        s2 = discretize_state(env)
        # At minimum the time bucket or vehicle position should differ
        assert s1 != s2


class TestQLearningInit:
    def test_default_config(self):
        env = FoodRescueEnv()
        agent = QLearningAgent(num_actions=env.action_space.n)
        assert agent.num_actions == 11
        assert agent.config.learning_rate == 0.1
        assert agent.table_size() == 0

    def test_custom_config(self):
        cfg = QLearningConfig(learning_rate=0.5, discount=0.9, epsilon_decay_episodes=200)
        agent = QLearningAgent(num_actions=11, config=cfg)
        assert agent.config.learning_rate == 0.5
        assert agent.config.discount == 0.9


class TestEpsilonDecay:
    def test_epsilon_starts_high(self):
        agent = QLearningAgent(num_actions=11)
        assert agent.epsilon() == agent.config.epsilon_start

    def test_epsilon_decays(self):
        cfg = QLearningConfig(epsilon_decay_episodes=10)
        agent = QLearningAgent(num_actions=11, config=cfg)
        for _ in range(5):
            agent.end_episode()
        eps_mid = agent.epsilon()
        assert agent.config.epsilon_end < eps_mid < agent.config.epsilon_start

    def test_epsilon_clamps_at_end(self):
        cfg = QLearningConfig(epsilon_decay_episodes=10)
        agent = QLearningAgent(num_actions=11, config=cfg)
        for _ in range(20):
            agent.end_episode()
        assert agent.epsilon() == agent.config.epsilon_end


class TestActionSelection:
    def test_action_in_range(self):
        env = FoodRescueEnv()
        obs, _ = env.reset(seed=42)
        agent = QLearningAgent(num_actions=env.action_space.n, seed=0)
        for _ in range(50):
            a = agent.select_action(env, obs)
            assert 0 <= a < env.action_space.n

    def test_eval_mode_is_deterministic(self):
        env = FoodRescueEnv()
        obs, _ = env.reset(seed=42)
        agent = QLearningAgent(num_actions=env.action_space.n, seed=0)
        agent.set_training(False)
        # In eval mode with no Q-table populated, _argmax_q falls back to random.
        # To make it deterministic, populate one entry.
        s = discretize_state(env)
        q_row = agent._ensure_q_row(s)
        q_row[3] = 5.0  # action 3 has high Q
        a = agent.select_action(env, obs)
        assert a == 3


class TestQUpdate:
    def test_update_changes_q_value(self):
        """Update should grow the table by writing to env_before's discretized state."""
        env = FoodRescueEnv()
        env.reset(seed=42)
        agent = QLearningAgent(num_actions=env.action_space.n, seed=0)
        assert agent.table_size() == 0

        # Snapshot the env state, then step. After step, env_after is current env.
        # The update writes to whatever env_before's discretization is.
        # Since we can't easily clone env, this test calls update with the SAME env
        # for both args — so state_before == state_after, but the row should still
        # be created and modified.
        agent.update_from_transition(
            env_before=env, action=0, reward=10.0,
            env_after=env, done=False,
        )
        # At least one row was created
        assert agent.table_size() >= 1
        # The row for the env's current discretized state should have action 0 != 0.0
        from agents.q_learning import discretize_state as ds
        state = ds(env)
        assert state in agent._q_table
        # With lr=0.1, gamma=0.95, optimistic_init=0, reward=10:
        # target = 10 + 0.95 * max(row_after) = 10 + 0.95 * 0 = 10 (since same state's row was just created)
        # update: 0 + 0.1 * (10 - 0) = 1.0
        # But because state_before == state_after and we just wrote action 0 = 1.0
        # before the bootstrap reads it... actually order matters: _ensure_q_row
        # creates state_before's row first (all zeros), then we read state_after's
        # row (same row, now zeros), so max_next = 0 -> target = 10 -> q[0] = 1.0
        assert agent._q_table[state][0] != 0.0

    def test_update_with_high_reward_increases_q(self):
        """A positive reward should push Q-value up."""
        env = FoodRescueEnv()
        env.reset(seed=42)
        agent = QLearningAgent(num_actions=env.action_space.n, seed=0)
        from agents.q_learning import discretize_state as ds

        state = ds(env)
        # Pre-create the row
        agent._ensure_q_row(state)
        initial = agent._q_table[state][0]
        agent.update_from_transition(env_before=env, action=0, reward=100.0,
                                     env_after=env, done=False)
        after = agent._q_table[state][0]
        assert after > initial

    def test_update_with_negative_reward_decreases_q(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        agent = QLearningAgent(num_actions=env.action_space.n, seed=0)
        from agents.q_learning import discretize_state as ds

        state = ds(env)
        agent._ensure_q_row(state)
        agent._q_table[state][0] = 5.0  # start positive
        agent.update_from_transition(env_before=env, action=0, reward=-10.0,
                                     env_after=env, done=False)
        after = agent._q_table[state][0]
        assert after < 5.0

    def test_no_update_in_eval_mode(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        agent = QLearningAgent(num_actions=env.action_space.n, seed=0)
        agent.set_training(False)
        agent.update_from_transition(env_before=env, action=0, reward=10.0,
                                     env_after=env, done=False)
        assert agent.table_size() == 0

    def test_no_update_in_eval_mode(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        agent = QLearningAgent(num_actions=env.action_space.n, seed=0)
        agent.set_training(False)
        agent.update_from_transition(env_before=env, action=0, reward=10.0,
                                     env_after=env, done=False)
        assert agent.table_size() == 0


class TestSaveLoad:
    def test_save_load_roundtrip(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        agent = QLearningAgent(num_actions=env.action_space.n, seed=0)
        # Populate a few cells
        for _ in range(20):
            from agents.q_learning import discretize_state as ds
            s = ds(env)
            q = agent._ensure_q_row(s)
            q[0] += 0.5
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                break

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "q.pkl")
            agent.save(path)
            loaded = QLearningAgent.load(path)
            assert loaded.table_size() == agent.table_size()
            assert loaded.config.learning_rate == agent.config.learning_rate
            for state, row in agent._q_table.items():
                np.testing.assert_array_almost_equal(loaded._q_table[state], row)

    def test_loaded_agent_is_eval_mode(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        agent = QLearningAgent(num_actions=env.action_space.n, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "q.pkl")
            agent.save(path)
            loaded = QLearningAgent.load(path)
            assert loaded._training is False


class TestStatsAndDiagnostics:
    def test_stats_empty_table(self):
        agent = QLearningAgent(num_actions=11)
        s = agent.stats()
        assert s["table_size"] == 0

    def test_stats_after_population(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        agent = QLearningAgent(num_actions=env.action_space.n, seed=0)
        from agents.q_learning import discretize_state as ds
        for _ in range(10):
            s = ds(env)
            agent._ensure_q_row(s)
            env.step(env.action_space.sample())
        stats = agent.stats()
        assert stats["table_size"] > 0
        assert "mean_q" in stats
        assert "epsilon_current" in stats
