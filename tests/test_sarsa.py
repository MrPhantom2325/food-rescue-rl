
"""Tests for SARSA agent (subclass of Q-learning)."""

import os
import tempfile

import numpy as np

from agents.q_learning import QLearningConfig, discretize_state
from agents.sarsa import SARSAAgent
from sim.environment import FoodRescueEnv


class TestSARSAInheritance:
    def test_inherits_from_qlearning(self):
        from agents.q_learning import QLearningAgent
        agent = SARSAAgent(num_actions=11)
        assert isinstance(agent, QLearningAgent)

    def test_name_is_sarsa(self):
        agent = SARSAAgent(num_actions=11)
        assert agent.name == "sarsa"

    def test_action_selection_works(self):
        env = FoodRescueEnv()
        obs, _ = env.reset(seed=42)
        agent = SARSAAgent(num_actions=env.action_space.n, seed=0)
        for _ in range(20):
            a = agent.select_action(env, obs)
            assert 0 <= a < env.action_space.n


class TestSARSAUpdate:
    def test_update_uses_next_action_value(self):
        """SARSA bootstrap should use Q(s', next_action), not max over actions."""
        env = FoodRescueEnv()
        env.reset(seed=42)
        agent = SARSAAgent(
            num_actions=env.action_space.n,
            config=QLearningConfig(learning_rate=1.0, discount=1.0, optimistic_init=0.0),
            seed=0,
        )
        agent.set_training(True)

        # Both env_before and env_after refer to the same env state for this unit test.
        # state_before == state_after, but that's fine — we're testing the math of
        # the target computation, not state transitions.
        state = discretize_state(env)

        # Pre-create the row so the bootstrap reads our planted values
        agent._ensure_q_row(state)
        # Plant: action 0 has high Q, action 5 has very negative Q
        agent._q_table[state][0] = 100.0
        agent._q_table[state][5] = -100.0

        # Take action 0, get reward=1, bootstrap from action 5 (low value)
        # SARSA target = reward + 1.0 * Q(state, 5) = 1.0 + (-100.0) = -99.0
        # Update for Q(state, 0): old=100, lr=1.0
        #   new = 100 + 1.0 * (-99.0 - 100) = -99.0
        agent.update_from_transition(
            env_before=env, action=0, reward=1.0,
            env_after=env, done=False, next_action=5,
        )

        # The Q-value of action 0 should now reflect the LOW bootstrap (next_action=5)
        # If SARSA had used max instead (which would be Q[0]=100 itself), the update
        # would have been: 100 + 1.0 * (1.0 + 100 - 100) = 101, not -99.
        # So a negative or near-zero value here proves SARSA used next_action.
        result = agent._q_table[state][0]
        assert result < 0, (
            f"SARSA update should bootstrap from Q(s', next_action=5)=-100, "
            f"giving target near -99. Got Q[0] = {result}."
        )

    def test_update_with_high_next_action_value(self):
        """Mirror test: bootstrapping from a high-value next_action should push Q up."""
        env = FoodRescueEnv()
        env.reset(seed=42)
        agent = SARSAAgent(
            num_actions=env.action_space.n,
            config=QLearningConfig(learning_rate=1.0, discount=1.0, optimistic_init=0.0),
            seed=0,
        )
        agent.set_training(True)

        state = discretize_state(env)
        agent._ensure_q_row(state)
        agent._q_table[state][0] = 0.0
        agent._q_table[state][7] = 50.0

        # SARSA target = 1.0 + 1.0 * Q(state, 7) = 1.0 + 50.0 = 51.0
        # Update Q(state, 0): 0 + 1.0 * (51.0 - 0) = 51.0
        agent.update_from_transition(
            env_before=env, action=0, reward=1.0,
            env_after=env, done=False, next_action=7,
        )
        assert agent._q_table[state][0] > 40.0

    def test_no_update_in_eval_mode(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        agent = SARSAAgent(num_actions=env.action_space.n, seed=0)
        agent.set_training(False)
        agent.update_from_transition(env_before=env, action=0, reward=10.0,
                                     env_after=env, done=False, next_action=0)
        assert agent.table_size() == 0

    def test_no_update_in_eval_mode(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        agent = SARSAAgent(num_actions=env.action_space.n, seed=0)
        agent.set_training(False)
        agent.update_from_transition(env_before=env, action=0, reward=10.0,
                                     env_after=env, done=False, next_action=0)
        assert agent.table_size() == 0


class TestSARSASaveLoad:
    def test_save_load_roundtrip(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        agent = SARSAAgent(num_actions=env.action_space.n, seed=0)

        # Populate a couple entries
        s = discretize_state(env)
        agent._ensure_q_row(s)[2] = 7.5

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sarsa.pkl")
            agent.save(path)
            loaded = SARSAAgent.load(path)
            assert loaded.table_size() == agent.table_size()
            np.testing.assert_array_almost_equal(loaded._q_table[s], agent._q_table[s])
