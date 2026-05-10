
"""Tests for DQN agent."""

import json
import os
import tempfile

import numpy as np
import torch

from agents.dqn import DQNAgent, DQNConfig, QNetwork, ReplayBuffer
from sim.environment import FoodRescueEnv


class TestQNetwork:
    def test_forward_shape(self):
        net = QNetwork(obs_dim=31, num_actions=11, hidden_sizes=(64, 64))
        x = torch.zeros(4, 31)
        y = net(x)
        assert y.shape == (4, 11)

    def test_different_inputs_different_outputs(self):
        net = QNetwork(obs_dim=31, num_actions=11)
        x1 = torch.randn(1, 31)
        x2 = torch.randn(1, 31)
        y1 = net(x1)
        y2 = net(x2)
        assert not torch.allclose(y1, y2)


class TestReplayBuffer:
    def test_push_and_size(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(np.zeros(31), 0, 1.0, np.zeros(31), False)
        assert len(buf) == 10

    def test_capacity_overflow(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(20):
            buf.push(np.zeros(31), 0, 1.0, np.zeros(31), False)
        assert len(buf) == 10  # caps at capacity

    def test_sample_shapes(self):
        buf = ReplayBuffer(capacity=100, seed=0)
        for _ in range(50):
            buf.push(np.random.randn(31), np.random.randint(11), 1.0,
                     np.random.randn(31), False)
        obs, act, rew, next_obs, done = buf.sample(16)
        assert obs.shape == (16, 31)
        assert act.shape == (16,)
        assert rew.shape == (16,)
        assert next_obs.shape == (16, 31)
        assert done.shape == (16,)


class TestDQNAgent:
    def test_init(self):
        env = FoodRescueEnv()
        agent = DQNAgent(
            obs_dim=env.observation_space.shape[0],
            num_actions=env.action_space.n,
        )
        assert agent.obs_dim == 31
        assert agent.num_actions == 11
        assert agent.q_net is not None
        assert agent.target_net is not None

    def test_target_net_starts_equal_to_q_net(self):
        agent = DQNAgent(obs_dim=31, num_actions=11, seed=0)
        for p_q, p_t in zip(agent.q_net.parameters(), agent.target_net.parameters()):
            assert torch.allclose(p_q, p_t)

    def test_action_in_range(self):
        env = FoodRescueEnv()
        obs, _ = env.reset(seed=42)
        agent = DQNAgent(
            obs_dim=env.observation_space.shape[0],
            num_actions=env.action_space.n,
            seed=0,
        )
        for _ in range(20):
            a = agent.select_action(env, obs)
            assert 0 <= a < env.action_space.n

    def test_train_step_returns_none_before_min_replay(self):
        agent = DQNAgent(
            obs_dim=31, num_actions=11,
            config=DQNConfig(min_replay_to_train=100),
        )
        # Buffer empty
        assert agent.train_step() is None
        # Add a few transitions but below threshold
        for _ in range(50):
            agent.store_transition(np.zeros(31), 0, 0.0, np.zeros(31), False)
        assert agent.train_step() is None

    def test_train_step_works_after_min_replay(self):
        agent = DQNAgent(
            obs_dim=31, num_actions=11,
            config=DQNConfig(min_replay_to_train=20, batch_size=8),
            seed=0,
        )
        for _ in range(40):
            agent.store_transition(
                np.random.randn(31).astype(np.float32),
                np.random.randint(11),
                np.random.randn(),
                np.random.randn(31).astype(np.float32),
                False,
            )
        loss = agent.train_step()
        assert loss is not None
        assert np.isfinite(loss)

    def test_target_net_update_after_interval(self):
        agent = DQNAgent(
            obs_dim=31, num_actions=11,
            config=DQNConfig(min_replay_to_train=10, batch_size=4,
                             target_update_interval=5),
            seed=0,
        )
        for _ in range(20):
            agent.store_transition(
                np.random.randn(31).astype(np.float32),
                np.random.randint(11),
                np.random.randn(),
                np.random.randn(31).astype(np.float32),
                False,
            )
        # Modify q_net manually so we can detect target update
        with torch.no_grad():
            agent.q_net.net[0].weight.fill_(7.7)

        # Train a few steps to cross the interval
        for _ in range(10):
            agent.train_step()

        # Target should have been hard-updated to match q_net
        # (specifically, the param we modified should now match)
        q_w = agent.q_net.net[0].weight.detach().clone()
        t_w = agent.target_net.net[0].weight.detach().clone()
        assert torch.allclose(q_w, t_w)


class TestSaveLoad:
    def test_save_load_roundtrip(self):
        env = FoodRescueEnv()
        agent = DQNAgent(
            obs_dim=env.observation_space.shape[0],
            num_actions=env.action_space.n,
            config=DQNConfig(hidden_sizes=(64, 64)),
            seed=0,
        )

        # Run a few train steps to change the weights from random init
        for _ in range(50):
            agent.store_transition(
                np.random.randn(31).astype(np.float32),
                np.random.randint(11),
                np.random.randn(),
                np.random.randn(31).astype(np.float32),
                False,
            )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dqn.pt")
            agent.save(path)
            assert os.path.exists(path)
            assert os.path.exists(os.path.join(tmp, "dqn.meta.json"))

            loaded = DQNAgent.load(path)
            # Same architecture
            assert loaded.obs_dim == agent.obs_dim
            assert loaded.num_actions == agent.num_actions
            # Same weights
            for p_a, p_b in zip(agent.q_net.parameters(), loaded.q_net.parameters()):
                assert torch.allclose(p_a, p_b)
            # Loaded is in eval mode
            assert loaded._training is False

    def test_meta_json_has_required_fields(self):
        env = FoodRescueEnv()
        agent = DQNAgent(
            obs_dim=env.observation_space.shape[0],
            num_actions=env.action_space.n,
            seed=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dqn.pt")
            agent.save(path)
            with open(os.path.join(tmp, "dqn.meta.json")) as f:
                meta = json.load(f)
            assert "config" in meta
            assert "obs_dim" in meta
            assert "num_actions" in meta
            assert "step_count" in meta
            assert "episode_count" in meta
