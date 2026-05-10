
"""Unit tests for sim.environment (Step 7: structure + reset only)."""

import numpy as np
import pytest

from sim.environment import EnvConfig, FoodRescueEnv


# -----------------------------
# Construction & spaces
# -----------------------------

class TestEnvConstruction:
    def test_default_config(self):
        env = FoodRescueEnv()
        assert env.config.scenario_name == "weekday"
        assert env.num_donors == 5
        assert env.num_shelters == 5
        assert env.num_vehicles == 2
        assert env.grid_size == 10

    def test_custom_scenario(self):
        env = FoodRescueEnv(EnvConfig(scenario_name="holiday_rush"))
        assert env.config.scenario_name == "holiday_rush"
        assert env.num_donors == 5
        assert env.num_shelters == 5

    def test_action_space_size(self):
        env = FoodRescueEnv()
        # 5 donors + 5 shelters + 1 idle = 11
        assert env.action_space.n == 11

    def test_observation_space_shape(self):
        env = FoodRescueEnv()
        # 4 vehicle + 3*5 donors + 2*5 shelters + 2 (time, vehicle_idx) = 31
        assert env.observation_space.shape == (31,)
        assert env.observation_space.dtype == np.float32


# -----------------------------
# Action decoding
# -----------------------------

class TestActionDecoding:
    def test_donor_actions(self):
        env = FoodRescueEnv()
        for i in range(env.num_donors):
            kind, idx = env._decode_action(i)
            assert kind == "donor"
            assert idx == i

    def test_shelter_actions(self):
        env = FoodRescueEnv()
        for j in range(env.num_shelters):
            kind, idx = env._decode_action(env.num_donors + j)
            assert kind == "shelter"
            assert idx == j

    def test_idle_action(self):
        env = FoodRescueEnv()
        kind, idx = env._decode_action(env.num_donors + env.num_shelters)
        assert kind == "idle"
        assert idx is None

    def test_out_of_range_raises(self):
        env = FoodRescueEnv()
        with pytest.raises(ValueError):
            env._decode_action(-1)
        with pytest.raises(ValueError):
            env._decode_action(env.action_space.n)


# -----------------------------
# Reset
# -----------------------------

class TestReset:
    def test_reset_returns_obs_and_info(self):
        env = FoodRescueEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert "scenario" in info
        assert info["scenario"] == "weekday"
        assert info["step"] == 0

    def test_reset_initializes_state(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        assert env.scenario is not None
        assert len(env.vehicles) == env.num_vehicles
        assert env.current_step == 0
        assert env.current_vehicle_idx == 0
        assert env.batches == []
        assert env.next_batch_id == 0

    def test_reset_metrics_zeroed(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        m = env._episode_metrics
        assert m["total_generated"] == 0
        assert m["total_delivered_units"] == 0.0
        assert m["total_spoiled_units"] == 0.0
        assert m["total_distance"] == 0

    def test_reset_deterministic_with_same_seed(self):
        env = FoodRescueEnv()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_creates_fresh_entities(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        donors_a = env.scenario.donors
        env.reset(seed=42)
        donors_b = env.scenario.donors
        # Same logical donors, but distinct objects (no state leakage between episodes)
        assert donors_a[0].donor_id == donors_b[0].donor_id
        assert donors_a[0] is not donors_b[0]


# -----------------------------
# Observation
# -----------------------------

class TestObservation:
    def test_obs_in_box_bounds(self):
        env = FoodRescueEnv()
        obs, _ = env.reset(seed=42)
        # Check every value is within the box bounds
        assert np.all(obs >= env.observation_space.low)
        assert np.all(obs <= env.observation_space.high)

    def test_obs_layout_vehicle_at_center(self):
        # Default start strategy is 'center', so vehicle is at (5, 5) on a 10-grid.
        # Normalized: 0.5, 0.5
        env = FoodRescueEnv()
        obs, _ = env.reset(seed=42)
        assert obs[0] == pytest.approx(0.5)  # vehicle x
        assert obs[1] == pytest.approx(0.5)  # vehicle y
        assert obs[2] == 0.0                  # load
        assert obs[3] == 1.0                  # idle flag

    def test_normalized_time_starts_at_zero(self):
        env = FoodRescueEnv()
        obs, _ = env.reset(seed=42)
        # Normalized time is the second-to-last feature
        assert obs[-2] == pytest.approx(0.0)


# -----------------------------
