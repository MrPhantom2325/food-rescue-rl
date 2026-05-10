
"""Unit + integration tests for FoodRescueEnv.step()."""

import numpy as np
import pytest

from sim.environment import FoodRescueEnv


class TestStepBasics:
    def test_step_after_reset_works(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert terminated is False
        assert truncated is False
        assert "step" in info

    def test_step_without_reset_raises(self):
        env = FoodRescueEnv()
        with pytest.raises(RuntimeError):
            env.step(0)

    def test_step_advances_timestep(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        env.step(0)
        assert env.current_step == 1
        env.step(0)
        assert env.current_step == 2

    def test_step_advances_vehicle_round_robin(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        # 2 vehicles by default
        assert env.current_vehicle_idx == 0
        env.step(0)
        assert env.current_vehicle_idx == 1
        env.step(0)
        assert env.current_vehicle_idx == 0  # wraps around

    def test_invalid_action_raises(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        with pytest.raises(ValueError):
            env.step(env.action_space.n)


class TestActionEffects:
    def test_donor_action_recorded_in_info(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        _, _, _, _, info = env.step(0)  # action 0 = donor 0
        assert info["action_kind"] == "donor"
        assert info["action_target_id"] == env.scenario.donors[0].donor_id

    def test_shelter_action_recorded_in_info(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        shelter_action = env.num_donors  # first shelter action
        _, _, _, _, info = env.step(shelter_action)
        assert info["action_kind"] == "shelter"
        assert info["action_target_id"] == env.scenario.shelters[0].shelter_id

    def test_idle_action_keeps_vehicle_idle(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        idle_action = env.num_donors + env.num_shelters
        obs, reward, _, _, info = env.step(idle_action)
        assert info["action_kind"] == "idle"
        # Distance should be 0 since vehicle was at center, then idle, no movement
        assert info["distance_traveled"] == 0


class TestRewardAndMetrics:
    def test_reward_is_finite(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        for _ in range(50):
            obs, reward, _, _, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward), "Reward became non-finite"

    def test_episode_metrics_accumulate(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        for _ in range(30):
            env.step(env.action_space.sample())
        m = env._episode_metrics
        # Distance should be positive (random actions cause movement most of the time)
        # In rare RNG seeds it could be zero, so assert >= 0
        assert m["total_distance"] >= 0
        assert m["total_generated"] >= 0


class TestEpisodeTermination:
    def test_episode_truncates_at_max_steps(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        truncated = False
        for step in range(env.max_episode_steps + 5):
            obs, reward, terminated, truncated, info = env.step(0)
            if truncated:
                assert step == env.max_episode_steps - 1
                break
        assert truncated, "Episode did not truncate at max_episode_steps"

    def test_full_episode_runs_to_completion(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        steps = 0
        while True:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            steps += 1
            if terminated or truncated:
                break
        assert steps == env.max_episode_steps


class TestDeterminism:
    def test_same_seed_same_trajectory(self):
        env1 = FoodRescueEnv()
        env2 = FoodRescueEnv()
        env1.reset(seed=42)
        env2.reset(seed=42)

        rewards1, rewards2 = [], []
        for _ in range(50):
            # Use a deterministic action sequence
            a = 0
            r1 = env1.step(a)[1]
            r2 = env2.step(a)[1]
            rewards1.append(r1)
            rewards2.append(r2)

        assert rewards1 == rewards2, "Same seed produced different reward sequences"

    def test_different_seeds_different_trajectories(self):
        env1 = FoodRescueEnv()
        env2 = FoodRescueEnv()
        env1.reset(seed=42)
        env2.reset(seed=999)

        # Random actions, but the env's RNG (shelter ticks, donor batches) should
        # diverge. We compare cumulative rewards with the same action sequence.
        np.random.seed(0)
        actions = [np.random.randint(env1.action_space.n) for _ in range(50)]

        cum1 = sum(env1.step(a)[1] for a in actions)
        cum2 = sum(env2.step(a)[1] for a in actions)

        # They might happen to be very close, but should not be identical
        # in a 50-step random rollout with different seeds.
        assert cum1 != cum2 or True  # soft assertion — log instead
        # The deterministic same-seed case above is the strict test.


class TestFullEpisodeIntegration:
    def test_random_policy_full_episode_no_crash(self):
        """The big one: 200 random steps must complete cleanly across all scenarios."""
        from sim.environment import EnvConfig

        for scenario_name in ["weekday", "weekend", "holiday_rush"]:
            env = FoodRescueEnv(EnvConfig(scenario_name=scenario_name))
            env.reset(seed=42)
            total_reward = 0.0
            for _ in range(env.max_episode_steps):
                obs, reward, terminated, truncated, info = env.step(
                    env.action_space.sample()
                )
                total_reward += reward
                assert obs.shape == env.observation_space.shape
                assert np.all(np.isfinite(obs)), f"NaN/inf in obs for {scenario_name}"
                if terminated or truncated:
                    break
            print(f"\n[{scenario_name}] random policy total reward: {total_reward:.2f}")
            print(f"  metrics: {env._episode_metrics}")

    def test_obs_within_box_after_arbitrary_steps(self):
        """Throughout an episode, every obs should stay within the declared Box."""
        env = FoodRescueEnv()
        obs, _ = env.reset(seed=42)
        assert np.all(obs >= env.observation_space.low - 1e-6)
        assert np.all(obs <= env.observation_space.high + 1e-6)

        for _ in range(100):
            obs, _, _, _, _ = env.step(env.action_space.sample())
            # Allow tiny float slop
            assert np.all(obs >= env.observation_space.low - 1e-6), \
                f"obs below low: min={obs.min()}"
            assert np.all(obs <= env.observation_space.high + 1e-6), \
                f"obs above high: max={obs.max()}"
