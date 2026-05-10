"""Unit tests for baseline policies."""

import numpy as np
import pytest

from sim.environment import FoodRescueEnv
from agents.baseline import GreedyPolicy, RandomPolicy


# -----------------------------
# RandomPolicy
# -----------------------------

class TestRandomPolicy:
    def test_select_action_in_range(self):
        env = FoodRescueEnv()
        obs, _ = env.reset(seed=42)
        policy = RandomPolicy(seed=0)
        for _ in range(100):
            a = policy.select_action(env, obs)
            assert 0 <= a < env.action_space.n

    def test_seed_reproducibility(self):
        env = FoodRescueEnv()
        obs, _ = env.reset(seed=42)
        p1 = RandomPolicy(seed=99)
        p2 = RandomPolicy(seed=99)
        actions1 = [p1.select_action(env, obs) for _ in range(50)]
        actions2 = [p2.select_action(env, obs) for _ in range(50)]
        assert actions1 == actions2


# -----------------------------
# GreedyPolicy
# -----------------------------

class TestGreedyPolicyLogic:
    """White-box tests of the heuristic decision rules."""

    def test_idle_when_empty_and_no_donors_have_food(self):
        env = FoodRescueEnv()
        obs, _ = env.reset(seed=42)
        # Force all donors empty
        for d in env.scenario.donors:
            d.pending_batches = []
        # Vehicle 0 has no cargo (just reset)
        policy = GreedyPolicy()
        a = policy.select_action(env, obs)
        idle_action = env.num_donors + env.num_shelters
        assert a == idle_action

    def test_idle_when_loaded_and_no_demand(self):
        env = FoodRescueEnv()
        obs, _ = env.reset(seed=42)
        # Manually load vehicle 0
        from sim.entities import BatchStatus, FoodBatch
        v = env.vehicles[0]
        v.cargo.append(
            FoodBatch(batch_id=999, quantity=5, shelf_life=20,
                      origin_donor_id="D001", status=BatchStatus.IN_VEHICLE)
        )
        # Zero out shelter demand
        for s in env.scenario.shelters:
            s.current_demand = 0.0
        policy = GreedyPolicy()
        a = policy.select_action(env, obs)
        idle_action = env.num_donors + env.num_shelters
        assert a == idle_action

    def test_picks_donor_with_pending_food(self):
        env = FoodRescueEnv()
        obs, _ = env.reset(seed=42)
        # Empty all donors, then put food only in donor 2
        for d in env.scenario.donors:
            d.pending_batches = []
        from sim.entities import FoodBatch
        env.scenario.donors[2].pending_batches.append(
            FoodBatch(batch_id=1, quantity=10, shelf_life=30, origin_donor_id="D003")
        )
        policy = GreedyPolicy()
        a = policy.select_action(env, obs)
        assert a == 2  # donor index 2 → action 2

    def test_picks_shelter_with_demand(self):
        env = FoodRescueEnv()
        obs, _ = env.reset(seed=42)
        # Load vehicle 0
        from sim.entities import BatchStatus, FoodBatch
        v = env.vehicles[0]
        v.cargo.append(
            FoodBatch(batch_id=1, quantity=10, shelf_life=20,
                      origin_donor_id="D001", status=BatchStatus.IN_VEHICLE)
        )
        # Zero all shelters except shelter 3
        for j, s in enumerate(env.scenario.shelters):
            s.current_demand = 0.0
            s.priority = 2  # neutralize priority effect for this test
        env.scenario.shelters[3].current_demand = 50.0
        env.scenario.shelters[3].priority = 1
        policy = GreedyPolicy()
        a = policy.select_action(env, obs)
        assert a == env.num_donors + 3  # shelter index 3

    def test_priority_shelter_preferred(self):
        env = FoodRescueEnv()
        obs, _ = env.reset(seed=42)
        from sim.entities import BatchStatus, FoodBatch
        v = env.vehicles[0]
        v.cargo.append(
            FoodBatch(batch_id=1, quantity=10, shelf_life=20,
                      origin_donor_id="D001", status=BatchStatus.IN_VEHICLE)
        )
        # Two shelters at the SAME distance (ish): same demand but different priority
        # Make shelter 0 priority 1, shelter 4 priority 2, both with demand 30
        for s in env.scenario.shelters:
            s.current_demand = 0.0
        env.scenario.shelters[0].current_demand = 30.0
        env.scenario.shelters[0].priority = 1
        env.scenario.shelters[4].current_demand = 30.0
        env.scenario.shelters[4].priority = 2

        # Manually override locations to be equidistant from vehicle
        v.location = (5, 5)
        env.scenario.shelters[0].location = (3, 5)  # distance 2
        env.scenario.shelters[4].location = (7, 5)  # distance 2

        policy = GreedyPolicy()
        a = policy.select_action(env, obs)
        # Should pick shelter 0 (priority 1) over shelter 4 (priority 2)
        assert a == env.num_donors + 0


class TestGreedyVsRandomFullEpisode:
    """The acid test: greedy beats random by a wide margin on full episodes."""

    @pytest.mark.parametrize("scenario", ["weekday", "weekend", "holiday_rush"])
    def test_greedy_beats_random(self, scenario):
        from sim.environment import EnvConfig

        def run_episode(policy, scenario_name, seed):
            env = FoodRescueEnv(EnvConfig(scenario_name=scenario_name))
            obs, _ = env.reset(seed=seed)
            policy.reset()
            total_reward = 0.0
            while True:
                a = policy.select_action(env, obs)
                obs, r, term, trunc, _ = env.step(a)
                total_reward += r
                if term or trunc:
                    break
            return total_reward, env._episode_metrics

        random_reward, random_m = run_episode(RandomPolicy(seed=0), scenario, seed=42)
        greedy_reward, greedy_m = run_episode(GreedyPolicy(), scenario, seed=42)

        print(
            f"\n[{scenario}] random={random_reward:+.2f} (delivered {random_m['total_delivered_units']:.0f}), "
            f"greedy={greedy_reward:+.2f} (delivered {greedy_m['total_delivered_units']:.0f})"
        )

        # Greedy should beat random by a meaningful margin
        assert greedy_reward > random_reward, (
            f"Greedy ({greedy_reward}) failed to beat Random ({random_reward}) "
            f"on scenario {scenario}. Reward function or env logic is suspect."
        )

        # Greedy should deliver substantially more food
        assert greedy_m["total_delivered_units"] > random_m["total_delivered_units"], (
            f"Greedy delivered fewer units than Random on {scenario}"
        )

        # Greedy should let less food spoil (proportionally)
        # Allowing some slack since spoilage is partly driven by env dynamics
        # and the random policy may stumble into pickups
        gen_random = max(1, random_m["total_generated"])
        gen_greedy = max(1, greedy_m["total_generated"])
        spoil_rate_random = random_m["total_spoiled_units"] / gen_random
        spoil_rate_greedy = greedy_m["total_spoiled_units"] / gen_greedy
        # Greedy spoilage rate should be lower
        assert spoil_rate_greedy < spoil_rate_random, (
            f"Greedy spoilage rate ({spoil_rate_greedy:.2f}) ≥ "
            f"Random spoilage rate ({spoil_rate_random:.2f}) on {scenario}"
        )
