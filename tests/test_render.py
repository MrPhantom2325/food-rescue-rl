"""Tests for sim.render. We focus on 'doesn't crash' and 'produces a Figure'.

Pixel-level rendering tests would be brittle and slow. The acceptance test for
visualization is human-eyeball: open the saved PNG and look at it. Tests here
catch regressions like "renderer crashes when shelter has zero demand."
"""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from agents.baseline import GreedyPolicy
from sim.environment import EnvConfig, FoodRescueEnv
from sim.render import FrameRenderer, PALETTE


class TestFrameRenderer:
    def test_render_after_reset(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        renderer = FrameRenderer(env)
        fig = renderer.render()
        assert fig is not None
        plt.close(fig)

    def test_render_before_reset_raises(self):
        env = FoodRescueEnv()
        renderer = FrameRenderer(env)
        with pytest.raises(RuntimeError):
            renderer.render()

    def test_render_after_steps(self):
        env = FoodRescueEnv()
        obs, _ = env.reset(seed=42)
        for _ in range(20):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                break
        renderer = FrameRenderer(env)
        fig = renderer.render()
        plt.close(fig)

    def test_render_with_reward_info(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        obs, r, _, _, info = env.step(0)
        renderer = FrameRenderer(env)
        fig = renderer.render(reward=r, total_reward=r, step_info=info)
        plt.close(fig)

    def test_render_save_to_disk(self):
        env = FoodRescueEnv()
        env.reset(seed=42)
        renderer = FrameRenderer(env)
        fig = renderer.render(title_extra="test")
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "test.png")
            fig.savefig(out_path, dpi=72, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 1000  # non-trivially sized PNG
        plt.close(fig)

    def test_render_all_scenarios(self):
        for scenario in ["weekday", "weekend", "holiday_rush"]:
            env = FoodRescueEnv(EnvConfig(scenario_name=scenario))
            env.reset(seed=42)
            renderer = FrameRenderer(env)
            fig = renderer.render()
            plt.close(fig)

    def test_render_with_loaded_vehicle(self):
        """Vehicle with cargo should render in 'loaded' style."""
        env = FoodRescueEnv()
        env.reset(seed=42)
        # Run greedy for enough steps that vehicles pick up food
        policy = GreedyPolicy()
        obs, _ = env.reset(seed=42)
        for _ in range(50):
            a = policy.select_action(env, obs)
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        renderer = FrameRenderer(env)
        fig = renderer.render()
        plt.close(fig)

    def test_render_with_priority_shelters(self):
        """Priority shelters should be present (rendering exercises priority halo)."""
        env = FoodRescueEnv()
        env.reset(seed=42)
        priorities = [s.priority for s in env.scenario.shelters]
        assert 1 in priorities  # at least one priority-1 shelter exists
        renderer = FrameRenderer(env)
        fig = renderer.render()
        plt.close(fig)


class TestPalette:
    def test_palette_has_required_keys(self):
        # Smoke check: palette is a frozen dataclass with sensible color codes
        assert PALETTE.background.startswith("#")
        assert PALETTE.donor.startswith("#")
        assert PALETTE.shelter.startswith("#")
        assert PALETTE.vehicle_loaded.startswith("#")

    def test_palette_is_frozen(self):
        with pytest.raises(Exception):
            PALETTE.donor = "#000000"  # type: ignore[misc]
