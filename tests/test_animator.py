"""Tests for sim.render.EpisodeAnimator."""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import pytest

from agents.baseline import GreedyPolicy, RandomPolicy
from sim.environment import FoodRescueEnv
from sim.render import EpisodeAnimator


class TestEpisodeAnimatorRecord:
    def test_record_full_episode(self):
        env = FoodRescueEnv()
        animator = EpisodeAnimator(env, GreedyPolicy())
        animator.record(seed=42)
        # 200 steps + 1 initial snapshot
        assert len(animator._snapshots) == env.max_episode_steps + 1

    def test_record_max_steps_truncates(self):
        env = FoodRescueEnv()
        animator = EpisodeAnimator(env, GreedyPolicy())
        animator.record(seed=42, max_steps=30)
        # 30 steps + 1 initial snapshot
        assert len(animator._snapshots) == 31

    def test_snapshot_contains_required_keys(self):
        env = FoodRescueEnv()
        animator = EpisodeAnimator(env, GreedyPolicy())
        animator.record(seed=42, max_steps=10)
        snap = animator._snapshots[0]
        for key in ["step", "current_vehicle_idx", "reward", "total_reward",
                    "donors", "shelters", "vehicles", "episode_metrics"]:
            assert key in snap

    def test_snapshot_donor_state(self):
        env = FoodRescueEnv()
        animator = EpisodeAnimator(env, GreedyPolicy())
        animator.record(seed=42, max_steps=10)
        snap = animator._snapshots[5]
        assert len(snap["donors"]) == env.num_donors
        for d in snap["donors"]:
            assert "donor_id" in d
            assert "location" in d
            assert "total_pending_quantity" in d


class TestEpisodeAnimatorSave:
    @pytest.mark.skipif(
        not pytest.importorskip("matplotlib.animation").FFMpegWriter.isAvailable(),
        reason="ffmpeg not available",
    )
    def test_save_mp4(self):
        env = FoodRescueEnv()
        animator = EpisodeAnimator(env, RandomPolicy(seed=0))
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.mp4")
            animator.save_mp4(out, seed=42, fps=8, max_steps=20)
            assert os.path.exists(out)
            assert os.path.getsize(out) > 5000  # non-trivial MP4

    def test_save_gif(self):
        env = FoodRescueEnv()
        animator = EpisodeAnimator(env, RandomPolicy(seed=0))
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.gif")
            animator.save_gif(out, seed=42, fps=6, max_steps=20)
            assert os.path.exists(out)
            assert os.path.getsize(out) > 5000  # non-trivial GIF


class TestSnapshotIndependence:
    def test_snapshots_dont_share_state_with_env(self):
        """After recording, modifying env should NOT affect snapshots."""
        env = FoodRescueEnv()
        animator = EpisodeAnimator(env, GreedyPolicy())
        animator.record(seed=42, max_steps=10)
        snap5 = animator._snapshots[5]
        original_qty = snap5["donors"][0]["total_pending_quantity"]

        # Mess with the env after recording
        env.scenario.donors[0].pending_batches = []

        # Snapshot should be unchanged
        assert animator._snapshots[5]["donors"][0]["total_pending_quantity"] == original_qty
