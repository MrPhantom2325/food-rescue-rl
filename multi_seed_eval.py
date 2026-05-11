"""
Multi-seed training and evaluation — RL's analog to cross-validation.

For each tuned config, we train N times with different seeds and aggregate the
eval results. This is what gets us "Excellent" on CO1's cross-validation
requirement: instead of one number per algorithm, we report mean ± std across
5 independent training runs, each evaluated on 5 held-out seeds.

Usage:
    python multi_seed_eval.py --config configs/q_learning_tuned.yaml --n-seeds 5
    python multi_seed_eval.py --config configs/dqn_v1.yaml --n-seeds 5

Output
------
- experiments/multi_seed/<run_id>/seed_<i>.json per training seed
- experiments/multi_seed/<run_id>/summary.json with aggregated stats
- MLflow parent run "multi_seed_<run_id>" with 5 nested training runs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

from configs_loader import ExperimentConfig, load_config
from mlops_tracking import (
    configure_mlflow,
    log_metrics_safe,
    log_params_safe,
)
from train import build_agent, build_env, evaluate, train


def train_and_eval_one_seed(
    cfg: ExperimentConfig,
    train_seed: int,
    eval_seeds: list[int],
) -> dict[str, Any]:
    """Train one policy from scratch with the given seed; return eval metrics."""
    env = build_env(cfg)
    agent = build_agent(cfg, env)

    # Train using the agent-specific loop. We use train_seed for both env reset
    # variation (seed=train_seed+ep) and any agent RNGs already initialized.
    train_history = train(env, agent, cfg.run.num_episodes, train_seed, cfg.run.agent)

    # Evaluate on the held-out eval seeds
    eval_summary = evaluate(env, agent, eval_seeds)

    # Compute final training average (last 50 episodes)
    final_train_rewards = [h["total_reward"] for h in train_history[-50:]]
    final_train_mean = float(np.mean(final_train_rewards)) if final_train_rewards else 0.0
    final_train_std = float(np.std(final_train_rewards)) if final_train_rewards else 0.0

    return {
        "train_seed": train_seed,
        "eval_mean_reward": eval_summary["eval_mean_reward"],
        "eval_std_reward": eval_summary["eval_std_reward"],
        "eval_mean_delivered": eval_summary["eval_mean_delivered"],
        "eval_mean_spoiled": eval_summary["eval_mean_spoiled"],
        "eval_per_seed": eval_summary["per_seed"],
        "final_train_mean_reward": final_train_mean,
        "final_train_std_reward": final_train_std,
        "table_size": getattr(agent, "table_size", lambda: 0)(),
    }


def aggregate_results(per_seed: list[dict]) -> dict[str, Any]:
    """Compute mean ± std across training seeds for every key metric."""
    metric_keys = [
        "eval_mean_reward", "eval_mean_delivered", "eval_mean_spoiled",
        "final_train_mean_reward",
    ]
    out = {
        "n_train_seeds": len(per_seed),
        "train_seeds": [r["train_seed"] for r in per_seed],
    }
    for key in metric_keys:
        values = [r[key] for r in per_seed]
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
        out[f"{key}_min"] = float(np.min(values))
        out[f"{key}_max"] = float(np.max(values))

    # Also flatten all 25 eval episodes (5 seeds × 5 eval seeds) for histogram
    all_eval_rewards = []
    for r in per_seed:
        all_eval_rewards.extend([e["total_reward"] for e in r["eval_per_seed"]])
    out["all_eval_rewards"] = all_eval_rewards
    out["all_eval_mean"] = float(np.mean(all_eval_rewards))
    out["all_eval_std"] = float(np.std(all_eval_rewards))

    return out


def run_multi_seed(
    config_path: str,
    n_seeds: int = 5,
    base_train_seed: int = 42,
) -> dict[str, Any]:
    """Train+eval a config N times with different seeds; return aggregated results."""
    cfg = load_config(config_path)

    train_seeds = [base_train_seed + i for i in range(n_seeds)]
    eval_seeds = cfg.eval.eval_seeds[:cfg.eval.n_episodes]

    print(f"\n{'=' * 70}")
    print(f"Multi-seed eval: {cfg.run.run_id}")
    print(f"  Agent:        {cfg.run.agent}")
    print(f"  Scenario:     {cfg.run.scenario}")
    print(f"  Episodes/run: {cfg.run.num_episodes}")
    print(f"  Train seeds:  {train_seeds}")
    print(f"  Eval seeds:   {eval_seeds}")
    print(f"{'=' * 70}\n")

    configure_mlflow(experiment_name="food_rescue_multi_seed")

    start_time = time.time()
    per_seed_results = []

    with mlflow.start_run(
        run_name=f"multi_seed_{cfg.run.run_id}",
        tags={"agent": cfg.run.agent, "scenario": cfg.run.scenario,
              "config_file": Path(config_path).name},
    ):
        log_params_safe({
            "run_id": cfg.run.run_id,
            "agent": cfg.run.agent,
            "scenario": cfg.run.scenario,
            "num_episodes": cfg.run.num_episodes,
            "n_train_seeds": n_seeds,
            "base_train_seed": base_train_seed,
        })

        for i, train_seed in enumerate(train_seeds):
            print(f"\n[seed {i+1}/{n_seeds}] training with seed={train_seed}")
            with mlflow.start_run(
                run_name=f"seed_{train_seed}",
                nested=True,
            ):
                log_params_safe({"train_seed": train_seed})
                seed_result = train_and_eval_one_seed(cfg, train_seed, eval_seeds)
                log_metrics_safe({
                    "eval_mean_reward": seed_result["eval_mean_reward"],
                    "eval_mean_delivered": seed_result["eval_mean_delivered"],
                    "eval_mean_spoiled": seed_result["eval_mean_spoiled"],
                    "final_train_mean_reward": seed_result["final_train_mean_reward"],
                })
                per_seed_results.append(seed_result)
                print(f"  seed {train_seed}: eval_mean_reward = "
                      f"{seed_result['eval_mean_reward']:+.2f}")

        # Aggregate
        agg = aggregate_results(per_seed_results)
        wall_time = time.time() - start_time
        agg["wall_time_seconds"] = wall_time
        agg["config_path"] = config_path
        agg["run_id"] = cfg.run.run_id

        # Log aggregated metrics to parent run
        log_metrics_safe({
            "agg_eval_mean_reward": agg["eval_mean_reward_mean"],
            "agg_eval_std_reward": agg["eval_mean_reward_std"],
            "agg_eval_mean_delivered": agg["eval_mean_delivered_mean"],
            "agg_eval_mean_spoiled": agg["eval_mean_spoiled_mean"],
            "agg_eval_min_reward": agg["eval_mean_reward_min"],
            "agg_eval_max_reward": agg["eval_mean_reward_max"],
            "wall_time_seconds": wall_time,
        })

        # Write per-seed and summary JSONs
        out_dir = Path(cfg.run.output_dir) / "multi_seed" / cfg.run.run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, r in enumerate(per_seed_results):
            with open(out_dir / f"seed_{r['train_seed']}.json", "w") as f:
                json.dump(r, f, indent=2, default=str)

        summary_path = out_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(agg, f, indent=2, default=str)

        # Log the summary file as an artifact too
        mlflow.log_artifact(str(summary_path), artifact_path="multi_seed_summary")

        print(f"\n{'=' * 70}")
        print(f"Multi-seed eval complete. Wall time: {wall_time:.1f}s")
        print(f"  Eval mean reward (across {n_seeds} seeds): "
              f"{agg['eval_mean_reward_mean']:+.2f} ± {agg['eval_mean_reward_std']:.2f}")
        print(f"  Range: [{agg['eval_mean_reward_min']:+.2f}, "
              f"{agg['eval_mean_reward_max']:+.2f}]")
        print(f"  Eval mean delivered: {agg['eval_mean_delivered_mean']:.1f} units")
        print(f"  Saved: {summary_path}")
        print(f"{'=' * 70}\n")

    return agg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-seed training and evaluation.")
    parser.add_argument("--config", required=True,
                        help="Path to config YAML (e.g., configs/q_learning_tuned.yaml)")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of independent training seeds (default 5)")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Base training seed; actual seeds are base+0, base+1, ...")
    args = parser.parse_args(argv)

    try:
        run_multi_seed(
            config_path=args.config,
            n_seeds=args.n_seeds,
            base_train_seed=args.base_seed,
        )
        return 0
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
