"""
Side-by-side comparison: original DQN vs reward-normalized DQN.

Trains both configs with the same seed, runs the same eval seeds against each,
and prints a results table you can drop into the final report.

Usage:
    python scripts/compare_dqn_configs.py            # full (slow, ~30 min)
    python scripts/compare_dqn_configs.py --quick    # 150 episodes each (~3 min)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

# Make the project root importable when run from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import run_experiment  # noqa: E402


CONFIGS = {
    "baseline (dqn_v1)": "configs/dqn_v1.yaml",
    "normalized (dqn_v3)": "configs/dqn_v3_normalized.yaml",
}


def _quick_override(cfg_path: str, out_dir: Path) -> Path:
    """Make a copy of the config with reduced episodes for fast iteration."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["run"]["num_episodes"] = 150
    cfg["run"]["output_dir"] = str(out_dir)
    cfg["run"]["run_id"] = cfg["run"]["run_id"] + "_quick"
    cfg["agent_params"]["epsilon_decay_episodes"] = 100
    cfg["agent_params"]["min_replay_to_train"] = 500
    cfg["eval"]["n_episodes"] = 3
    cfg["eval"]["eval_seeds"] = [100, 101, 102]
    qpath = out_dir / Path(cfg_path).name
    with open(qpath, "w") as f:
        yaml.safe_dump(cfg, f)
    return qpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Use 150 episodes (fast sanity check)")
    ap.add_argument("--out", default="experiments/comparisons", help="Output dir")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, cfg_path in CONFIGS.items():
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
        if args.quick:
            cfg_path = str(_quick_override(cfg_path, out_dir))
        t0 = time.time()
        info = run_experiment(cfg_path)
        elapsed = time.time() - t0
        es = info["eval_summary"]
        results[label] = {
            "config": cfg_path,
            "wall_time_s": round(elapsed, 1),
            "eval_reward_mean": es["eval_mean_reward"],
            "eval_reward_std": es["eval_std_reward"],
            "eval_delivered_mean": es["eval_mean_delivered"],
            "eval_spoiled_mean": es["eval_mean_spoiled"],
        }

    print("\n\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Config':<25} {'Eval reward':>20} {'Delivered':>12} {'Spoiled':>10}")
    print("-" * 70)
    for label, r in results.items():
        m = r["eval_reward_mean"]
        s = r["eval_reward_std"]
        d = r["eval_delivered_mean"]
        sp = r["eval_spoiled_mean"]
        print(f"{label:<25} {m:>10.1f} ± {s:>5.1f}     {d:>10.1f}   {sp:>8.1f}")

    summary_path = out_dir / "comparison.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results: {summary_path}")


if __name__ == "__main__":
    main()