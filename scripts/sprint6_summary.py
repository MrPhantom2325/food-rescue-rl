"""
Generate Sprint 6's headline comparison artifact.

Pulls all multi-seed eval summaries from experiments/multi_seed/, combines them
with the baseline (random, greedy) results from the Sprint 5 MLflow runs, and
produces:

1. A markdown table comparing all approaches with mean ± std
2. A matplotlib comparison plot (bar chart with error bars)
3. A CSV with the same data for further analysis

Usage:
    python scripts/sprint6_summary.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from mlops_tracking import configure_mlflow


MULTI_SEED_DIR = Path("experiments/multi_seed")
OUTPUT_DIR = Path("experiments/figures")


def load_multi_seed_summaries() -> list[dict]:
    """Load all summary.json files from experiments/multi_seed/."""
    summaries = []
    if not MULTI_SEED_DIR.exists():
        return summaries

    for run_dir in sorted(MULTI_SEED_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            data = json.load(f)
        data["source"] = "multi_seed"
        summaries.append(data)
    return summaries


def fetch_baseline_from_mlflow(experiment_name: str = "food_rescue_rl") -> list[dict]:
    """
    Fetch greedy and random baseline results from the Sprint 5 MLflow experiment.
    """
    configure_mlflow(experiment_name=experiment_name)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.eval_mean_reward DESC"],
    )

    baselines = []
    for r in runs:
        agent = r.data.tags.get("agent", "")
        if agent not in {"random", "greedy"}:
            continue
        baselines.append({
            "run_id": r.data.tags.get("mlflow.runName", "?"),
            "agent": agent,
            "scenario": r.data.tags.get("scenario", "?"),
            "eval_mean_reward_mean": r.data.metrics.get("eval_mean_reward", 0.0),
            "eval_mean_reward_std": r.data.metrics.get("eval_std_reward", 0.0),
            "eval_mean_delivered_mean": r.data.metrics.get("eval_mean_delivered", 0.0),
            "eval_mean_spoiled_mean": r.data.metrics.get("eval_mean_spoiled", 0.0),
            "n_train_seeds": 1,
            "source": "mlflow_baseline",
        })
    return baselines


def build_combined_table(multi_seed: list[dict], baselines: list[dict]) -> list[dict]:
    """Normalize records from both sources into one flat list for table/plot."""
    rows = []

    for ms in multi_seed:
        rows.append({
            "run_id": ms.get("run_id", "?"),
            "scenario": "(varies)",
            "n_seeds": ms["n_train_seeds"],
            "reward_mean": ms["eval_mean_reward_mean"],
            "reward_std": ms["eval_mean_reward_std"],
            "delivered_mean": ms["eval_mean_delivered_mean"],
            "spoiled_mean": ms["eval_mean_spoiled_mean"],
            "source": "multi-seed (5)",
        })

    for b in baselines:
        rows.append({
            "run_id": b["run_id"],
            "scenario": b["scenario"],
            "n_seeds": 1,
            "reward_mean": b["eval_mean_reward_mean"],
            "reward_std": b["eval_mean_reward_std"],
            "delivered_mean": b["eval_mean_delivered_mean"],
            "spoiled_mean": b["eval_mean_spoiled_mean"],
            "source": "single-seed baseline",
        })

    rows.sort(key=lambda r: r["reward_mean"], reverse=True)
    return rows


def write_markdown_table(rows: list[dict], out_path: Path) -> None:
    """Write a human-readable markdown comparison table."""
    lines = [
        "# Food Rescue RL — Comparison",
        "",
        "All learning methods are evaluated across multiple training seeds, then on"
        " 5 held-out eval seeds per training run. Baselines are reported from"
        " single training runs (no training randomness for greedy/random).",
        "",
        "| Method | Scenario | Eval Reward (mean ± std) | Delivered | Spoiled | Source |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        reward_str = f"{r['reward_mean']:+.1f} ± {r['reward_std']:.1f}"
        lines.append(
            f"| `{r['run_id']}` | {r['scenario']} | {reward_str} | "
            f"{r['delivered_mean']:.1f} | {r['spoiled_mean']:.1f} | "
            f"{r['source']} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote: {out_path}")


def write_csv(rows: list[dict], out_path: Path) -> None:
    """Write the same data as CSV for downstream analysis."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {out_path}")


def plot_comparison(rows: list[dict], out_path: Path) -> None:
    """Bar chart with error bars showing eval reward per method."""
    if not rows:
        print("No rows to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    labels = [r["run_id"] for r in rows]
    means = [r["reward_mean"] for r in rows]
    stds = [r["reward_std"] for r in rows]

    colors = [
        "#94a3b8" if r["source"] == "single-seed baseline" else "#22c55e"
        for r in rows
    ]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor="#e2e8f0", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", color="#e2e8f0", fontsize=9)
    ax.set_ylabel("Eval Mean Reward", color="#e2e8f0", fontsize=11)
    ax.set_title("Food Rescue RL — Methods Comparison",
                 color="#e2e8f0", fontsize=13, pad=15)

    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#1e293b")

    ax.axhline(y=0, color="#64748b", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.grid(axis="y", color="#1e293b", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(abs(min(means)), abs(max(means))) * 0.03),
            f"{mean_val:+.0f}",
            ha="center", color="#e2e8f0", fontsize=8, fontweight="bold",
        )

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#22c55e", edgecolor="#e2e8f0",
              label="Learned policy (multi-seed)"),
        Patch(facecolor="#94a3b8", edgecolor="#e2e8f0",
              label="Heuristic baseline (single-seed)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right",
              facecolor="#1e293b", edgecolor="#1e293b",
              labelcolor="#e2e8f0", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote: {out_path}")


def main() -> int:
    print("Loading multi-seed summaries...")
    multi_seed = load_multi_seed_summaries()
    print(f"  Found {len(multi_seed)} multi-seed eval runs")

    print("Loading baselines from MLflow...")
    baselines = fetch_baseline_from_mlflow()
    print(f"  Found {len(baselines)} baseline runs (greedy/random)")

    if not multi_seed and not baselines:
        print("No data to summarize. Run multi_seed_eval.py and train.py first.",
              file=sys.stderr)
        return 1

    rows = build_combined_table(multi_seed, baselines)

    print("\nComparison:")
    for r in rows:
        print(f"  {r['run_id']:<30} reward={r['reward_mean']:+8.1f} ± "
              f"{r['reward_std']:.1f}  delivered={r['delivered_mean']:.1f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_markdown_table(rows, OUTPUT_DIR / "sprint6_comparison.md")
    write_csv(rows, OUTPUT_DIR / "sprint6_comparison.csv")
    plot_comparison(rows, OUTPUT_DIR / "sprint6_comparison.png")

    print("\nSprint 6 summary artifacts generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
