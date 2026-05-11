"""
Optuna-based hyperparameter tuning for the food rescue project.

Usage:
    python tune.py --agent q_learning --n-trials 30 --scenario weekday
    python tune.py --agent sarsa --n-trials 30 --scenario weekday
    python tune.py --agent dqn --n-trials 20 --scenario weekday

Architecture
------------
For each trial:
1. Optuna proposes a config (sampled from the per-agent search space)
2. We instantiate env + agent, train for a reduced number of episodes
3. We evaluate the trained agent on `n_eval_seeds` (default 3) seeds
4. The trial's score = mean eval reward across seeds
5. Optuna receives the score, updates its TPE model, proposes next trial

Each trial is logged as a NESTED MLflow run under a parent study run, so the
MLflow UI shows a clean hierarchy:
    food_rescue_tuning (experiment)
        study_qlearning_20260510 (parent run)
            trial_001 (nested)
            trial_002 (nested)
            ...

After all trials complete, the best params are written to
configs/<agent>_tuned.yaml so multi_seed_eval.py can pick it up directly.

Cost control
------------
Tuning trials use REDUCED episode counts vs the production configs in Sprint 5:
- q_learning / sarsa: 600 episodes per trial (vs 1500 in production)
- dqn: 300 episodes per trial (vs 800 in production)

This is intentional: tuning needs to be cheap. The relative ranking of configs
generally holds at reduced budgets. After tuning, we re-train the winner with
the full episode budget for multi-seed eval (Sprint 6 Step 23).
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
import optuna
import yaml
from optuna.samplers import TPESampler

from agents.q_learning import discretize_state  # noqa: F401 (used indirectly)
from configs_loader import ExperimentConfig, EvalConfig, RunConfig
from mlops_tracking import (
    configure_mlflow,
    log_metrics_safe,
    log_params_safe,
)
from train import build_agent, build_env, evaluate, train


# -----------------------------
# Per-agent search spaces
# -----------------------------

def sample_q_learning_params(trial: optuna.Trial) -> dict[str, Any]:
    """Hyperparameter search space for tabular Q-learning."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "discount": trial.suggest_float("discount", 0.9, 0.99),
        "epsilon_start": 1.0,
        "epsilon_end": trial.suggest_float("epsilon_end", 0.05, 0.25),
        "epsilon_decay_episodes": trial.suggest_int("epsilon_decay_episodes", 400, 1200, step=100),
        "optimistic_init": 0.0,
        "pos_buckets": 3,
        "load_buckets": 3,
    }


def sample_sarsa_params(trial: optuna.Trial) -> dict[str, Any]:
    """SARSA uses the same hyperparameters as Q-learning."""
    return sample_q_learning_params(trial)


def sample_dqn_params(trial: optuna.Trial) -> dict[str, Any]:
    """Hyperparameter search space for DQN."""
    # hidden_sizes is sampled as a discrete choice from a small set
    hidden_choices = [(64, 64), (128, 128), (256, 128), (128, 64)]
    hidden_idx = trial.suggest_categorical("hidden_sizes_idx", list(range(len(hidden_choices))))
    hidden_sizes = hidden_choices[hidden_idx]

    return {
        "hidden_sizes": hidden_sizes,
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "discount": trial.suggest_float("discount", 0.9, 0.99),
        "epsilon_start": 1.0,
        "epsilon_end": trial.suggest_float("epsilon_end", 0.05, 0.2),
        "epsilon_decay_episodes": trial.suggest_int("epsilon_decay_episodes", 300, 700, step=50),
        "replay_buffer_size": 50_000,
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "min_replay_to_train": 1_000,
        "target_update_interval": trial.suggest_int("target_update_interval", 200, 1000, step=100),
        "grad_clip": 1.0,
        "device": "auto",
    }


SEARCH_SPACES = {
    "q_learning": sample_q_learning_params,
    "sarsa": sample_sarsa_params,
    "dqn": sample_dqn_params,
}

# Default trial-time episode budgets (reduced from production)
TRIAL_EPISODES = {
    "q_learning": 600,
    "sarsa": 600,
    "dqn": 300,
}


# -----------------------------
# Default reward weights (held constant across tuning trials)
# -----------------------------

DEFAULT_REWARD_WEIGHTS = {
    "delivery": 10.0,
    "spoilage": 5.0,
    "distance": 0.1,
    "unmet_demand": 1.0,
    "priority_bonus": 0.5,
    "oversupply_penalty": 0.3,
}


# -----------------------------
# Trial execution
# -----------------------------

def run_trial(
    trial: optuna.Trial,
    agent_kind: str,
    scenario: str,
    num_episodes: int,
    n_eval_seeds: int,
    base_seed: int,
) -> float:
    """
    Run one Optuna trial: build agent with sampled params, train, evaluate.

    Returns the mean eval reward (Optuna will maximize this).
    """
    sampler_fn = SEARCH_SPACES[agent_kind]
    agent_params = sampler_fn(trial)

    # Build a temporary ExperimentConfig
    run_cfg = RunConfig(
        run_id=f"trial_{trial.number:03d}",
        agent=agent_kind,
        scenario=scenario,
        num_episodes=num_episodes,
        seed=base_seed + trial.number,  # vary seed per trial for diversity
        output_dir="experiments/tuning_tmp",
    )
    eval_cfg = EvalConfig(
        n_episodes=n_eval_seeds,
        eval_seeds=[base_seed + 1000 + i for i in range(n_eval_seeds)],
    )
    cfg = ExperimentConfig(
        run=run_cfg,
        agent_params=agent_params,
        reward_weights=DEFAULT_REWARD_WEIGHTS,
        eval=eval_cfg,
        raw={},
    )

    # Build env + agent
    env = build_env(cfg)
    agent = build_agent(cfg, env)

    # Train (no MLflow per-step logging — too verbose for tuning trials)
    history = train(env, agent, num_episodes, run_cfg.seed, agent_kind)

    # Evaluate
    eval_summary = evaluate(env, agent, eval_cfg.eval_seeds)
    score = eval_summary["eval_mean_reward"]

    # Log this trial as a nested MLflow run
    with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
        # Flatten params for MLflow
        flat = {"trial_number": trial.number, "agent": agent_kind,
                "scenario": scenario, "num_episodes": num_episodes}
        for k, v in agent_params.items():
            flat[f"agent.{k}"] = v
        log_params_safe(flat)

        log_metrics_safe({
            "eval_mean_reward": eval_summary["eval_mean_reward"],
            "eval_std_reward": eval_summary["eval_std_reward"],
            "eval_mean_delivered": eval_summary["eval_mean_delivered"],
            "eval_mean_spoiled": eval_summary["eval_mean_spoiled"],
            "train_final_reward": float(np.mean([h["total_reward"] for h in history[-50:]])),
        })

    # Report to Optuna (also enables pruning if we add it later)
    trial.report(score, step=0)

    return score


# -----------------------------
# Main: run a study
# -----------------------------

def run_study(
    agent_kind: str,
    n_trials: int,
    scenario: str,
    num_episodes: int | None = None,
    n_eval_seeds: int = 3,
    base_seed: int = 42,
) -> dict[str, Any]:
    """
    Run an Optuna study for one agent type. Returns best params + best score.
    """
    if agent_kind not in SEARCH_SPACES:
        raise ValueError(
            f"Unknown agent kind: {agent_kind}. "
            f"Supported: {sorted(SEARCH_SPACES.keys())}"
        )

    if num_episodes is None:
        num_episodes = TRIAL_EPISODES[agent_kind]

    # Configure MLflow with a study-specific experiment
    configure_mlflow(experiment_name="food_rescue_tuning")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    study_name = f"study_{agent_kind}_{timestamp}"

    print(f"\n{'=' * 70}")
    print(f"Optuna study: {study_name}")
    print(f"  Agent:        {agent_kind}")
    print(f"  Scenario:     {scenario}")
    print(f"  Trials:       {n_trials}")
    print(f"  Episodes/trial: {num_episodes}")
    print(f"  Eval seeds/trial: {n_eval_seeds}")
    print(f"{'=' * 70}\n")

    # Sampler with fixed seed for reproducibility
    sampler = TPESampler(seed=base_seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
    )

    # Parent MLflow run that contains all the nested trial runs
    with mlflow.start_run(
        run_name=study_name,
        tags={"study_agent": agent_kind, "study_scenario": scenario,
              "n_trials": str(n_trials)},
    ):
        log_params_safe({
            "agent_kind": agent_kind,
            "scenario": scenario,
            "n_trials": n_trials,
            "num_episodes_per_trial": num_episodes,
            "n_eval_seeds_per_trial": n_eval_seeds,
            "base_seed": base_seed,
            "sampler": "TPESampler",
        })

        def objective(trial: optuna.Trial) -> float:
            return run_trial(
                trial,
                agent_kind=agent_kind,
                scenario=scenario,
                num_episodes=num_episodes,
                n_eval_seeds=n_eval_seeds,
                base_seed=base_seed,
            )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Log best results to the parent run
        best_trial = study.best_trial
        log_metrics_safe({
            "best_score": best_trial.value,
            "best_trial_number": best_trial.number,
        })

        # Log the best params with a clear prefix
        best_params_flat = {f"best.{k}": v for k, v in best_trial.params.items()}
        log_params_safe(best_params_flat)

    print(f"\n{'=' * 70}")
    print("Study complete.")
    print(f"  Best trial:   #{best_trial.number}")
    print(f"  Best score:   {best_trial.value:+.2f}")
    print(f"  Best params:  {json.dumps(best_trial.params, indent=2)}")
    print(f"{'=' * 70}\n")

    # Write best params to a YAML for downstream use (multi_seed_eval, etc.)
    save_best_config(agent_kind, scenario, best_trial, study, num_episodes)

    return {
        "study_name": study_name,
        "best_trial_number": best_trial.number,
        "best_score": float(best_trial.value),
        "best_params": dict(best_trial.params),
    }


def save_best_config(
    agent_kind: str,
    scenario: str,
    best_trial: optuna.Trial,
    study: optuna.Study,
    num_episodes: int,
) -> None:
    """
    Write configs/<agent>_tuned.yaml with the best params from this study.

    The tuned config uses the FULL production episode budget — we tuned at a
    reduced budget to be fast, but the deployment config trains the winner
    properly.
    """
    # Map the trial params back into the agent_params shape that ExperimentConfig wants
    # We need to handle dqn's hidden_sizes_idx -> hidden_sizes specially.
    raw_params = dict(best_trial.params)

    if agent_kind == "dqn":
        hidden_choices = [(64, 64), (128, 128), (256, 128), (128, 64)]
        hidden_sizes_idx = raw_params.pop("hidden_sizes_idx", 1)
        hidden_sizes = list(hidden_choices[hidden_sizes_idx])

        # Fill in non-tuned defaults
        agent_params = {
            "hidden_sizes": hidden_sizes,
            "learning_rate": raw_params["learning_rate"],
            "discount": raw_params["discount"],
            "epsilon_start": 1.0,
            "epsilon_end": raw_params["epsilon_end"],
            "epsilon_decay_episodes": raw_params["epsilon_decay_episodes"],
            "replay_buffer_size": 50_000,
            "batch_size": raw_params["batch_size"],
            "min_replay_to_train": 1_000,
            "target_update_interval": raw_params["target_update_interval"],
            "grad_clip": 1.0,
            "device": "auto",
        }
        production_episodes = 800
    else:
        # q_learning / sarsa
        agent_params = {
            "learning_rate": raw_params["learning_rate"],
            "discount": raw_params["discount"],
            "epsilon_start": 1.0,
            "epsilon_end": raw_params["epsilon_end"],
            "epsilon_decay_episodes": raw_params["epsilon_decay_episodes"],
            "optimistic_init": 0.0,
            "pos_buckets": 3,
            "load_buckets": 3,
        }
        production_episodes = 1500

    out = {
        "run": {
            "run_id": f"{agent_kind}_tuned",
            "agent": agent_kind,
            "scenario": scenario,
            "num_episodes": production_episodes,
            "seed": 42,
            "output_dir": "experiments",
            "description": (
                f"Optuna-tuned hyperparameters for {agent_kind} on {scenario}. "
                f"Best of {len(study.trials)} trials, score = {best_trial.value:+.2f}."
            ),
        },
        "agent_params": agent_params,
        "reward_weights": DEFAULT_REWARD_WEIGHTS,
        "eval": {
            "n_episodes": 5,
            "eval_seeds": [100, 101, 102, 103, 104],
        },
    }

    out_path = Path("configs") / f"{agent_kind}_tuned.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f"Tuned config saved to: {out_path}")


# -----------------------------
# CLI
# -----------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an Optuna hyperparameter study.")
    parser.add_argument("--agent", required=True,
                        choices=sorted(SEARCH_SPACES.keys()),
                        help="Which agent to tune.")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Number of Optuna trials.")
    parser.add_argument("--scenario", default="weekday",
                        help="Scenario name from data/processed/.")
    parser.add_argument("--num-episodes", type=int, default=None,
                        help="Override episodes per trial (default: agent-specific).")
    parser.add_argument("--n-eval-seeds", type=int, default=3,
                        help="Seeds to average over for each trial's score.")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Base seed for sampler + train/eval.")
    args = parser.parse_args(argv)

    try:
        result = run_study(
            agent_kind=args.agent,
            n_trials=args.n_trials,
            scenario=args.scenario,
            num_episodes=args.num_episodes,
            n_eval_seeds=args.n_eval_seeds,
            base_seed=args.base_seed,
        )
        print(f"\nResult: {json.dumps(result, indent=2)}")
        return 0
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
