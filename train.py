"""
Unified training entry point for the food rescue project.

Usage:
    python train.py --config configs/qlearning_v1.yaml

The training loop:
1. Load config and configure MLflow
2. Build env (with reward weights from config) and agent (from agent_params)
3. For each episode:
   - reset env, run until termination, collect transitions
   - update agent (algorithm-specific)
   - log per-episode metrics to MLflow + CSV
4. Save policy artifact, log it as MLflow artifact
5. Run evaluation episodes with the trained policy, log eval metrics
6. Write run metadata JSON

Supports 5 agents:
  - random / greedy: no learning, no policy artifact
  - q_learning / sarsa: tabular Bellman updates
  - dqn: replay buffer + neural net training
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from agents.baseline import GreedyPolicy, RandomPolicy
from agents.dqn import DQNAgent, DQNConfig
from agents.q_learning import QLearningAgent, QLearningConfig, discretize_state
from agents.sarsa import SARSAAgent
from configs_loader import ExperimentConfig, load_config
from mlops_tracking import (
    configure_mlflow,
    log_metric_safe,
    log_metrics_safe,
    log_params_safe,
    start_run,
)
from sim.environment import EnvConfig, FoodRescueEnv, RewardWeights

import mlflow


# -----------------------------
# Agent factory
# -----------------------------

def build_agent(cfg: ExperimentConfig, env: FoodRescueEnv):
    """Construct the agent type specified by the config."""
    name = cfg.run.agent
    params = cfg.agent_params
    seed = cfg.run.seed

    if name == "random":
        return RandomPolicy(seed=seed)

    if name == "greedy":
        return GreedyPolicy(**params)

    if name == "q_learning":
        return QLearningAgent(
            num_actions=int(env.action_space.n),
            config=QLearningConfig(**params),
            seed=seed,
        )

    if name == "sarsa":
        return SARSAAgent(
            num_actions=int(env.action_space.n),
            config=QLearningConfig(**params),
            seed=seed,
        )

    if name == "dqn":
        # hidden_sizes might come in as a list from YAML; ensure tuple
        params = dict(params)
        if "hidden_sizes" in params and isinstance(params["hidden_sizes"], list):
            params["hidden_sizes"] = tuple(params["hidden_sizes"])
        return DQNAgent(
            obs_dim=int(env.observation_space.shape[0]),
            num_actions=int(env.action_space.n),
            config=DQNConfig(**params),
            seed=seed,
        )

    raise ValueError(f"Unknown agent: {name}")


def build_env(cfg: ExperimentConfig) -> FoodRescueEnv:
    """Construct the environment with reward weights from the config."""
    rw = RewardWeights(**cfg.reward_weights) if cfg.reward_weights else RewardWeights()
    env_config = EnvConfig(
        scenario_name=cfg.run.scenario,
        reward_weights=rw,
    )
    return FoodRescueEnv(config=env_config)


# -----------------------------
# Training loops (algorithm-specific)
# -----------------------------

def train_qlearning(env: FoodRescueEnv, agent, num_episodes: int, seed: int) -> list[dict]:
    """Train Q-learning. Returns list of per-episode metric dicts."""
    history = []
    for ep in tqdm(range(num_episodes), desc="qlearning", leave=False):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        steps = 0

        while True:
            state_before = discretize_state(env)
            action = agent.select_action(env, obs)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            total_reward += reward
            steps += 1

            # Q-learning Bellman update with the captured s_before
            if agent._training:
                state_after = discretize_state(env)
                q_row = agent._ensure_q_row(state_before)
                if done:
                    target = reward
                else:
                    q_next = agent._q_table.get(state_after)
                    max_next = float(q_next.max()) if q_next is not None else agent.config.optimistic_init
                    target = reward + agent.config.discount * max_next
                q_row[action] = q_row[action] + agent.config.learning_rate * (target - q_row[action])

            obs = next_obs
            if done:
                break

        agent.end_episode()
        em = env._episode_metrics
        history.append({
            "episode": ep,
            "total_reward": total_reward,
            "steps": steps,
            "epsilon": agent.epsilon(),
            "table_size": agent.table_size(),
            "delivered_units": em["total_delivered_units"],
            "spoiled_units": em["total_spoiled_units"],
            "deliveries_count": em["deliveries_count"],
            "distance": em["total_distance"],
        })
    return history


def train_sarsa(env: FoodRescueEnv, agent, num_episodes: int, seed: int) -> list[dict]:
    """Train SARSA. Same structure as Q-learning but uses next_action in target."""
    history = []
    for ep in tqdm(range(num_episodes), desc="sarsa", leave=False):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        steps = 0

        state_before = discretize_state(env)
        action = agent.select_action(env, obs)

        while True:
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            total_reward += reward
            steps += 1

            state_after = discretize_state(env)

            if done:
                # Terminal update: target = reward only
                if agent._training:
                    q_row = agent._ensure_q_row(state_before)
                    q_row[action] = q_row[action] + agent.config.learning_rate * (
                        reward - q_row[action]
                    )
                break

            next_action = agent.select_action(env, next_obs)
            if agent._training:
                q_row = agent._ensure_q_row(state_before)
                q_next_row = agent._q_table.get(state_after)
                next_value = float(q_next_row[next_action]) if q_next_row is not None else agent.config.optimistic_init
                target = reward + agent.config.discount * next_value
                q_row[action] = q_row[action] + agent.config.learning_rate * (target - q_row[action])

            obs = next_obs
            state_before = state_after
            action = next_action

        agent.end_episode()
        em = env._episode_metrics
        history.append({
            "episode": ep,
            "total_reward": total_reward,
            "steps": steps,
            "epsilon": agent.epsilon(),
            "table_size": agent.table_size(),
            "delivered_units": em["total_delivered_units"],
            "spoiled_units": em["total_spoiled_units"],
            "deliveries_count": em["deliveries_count"],
            "distance": em["total_distance"],
        })
    return history


def train_dqn(env: FoodRescueEnv, agent, num_episodes: int, seed: int) -> list[dict]:
    """Train DQN. Uses store_transition + train_step (gradient steps)."""
    history = []
    for ep in tqdm(range(num_episodes), desc="dqn", leave=False):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        steps = 0
        ep_losses = []

        while True:
            action = agent.select_action(env, obs)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            total_reward += reward
            steps += 1

            agent.store_transition(obs, action, reward, next_obs, float(done))
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            obs = next_obs
            if done:
                break

        agent.end_episode()
        em = env._episode_metrics
        history.append({
            "episode": ep,
            "total_reward": total_reward,
            "steps": steps,
            "epsilon": agent.epsilon(),
            "replay_size": len(agent.replay),
            "mean_loss": float(np.mean(ep_losses)) if ep_losses else 0.0,
            "delivered_units": em["total_delivered_units"],
            "spoiled_units": em["total_spoiled_units"],
            "deliveries_count": em["deliveries_count"],
            "distance": em["total_distance"],
        })
    return history


def run_baseline_episodes(env: FoodRescueEnv, agent, num_episodes: int, seed: int) -> list[dict]:
    """Run a non-learning policy for num_episodes — no updates, just collect metrics."""
    history = []
    for ep in tqdm(range(num_episodes), desc=agent.name, leave=False):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        steps = 0
        while True:
            action = agent.select_action(env, obs)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            steps += 1
            if term or trunc:
                break
        em = env._episode_metrics
        history.append({
            "episode": ep,
            "total_reward": total_reward,
            "steps": steps,
            "delivered_units": em["total_delivered_units"],
            "spoiled_units": em["total_spoiled_units"],
            "deliveries_count": em["deliveries_count"],
            "distance": em["total_distance"],
        })
    return history


def train(env: FoodRescueEnv, agent, num_episodes: int, seed: int, agent_kind: str) -> list[dict]:
    """Dispatch to the correct training function based on agent type."""
    if agent_kind in {"random", "greedy"}:
        return run_baseline_episodes(env, agent, num_episodes, seed)
    if agent_kind == "q_learning":
        return train_qlearning(env, agent, num_episodes, seed)
    if agent_kind == "sarsa":
        return train_sarsa(env, agent, num_episodes, seed)
    if agent_kind == "dqn":
        return train_dqn(env, agent, num_episodes, seed)
    raise ValueError(f"No training loop for agent kind: {agent_kind}")


# -----------------------------
# Evaluation
# -----------------------------

def evaluate(env: FoodRescueEnv, agent, eval_seeds: list[int]) -> dict:
    """
    Evaluate the trained policy in eval mode (no exploration, no updates).

    Returns aggregated metrics across all eval seeds: mean and std of reward,
    delivered, spoiled, etc.
    """
    if hasattr(agent, "set_training"):
        agent.set_training(False)

    per_seed = []
    for seed in eval_seeds:
        obs, _ = env.reset(seed=seed)
        total_reward = 0.0
        while True:
            action = agent.select_action(env, obs)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            if term or trunc:
                break
        em = env._episode_metrics
        per_seed.append({
            "seed": seed,
            "total_reward": total_reward,
            "delivered_units": em["total_delivered_units"],
            "spoiled_units": em["total_spoiled_units"],
            "deliveries_count": em["deliveries_count"],
            "distance": em["total_distance"],
        })

    if hasattr(agent, "set_training"):
        agent.set_training(True)  # restore training mode for any later use

    rewards = np.array([r["total_reward"] for r in per_seed])
    delivered = np.array([r["delivered_units"] for r in per_seed])
    spoiled = np.array([r["spoiled_units"] for r in per_seed])

    return {
        "eval_mean_reward": float(rewards.mean()),
        "eval_std_reward": float(rewards.std()),
        "eval_mean_delivered": float(delivered.mean()),
        "eval_mean_spoiled": float(spoiled.mean()),
        "eval_n_episodes": len(per_seed),
        "per_seed": per_seed,
    }


# -----------------------------
# Output writers
# -----------------------------

def write_results_csv(history: list[dict], out_path: str) -> None:
    if not history:
        return
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def write_meta_json(cfg: ExperimentConfig, eval_summary: dict, wall_time: float, out_path: str) -> None:
    meta = {
        "run_id": cfg.run.run_id,
        "agent": cfg.run.agent,
        "scenario": cfg.run.scenario,
        "num_episodes": cfg.run.num_episodes,
        "seed": cfg.run.seed,
        "git_commit": get_git_hash(),
        "wall_time_seconds": wall_time,
        "eval_summary": {k: v for k, v in eval_summary.items() if k != "per_seed"},
        "eval_per_seed": eval_summary["per_seed"],
        "config_raw": cfg.raw,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


# -----------------------------
# Main
# -----------------------------

def run_experiment(config_path: str) -> dict:
    """End-to-end: load config, train, evaluate, save artifacts. Return summary."""
    cfg = load_config(config_path)

    print(f"\n{'=' * 70}")
    print(f"Running experiment: {cfg.run.run_id}")
    print(f"  Agent:    {cfg.run.agent}")
    print(f"  Scenario: {cfg.run.scenario}")
    print(f"  Episodes: {cfg.run.num_episodes}")
    print(f"  Seed:     {cfg.run.seed}")
    print(f"{'=' * 70}\n")

    # Configure MLflow
    configure_mlflow(experiment_name="food_rescue_rl")

    env = build_env(cfg)
    agent = build_agent(cfg, env)

    start_time = time.time()

    with start_run(
        run_name=cfg.run.run_id,
        tags={
            "agent": cfg.run.agent,
            "scenario": cfg.run.scenario,
            "config_file": Path(config_path).name,
        },
    ) as run:
        # Log all params (flattened for MLflow's flat namespace)
        flat_params = {"run_id": cfg.run.run_id, "agent": cfg.run.agent,
                       "scenario": cfg.run.scenario, "num_episodes": cfg.run.num_episodes,
                       "seed": cfg.run.seed}
        for k, v in cfg.agent_params.items():
            flat_params[f"agent.{k}"] = v
        for k, v in cfg.reward_weights.items():
            flat_params[f"reward.{k}"] = v
        log_params_safe(flat_params)

        # Train
        history = train(env, agent, cfg.run.num_episodes, cfg.run.seed, cfg.run.agent)

        # Log per-episode metrics to MLflow
        for ep_record in history:
            metrics = {k: v for k, v in ep_record.items() if k != "episode"}
            log_metrics_safe(metrics, step=ep_record["episode"])

        # Save policy if it's a learning agent
        policy_path = cfg.policy_path()
        if cfg.run.agent in {"q_learning", "sarsa", "dqn"}:
            Path(policy_path).parent.mkdir(parents=True, exist_ok=True)
            agent.save(policy_path)
            mlflow.log_artifact(policy_path, artifact_path="policy")
            # For DQN, log the meta JSON sidecar too
            if cfg.run.agent == "dqn":
                meta_sidecar = Path(policy_path).with_suffix(".meta.json")
                if meta_sidecar.exists():
                    mlflow.log_artifact(str(meta_sidecar), artifact_path="policy")

        # Write the per-episode CSV
        csv_path = cfg.results_csv_path()
        write_results_csv(history, csv_path)
        mlflow.log_artifact(csv_path, artifact_path="results")

        # Evaluate
        print(f"\nEvaluating trained policy on {cfg.eval.n_episodes} seeds...")
        eval_summary = evaluate(env, agent, cfg.eval.eval_seeds[:cfg.eval.n_episodes])
        log_metrics_safe(
            {k: v for k, v in eval_summary.items() if k not in {"per_seed"}},
        )

        wall_time = time.time() - start_time
        log_metric_safe("wall_time_seconds", wall_time)

        # Meta JSON
        meta_path = cfg.meta_json_path()
        write_meta_json(cfg, eval_summary, wall_time, meta_path)
        mlflow.log_artifact(meta_path, artifact_path="results")

        print(f"\nDone. Wall time: {wall_time:.1f}s")
        print(f"Eval mean reward over {eval_summary['eval_n_episodes']} seeds: "
              f"{eval_summary['eval_mean_reward']:+.2f} ± {eval_summary['eval_std_reward']:.2f}")
        print(f"Eval mean delivered: {eval_summary['eval_mean_delivered']:.1f} units")
        print(f"Eval mean spoiled:   {eval_summary['eval_mean_spoiled']:.1f} units")
        print(f"\nMLflow run ID: {run.info.run_id}")
        print(f"Artifacts saved to: {policy_path}, {csv_path}, {meta_path}")

        return {
            "run_id": cfg.run.run_id,
            "mlflow_run_id": run.info.run_id,
            "eval_summary": eval_summary,
            "wall_time": wall_time,
        }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train an RL agent on food rescue.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args(argv)

    try:
        run_experiment(args.config)
        return 0
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
