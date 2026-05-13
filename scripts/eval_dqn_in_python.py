"""
Sanity-check a trained DQN by running it in the Python env (the env it was
trained on). Strips away any JS-side observation building / wiring concerns.

If this prints decent delivered/spoiled numbers, the model is fine and the
browser demo's worse behavior is due to JS observation mismatch.

If this prints garbage, the model itself never learned well.

Usage:
    python scripts/eval_dqn_in_python.py
    python scripts/eval_dqn_in_python.py --policy experiments/policies/dqn_v1.pt
    python scripts/eval_dqn_in_python.py --verbose
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Make project root importable when run from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.policy_loader import load_policy_from_env  # noqa: E402
from sim.environment import FoodRescueEnv  # noqa: E402


def evaluate_one_seed(policy, env: FoodRescueEnv, seed: int, verbose: bool = False) -> dict:
    """Run one episode greedily under the policy, return summary metrics."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    action_counts = {"donor": 0, "shelter": 0, "idle": 0}
    n_donors = env.num_donors
    n_shelters = env.num_shelters
    q_spreads = []

    # Detect which device the policy's weights live on (cpu / mps / cuda).
    device = next(policy.q_net.parameters()).device

    for t in range(env.max_episode_steps):
        # Use the agent's official select_action so we match training-eval behavior
        # exactly (including random tie-breaking, eval mode handling, etc).
        action = policy.select_action(env, obs)

        # Also pull the raw Q-values for the diagnostic (q-spread is what told us
        # the model was flat).
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            q = policy.q_net(obs_t).squeeze(0).cpu().numpy()
        q_spreads.append(float(q.max() - q.min()))

        # Classify what kind of action this was
        if action < n_donors:
            action_counts["donor"] += 1
        elif action < n_donors + n_shelters:
            action_counts["shelter"] += 1
        else:
            action_counts["idle"] += 1

        if verbose and t < 5:
            v = env.vehicles[env.current_vehicle_idx]
            # Try common attribute names for vehicle id, fall back to index.
            v_label = (
                getattr(v, "id", None)
                or getattr(v, "vehicle_id", None)
                or f"V{env.current_vehicle_idx}"
            )
            try:
                load = v.current_load()
            except Exception:
                load = getattr(v, "load", "?")
            print(f"  step {t}: vehicle={v_label} load={load} "
                  f"action={action} q-spread={q.max()-q.min():.2f} "
                  f"q-top3={[round(x, 2) for x in sorted(q.tolist(), reverse=True)[:3]]}")

        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        if term or trunc:
            break

    em = env._episode_metrics
    return {
        "seed": seed,
        "total_reward": round(total_reward, 2),
        "delivered_units": round(em["total_delivered_units"], 1),
        "spoiled_units": round(em["total_spoiled_units"], 1),
        "deliveries_count": em["deliveries_count"],
        "action_counts": action_counts,
        "mean_q_spread": round(float(np.mean(q_spreads)), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--policy",
        default="experiments/policies/dqn_v3_normalized.pt",
        help="Path to the .pt policy file",
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102, 103, 104])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import os
    print(f"Loading policy from: {args.policy}")
    # The public loader reads FOOD_RESCUE_MODEL_PATH from env — set it from CLI arg.
    os.environ["FOOD_RESCUE_MODEL_PATH"] = args.policy
    # Make sure we're not also pointed at the registry; local path wins.
    os.environ.pop("FOOD_RESCUE_MODEL_NAME", None)
    policy, info = load_policy_from_env()
    if hasattr(policy, "set_training"):
        policy.set_training(False)
    print(f"  Model: {info.get('agent_kind')} | obs_dim={info.get('obs_dim')} "
          f"| num_actions={info.get('num_actions')}")
    print()

    env = FoodRescueEnv()
    print(f"Env: {env.num_donors} donors, {env.num_shelters} shelters, "
          f"{env.max_episode_steps} steps/episode")
    print()

    results = []
    for seed in args.seeds:
        if args.verbose:
            print(f"--- seed {seed} (first 5 steps) ---")
        r = evaluate_one_seed(policy, env, seed, verbose=args.verbose)
        results.append(r)
        print(f"seed={seed}: reward={r['total_reward']:+.1f}  "
              f"delivered={r['delivered_units']:.0f}u  "
              f"spoiled={r['spoiled_units']:.0f}u  "
              f"deliveries={r['deliveries_count']}  "
              f"q-spread-avg={r['mean_q_spread']:.2f}  "
              f"actions={r['action_counts']}")

    # Aggregate
    rewards = [r["total_reward"] for r in results]
    deliv = [r["delivered_units"] for r in results]
    spoil = [r["spoiled_units"] for r in results]
    spreads = [r["mean_q_spread"] for r in results]

    print()
    print("=" * 60)
    print(f"AGGREGATE over {len(args.seeds)} seeds:")
    print(f"  Reward:    {np.mean(rewards):+.2f} ± {np.std(rewards):.2f}")
    print(f"  Delivered: {np.mean(deliv):.1f} ± {np.std(deliv):.1f} units")
    print(f"  Spoiled:   {np.mean(spoil):.1f} ± {np.std(spoil):.1f} units")
    print(f"  Q-spread:  {np.mean(spreads):.2f} (avg max-min Q-value per step)")
    print("=" * 60)


if __name__ == "__main__":
    main()