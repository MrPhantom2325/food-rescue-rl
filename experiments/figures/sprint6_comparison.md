# Food Rescue RL — Comparison

All learning methods are evaluated across multiple training seeds, then on 5 held-out eval seeds per training run. Baselines are reported from single training runs (no training randomness for greedy/random).

| Method | Scenario | Eval Reward (mean ± std) | Delivered | Spoiled | Source |
|---|---|---:|---:|---:|---|
| `greedy_baseline` | weekday | +1337.3 ± 167.5 | 166.6 | 88.4 | single-seed baseline |
| `dqn_v1` | (varies) | -448.0 ± 141.4 | 67.2 | 230.6 | multi-seed (5) |
| `q_learning_tuned` | (varies) | -612.7 ± 349.7 | 60.0 | 245.8 | multi-seed (5) |
| `sarsa_tuned` | (varies) | -1048.1 ± 250.0 | 32.5 | 272.4 | multi-seed (5) |
| `random_baseline` | weekday | -1546.6 ± 261.3 | 6.0 | 312.2 | single-seed baseline |
