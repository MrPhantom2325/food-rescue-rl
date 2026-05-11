# 🥫 food-rescue-rl

> *Two trucks. Five donors. Five shelters. 200 timesteps. Zero tolerance for wasted food.*

A **Reinforcement Learning system** that routes food rescue vehicles across a city grid — deciding in real time which donor to collect from, which shelter to deliver to, and how to stop food from becoming landfill. Built as part of a college MLOps course, aligned with **UN SDGs 2, 11, and 12**.

---

## The Problem (in one paragraph)

Every day, surplus food sits at restaurants, grocery stores, and farms while shelters run short. Human dispatchers can't optimize across multiple vehicles, changing food shelf lives, fluctuating shelter demand, and traffic — simultaneously, in real time. This project trains RL agents to do exactly that: minimize spoilage, maximize delivery, and keep shelters fed, even on the brutal `holiday_rush` scenario where supply overwhelms fleet capacity.

---

## Agents Implemented

| Agent | Type | Best Scenario | Notes |
|---|---|---|---|
| `RandomPolicy` | Baseline | — | Uniform random actions. The floor. |
| `GreedyPolicy` | Baseline | weekday (+1119) | Closest-donor-first with fleet deconfliction |
| `QLearningAgent` | Tabular RL | weekday | Discretized obs, ε-greedy, 1500 episodes |
| `SARSAAgent` | Tabular RL | weekday | On-policy variant, same table structure |
| `DQNAgent` | Deep RL | holiday_rush | MLP Q-network, replay buffer, target net |

---

## Architecture at a Glance

```
┌──────────────────────────────────────────────────────┐
│                  FoodRescueEnv (Gym)                 │
│  10×10 grid · 5 donors · 5 shelters · 2 vehicles    │
│  Obs: 31 floats   Action: Discrete(11)               │
│  Episode: 200 timesteps · Round-robin vehicle control│
└───────────────────────────┬──────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │         Agents            │
              │  Q-Learning / SARSA / DQN │
              └─────────────┬─────────────┘
                            │
        ┌───────────────────▼──────────────────┐
        │            MLflow Tracking            │
        │  Params · Metrics · Artifacts · Registry│
        └───────────────────┬──────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │       FastAPI Server       │
              │  POST /predict · /health  │
              │  Drift monitoring · Logs  │
              └───────────────────────────┘
```

---

## Reward Function

Every timestep, each vehicle earns:

```
R = α · food_delivered
  + priority_bonus · priority_deliveries
  − β · food_spoiled
  − γ · distance_traveled
  − δ · unmet_demand
  − oversupply_penalty · excess_delivered
```

| Weight | Symbol | Default | What it controls |
|---|---|---|---|
| `delivery` | α | 10.0 | Primary incentive — get food to shelters |
| `spoilage` | β | 5.0 | Penalise letting food rot |
| `distance` | γ | 0.1 | Gentle fuel/emissions cost |
| `unmet_demand` | δ | 1.0 | Equity — don't leave shelters empty |
| `priority_bonus` | — | 0.5 | Extra for high-priority shelter deliveries |
| `oversupply_penalty` | — | 0.3 | Don't dump excess on one shelter |

All weights are YAML-configurable and swept via Optuna.

---

## Scenarios

| Scenario | Supply/Episode | Greedy Reward | Difficulty |
|---|---|---|---|
| `weekday` | ~440 units | **+1119** | Easy — supply ≈ fleet capacity |
| `weekend` | ~620 units | **+684** | Medium — slight oversupply |
| `holiday_rush` | ~920 units | **−2710** | Hard — supply saturates fleet; triage required |

The `holiday_rush` scenario is intentionally unsolvable by simple heuristics. This is where DQN earns its keep.

---

## Observation Space (31 floats)

```
[ vehicle_x, vehicle_y, vehicle_load_pct, vehicle_idle_flag,   ← 4 per vehicle × 2
  donor_0_qty, donor_0_shelf_life, donor_0_dist,               ← 3 per donor × 5
  shelter_0_demand_pct, shelter_0_dist,                        ← 2 per shelter × 5
  normalized_time, current_vehicle_idx ]                       ← 2 global
```

All values normalized to [0, 1]. Tabular agents further discretize position, load, and demand into buckets.

---

## Repo Structure

```
food-rescue-rl/
├── agents/
│   ├── baseline.py        # Policy ABC, RandomPolicy, GreedyPolicy
│   ├── q_learning.py      # QLearningAgent + state discretization
│   ├── sarsa.py           # On-policy SARSA (inherits Q-learning table)
│   └── dqn.py             # DQNAgent, QNetwork, ReplayBuffer
├── sim/
│   ├── entities.py        # FoodBatch, Donor, Shelter, Vehicle
│   ├── city.py            # City, Scenario, ScenarioLoader
│   ├── environment.py     # FoodRescueEnv (Gymnasium), EnvConfig, RewardWeights
│   └── render.py          # FrameRenderer, EpisodeAnimator (MP4 / GIF)
├── api/                   # FastAPI serving endpoint
├── configs/               # 5 experiment YAMLs
├── data/
│   ├── raw/               # weekday / weekend / holiday_rush CSVs
│   ├── processed/         # outputs of data_prep.py
│   └── VERSIONS.md        # data changelog
├── scripts/
│   ├── compare_runs.py    # MLflow leaderboard printer
│   └── register_models.py # MLflow Model Registry
├── tests/                 # 175+ tests
├── experiments/           # gitignored: policies, results, videos, figures
├── train.py               # main entrypoint
├── data_prep.py           # validation + feature engineering
├── configs_loader.py      # YAML → dataclasses
└── mlops_tracking.py      # MLflow helpers
```

---

## Quickstart

```bash
# 1. Clone and set up
git clone https://github.com/yourname/food-rescue-rl.git
cd food-rescue-rl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Prepare data
python data_prep.py

# 3. Train an agent
python train.py --config configs/qlearning_v1.yaml    # tabular Q-learning
python train.py --config configs/dqn_v2_holiday.yaml  # DQN on hard scenario

# 4. Compare runs
python scripts/compare_runs.py

# 5. Serve predictions
uvicorn api.main:app --reload
# → POST http://localhost:8000/predict
```

---

## Experiment Configs

| Config file | Agent | Scenario | Episodes |
|---|---|---|---|
| `qlearning_v1.yaml` | Q-Learning | weekday | 1500 |
| `qlearning_v2_explored.yaml` | Q-Learning | weekday | 2000 (more exploration) |
| `sarsa_v1.yaml` | SARSA | weekday | 1500 |
| `dqn_v1.yaml` | DQN | weekday | 500 |
| `dqn_v2_holiday.yaml` | DQN | holiday_rush | 800 |

---

## MLflow Tracking

Every run logs:
- **Params:** all agent hyperparameters + reward weights + scenario name
- **Metrics:** reward per episode, delivered units, spoilage rate, unmet demand
- **Artifacts:** trained policy file (`.pkl` or `.pt`) + eval summary JSON

```bash
mlflow ui          # → http://localhost:5000
python scripts/compare_runs.py   # terminal leaderboard
python scripts/register_models.py  # promote best run to Model Registry
```

---

## Docker

```bash
# Build both images
docker compose build

# Run MLflow UI only
docker compose up mlflow

# Train inside Docker (CPU mode, fully reproducible)
docker compose --profile train run -e CONFIG=configs/dqn_v1.yaml train

# Serve the API
docker compose --profile serve up
# → http://localhost:8000/docs
```

Training inside Docker uses CPU (MPS unavailable in containers). Identical results, slower clock — use it for CI reproducibility, not iteration speed.

---

## API

```bash
POST /predict
Content-Type: application/json

{
  "observation": [0.5, 0.3, 0.8, 0.0, ...]   # 31 floats
}

→ { "action": 3, "agent": "dqn_v1", "timestamp": "..." }
```

```bash
GET /health     # liveness probe
GET /metrics    # prediction count, drift alerts
```

---

## Tests

```bash
pytest                        # all 175+ tests
pytest tests/test_dqn.py -v  # specific module
pytest --cov=. --cov-report=term-missing  # with coverage
```

Tests cover entity logic, environment step semantics, reward computation, agent save/load, MLflow cast boundaries, API response schema, and animation rendering.

---

## Tech Stack

| Layer | Library | Version |
|---|---|---|
| RL environment | Gymnasium | 0.29.1 |
| Deep learning | PyTorch | 2.2.0 |
| Experiment tracking | MLflow | 2.10.0 |
| Hyperparam sweep | Optuna | 3.5.0 |
| API serving | FastAPI + Uvicorn | 0.109 / 0.27 |
| Data | NumPy + Pandas | 1.26.4 / 2.2.0 |
| Visualization | Matplotlib + Seaborn | 3.8.2 / 0.13.2 |
| Testing | pytest | 8.0.0 |
| Linting | ruff | 0.2.0 |

---

## UN SDG Alignment

- **SDG 2 — Zero Hunger:** Routes food to shelters efficiently; prioritizes high-need sites
- **SDG 11 — Sustainable Cities:** Reduces unnecessary vehicle distance; models urban last-mile logistics
- **SDG 12 — Responsible Consumption:** Directly minimizes food spoilage in the reward signal

---

## Branch Strategy

```
main          ← stable, squash-merged sprints only
dev           ← integration branch, always green
feature/xyz   ← one branch per sprint, PR into dev
```

All merges go through PRs. No direct pushes to `main`. Sprint history in `CONTRIBUTING.md`.

---

## MLOps Course Rubric Coverage

| Requirement | Where |
|---|---|
| Multiple models | Q-Learning, SARSA, DQN + 2 baselines |
| Hyperparam tuning | Optuna sweeps via `configs/` YAMLs |
| MLflow tracking | `mlops_tracking.py` + `mlruns/` |
| Model Registry | `scripts/register_models.py` |
| Reproducible script | `train.py --config` |
| FastAPI endpoint | `api/` |
| Drift monitoring | prediction log comparison in `/metrics` |
| Docker + compose | `Dockerfile.train`, `Dockerfile.serve`, `docker-compose.yml` |
| CI/CD | `.github/workflows/` (GitHub Actions) |
| Git branching | main / dev / feature/* |
| Config-as-code | all experiment params in YAML |
| Data versioning | `data/VERSIONS.md` changelog |

---

*Built with too much caffeine and genuine concern about food waste.*
