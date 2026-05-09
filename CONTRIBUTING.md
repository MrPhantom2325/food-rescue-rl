# Contributing Guide

## Branch Strategy

This project follows a three-tier branching model:

- **`main`** — production-ready, released code. Protected. Only updated via PRs from `dev`. Tagged on every release.
- **`dev`** — integration branch. All feature branches PR into here. Considered "stable but not yet released."
- **`feature/<short-name>`** — individual features, fixes, or chores. Created from `dev`, merged back into `dev` via PR.

## Workflow

### Starting a new feature

```bash
git checkout dev
git pull
git checkout -b feature/your-feature-name
```

### Committing

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `test:` add or fix tests
- `chore:` tooling, deps, refactor with no behavior change
- `refactor:` code restructure with no behavior change
- `perf:` performance improvement

Example: `feat(env): add donor batch expiry mechanic`

### Opening a PR

1. Push your branch: `git push -u origin feature/your-feature-name`
2. Open PR on GitHub against `dev`
3. PR description must include:
   - **What** changed
   - **Why** it changed
   - **How to test** the change
   - **Closes #N** (link to issue if applicable)
4. Self-review your own diff. Leave at least one comment on something non-trivial.
5. Wait for CI to pass (once CI is set up).
6. Squash-merge into `dev`.
7. Delete the feature branch (GitHub offers this button after merge).

### Releasing

Periodically, `dev` is PR'd into `main` as a release:

1. Open PR `dev` → `main` titled "Release v0.X-name"
2. PR description summarizes what's in the release
3. After merge, tag the release:
```bash
   git checkout main
   git pull
   git tag -a v0.1-rl-baseline -m "RL baseline release"
   git push --tags
```

## Sprint Plan & Branch Routing

| Sprint | Feature Branch | Closes Issues |
|---|---|---|
| 1. Data + Entities | `feature/data-and-entities` | #1 (partial) |
| 2. Gym Environment | `feature/gym-environment` | #1 |
| 3. Visualization | `feature/visualization` | — |
| 4. RL Agents | `feature/rl-agents` | #2, #3 |
| 5. MLflow Tracking | `feature/mlflow-tracking` | #4 |
| **Release `v0.1-rl-baseline`** | `dev` → `main` | — |
| 6. Hyperparam Tuning | `feature/hyperparam-tuning` | #8 |
| 7. Serving API | `feature/serving-api` | #5, #7 |
| 8. Containerization | `feature/containerization` | — |
| 9. CI/CD | `feature/cicd` | #6 |
| 10. K8s + Polish | `feature/k8s-and-polish` | — |
| **Release `v1.0-phase1-final`** | `dev` → `main` | — |

## Tag Conventions

- **Release tags** (on `main`): `v<major>.<minor>-<short-name>`
  - Examples: `v0.1-rl-baseline`, `v1.0-phase1-final`
- **Experiment tags** (anywhere, mark code that produced a specific result): `exp-<algorithm>-<n>`
  - Examples: `exp-qlearning-1`, `exp-qlearning-2`, `exp-dqn-1`, `exp-sarsa-1`

## Code Quality Standards

- All Python code must pass `ruff check .` (linting)
- All tests must pass: `pytest tests/`
- Test coverage should not drop below 70%: `pytest --cov=. tests/`
- Type hints encouraged but not required for Phase 1

## Issue Tracking

We use GitHub Issues with these labels:

- `bug` — something isn't working
- `feature` — new functionality
- `refactor` — code restructuring
- `docs` — documentation
- `mlops` — MLOps infrastructure (CI/CD, monitoring, serving)
- `rl` — reinforcement learning logic (agents, environment, training)

Every PR should reference an issue with `Closes #N` where applicable.

## PR Review Checklist (for self-review or peer review)

- [ ] Code follows the conventional commit style
- [ ] No print statements or debug code left in
- [ ] Tests added/updated for new behavior
- [ ] Linter passes locally (`ruff check .`)
- [ ] Tests pass locally (`pytest`)
- [ ] Documentation updated if behavior changed
- [ ] Linked issue exists and is closed by this PR
