# Scenario Data Versions

This document tracks the version history of all scenario data. When you modify
scenario CSVs or configs, bump the version field in the scenario's
`scenario_config.yaml` and add an entry below.

## Why not DVC?

We initially considered DVC for data versioning but rejected it for this
project: dataset sizes are small (<10KB total), and DVC adds friction for team
collaborators who haven't installed it. Plain Git handles this volume cleanly,
and explicit version logging here is more readable than `.dvc` pointer files.

For a production version of this system with larger datasets (e.g., real GPS
traces or historical donation records), DVC or Git-LFS would be appropriate.

## Versioned Scenarios

### `weekday` v1.0 (initial)
- 5 donors, 5 shelters
- Moderate arrival/demand rates
- Standard time-of-day modifiers
- Random seed: 42

### `weekend` v1.0 (initial)
- 5 donors, 5 shelters
- Higher restaurant donations, higher shelter demand
- Stronger afternoon/evening rush
- Random seed: 43

### `holiday_rush` v1.0 (initial)
- 5 donors, 5 shelters
- High donations AND high demand
- Shorter shelf lives (prepared foods spoil faster)
- Strongest rush effects
- Random seed: 44

## Change Log Template

When you modify a scenario, append an entry:

### `<scenario>` v<X.Y> — YYYY-MM-DD
- What changed
- Why
- Migration notes (if behavior of trained policies changes)

### `weekday` v1.1 — 2026-05-10
- **Rebalanced supply to be feasible for the fleet.**
- Dropped donor arrival rates and avg quantities by ~40-50%.
- Reason: v1.0 generated 1500+ units of supply per episode while a 2-vehicle
  fleet can only move ~520 units max. Result was 80%+ guaranteed spoilage and
  RL having no room to demonstrate improvement over a greedy baseline.
- v1.1 expected supply: ~440 units, fleet throughput: ~520. Feasible.

### `weekend` v1.1 — 2026-05-10
- Same rebalance as weekday, scaled up modestly.
- v1.1 expected supply: ~620 units. Slight spoilage even with good policy
  preserves the "weekend has more donations than fleet can perfectly handle"
  flavor.

### `holiday_rush` v1.1 — 2026-05-10
- Same rebalance, scaled up further but with shorter shelf lives.
- v1.1 expected supply: ~920 units. Genuinely over-supplied — even good
  policies should see some spoilage, making this the hardest scenario.
- Shorter shelf lives (15-90 vs 20-120 originally) keep time pressure.
