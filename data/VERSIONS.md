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
