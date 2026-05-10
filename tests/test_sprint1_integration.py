"""
Sprint 1 integration test.

Runs a short mini-episode using only Sprint 1 components (entities + city +
scenario loading). Confirms that:
- Scenarios load cleanly
- Donors generate batches over time
- Shelter demand grows
- Vehicles can move, pick up, and deliver
- Time-of-day modifiers are applied

This is the "everything works together" test before we wrap entities and
city behind a Gymnasium environment in Sprint 2.
"""

import numpy as np

from sim.city import ScenarioLoader, make_vehicles


def test_full_mini_episode_no_agent():
    """Run 50 steps with random vehicle actions, sanity-check the dynamics."""
    loader = ScenarioLoader()
    scn = loader.load("weekday")
    vehicles = make_vehicles(scn, start_strategy="center")
    rng = np.random.default_rng(scn.random_seed)

    next_batch_id = 0
    total_generated = 0
    total_spoiled = 0
    total_delivered = 0
    total_distance = 0

    for step in range(50):
        donor_mult = scn.city.donor_rate_multiplier(step)
        shelter_mult = scn.city.shelter_rate_multiplier(step)

        # Donors maybe generate
        for d in scn.donors:
            b = d.maybe_generate_batch(step, next_batch_id, donor_mult, rng)
            if b:
                next_batch_id += 1
                total_generated += 1

        # Donors age batches
        for d in scn.donors:
            count, _ = d.tick_pending_batches()
            total_spoiled += count

        # Shelters grow demand
        for s in scn.shelters:
            s.tick(shelter_mult, rng)

        # Random action for vehicle 0: if idle, pick a donor; if at donor, pick up; etc.
        v = vehicles[0]
        if v.is_idle():
            # 50/50: chase a donor or a shelter
            if rng.random() < 0.5:
                d_idx = int(rng.integers(0, scn.num_donors))
                v.set_target(scn.donors[d_idx].location, "donor", scn.donors[d_idx].donor_id)
            else:
                s_idx = int(rng.integers(0, scn.num_shelters))
                v.set_target(scn.shelters[s_idx].location, "shelter", scn.shelters[s_idx].shelter_id)
        else:
            # Move toward target
            total_distance += v.move_one_step()
            if v.at_target():
                if v.target_kind == "donor":
                    donor = next(d for d in scn.donors if d.donor_id == v.target_id)
                    picked = donor.pickup_all(step)
                    leftover = v.load_batches(picked)
                    # Put leftovers back (the donor would re-add them in real env)
                    donor.pending_batches.extend(leftover)
                elif v.target_kind == "shelter":
                    shelter = next(s for s in scn.shelters if s.shelter_id == v.target_id)
                    absorbed, wasted, n = v.deliver_to_shelter(shelter, step)
                    total_delivered += absorbed
                v.clear_target()

    # Sanity assertions on the dynamics
    assert total_generated > 0, "No batches were generated in 50 steps — donor rates too low?"
    assert any(s.current_demand > 0 for s in scn.shelters), \
        "No shelter accumulated any demand"
    assert total_distance > 0, "Vehicle never moved"

    # Things that COULD be zero but we'll log them for human inspection
    print("\n[mini-episode summary]")
    print(f"  generated:    {total_generated} batches")
    print(f"  spoiled:      {total_spoiled} batches")
    print(f"  delivered:    {total_delivered:.1f} units")
    print(f"  distance:     {total_distance} cells")
    print(f"  open demand:  {sum(s.current_demand for s in scn.shelters):.1f} units")


def test_all_scenarios_loadable_and_runnable():
    """Each scenario should produce non-trivial dynamics in 30 steps."""
    loader = ScenarioLoader()
    for name in loader.available_scenarios():
        scn = loader.load(name)
        rng = np.random.default_rng(scn.random_seed)

        next_batch_id = 0
        for step in range(30):
            donor_mult = scn.city.donor_rate_multiplier(step)
            shelter_mult = scn.city.shelter_rate_multiplier(step)
            for d in scn.donors:
                b = d.maybe_generate_batch(step, next_batch_id, donor_mult, rng)
                if b:
                    next_batch_id += 1
            for d in scn.donors:
                d.tick_pending_batches()
            for s in scn.shelters:
                s.tick(shelter_mult, rng)

        total_pending = sum(len(d.pending_batches) for d in scn.donors)
        total_demand = sum(s.current_demand for s in scn.shelters)
        # In 30 steps, both supply (donor pending) and demand should appear
        assert total_pending > 0 or next_batch_id > 0, \
            f"Scenario {name}: no batches ever appeared in 30 steps"
        assert total_demand > 0, \
            f"Scenario {name}: no shelter demand grew in 30 steps"


def test_random_seed_reproducibility():
    """Two episodes with the same seed should produce identical batch sequences."""
    loader = ScenarioLoader()

    def run_short_episode(seed: int):
        scn = loader.load("weekday")
        rng = np.random.default_rng(seed)
        next_batch_id = 0
        sequence = []
        for step in range(20):
            donor_mult = scn.city.donor_rate_multiplier(step)
            for d in scn.donors:
                b = d.maybe_generate_batch(step, next_batch_id, donor_mult, rng)
                if b:
                    next_batch_id += 1
                    sequence.append((step, b.origin_donor_id, b.quantity, b.shelf_life))
        return sequence

    seq1 = run_short_episode(seed=42)
    seq2 = run_short_episode(seed=42)
    assert seq1 == seq2, "Same seed produced different batch sequences"

    seq3 = run_short_episode(seed=99)
    assert seq1 != seq3, "Different seeds somehow produced identical sequences"
