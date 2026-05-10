"""Unit tests for sim.entities."""

import numpy as np

from sim.entities import BatchStatus, Donor, FoodBatch, Shelter, Vehicle


# -----------------------------
# FoodBatch
# -----------------------------

class TestFoodBatch:
    def test_tick_decrements_shelf_life(self):
        b = FoodBatch(batch_id=1, quantity=10, shelf_life=5, origin_donor_id="D001")
        b.tick()
        assert b.shelf_life == 4
        assert b.status == BatchStatus.AT_DONOR

    def test_tick_marks_spoiled_at_zero(self):
        b = FoodBatch(batch_id=1, quantity=10, shelf_life=1, origin_donor_id="D001")
        b.tick()
        assert b.status == BatchStatus.SPOILED
        assert b.location is None

    def test_tick_no_op_after_delivered(self):
        b = FoodBatch(batch_id=1, quantity=10, shelf_life=1, origin_donor_id="D001")
        b.status = BatchStatus.DELIVERED
        b.tick()
        assert b.shelf_life == 1  # unchanged

    def test_is_active(self):
        b = FoodBatch(batch_id=1, quantity=10, shelf_life=5, origin_donor_id="D001")
        assert b.is_active()
        b.status = BatchStatus.DELIVERED
        assert not b.is_active()
        b.status = BatchStatus.SPOILED
        assert not b.is_active()

    def test_waiting_time(self):
        b = FoodBatch(batch_id=1, quantity=10, shelf_life=5, origin_donor_id="D001",
                      created_at_step=10)
        assert b.waiting_time(current_step=15) == 5
        b.delivered_at_step = 20
        assert b.waiting_time(current_step=100) == 10  # uses delivery time


# -----------------------------
# Donor
# -----------------------------

class TestDonor:
    def _make_donor(self, arrival_rate=0.5, avg_quantity=10):
        return Donor(
            donor_id="D001", name="Test", type="restaurant",
            location=(2, 3), arrival_rate=arrival_rate,
            avg_quantity=avg_quantity, shelf_life_min=20, shelf_life_max=40,
        )

    def test_maybe_generate_batch_can_produce(self):
        d = self._make_donor(arrival_rate=1.0)  # always fires
        rng = np.random.default_rng(42)
        b = d.maybe_generate_batch(0, 0, 1.0, rng)
        assert b is not None
        assert b.origin_donor_id == "D001"
        assert b.location == (2, 3)
        assert 20 <= b.shelf_life <= 40
        assert len(d.pending_batches) == 1

    def test_maybe_generate_batch_can_skip(self):
        d = self._make_donor(arrival_rate=0.0)  # never fires
        rng = np.random.default_rng(42)
        b = d.maybe_generate_batch(0, 0, 1.0, rng)
        assert b is None
        assert len(d.pending_batches) == 0

    def test_rate_multiplier_clamps(self):
        d = self._make_donor(arrival_rate=0.5)
        rng = np.random.default_rng(42)
        # multiplier of 10 would give effective rate 5.0 → should clamp to 1.0
        # we can't directly check the rate, but rng.random() < 1.0 is always true,
        # so 100/100 trials should produce a batch
        produced = 0
        for i in range(100):
            d.pending_batches = []
            if d.maybe_generate_batch(i, i, rate_multiplier=10.0, rng=rng):
                produced += 1
        assert produced == 100

    def test_tick_pending_batches_spoils(self):
        d = self._make_donor()
        b = FoodBatch(batch_id=1, quantity=10, shelf_life=1, origin_donor_id="D001")
        d.pending_batches.append(b)
        count, qty = d.tick_pending_batches()
        assert count == 1
        assert qty == 10.0
        assert len(d.pending_batches) == 0

    def test_pickup_all(self):
        d = self._make_donor()
        b1 = FoodBatch(batch_id=1, quantity=5, shelf_life=10, origin_donor_id="D001")
        b2 = FoodBatch(batch_id=2, quantity=8, shelf_life=15, origin_donor_id="D001")
        d.pending_batches = [b1, b2]
        picked = d.pickup_all(current_step=5)
        assert len(picked) == 2
        assert all(b.status == BatchStatus.IN_VEHICLE for b in picked)
        assert len(d.pending_batches) == 0


# -----------------------------
# Shelter
# -----------------------------

class TestShelter:
    def _make_shelter(self, capacity=100.0):
        return Shelter(
            shelter_id="S001", name="Test", type="homeless",
            location=(5, 5), demand_rate=2.0, capacity=capacity, priority=1,
        )

    def test_tick_grows_demand(self):
        s = self._make_shelter()
        rng = np.random.default_rng(42)
        s.tick(rate_multiplier=1.0, rng=rng)
        assert s.current_demand > 0
        assert s.total_demand_accumulated > 0

    def test_demand_capped_at_capacity(self):
        s = self._make_shelter(capacity=5.0)
        rng = np.random.default_rng(42)
        for _ in range(100):
            s.tick(rate_multiplier=10.0, rng=rng)
        assert s.current_demand <= 5.0

    def test_receive_delivery_normal(self):
        s = self._make_shelter()
        s.current_demand = 20.0
        absorbed = s.receive_delivery(15.0)
        assert absorbed == 15.0
        assert s.current_demand == 5.0
        assert s.total_delivered == 15.0

    def test_receive_delivery_oversupply(self):
        s = self._make_shelter()
        s.current_demand = 5.0
        absorbed = s.receive_delivery(20.0)
        assert absorbed == 5.0
        assert s.current_demand == 0.0
        # Note: 15 units were "wasted" — the caller (vehicle) computes that.

    def test_utilization(self):
        s = self._make_shelter(capacity=100.0)
        s.current_demand = 25.0
        assert s.utilization() == 0.25


# -----------------------------
# Vehicle
# -----------------------------

class TestVehicle:
    def _make_vehicle(self, capacity=20.0, location=(0, 0)):
        return Vehicle(vehicle_id=0, location=location, capacity=capacity)

    def test_starts_idle(self):
        v = self._make_vehicle()
        assert v.is_idle()
        assert v.current_load() == 0
        assert v.remaining_capacity() == 20.0

    def test_set_and_clear_target(self):
        v = self._make_vehicle()
        v.set_target((5, 5), "donor", "D001")
        assert v.target == (5, 5)
        assert v.target_kind == "donor"
        assert not v.is_idle()
        v.clear_target()
        assert v.is_idle()

    def test_move_one_step_x_axis(self):
        v = self._make_vehicle(location=(0, 0))
        v.set_target((3, 0), "donor", "D001")
        v.move_one_step()
        assert v.location == (1, 0)
        assert v.distance_traveled == 1

    def test_move_one_step_y_axis(self):
        v = self._make_vehicle(location=(0, 0))
        v.set_target((0, 3), "donor", "D001")
        v.move_one_step()
        assert v.location == (0, 1)

    def test_move_prefers_larger_gap(self):
        v = self._make_vehicle(location=(0, 0))
        v.set_target((5, 2), "donor", "D001")
        v.move_one_step()
        assert v.location == (1, 0)  # x has larger gap

    def test_move_no_op_when_at_target(self):
        v = self._make_vehicle(location=(5, 5))
        v.set_target((5, 5), "donor", "D001")
        moved = v.move_one_step()
        assert moved == 0
        assert v.location == (5, 5)
        assert v.distance_traveled == 0

    def test_at_target(self):
        v = self._make_vehicle(location=(5, 5))
        v.set_target((5, 5), "donor", "D001")
        assert v.at_target()

    def test_load_batches_within_capacity(self):
        v = self._make_vehicle(capacity=20.0)
        b1 = FoodBatch(batch_id=1, quantity=8, shelf_life=10, origin_donor_id="D001")
        b2 = FoodBatch(batch_id=2, quantity=5, shelf_life=15, origin_donor_id="D001")
        leftover = v.load_batches([b1, b2])
        assert leftover == []
        assert v.current_load() == 13.0

    def test_load_batches_exceeds_capacity(self):
        v = self._make_vehicle(capacity=10.0)
        b1 = FoodBatch(batch_id=1, quantity=8, shelf_life=10, origin_donor_id="D001")
        b2 = FoodBatch(batch_id=2, quantity=5, shelf_life=15, origin_donor_id="D001")
        leftover = v.load_batches([b1, b2])
        assert len(leftover) == 1
        assert leftover[0].batch_id == 2  # second one didn't fit
        assert v.current_load() == 8.0

    def test_deliver_to_shelter(self):
        v = self._make_vehicle()
        b = FoodBatch(batch_id=1, quantity=10, shelf_life=5, origin_donor_id="D001",
                      status=BatchStatus.IN_VEHICLE)
        v.cargo.append(b)
        s = Shelter(shelter_id="S001", name="Test", type="homeless",
                    location=(5, 5), demand_rate=1.0, capacity=100.0, priority=1)
        s.current_demand = 7.0
        absorbed, wasted, n = v.deliver_to_shelter(s, current_step=10)
        assert absorbed == 7.0
        assert wasted == 3.0
        assert n == 1
        assert v.current_load() == 0
        assert b.status == BatchStatus.DELIVERED
        assert b.delivered_at_step == 10
