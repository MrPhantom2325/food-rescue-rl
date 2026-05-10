"""
Core entities for the food rescue simulation.

Four classes:
- FoodBatch: a chunk of food with quantity, shelf life, and location
- Donor: stationary food source that generates batches stochastically
- Shelter: stationary demand sink with growing unmet demand
- Vehicle: moves on the grid, picks up batches from donors, delivers to shelters

Entities own their own state and methods. The Gymnasium environment
(sim/environment.py) orchestrates them and exposes the standard RL interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# -----------------------------
# FoodBatch
# -----------------------------

class BatchStatus(Enum):
    """Lifecycle states for a food batch."""
    AT_DONOR = "at_donor"        # waiting at the donor
    IN_VEHICLE = "in_vehicle"    # being transported
    DELIVERED = "delivered"      # successfully delivered to a shelter
    SPOILED = "spoiled"          # shelf life expired before delivery


@dataclass
class FoodBatch:
    """
    A single batch of food.

    Attributes
    ----------
    batch_id : int
        Unique identifier within an episode.
    quantity : float
        Units of food (whatever unit; what matters is consistency).
    shelf_life : int
        Timesteps remaining before this batch spoils. Decrements every step.
    origin_donor_id : str
        Which donor produced this batch. Useful for analytics and rendering.
    status : BatchStatus
        Where the batch is in its lifecycle.
    location : tuple[int, int] | None
        Current (x, y) on the grid. None if delivered or spoiled.
    created_at_step : int
        Episode timestep when the batch was generated. Used for waiting-time metrics.
    delivered_at_step : int | None
        Episode timestep when delivered, if delivered.
    """
    batch_id: int
    quantity: float
    shelf_life: int
    origin_donor_id: str
    status: BatchStatus = BatchStatus.AT_DONOR
    location: Optional[tuple[int, int]] = None
    created_at_step: int = 0
    delivered_at_step: Optional[int] = None

    def tick(self) -> None:
        """Advance shelf life by one timestep. Mark as spoiled if it expires."""
        if self.status in (BatchStatus.DELIVERED, BatchStatus.SPOILED):
            return
        self.shelf_life -= 1
        if self.shelf_life <= 0:
            self.status = BatchStatus.SPOILED
            self.location = None

    def is_active(self) -> bool:
        """Is this batch still in play (not delivered or spoiled)?"""
        return self.status in (BatchStatus.AT_DONOR, BatchStatus.IN_VEHICLE)

    def waiting_time(self, current_step: int) -> int:
        """How long has this batch been waiting (created until now or delivery)?"""
        end = self.delivered_at_step if self.delivered_at_step is not None else current_step
        return end - self.created_at_step


# -----------------------------
# Donor
# -----------------------------

@dataclass
class Donor:
    """
    A fixed-location food source.

    Generates new FoodBatches stochastically each timestep based on arrival_rate.
    Holds AT_DONOR batches until a vehicle picks them up.

    Attributes
    ----------
    donor_id : str
        Unique key from the scenario CSV.
    name : str
        Human-readable label for rendering and logs.
    type : str
        Restaurant / supermarket / cafe / etc. Display only.
    location : tuple[int, int]
        (x, y) on the grid.
    arrival_rate : float
        Per-timestep Poisson rate of new batch generation (0..1).
    avg_quantity : float
        Mean of the quantity distribution for new batches.
    shelf_life_min : int
        Lower bound of uniform shelf-life distribution.
    shelf_life_max : int
        Upper bound of uniform shelf-life distribution.
    pending_batches : list[FoodBatch]
        Batches currently waiting at this donor.
    """
    donor_id: str
    name: str
    type: str
    location: tuple[int, int]
    arrival_rate: float
    avg_quantity: float
    shelf_life_min: int
    shelf_life_max: int
    pending_batches: list[FoodBatch] = field(default_factory=list)

    def maybe_generate_batch(
        self,
        current_step: int,
        next_batch_id: int,
        rate_multiplier: float,
        rng: np.random.Generator,
    ) -> Optional[FoodBatch]:
        """
        Possibly generate a new batch this timestep.

        Returns the new FoodBatch if one was generated, else None.
        Caller is responsible for incrementing the global batch counter.
        """
        effective_rate = min(self.arrival_rate * rate_multiplier, 1.0)
        if rng.random() >= effective_rate:
            return None

        # Quantity: Poisson around avg_quantity, clipped to a sensible minimum
        quantity = max(1.0, float(rng.poisson(self.avg_quantity)))

        # Shelf life: uniform in [min, max]
        shelf_life = int(rng.integers(self.shelf_life_min, self.shelf_life_max + 1))

        batch = FoodBatch(
            batch_id=next_batch_id,
            quantity=quantity,
            shelf_life=shelf_life,
            origin_donor_id=self.donor_id,
            status=BatchStatus.AT_DONOR,
            location=self.location,
            created_at_step=current_step,
        )
        self.pending_batches.append(batch)
        return batch

    def tick_pending_batches(self) -> tuple[int, float]:
        """
        Age all pending batches by one step.

        Returns
        -------
        spoiled_count : int
            Number of batches that spoiled this tick.
        spoiled_quantity : float
            Total quantity (units) that spoiled this tick.
        """
        spoiled_count = 0
        spoiled_quantity = 0.0
        surviving: list[FoodBatch] = []
        for b in self.pending_batches:
            b.tick()
            if b.status == BatchStatus.SPOILED:
                spoiled_count += 1
                spoiled_quantity += b.quantity
            else:
                surviving.append(b)
        self.pending_batches = surviving
        return spoiled_count, spoiled_quantity

    def total_pending_quantity(self) -> float:
        return sum(b.quantity for b in self.pending_batches)

    def min_pending_shelf_life(self) -> int:
        """Most-urgent batch's remaining shelf life. Returns a large number if no batches."""
        if not self.pending_batches:
            return 9999
        return min(b.shelf_life for b in self.pending_batches)

    def pickup_all(self, current_step: int) -> list[FoodBatch]:
        """
        Vehicle picks up everything currently pending. Marks them IN_VEHICLE.
        Caller is responsible for capacity checks BEFORE calling this.
        """
        picked = self.pending_batches
        for b in picked:
            b.status = BatchStatus.IN_VEHICLE
            b.location = None  # vehicle's location is the source of truth now
        self.pending_batches = []
        return picked


# -----------------------------
# Shelter
# -----------------------------

@dataclass
class Shelter:
    """
    A fixed-location demand sink.

    Demand grows each timestep at demand_rate (modulated by time-of-day).
    Capped at `capacity` to prevent unbounded accumulation.
    When a vehicle delivers food, demand is reduced by the delivered quantity.

    Attributes
    ----------
    shelter_id : str
    name : str
    type : str
        Homeless / children / elderly / foodbank / refugee. Display + reward weighting.
    location : tuple[int, int]
    demand_rate : float
        Per-timestep growth rate of unmet demand.
    capacity : float
        Hard cap on unmet demand.
    priority : int
        1 = high (priority shelter), 2 = standard. Used in reward shaping.
    current_demand : float
        Current unmet demand level. Updated each step.
    total_demand_accumulated : float
        Cumulative demand over the episode (for analytics).
    total_delivered : float
        Cumulative food delivered to this shelter (for analytics).
    """
    shelter_id: str
    name: str
    type: str
    location: tuple[int, int]
    demand_rate: float
    capacity: float
    priority: int
    current_demand: float = 0.0
    total_demand_accumulated: float = 0.0
    total_delivered: float = 0.0

    def tick(self, rate_multiplier: float, rng: np.random.Generator) -> None:
        """
        Advance demand by one timestep.

        Demand grows by Poisson(demand_rate * rate_multiplier), capped at capacity.
        """
        increment = float(rng.poisson(self.demand_rate * rate_multiplier))
        self.current_demand = min(self.current_demand + increment, self.capacity)
        self.total_demand_accumulated += increment

    def receive_delivery(self, quantity: float) -> float:
        """
        Apply a delivery. Returns the actual quantity that satisfied demand
        (i.e., min(quantity, current_demand)). Excess is wasted (over-supply).
        """
        absorbed = min(quantity, self.current_demand)
        self.current_demand -= absorbed
        self.total_delivered += absorbed
        return absorbed

    def utilization(self) -> float:
        """current_demand / capacity, useful for state representation."""
        return self.current_demand / self.capacity if self.capacity > 0 else 0.0


# -----------------------------
# Vehicle
# -----------------------------

@dataclass
class Vehicle:
    """
    A delivery vehicle that moves on the grid.

    Each timestep it can move one unit toward a target (Manhattan-style:
    we move along x or y, prefer the larger gap). When it reaches a target
    that is a Donor, it picks up. When it reaches a Shelter, it delivers.

    Movement is intentionally simple — the RL problem is *what to target next*,
    not pathfinding. Pathfinding on a grid is trivial.

    Attributes
    ----------
    vehicle_id : int
    location : tuple[int, int]
        Current (x, y).
    capacity : float
        Max total quantity it can carry across all batches.
    cargo : list[FoodBatch]
        Batches currently being carried (status IN_VEHICLE).
    target : tuple[int, int] | None
        Where the vehicle is currently headed. None means idle.
    target_kind : str | None
        'donor', 'shelter', or None.
    target_id : str | None
        ID of the target donor or shelter.
    distance_traveled : int
        Cumulative grid cells moved (for cost analytics).
    """
    vehicle_id: int
    location: tuple[int, int]
    capacity: float
    cargo: list[FoodBatch] = field(default_factory=list)
    target: Optional[tuple[int, int]] = None
    target_kind: Optional[str] = None
    target_id: Optional[str] = None
    distance_traveled: int = 0

    def current_load(self) -> float:
        return sum(b.quantity for b in self.cargo)

    def remaining_capacity(self) -> float:
        return self.capacity - self.current_load()

    def is_idle(self) -> bool:
        return self.target is None

    def set_target(self, target: tuple[int, int], kind: str, target_id: str) -> None:
        """Assign a navigation goal."""
        self.target = target
        self.target_kind = kind
        self.target_id = target_id

    def clear_target(self) -> None:
        self.target = None
        self.target_kind = None
        self.target_id = None

    def move_one_step(self) -> int:
        """
        Move one grid cell toward the target (greedy Manhattan walk).
        Returns 1 if it moved, 0 if already at target or idle.
        """
        if self.target is None:
            return 0

        x, y = self.location
        tx, ty = self.target
        dx, dy = tx - x, ty - y

        if dx == 0 and dy == 0:
            return 0  # already there

        # Move along the larger of the two gaps
        if abs(dx) >= abs(dy):
            x += 1 if dx > 0 else -1
        else:
            y += 1 if dy > 0 else -1

        self.location = (x, y)
        self.distance_traveled += 1
        return 1

    def at_target(self) -> bool:
        return self.target is not None and self.location == self.target

    def load_batches(self, batches: list[FoodBatch]) -> list[FoodBatch]:
        """
        Try to load all given batches. If total quantity exceeds remaining capacity,
        loads in order until full and returns the batches that didn't fit.
        Loaded batches have their status updated to IN_VEHICLE.
        """
        leftover: list[FoodBatch] = []
        for b in batches:
            if b.quantity <= self.remaining_capacity():
                b.status = BatchStatus.IN_VEHICLE
                b.location = None
                self.cargo.append(b)
            else:
                leftover.append(b)
        return leftover

    def deliver_to_shelter(
        self,
        shelter: Shelter,
        current_step: int,
    ) -> tuple[float, float, int]:
        """
        Deliver all cargo to a shelter.

        Returns
        -------
        delivered_quantity : float
            Total quantity that satisfied demand.
        wasted_quantity : float
            Quantity delivered beyond the shelter's current demand.
        num_batches_delivered : int
        """
        total_attempted = self.current_load()
        absorbed = shelter.receive_delivery(total_attempted)
        wasted = total_attempted - absorbed
        num_batches = len(self.cargo)

        for b in self.cargo:
            b.status = BatchStatus.DELIVERED
            b.delivered_at_step = current_step
            b.location = None

        self.cargo = []
        return absorbed, wasted, num_batches
