"""Unit tests for sim.city."""

import numpy as np
import pytest

from sim.city import City, ScenarioLoader


# -----------------------------
# City
# -----------------------------

class TestCity:
    def _make_city(self, grid_size=10, episode_length=200):
        return City(
            grid_size=grid_size,
            episode_length=episode_length,
            time_modifiers={
                "morning":   {"donor_arrival_mult": 0.5, "shelter_demand_mult": 0.7},
                "afternoon": {"donor_arrival_mult": 1.5, "shelter_demand_mult": 1.0},
                "evening":   {"donor_arrival_mult": 1.0, "shelter_demand_mult": 1.5},
            },
        )

    def test_is_valid_position_inside(self):
        c = self._make_city()
        assert c.is_valid_position(0, 0)
        assert c.is_valid_position(5, 5)
        assert c.is_valid_position(9, 9)

    def test_is_valid_position_outside(self):
        c = self._make_city()
        assert not c.is_valid_position(-1, 0)
        assert not c.is_valid_position(10, 0)
        assert not c.is_valid_position(0, 10)

    def test_time_bucket_morning(self):
        c = self._make_city(episode_length=300)
        # third = 100, so morning is [0, 100)
        assert c.time_bucket(0) == "morning"
        assert c.time_bucket(50) == "morning"
        assert c.time_bucket(99) == "morning"

    def test_time_bucket_afternoon(self):
        c = self._make_city(episode_length=300)
        assert c.time_bucket(100) == "afternoon"
        assert c.time_bucket(150) == "afternoon"
        assert c.time_bucket(199) == "afternoon"

    def test_time_bucket_evening(self):
        c = self._make_city(episode_length=300)
        assert c.time_bucket(200) == "evening"
        assert c.time_bucket(299) == "evening"

    def test_donor_rate_multiplier(self):
        c = self._make_city(episode_length=300)
        assert c.donor_rate_multiplier(50) == 0.5    # morning
        assert c.donor_rate_multiplier(150) == 1.5   # afternoon
        assert c.donor_rate_multiplier(250) == 1.0   # evening

    def test_shelter_rate_multiplier(self):
        c = self._make_city(episode_length=300)
        assert c.shelter_rate_multiplier(50) == 0.7
        assert c.shelter_rate_multiplier(150) == 1.0
        assert c.shelter_rate_multiplier(250) == 1.5


# -----------------------------
# ScenarioLoader
# -----------------------------
# These tests assume `python data_prep.py --scenario all` has been run.

class TestScenarioLoader:
    def test_available_scenarios(self):
        loader = ScenarioLoader()
        scenarios = loader.available_scenarios()
        assert "weekday" in scenarios
        assert "weekend" in scenarios
        assert "holiday_rush" in scenarios

    def test_load_weekday(self):
        loader = ScenarioLoader()
        scn = loader.load("weekday")
        assert scn.name == "weekday"
        assert scn.num_donors == 5
        assert scn.num_shelters == 5
        assert scn.num_vehicles == 2
        assert scn.vehicle_capacity == 20.0
        assert scn.random_seed == 42

    def test_load_returns_fresh_entities(self):
        loader = ScenarioLoader()
        s1 = loader.load("weekday")
        s2 = loader.load("weekday")
        # Same data, different objects
        assert s1.donors[0].donor_id == s2.donors[0].donor_id
        assert s1.donors[0] is not s2.donors[0]
        assert s1.shelters[0] is not s2.shelters[0]

    def test_distance_matrix_shape(self):
        loader = ScenarioLoader()
        scn = loader.load("weekday")
        assert scn.distance_matrix.shape == (5, 5)
        assert scn.donor_distance_matrix.shape == (5, 5)

    def test_distance_matrix_symmetric_donor_donor(self):
        loader = ScenarioLoader()
        scn = loader.load("weekday")
        # Donor-to-donor distances should be symmetric and zero on diagonal
        np.testing.assert_array_equal(
            scn.donor_distance_matrix, scn.donor_distance_matrix.T
        )
        assert all(scn.donor_distance_matrix[i, i] == 0 for i in range(5))

    def test_load_unknown_scenario_raises(self):
        loader = ScenarioLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_scenario")

    def test_donor_shelter_distance_lookup(self):
        loader = ScenarioLoader()
        scn = loader.load("weekday")
        # D001 is at (2, 3), S001 is at (3, 1) -> Manhattan distance = 1 + 2 = 3
        assert scn.donor_shelter_distance(0, 0) == 3

    def test_summary_stats_loaded(self):
        loader = ScenarioLoader()
        scn = loader.load("weekday")
        assert scn.summary_stats  # non-empty
        assert scn.summary_stats["scenario"] == "weekday"
        assert scn.summary_stats["num_donors"] == 5
