"""
Distribution drift detector for the food rescue prediction service.

Compares live request observations (from prediction_log.db) against the
training distribution (computed from the scenarios used during training).

Method: per-feature two-sample Kolmogorov-Smirnov test.
- p < 0.05 on a feature => that feature has drifted
- Overall drift flag: any feature drifted

Usage:
    from monitoring.drift_detector import DriftDetector
    detector = DriftDetector()
    report = detector.run()
    print(report)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DriftReport:
    n_live: int
    n_reference: int
    feature_pvalues: list[float]       # one per obs dimension
    drifted_features: list[int]        # indices where p < threshold
    drift_detected: bool
    threshold: float = 0.05
    message: str = ""

    def summary(self) -> str:
        if self.n_live < 30:
            return (
                f"Insufficient data: only {self.n_live} live predictions "
                f"(need ≥30 for reliable KS test)."
            )
        if not self.drift_detected:
            return (
                f"No drift detected across {len(self.feature_pvalues)} features "
                f"({self.n_live} live vs {self.n_reference} reference observations)."
            )
        return (
            f"DRIFT DETECTED on {len(self.drifted_features)} feature(s): "
            f"indices {self.drifted_features} "
            f"({self.n_live} live vs {self.n_reference} reference observations)."
        )


class DriftDetector:
    """
    Computes per-feature KS drift between reference data and live predictions.

    Reference data is built by rolling out the training scenarios and
    collecting obs vectors. This happens once on first call and is cached
    in memory.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        min_live_samples: int = 30,
        n_reference_episodes: int = 20,
    ):
        self.threshold = threshold
        self.min_live_samples = min_live_samples
        self.n_reference_episodes = n_reference_episodes
        self._reference: Optional[np.ndarray] = None  # shape (N, obs_dim)

    # ------------------------------------------------------------------
    # Reference distribution
    # ------------------------------------------------------------------

    def _build_reference(self) -> np.ndarray:
        """
        Collect obs vectors from early-episode steps across all scenarios.

        We deliberately mirror how the API receives observations in practice:
        env.reset() followed by a short number of steps. Full rollouts reach
        feature values (e.g. high time-remaining counters) that live requests
        never see, which causes spurious drift alerts.
        """
        from sim.environment import FoodRescueEnv, EnvConfig

        all_obs: list[np.ndarray] = []
        scenarios = ["weekday", "weekend", "holiday_rush"]
        steps_per_episode = 20  # match typical live request horizon

        for scenario in scenarios:
            try:
                env = FoodRescueEnv(config=EnvConfig(scenario_name=scenario))
            except Exception:
                continue

            eps_per_scenario = max(1, self.n_reference_episodes // len(scenarios))
            for ep in range(eps_per_scenario):
                obs, _ = env.reset(seed=ep)
                all_obs.append(obs.copy())
                for step in range(steps_per_episode):
                    action = env.action_space.sample()
                    obs, _, terminated, truncated, _ = env.step(action)
                    all_obs.append(obs.copy())
                    if terminated or truncated:
                        break

        if not all_obs:
            raise RuntimeError("Could not build reference distribution — no scenarios loaded.")

        return np.array(all_obs, dtype=np.float32)

    def _get_reference(self) -> np.ndarray:
        if self._reference is None:
            print("DriftDetector: building reference distribution...")
            self._reference = self._build_reference()
            print(f"DriftDetector: reference built — {self._reference.shape[0]} obs vectors.")
        return self._reference

    # ------------------------------------------------------------------
    # KS test
    # ------------------------------------------------------------------

    def run(self, live_obs: Optional[list[list[float]]] = None) -> DriftReport:
        """
        Run the KS drift test.

        Args:
            live_obs: override live observations (for testing). If None,
                      reads from the prediction log DB.
        """
        from scipy.stats import ks_2samp

        if live_obs is None:
            from api.prediction_log import fetch_observations
            live_obs = fetch_observations(500)

        n_live = len(live_obs)

        reference = self._get_reference()
        n_reference = reference.shape[0]

        if n_live < self.min_live_samples:
            obs_dim = reference.shape[1]
            return DriftReport(
                n_live=n_live,
                n_reference=n_reference,
                feature_pvalues=[1.0] * obs_dim,
                drifted_features=[],
                drift_detected=False,
                threshold=self.threshold,
                message=f"Need ≥{self.min_live_samples} live samples; have {n_live}.",
            )

        live_arr = np.array(live_obs, dtype=np.float32)
        obs_dim = reference.shape[1]

        pvalues: list[float] = []
        for i in range(obs_dim):
            _, p = ks_2samp(reference[:, i], live_arr[:, i])
            pvalues.append(float(p))

        drifted = [i for i, p in enumerate(pvalues) if p < self.threshold]

        return DriftReport(
            n_live=n_live,
            n_reference=n_reference,
            feature_pvalues=pvalues,
            drifted_features=drifted,
            drift_detected=len(drifted) > 0,
            threshold=self.threshold,
        )
