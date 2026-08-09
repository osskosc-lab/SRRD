from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srrd_phase2.features import build_features
from srrd_phase2.generator import (
    PUBLIC_KEYS,
    SCENARIOS,
    generate_observables,
    shuffled_history_copy,
)
from srrd_phase2.models import RidgePredictor, model_specs
from srrd_phase2.statistics import paired_state_metrics, update_interaction


class Phase2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.aligned = SCENARIOS[0]
        self.train = generate_observables(
            self.aligned, seed=3, n_rows=96, horizon=6, split="train"
        )
        self.test = generate_observables(
            self.aligned, seed=3, n_rows=80, horizon=6, split="test"
        )

    def test_generator_exposes_observables_only(self) -> None:
        self.assertEqual(set(self.train), set(PUBLIC_KEYS))
        forbidden = {"true_rule", "latent_rule", "slow_state", "mechanism"}
        self.assertTrue(forbidden.isdisjoint(self.train))

    def test_ood_intervention_is_strictly_held_out(self) -> None:
        self.assertLessEqual(float(np.abs(self.train["u2"]).max()), 0.8)
        np.testing.assert_allclose(self.test["u2"], 1.2)

    def test_history_shuffle_preserves_values_and_breaks_order(self) -> None:
        shuffled = shuffled_history_copy(self.test, seed=11)
        np.testing.assert_allclose(
            np.sort(shuffled["history"], axis=1),
            np.sort(self.test["history"], axis=1),
        )
        self.assertFalse(np.allclose(shuffled["history"], self.test["history"]))

    def test_feature_budgets_are_exact(self) -> None:
        for spec in model_specs(24):
            name = "flat_rnn" if spec.name.startswith("flat_rnn") else spec.name
            features = build_features(name, self.train, spec.feature_dim)
            self.assertEqual(features.shape, (96, spec.feature_dim))

    def test_nominal_capacity_is_equal(self) -> None:
        specs = {spec.name: spec for spec in model_specs(24)}
        self.assertEqual(specs["flat_rnn_1x"].feature_dim, specs["srrd_bilevel"].feature_dim)

    def test_state_matching_detects_residual_confound(self) -> None:
        matched = paired_state_metrics(self.test, permutations=99, seed=1)
        residual = generate_observables(
            SCENARIOS[6], seed=3, n_rows=80, horizon=6, split="test"
        )
        confounded = paired_state_metrics(residual, permutations=99, seed=1)
        self.assertLess(matched["state_max_smd"], 0.10)
        self.assertGreater(confounded["state_max_smd"], 0.10)

    def test_update_interaction_positive_only_when_observable(self) -> None:
        aligned = update_interaction(self.test)
        orthogonal = generate_observables(
            SCENARIOS[1], seed=3, n_rows=160, horizon=6, split="test"
        )
        null = update_interaction(orthogonal)
        self.assertGreater(aligned["abs_psi_update"], 0.20)
        self.assertLess(null["abs_psi_update"], 0.20)

    def test_models_are_deterministic(self) -> None:
        spec = next(s for s in model_specs(24) if s.name == "srrd_bilevel")
        first = RidgePredictor(spec, ridge=1.0, sigma_floor=0.5).fit(self.train)
        second = RidgePredictor(spec, ridge=1.0, sigma_floor=0.5).fit(self.train)
        np.testing.assert_allclose(first.predict(self.test), second.predict(self.test))


if __name__ == "__main__":
    unittest.main()
