from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srrd_phase2.generator import SCENARIOS, generate_observables
from srrd_phase2b_hard.models import (
    MODEL_SPECS,
    NOMINAL_BASELINES,
    TARGET_NAME,
    TrainBudget,
    matched_hidden_size,
    stratified_fit_calibration_indices,
    tune_predictor,
)


class Phase2BHardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = generate_observables(SCENARIOS[0], seed=5, n_rows=128, horizon=6, split="train")
        self.test = generate_observables(SCENARIOS[0], seed=5, n_rows=64, horizon=6, split="test")

    def test_nominal_parameter_matching_is_within_frozen_tolerance(self) -> None:
        counts = {}
        for spec in MODEL_SPECS:
            if spec.parameter_multiplier != 1.0:
                continue
            _, counts[spec.name] = matched_hidden_size(spec.family, 2500, 6)
        target = counts[TARGET_NAME]
        for name in NOMINAL_BASELINES:
            self.assertLessEqual(abs(counts[name] / target - 1.0), 0.10)

    def test_nested_split_is_disjoint_and_train_only(self) -> None:
        fit, cal = stratified_fit_calibration_indices(self.train, 0.25)
        self.assertEqual(np.intersect1d(fit, cal).size, 0)
        self.assertEqual(np.union1d(fit, cal).size, self.train["y"].shape[0])
        self.assertLessEqual(float(np.abs(self.train["u2"]).max()), 0.8)
        np.testing.assert_allclose(self.test["u2"], 1.2)

    def test_end_to_end_predictor_smoke(self) -> None:
        spec = next(s for s in MODEL_SPECS if s.name == TARGET_NAME)
        predictor = tune_predictor(
            spec,
            self.train,
            horizon=6,
            nominal_target_params=2500,
            budget=TrainBudget(trials=1, max_epochs=2, patience=1),
            sigma_floor=0.5,
            seed=11,
        )
        pred = predictor.predict(self.test)
        frozen = predictor.predict(self.test, frozen_update=True)
        self.assertEqual(pred.shape, (64, 6))
        self.assertEqual(frozen.shape, (64, 6))
        self.assertTrue(np.isfinite(predictor.standardized_nll(self.test)))

    def test_history_mpc_selects_actions_from_frozen_grid(self) -> None:
        spec = next(s for s in MODEL_SPECS if s.name == "history_mpc_1x")
        predictor = tune_predictor(
            spec,
            self.train,
            horizon=6,
            nominal_target_params=2500,
            budget=TrainBudget(trials=1, max_epochs=2, patience=1),
            sigma_floor=0.5,
            seed=12,
        )
        grid = np.asarray([-1.2, -0.4, 0.0, 0.4, 1.2])
        actions = predictor.select_action(self.test, grid)
        self.assertEqual(actions.shape, (64,))
        self.assertTrue(set(np.unique(actions)).issubset(set(grid)))


if __name__ == "__main__":
    unittest.main()
