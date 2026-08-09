from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srrd_phase1.metrics import cvp_metric, srrd_mechanism_metrics, sriv_metric
from srrd_phase1.model import (
    HISTORY_AB,
    HISTORY_BA,
    SCENARIOS,
    history_conditioned_rule_targets,
    predict_probabilities,
    reconstruct_rule,
    recursive_history_code,
)


class Phase1Tests(unittest.TestCase):
    def test_history_order_is_noncommutative_after_common_suffix(self) -> None:
        ab = recursive_history_code(
            HISTORY_AB, contraction=0.4, drive=1.5, common_suffix_decay=0.9
        )
        ba = recursive_history_code(
            HISTORY_BA, contraction=0.4, drive=1.5, common_suffix_decay=0.9
        )
        self.assertNotEqual(ab, ba)
        self.assertAlmostEqual(ab, -ba)

    def test_state_matching_is_exact_at_intervention(self) -> None:
        scenario = next(s for s in SCENARIOS if s.name == "orthogonal_counterexample")
        target_ab, target_ba = history_conditioned_rule_targets(
            scenario, contraction=0.4, common_suffix_decay=0.9
        )
        path_ab = reconstruct_rule(target_ab, horizon=12, recovery_rate=0.2)
        path_ba = reconstruct_rule(target_ba, horizon=12, recovery_rate=0.2)
        probes = np.random.default_rng(1).normal(size=(128, 2))
        p_ab = predict_probabilities(probes, path_ab, inverse_temperature=1.5)
        p_ba = predict_probabilities(probes, path_ba, inverse_temperature=1.5)
        np.testing.assert_allclose(p_ab[0], p_ba[0], atol=0.0, rtol=0.0)

    def test_mechanism_metric_does_not_accept_observables(self) -> None:
        q_ab = np.array([0.0, 1.5])
        q_ba = np.array([0.0, -1.5])
        path_ab = reconstruct_rule(q_ab, horizon=24, recovery_rate=0.2)
        path_ba = reconstruct_rule(q_ba, horizon=24, recovery_rate=0.2)
        result = srrd_mechanism_metrics(q_ab, q_ba, path_ab, path_ba)
        self.assertGreater(result["rule_separation"], 1.0)
        self.assertGreater(result["recovery_fraction"], 0.7)

    def test_constructive_high_srrd_low_cvp(self) -> None:
        q_ab = np.array([0.0, 1.5])
        q_ba = np.array([0.0, -1.5])
        path_ab = reconstruct_rule(q_ab, horizon=24, recovery_rate=0.2)
        path_ba = reconstruct_rule(q_ba, horizon=24, recovery_rate=0.2)
        diagnostic = np.random.default_rng(2).normal(size=(512, 2))
        p_ab = predict_probabilities(diagnostic, path_ab[1:], inverse_temperature=1.5)
        p_ba = predict_probabilities(diagnostic, path_ba[1:], inverse_temperature=1.5)
        self.assertGreater(sriv_metric(p_ab, p_ba), 0.02)

        x1 = np.random.default_rng(3).normal(size=512)
        viability = np.column_stack([x1, np.zeros_like(x1)])
        labels = np.broadcast_to((x1 > 0).astype(float), (24, x1.size))
        full = predict_probabilities(viability, path_ab[1:], inverse_temperature=1.5)
        ablated = np.full_like(full, 0.5)
        self.assertAlmostEqual(cvp_metric(labels, full, ablated), 0.0, places=12)

    def test_jsd_is_bounded(self) -> None:
        p = np.array([0.001, 0.2, 0.5, 0.9])
        q = 1.0 - p
        value = sriv_metric(p, q)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()

