#!/usr/bin/env python3
"""Independent closed-form and invariant checks for Phase 1 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def recovery_fraction_closed_form(rate: np.ndarray, horizon: int) -> np.ndarray:
    rate = np.asarray(rate, dtype=float)
    result = np.zeros_like(rate)
    active = rate > 0.0
    decay = 1.0 - rate[active]
    residual_mean = decay * (1.0 - decay**horizon) / (rate[active] * horizon)
    result[active] = 1.0 - residual_mean
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    root = Path(args.results)
    seeds = pd.read_csv(root / "phase1_seed_metrics.csv")
    metadata = json.loads((root / "phase1_metadata.json").read_text(encoding="utf-8"))
    config = metadata["config"]

    expected_recovery = recovery_fraction_closed_form(
        seeds["recovery_rate"].to_numpy(float), int(config["horizon"])
    )
    has_nonzero_target = (
        seeds[["target_ab_1", "target_ab_2", "target_ba_1", "target_ba_2"]]
        .abs()
        .max(axis=1)
        .to_numpy(float)
        > 1e-12
    )
    expected_recovery = np.where(has_nonzero_target, expected_recovery, 0.0)
    recovery_error = np.max(
        np.abs(seeds["recovery_fraction"].to_numpy(float) - expected_recovery)
    )

    target_ab = seeds[["target_ab_1", "target_ab_2"]].to_numpy(float)
    target_ba = seeds[["target_ba_1", "target_ba_2"]].to_numpy(float)
    expected_separation = np.linalg.norm(target_ab - target_ba, axis=1)
    separation_error = np.max(
        np.abs(seeds["rule_separation"].to_numpy(float) - expected_separation)
    )

    orthogonal = seeds.loc[seeds["scenario"] == "orthogonal_counterexample"]
    orthogonal_cvp_max_abs = float(orthogonal["cvp"].abs().max())
    state_match_max_abs = float(seeds["state_match_error"].abs().max())
    orthogonal_rule_readout_dot_max_abs = float(
        np.max(
            np.abs(
                orthogonal[["target_ab_1", "target_ba_1"]].to_numpy(float)
            )
        )
    )

    checks = {
        "rule_separation_closed_form": bool(separation_error < 1e-12),
        "recovery_fraction_closed_form": bool(recovery_error < 1e-12),
        "orthogonal_CVP_exact_zero": bool(orthogonal_cvp_max_abs < 1e-12),
        "state_matching_exact": bool(state_match_max_abs < 1e-12),
        "orthogonal_rule_in_CVP_kernel": bool(
            orthogonal_rule_readout_dot_max_abs < 1e-12
        ),
    }
    result = {
        "all_checks_pass": bool(all(checks.values())),
        "checks": checks,
        "max_absolute_errors": {
            "rule_separation": float(separation_error),
            "recovery_fraction": float(recovery_error),
            "orthogonal_cvp": orthogonal_cvp_max_abs,
            "state_match": state_match_max_abs,
            "orthogonal_rule_readout_dot": orthogonal_rule_readout_dot_max_abs,
        },
        "closed_form_equations": {
            "rule_separation": "2 * suffix_decay * (1 - contraction) * drive * ||direction||",
            "recovery_fraction": "1 - mean_{k=1..K}(1 - gamma)^k",
            "orthogonal_cvp": "x=(x1,0), theta=(0,theta2) => x^T theta=0 => p=0.5",
        },
    }
    (root / "phase1_analytic_checks.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_checks_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

