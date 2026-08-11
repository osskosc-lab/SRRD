#!/usr/bin/env python3
"""Independent invariants, ratio, budget, freeze, and decision audit for Phase 2B-hard."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value(summary: pd.DataFrame, scenario: str, metric: str, field: str) -> float:
    rows = summary.loc[(summary["scenario"] == scenario) & (summary["metric"] == metric), field]
    if rows.shape[0] != 1:
        raise RuntimeError(f"missing {scenario}/{metric}/{field}")
    return float(rows.iloc[0])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/phase2b_hard_smoke")
    p.add_argument("--preregistration", default="preregistration/phase2b_hard.yaml")
    p.add_argument("--freeze", default="preregistration/phase2b_hard.freeze.json")
    p.add_argument("--mode", choices=("smoke", "development", "confirmatory"), default="smoke")
    args = p.parse_args()

    root = Path(args.results)
    prereg_path = Path(args.preregistration)
    freeze_path = Path(args.freeze)
    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    seeds = pd.read_csv(root / "phase2b_hard_seed_metrics.csv.gz")
    summary = pd.read_csv(root / "phase2b_hard_summary.csv")
    gates = json.loads((root / "phase2b_hard_gates.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "phase2b_hard_metadata.json").read_text(encoding="utf-8"))

    nominal = ["flat_gru_1x", "flat_lstm_1x", "learned_psr_1x", "history_mpc_1x"]
    cap4 = ["flat_gru_4x", "flat_lstm_4x"]
    strongest_nominal = seeds[[f"loss_{n}" for n in nominal]].min(axis=1).to_numpy(float)
    strongest_4x = seeds[[f"loss_{n}" for n in cap4]].min(axis=1).to_numpy(float)
    srrd = seeds["loss_srrd_bilevel_e2e"].to_numpy(float)
    r_ood = srrd / strongest_nominal
    r_4x = srrd / strongest_4x
    r_shuffle = seeds["loss_srrd_shuffle"].to_numpy(float) / srrd
    r_frozen = seeds["loss_srrd_frozen"].to_numpy(float) / srrd
    srrd_params = seeds["params_srrd_bilevel_e2e"].to_numpy(float)
    deviations = np.column_stack(
        [np.abs(seeds[f"params_{n}"].to_numpy(float) / srrd_params - 1.0) for n in nominal]
    ).max(axis=1)
    ratio_errors = {
        "strongest_nominal": float(np.max(np.abs(strongest_nominal - seeds["loss_strongest_nominal_baseline"].to_numpy(float)))),
        "strongest_4x": float(np.max(np.abs(strongest_4x - seeds["loss_strongest_4x_recurrent"].to_numpy(float)))),
        "r_ood": float(np.max(np.abs(r_ood - seeds["r_ood"].to_numpy(float)))),
        "r_capacity_4x": float(np.max(np.abs(r_4x - seeds["r_capacity_4x"].to_numpy(float)))),
        "r_shuffle": float(np.max(np.abs(r_shuffle - seeds["r_shuffle"].to_numpy(float)))),
        "r_frozen": float(np.max(np.abs(r_frozen - seeds["r_frozen"].to_numpy(float)))),
        "param_deviation": float(np.max(np.abs(deviations - seeds["nominal_param_ratio_max_abs_deviation"].to_numpy(float)))),
    }

    t = prereg["thresholds"]
    state = prereg["design"]["state_matching"]
    aligned = "true_srrd_aligned"
    independent = {
        "G0_data_integrity": bool(
            seeds["seed_id"].is_unique
            and int(seeds["ood_contamination"].sum()) == 0
            and float(seeds["u_train_abs_max"].max()) <= 0.8 + 1e-12
            and float(seeds["u_test_abs_min"].min()) >= 1.2 - 1e-12
            and np.isfinite(seeds.filter(regex=r"^loss_").to_numpy(float)).all()
        ),
        "G1_observable_state_equivalence": bool(
            value(summary, aligned, "state_max_smd", "ci_high") <= float(state["max_smd_margin"])
            and value(summary, aligned, "state_energy_distance", "ci_high") <= float(state["energy_distance_margin"])
            and value(summary, aligned, "state_permutation_reject", "mean") <= float(state["maximum_false_rejection_rate"])
        ),
        "G2_nested_tuning_isolation": bool(
            (seeds["no_ood_used_for_tuning"] == 1).all() and (seeds["optimization_budget_equal"] == 1).all()
        ),
        "G3_nominal_parameter_fairness": bool(
            value(summary, aligned, "nominal_param_ratio_max_abs_deviation", "ci_high")
            <= float(t["nominal_parameter_tolerance_fraction"])
        ),
        "G4_mechanism_positive_control": bool(
            value(summary, aligned, "r_shuffle", "ci_low") >= float(t["shuffle_necessity_lower_CI"])
            and value(summary, aligned, "r_frozen", "ci_low") >= float(t["frozen_necessity_lower_CI"])
            and value(summary, aligned, "abs_psi_update", "ci_low") >= float(t["update_minimum_effect_lower_CI"])
        ),
        "G5_negative_control_specificity": bool(
            all(
                value(summary, s, "abs_psi_update", "ci_high") <= float(t["negative_update_equivalence_margin"])
                for s in ("pure_null", "frozen_rule", "order_invariant_memory")
            )
            and value(summary, "frozen_rule", "r_frozen", "ci_high") < float(t["frozen_necessity_lower_CI"])
        ),
        "G6_OOD_superiority": bool(
            value(summary, aligned, "r_ood", "ci_high") <= float(t["OOD_superiority_upper_CI"])
        ),
        "G7_4x_capacity_robustness": bool(
            value(summary, aligned, "r_capacity_4x", "ci_high") <= float(t["capacity_robustness_upper_CI"])
        ),
        "G8_training_stability": bool(int(seeds["training_failures"].sum()) == 0),
        "G9_residual_imbalance_robustness": bool(
            value(summary, "residual_state_imbalance", "state_max_smd", "ci_low") > float(state["max_smd_margin"])
            and value(summary, "residual_state_imbalance", "state_permutation_reject", "ci_low") > 0.80
            and value(summary, "residual_state_imbalance", "abs_psi_update", "ci_high") <= float(t["negative_update_equivalence_margin"])
        ),
        "G10_history_mpc_control_competence": bool(
            value(summary, aligned, "control_random_gain_hmpc", "ci_low") >= float(t["control_random_gain_lower_CI"])
        ),
    }
    if not independent["G0_data_integrity"] or not independent["G2_nested_tuning_isolation"] or not independent["G8_training_stability"]:
        classification = "F_implementation_failure"
    elif not independent["G1_observable_state_equivalence"] or not independent["G3_nominal_parameter_fairness"] or not independent["G10_history_mpc_control_competence"]:
        classification = "E_unidentifiable_or_unfair_comparison"
    elif not independent["G4_mechanism_positive_control"]:
        classification = "D_SRRD_mechanism_falsified"
    elif independent["G6_OOD_superiority"] and independent["G7_4x_capacity_robustness"] and independent["G5_negative_control_specificity"] and independent["G9_residual_imbalance_robustness"]:
        classification = "A_hard_operationally_identified"
    elif independent["G6_OOD_superiority"] and not independent["G7_4x_capacity_robustness"]:
        classification = "C_capacity_sensitive_not_unique"
    else:
        classification = "B_history_survives_SRRD_not_uniquely_required"

    checks = {
        "freeze_hash_matches": sha256(prereg_path) == freeze["sha256"],
        "run_metadata_hash_matches": sha256(prereg_path) == metadata["preregistration_sha256"],
        "all_ratio_recomputations_within_1e_10": max(ratio_errors.values()) < 1e-10,
        "forbidden_inputs_recorded": set(metadata["forbidden_model_inputs"]) == {"true_rule", "latent_rule", "slow_state", "scenario_mechanism"},
        "OOD_not_used_for_tuning": metadata["ood_used_for_tuning"] is False and (seeds["no_ood_used_for_tuning"] == 1).all(),
        "independent_gates_match": independent == gates["gates"],
        "classification_matches": classification == gates["classification"],
        "confirmatory_row_count": True if args.mode != "confirmatory" else seeds.shape[0] == 2400,
    }
    independent = {key: bool(val) for key, val in independent.items()}
    checks = {key: bool(val) for key, val in checks.items()}
    result = {
        "all_checks_pass": bool(all(checks.values())),
        "checks": checks,
        "ratio_max_absolute_errors": ratio_errors,
        "independent_gates": independent,
        "independent_classification": classification,
        "recorded_classification": gates["classification"],
    }
    (root / "phase2b_hard_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_checks_pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
