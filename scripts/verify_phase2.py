#!/usr/bin/env python3
"""Independent invariant and decision audit for SRRD Phase 2 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary_value(summary: pd.DataFrame, scenario: str, metric: str, field: str) -> float:
    rows = summary.loc[
        (summary["scenario"] == scenario) & (summary["metric"] == metric), field
    ]
    if rows.shape[0] != 1:
        raise RuntimeError(f"missing {scenario}/{metric}/{field}")
    return float(rows.iloc[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/phase2_confirmatory")
    parser.add_argument("--preregistration", default="preregistration/phase2.yaml")
    parser.add_argument("--freeze", default="preregistration/phase2.freeze.json")
    parser.add_argument("--external", default="external/phase2d_eligibility.json")
    args = parser.parse_args()

    root = Path(args.results)
    prereg_path = Path(args.preregistration)
    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    metadata = json.loads((root / "phase2_metadata.json").read_text(encoding="utf-8"))
    recorded_gates = json.loads((root / "phase2_gates.json").read_text(encoding="utf-8"))
    external = json.loads(Path(args.external).read_text(encoding="utf-8"))
    seeds = pd.read_csv(root / "phase2_seed_metrics.csv.gz")
    summary = pd.read_csv(root / "phase2_summary.csv")
    rotation = pd.read_csv(root / "phase2_rotation_summary.csv")

    baseline_columns = [
        "loss_markov_ssm",
        "loss_flat_rnn_0_5x",
        "loss_flat_rnn_1x",
        "loss_flat_rnn_2x",
        "loss_flat_rnn_4x",
        "loss_adaptive_psr",
        "loss_history_mpc",
    ]
    recomputed_strongest = seeds[baseline_columns].min(axis=1).to_numpy(float)
    r_ood = seeds["loss_srrd_bilevel"].to_numpy(float) / recomputed_strongest
    r_shuffle = seeds["loss_srrd_shuffle"].to_numpy(float) / seeds[
        "loss_srrd_bilevel"
    ].to_numpy(float)
    r_frozen = seeds["loss_srrd_frozen"].to_numpy(float) / seeds[
        "loss_srrd_bilevel"
    ].to_numpy(float)
    r_flat4 = seeds["loss_srrd_bilevel"].to_numpy(float) / seeds[
        "loss_flat_rnn_4x"
    ].to_numpy(float)

    ratio_errors = {
        "strongest_baseline": float(
            np.max(
                np.abs(
                    recomputed_strongest
                    - seeds["loss_strongest_baseline"].to_numpy(float)
                )
            )
        ),
        "r_ood": float(np.max(np.abs(r_ood - seeds["r_ood"].to_numpy(float)))),
        "r_shuffle": float(
            np.max(np.abs(r_shuffle - seeds["r_shuffle"].to_numpy(float)))
        ),
        "r_frozen": float(
            np.max(np.abs(r_frozen - seeds["r_frozen"].to_numpy(float)))
        ),
        "srrd_vs_flat_rnn_4x": float(
            np.max(np.abs(r_flat4 - seeds["srrd_vs_flat_rnn_4x"].to_numpy(float)))
        ),
    }

    thresholds = prereg["thresholds"]
    state = prereg["design"]["state_matching"]
    aligned = "true_srrd_aligned"
    independent_gates = {
        "G0_data_integrity": bool(
            seeds["seed_id"].is_unique
            and seeds.shape[0] == 2400
            and int(seeds["ood_contamination"].sum()) == 0
            and float(seeds["u_train_abs_max"].max()) <= 0.8 + 1e-12
            and float(seeds["u_test_abs_min"].min()) >= 1.2 - 1e-12
        ),
        "G1_observable_state_equivalence": bool(
            summary_value(summary, aligned, "state_max_smd", "ci_high")
            <= float(state["max_smd_margin"])
            and summary_value(summary, aligned, "state_energy_distance", "ci_high")
            <= float(state["energy_distance_margin"])
            and summary_value(summary, aligned, "state_permutation_reject", "mean")
            <= float(state["maximum_false_rejection_rate"])
        ),
        "G4_OOD_superiority": bool(
            summary_value(summary, aligned, "r_ood", "ci_high")
            <= float(thresholds["OOD_superiority_upper_CI"])
        ),
        "G5_order_necessity": bool(
            summary_value(summary, aligned, "r_shuffle", "ci_low")
            >= float(thresholds["shuffle_necessity_lower_CI"])
        ),
        "G6_reconstruction_necessity": bool(
            summary_value(summary, aligned, "r_frozen", "ci_low")
            >= float(thresholds["frozen_necessity_lower_CI"])
        ),
        "G7_update_interaction": bool(
            summary_value(summary, aligned, "abs_psi_update", "ci_low")
            >= float(thresholds["update_minimum_effect_lower_CI"])
        ),
        "G8_capacity_robustness": bool(
            summary_value(summary, aligned, "srrd_vs_flat_rnn_4x", "ci_high")
            <= float(thresholds["OOD_superiority_upper_CI"])
        ),
    }
    independent_gates["G2_positive_control_detectability"] = bool(
        independent_gates["G5_order_necessity"]
        and independent_gates["G6_reconstruction_necessity"]
        and independent_gates["G7_update_interaction"]
    )
    negative_margin = float(thresholds["negative_update_equivalence_margin"])
    independent_gates["G3_negative_control_specificity"] = bool(
        all(
            summary_value(summary, scenario, "abs_psi_update", "ci_high")
            <= negative_margin
            for scenario in ("pure_null", "frozen_rule", "order_invariant_memory")
        )
        and summary_value(summary, "frozen_rule", "r_frozen", "ci_high")
        < float(thresholds["frozen_necessity_lower_CI"])
    )
    independent_gates["G9_residual_imbalance_robustness"] = bool(
        summary_value(
            summary, "residual_state_imbalance", "state_max_smd", "ci_low"
        )
        > float(state["max_smd_margin"])
        and summary_value(
            summary,
            "residual_state_imbalance",
            "state_permutation_reject",
            "ci_low",
        )
        > 0.80
        and summary_value(
            summary, "residual_state_imbalance", "abs_psi_update", "ci_high"
        )
        <= negative_margin
    )

    rot_psi = rotation.loc[rotation["metric"] == "abs_psi_update"].sort_values(
        "angle_degrees"
    )
    rot_kappa = rotation.loc[rotation["metric"] == "kappa_obs"].sort_values(
        "angle_degrees"
    )
    rot_shuffle = rotation.loc[rotation["metric"] == "r_shuffle"].sort_values(
        "angle_degrees"
    )
    rho_psi = float(spearmanr(rot_psi["coupling_cosine"], rot_psi["mean"]).statistic)
    rho_kappa = float(
        spearmanr(rot_kappa["coupling_cosine"], np.abs(rot_kappa["mean"])).statistic
    )
    endpoint_psi = rot_psi.loc[rot_psi["angle_degrees"] == 90].iloc[0]
    endpoint_shuffle = rot_shuffle.loc[rot_shuffle["angle_degrees"] == 90].iloc[0]
    independent_gates["G10_observation_boundary"] = bool(
        rho_psi >= 0.95
        and rho_kappa >= 0.95
        and float(endpoint_psi["ci_high"])
        <= float(thresholds["observation_zero_margin"])
        and 0.90 <= float(endpoint_shuffle["ci_low"])
        and float(endpoint_shuffle["ci_high"]) <= 1.10
    )

    ordered = {key: independent_gates[key] for key in recorded_gates["gates"]}
    gate_agreement = ordered == recorded_gates["gates"]
    checks = {
        "preregistration_hash_matches_freeze": sha256(prereg_path) == freeze["sha256"],
        "preregistration_hash_matches_run_metadata": sha256(prereg_path)
        == metadata["preregistration_sha256"],
        "all_ratio_recomputations_within_1e-10": max(ratio_errors.values()) < 1e-10,
        "composite_seed_ids_unique": bool(seeds["seed_id"].is_unique),
        "nominal_feature_budgets_equal": bool(
            (seeds["feature_dim_srrd"] == seeds["feature_dim_flat_rnn_1x"]).all()
        ),
        "four_x_feature_budget_exact": bool(
            (
                seeds["feature_dim_flat_rnn_4x"]
                == 4 * seeds["feature_dim_srrd"]
            ).all()
        ),
        "forbidden_model_inputs_recorded": set(metadata["forbidden_model_inputs"])
        == {"true_rule", "latent_rule", "slow_state", "scenario_mechanism"},
        "external_candidates_not_misclassified_as_confirmatory": external["decision"]
        == "no_inspected_public_dataset_is_phase2d_confirmatory_eligible",
        "independent_gate_recalculation_matches": gate_agreement,
        "classification_is_C": recorded_gates["classification"]
        == "C_history_survives_SRRD_decomposition_does_not",
    }
    result = {
        "all_checks_pass": bool(all(checks.values())),
        "checks": checks,
        "ratio_max_absolute_errors": ratio_errors,
        "independent_gates": ordered,
        "recorded_classification": recorded_gates["classification"],
        "rotation_spearman": {
            "coupling_vs_abs_psi": rho_psi,
            "coupling_vs_abs_kappa": rho_kappa,
        },
    }
    (root / "phase2_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_checks_pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
