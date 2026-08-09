"""Preregistered Phase 2 black-box falsification experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .generator import SCENARIOS, generate_observables, shuffled_history_copy
from .models import RidgePredictor, model_specs
from .statistics import bootstrap_mean_ci, paired_state_metrics, update_interaction


@dataclass(frozen=True)
class Phase2Config:
    base_seeds: int = 200
    confirmation_seeds: int = 400
    confirmation_scenarios: tuple[str, ...] = (
        "true_srrd_aligned",
        "true_srrd_orthogonal",
        "flat_high_dim_markov",
        "residual_state_imbalance",
    )
    n_train: int = 192
    n_test: int = 160
    horizon: int = 6
    nominal_feature_budget: int = 24
    ridge: float = 1.0
    sigma_floor: float = 0.50
    state_permutations: int = 199
    bootstrap_replicates: int = 4000
    rotation_seeds: int = 80
    rotation_angles: tuple[int, ...] = (0, 15, 30, 45, 60, 75, 90)
    state_smd_margin: float = 0.10
    state_energy_margin: float = 0.10
    state_reject_rate_max: float = 0.10
    ood_ratio_threshold: float = 0.90
    shuffle_ratio_threshold: float = 1.10
    frozen_ratio_threshold: float = 1.10
    update_min_effect: float = 0.20
    negative_update_margin: float = 0.10
    observation_zero_margin: float = 0.10
    master_seed: int = 20260809


BASELINE_NAMES = (
    "markov_ssm",
    "flat_rnn_0_5x",
    "flat_rnn_1x",
    "flat_rnn_2x",
    "flat_rnn_4x",
    "adaptive_psr",
    "history_mpc",
)


def _n_seeds(config: Phase2Config, scenario_name: str) -> int:
    if scenario_name in config.confirmation_scenarios:
        return config.confirmation_seeds
    return config.base_seeds


def _run_seed(
    config: Phase2Config, scenario_index: int, seed: int
) -> dict[str, float | int | str]:
    scenario = SCENARIOS[scenario_index]
    train = generate_observables(
        scenario,
        seed=seed,
        n_rows=config.n_train,
        horizon=config.horizon,
        split="train",
    )
    test = generate_observables(
        scenario,
        seed=seed,
        n_rows=config.n_test,
        horizon=config.horizon,
        split="test",
    )
    matching = paired_state_metrics(
        test, permutations=config.state_permutations, seed=seed + 10000 * scenario_index
    )
    interaction = update_interaction(test)

    losses: dict[str, float] = {}
    predictors: dict[str, RidgePredictor] = {}
    for spec in model_specs(config.nominal_feature_budget):
        predictor = RidgePredictor(
            spec, ridge=config.ridge, sigma_floor=config.sigma_floor
        ).fit(train)
        predictors[spec.name] = predictor
        losses[spec.name] = predictor.standardized_nll(test)

    srrd = predictors["srrd_bilevel"]
    shuffled = shuffled_history_copy(
        test, seed=seed + scenario_index * config.confirmation_seeds
    )
    loss_srrd = losses["srrd_bilevel"]
    loss_shuffle = srrd.standardized_nll(shuffled)
    loss_frozen = srrd.standardized_nll(test, frozen_update=True)
    strongest = min(losses[name] for name in BASELINE_NAMES)

    row: dict[str, float | int | str] = {
        "scenario": scenario.name,
        "scenario_label": scenario.label,
        "expected_role": scenario.expected_role,
        "seed": seed,
        "seed_id": f"{scenario.name}:{seed}",
        "n_train": config.n_train,
        "n_test": config.n_test,
        "u_train_abs_max": float(np.max(np.abs(train["u2"]))),
        "u_test_abs_min": float(np.min(np.abs(test["u2"]))),
        "ood_contamination": int(np.any(np.isclose(np.abs(train["u2"]), 1.2))),
        **matching,
        "state_permutation_reject": int(
            matching["state_pair_permutation_p"] < 0.05
        ),
        **interaction,
        "loss_srrd_bilevel": loss_srrd,
        "loss_srrd_shuffle": loss_shuffle,
        "loss_srrd_frozen": loss_frozen,
        "loss_strongest_baseline": strongest,
        "r_ood": loss_srrd / strongest,
        "r_shuffle": loss_shuffle / loss_srrd,
        "r_frozen": loss_frozen / loss_srrd,
        "srrd_vs_flat_rnn_4x": loss_srrd / losses["flat_rnn_4x"],
    }
    for name, value in losses.items():
        row[f"loss_{name}"] = value
    specs = {spec.name: spec for spec in model_specs(config.nominal_feature_budget)}
    row["params_srrd"] = predictors["srrd_bilevel"].trainable_parameter_count(
        config.horizon
    )
    row["params_flat_rnn_1x"] = predictors[
        "flat_rnn_1x"
    ].trainable_parameter_count(config.horizon)
    row["params_flat_rnn_4x"] = predictors[
        "flat_rnn_4x"
    ].trainable_parameter_count(config.horizon)
    row["feature_dim_srrd"] = specs["srrd_bilevel"].feature_dim
    row["feature_dim_flat_rnn_1x"] = specs["flat_rnn_1x"].feature_dim
    row["feature_dim_flat_rnn_4x"] = specs["flat_rnn_4x"].feature_dim
    return row


SUMMARY_METRICS = (
    "loss_srrd_bilevel",
    "loss_strongest_baseline",
    "loss_flat_rnn_4x",
    "r_ood",
    "r_shuffle",
    "r_frozen",
    "srrd_vs_flat_rnn_4x",
    "psi_update",
    "abs_psi_update",
    "kappa_obs",
    "history_effect_sham",
    "history_effect_probe",
    "state_max_smd",
    "state_energy_distance",
    "state_permutation_reject",
)


def _summarize(
    config: Phase2Config, seed_metrics: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        frame = seed_metrics.loc[seed_metrics["scenario"] == scenario.name]
        for metric_index, metric in enumerate(SUMMARY_METRICS):
            mean, low, high = bootstrap_mean_ci(
                frame[metric].to_numpy(float),
                replicates=config.bootstrap_replicates,
                seed=1000 * scenario_index + metric_index,
            )
            rows.append(
                {
                    "scenario": scenario.name,
                    "scenario_label": scenario.label,
                    "expected_role": scenario.expected_role,
                    "metric": metric,
                    "n": int(frame.shape[0]),
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def _lookup(summary: pd.DataFrame, scenario: str, metric: str) -> pd.Series:
    rows = summary.loc[
        (summary["scenario"] == scenario) & (summary["metric"] == metric)
    ]
    if rows.shape[0] != 1:
        raise RuntimeError(f"missing summary row for {scenario}/{metric}")
    return rows.iloc[0]


def _run_rotation_seed(
    config: Phase2Config, angle: int, seed: int
) -> dict[str, float | int]:
    scenario = SCENARIOS[0]
    train = generate_observables(
        scenario,
        seed=seed,
        n_rows=config.n_train,
        horizon=config.horizon,
        split="train",
        observation_angle_override=float(angle),
    )
    test = generate_observables(
        scenario,
        seed=seed,
        n_rows=config.n_test,
        horizon=config.horizon,
        split="test",
        observation_angle_override=float(angle),
    )
    specs = {spec.name: spec for spec in model_specs(config.nominal_feature_budget)}
    srrd = RidgePredictor(
        specs["srrd_bilevel"], ridge=config.ridge, sigma_floor=config.sigma_floor
    ).fit(train)
    flat = RidgePredictor(
        specs["flat_rnn_4x"], ridge=config.ridge, sigma_floor=config.sigma_floor
    ).fit(train)
    shuffled = shuffled_history_copy(test, seed=seed + 900000 + angle)
    loss_srrd = srrd.standardized_nll(test)
    interaction = update_interaction(test)
    return {
        "angle_degrees": angle,
        "seed": seed,
        "coupling_cosine": float(np.cos(np.deg2rad(angle))),
        **interaction,
        "r_shuffle": srrd.standardized_nll(shuffled) / loss_srrd,
        "srrd_vs_flat_rnn_4x": loss_srrd / flat.standardized_nll(test),
    }


def _summarize_rotation(
    config: Phase2Config, rotation: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for angle_index, angle in enumerate(config.rotation_angles):
        frame = rotation.loc[rotation["angle_degrees"] == angle]
        for metric_index, metric in enumerate(
            ("abs_psi_update", "kappa_obs", "r_shuffle", "srrd_vs_flat_rnn_4x")
        ):
            mean, low, high = bootstrap_mean_ci(
                frame[metric].to_numpy(float),
                replicates=config.bootstrap_replicates,
                seed=9000 + angle_index * 100 + metric_index,
            )
            rows.append(
                {
                    "angle_degrees": angle,
                    "coupling_cosine": float(np.cos(np.deg2rad(angle))),
                    "metric": metric,
                    "n": int(frame.shape[0]),
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def evaluate_gates(
    config: Phase2Config,
    seeds: pd.DataFrame,
    summary: pd.DataFrame,
    rotation_summary: pd.DataFrame,
) -> dict[str, object]:
    def low(scenario: str, metric: str) -> float:
        return float(_lookup(summary, scenario, metric)["ci_low"])

    def high(scenario: str, metric: str) -> float:
        return float(_lookup(summary, scenario, metric)["ci_high"])

    integrity = bool(
        seeds["seed_id"].is_unique
        and int(seeds["ood_contamination"].sum()) == 0
        and float(seeds["u_train_abs_max"].max()) <= 0.8 + 1e-12
        and float(seeds["u_test_abs_min"].min()) >= 1.2 - 1e-12
    )
    state_equivalence = bool(
        high("true_srrd_aligned", "state_max_smd") <= config.state_smd_margin
        and high("true_srrd_aligned", "state_energy_distance")
        <= config.state_energy_margin
        # The preregistered false-rejection ceiling applies to the observed
        # seed rate. SMD and energy-distance margins, by contrast, explicitly
        # use upper confidence bounds. Applying a second confidence bound to
        # this diagnostic rate was the development evaluator bug fixed here.
        and float(
            _lookup(summary, "true_srrd_aligned", "state_permutation_reject")[
                "mean"
            ]
        )
        <= config.state_reject_rate_max
    )
    order_necessity = bool(
        low("true_srrd_aligned", "r_shuffle")
        >= config.shuffle_ratio_threshold
    )
    reconstruction_necessity = bool(
        low("true_srrd_aligned", "r_frozen") >= config.frozen_ratio_threshold
    )
    update_interaction_gate = bool(
        low("true_srrd_aligned", "abs_psi_update") >= config.update_min_effect
    )
    positive_detectability = bool(
        order_necessity and reconstruction_necessity and update_interaction_gate
    )
    negative_specificity = bool(
        high("pure_null", "abs_psi_update") <= config.negative_update_margin
        and high("frozen_rule", "abs_psi_update") <= config.negative_update_margin
        and high("order_invariant_memory", "abs_psi_update")
        <= config.negative_update_margin
        and high("frozen_rule", "r_frozen") < config.frozen_ratio_threshold
    )
    ood_superiority = bool(
        high("true_srrd_aligned", "r_ood") <= config.ood_ratio_threshold
    )
    capacity_robustness = bool(
        high("true_srrd_aligned", "srrd_vs_flat_rnn_4x")
        <= config.ood_ratio_threshold
    )
    residual_robustness = bool(
        low("residual_state_imbalance", "state_max_smd")
        > config.state_smd_margin
        and low("residual_state_imbalance", "state_permutation_reject") > 0.80
        and high("residual_state_imbalance", "abs_psi_update")
        <= config.negative_update_margin
    )

    rotation_psi = rotation_summary.loc[
        rotation_summary["metric"] == "abs_psi_update"
    ].sort_values("angle_degrees")
    rotation_kappa = rotation_summary.loc[
        rotation_summary["metric"] == "kappa_obs"
    ].sort_values("angle_degrees")
    rotation_shuffle = rotation_summary.loc[
        rotation_summary["metric"] == "r_shuffle"
    ].sort_values("angle_degrees")
    rho_psi = float(
        spearmanr(rotation_psi["coupling_cosine"], rotation_psi["mean"]).statistic
    )
    rho_kappa = float(
        spearmanr(
            rotation_kappa["coupling_cosine"], np.abs(rotation_kappa["mean"])
        ).statistic
    )
    endpoint_psi = rotation_psi.loc[rotation_psi["angle_degrees"] == 90].iloc[0]
    endpoint_shuffle = rotation_shuffle.loc[
        rotation_shuffle["angle_degrees"] == 90
    ].iloc[0]
    observation_boundary = bool(
        rho_psi >= 0.95
        and rho_kappa >= 0.95
        and float(endpoint_psi["ci_high"]) <= config.observation_zero_margin
        and float(endpoint_shuffle["ci_low"]) >= 0.90
        and float(endpoint_shuffle["ci_high"]) <= 1.10
    )

    gates = {
        "G0_data_integrity": integrity,
        "G1_observable_state_equivalence": state_equivalence,
        "G2_positive_control_detectability": positive_detectability,
        "G3_negative_control_specificity": negative_specificity,
        "G4_OOD_superiority": ood_superiority,
        "G5_order_necessity": order_necessity,
        "G6_reconstruction_necessity": reconstruction_necessity,
        "G7_update_interaction": update_interaction_gate,
        "G8_capacity_robustness": capacity_robustness,
        "G9_residual_imbalance_robustness": residual_robustness,
        "G10_observation_boundary": observation_boundary,
    }
    if not integrity:
        classification = "F_implementation_failure"
    elif not state_equivalence:
        classification = "E_unidentifiable"
    elif all(gates.values()):
        classification = "A_operationally_identified"
    elif positive_detectability and (not ood_superiority or not capacity_robustness):
        classification = "C_history_survives_SRRD_decomposition_does_not"
    elif not order_necessity or not reconstruction_necessity or not update_interaction_gate:
        classification = "D_mechanism_falsified"
    elif observation_boundary:
        classification = "B_conditionally_identified"
    else:
        classification = "E_unidentifiable"
    return {
        "all_gates_pass": bool(all(gates.values())),
        "gates": gates,
        "classification": classification,
        "rotation_diagnostics": {
            "spearman_cosine_vs_abs_psi": rho_psi,
            "spearman_cosine_vs_abs_kappa": rho_kappa,
        },
        "interpretation": (
            "Operational identification requires both mechanism-sensitive ablations and "
            "a >=10% held-out OOD advantage over every strong baseline."
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_phase2(
    config: Phase2Config,
    output_dir: Path,
    *,
    preregistration_path: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict[str, float | int | str]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for seed in range(_n_seeds(config, scenario.name)):
            seed_rows.append(_run_seed(config, scenario_index, seed))
    seeds = pd.DataFrame(seed_rows)
    summary = _summarize(config, seeds)

    rotation_rows = [
        _run_rotation_seed(config, angle, seed)
        for angle in config.rotation_angles
        for seed in range(config.rotation_seeds)
    ]
    rotation = pd.DataFrame(rotation_rows)
    rotation_summary = _summarize_rotation(config, rotation)
    gates = evaluate_gates(config, seeds, summary, rotation_summary)

    seeds.to_csv(
        output_dir / "phase2_seed_metrics.csv.gz",
        index=False,
        compression="gzip",
        float_format="%.12g",
    )
    summary.to_csv(output_dir / "phase2_summary.csv", index=False, float_format="%.12g")
    rotation.to_csv(
        output_dir / "phase2_rotation_seed_metrics.csv.gz",
        index=False,
        compression="gzip",
        float_format="%.12g",
    )
    rotation_summary.to_csv(
        output_dir / "phase2_rotation_summary.csv", index=False, float_format="%.12g"
    )
    (output_dir / "phase2_gates.json").write_text(
        json.dumps(gates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    budgets = [
        {
            "name": spec.name,
            "family": spec.family,
            "feature_dim": spec.feature_dim,
            "trainable_parameters": int(
                (spec.feature_dim + 1) * config.horizon + config.horizon
            ),
        }
        for spec in model_specs(config.nominal_feature_budget)
    ]
    (output_dir / "phase2_model_budgets.json").write_text(
        json.dumps(budgets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "study_id": "SRRD-P2-BLACKBOX-2026-08-09",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "preregistration_path": str(
            preregistration_path.relative_to(preregistration_path.parents[1])
        ),
        "preregistration_sha256": _sha256(preregistration_path),
        "model_input_contract": ["history", "x_obs", "c1", "u2"],
        "forbidden_model_inputs": [
            "true_rule",
            "latent_rule",
            "slow_state",
            "scenario_mechanism",
        ],
        "common_random_numbers_across_scenarios": True,
        "n_seed_runs": int(seeds.shape[0]),
        "n_rotation_runs": int(rotation.shape[0]),
    }
    (output_dir / "phase2_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "output_dir": str(output_dir),
        "n_seed_runs": int(seeds.shape[0]),
        "n_rotation_runs": int(rotation.shape[0]),
        "classification": gates["classification"],
        "all_gates_pass": gates["all_gates_pass"],
        "gates": gates["gates"],
    }
