"""SRRD Phase 2B-hard: end-to-end operational identifiability falsification."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from srrd_phase2.generator import SCENARIOS, generate_observables, shuffled_history_copy
from srrd_phase2.statistics import bootstrap_mean_ci, paired_state_metrics, update_interaction

from .control import control_audit
from .models import (
    CAPACITY_BASELINES,
    MODEL_SPECS,
    NOMINAL_BASELINES,
    TARGET_NAME,
    TrainBudget,
    tune_predictor,
)


@dataclass(frozen=True)
class Phase2BHardConfig:
    base_seeds: int = 200
    confirmation_seeds: int = 400
    confirmation_scenarios: tuple[str, ...] = (
        "true_srrd_aligned",
        "true_srrd_orthogonal",
        "flat_high_dim_markov",
        "residual_state_imbalance",
    )
    n_train: int = 512
    n_test: int = 256
    horizon: int = 6
    nominal_target_params: int = 2500
    tuning_trials: int = 6
    max_epochs: int = 80
    patience: int = 10
    calibration_fraction: float = 0.25
    sigma_floor: float = 0.50
    state_permutations: int = 199
    bootstrap_replicates: int = 4000
    state_smd_margin: float = 0.10
    state_energy_margin: float = 0.10
    state_reject_rate_max: float = 0.10
    nominal_parameter_tolerance: float = 0.10
    ood_ratio_threshold: float = 0.90
    shuffle_ratio_threshold: float = 1.10
    frozen_ratio_threshold: float = 1.10
    update_min_effect: float = 0.20
    negative_update_margin: float = 0.10
    control_random_gain_threshold: float = 1.05
    action_grid: tuple[float, ...] = (-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2)
    action_penalty: float = 0.02
    master_seed: int = 20260811


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_frozen(preregistration_path: Path, freeze_path: Path) -> dict[str, object]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    actual = sha256(preregistration_path)
    if actual != freeze["sha256"]:
        raise RuntimeError("preregistration hash does not match frozen Phase 2B-hard protocol")
    return freeze


def _n_seeds(config: Phase2BHardConfig, scenario_name: str) -> int:
    return config.confirmation_seeds if scenario_name in config.confirmation_scenarios else config.base_seeds


def _run_seed(config: Phase2BHardConfig, scenario_index: int, seed: int) -> dict[str, float | int | str]:
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
        test,
        permutations=config.state_permutations,
        seed=seed + 10000 * scenario_index,
    )
    interaction = update_interaction(test)
    budget = TrainBudget(
        trials=config.tuning_trials,
        max_epochs=config.max_epochs,
        patience=config.patience,
        calibration_fraction=config.calibration_fraction,
    )

    predictors = {}
    losses: dict[str, float] = {}
    metadata: dict[str, dict[str, float | int | str]] = {}
    for spec_index, spec in enumerate(MODEL_SPECS):
        predictor = tune_predictor(
            spec,
            train,
            horizon=config.horizon,
            nominal_target_params=config.nominal_target_params,
            budget=budget,
            sigma_floor=config.sigma_floor,
            seed=config.master_seed + seed * 1000 + scenario_index * 100 + spec_index,
        )
        predictors[spec.name] = predictor
        metadata[spec.name] = predictor.metadata
        losses[spec.name] = predictor.standardized_nll(test)
        if not np.isfinite(losses[spec.name]):
            raise FloatingPointError(f"non-finite OOD loss: {spec.name}")

    srrd = predictors[TARGET_NAME]
    shuffled = shuffled_history_copy(test, seed=seed + scenario_index * 100000 + 87)
    loss_srrd = losses[TARGET_NAME]
    loss_shuffle = srrd.standardized_nll(shuffled)
    loss_frozen = srrd.standardized_nll(test, frozen_update=True)
    strongest_nominal = min(losses[name] for name in NOMINAL_BASELINES)
    strongest_capacity = min(losses[name] for name in CAPACITY_BASELINES)

    srrd_params = int(metadata[TARGET_NAME]["trainable_params"])
    nominal_param_ratios = np.asarray(
        [int(metadata[name]["trainable_params"]) / srrd_params for name in NOMINAL_BASELINES],
        dtype=float,
    )
    trials = [int(metadata[name]["tuning_trials"]) for name in metadata]
    max_epochs = [int(metadata[name]["max_epochs"]) for name in metadata]

    control_values = {
        "control_cost_hmpc": np.nan,
        "control_regret_hmpc": np.nan,
        "control_random_gain_hmpc": np.nan,
        "control_cost_srrd": np.nan,
        "control_regret_srrd": np.nan,
    }
    if scenario.name == "true_srrd_aligned":
        hmpc_control = control_audit(
            predictors["history_mpc_1x"],
            scenario.name,
            test,
            horizon=config.horizon,
            seed=seed,
            action_grid=config.action_grid,
            action_penalty=config.action_penalty,
        )
        srrd_control = control_audit(
            predictors[TARGET_NAME],
            scenario.name,
            test,
            horizon=config.horizon,
            seed=seed + 1,
            action_grid=config.action_grid,
            action_penalty=config.action_penalty,
        )
        control_values = {
            "control_cost_hmpc": hmpc_control["control_cost"],
            "control_regret_hmpc": hmpc_control["control_regret"],
            "control_random_gain_hmpc": hmpc_control["control_random_gain"],
            "control_cost_srrd": srrd_control["control_cost"],
            "control_regret_srrd": srrd_control["control_regret"],
        }

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
        "no_ood_used_for_tuning": 1,
        "optimization_budget_equal": int(len(set(trials)) == 1 and len(set(max_epochs)) == 1),
        "nominal_param_ratio_max_abs_deviation": float(np.max(np.abs(nominal_param_ratios - 1.0))),
        "training_failures": 0,
        **matching,
        "state_permutation_reject": int(matching["state_pair_permutation_p"] < 0.05),
        **interaction,
        **control_values,
        "loss_srrd_shuffle": float(loss_shuffle),
        "loss_srrd_frozen": float(loss_frozen),
        "loss_strongest_nominal_baseline": float(strongest_nominal),
        "loss_strongest_4x_recurrent": float(strongest_capacity),
        "r_ood": float(loss_srrd / strongest_nominal),
        "r_shuffle": float(loss_shuffle / loss_srrd),
        "r_frozen": float(loss_frozen / loss_srrd),
        "r_capacity_4x": float(loss_srrd / strongest_capacity),
    }
    for name, value in losses.items():
        row[f"loss_{name}"] = float(value)
    for name, meta in metadata.items():
        row[f"params_{name}"] = int(meta["trainable_params"])
        row[f"hidden_{name}"] = int(meta["hidden"])
        row[f"trial_{name}"] = int(meta["selected_trial"])
        row[f"lr_{name}"] = float(meta["selected_lr"])
        row[f"wd_{name}"] = float(meta["selected_weight_decay"])
        row[f"epochs_{name}"] = int(meta["selected_epochs"])
    return row


SUMMARY_METRICS = (
    "loss_srrd_bilevel_e2e",
    "loss_strongest_nominal_baseline",
    "loss_strongest_4x_recurrent",
    "r_ood",
    "r_shuffle",
    "r_frozen",
    "r_capacity_4x",
    "psi_update",
    "abs_psi_update",
    "kappa_obs",
    "state_max_smd",
    "state_energy_distance",
    "state_permutation_reject",
    "nominal_param_ratio_max_abs_deviation",
    "control_cost_hmpc",
    "control_regret_hmpc",
    "control_random_gain_hmpc",
    "control_cost_srrd",
    "control_regret_srrd",
)


def _summarize(config: Phase2BHardConfig, seeds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        frame = seeds.loc[seeds["scenario"] == scenario.name]
        if frame.empty:
            continue
        for metric_index, metric in enumerate(SUMMARY_METRICS):
            values = frame[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            mean, low, high = bootstrap_mean_ci(
                values,
                replicates=config.bootstrap_replicates,
                seed=30000 + scenario_index * 100 + metric_index,
            )
            rows.append(
                {
                    "scenario": scenario.name,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def _value(summary: pd.DataFrame, scenario: str, metric: str, field: str) -> float:
    rows = summary.loc[(summary["scenario"] == scenario) & (summary["metric"] == metric), field]
    if rows.shape[0] != 1:
        raise RuntimeError(f"missing summary value {scenario}/{metric}/{field}")
    return float(rows.iloc[0])


def evaluate_gates(config: Phase2BHardConfig, seeds: pd.DataFrame, summary: pd.DataFrame) -> dict[str, object]:
    aligned = "true_srrd_aligned"
    integrity = bool(
        seeds["seed_id"].is_unique
        and int(seeds["ood_contamination"].sum()) == 0
        and float(seeds["u_train_abs_max"].max()) <= 0.8 + 1e-12
        and float(seeds["u_test_abs_min"].min()) >= 1.2 - 1e-12
        and np.isfinite(seeds.filter(regex=r"^loss_").to_numpy(float)).all()
    )
    state_equivalence = bool(
        _value(summary, aligned, "state_max_smd", "ci_high") <= config.state_smd_margin
        and _value(summary, aligned, "state_energy_distance", "ci_high") <= config.state_energy_margin
        and _value(summary, aligned, "state_permutation_reject", "mean") <= config.state_reject_rate_max
    )
    optimization_isolation = bool(
        (seeds["no_ood_used_for_tuning"] == 1).all()
        and (seeds["optimization_budget_equal"] == 1).all()
    )
    capacity_fairness = bool(
        _value(summary, aligned, "nominal_param_ratio_max_abs_deviation", "ci_high")
        <= config.nominal_parameter_tolerance
    )
    mechanism_positive = bool(
        _value(summary, aligned, "r_shuffle", "ci_low") >= config.shuffle_ratio_threshold
        and _value(summary, aligned, "r_frozen", "ci_low") >= config.frozen_ratio_threshold
        and _value(summary, aligned, "abs_psi_update", "ci_low") >= config.update_min_effect
    )
    negative_specificity = bool(
        all(
            _value(summary, scenario, "abs_psi_update", "ci_high") <= config.negative_update_margin
            for scenario in ("pure_null", "frozen_rule", "order_invariant_memory")
        )
        and _value(summary, "frozen_rule", "r_frozen", "ci_high") < config.frozen_ratio_threshold
    )
    ood_superiority = bool(
        _value(summary, aligned, "r_ood", "ci_high") <= config.ood_ratio_threshold
    )
    capacity_robustness = bool(
        _value(summary, aligned, "r_capacity_4x", "ci_high") <= config.ood_ratio_threshold
    )
    training_stability = bool(int(seeds["training_failures"].sum()) == 0)
    residual_robustness = bool(
        _value(summary, "residual_state_imbalance", "state_max_smd", "ci_low") > config.state_smd_margin
        and _value(summary, "residual_state_imbalance", "state_permutation_reject", "ci_low") > 0.80
        and _value(summary, "residual_state_imbalance", "abs_psi_update", "ci_high") <= config.negative_update_margin
    )
    control_competence = bool(
        _value(summary, aligned, "control_random_gain_hmpc", "ci_low")
        >= config.control_random_gain_threshold
    )
    gates = {
        "G0_data_integrity": integrity,
        "G1_observable_state_equivalence": state_equivalence,
        "G2_nested_tuning_isolation": optimization_isolation,
        "G3_nominal_parameter_fairness": capacity_fairness,
        "G4_mechanism_positive_control": mechanism_positive,
        "G5_negative_control_specificity": negative_specificity,
        "G6_OOD_superiority": ood_superiority,
        "G7_4x_capacity_robustness": capacity_robustness,
        "G8_training_stability": training_stability,
        "G9_residual_imbalance_robustness": residual_robustness,
        "G10_history_mpc_control_competence": control_competence,
    }
    if not integrity or not optimization_isolation or not training_stability:
        classification = "F_implementation_failure"
    elif not state_equivalence or not capacity_fairness or not control_competence:
        classification = "E_unidentifiable_or_unfair_comparison"
    elif not mechanism_positive:
        classification = "D_SRRD_mechanism_falsified"
    elif ood_superiority and capacity_robustness and negative_specificity and residual_robustness:
        classification = "A_hard_operationally_identified"
    elif ood_superiority and not capacity_robustness:
        classification = "C_capacity_sensitive_not_unique"
    else:
        classification = "B_history_survives_SRRD_not_uniquely_required"
    return {"gates": gates, "classification": classification}


def run_phase2b_hard(
    config: Phase2BHardConfig,
    output_dir: Path,
    *,
    preregistration_path: Path,
    freeze_path: Path,
    require_frozen: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze = assert_frozen(preregistration_path, freeze_path) if require_frozen else None
    failures: list[dict[str, object]] = []
    rows: list[dict[str, float | int | str]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for seed in range(_n_seeds(config, scenario.name)):
            try:
                rows.append(_run_seed(config, scenario_index, seed))
            except Exception as exc:
                failures.append({"scenario": scenario.name, "seed": seed, "error": repr(exc)})
                if require_frozen:
                    break
        if failures and require_frozen:
            break

    failure_path = output_dir / "phase2b_hard_failures.json"
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        result = {
            "classification": "F_implementation_failure",
            "gates": {"G0_data_integrity": False, "G8_training_stability": False},
            "failure_count": len(failures),
        }
        (output_dir / "phase2b_hard_gates.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return result

    seeds = pd.DataFrame(rows)
    summary = _summarize(config, seeds)
    decision = evaluate_gates(config, seeds, summary)
    seeds.to_csv(output_dir / "phase2b_hard_seed_metrics.csv.gz", index=False, compression="gzip")
    summary.to_csv(output_dir / "phase2b_hard_summary.csv", index=False)
    (output_dir / "phase2b_hard_gates.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "study_id": "SRRD-P2B-HARD-2026-08-11",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "preregistration_sha256": sha256(preregistration_path),
        "freeze": freeze,
        "allowed_model_inputs": ["history", "x_obs", "c1", "u2"],
        "forbidden_model_inputs": ["true_rule", "latent_rule", "slow_state", "scenario_mechanism"],
        "ood_used_for_tuning": False,
        "primary_endpoint": "multi-horizon standardized Gaussian NLL",
        "scientific_wording_before_run": "history matters; SRRD is not uniquely required",
    }
    (output_dir / "phase2b_hard_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return decision
