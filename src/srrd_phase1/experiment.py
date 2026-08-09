"""Monte Carlo experiment and preregistered falsification gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import cvp_metric, srrd_mechanism_metrics, sriv_metric
from .model import (
    SCENARIOS,
    Scenario,
    history_conditioned_rule_targets,
    predict_probabilities,
    reconstruct_rule,
)


@dataclass(frozen=True)
class ExperimentConfig:
    n_seeds: int = 400
    horizon: int = 24
    n_diagnostic_probes: int = 768
    n_viability_probes: int = 768
    history_contraction: float = 0.40
    common_suffix_decay: float = 0.90
    inverse_temperature: float = 1.50
    drive_log_sd: float = 0.05
    recovery_rate_sd: float = 0.01
    bootstrap_replicates: int = 4000
    state_match_tolerance: float = 1e-12
    rule_separation_threshold: float = 1.00
    recovery_fraction_threshold: float = 0.70
    sriv_threshold: float = 0.02
    cvp_positive_threshold: float = 0.10
    cvp_equivalence_margin: float = 0.02
    random_seed: int = 20260809


def _seed_rng(master_seed: int, scenario_index: int, seed: int, stream: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([master_seed, scenario_index, seed, stream])
    return np.random.default_rng(sequence)


def _run_seed(
    config: ExperimentConfig,
    scenario: Scenario,
    scenario_index: int,
    seed: int,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    parameter_rng = _seed_rng(config.random_seed, scenario_index, seed, 0)
    diagnostic_rng = _seed_rng(config.random_seed, scenario_index, seed, 1)
    viability_rng = _seed_rng(config.random_seed, scenario_index, seed, 2)

    drive_multiplier = float(np.exp(parameter_rng.normal(0.0, config.drive_log_sd)))
    recovery_rate = float(
        np.clip(
            scenario.recovery_rate
            + parameter_rng.normal(0.0, config.recovery_rate_sd)
            if scenario.recovery_rate > 0.0
            else 0.0,
            0.0,
            0.95,
        )
    )
    target_ab, target_ba = history_conditioned_rule_targets(
        scenario,
        contraction=config.history_contraction,
        common_suffix_decay=config.common_suffix_decay,
        drive_multiplier=drive_multiplier,
    )
    path_ab = reconstruct_rule(
        target_ab,
        horizon=config.horizon,
        recovery_rate=recovery_rate,
    )
    path_ba = reconstruct_rule(
        target_ba,
        horizon=config.horizon,
        recovery_rate=recovery_rate,
    )

    mechanism = srrd_mechanism_metrics(target_ab, target_ba, path_ab, path_ba)

    diagnostic_probes = diagnostic_rng.normal(
        0.0, 1.0, size=(config.n_diagnostic_probes, 2)
    )
    diagnostic_ab = predict_probabilities(
        diagnostic_probes,
        path_ab,
        inverse_temperature=config.inverse_temperature,
    )
    diagnostic_ba = predict_probabilities(
        diagnostic_probes,
        path_ba,
        inverse_temperature=config.inverse_temperature,
    )
    state_match_error = float(np.max(np.abs(diagnostic_ab[0] - diagnostic_ba[0])))
    sriv = sriv_metric(diagnostic_ab[1:], diagnostic_ba[1:])

    # CVP sees a distinct held-out task distribution: x2 is exactly zero and
    # labels depend only on x1. The orthogonal counterexample reconstructs rules
    # entirely in the x2 direction, which is in the CVP observation kernel.
    x1 = viability_rng.normal(0.0, 1.0, size=config.n_viability_probes)
    viability_probes = np.column_stack([x1, np.zeros_like(x1)])
    labels_one_step = (x1 > 0.0).astype(float)
    labels = np.broadcast_to(labels_one_step, (config.horizon, x1.size))
    full_ab = predict_probabilities(
        viability_probes,
        path_ab[1:],
        inverse_temperature=config.inverse_temperature,
    )
    full_ba = predict_probabilities(
        viability_probes,
        path_ba[1:],
        inverse_temperature=config.inverse_temperature,
    )
    ablated = np.full_like(full_ab, 0.5)
    cvp_ab = cvp_metric(labels, full_ab, ablated)
    cvp_ba = cvp_metric(labels, full_ba, ablated)
    cvp = float(0.5 * (cvp_ab + cvp_ba))

    seed_row: dict[str, float | int | str] = {
        "scenario": scenario.name,
        "scenario_label": scenario.label,
        "expected_class": scenario.expected_class,
        "seed": seed,
        "drive_multiplier": drive_multiplier,
        "recovery_rate": recovery_rate,
        "rule_separation": mechanism["rule_separation"],
        "recovery_fraction": mechanism["recovery_fraction"],
        "sriv": sriv,
        "cvp": cvp,
        "cvp_ab": cvp_ab,
        "cvp_ba": cvp_ba,
        "state_match_error": state_match_error,
        "target_ab_1": float(target_ab[0]),
        "target_ab_2": float(target_ab[1]),
        "target_ba_1": float(target_ba[0]),
        "target_ba_2": float(target_ba[1]),
    }
    history_direction = np.asarray(scenario.history_direction, dtype=float)
    direction_norm = float(np.linalg.norm(history_direction))
    if direction_norm > 0.0:
        history_axis = history_direction / direction_norm
    else:
        history_axis = np.zeros_like(history_direction)
    trajectory_rows: list[dict[str, float | int | str]] = []
    for step in range(config.horizon + 1):
        trajectory_rows.append(
            {
                "scenario": scenario.name,
                "scenario_label": scenario.label,
                "seed": seed,
                "step": step,
                "rule_norm_ab": float(np.linalg.norm(path_ab[step])),
                "rule_norm_ba": float(np.linalg.norm(path_ba[step])),
                "target_norm_ab": float(np.linalg.norm(target_ab)),
                "target_norm_ba": float(np.linalg.norm(target_ba)),
                "history_axis_rule_ab": float(path_ab[step] @ history_axis),
                "history_axis_rule_ba": float(path_ba[step] @ history_axis),
                "history_axis_target_ab": float(target_ab @ history_axis),
                "history_axis_target_ba": float(target_ba @ history_axis),
            }
        )
    return seed_row, trajectory_rows


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap values must be a non-empty 1D array")
    if np.all(values == values[0]):
        point = float(values[0])
        return point, point, point
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    means = values[indices].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(values.mean()), float(low), float(high)


def _summarize(config: ExperimentConfig, seeds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    metrics = ("rule_separation", "recovery_fraction", "sriv", "cvp", "state_match_error")
    for scenario_index, scenario in enumerate(SCENARIOS):
        frame = seeds.loc[seeds["scenario"] == scenario.name]
        for metric_index, metric in enumerate(metrics):
            rng = _seed_rng(config.random_seed, scenario_index, metric_index, 99)
            mean, low, high = bootstrap_mean_ci(
                frame[metric].to_numpy(float),
                replicates=config.bootstrap_replicates,
                rng=rng,
            )
            rows.append(
                {
                    "scenario": scenario.name,
                    "scenario_label": scenario.label,
                    "expected_class": scenario.expected_class,
                    "metric": metric,
                    "n": int(frame.shape[0]),
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def _lookup(summary: pd.DataFrame, scenario: str, metric: str) -> pd.Series:
    row = summary.loc[
        (summary["scenario"] == scenario) & (summary["metric"] == metric)
    ]
    if row.shape[0] != 1:
        raise RuntimeError(f"Expected one summary row for {scenario}/{metric}")
    return row.iloc[0]


def evaluate_gates(config: ExperimentConfig, summary: pd.DataFrame) -> dict[str, object]:
    def lower(scenario: str, metric: str) -> float:
        return float(_lookup(summary, scenario, metric)["ci_low"])

    def upper(scenario: str, metric: str) -> float:
        return float(_lookup(summary, scenario, metric)["ci_high"])

    margin = config.cvp_equivalence_margin
    gates = {
        "G0_state_matching": upper("orthogonal_counterexample", "state_match_error")
        <= config.state_match_tolerance,
        "G1_aligned_high_SRRD": (
            lower("aligned_positive", "rule_separation")
            >= config.rule_separation_threshold
            and lower("aligned_positive", "recovery_fraction")
            >= config.recovery_fraction_threshold
        ),
        "G2_counterexample_high_SRRD": (
            lower("orthogonal_counterexample", "rule_separation")
            >= config.rule_separation_threshold
            and lower("orthogonal_counterexample", "recovery_fraction")
            >= config.recovery_fraction_threshold
        ),
        "G3_counterexample_has_SRIV": lower("orthogonal_counterexample", "sriv")
        >= config.sriv_threshold,
        "G4_counterexample_low_CVP": (
            lower("orthogonal_counterexample", "cvp") > -margin
            and upper("orthogonal_counterexample", "cvp") < margin
        ),
        "G5_aligned_positive_CVP": lower("aligned_positive", "cvp")
        >= config.cvp_positive_threshold,
        "G6_converse_separation": (
            upper("order_invariant_viable", "rule_separation") < 0.10
            and lower("order_invariant_viable", "cvp")
            >= config.cvp_positive_threshold
        ),
        "G7_frozen_rule_fails_reconstruction": (
            upper("frozen_rule", "recovery_fraction") < 0.10
            and lower("frozen_rule", "cvp") > -margin
            and upper("frozen_rule", "cvp") < margin
        ),
    }
    return {
        "all_gates_pass": bool(all(gates.values())),
        "gates": gates,
        "interpretation": (
            "The unconditional implication SRRD -> CVP is falsified by construction; "
            "the hierarchy survives only with an explicit observation/viability coupling condition."
        ),
    }


def _trajectory_summary(trajectories: pd.DataFrame) -> pd.DataFrame:
    grouped = trajectories.groupby(["scenario", "scenario_label", "step"], as_index=False)
    return grouped.agg(
        rule_norm_ab=("rule_norm_ab", "mean"),
        rule_norm_ba=("rule_norm_ba", "mean"),
        target_norm_ab=("target_norm_ab", "mean"),
        target_norm_ba=("target_norm_ba", "mean"),
        history_axis_rule_ab=("history_axis_rule_ab", "mean"),
        history_axis_rule_ba=("history_axis_rule_ba", "mean"),
        history_axis_target_ab=("history_axis_target_ab", "mean"),
        history_axis_target_ba=("history_axis_target_ba", "mean"),
    )


def run_experiment(config: ExperimentConfig, output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict[str, float | int | str]] = []
    trajectory_rows: list[dict[str, float | int | str]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for seed in range(config.n_seeds):
            seed_row, seed_trajectories = _run_seed(
                config, scenario, scenario_index, seed
            )
            seed_rows.append(seed_row)
            trajectory_rows.extend(seed_trajectories)

    seeds = pd.DataFrame(seed_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    summary = _summarize(config, seeds)
    trajectory_summary = _trajectory_summary(trajectories)
    gates = evaluate_gates(config, summary)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    metadata = {
        "generated_at": generated_at,
        "config": asdict(config),
        "scenario_count": len(SCENARIOS),
        "seed_rows": int(seeds.shape[0]),
        "trajectory_rows": int(trajectories.shape[0]),
        "independent_random_streams": ["parameters", "diagnostic", "viability"],
    }

    seeds.to_csv(output / "phase1_seed_metrics.csv", index=False)
    summary.to_csv(output / "phase1_summary.csv", index=False)
    trajectory_summary.to_csv(output / "phase1_trajectories.csv", index=False)
    (output / "phase1_gates.json").write_text(
        json.dumps(gates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "phase1_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"metadata": metadata, "gates": gates}
