"""Minimal state-matched, two-timescale model for SRRD Phase 1.

The model deliberately separates three levels:

* ``q``: a slow, history-conditioned rule target (mechanism level),
* ``theta``: the working distinction rule reconstructed after intervention,
* ``p(y=1|x)``: observable predictions produced by ``theta``.

The common intervention resets the fast state and working rule, but not ``q``.
This makes the pre-recovery observable state exactly matched while preserving a
testable, latent history difference in the synthetic ground-truth model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


HISTORY_AB = ("A", "B", "C")
HISTORY_BA = ("B", "A", "C")


@dataclass(frozen=True)
class Scenario:
    """One preregistered synthetic data-generating scenario."""

    name: str
    label: str
    base_rule: tuple[float, float]
    history_direction: tuple[float, float]
    history_drive: float
    recovery_rate: float
    expected_class: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="aligned_positive",
        label="Aligned positive control",
        base_rule=(1.60, 0.0),
        history_direction=(1.0, 0.0),
        history_drive=1.50,
        recovery_rate=0.20,
        expected_class="high_srrd_high_cvp",
    ),
    Scenario(
        name="orthogonal_counterexample",
        label="Orthogonal counterexample",
        base_rule=(0.0, 0.0),
        history_direction=(0.0, 1.0),
        history_drive=2.50,
        recovery_rate=0.20,
        expected_class="high_srrd_low_cvp",
    ),
    Scenario(
        name="order_invariant_viable",
        label="Order-invariant viable control",
        base_rule=(1.60, 0.0),
        history_direction=(0.0, 0.0),
        history_drive=0.0,
        recovery_rate=0.20,
        expected_class="low_srrd_high_cvp",
    ),
    Scenario(
        name="history_shuffle",
        label="History-shuffle negative control",
        base_rule=(0.0, 0.0),
        history_direction=(0.0, 0.0),
        history_drive=0.0,
        recovery_rate=0.20,
        expected_class="low_srrd_low_cvp",
    ),
    Scenario(
        name="frozen_rule",
        label="Frozen-rule ablation",
        base_rule=(1.60, 0.0),
        history_direction=(1.0, 0.0),
        history_drive=1.50,
        recovery_rate=0.0,
        expected_class="separation_without_reconstruction",
    ),
    Scenario(
        name="flat_state_matched",
        label="State-matched flat-state baseline",
        base_rule=(0.0, 0.0),
        history_direction=(0.0, 0.0),
        history_drive=0.0,
        recovery_rate=0.0,
        expected_class="low_srrd_low_cvp",
    ),
)


def recursive_history_code(
    history: tuple[str, ...],
    *,
    contraction: float,
    drive: float,
    common_suffix_decay: float,
) -> float:
    """Encode order with recursive affine updates and a common suffix.

    ``A`` and ``B`` occur once in both histories, and both histories end in the
    same ``C`` suffix. Therefore a non-zero difference cannot be attributed to
    task counts or the immediately preceding task.
    """

    z = 0.0
    for token in history:
        if token == "A":
            z = contraction * z + drive
        elif token == "B":
            z = contraction * z - drive
        elif token == "C":
            z = common_suffix_decay * z
        else:
            raise ValueError(f"Unknown history token: {token}")
    return float(z)


def history_conditioned_rule_targets(
    scenario: Scenario,
    *,
    contraction: float,
    common_suffix_decay: float,
    drive_multiplier: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return slow rule targets for matched AB-C and BA-C histories."""

    base = np.asarray(scenario.base_rule, dtype=float)
    direction = np.asarray(scenario.history_direction, dtype=float)
    drive = scenario.history_drive * drive_multiplier
    z_ab = recursive_history_code(
        HISTORY_AB,
        contraction=contraction,
        drive=drive,
        common_suffix_decay=common_suffix_decay,
    )
    z_ba = recursive_history_code(
        HISTORY_BA,
        contraction=contraction,
        drive=drive,
        common_suffix_decay=common_suffix_decay,
    )
    return base + direction * z_ab, base + direction * z_ba


def reconstruct_rule(
    target: np.ndarray,
    *,
    horizon: int,
    recovery_rate: float,
) -> np.ndarray:
    """Reconstruct a working rule after the common ``do(theta=0)`` reset."""

    path = np.zeros((horizon + 1, target.size), dtype=float)
    for step in range(horizon):
        path[step + 1] = path[step] + recovery_rate * (target - path[step])
    return path


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def predict_probabilities(
    probes: np.ndarray,
    rule_path: np.ndarray,
    *,
    inverse_temperature: float,
) -> np.ndarray:
    """Return ``p(y=1|x)`` for every recovery step and probe."""

    logits = inverse_temperature * (rule_path @ probes.T)
    return sigmoid(logits)

