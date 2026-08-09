"""Latent-blind synthetic generators for SRRD Phase 2.

The generator uses private slow variables internally, but returns observables only.
Model code receives histories, current observable state, interventions, and outcomes;
it never receives the private rule, slow state, or scenario mechanism parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeneratorScenario:
    name: str
    label: str
    mechanism: str
    expected_role: str
    observation_angle_degrees: float = 0.0
    residual_state_shift: float = 0.0


SCENARIOS: tuple[GeneratorScenario, ...] = (
    GeneratorScenario(
        "true_srrd_aligned",
        "True SRRD / aligned observation",
        "adaptive_rule",
        "positive_control",
        0.0,
    ),
    GeneratorScenario(
        "true_srrd_orthogonal",
        "True SRRD / orthogonal observation",
        "adaptive_rule",
        "observation_boundary",
        90.0,
    ),
    GeneratorScenario(
        "frozen_rule",
        "Frozen rule",
        "frozen_rule",
        "negative_control",
    ),
    GeneratorScenario(
        "order_invariant_memory",
        "Order-invariant memory",
        "order_invariant",
        "negative_control",
    ),
    GeneratorScenario(
        "flat_high_dim_markov",
        "Flat high-dimensional Markov latent",
        "flat_markov",
        "capacity_control",
    ),
    GeneratorScenario(
        "persistent_history_no_update",
        "Persistent history without second-order update",
        "persistent_history",
        "history_only_control",
    ),
    GeneratorScenario(
        "residual_state_imbalance",
        "Residual observable-state imbalance",
        "residual_imbalance",
        "matching_confound",
        residual_state_shift=0.24,
    ),
    GeneratorScenario(
        "pure_null",
        "Pure null",
        "pure_null",
        "negative_control",
    ),
)


PUBLIC_KEYS = frozenset(
    {
        "history",
        "x_obs",
        "c1",
        "u2",
        "y",
        "history_group",
        "pair_id",
        "row_id",
    }
)


def _history_template(group: int, rng: np.random.Generator) -> np.ndarray:
    """Return A->B->C or B->A->C with equal counts and common suffix."""

    if group not in (-1, 1):
        raise ValueError("history group must be -1 or 1")
    first = np.full(6, 1.0 if group == 1 else -1.0)
    second = -first.copy()
    suffix = np.zeros(4)
    values = np.concatenate([first, second, suffix])
    active = values != 0.0
    values[active] += rng.normal(0.0, 0.04, size=int(active.sum()))
    return values


def _slow_history_code(history: np.ndarray, decay: float = 0.72) -> np.ndarray:
    state = np.zeros(history.shape[0], dtype=float)
    for step in range(history.shape[1]):
        state = decay * state + history[:, step]
    return state


def _balanced_design(
    rng: np.random.Generator,
    n_rows: int,
    residual_state_shift: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_rows % 4:
        raise ValueError("n_rows must be divisible by four")
    quartets = n_rows // 4
    history_group = np.tile(np.array([1, 1, -1, -1], dtype=int), quartets)
    c1 = np.tile(np.array([0.0, 1.0, 0.0, 1.0]), quartets)
    base = rng.normal(0.0, 1.0, size=(quartets, 5))
    x_obs = np.repeat(base, 4, axis=0)
    x_obs += rng.normal(0.0, 0.04, size=x_obs.shape)
    if residual_state_shift:
        x_obs[:, 0] += residual_state_shift * history_group
    pair_id = np.empty(n_rows, dtype=int)
    for index in range(quartets):
        start = 4 * index
        pair_id[start : start + 4] = (2 * index, 2 * index + 1, 2 * index, 2 * index + 1)
    return history_group, c1, x_obs, pair_id


def generate_observables(
    scenario: GeneratorScenario,
    *,
    seed: int,
    n_rows: int,
    horizon: int,
    split: str,
    observation_angle_override: float | None = None,
) -> dict[str, np.ndarray]:
    """Generate a train or OOD test split containing observables only."""

    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    stream = 0 if split == "train" else 1
    rng = np.random.default_rng(np.random.SeedSequence([20260809, seed, stream]))
    history_group, c1, x_obs, pair_id = _balanced_design(
        rng, n_rows, scenario.residual_state_shift
    )
    history = np.stack([_history_template(int(group), rng) for group in history_group])

    if split == "train":
        u2 = rng.choice(np.array([-0.8, -0.4, 0.4, 0.8]), size=n_rows)
    else:
        u2 = np.full(n_rows, 1.2)

    private_slow = _slow_history_code(history)
    angle = (
        scenario.observation_angle_degrees
        if observation_angle_override is None
        else observation_angle_override
    )
    coupling = float(np.cos(np.deg2rad(angle)))
    k = np.arange(1, horizon + 1, dtype=float)
    fast_decay = 0.58**k
    response_profile = 0.78 + 0.055 * k
    base_scalar = (
        0.65 * x_obs[:, 0]
        + 0.25 * x_obs[:, 1]
        + 0.10 * x_obs[:, 4]
        + 0.15 * u2
        + 0.08 * x_obs[:, 0] * u2
    )
    y = base_scalar[:, None] * fast_decay[None, :]
    y += 0.08 * np.sin(x_obs[:, 2])[:, None]
    y += 0.05 * c1[:, None]

    if scenario.mechanism == "adaptive_rule":
        private_post = private_slow * (1.0 + 0.80 * c1)
        y += (
            coupling
            * 0.65
            * private_post[:, None]
            * u2[:, None]
            * response_profile[None, :]
        )
    elif scenario.mechanism == "frozen_rule":
        y += (
            0.65
            * private_slow[:, None]
            * u2[:, None]
            * response_profile[None, :]
        )
    elif scenario.mechanism == "order_invariant":
        order_free = np.mean(np.abs(history), axis=1)
        y += 0.25 * order_free[:, None] * u2[:, None] * response_profile[None, :]
    elif scenario.mechanism == "flat_markov":
        nonlinear = (
            0.35 * np.sin(1.4 * x_obs[:, 0]) * u2
            + 0.25 * x_obs[:, 2] * x_obs[:, 3]
        )
        y += nonlinear[:, None] * response_profile[None, :]
    elif scenario.mechanism == "persistent_history":
        y += (
            0.55
            * np.sign(private_slow)[:, None]
            * u2[:, None]
            * response_profile[None, :]
        )
    elif scenario.mechanism in {"residual_imbalance", "pure_null"}:
        pass
    else:
        raise ValueError(f"unknown mechanism: {scenario.mechanism}")

    y += rng.normal(0.0, 0.35, size=y.shape)
    row_id = np.arange(n_rows, dtype=int)
    order = rng.permutation(n_rows)
    result = {
        "history": history[order],
        "x_obs": x_obs[order],
        "c1": c1[order],
        "u2": u2[order],
        "y": y[order],
        "history_group": history_group[order],
        "pair_id": pair_id[order],
        "row_id": row_id[order],
    }
    if set(result) != PUBLIC_KEYS:
        raise RuntimeError("generator exposed an unexpected field")
    return result


def shuffled_history_copy(data: dict[str, np.ndarray], *, seed: int) -> dict[str, np.ndarray]:
    """Break within-row order while preserving every observed history value."""

    rng = np.random.default_rng(np.random.SeedSequence([20260809, seed, 991]))
    copied = {key: value.copy() for key, value in data.items()}
    for row in range(copied["history"].shape[0]):
        copied["history"][row] = copied["history"][row, rng.permutation(copied["history"].shape[1])]
    return copied
