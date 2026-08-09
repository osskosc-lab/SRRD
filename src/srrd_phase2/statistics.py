"""Seed-level estimands, matching diagnostics, and bootstrap utilities."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def paired_state_metrics(
    data: dict[str, np.ndarray], *, permutations: int, seed: int
) -> dict[str, float]:
    x = np.asarray(data["x_obs"], dtype=float)
    group = np.asarray(data["history_group"], dtype=int)
    scale = x.std(axis=0, ddof=1)
    scale = np.where(scale < 1e-9, 1.0, scale)
    left = x[group == 1]
    right = x[group == -1]
    smd = np.abs(left.mean(axis=0) - right.mean(axis=0)) / scale

    xs = x / scale
    left_s = xs[group == 1]
    right_s = xs[group == -1]
    energy = float(
        2.0 * cdist(left_s, right_s).mean()
        - cdist(left_s, left_s).mean()
        - cdist(right_s, right_s).mean()
    )

    differences: list[np.ndarray] = []
    for pair in np.unique(data["pair_id"]):
        rows = np.flatnonzero(data["pair_id"] == pair)
        if rows.size != 2:
            raise ValueError("each matching pair must contain two observations")
        positive = rows[group[rows] == 1]
        negative = rows[group[rows] == -1]
        if positive.size != 1 or negative.size != 1:
            raise ValueError("matching pair must contain both histories")
        differences.append((x[positive[0]] - x[negative[0]]) / scale)
    diff = np.asarray(differences)
    observed = float(np.linalg.norm(diff.mean(axis=0)))
    rng = np.random.default_rng(np.random.SeedSequence([20260809, seed, 777]))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, diff.shape[0]))
    null = np.linalg.norm((signs @ diff) / diff.shape[0], axis=1)
    p_value = float((1.0 + np.sum(null >= observed)) / (permutations + 1.0))
    return {
        "state_max_smd": float(np.max(smd)),
        "state_energy_distance": max(energy, 0.0),
        "state_pair_permutation_p": p_value,
        "state_paired_mean_norm": observed,
    }


def update_interaction(data: dict[str, np.ndarray]) -> dict[str, float]:
    response = np.mean(data["y"], axis=1)
    group = data["history_group"]
    c1 = data["c1"]

    def mean_at(history: int, probe: int) -> float:
        mask = (group == history) & (c1 == probe)
        if not np.any(mask):
            raise ValueError("empty History x C1 cell")
        return float(response[mask].mean())

    psi = (mean_at(1, 1) - mean_at(1, 0)) - (
        mean_at(-1, 1) - mean_at(-1, 0)
    )
    z = group * (c1 - 0.5) * data["u2"]
    centered = z - z.mean()
    kappa = float(np.mean(centered * (response - response.mean())) / np.mean(centered**2))
    history_sham = mean_at(1, 0) - mean_at(-1, 0)
    history_probe = mean_at(1, 1) - mean_at(-1, 1)
    return {
        "psi_update": float(psi),
        "abs_psi_update": float(abs(psi)),
        "kappa_obs": kappa,
        "history_effect_sham": history_sham,
        "history_effect_probe": history_probe,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values must be finite and one-dimensional")
    if np.all(values == values[0]):
        point = float(values[0])
        return point, point, point
    rng = np.random.default_rng(np.random.SeedSequence([20260809, seed, 4000]))
    means = np.empty(replicates, dtype=float)
    block = 250
    for start in range(0, replicates, block):
        stop = min(start + block, replicates)
        index = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[index].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(values.mean()), float(low), float(high)
