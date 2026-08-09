"""Strictly separated mechanism, phenomenon, and observable metrics."""

from __future__ import annotations

import numpy as np


EPS = 1e-12


def srrd_mechanism_metrics(
    rule_target_ab: np.ndarray,
    rule_target_ba: np.ndarray,
    rule_path_ab: np.ndarray,
    rule_path_ba: np.ndarray,
) -> dict[str, float]:
    """Measure SRRD using rule-space variables only.

    This function cannot access probes, predictions, labels, utilities, or CVP.
    A system is classified as high-SRRD only if it has both rule-level order
    separation and post-reset rule reconstruction.
    """

    rule_separation = float(np.linalg.norm(rule_target_ab - rule_target_ba))
    fractions: list[float] = []
    for target, path in (
        (rule_target_ab, rule_path_ab),
        (rule_target_ba, rule_path_ba),
    ):
        initial_error = float(np.linalg.norm(target - path[0]))
        if initial_error <= EPS:
            fractions.append(0.0)
            continue
        residual = np.linalg.norm(target[None, :] - path[1:], axis=1)
        fractions.append(float(1.0 - np.mean(residual) / initial_error))
    return {
        "rule_separation": rule_separation,
        "recovery_fraction": float(np.mean(fractions)),
    }


def bernoulli_jsd(prob_ab: np.ndarray, prob_ba: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence for Bernoulli predictions, normalized to [0,1]."""

    p = np.clip(np.asarray(prob_ab, dtype=float), EPS, 1.0 - EPS)
    q = np.clip(np.asarray(prob_ba, dtype=float), EPS, 1.0 - EPS)
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return left * np.log(left / right) + (1.0 - left) * np.log(
            (1.0 - left) / (1.0 - right)
        )

    return 0.5 * (kl(p, midpoint) + kl(q, midpoint)) / np.log(2.0)


def sriv_metric(prob_ab: np.ndarray, prob_ba: np.ndarray) -> float:
    """State-matched rule-induced intervention variance.

    SRIV uses diagnostic predictions only. It does not access viability labels,
    rewards, losses, or latent rule variables.
    """

    if prob_ab.shape != prob_ba.shape:
        raise ValueError("SRIV probability arrays must have identical shapes")
    return float(np.mean(bernoulli_jsd(prob_ab, prob_ba)))


def binary_cross_entropy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(labels, dtype=float)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def cvp_metric(
    labels: np.ndarray,
    full_probabilities: np.ndarray,
    ablated_probabilities: np.ndarray,
) -> float:
    """Counterfactual viability preservation from observable arrays only.

    CVP is the relative reduction in held-out log loss versus the no-recovery
    ablation. This function cannot access history codes, slow rules, working
    rules, or SRRD/SRIV values.
    """

    baseline_loss = binary_cross_entropy(labels, ablated_probabilities)
    full_loss = binary_cross_entropy(labels, full_probabilities)
    if baseline_loss <= EPS:
        raise ValueError("CVP baseline loss must be positive")
    return float((baseline_loss - full_loss) / baseline_loss)

