"""Capacity-controlled predictive models for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import build_features


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_dim: int
    family: str


def model_specs(nominal_budget: int) -> tuple[ModelSpec, ...]:
    return (
        ModelSpec("markov_ssm", nominal_budget, "current_state"),
        ModelSpec("flat_rnn_0_5x", nominal_budget // 2, "flat_recurrent"),
        ModelSpec("flat_rnn_1x", nominal_budget, "flat_recurrent"),
        ModelSpec("flat_rnn_2x", nominal_budget * 2, "flat_recurrent"),
        ModelSpec("flat_rnn_4x", nominal_budget * 4, "flat_recurrent"),
        ModelSpec("adaptive_psr", nominal_budget, "predictive_state"),
        ModelSpec("history_mpc", nominal_budget, "history_control"),
        ModelSpec("srrd_bilevel", nominal_budget, "bilevel_reconstruction"),
    )


def _feature_dispatch_name(spec_name: str) -> str:
    return "flat_rnn" if spec_name.startswith("flat_rnn") else spec_name


class RidgePredictor:
    def __init__(self, spec: ModelSpec, *, ridge: float, sigma_floor: float) -> None:
        self.spec = spec
        self.ridge = float(ridge)
        self.sigma_floor = float(sigma_floor)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.sigma_: np.ndarray | None = None

    def _features(
        self, data: dict[str, np.ndarray], *, frozen_update: bool = False
    ) -> np.ndarray:
        return build_features(
            _feature_dispatch_name(self.spec.name),
            data,
            self.spec.feature_dim,
            frozen_update=frozen_update,
        )

    def fit(self, data: dict[str, np.ndarray]) -> "RidgePredictor":
        features = self._features(data)
        self.mean_ = features.mean(axis=0)
        self.scale_ = features.std(axis=0, ddof=0)
        self.scale_ = np.where(self.scale_ < 1e-9, 1.0, self.scale_)
        design = (features - self.mean_) / self.scale_
        design = np.column_stack([np.ones(design.shape[0]), design])
        penalty = np.eye(design.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(design.T @ design + penalty, design.T @ data["y"])
        residual = data["y"] - design @ self.coef_
        self.sigma_ = np.maximum(np.sqrt(np.mean(residual**2, axis=0)), self.sigma_floor)
        return self

    def predict(
        self, data: dict[str, np.ndarray], *, frozen_update: bool = False
    ) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coef_ is None:
            raise RuntimeError("model is not fitted")
        features = self._features(data, frozen_update=frozen_update)
        design = (features - self.mean_) / self.scale_
        design = np.column_stack([np.ones(design.shape[0]), design])
        return design @ self.coef_

    def standardized_nll(
        self, data: dict[str, np.ndarray], *, frozen_update: bool = False
    ) -> float:
        if self.sigma_ is None:
            raise RuntimeError("model is not fitted")
        prediction = self.predict(data, frozen_update=frozen_update)
        z = (data["y"] - prediction) / self.sigma_[None, :]
        values = 0.5 * np.log(2.0 * np.pi * self.sigma_[None, :] ** 2) + 0.5 * z**2
        return float(np.mean(values))

    def trainable_parameter_count(self, horizon: int) -> int:
        return int((self.spec.feature_dim + 1) * horizon + horizon)
