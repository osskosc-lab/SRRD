"""End-to-end capacity-matched models for SRRD Phase 2B-hard.

Model code receives observables only.  Hyperparameters are selected inside the
training split; the frozen OOD split is never used for tuning or calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn


torch.set_num_threads(1)


@dataclass(frozen=True)
class TrainBudget:
    trials: int
    max_epochs: int
    patience: int
    calibration_fraction: float = 0.25


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    parameter_multiplier: float


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("flat_gru_1x", "flat_gru", 1.0),
    ModelSpec("flat_lstm_1x", "flat_lstm", 1.0),
    ModelSpec("learned_psr_1x", "learned_psr", 1.0),
    ModelSpec("history_mpc_1x", "history_mpc", 1.0),
    ModelSpec("srrd_bilevel_e2e", "srrd_bilevel", 1.0),
    ModelSpec("flat_gru_4x", "flat_gru", 4.0),
    ModelSpec("flat_lstm_4x", "flat_lstm", 4.0),
)

NOMINAL_BASELINES = (
    "flat_gru_1x",
    "flat_lstm_1x",
    "learned_psr_1x",
    "history_mpc_1x",
)
CAPACITY_BASELINES = ("flat_gru_4x", "flat_lstm_4x")
TARGET_NAME = "srrd_bilevel_e2e"


def _context(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(data["x_obs"], dtype=np.float32),
            np.asarray(data["c1"], dtype=np.float32),
            np.asarray(data["u2"], dtype=np.float32),
        ]
    )


def _to_tensors(data: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history = torch.as_tensor(np.asarray(data["history"], dtype=np.float32)[:, :, None])
    context = torch.as_tensor(_context(data))
    target = torch.as_tensor(np.asarray(data["y"], dtype=np.float32))
    return history, context, target


class FlatGRU(nn.Module):
    def __init__(self, hidden: int, horizon: int) -> None:
        super().__init__()
        self.history = nn.GRU(1, hidden, batch_first=True)
        self.context = nn.Sequential(nn.Linear(7, hidden), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, horizon))

    def forward(self, history: torch.Tensor, context: torch.Tensor, *, frozen_update: bool = False) -> torch.Tensor:
        del frozen_update
        _, h = self.history(history)
        z = torch.cat([h[-1], self.context(context)], dim=1)
        return self.head(z)


class FlatLSTM(nn.Module):
    def __init__(self, hidden: int, horizon: int) -> None:
        super().__init__()
        self.history = nn.LSTM(1, hidden, batch_first=True)
        self.context = nn.Sequential(nn.Linear(7, hidden), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, horizon))

    def forward(self, history: torch.Tensor, context: torch.Tensor, *, frozen_update: bool = False) -> torch.Tensor:
        del frozen_update
        _, (h, _) = self.history(history)
        z = torch.cat([h[-1], self.context(context)], dim=1)
        return self.head(z)


class LearnedPSR(nn.Module):
    """Learn a predictive-state bottleneck directly from ordered history + observables."""

    def __init__(self, hidden: int, horizon: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(16 + 7, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden),
            nn.Tanh(),
        )
        self.transition = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, horizon)

    def forward(self, history: torch.Tensor, context: torch.Tensor, *, frozen_update: bool = False) -> torch.Tensor:
        del frozen_update
        flat = history.squeeze(-1)
        ps = self.encoder(torch.cat([flat, context], dim=1))
        ps = self.transition(torch.cat([ps, context[:, -2:]], dim=1))
        return self.head(ps)


class HistoryMPC(nn.Module):
    """Action-conditioned learned dynamics with an explicit multi-step rollout."""

    def __init__(self, hidden: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.history = nn.GRU(1, hidden, batch_first=True)
        self.state = nn.Sequential(nn.Linear(5, hidden), nn.Tanh())
        self.init = nn.Linear(2 * hidden, hidden)
        self.cell = nn.GRUCell(2, hidden)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, history: torch.Tensor, context: torch.Tensor, *, frozen_update: bool = False) -> torch.Tensor:
        del frozen_update
        _, h = self.history(history)
        fast = self.state(context[:, :5])
        z = torch.tanh(self.init(torch.cat([h[-1], fast], dim=1)))
        intervention = context[:, -2:]
        outputs: list[torch.Tensor] = []
        for _ in range(self.horizon):
            z = self.cell(intervention, z)
            outputs.append(self.readout(z))
        return torch.cat(outputs, dim=1)


class SRRDBilevel(nn.Module):
    """Explicit fast-state / slow-rule decomposition with a learnable update gate."""

    def __init__(self, hidden: int, horizon: int) -> None:
        super().__init__()
        self.slow = nn.GRU(1, hidden, batch_first=True)
        self.fast = nn.Sequential(nn.Linear(5, hidden), nn.Tanh())
        self.delta = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(2 * hidden + 2, hidden), nn.Tanh(), nn.Linear(hidden, horizon))

    def forward(self, history: torch.Tensor, context: torch.Tensor, *, frozen_update: bool = False) -> torch.Tensor:
        _, h = self.slow(history)
        slow_pre = h[-1]
        intervention = context[:, -2:]
        update_input = torch.cat([slow_pre, intervention], dim=1)
        if frozen_update:
            slow_post = slow_pre
        else:
            slow_post = slow_pre + self.gate(update_input) * self.delta(update_input)
        fast = self.fast(context[:, :5])
        return self.head(torch.cat([fast, slow_post, intervention], dim=1))


def build_network(family: str, hidden: int, horizon: int) -> nn.Module:
    if family == "flat_gru":
        return FlatGRU(hidden, horizon)
    if family == "flat_lstm":
        return FlatLSTM(hidden, horizon)
    if family == "learned_psr":
        return LearnedPSR(hidden, horizon)
    if family == "history_mpc":
        return HistoryMPC(hidden, horizon)
    if family == "srrd_bilevel":
        return SRRDBilevel(hidden, horizon)
    raise ValueError(f"unknown family: {family}")


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def matched_hidden_size(family: str, target_params: int, horizon: int) -> tuple[int, int]:
    candidates: list[tuple[int, int, int]] = []
    for hidden in range(4, 129):
        count = parameter_count(build_network(family, hidden, horizon))
        candidates.append((abs(count - target_params), hidden, count))
    _, hidden, count = min(candidates)
    return hidden, count


def stratified_fit_calibration_indices(data: dict[str, np.ndarray], fraction: float) -> tuple[np.ndarray, np.ndarray]:
    group = np.asarray(data["history_group"], dtype=int)
    c1 = np.asarray(data["c1"], dtype=float)
    u2 = np.asarray(data["u2"], dtype=float)
    row_id = np.asarray(data["row_id"], dtype=int)
    fit: list[int] = []
    cal: list[int] = []
    for h in (-1, 1):
        for probe in (0.0, 1.0):
            for action in (-0.8, -0.4, 0.4, 0.8):
                idx = np.flatnonzero((group == h) & (c1 == probe) & np.isclose(u2, action))
                idx = idx[np.argsort(row_id[idx])]
                if idx.size == 0:
                    continue
                n_cal = max(1, int(round(fraction * idx.size)))
                n_cal = min(n_cal, max(1, idx.size - 1)) if idx.size > 1 else 1
                cal.extend(idx[-n_cal:].tolist())
                fit.extend(idx[:-n_cal].tolist())
    if not fit or not cal:
        order = np.argsort(row_id)
        cut = max(1, int(round((1.0 - fraction) * order.size)))
        fit = order[:cut].tolist()
        cal = order[cut:].tolist()
    return np.asarray(fit, dtype=int), np.asarray(cal, dtype=int)


def subset(data: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[idx] for key, value in data.items()}


class FittedPredictor:
    def __init__(self, model: nn.Module, sigma: np.ndarray, metadata: dict[str, float | int | str]) -> None:
        self.model = model.eval()
        self.sigma = np.asarray(sigma, dtype=float)
        self.metadata = metadata

    def predict(self, data: dict[str, np.ndarray], *, frozen_update: bool = False) -> np.ndarray:
        history, context, _ = _to_tensors(data)
        with torch.no_grad():
            out = self.model(history, context, frozen_update=frozen_update)
        return out.detach().cpu().numpy().astype(float)

    def standardized_nll(self, data: dict[str, np.ndarray], *, frozen_update: bool = False) -> float:
        pred = self.predict(data, frozen_update=frozen_update)
        target = np.asarray(data["y"], dtype=float)
        sigma = np.maximum(self.sigma, 1e-6)
        z = (target - pred) / sigma[None, :]
        values = 0.5 * np.log(2.0 * np.pi * sigma[None, :] ** 2) + 0.5 * z**2
        return float(np.mean(values))

    def select_action(self, data: dict[str, np.ndarray], action_grid: np.ndarray, *, action_penalty: float = 0.02) -> np.ndarray:
        costs = []
        for action in action_grid:
            candidate = {key: np.asarray(value).copy() for key, value in data.items()}
            candidate["u2"] = np.full(candidate["u2"].shape, float(action))
            prediction = self.predict(candidate)
            cost = np.mean(prediction**2, axis=1) + action_penalty * float(action) ** 2
            costs.append(cost)
        stacked = np.stack(costs, axis=1)
        return np.asarray(action_grid, dtype=float)[np.argmin(stacked, axis=1)]


def _fit_one(
    family: str,
    hidden: int,
    horizon: int,
    fit_data: dict[str, np.ndarray],
    cal_data: dict[str, np.ndarray],
    *,
    seed: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
) -> tuple[nn.Module, float, int]:
    torch.manual_seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    model = build_network(family, hidden, horizon)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    h_fit, c_fit, y_fit = _to_tensors(fit_data)
    h_cal, c_cal, y_cal = _to_tensors(cal_data)
    best_state = deepcopy(model.state_dict())
    best_loss = math.inf
    bad = 0
    epochs_run = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(h_fit, c_fit)
        loss = criterion(pred, y_fit)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            cal_loss = float(criterion(model(h_cal, c_cal), y_cal).item())
        epochs_run = epoch + 1
        if cal_loss < best_loss - 1e-7:
            best_loss = cal_loss
            best_state = deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_loss, epochs_run


def tune_predictor(
    spec: ModelSpec,
    data: dict[str, np.ndarray],
    *,
    horizon: int,
    nominal_target_params: int,
    budget: TrainBudget,
    sigma_floor: float,
    seed: int,
) -> FittedPredictor:
    target_params = int(round(nominal_target_params * spec.parameter_multiplier))
    hidden, actual_params = matched_hidden_size(spec.family, target_params, horizon)
    fit_idx, cal_idx = stratified_fit_calibration_indices(data, budget.calibration_fraction)
    fit_data = subset(data, fit_idx)
    cal_data = subset(data, cal_idx)
    grid = (
        (1e-3, 0.0),
        (3e-3, 0.0),
        (1e-3, 1e-4),
        (3e-3, 1e-4),
        (5e-4, 1e-4),
        (2e-3, 5e-4),
    )
    if budget.trials > len(grid):
        raise ValueError("requested tuning trials exceed frozen grid")
    best: tuple[float, nn.Module, float, float, int, int] | None = None
    for trial, (lr, wd) in enumerate(grid[: budget.trials]):
        model, val_loss, epochs_run = _fit_one(
            spec.family,
            hidden,
            horizon,
            fit_data,
            cal_data,
            seed=seed * 100 + trial + 17,
            lr=lr,
            weight_decay=wd,
            max_epochs=budget.max_epochs,
            patience=budget.patience,
        )
        candidate = (val_loss, model, lr, wd, epochs_run, trial)
        if best is None or val_loss < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("no tuning trial completed")
    val_loss, model, lr, wd, epochs_run, trial = best
    h_cal, c_cal, y_cal = _to_tensors(cal_data)
    model.eval()
    with torch.no_grad():
        residual = y_cal - model(h_cal, c_cal)
    sigma = np.maximum(
        residual.detach().cpu().numpy().std(axis=0, ddof=0),
        sigma_floor,
    )
    metadata: dict[str, float | int | str] = {
        "family": spec.family,
        "hidden": hidden,
        "trainable_params": actual_params,
        "target_params": target_params,
        "selected_trial": trial,
        "selected_lr": lr,
        "selected_weight_decay": wd,
        "selected_validation_mse": val_loss,
        "selected_epochs": epochs_run,
        "tuning_trials": budget.trials,
        "max_epochs": budget.max_epochs,
        "n_fit": int(fit_idx.size),
        "n_calibration": int(cal_idx.size),
    }
    return FittedPredictor(model, sigma, metadata)
