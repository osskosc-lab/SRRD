"""Secondary actual-control audit for the learned History-MPC baseline.

The evaluator owns the simulator mechanism; candidate models only receive the public
observable dictionary.  The primary OOD prediction endpoint remains frozen at C2=1.2.
"""
from __future__ import annotations

import numpy as np

from srrd_phase2.generator import SCENARIOS


def _slow_history_code(history: np.ndarray, decay: float = 0.72) -> np.ndarray:
    state = np.zeros(history.shape[0], dtype=float)
    for step in range(history.shape[1]):
        state = decay * state + history[:, step]
    return state


def counterfactual_mean(
    scenario_name: str,
    data: dict[str, np.ndarray],
    actions: np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    scenario = next(s for s in SCENARIOS if s.name == scenario_name)
    x_obs = np.asarray(data["x_obs"], dtype=float)
    c1 = np.asarray(data["c1"], dtype=float)
    actions = np.asarray(actions, dtype=float)
    history = np.asarray(data["history"], dtype=float)
    private_slow = _slow_history_code(history)
    k = np.arange(1, horizon + 1, dtype=float)
    fast_decay = 0.58**k
    response_profile = 0.78 + 0.055 * k
    base_scalar = (
        0.65 * x_obs[:, 0]
        + 0.25 * x_obs[:, 1]
        + 0.10 * x_obs[:, 4]
        + 0.15 * actions
        + 0.08 * x_obs[:, 0] * actions
    )
    y = base_scalar[:, None] * fast_decay[None, :]
    y += 0.08 * np.sin(x_obs[:, 2])[:, None]
    y += 0.05 * c1[:, None]
    if scenario.mechanism == "adaptive_rule":
        coupling = float(np.cos(np.deg2rad(scenario.observation_angle_degrees)))
        private_post = private_slow * (1.0 + 0.80 * c1)
        y += coupling * 0.65 * private_post[:, None] * actions[:, None] * response_profile[None, :]
    elif scenario.mechanism == "frozen_rule":
        y += 0.65 * private_slow[:, None] * actions[:, None] * response_profile[None, :]
    elif scenario.mechanism == "order_invariant":
        order_free = np.mean(np.abs(history), axis=1)
        y += 0.25 * order_free[:, None] * actions[:, None] * response_profile[None, :]
    elif scenario.mechanism == "flat_markov":
        nonlinear = 0.35 * np.sin(1.4 * x_obs[:, 0]) * actions + 0.25 * x_obs[:, 2] * x_obs[:, 3]
        y += nonlinear[:, None] * response_profile[None, :]
    elif scenario.mechanism == "persistent_history":
        y += 0.55 * np.sign(private_slow)[:, None] * actions[:, None] * response_profile[None, :]
    elif scenario.mechanism in {"residual_imbalance", "pure_null"}:
        pass
    else:
        raise ValueError(f"unknown mechanism: {scenario.mechanism}")
    return y


def realized_cost(
    scenario_name: str,
    data: dict[str, np.ndarray],
    actions: np.ndarray,
    *,
    horizon: int,
    action_penalty: float,
) -> np.ndarray:
    y = counterfactual_mean(scenario_name, data, actions, horizon=horizon)
    return np.mean(y**2, axis=1) + action_penalty * np.asarray(actions, dtype=float) ** 2


def control_audit(
    predictor,
    scenario_name: str,
    data: dict[str, np.ndarray],
    *,
    horizon: int,
    seed: int,
    action_grid: tuple[float, ...],
    action_penalty: float,
) -> dict[str, float]:
    grid = np.asarray(action_grid, dtype=float)
    selected = predictor.select_action(data, grid, action_penalty=action_penalty)
    selected_cost = realized_cost(
        scenario_name, data, selected, horizon=horizon, action_penalty=action_penalty
    )
    all_costs = np.stack(
        [
            realized_cost(
                scenario_name,
                data,
                np.full(data["u2"].shape, action),
                horizon=horizon,
                action_penalty=action_penalty,
            )
            for action in grid
        ],
        axis=1,
    )
    oracle_cost = np.min(all_costs, axis=1)
    rng = np.random.default_rng(np.random.SeedSequence([20260811, seed, 921]))
    random_actions = rng.choice(grid, size=data["u2"].shape[0])
    random_cost = realized_cost(
        scenario_name, data, random_actions, horizon=horizon, action_penalty=action_penalty
    )
    eps = 1e-9
    return {
        "control_cost": float(np.mean(selected_cost)),
        "control_oracle_cost": float(np.mean(oracle_cost)),
        "control_random_cost": float(np.mean(random_cost)),
        "control_regret": float(np.mean(selected_cost - oracle_cost)),
        "control_random_gain": float(np.mean(random_cost) / max(np.mean(selected_cost), eps)),
    }
