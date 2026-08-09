"""Observable-only feature maps for the five Phase 2 model families."""

from __future__ import annotations

import hashlib

import numpy as np


def _stable_rng(key: str) -> np.random.Generator:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


def _fill_budget(raw: np.ndarray, target_dim: int, key: str) -> np.ndarray:
    if raw.shape[1] > target_dim:
        raise ValueError(f"raw feature map {key} exceeds fixed budget")
    if raw.shape[1] == target_dim:
        return raw
    needed = target_dim - raw.shape[1]
    rng = _stable_rng(f"fill:{key}:{raw.shape[1]}:{target_dim}")
    weights = rng.normal(0.0, 1.0 / np.sqrt(max(raw.shape[1], 1)), size=(raw.shape[1], needed))
    return np.column_stack([raw, np.tanh(raw @ weights)])


def _common(data: dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(data["x_obs"], dtype=float)
    c1 = np.asarray(data["c1"], dtype=float)[:, None]
    u2 = np.asarray(data["u2"], dtype=float)[:, None]
    return np.column_stack(
        [
            x,
            x**2,
            c1,
            u2,
            c1 * u2,
            x[:, [0]] * u2,
            x[:, [1]] * u2,
        ]
    )


def _small_common(data: dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(data["x_obs"], dtype=float)
    c1 = np.asarray(data["c1"], dtype=float)[:, None]
    u2 = np.asarray(data["u2"], dtype=float)[:, None]
    return np.column_stack([x, c1, u2, c1 * u2])


def _recursive_code(history: np.ndarray, decay: float) -> np.ndarray:
    state = np.zeros(history.shape[0], dtype=float)
    for step in range(history.shape[1]):
        state = decay * state + history[:, step]
    return state


def markov_features(data: dict[str, np.ndarray], target_dim: int) -> np.ndarray:
    return _fill_budget(_common(data), target_dim, "markov")


def reservoir_features(data: dict[str, np.ndarray], target_dim: int) -> np.ndarray:
    common = _small_common(data)
    hidden_dim = max(1, (target_dim - common.shape[1]) // 3)
    rng = _stable_rng(f"reservoir:{hidden_dim}")
    w_in = rng.normal(0.0, 0.65, size=hidden_dim)
    w_rec = rng.normal(0.0, 1.0, size=(hidden_dim, hidden_dim))
    radius = float(np.max(np.abs(np.linalg.eigvals(w_rec))))
    if radius > 0.0:
        w_rec *= 0.82 / radius
    bias = rng.normal(0.0, 0.10, size=hidden_dim)
    state = np.zeros((data["history"].shape[0], hidden_dim), dtype=float)
    for step in range(data["history"].shape[1]):
        state = np.tanh(
            data["history"][:, [step]] * w_in[None, :] + state @ w_rec.T + bias
        )
    u2 = data["u2"][:, None]
    c1 = data["c1"][:, None]
    raw = np.column_stack([common, state, state * u2, state * c1 * u2])
    return _fill_budget(raw, target_dim, f"reservoir:{target_dim}")


def adaptive_psr_features(data: dict[str, np.ndarray], target_dim: int) -> np.ndarray:
    history = np.asarray(data["history"], dtype=float)
    tests = [_recursive_code(history, decay) for decay in (0.25, 0.50, 0.75, 0.90)]
    tests.extend(
        [
            np.mean(history[:, : history.shape[1] // 2], axis=1),
            np.mean(history[:, history.shape[1] // 2 :], axis=1),
            np.mean(history[:, 1:] * history[:, :-1], axis=1),
        ]
    )
    raw = np.column_stack([_common(data), *tests])
    return _fill_budget(raw, target_dim, "adaptive_psr")


def history_mpc_features(data: dict[str, np.ndarray], target_dim: int) -> np.ndarray:
    history = np.asarray(data["history"], dtype=float)
    tail = history[:, -5:]
    first = np.mean(history[:, : history.shape[1] // 2], axis=1)
    second = np.mean(history[:, history.shape[1] // 2 :], axis=1)
    raw = np.column_stack([_common(data), tail, first, second])
    return _fill_budget(raw, target_dim, "history_mpc")


def srrd_features(
    data: dict[str, np.ndarray],
    target_dim: int,
    *,
    frozen_update: bool = False,
) -> np.ndarray:
    history = np.asarray(data["history"], dtype=float)
    r_pre = _recursive_code(history, 0.75)
    c1 = np.asarray(data["c1"], dtype=float)
    u2 = np.asarray(data["u2"], dtype=float)
    r_post = r_pre if frozen_update else r_pre * (1.0 + 0.65 * c1)
    raw = np.column_stack(
        [
            _common(data),
            r_pre,
            r_post,
            r_pre * u2,
            r_post * u2,
            c1 * r_pre,
        ]
    )
    # Keep the same filler map for the ablation so only the slow-update terms change.
    return _fill_budget(raw, target_dim, "srrd")


def build_features(
    model_name: str,
    data: dict[str, np.ndarray],
    target_dim: int,
    *,
    frozen_update: bool = False,
) -> np.ndarray:
    if model_name == "markov_ssm":
        return markov_features(data, target_dim)
    if model_name.startswith("flat_rnn"):
        return reservoir_features(data, target_dim)
    if model_name == "adaptive_psr":
        return adaptive_psr_features(data, target_dim)
    if model_name == "history_mpc":
        return history_mpc_features(data, target_dim)
    if model_name == "srrd_bilevel":
        return srrd_features(data, target_dim, frozen_update=frozen_update)
    raise ValueError(f"unknown model: {model_name}")
