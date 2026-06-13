"""Energy calibration and repair utilities."""

from __future__ import annotations

import numpy as np

from .toy_envs import ArrayDict


def calibration_features(pool: ArrayDict, energy: np.ndarray) -> np.ndarray:
    energy = np.asarray(energy, dtype=float)
    return np.column_stack(
        [
            np.ones(len(energy)),
            -energy,
            np.asarray(pool["visible_score"], dtype=float),
            np.asarray(pool["support_penalty"], dtype=float),
            np.asarray(pool["contact_penalty"], dtype=float),
            np.asarray(pool["jerk_penalty"], dtype=float),
            np.asarray(pool["shortcut_score"], dtype=float),
            np.asarray(pool["action_norm"], dtype=float),
        ]
    )


def fit_energy_calibrator(
    pool: ArrayDict,
    raw_energy: np.ndarray,
    pilot_size: int = 256,
    seed: int = 0,
    ridge: float = 1e-3,
) -> dict[str, np.ndarray | float | int]:
    """Fit a ridge linear map from energy/diagnostic features to real utility."""

    rng = np.random.default_rng(seed)
    n = len(raw_energy)
    if pilot_size < 4:
        raise ValueError("pilot_size must be at least 4")
    idx = rng.choice(n, size=min(pilot_size, n), replace=False)
    x = calibration_features(pool, raw_energy)[idx]
    y = np.asarray(pool["utility"], dtype=float)[idx]
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    pred = x @ coef
    mae = float(np.mean(np.abs(pred - y)))
    return {"coef": coef, "pilot_mae": mae, "pilot_size": int(len(idx))}


def apply_energy_calibrator(pool: ArrayDict, raw_energy: np.ndarray, calibrator: dict[str, np.ndarray | float | int]) -> np.ndarray:
    coef = np.asarray(calibrator["coef"], dtype=float)
    predicted_utility = calibration_features(pool, raw_energy) @ coef
    return -predicted_utility


def tail_aware_calibrated_energy(pool: ArrayDict, raw_energy: np.ndarray) -> np.ndarray:
    """A simple analytic tail-aware repair for toy ablations."""

    return (
        np.asarray(raw_energy, dtype=float)
        + 1.15 * np.asarray(pool["shortcut_score"], dtype=float)
        + 0.95 * np.asarray(pool["support_penalty"], dtype=float)
        + 0.55 * np.asarray(pool["contact_penalty"], dtype=float)
    )


def conservative_energy_threshold(energy: np.ndarray, quantile: float = 0.02) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    return float(np.quantile(np.asarray(energy, dtype=float), quantile))
