"""Exact finite Best-of-N laws for score-maximizing selection.

The theorem code always maximizes a score S. For EBM policies the convention is
S(o, a) = -E_theta(o, a), so minimum-energy selection is exactly maximum-score
selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class TieGroup:
    """A tied score group in ascending score order."""

    score: float
    start: int
    stop: int
    r_min: int
    r_max: int

    @property
    def size(self) -> int:
        return self.stop - self.start


def _as_1d_float(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_n_values(n_values: Iterable[int]) -> list[int]:
    out: list[int] = []
    for n in n_values:
        if int(n) != n:
            raise ValueError("all N values must be integers")
        out.append(int(n))
    if not out:
        raise ValueError("n_values must be non-empty")
    if any(n < 1 for n in out):
        raise ValueError("all N values must be >= 1")
    return out


def _validate_tie_policy(tie_policy: str) -> str:
    if tie_policy != "mean":
        raise ValueError("only deterministic tie_policy='mean' is supported")
    return tie_policy


def _sorted_groups(scores: np.ndarray) -> tuple[np.ndarray, list[TieGroup]]:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    groups: list[TieGroup] = []
    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        groups.append(
            TieGroup(
                score=float(sorted_scores[i]),
                start=i,
                stop=j,
                r_min=i + 1,
                r_max=j,
            )
        )
        i = j
    return order, groups


def score_from_energy(energy: Iterable[float] | np.ndarray) -> np.ndarray:
    """Convert EBM energy to the score convention used by theorem code."""

    return -_as_1d_float(energy, "energy")


def utility_best_of_n_finite(
    scores: Iterable[float] | np.ndarray,
    utilities: Iterable[float] | np.ndarray,
    n_values: Iterable[int],
    tie_policy: str = "mean",
) -> dict[int, float]:
    """Expected selected utility from a finite pool under Best-of-N.

    Sampling is i.i.d. with replacement from the finite pool. The selected item
    is the maximum-score item among N samples. If the selected maximum score is
    tied, deterministic tie handling uses the mean utility of that score group.
    """

    _validate_tie_policy(tie_policy)
    scores_arr = _as_1d_float(scores, "scores")
    utilities_arr = _as_1d_float(utilities, "utilities")
    if scores_arr.shape != utilities_arr.shape:
        raise ValueError("scores and utilities must have the same length")
    ns = _as_n_values(n_values)

    m = len(scores_arr)
    order, groups = _sorted_groups(scores_arr)
    sorted_utilities = utilities_arr[order]
    out: dict[int, float] = {}
    for n in ns:
        value = 0.0
        for group in groups:
            mass = (group.r_max / m) ** n - ((group.r_min - 1) / m) ** n
            group_mean = float(np.mean(sorted_utilities[group.start : group.stop]))
            value += group_mean * mass
        out[n] = float(value)
    return out


def binary_success_best_of_n(
    scores: Iterable[float] | np.ndarray,
    success: Iterable[float] | np.ndarray,
    n_values: Iterable[int],
    tie_policy: str = "mean",
) -> dict[int, float]:
    """Exact finite Best-of-N success probability for binary utilities."""

    success_arr = _as_1d_float(success, "success")
    if not np.all((success_arr == 0.0) | (success_arr == 1.0)):
        raise ValueError("success must be binary values in {0, 1}")
    return utility_best_of_n_finite(scores, success_arr, n_values, tie_policy=tie_policy)


def monte_carlo_best_of_n(
    scores: Iterable[float] | np.ndarray,
    utilities: Iterable[float] | np.ndarray,
    n_values: Iterable[int],
    n_trials: int = 20_000,
    seed: int | None = 0,
    tie_policy: str = "mean",
) -> dict[int, dict[str, float]]:
    """Monte Carlo estimate and normal-approximation CI for Best-of-N utility."""

    _validate_tie_policy(tie_policy)
    scores_arr = _as_1d_float(scores, "scores")
    utilities_arr = _as_1d_float(utilities, "utilities")
    if scores_arr.shape != utilities_arr.shape:
        raise ValueError("scores and utilities must have the same length")
    ns = _as_n_values(n_values)
    if int(n_trials) < 1:
        raise ValueError("n_trials must be >= 1")

    rng = np.random.default_rng(seed)
    out: dict[int, dict[str, float]] = {}
    m = len(scores_arr)
    has_ties = len(np.unique(scores_arr)) < len(scores_arr)
    for n in ns:
        idx = rng.integers(0, m, size=(int(n_trials), n))
        sampled_scores = scores_arr[idx]
        if not has_ties:
            best = np.argmax(sampled_scores, axis=1)
            selected_values = utilities_arr[idx[np.arange(int(n_trials)), best]]
        else:
            max_scores = np.max(sampled_scores, axis=1)
            selected_values = np.empty(int(n_trials), dtype=float)
            for row in range(int(n_trials)):
                tied_positions = np.flatnonzero(sampled_scores[row] == max_scores[row])
                selected_values[row] = float(np.mean(utilities_arr[idx[row, tied_positions]]))
        mean = float(np.mean(selected_values))
        se = float(np.std(selected_values, ddof=1) / sqrt(int(n_trials))) if n_trials > 1 else 0.0
        out[n] = {
            "mean": mean,
            "se": se,
            "ci_low": mean - 1.96 * se,
            "ci_high": mean + 1.96 * se,
        }
    return out


def exact_law_prediction_error(
    exact: Mapping[int, float] | Mapping[str, float],
    empirical: Mapping[int, float] | Mapping[int, Mapping[str, float]] | Mapping[str, float],
) -> dict[str, float]:
    """Compute absolute-error diagnostics between exact and empirical curves."""

    errors: list[float] = []
    for key, exact_value in exact.items():
        empirical_value = empirical[key]  # type: ignore[index]
        if isinstance(empirical_value, Mapping):
            observed = float(empirical_value["mean"])
        else:
            observed = float(empirical_value)
        err = abs(float(exact_value) - observed)
        errors.append(err)
    if not errors:
        raise ValueError("exact and empirical curves must share at least one N value")
    arr = np.asarray(errors, dtype=float)
    return {
        "mean_abs_error": float(np.mean(arr)),
        "max_abs_error": float(np.max(arr)),
        "rmse": float(np.sqrt(np.mean(arr * arr))),
    }
