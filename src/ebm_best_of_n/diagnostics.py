"""Diagnostics for low-energy selected tails."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .best_of_n import (
    binary_success_best_of_n,
    exact_law_prediction_error,
    monte_carlo_best_of_n,
    score_from_energy,
    utility_best_of_n_finite,
)
from .gates import deployment_gate
from .toy_envs import ArrayDict


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearmanr(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    if len(x_arr) < 3:
        return 0.0
    rx = _rankdata(x_arr)
    ry = _rankdata(y_arr)
    rx = rx - np.mean(rx)
    ry = ry - np.mean(ry)
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(rx * ry) / denom)


def low_energy_tail_mask(energy: np.ndarray, tail_fraction: float) -> np.ndarray:
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    cutoff = np.quantile(np.asarray(energy, dtype=float), tail_fraction)
    return np.asarray(energy, dtype=float) <= cutoff


def tail_rank_correlation(energy: np.ndarray, utility: np.ndarray, tail_fraction: float) -> float:
    mask = low_energy_tail_mask(energy, tail_fraction)
    if int(np.sum(mask)) < 3:
        return 0.0
    return spearmanr(-np.asarray(energy)[mask], np.asarray(utility)[mask])


def curve_rows_for_energy(
    pool: ArrayDict,
    energy: np.ndarray,
    n_values: Iterable[int],
    model: str,
    seed: int,
    exact_mc_trials: int = 8000,
) -> tuple[list[dict[str, float | int | str]], dict[str, object]]:
    """Compute exact selected metrics for one energy model and pool."""

    energy = np.asarray(energy, dtype=float)
    score = score_from_energy(energy)
    utility = np.asarray(pool["utility"], dtype=float)
    success = np.asarray(pool["success"], dtype=float)
    invalid = np.asarray(pool["invalid"], dtype=float)
    jerk = np.asarray(pool["jerk_penalty"], dtype=float)
    contact = np.asarray(pool["contact_penalty"], dtype=float)

    selected_utility = utility_best_of_n_finite(score, utility, n_values)
    selected_energy = utility_best_of_n_finite(score, energy, n_values)
    selected_score = utility_best_of_n_finite(score, score, n_values)
    selected_success = binary_success_best_of_n(score, success, n_values)
    selected_invalid = utility_best_of_n_finite(score, invalid, n_values)
    selected_jerk = utility_best_of_n_finite(score, jerk, n_values)
    selected_contact = utility_best_of_n_finite(score, contact, n_values)
    oracle_utility = utility_best_of_n_finite(utility, utility, n_values)
    random_utility = float(np.mean(utility))

    mc = monte_carlo_best_of_n(score, utility, n_values, n_trials=exact_mc_trials, seed=seed)
    pred_err = exact_law_prediction_error(selected_utility, mc)
    average_rank_corr = spearmanr(score, utility)

    rows: list[dict[str, float | int | str]] = []
    prev_n: int | None = None
    prev_u: float | None = None
    for n in n_values:
        tail_fraction = max(1.0 / int(n), min(0.10, 64.0 / len(energy)))
        mask = low_energy_tail_mask(energy, tail_fraction)
        tail_u = float(np.mean(utility[mask]))
        tail_invalid = float(np.mean(invalid[mask]))
        tail_corr = tail_rank_correlation(energy, utility, tail_fraction)
        selected_u = selected_utility[int(n)]
        marginal = selected_u - (prev_u if prev_u is not None else random_utility)
        eval_delta = int(n) - (prev_n if prev_n is not None else 0)
        high_regret = oracle_utility[int(n)] - selected_u
        rows.append(
            {
                "seed": int(seed),
                "model": model,
                "N": int(n),
                "selected_energy": selected_energy[int(n)],
                "selected_score": selected_score[int(n)],
                "selected_real_utility": selected_u,
                "success_rate": selected_success[int(n)],
                "exact_expected_utility": selected_u,
                "mc_mean_utility": mc[int(n)]["mean"],
                "mc_ci_low": mc[int(n)]["ci_low"],
                "mc_ci_high": mc[int(n)]["ci_high"],
                "exact_law_abs_error": abs(selected_u - mc[int(n)]["mean"]),
                "low_energy_tail_real_utility": tail_u,
                "low_energy_tail_invalid_rate": tail_invalid,
                "tail_rank_correlation": tail_corr,
                "energy_utility_rank_correlation": average_rank_corr,
                "invalid_action_rate": selected_invalid[int(n)],
                "jerk_smoothness_penalty": selected_jerk[int(n)],
                "contact_feasibility_penalty": selected_contact[int(n)],
                "oracle_minus_energy_gap": high_regret,
                "high_n_regret": high_regret,
                "marginal_utility_per_added_candidate": marginal / max(eval_delta, 1),
                "compute_latency_proxy": float(n),
                "energy_evaluations": int(n),
            }
        )
        prev_n = int(n)
        prev_u = selected_u

    high = rows[-1]
    gate = deployment_gate(
        tail_rank_correlation=float(high["tail_rank_correlation"]),
        high_n_regret=float(high["high_n_regret"]),
        invalid_action_rate=float(high["invalid_action_rate"]),
        marginal_utility_high_n=float(high["marginal_utility_per_added_candidate"]),
        calibration_available=model in {"calibrated", "value_shaped", "support_penalized", "oracle", "good"},
    )
    for row in rows:
        row["deployment_gate"] = gate

    summary = {
        "model": model,
        "seed": int(seed),
        "gate": gate,
        "prediction_error": pred_err,
        "selected_energy_by_N": {str(row["N"]): row["selected_energy"] for row in rows},
        "selected_real_utility_by_N": {str(row["N"]): row["selected_real_utility"] for row in rows},
        "tail_rank_correlation_high_N": float(high["tail_rank_correlation"]),
        "low_energy_tail_real_utility_high_N": float(high["low_energy_tail_real_utility"]),
        "high_N_regret": float(high["high_n_regret"]),
        "invalid_action_rate_high_N": float(high["invalid_action_rate"]),
    }
    return rows, summary


def aggregate_curve_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model"]), int(row["N"]))].append(row)
    out: list[dict[str, object]] = []
    metric_keys = [
        "selected_energy",
        "selected_score",
        "selected_real_utility",
        "success_rate",
        "exact_law_abs_error",
        "low_energy_tail_real_utility",
        "low_energy_tail_invalid_rate",
        "tail_rank_correlation",
        "energy_utility_rank_correlation",
        "invalid_action_rate",
        "jerk_smoothness_penalty",
        "contact_feasibility_penalty",
        "oracle_minus_energy_gap",
        "high_n_regret",
        "marginal_utility_per_added_candidate",
        "compute_latency_proxy",
        "energy_evaluations",
    ]
    for (model, n), group_rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        out_row: dict[str, object] = {"model": model, "N": n, "num_seeds": len(group_rows)}
        for key in metric_keys:
            vals = np.asarray([float(r[key]) for r in group_rows], dtype=float)
            mean = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            out_row[key] = mean
            out_row[f"{key}_ci_low"] = mean - 1.96 * se
            out_row[f"{key}_ci_high"] = mean + 1.96 * se
        gates = [str(r["deployment_gate"]) for r in group_rows]
        out_row["deployment_gate"] = max(set(gates), key=gates.count)
        out.append(out_row)
    return out


def exact_law_validation_table(pool: ArrayDict, energy: np.ndarray, n_values: Iterable[int], seed: int = 0) -> tuple[list[dict[str, object]], dict[str, float]]:
    score = score_from_energy(energy)
    exact = utility_best_of_n_finite(score, pool["utility"], n_values)
    mc = monte_carlo_best_of_n(score, pool["utility"], n_values, n_trials=30_000, seed=seed)
    rows = []
    for n in n_values:
        rows.append(
            {
                "N": int(n),
                "predicted_selected_utility": exact[int(n)],
                "empirical_selected_utility": mc[int(n)]["mean"],
                "mc_ci_low": mc[int(n)]["ci_low"],
                "mc_ci_high": mc[int(n)]["ci_high"],
                "prediction_error": abs(exact[int(n)] - mc[int(n)]["mean"]),
            }
        )
    return rows, exact_law_prediction_error(exact, mc)
