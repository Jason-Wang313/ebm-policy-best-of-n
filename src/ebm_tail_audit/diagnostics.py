"""Diagnostics for low-energy selected tails."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .tail_selection import (
    binary_success_tail_selection,
    exact_law_prediction_error,
    monte_carlo_tail_selection,
    score_from_energy,
    utility_tail_selection_finite,
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


def _tail_selection_expectations_by_score(
    score: np.ndarray,
    metrics: dict[str, np.ndarray],
    n_values: Iterable[int],
) -> dict[str, dict[int, float]]:
    """Compute exact tail selection expectations for many metrics with one sort."""

    score_arr = np.asarray(score, dtype=float)
    order = np.argsort(score_arr, kind="mergesort")
    sorted_score = score_arr[order]
    n = len(sorted_score)
    starts: list[int] = []
    stops: list[int] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_score[j] == sorted_score[i]:
            j += 1
        starts.append(i)
        stops.append(j)
        i = j
    starts_arr = np.asarray(starts, dtype=float)
    stops_arr = np.asarray(stops, dtype=float)
    metric_names = list(metrics)
    group_means = []
    for name in metric_names:
        values = np.asarray(metrics[name], dtype=float)[order]
        group_means.append(np.asarray([float(np.mean(values[int(a) : int(b)])) for a, b in zip(starts_arr, stops_arr)], dtype=float))
    means = np.vstack(group_means)
    out = {name: {} for name in metric_names}
    for n_value in [int(v) for v in n_values]:
        mass = (stops_arr / n) ** n_value - (starts_arr / n) ** n_value
        vals = means @ mass
        for idx, name in enumerate(metric_names):
            out[name][n_value] = float(vals[idx])
    return out


def repair_recovery_ratio(repair_utility: float, raw_utility: float, oracle_utility: float, eps: float = 1e-12) -> float:
    """Fraction of the oracle-minus-raw utility gap recovered by a repair.

    If there is no positive oracle-minus-raw gap, there is nothing meaningful to
    recover. In that degenerate case, a non-degrading repair gets ratio 1.0 and
    a degrading repair gets 0.0.
    """

    repair = float(repair_utility)
    raw = float(raw_utility)
    oracle = float(oracle_utility)
    gap = oracle - raw
    if gap <= eps:
        return 1.0 if repair >= raw - eps else 0.0
    return float((repair - raw) / gap)


def _repair_default_fields() -> dict[str, float | int]:
    return {
        "requested_N": 0,
        "gate_selected_N": 0,
        "conservative_high_N_gate": 0,
        "repair_recovery_ratio": 0.0,
        "repair_recovery_ratio_reported": 0.0,
        "recovery_ge_95": 0,
        "utility_not_degraded": 0,
        "tail_rank_improved": 0,
        "support_distance_reduced": 0,
    }


def _context_key(row: dict[str, object], include_n: bool = True) -> tuple[object, ...]:
    keys = ["benchmark", "task", "seed"]
    if include_n:
        keys.append("N")
    return tuple(row.get(key, "") for key in keys)


def annotate_repair_effectiveness(
    rows: list[dict[str, object]],
    raw_model: str,
    oracle_model: str,
    repair_models: set[str] | list[str] | tuple[str, ...] | None = None,
    recovery_threshold: float = 0.95,
) -> None:
    """Annotate curve rows with recovery-ratio and pass/fail fields.

    Rows are matched by benchmark, task, seed, and N. The raw ratio is preserved
    even when it is above 1.0; ``repair_recovery_ratio_reported`` is clipped to
    [0, 1] for plots and compact reporting.
    """

    repair_set = set(repair_models or [])
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        for key, value in _repair_default_fields().items():
            row.setdefault(key, value)
        row["requested_N"] = int(float(row.get("N", 0)))
        row.setdefault("gate_selected_N", int(float(row.get("N", 0))))
        groups[_context_key(row)].append(row)

    for group_rows in groups.values():
        raw_row = next((r for r in group_rows if str(r.get("model")) == raw_model), None)
        oracle_row = next((r for r in group_rows if str(r.get("model")) == oracle_model), None)
        if raw_row is None or oracle_row is None:
            continue
        raw_u = float(raw_row["selected_real_utility"])
        oracle_u = float(oracle_row["selected_real_utility"])
        raw_tail = float(raw_row.get("tail_rank_correlation", 0.0))
        raw_support = float(raw_row.get("support_distance", 0.0))
        for row in group_rows:
            model = str(row.get("model"))
            utility = float(row["selected_real_utility"])
            ratio = repair_recovery_ratio(utility, raw_u, oracle_u)
            row["repair_recovery_ratio"] = ratio
            row["repair_recovery_ratio_reported"] = float(np.clip(ratio, 0.0, 1.0))
            row["recovery_ge_95"] = int(ratio >= float(recovery_threshold))
            row["utility_not_degraded"] = int(utility >= raw_u - 1e-12)
            row["tail_rank_improved"] = int(float(row.get("tail_rank_correlation", 0.0)) >= raw_tail - 1e-12)
            row["support_distance_reduced"] = int(float(row.get("support_distance", 0.0)) <= raw_support + 1e-12)
            row["is_repair_model"] = int(model in repair_set)


def make_conservative_gated_rows(
    rows: list[dict[str, object]],
    source_model: str,
    gated_model: str,
    marginal_floor: float = -1e-9,
) -> list[dict[str, object]]:
    """Create explicit conservative-high-N rows from an existing repair curve."""

    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row.get("model")) == source_model:
            groups[_context_key(row, include_n=False)].append(row)

    gated_rows: list[dict[str, object]] = []
    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda r: int(float(r["N"])))
        if not ordered:
            continue
        chosen = ordered[0]
        for row in ordered:
            candidate_ok = (
                float(row.get("invalid_action_rate", 0.0)) <= 0.35
                and float(row.get("tail_rank_correlation", 0.0)) >= -0.10
                and float(row.get("high_n_regret", 0.0)) <= 0.70
                and float(row.get("marginal_utility_per_added_candidate", 0.0)) >= marginal_floor
            )
            if candidate_ok and float(row["selected_real_utility"]) >= float(chosen["selected_real_utility"]) - 1e-12:
                chosen = row
            out = dict(chosen)
            requested_n = int(float(row["N"]))
            selected_n = int(float(chosen["N"]))
            out["model"] = gated_model
            out["N"] = requested_n
            out["requested_N"] = requested_n
            out["gate_selected_N"] = selected_n
            out["conservative_high_N_gate"] = 1
            out["source_repair_model"] = source_model
            out["energy_evaluations"] = selected_n
            out["compute_latency_proxy"] = float(selected_n)
            if selected_n < requested_n:
                out["deployment_gate"] = "stop_early"
            gated_rows.append(out)
    return gated_rows


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
    support = np.asarray(pool.get("support_distance", pool["support_penalty"]), dtype=float)

    selected = _tail_selection_expectations_by_score(
        score,
        {
            "utility": utility,
            "energy": energy,
            "score": score,
            "success": success,
            "invalid": invalid,
            "jerk": jerk,
            "contact": contact,
            "support": support,
        },
        n_values,
    )
    selected_utility = selected["utility"]
    selected_energy = selected["energy"]
    selected_score = selected["score"]
    selected_success = selected["success"]
    selected_invalid = selected["invalid"]
    selected_jerk = selected["jerk"]
    selected_contact = selected["contact"]
    selected_support = selected["support"]
    oracle_utility = _tail_selection_expectations_by_score(utility, {"utility": utility}, n_values)["utility"]
    random_utility = float(np.mean(utility))

    mc = monte_carlo_tail_selection(score, utility, n_values, n_trials=exact_mc_trials, seed=seed)
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
        tail_contact = float(np.mean(contact[mask]))
        tail_support = float(np.mean(support[mask]))
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
                "requested_N": int(n),
                "gate_selected_N": int(n),
                "conservative_high_N_gate": 0,
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
                "low_energy_tail_contact_failure_rate": tail_contact,
                "low_energy_tail_support_distance": tail_support,
                "tail_rank_correlation": tail_corr,
                "energy_utility_rank_correlation": average_rank_corr,
                "invalid_action_rate": selected_invalid[int(n)],
                "contact_failure_rate": selected_contact[int(n)],
                "support_distance": selected_support[int(n)],
                "jerk_smoothness_penalty": selected_jerk[int(n)],
                "contact_feasibility_penalty": selected_contact[int(n)],
                "oracle_minus_energy_gap": high_regret,
                "high_n_regret": high_regret,
                "marginal_utility_per_added_candidate": marginal / max(eval_delta, 1),
                "compute_latency_proxy": float(n),
                "energy_evaluations": int(n),
                "repair_recovery_ratio": 0.0,
                "repair_recovery_ratio_reported": 0.0,
                "recovery_ge_95": 0,
                "utility_not_degraded": 0,
                "tail_rank_improved": 0,
                "support_distance_reduced": 0,
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
        "contact_failure_rate_high_N": float(high["contact_failure_rate"]),
        "support_distance_high_N": float(high["support_distance"]),
    }
    return rows, summary


def energy_reliability_rows(
    pool: ArrayDict,
    energy: np.ndarray,
    model: str,
    seed: int,
    task: str,
    num_buckets: int = 20,
    benchmark: str = "toy",
    training_source: str = "",
) -> list[dict[str, object]]:
    """Bucket candidates by energy quantile and report real utility per bucket.

    Buckets are ordered from lowest energy to highest energy. With the default
    20 buckets, bucket 0 is the 5% extreme low-energy tail.
    """

    if int(num_buckets) < 2:
        raise ValueError("num_buckets must be >= 2")
    energy_arr = np.asarray(energy, dtype=float)
    utility = np.asarray(pool["utility"], dtype=float)
    success = np.asarray(pool["success"], dtype=float)
    invalid = np.asarray(pool["invalid"], dtype=float)
    contact = np.asarray(pool["contact_penalty"], dtype=float)
    support = np.asarray(pool.get("support_distance", pool["support_penalty"]), dtype=float)
    if len(energy_arr) != len(utility):
        raise ValueError("energy and pool utility must have the same length")
    if len(energy_arr) == 0:
        return []

    order = np.argsort(energy_arr, kind="mergesort")
    splits = np.array_split(order, int(num_buckets))
    rows: list[dict[str, object]] = []
    for bucket_idx, idx in enumerate(splits):
        if len(idx) == 0:
            continue
        q_low = bucket_idx / int(num_buckets)
        q_high = (bucket_idx + 1) / int(num_buckets)
        rows.append(
            {
                "benchmark": benchmark,
                "task": task,
                "seed": int(seed),
                "model": model,
                "training_source": training_source,
                "bucket": int(bucket_idx),
                "energy_quantile_low": float(q_low),
                "energy_quantile_high": float(q_high),
                "is_extreme_low_energy_tail": bool(bucket_idx == 0),
                "bucket_size": int(len(idx)),
                "mean_energy": float(np.mean(energy_arr[idx])),
                "mean_real_utility": float(np.mean(utility[idx])),
                "success_rate": float(np.mean(success[idx])),
                "invalid_rate": float(np.mean(invalid[idx])),
                "contact_failure_rate": float(np.mean(contact[idx])),
                "support_distance": float(np.mean(support[idx])),
            }
        )
    return rows


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
        "low_energy_tail_contact_failure_rate",
        "low_energy_tail_support_distance",
        "tail_rank_correlation",
        "energy_utility_rank_correlation",
        "invalid_action_rate",
        "contact_failure_rate",
        "support_distance",
        "jerk_smoothness_penalty",
        "contact_feasibility_penalty",
        "oracle_minus_energy_gap",
        "high_n_regret",
        "marginal_utility_per_added_candidate",
        "compute_latency_proxy",
        "energy_evaluations",
        "requested_N",
        "gate_selected_N",
        "conservative_high_N_gate",
        "repair_recovery_ratio",
        "repair_recovery_ratio_reported",
        "recovery_ge_95",
        "utility_not_degraded",
        "tail_rank_improved",
        "support_distance_reduced",
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
    exact = utility_tail_selection_finite(score, pool["utility"], n_values)
    mc = monte_carlo_tail_selection(score, pool["utility"], n_values, n_trials=30_000, seed=seed)
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
