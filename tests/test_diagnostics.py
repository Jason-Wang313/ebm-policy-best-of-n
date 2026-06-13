from __future__ import annotations

import importlib.util
from pathlib import Path

from ebm_tail_audit.calibration import apply_energy_calibrator, fit_energy_calibrator
from ebm_tail_audit.diagnostics import (
    annotate_repair_effectiveness,
    curve_rows_for_energy,
    energy_reliability_rows,
    make_conservative_gated_rows,
    repair_recovery_ratio,
    tail_rank_correlation,
)
from ebm_tail_audit.energy_models import oracle_energy, raw_miscalibrated_energy, support_penalized_energy
from ebm_tail_audit.toy_envs import generate_contact_push_pool


_METAWORLD_SPEC = importlib.util.spec_from_file_location(
    "run_metaworld_benchmark",
    Path("experiments/run_metaworld_benchmark.py"),
)
assert _METAWORLD_SPEC and _METAWORLD_SPEC.loader
_METAWORLD = importlib.util.module_from_spec(_METAWORLD_SPEC)
_METAWORLD_SPEC.loader.exec_module(_METAWORLD)
_closed_loop_dependency_payload = _METAWORLD._closed_loop_dependency_payload


def test_tail_rank_correlation_detects_anti_aligned_tail():
    energy = [0.0, 0.1, 0.2, 5.0, 6.0]
    utility = [-1.0, 1.0, 3.0, 0.0, 0.0]
    assert tail_rank_correlation(energy, utility, tail_fraction=0.6) < 0.0


def test_curve_rows_include_required_ebm_metrics():
    pool = generate_contact_push_pool(seed=0, n=256)
    energy = raw_miscalibrated_energy(pool, seed=0)
    rows, summary = curve_rows_for_energy(pool, energy, [1, 8], "raw", seed=0, exact_mc_trials=100)
    assert rows[0]["selected_energy"] > rows[-1]["selected_energy"]
    for key in [
        "low_energy_tail_real_utility",
        "tail_rank_correlation",
        "invalid_action_rate",
        "jerk_smoothness_penalty",
        "contact_feasibility_penalty",
        "compute_latency_proxy",
        "deployment_gate",
    ]:
        assert key in rows[-1]
    assert "prediction_error" in summary


def test_energy_reliability_rows_mark_low_energy_tail():
    pool = generate_contact_push_pool(seed=2, n=100)
    energy = raw_miscalibrated_energy(pool, seed=2)
    rows = energy_reliability_rows(pool, energy, model="raw", seed=2, task="contact_push", num_buckets=10)
    assert len(rows) == 10
    assert rows[0]["is_extreme_low_energy_tail"] is True
    assert rows[0]["energy_quantile_low"] == 0.0
    assert rows[-1]["energy_quantile_high"] == 1.0
    assert "mean_real_utility" in rows[0]


def test_repair_recovery_ratio_edge_cases():
    assert repair_recovery_ratio(1.0, 1.0, 1.0) == 1.0
    assert repair_recovery_ratio(0.5, 1.0, 1.0) == 0.0
    assert repair_recovery_ratio(0.0, -2.0, 2.0) == 0.5
    assert repair_recovery_ratio(3.0, -1.0, 1.0) == 2.0


def test_combined_repair_improves_and_gates_false_positive_tail():
    pool = generate_contact_push_pool(seed=3, n=512)
    raw = raw_miscalibrated_energy(pool, seed=3)
    calibrated = apply_energy_calibrator(pool, raw, fit_energy_calibrator(pool, raw, pilot_size=96, seed=3))
    combined = support_penalized_energy(calibrated, pool, weight=2.0)
    rows = []
    for model, energy in [
        ("raw_ebm", raw),
        ("calibrated_support_penalized", combined),
        ("oracle", oracle_energy(pool)),
    ]:
        model_rows, _ = curve_rows_for_energy(pool, energy, [1, 64], model, seed=3, exact_mc_trials=50)
        for row in model_rows:
            row["task"] = "contact_push"
            rows.append(row)
    rows.extend(
        make_conservative_gated_rows(
            rows,
            "calibrated_support_penalized",
            "calibrated_support_penalized_conservative_gate",
        )
    )
    annotate_repair_effectiveness(
        rows,
        "raw_ebm",
        "oracle",
        {"calibrated_support_penalized_conservative_gate"},
    )
    raw_high = next(r for r in rows if r["model"] == "raw_ebm" and r["N"] == 64)
    repaired_high = next(
        r for r in rows if r["model"] == "calibrated_support_penalized_conservative_gate" and r["N"] == 64
    )
    assert repaired_high["selected_real_utility"] > raw_high["selected_real_utility"] + 1.0
    assert repaired_high["utility_not_degraded"] == 1
    assert repaired_high["support_distance_reduced"] == 1


def test_closed_loop_dependency_payload_thresholds():
    tasks = ["reach-v3", "push-v3"]
    rows = []
    for task in tasks:
        for policy, success in [
            ("expert_centered_gate", 1.0),
            ("expert_centered_no_gate", 0.5),
            ("mixed_expert_local_no_gate", 0.6),
            ("local_gaussian_no_gate", 0.0),
            ("learned_demo_proposal_no_gate", 0.8),
            ("learned_demo_proposal_direct", 0.8),
            ("learned_bc_proposal_no_gate", 0.7),
            ("learned_bc_proposal_direct", 0.7),
            ("state_heuristic_proposal_no_gate", 0.9),
            ("state_heuristic_direct", 0.9),
            ("random_uniform", 0.0),
            ("scripted_expert", 1.0),
        ]:
            rows.append(
                {
                    "task": task,
                    "policy": policy,
                    "mean_success_rate": success,
                    "success_nonzero": success > 0,
                    "runtime_scripted_expert_dependency": policy
                    in {"expert_centered_gate", "expert_centered_no_gate", "mixed_expert_local_no_gate", "scripted_expert"},
                }
            )
    payload = _closed_loop_dependency_payload(rows, tasks)
    assert payload["status"] == "SUPPORTED"
    assert payload["low_dependency_success_threshold_met"] is True
    assert payload["autonomous_success_threshold_met"] is True
    assert payload["learned_ebm_low_dependency_success_threshold_met"] is True
