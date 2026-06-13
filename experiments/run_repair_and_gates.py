from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ebm_tail_audit.calibration import apply_energy_calibrator, fit_energy_calibrator
from ebm_tail_audit.diagnostics import (
    aggregate_curve_rows,
    annotate_repair_effectiveness,
    curve_rows_for_energy,
    make_conservative_gated_rows,
)
from ebm_tail_audit.energy_models import oracle_energy, raw_miscalibrated_energy, support_penalized_energy, value_shaped_energy
from ebm_tail_audit.toy_envs import generate_contact_push_pool
from ebm_tail_audit.utils import N_VALUES, mean_ci, write_csv, write_json


MAIN_REPAIR_MODEL = "calibrated_support_penalized_conservative_gate"
REPAIR_MODELS = {
    "calibrated",
    "support_penalized",
    "support_penalized_conservative_gate",
    "calibrated_support_penalized",
    MAIN_REPAIR_MODEL,
    "value_shaped",
    "oracle_value_shaped_upper",
    "oracle",
}


def _effectiveness_rows(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    context: dict[tuple[object, ...], dict[str, dict[str, object]]] = {}
    for row in seed_rows:
        key = (row.get("benchmark", ""), row.get("task", ""), row.get("seed", ""), row.get("N", ""))
        context.setdefault(key, {})[str(row.get("model"))] = row

    out: list[dict[str, object]] = []
    for key, by_model in context.items():
        raw = by_model.get("raw_ebm")
        oracle = by_model.get("oracle")
        if raw is None or oracle is None:
            continue
        for model in sorted(REPAIR_MODELS | {"raw_ebm", "random"}):
            row = by_model.get(model)
            if row is None:
                continue
            out.append(
                {
                    "benchmark": key[0],
                    "task": key[1],
                    "seed": row.get("seed"),
                    "N": row.get("N"),
                    "model": model,
                    "raw_utility": raw.get("selected_real_utility"),
                    "repair_utility": row.get("selected_real_utility"),
                    "oracle_utility": oracle.get("selected_real_utility"),
                    "oracle_minus_raw_gap": float(oracle["selected_real_utility"]) - float(raw["selected_real_utility"]),
                    "repair_recovery_ratio": row.get("repair_recovery_ratio"),
                    "repair_recovery_ratio_reported": row.get("repair_recovery_ratio_reported"),
                    "recovery_ge_95": row.get("recovery_ge_95"),
                    "utility_not_degraded": row.get("utility_not_degraded"),
                    "tail_rank_improved": row.get("tail_rank_improved"),
                    "support_distance_reduced": row.get("support_distance_reduced"),
                    "gate_selected_N": row.get("gate_selected_N"),
                    "requested_N": row.get("requested_N"),
                    "deployment_gate": row.get("deployment_gate"),
                }
            )
    return out


def _write_repair_effectiveness(seed_rows: list[dict[str, object]], smoke: bool) -> dict[str, object]:
    out_dir = Path("results") / "repair_effectiveness"
    stress_rows = _effectiveness_rows(seed_rows)
    high_n = max(N_VALUES)
    summary_rows: list[dict[str, object]] = []
    for model in sorted(REPAIR_MODELS):
        group = [r for r in stress_rows if str(r["model"]) == model and int(r["N"]) == high_n]
        if not group:
            continue
        ratios = [float(r["repair_recovery_ratio"]) for r in group]
        ci = mean_ci(ratios)
        utility_degrading = [r for r in group if int(float(r["utility_not_degraded"])) == 0]
        failures = [
            r
            for r in group
            if int(float(r["recovery_ge_95"])) == 0
            or int(float(r["utility_not_degraded"])) == 0
            or int(float(r["tail_rank_improved"])) == 0
            or int(float(r["support_distance_reduced"])) == 0
        ]
        summary_rows.append(
            {
                "model": model,
                "N": high_n,
                "num_rows": len(group),
                "mean_repair_recovery_ratio": ci["mean"],
                "repair_recovery_ratio_ci_low": ci["ci_low"],
                "repair_recovery_ratio_ci_high": ci["ci_high"],
                "worst_case_recovery_ratio": float(np.min(ratios)),
                "num_repair_failures": len(failures),
                "num_utility_degrading_rows": len(utility_degrading),
                "recovery_ge_95_rate": float(np.mean([float(r["recovery_ge_95"]) for r in group])),
                "utility_not_degraded_rate": float(np.mean([float(r["utility_not_degraded"]) for r in group])),
                "tail_rank_improved_rate": float(np.mean([float(r["tail_rank_improved"]) for r in group])),
                "support_distance_reduced_rate": float(np.mean([float(r["support_distance_reduced"]) for r in group])),
            }
        )

    main = next((r for r in summary_rows if r["model"] == MAIN_REPAIR_MODEL), {})
    near_complete_supported = bool(
        main
        and float(main["mean_repair_recovery_ratio"]) >= 0.95
        and float(main["worst_case_recovery_ratio"]) >= 0.85
        and int(main["num_utility_degrading_rows"]) == 0
    )
    payload = {
        "experiment": "repair_effectiveness",
        "smoke": smoke,
        "metric_definition": "repair_recovery_ratio = (repair_utility - raw_utility) / (oracle_utility - raw_utility)",
        "main_repair_stack": MAIN_REPAIR_MODEL,
        "supported_scope": "local controlled contact-push stress tests; no real-robot or universal manipulation claim",
        "near_complete_repair_supported": near_complete_supported,
        "mean_recovery_ratio": float(main.get("mean_repair_recovery_ratio", 0.0)) if main else 0.0,
        "worst_case_recovery_ratio": float(main.get("worst_case_recovery_ratio", 0.0)) if main else 0.0,
        "num_repair_failures": int(main.get("num_repair_failures", 0)) if main else 0,
        "num_utility_degrading_rows": int(main.get("num_utility_degrading_rows", 0)) if main else 0,
        "thresholds": {
            "mean_recovery_ratio": 0.95,
            "worst_case_supported_stress_recovery_ratio": 0.85,
            "utility_degrading_rows": 0,
        },
        "summary_rows": summary_rows,
        "artifacts": {
            "summary_csv": str(out_dir / "summary.csv"),
            "stress_matrix_csv": str(out_dir / "stress_matrix.csv"),
            "summary_json": str(out_dir / "summary.json"),
        },
    }
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "stress_matrix.csv", stress_rows)
    write_json(out_dir / "summary.json", payload)
    return payload


def run(smoke: bool = False) -> dict[str, object]:
    out_dir = Path("results") / "repair"
    seeds = [0] if smoke else [0, 1, 2, 3, 4]
    pool_n = 1024 if smoke else 1536
    mc_trials = 120 if smoke else 80
    seed_rows: list[dict[str, object]] = []
    for seed in seeds:
        pool = generate_contact_push_pool(seed, pool_n)
        raw = raw_miscalibrated_energy(pool, seed)
        calibrator = fit_energy_calibrator(pool, raw, pilot_size=160 if smoke else 224, seed=seed)
        calibrated = apply_energy_calibrator(pool, raw, calibrator)
        oracle = oracle_energy(pool)
        energies = {
            "raw_ebm": raw,
            "calibrated": calibrated,
            "value_shaped": value_shaped_energy(raw, pool),
            "support_penalized": support_penalized_energy(raw, pool, weight=2.0),
            "calibrated_support_penalized": support_penalized_energy(calibrated, pool, weight=2.0),
            "oracle_value_shaped_upper": value_shaped_energy(oracle, pool),
            "random": np.zeros_like(raw),
            "oracle": oracle,
        }
        for model_name, energy in energies.items():
            rows, _summary = curve_rows_for_energy(pool, energy, N_VALUES, model_name, seed, exact_mc_trials=mc_trials)
            for row in rows:
                row["task"] = "contact_push"
                seed_rows.append(row)
    seed_rows.extend(
        make_conservative_gated_rows(seed_rows, "support_penalized", "support_penalized_conservative_gate")
    )
    seed_rows.extend(
        make_conservative_gated_rows(seed_rows, "calibrated_support_penalized", MAIN_REPAIR_MODEL)
    )
    annotate_repair_effectiveness(seed_rows, "raw_ebm", "oracle", REPAIR_MODELS)
    summary_rows = aggregate_curve_rows(seed_rows)
    write_csv(out_dir / "seed_level.csv", seed_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    effectiveness = _write_repair_effectiveness(seed_rows, smoke)
    high_n = max(N_VALUES)
    def mean_high(model: str) -> float:
        vals = [float(r["selected_real_utility"]) for r in seed_rows if r["model"] == model and int(r["N"]) == high_n]
        return float(np.mean(vals))
    raw_gate = next(r["deployment_gate"] for r in seed_rows if r["model"] == "raw_ebm" and int(r["N"]) == high_n)
    cal_gate = next(r["deployment_gate"] for r in seed_rows if r["model"] == "calibrated" and int(r["N"]) == high_n)
    payload = {
        "experiment": "repair_and_gates",
        "smoke": smoke,
        "n_values": N_VALUES,
        "seeds": seeds,
        "raw_high_N_gate": raw_gate,
        "calibrated_high_N_gate": cal_gate,
        "calibrated_high_N_utility_gain_over_raw": mean_high("calibrated") - mean_high("raw_ebm"),
        "value_shaped_high_N_utility_gain_over_raw": mean_high("value_shaped") - mean_high("raw_ebm"),
        "support_penalized_high_N_utility_gain_over_raw": mean_high("support_penalized") - mean_high("raw_ebm"),
        "main_repair_stack": MAIN_REPAIR_MODEL,
        "main_repair_recovery_ratio": effectiveness["mean_recovery_ratio"],
        "main_repair_worst_case_recovery_ratio": effectiveness["worst_case_recovery_ratio"],
        "main_repair_num_failures": effectiveness["num_repair_failures"],
        "artifacts": {
            "summary_csv": str(out_dir / "summary.csv"),
            "seed_level_csv": str(out_dir / "seed_level.csv"),
            "summary_json": str(out_dir / "summary.json"),
            "repair_effectiveness_summary_csv": "results/repair_effectiveness/summary.csv",
            "repair_effectiveness_stress_matrix_csv": "results/repair_effectiveness/stress_matrix.csv",
        },
    }
    write_json(out_dir / "summary.json", payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(smoke=args.smoke)
