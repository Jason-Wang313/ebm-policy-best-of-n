from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ebm_tail_audit.tail_selection import score_from_energy, utility_tail_selection_finite
from ebm_tail_audit.calibration import apply_energy_calibrator, fit_energy_calibrator
from ebm_tail_audit.diagnostics import annotate_repair_effectiveness, curve_rows_for_energy
from ebm_tail_audit.energy_models import oracle_energy, raw_miscalibrated_energy, support_penalized_energy
from ebm_tail_audit.toy_envs import generate_contact_push_pool
from ebm_tail_audit.utils import N_VALUES, write_csv, write_json


def run(smoke: bool = False) -> dict[str, object]:
    out_dir = Path("results") / "ablations"
    seeds = [0] if smoke else [0, 1, 2, 3, 4]
    pool_n = 1024 if smoke else 1536
    pilot_sizes = [16, 64, 128] if smoke else [16, 64, 128, 256, 512]
    support_weights = [0.0, 1.0, 2.0] if smoke else [0.0, 0.5, 1.0, 2.0, 4.0]
    high_n = max(N_VALUES)
    rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for seed in seeds:
        pool = generate_contact_push_pool(seed + 900, pool_n)
        raw = raw_miscalibrated_energy(pool, seed)
        raw_curve, _ = curve_rows_for_energy(pool, raw, [high_n], "raw_ebm", seed, exact_mc_trials=250 if smoke else 300)
        raw_row = raw_curve[0]
        seed_rows.append({"ablation": "raw_reference", "value": "raw", **raw_row})
        oracle_curve, _ = curve_rows_for_energy(
            pool, oracle_energy(pool), [high_n], "oracle", seed, exact_mc_trials=250 if smoke else 300
        )
        seed_rows.append({"ablation": "oracle_reference", "value": "oracle", **oracle_curve[0]})
        raw_utility = float(raw_row["selected_real_utility"])
        for pilot_size in pilot_sizes:
            calibrated = apply_energy_calibrator(
                pool,
                raw,
                fit_energy_calibrator(pool, raw, pilot_size=min(pilot_size, pool_n), seed=seed),
            )
            curve, _summary = curve_rows_for_energy(pool, calibrated, [high_n], "calibrated", seed, exact_mc_trials=250 if smoke else 300)
            row = dict(curve[0])
            row.update(
                {
                    "ablation": "pilot_labels",
                    "value": int(pilot_size),
                    "utility_gain_over_raw": float(row["selected_real_utility"]) - raw_utility,
                }
            )
            seed_rows.append(row)
        for weight in support_weights:
            repaired = support_penalized_energy(raw, pool, weight=weight)
            curve, _summary = curve_rows_for_energy(pool, repaired, [high_n], "support_penalized", seed, exact_mc_trials=250 if smoke else 300)
            row = dict(curve[0])
            row.update(
                {
                    "ablation": "support_weight",
                    "value": float(weight),
                    "utility_gain_over_raw": float(row["selected_real_utility"]) - raw_utility,
                }
            )
            seed_rows.append(row)

    annotate_repair_effectiveness(seed_rows, "raw_ebm", "oracle", {"calibrated", "support_penalized", "oracle"})
    for ablation, values in [
        ("pilot_labels", pilot_sizes),
        ("support_weight", support_weights),
    ]:
        for value in values:
            group = [r for r in seed_rows if r.get("ablation") == ablation and float(r["value"]) == float(value)]
            vals = np.asarray([float(r["selected_real_utility"]) for r in group], dtype=float)
            gains = np.asarray([float(r["utility_gain_over_raw"]) for r in group], dtype=float)
            recovery = np.asarray([float(r["repair_recovery_ratio"]) for r in group], dtype=float)
            gates = [str(r["deployment_gate"]) for r in group]
            mean = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            rows.append(
                {
                    "ablation": ablation,
                    "value": value,
                    "N": high_n,
                    "selected_real_utility": mean,
                    "selected_real_utility_ci_low": mean - 1.96 * se,
                    "selected_real_utility_ci_high": mean + 1.96 * se,
                    "utility_gain_over_raw": float(np.mean(gains)),
                    "mean_repair_recovery_ratio": float(np.mean(recovery)),
                    "worst_case_recovery_ratio": float(np.min(recovery)),
                    "num_repair_failures": int(
                        sum(
                            int(float(r["recovery_ge_95"])) == 0
                            or int(float(r["utility_not_degraded"])) == 0
                            or int(float(r["tail_rank_improved"])) == 0
                            or int(float(r["support_distance_reduced"])) == 0
                            for r in group
                        )
                    ),
                    "deployment_gate": max(set(gates), key=gates.count),
                    "num_seeds": len(group),
                }
            )

    write_csv(out_dir / "seed_level.csv", seed_rows)
    write_csv(out_dir / "summary.csv", rows)
    best_pilot = max([r for r in rows if r["ablation"] == "pilot_labels"], key=lambda r: float(r["selected_real_utility"]))
    best_support = max([r for r in rows if r["ablation"] == "support_weight"], key=lambda r: float(r["selected_real_utility"]))
    payload = {
        "experiment": "calibration_ablation",
        "smoke": smoke,
        "n_values": [high_n],
        "pilot_sizes": pilot_sizes,
        "support_weights": support_weights,
        "best_pilot_label_result": best_pilot,
        "best_support_weight_result": best_support,
        "artifacts": {
            "summary_csv": str(out_dir / "summary.csv"),
            "seed_level_csv": str(out_dir / "seed_level.csv"),
            "summary_json": str(out_dir / "summary.json"),
        },
    }
    write_json(out_dir / "summary.json", payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(smoke=args.smoke)
