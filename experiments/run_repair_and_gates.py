from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ebm_best_of_n.calibration import apply_energy_calibrator, fit_energy_calibrator
from ebm_best_of_n.diagnostics import aggregate_curve_rows, curve_rows_for_energy
from ebm_best_of_n.energy_models import oracle_energy, raw_miscalibrated_energy, support_penalized_energy, value_shaped_energy
from ebm_best_of_n.toy_envs import generate_contact_push_pool
from ebm_best_of_n.utils import N_VALUES, write_csv, write_json


def run(smoke: bool = False) -> dict[str, object]:
    out_dir = Path("results") / "repair"
    seeds = [0] if smoke else [0, 1, 2]
    pool_n = 1024 if smoke else 2048
    mc_trials = 250 if smoke else 650
    seed_rows: list[dict[str, object]] = []
    for seed in seeds:
        pool = generate_contact_push_pool(seed, pool_n)
        raw = raw_miscalibrated_energy(pool, seed)
        calibrator = fit_energy_calibrator(pool, raw, pilot_size=160 if smoke else 224, seed=seed)
        energies = {
            "raw_ebm": raw,
            "calibrated": apply_energy_calibrator(pool, raw, calibrator),
            "value_shaped": value_shaped_energy(raw, pool),
            "support_penalized": support_penalized_energy(raw, pool, weight=2.0),
            "random": np.zeros_like(raw),
            "oracle": oracle_energy(pool),
        }
        for model_name, energy in energies.items():
            rows, _summary = curve_rows_for_energy(pool, energy, N_VALUES, model_name, seed, exact_mc_trials=mc_trials)
            for row in rows:
                row["task"] = "contact_push"
                seed_rows.append(row)
    summary_rows = aggregate_curve_rows(seed_rows)
    write_csv(out_dir / "seed_level.csv", seed_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
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
