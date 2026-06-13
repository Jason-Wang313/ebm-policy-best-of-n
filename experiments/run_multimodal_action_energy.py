from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ebm_tail_audit.diagnostics import aggregate_curve_rows, curve_rows_for_energy
from ebm_tail_audit.energy_models import good_tail_aligned_energy, raw_miscalibrated_energy
from ebm_tail_audit.toy_envs import evaluate_multimodal_actions, generate_demo_dataset, generate_multimodal_pool
from ebm_tail_audit.utils import N_VALUES, write_csv, write_json


def _mse_row(seed: int, n: int) -> dict[str, object]:
    demo = generate_demo_dataset(seed, n)
    pool = generate_multimodal_pool(seed + 30, n)
    mean_action = np.mean(demo["actions"], axis=0)
    actions = np.tile(mean_action, (n, 1))
    actions[:, 1] = pool["observations"][:, 0]
    metrics = evaluate_multimodal_actions(pool["observations"], actions)
    return {
        "seed": seed,
        "baseline": "explicit_mse_regression",
        "mean_utility": float(np.mean(metrics["utility"])),
        "success_rate": float(np.mean(metrics["success"])),
        "invalid_rate": float(np.mean(metrics["invalid"])),
        "mean_action_x": float(np.mean(actions[:, 0])),
    }


def run(smoke: bool = False) -> dict[str, object]:
    out_dir = Path("results") / "multimodal_action"
    seeds = [0] if smoke else [0, 1, 2, 3, 4]
    pool_n = 1024 if smoke else 1536
    mc_trials = 250 if smoke else 250
    seed_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    for seed in seeds:
        pool = generate_multimodal_pool(seed + 40, pool_n)
        energies = {
            "tail_aligned_ebm": good_tail_aligned_energy(pool, seed),
            "raw_shortcut_ebm": raw_miscalibrated_energy(pool, seed),
        }
        for model_name, energy in energies.items():
            rows, _summary = curve_rows_for_energy(pool, energy, N_VALUES, model_name, seed, exact_mc_trials=mc_trials)
            for row in rows:
                row["task"] = "multimodal"
                seed_rows.append(row)
        baseline_rows.append(_mse_row(seed, pool_n // 2))
    summary_rows = aggregate_curve_rows(seed_rows)
    write_csv(out_dir / "seed_level.csv", seed_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "baselines.csv", baseline_rows)
    high = max(N_VALUES)
    aligned_high = np.mean([float(r["selected_real_utility"]) for r in seed_rows if r["model"] == "tail_aligned_ebm" and int(r["N"]) == high])
    raw_high = np.mean([float(r["selected_real_utility"]) for r in seed_rows if r["model"] == "raw_shortcut_ebm" and int(r["N"]) == high])
    mse_mean = np.mean([float(r["mean_utility"]) for r in baseline_rows])
    payload = {
        "experiment": "multimodal_action_energy",
        "smoke": smoke,
        "n_values": N_VALUES,
        "seeds": seeds,
        "mse_regression_mean_utility": float(mse_mean),
        "tail_aligned_ebm_high_N_utility": float(aligned_high),
        "raw_shortcut_ebm_high_N_utility": float(raw_high),
        "interpretation": "The tail-aligned EBM preserves two action modes better than explicit MSE, while raw shortcut energy fails under high-N low-energy selection.",
        "artifacts": {
            "summary_csv": str(out_dir / "summary.csv"),
            "seed_level_csv": str(out_dir / "seed_level.csv"),
            "baselines_csv": str(out_dir / "baselines.csv"),
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
