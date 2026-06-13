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
from ebm_tail_audit.energy_models import NumpyIBCEnergy, oracle_energy, support_penalized_energy, value_shaped_energy
from ebm_tail_audit.toy_envs import evaluate_multimodal_actions, generate_demo_dataset, generate_multimodal_pool
from ebm_tail_audit.utils import N_VALUES, write_csv, write_json


def _mse_baseline(seed: int, n: int = 512) -> dict[str, float | str]:
    demo = generate_demo_dataset(seed, n)
    eval_obs = generate_multimodal_pool(seed + 100, n)["observations"]
    action_mean = np.mean(demo["actions"], axis=0)
    actions = np.tile(action_mean, (len(eval_obs), 1))
    actions[:, 1] = eval_obs[:, 0] + float(np.mean(demo["actions"][:, 1] - demo["observations"][:, 0]))
    metrics = evaluate_multimodal_actions(eval_obs, actions)
    return {
        "baseline": "explicit_mse_regression",
        "mean_utility": float(np.mean(metrics["utility"])),
        "success_rate": float(np.mean(metrics["success"])),
        "invalid_rate": float(np.mean(metrics["invalid"])),
        "mean_action_x": float(np.mean(actions[:, 0])),
    }


def _nearest_neighbor_baseline(seed: int, n: int = 512) -> dict[str, float | str]:
    demo = generate_demo_dataset(seed, n)
    eval_pool = generate_multimodal_pool(seed + 110, n)
    obs = eval_pool["observations"]
    demo_obs = demo["observations"][:, 0]
    idx = np.argmin(np.abs(obs[:, 0:1] - demo_obs[None, :]), axis=1)
    actions = demo["actions"][idx]
    metrics = evaluate_multimodal_actions(obs, actions)
    return {
        "baseline": "nearest_neighbor_demo_action",
        "mean_utility": float(np.mean(metrics["utility"])),
        "success_rate": float(np.mean(metrics["success"])),
        "invalid_rate": float(np.mean(metrics["invalid"])),
        "mean_action_x": float(np.mean(actions[:, 0])),
    }


def run(smoke: bool = False) -> dict[str, object]:
    out_dir = Path("results") / "learned_ibc"
    seeds = [0] if smoke else [0, 1, 2, 3, 4]
    demo_n = 256 if smoke else 320
    pool_n = 1024 if smoke else 1536
    epochs = 35 if smoke else 50
    negatives = 12 if smoke else 16
    mc_trials = 250 if smoke else 300
    seed_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    metadata: dict[str, object] | None = None

    for seed in seeds:
        demos = generate_demo_dataset(seed, demo_n)
        model = NumpyIBCEnergy.fit(
            demos["observations"],
            demos["actions"],
            seed=seed,
            epochs=epochs,
            batch_size=64,
            negatives=negatives,
            lr=0.08,
        )
        metadata = model.metadata()
        pool = generate_multimodal_pool(seed + 10, pool_n)
        raw_energy = model.energy(pool["observations"], pool["actions"])
        calibrator = fit_energy_calibrator(pool, raw_energy, pilot_size=160 if smoke else 224, seed=seed)
        calibrated = apply_energy_calibrator(pool, raw_energy, calibrator)
        oracle = oracle_energy(pool)
        energies = {
            "raw_ebm": raw_energy,
            "calibrated": calibrated,
            "value_shaped": value_shaped_energy(raw_energy, pool),
            "support_penalized": support_penalized_energy(raw_energy, pool, weight=2.0),
            "calibrated_support_penalized": support_penalized_energy(calibrated, pool, weight=2.0),
            "oracle_value_shaped_upper": value_shaped_energy(oracle, pool),
            "random": np.zeros_like(raw_energy),
            "oracle": oracle,
        }
        for model_name, energy in energies.items():
            rows, _summary = curve_rows_for_energy(pool, energy, N_VALUES, model_name, seed, exact_mc_trials=mc_trials)
            for row in rows:
                row["task"] = "multimodal"
                row["learned_model"] = "numpy_ibc_linear_feature_ebm"
                seed_rows.append(row)
        baseline_rows.append(_mse_baseline(seed, n=pool_n // 2))
        baseline_rows.append(_nearest_neighbor_baseline(seed, n=pool_n // 2))

    seed_rows.extend(
        make_conservative_gated_rows(seed_rows, "support_penalized", "support_penalized_conservative_gate")
    )
    seed_rows.extend(
        make_conservative_gated_rows(
            seed_rows,
            "calibrated_support_penalized",
            "calibrated_support_penalized_conservative_gate",
        )
    )
    annotate_repair_effectiveness(
        seed_rows,
        "raw_ebm",
        "oracle",
        {
            "calibrated",
            "value_shaped",
            "support_penalized",
            "support_penalized_conservative_gate",
            "calibrated_support_penalized",
            "calibrated_support_penalized_conservative_gate",
            "oracle_value_shaped_upper",
            "oracle",
        },
    )
    summary_rows = aggregate_curve_rows(seed_rows)
    write_csv(out_dir / "seed_level.csv", seed_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "baselines.csv", baseline_rows)
    raw_high = [r for r in seed_rows if r["model"] == "raw_ebm" and int(r["N"]) == max(N_VALUES)]
    cal_high = [r for r in seed_rows if r["model"] == "calibrated" and int(r["N"]) == max(N_VALUES)]
    strongest = min(raw_high, key=lambda r: float(r["selected_real_utility"]))
    repair_gain = float(np.mean([float(r["selected_real_utility"]) for r in cal_high]) - np.mean([float(r["selected_real_utility"]) for r in raw_high]))
    payload = {
        "experiment": "learned_ibc_policy",
        "smoke": smoke,
        "model_metadata": metadata,
        "n_values": N_VALUES,
        "demo_n": demo_n,
        "pool_n": pool_n,
        "seeds": seeds,
        "strongest_learned_ibc_artifact": strongest,
        "calibrated_high_N_utility_gain_over_raw": repair_gain,
        "main_repair_stack": "calibrated_support_penalized_conservative_gate",
        "baselines": baseline_rows,
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
