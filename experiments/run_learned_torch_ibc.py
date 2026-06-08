from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ebm_best_of_n.calibration import apply_energy_calibrator, fit_energy_calibrator
from ebm_best_of_n.diagnostics import aggregate_curve_rows, curve_rows_for_energy
from ebm_best_of_n.energy_models import NumpyIBCEnergy, oracle_energy, support_penalized_energy, value_shaped_energy
from ebm_best_of_n.toy_envs import generate_demo_dataset, generate_multimodal_pool
from ebm_best_of_n.torch_models import TorchIBCEnergy, toy_negative_sampler
from ebm_best_of_n.utils import N_VALUES, write_csv, write_json


def run(smoke: bool = False) -> dict[str, object]:
    out_dir = Path("results") / "learned_torch_ibc"
    seeds = [0] if smoke else [0, 1, 2]
    demo_n = 256 if smoke else 512
    pool_n = 1024 if smoke else 3072
    epochs = 35 if smoke else 110
    negatives = 16 if smoke else 32
    mc_trials = 250 if smoke else 900
    seed_rows: list[dict[str, object]] = []
    metadata: dict[str, object] | None = None

    for seed in seeds:
        demos = generate_demo_dataset(seed + 200, demo_n)
        torch_model = TorchIBCEnergy.fit(
            demos["observations"],
            demos["actions"],
            seed=seed,
            epochs=epochs,
            batch_size=128,
            negatives=negatives,
            lr=2e-3,
            hidden_dim=96,
            feature_mode="toy_visible",
            negative_sampler=toy_negative_sampler,
        )
        numpy_model = NumpyIBCEnergy.fit(
            demos["observations"],
            demos["actions"],
            seed=seed,
            epochs=35 if smoke else 70,
            batch_size=64,
            negatives=12 if smoke else 16,
            lr=0.08,
        )
        metadata = torch_model.metadata()
        pool = generate_multimodal_pool(seed + 250, pool_n)
        raw_torch = torch_model.energy(pool["observations"], pool["actions"])
        raw_numpy = numpy_model.energy(pool["observations"], pool["actions"])
        calibrator = fit_energy_calibrator(pool, raw_torch, pilot_size=192 if smoke else 384, seed=seed)
        energies = {
            "raw_torch_ebm": raw_torch,
            "raw_numpy_ebm": raw_numpy,
            "calibrated_torch": apply_energy_calibrator(pool, raw_torch, calibrator),
            "value_shaped_torch": value_shaped_energy(raw_torch, pool),
            "support_penalized_torch": support_penalized_energy(raw_torch, pool, weight=2.0),
            "random": np.zeros_like(raw_torch),
            "oracle": oracle_energy(pool),
        }
        for model_name, energy in energies.items():
            rows, _summary = curve_rows_for_energy(pool, energy, N_VALUES, model_name, seed, exact_mc_trials=mc_trials)
            for row in rows:
                row["task"] = "multimodal"
                row["learned_model"] = "pytorch_mlp_ibc_ebm"
                seed_rows.append(row)

    summary_rows = aggregate_curve_rows(seed_rows)
    write_csv(out_dir / "seed_level.csv", seed_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    high = max(N_VALUES)
    raw_high = [r for r in seed_rows if r["model"] == "raw_torch_ebm" and int(r["N"]) == high]
    cal_high = [r for r in seed_rows if r["model"] == "calibrated_torch" and int(r["N"]) == high]
    strongest = min(raw_high, key=lambda r: float(r["selected_real_utility"]))
    payload = {
        "experiment": "learned_torch_ibc",
        "smoke": smoke,
        "model_metadata": metadata,
        "n_values": N_VALUES,
        "demo_n": demo_n,
        "pool_n": pool_n,
        "seeds": seeds,
        "strongest_torch_ibc_artifact": strongest,
        "calibrated_high_N_utility_gain_over_raw_torch": float(
            np.mean([float(r["selected_real_utility"]) for r in cal_high])
            - np.mean([float(r["selected_real_utility"]) for r in raw_high])
        ),
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
