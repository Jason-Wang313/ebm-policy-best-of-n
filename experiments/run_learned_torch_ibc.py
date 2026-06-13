from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
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
from ebm_tail_audit.toy_envs import generate_demo_dataset, generate_multimodal_pool
from ebm_tail_audit.torch_models import TorchIBCEnergy, toy_negative_sampler
from ebm_tail_audit.utils import N_VALUES, write_csv, write_json


def _settings(smoke: bool) -> dict[str, int]:
    return {
        "demo_n": 256 if smoke else 384,
        "pool_n": 1024 if smoke else 1536,
        "epochs": 35 if smoke else 32,
        "negatives": 16,
        "mc_trials": 250 if smoke else 300,
    }


def _run_seed(seed: int, smoke: bool) -> dict[str, object]:
    settings = _settings(smoke)
    demo_n = settings["demo_n"]
    pool_n = settings["pool_n"]
    epochs = settings["epochs"]
    negatives = settings["negatives"]
    mc_trials = settings["mc_trials"]
    seed_rows: list[dict[str, object]] = []

    print(f"learned_torch_ibc seed={seed} train", flush=True)
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
    print(f"learned_torch_ibc seed={seed} numpy_baseline", flush=True)
    numpy_model = NumpyIBCEnergy.fit(
        demos["observations"],
        demos["actions"],
        seed=seed,
        epochs=35 if smoke else 45,
        batch_size=64,
        negatives=12 if smoke else 16,
        lr=0.08,
    )
    metadata = torch_model.metadata()
    pool = generate_multimodal_pool(seed + 250, pool_n)
    raw_torch = torch_model.energy(pool["observations"], pool["actions"])
    raw_numpy = numpy_model.energy(pool["observations"], pool["actions"])
    calibrator = fit_energy_calibrator(pool, raw_torch, pilot_size=192 if smoke else 384, seed=seed)
    calibrated = apply_energy_calibrator(pool, raw_torch, calibrator)
    oracle = oracle_energy(pool)
    energies = {
        "raw_torch_ebm": raw_torch,
        "raw_numpy_ebm": raw_numpy,
        "calibrated_torch": calibrated,
        "value_shaped_torch": value_shaped_energy(raw_torch, pool),
        "support_penalized_torch": support_penalized_energy(raw_torch, pool, weight=2.0),
        "calibrated_support_penalized_torch": support_penalized_energy(calibrated, pool, weight=2.0),
        "oracle_value_shaped_upper": value_shaped_energy(oracle, pool),
        "random": np.zeros_like(raw_torch),
        "oracle": oracle,
    }
    for model_name, energy in energies.items():
        print(f"learned_torch_ibc seed={seed} curves {model_name}", flush=True)
        rows, _summary = curve_rows_for_energy(pool, energy, N_VALUES, model_name, seed, exact_mc_trials=mc_trials)
        for row in rows:
            row["task"] = "multimodal"
            row["learned_model"] = "pytorch_mlp_ibc_ebm"
            seed_rows.append(row)
    del torch_model, numpy_model, pool, raw_torch, raw_numpy, calibrated, oracle, energies
    gc.collect()
    return {"seed": seed, "rows": seed_rows, "metadata": metadata}


def _run_seed_subprocess(seed: int, output_path: Path) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-seed",
        str(seed),
        "--single-seed-output",
        str(output_path),
    ]
    last_result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, 4):
        print(f"learned_torch_ibc seed={seed} subprocess attempt={attempt}", flush=True)
        result = subprocess.run(args, env=env, text=True, capture_output=True)
        last_result = result
        if result.stdout:
            print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr, flush=True)
        if result.returncode == 0 and output_path.exists():
            break
        time.sleep(8.0 * attempt)
    else:
        assert last_result is not None
        raise subprocess.CalledProcessError(
            last_result.returncode,
            last_result.args,
            output=last_result.stdout,
            stderr=last_result.stderr,
        )
    time.sleep(2.0)
    with output_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run(smoke: bool = False) -> dict[str, object]:
    out_dir = Path("results") / "learned_torch_ibc"
    seeds = [0] if smoke else [0, 1, 2, 3, 4]
    settings = _settings(smoke)
    demo_n = settings["demo_n"]
    pool_n = settings["pool_n"]
    seed_rows: list[dict[str, object]] = []
    metadata: dict[str, object] | None = None

    for seed in seeds:
        if smoke:
            result = _run_seed(seed, smoke=True)
        else:
            result = _run_seed_subprocess(seed, out_dir / "tmp" / f"seed_{seed}.json")
        metadata = result["metadata"]
        seed_rows.extend(result["rows"])

    seed_rows.extend(
        make_conservative_gated_rows(
            seed_rows,
            "support_penalized_torch",
            "support_penalized_torch_conservative_gate",
        )
    )
    seed_rows.extend(
        make_conservative_gated_rows(
            seed_rows,
            "calibrated_support_penalized_torch",
            "calibrated_support_penalized_torch_conservative_gate",
        )
    )
    annotate_repair_effectiveness(
        seed_rows,
        "raw_torch_ebm",
        "oracle",
        {
            "calibrated_torch",
            "value_shaped_torch",
            "support_penalized_torch",
            "support_penalized_torch_conservative_gate",
            "calibrated_support_penalized_torch",
            "calibrated_support_penalized_torch_conservative_gate",
            "oracle_value_shaped_upper",
            "oracle",
        },
    )
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
        "main_repair_stack": "calibrated_support_penalized_torch_conservative_gate",
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
    parser.add_argument("--single-seed", type=int, default=None)
    parser.add_argument("--single-seed-output", type=Path, default=None)
    args = parser.parse_args()
    if args.single_seed is not None:
        if args.single_seed_output is None:
            raise SystemExit("--single-seed-output is required with --single-seed")
        write_json(args.single_seed_output, _run_seed(args.single_seed, smoke=args.smoke))
    else:
        run(smoke=args.smoke)
