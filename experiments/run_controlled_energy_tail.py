from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ebm_tail_audit.diagnostics import aggregate_curve_rows, curve_rows_for_energy
from ebm_tail_audit.energy_models import (
    good_tail_aligned_energy,
    noisy_energy,
    ood_low_energy,
    oracle_energy,
    pilot_label_noise_energy,
    raw_miscalibrated_energy,
    shortcut_energy,
    smoothness_blind_energy,
    support_blind_energy,
)
from ebm_tail_audit.toy_envs import TASKS
from ebm_tail_audit.utils import N_VALUES, write_csv, write_json


def run(smoke: bool = False) -> dict[str, object]:
    out_dir = Path("results") / "controlled_energy_tail"
    seeds = [0] if smoke else [0, 1, 2, 3, 4]
    pool_n = 1024 if smoke else 1024
    mc_trials = 300 if smoke else 100
    seed_rows: list[dict[str, object]] = []
    model_summaries: list[dict[str, object]] = []
    models = [
        ("oracle", oracle_energy),
        ("good", good_tail_aligned_energy),
        ("raw_miscalibrated", raw_miscalibrated_energy),
        ("shortcut", shortcut_energy),
        ("smoothness_blind", smoothness_blind_energy),
        ("ood_low_energy", ood_low_energy),
        ("support_blind", support_blind_energy),
        ("noisy_energy", noisy_energy),
        ("pilot_label_noise", pilot_label_noise_energy),
    ]
    for task_name in ["contact_push", "multimodal"]:
        for seed in seeds:
            pool = TASKS[task_name].pool_fn(seed=seed, n=pool_n)
            for model_name, energy_fn in models:
                energy = energy_fn(pool, seed) if model_name != "oracle" else energy_fn(pool)
                rows, summary = curve_rows_for_energy(pool, energy, N_VALUES, model_name, seed, exact_mc_trials=mc_trials)
                for row in rows:
                    row["task"] = task_name
                    seed_rows.append(row)
                summary["task"] = task_name
                model_summaries.append(summary)

    summary_rows = aggregate_curve_rows(seed_rows)
    write_csv(out_dir / "seed_level.csv", seed_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    raw_high = [r for r in seed_rows if r["model"] == "raw_miscalibrated" and int(r["N"]) == max(N_VALUES)]
    strongest = min(raw_high, key=lambda r: float(r["selected_real_utility"]))
    payload = {
        "experiment": "controlled_energy_tail",
        "smoke": smoke,
        "tasks": ["contact_push", "multimodal"],
        "n_values": N_VALUES,
        "pool_n": pool_n,
        "seeds": seeds,
        "models": [name for name, _ in models],
        "strongest_low_energy_tail_miscalibration": strongest,
        "model_summaries": model_summaries,
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
