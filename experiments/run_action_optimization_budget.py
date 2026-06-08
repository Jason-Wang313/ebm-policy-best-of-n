from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ebm_best_of_n.best_of_n import score_from_energy, utility_best_of_n_finite
from ebm_best_of_n.energy_models import raw_miscalibrated_energy
from ebm_best_of_n.samplers import cem_like_refinement, gaussian_action_proposals, langevin_like_refinement
from ebm_best_of_n.utils import N_VALUES, write_csv, write_json


def _row(pool, energy, policy: str, seed: int, n: int, steps: int) -> dict[str, object]:
    score = score_from_energy(energy)
    utility_curve = utility_best_of_n_finite(score, pool["utility"], [n])
    energy_curve = utility_best_of_n_finite(score, energy, [n])
    evaluations = int(n * max(1, steps + 1))
    selected_utility = float(utility_curve[n])
    return {
        "seed": seed,
        "policy": policy,
        "N": n,
        "refinement_steps": steps,
        "energy_evaluations": evaluations,
        "latency_proxy": float(evaluations),
        "selected_real_utility": selected_utility,
        "selected_energy": float(energy_curve[n]),
        "utility_per_energy_eval": selected_utility / evaluations,
        "stop_rule": "stop_early" if n >= 32 and selected_utility < 0.2 else "continue",
    }


def run(smoke: bool = False) -> dict[str, object]:
    out_dir = Path("results") / "optimization_budget"
    seeds = [0] if smoke else [0, 1, 2]
    pool_n = 1024 if smoke else 2048
    n_values = [1, 2, 4, 8, 16, 32, 64] if smoke else N_VALUES
    rows: list[dict[str, object]] = []
    for seed in seeds:
        base = gaussian_action_proposals("contact_push", seed=seed + 70, n=pool_n)
        base_energy = raw_miscalibrated_energy(base, seed)
        cem_pool = cem_like_refinement(base, base_energy, "contact_push", seed=seed + 71, top_frac=0.18, noise_scale=0.08)
        cem_energy = raw_miscalibrated_energy(cem_pool, seed + 71)
        lang_pool = langevin_like_refinement(base, base_energy, "contact_push", seed=seed + 72, step_size=0.10)
        lang_energy = raw_miscalibrated_energy(lang_pool, seed + 72)
        for n in n_values:
            rows.append(_row(base, base_energy, "sample_only", seed, n, 0))
            rows.append(_row(cem_pool, cem_energy, "cem_like_topk_refinement", seed, n, 1))
            rows.append(_row(lang_pool, lang_energy, "langevin_like_refinement", seed, n, 1))
    write_csv(out_dir / "seed_level.csv", rows)
    # The summary keeps seed-level rows because the frontier plot benefits from
    # every observed compute point and the files stay small.
    write_csv(out_dir / "summary.csv", rows)
    best = max(rows, key=lambda r: float(r["selected_real_utility"]) / float(r["energy_evaluations"]))
    payload = {
        "experiment": "action_optimization_budget",
        "smoke": smoke,
        "n_values": n_values,
        "seeds": seeds,
        "best_utility_per_compute_region": best,
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
