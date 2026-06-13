from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ebm_tail_audit.diagnostics import exact_law_validation_table
from ebm_tail_audit.energy_models import raw_miscalibrated_energy
from ebm_tail_audit.toy_envs import generate_contact_push_pool
from ebm_tail_audit.utils import N_VALUES, write_csv, write_json


def run(smoke: bool = False) -> dict[str, object]:
    out_dir = Path("results") / "exact_law"
    pool = generate_contact_push_pool(seed=4, n=1024 if smoke else 2048)
    energy = raw_miscalibrated_energy(pool, seed=4)
    rows, error = exact_law_validation_table(pool, energy, N_VALUES, seed=4)
    write_csv(out_dir / "validation.csv", rows)
    payload = {
        "experiment": "exact_law_validation",
        "smoke": smoke,
        "prediction_error": error,
        "n_values": N_VALUES,
        "artifacts": {
            "validation_csv": str(out_dir / "validation.csv"),
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
