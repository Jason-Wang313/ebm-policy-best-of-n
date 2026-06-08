from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ebm_best_of_n.utils import write_csv, write_json


def run(smoke: bool = False) -> dict[str, object]:
    out_dir = Path("results") / "optional_benchmarks"
    candidates = ["gymnasium", "d4rl", "metaworld", "mani_skill", "robosuite"]
    rows = []
    for name in candidates:
        available = importlib.util.find_spec(name) is not None
        if name == "metaworld" and (Path("results") / "benchmarks" / "metaworld" / "summary.json").exists():
            status = "CORE_GUARDED_ARTIFACT_SEE_RESULTS_BENCHMARKS_METAWORLD"
        else:
            status = "UNSUPPORTED_FUTURE" if not available else "AVAILABLE_NOT_RUN_BY_CORE_SCRIPT"
        rows.append(
            {
                "benchmark": name,
                "available": available,
                "status": status,
                "claim_boundary": "Optional adapters are guarded and do not support real-robot claims.",
            }
        )
    write_csv(out_dir / "summary.csv", rows)
    payload = {
        "experiment": "optional_benchmarks",
        "smoke": smoke,
        "benchmarks": rows,
        "claim": "Meta-World support is audited in results/benchmarks/metaworld; other optional benchmark claims remain unsupported until artifacts exist.",
        "artifacts": {
            "summary_csv": str(out_dir / "summary.csv"),
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
