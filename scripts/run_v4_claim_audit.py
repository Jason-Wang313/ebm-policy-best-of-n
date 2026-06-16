from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_PDF = ROOT / "paper" / "final" / "ebm-policy-best-of-n-v4.pdf"
DESKTOP = Path.home() / "OneDrive" / "Desktop"
DESKTOP_PDF = DESKTOP / "ebm-policy-best-of-n-v4.pdf"
OLD_DESKTOP_PDF = DESKTOP / "ebm-policy-best-of-n-v2.pdf"
OLD_V3_DESKTOP_PDF = DESKTOP / "ebm-policy-best-of-n-v3.pdf"
SOURCE_MAP = DESKTOP / "PAPER_SOURCE_MAP.md"
LOG = ROOT / "paper" / "iclr2026" / "build" / "main.log"


def run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def page_count(path: Path) -> int:
    try:
        out = subprocess.check_output(["pdfinfo", str(path)], text=True, stderr=subprocess.STDOUT)
        match = re.search(r"^Pages:\s+(\d+)", out, re.MULTILINE)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    data = path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page\b", data))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    run([sys.executable, "scripts/claim_audit.py"])
    run([sys.executable, "experiments/v4_frozen_evidence.py"])

    summary = read_json(ROOT / "results" / "v4_frozen_evidence" / "summary.json")
    claims = read_json(ROOT / "results" / "claims_status.json")

    require(summary["supported_claims"] >= 16, "too few supported claims")
    require(summary["partial_claims"] == 0, "partial claims remain")
    require(summary["unsupported_claims"] == 0, "unsupported promoted claims remain")
    require(summary["non_claim_boundaries"] >= 3, "non-claim boundaries missing")
    require(summary["not_clone_checks_passed"] == summary["not_clone_checks_total"] >= 18, "not-clone audit failed")
    require(summary["exact_law_max_abs_error"] <= 0.02, "exact-law error too high")
    require(summary["raw_high_n_energy"] < -2.5, "high-N energy tail not extreme enough")
    require(summary["raw_high_n_utility"] < -1.0, "high-N raw utility failure too weak")
    require(summary["raw_high_n_invalid"] >= 0.99, "invalid-action tail failure too weak")
    require(summary["raw_high_n_regret"] >= 3.0, "high-N regret too weak")
    require(summary["main_repair_mean_recovery"] >= 0.95, "main repair mean recovery below threshold")
    require(summary["main_repair_worst_recovery"] >= 0.95, "main repair worst recovery below V4 threshold")
    require(summary["main_repair_utility_degrading_rows"] == 0, "repair degrades utility")
    require(summary["closed_loop_rows"] >= 48 and summary["closed_loop_policies"] >= 12, "closed-loop audit incomplete")
    require(summary["state_heuristic_success"] >= 0.99, "state-heuristic boundary no longer supported")
    require(summary["learned_demo_success"] < 0.5 and summary["learned_bc_success"] < 0.5, "learned-policy boundary disappeared")
    require(summary["optional_supported_suites"] >= 3, "optional adapter coverage too small")
    require(summary["optional_nonzero_success_rows"] >= 1, "optional scripted success artifact missing")
    require(summary["reliability_rows"] >= 1600, "reliability audit too small")
    require(summary["optimization_rows"] >= 150, "compute frontier audit too small")
    require(summary["benchmark_claim_rows"] >= 28, "benchmark claim matrix too small")
    require(summary["claim_gates_passed"] == summary["claim_gate_rows"] >= 9, "claim gates did not all pass")
    require(summary["compute_utility_drop_steps"] >= 5, "compute-frontier stress is too weak")
    require(summary["dependency_stress_rows"] >= 12, "dependency stress rows missing")
    require(summary["metaworld_ci_rows"] >= 28, "Meta-World CI rows missing")
    require(summary["reliability_gap_rows"] >= 16, "reliability gap rows missing")
    require(summary["v4_figure_files"] >= 9, "V4 figures missing")
    require(summary["artifact_files"] >= 100, "artifact inventory too small")
    require(claims["summary"]["supported"] >= 16 and claims["summary"]["unsupported"] == 0, "claim audit summary failed")

    require(REPO_PDF.exists(), f"missing repo final PDF: {REPO_PDF}")
    require(DESKTOP_PDF.exists(), f"missing Desktop final PDF: {DESKTOP_PDF}")
    require(not OLD_DESKTOP_PDF.exists(), f"stale Desktop v2 remains: {OLD_DESKTOP_PDF}")
    require(not OLD_V3_DESKTOP_PDF.exists(), f"stale Desktop v3 remains: {OLD_V3_DESKTOP_PDF}")
    repo_hash = sha256(REPO_PDF)
    desktop_hash = sha256(DESKTOP_PDF)
    require(repo_hash == desktop_hash, "repo/Desktop PDF hashes differ")
    pages = page_count(REPO_PDF)
    require(pages >= 25, f"final PDF has only {pages} pages")

    require(SOURCE_MAP.exists(), "Desktop source map missing")
    expected_row = (
        "| `ebm-policy-best-of-n-v4.pdf` | `C:\\Users\\wangz\\ebm-policy-best-of-n` | "
        "`Jason-Wang313/ebm-policy-best-of-n` |"
    )
    require(expected_row in SOURCE_MAP.read_text(encoding="utf-8"), "source map row is not V4/exact")
    require("ebm-policy-best-of-n-v3.pdf" not in SOURCE_MAP.read_text(encoding="utf-8"), "source map still contains EBM v3")

    if LOG.exists():
        log_text = LOG.read_text(encoding="utf-8", errors="ignore")
        blockers = [
            "! LaTeX Error",
            "Emergency stop",
            "Fatal error",
            "Undefined control sequence",
            "LaTeX Warning: There were undefined references",
            "Package natbib Warning: There were undefined citations",
        ]
        require(not any(blocker in log_text for blocker in blockers), "LaTeX log has blocking warnings/errors")

    print("EBM policy V4 audit passed")
    print(f"pages={pages} sha256={repo_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
