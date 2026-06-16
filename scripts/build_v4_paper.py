from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper" / "iclr2026"
BUILD_DIR = PAPER_DIR / "build"
REPO_FINAL = ROOT / "paper" / "final" / "ebm-policy-best-of-n-v4.pdf"
DESKTOP_FINAL = Path.home() / "OneDrive" / "Desktop" / "ebm-policy-best-of-n-v4.pdf"
STALE_FINAL_NAMES = ["ebm-policy-best-of-n-v3.pdf", "ebm-policy-best-of-n-v2.pdf"]


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("==>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def clean_stale_pdfs() -> None:
    for base in [ROOT / "paper" / "final", Path.home() / "OneDrive" / "Desktop"]:
        for name in STALE_FINAL_NAMES:
            path = base / name
            if path.exists():
                path.unlink()


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["SOURCE_DATE_EPOCH"] = "1700000000"
    env["FORCE_SOURCE_DATE"] = "1"

    clean_stale_pdfs()
    run([sys.executable, "scripts/claim_audit.py"], ROOT, env)
    run([sys.executable, "experiments/v4_frozen_evidence.py"], ROOT, env)

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "main.tex"], PAPER_DIR, env)
    run(["bibtex", "build/main"], PAPER_DIR, env)
    for _ in range(4):
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "main.tex"], PAPER_DIR, env)

    pdf = BUILD_DIR / "main.pdf"
    if not pdf.exists():
        raise FileNotFoundError(pdf)
    REPO_FINAL.parent.mkdir(parents=True, exist_ok=True)
    DESKTOP_FINAL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf, REPO_FINAL)
    shutil.copyfile(pdf, DESKTOP_FINAL)
    clean_stale_pdfs()
    print(f"Repo PDF: {REPO_FINAL}")
    print(f"Desktop PDF: {DESKTOP_FINAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
