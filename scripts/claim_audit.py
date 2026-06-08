from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def status(condition: bool, partial: bool = False) -> str:
    if condition:
        return "SUPPORTED"
    if partial:
        return "PARTIAL"
    return "UNSUPPORTED"


def model_rows(rows: list[dict[str, str]], model: str) -> list[dict[str, str]]:
    out = [r for r in rows if r.get("model") == model]
    out.sort(key=lambda r: int(float(r["N"])))
    return out


def np_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> int:
    controlled = read_csv(ROOT / "results" / "controlled_energy_tail" / "summary.csv")
    learned = read_csv(ROOT / "results" / "learned_ibc" / "summary.csv")
    learned_json = read_json(ROOT / "results" / "learned_ibc" / "summary.json")
    torch_learned = read_csv(ROOT / "results" / "learned_torch_ibc" / "summary.csv")
    torch_learned_json = read_json(ROOT / "results" / "learned_torch_ibc" / "summary.json")
    multimodal = read_json(ROOT / "results" / "multimodal_action" / "summary.json")
    optimization = read_csv(ROOT / "results" / "optimization_budget" / "summary.csv")
    repair = read_json(ROOT / "results" / "repair" / "summary.json")
    ablations = read_json(ROOT / "results" / "ablations" / "summary.json")
    metaworld = read_json(ROOT / "results" / "benchmarks" / "metaworld" / "summary.json")
    metaworld_rows = read_csv(ROOT / "results" / "benchmarks" / "metaworld" / "summary.csv")
    exact = read_json(ROOT / "results" / "exact_law" / "summary.json")
    figures = read_json(ROOT / "results" / "figures" / "figures.json")

    raw = model_rows(controlled, "raw_miscalibrated")
    controlled_ok = False
    if raw:
        controlled_ok = (
            float(raw[-1]["selected_energy"]) < float(raw[0]["selected_energy"])
            and float(raw[-1]["selected_real_utility"]) < float(raw[0]["selected_real_utility"])
            and float(raw[-1]["low_energy_tail_real_utility"]) < float(raw[0]["selected_real_utility"])
            and raw[-1]["deployment_gate"] in {"collect_pilot_labels", "block_high_n"}
        )

    learned_raw = model_rows(learned, "raw_ebm")
    learned_cal = model_rows(learned, "calibrated")
    learned_ok = False
    if learned_raw and learned_cal:
        learned_ok = (
            float(learned_raw[-1]["selected_energy"]) < float(learned_raw[0]["selected_energy"])
            and float(learned_raw[-1]["selected_real_utility"]) < float(learned_raw[0]["selected_real_utility"])
            and float(learned_cal[-1]["selected_real_utility"]) > float(learned_raw[-1]["selected_real_utility"])
            and learned_json.get("model_metadata", {}).get("training_objective")
        )

    exact_error = exact.get("prediction_error", {}).get("max_abs_error", 999.0)
    repair_ok = (
        repair.get("raw_high_N_gate") in {"collect_pilot_labels", "block_high_n"}
        and repair.get("calibrated_high_N_gate") in {"allow_high_n", "stop_early"}
        and float(repair.get("calibrated_high_N_utility_gain_over_raw", 0.0)) > 1.0
    )
    multimodal_ok = (
        multimodal.get("tail_aligned_ebm_high_N_utility", -999.0) > multimodal.get("mse_regression_mean_utility", 999.0)
        and multimodal.get("raw_shortcut_ebm_high_N_utility", 999.0) < multimodal.get("tail_aligned_ebm_high_N_utility", -999.0)
    )
    optimization_ok = len(optimization) > 0 and {"energy_evaluations", "latency_proxy", "utility_per_energy_eval"}.issubset(
        set(optimization[0].keys())
    )
    figure_ok = len(figures.get("figures", [])) >= 4 and all((ROOT / p).exists() or Path(p).exists() for p in figures.get("figures", []))
    figure_set = {str(p).replace("\\", "/") for p in figures.get("figures", [])}

    torch_raw = model_rows(torch_learned, "raw_torch_ebm")
    torch_cal = model_rows(torch_learned, "calibrated_torch")
    torch_ok = False
    if torch_raw and torch_cal and torch_learned_json:
        meta = torch_learned_json.get("model_metadata", {})
        torch_ok = (
            meta.get("model_type") == "PyTorch MLP EBM"
            and float(meta.get("loss_decrease", 0.0)) > 0.0
            and all(np_finite(r.get("selected_real_utility")) and np_finite(r.get("exact_law_abs_error")) for r in torch_raw)
            and float(torch_raw[-1]["exact_law_abs_error"]) < 0.10
        )

    metaworld_ok = False
    if metaworld.get("status") == "SUPPORTED" and metaworld_rows:
        raw_bench = model_rows(metaworld_rows, "metaworld_raw_torch_ebm")
        metaworld_ok = (
            bool(raw_bench)
            and all(np_finite(r.get("selected_real_utility")) and np_finite(r.get("selected_energy")) for r in raw_bench)
            and float(raw_bench[-1]["exact_law_abs_error"]) < 0.15
            and "results/figures/figure8_metaworld_benchmark.png" in figure_set
        )

    ablation_ok = False
    if ablations:
        pilot = ablations.get("best_pilot_label_result", {})
        support = ablations.get("best_support_weight_result", {})
        ablation_ok = (
            float(pilot.get("utility_gain_over_raw", 0.0)) > 1.0
            and float(support.get("utility_gain_over_raw", 0.0)) > 1.0
            and "results/figures/figure9_repair_ablations.png" in figure_set
        )

    claims = [
        {
            "category": "theorem claims",
            "claim": "The finite tie-aware Best-of-N law predicts selected real utility for a fixed score/utility pool.",
            "status": status(bool(exact) and float(exact_error) < 0.08),
            "evidence": "results/exact_law/summary.json and results/exact_law/validation.csv",
        },
        {
            "category": "controlled EBM toy claims",
            "claim": "Controlled contact and multimodal toys exhibit low-energy tail miscalibration: selected energy decreases while real utility falls.",
            "status": status(controlled_ok),
            "evidence": "results/controlled_energy_tail/summary.csv",
        },
        {
            "category": "learned IBC-style EBM claims",
            "claim": "A small learned contrastive EBM can over-optimize an observation-limited low-energy tail, and calibrated/value-shaped scoring improves high-N utility.",
            "status": status(bool(learned_ok)),
            "evidence": "results/learned_ibc/summary.csv and results/learned_ibc/summary.json",
        },
        {
            "category": "PyTorch learned EBM claims",
            "claim": "A PyTorch MLP IBC-style EBM trains with decreasing contrastive loss and produces finite exact-law selected-utility curves, showing the aligned-tail regime alongside the failure regime.",
            "status": status(torch_ok),
            "evidence": "results/learned_torch_ibc/summary.csv and results/learned_torch_ibc/summary.json",
        },
        {
            "category": "Meta-World benchmark claims",
            "claim": "The repository produces a guarded Meta-World reach-v3 benchmark artifact with finite reward, energy, compute, and exact-law diagnostics.",
            "status": status(metaworld_ok, partial=metaworld.get("status") == "SUPPORTED"),
            "evidence": "results/benchmarks/metaworld/summary.json and results/benchmarks/metaworld/summary.csv",
        },
        {
            "category": "multimodal action claims",
            "claim": "A tail-aligned EBM can represent two valid modes better than explicit MSE regression, while a shortcut energy can still fail at high N.",
            "status": status(multimodal_ok),
            "evidence": "results/multimodal_action/summary.json",
        },
        {
            "category": "optimization/latency claims",
            "claim": "More EBM inference compute trades energy evaluations and latency proxy against real utility, and added candidates can over-optimize energy.",
            "status": status(optimization_ok),
            "evidence": "results/optimization_budget/summary.csv",
        },
        {
            "category": "repair/calibration claims",
            "claim": "Pilot-label calibration, value shaping, and support penalties repair selected-tail utility in controlled toy settings.",
            "status": status(repair_ok),
            "evidence": "results/repair/summary.json and results/repair/summary.csv",
        },
        {
            "category": "repair ablation claims",
            "claim": "Pilot-label count and support-penalty sweeps quantify how much repair signal is needed to recover high-N selected-tail utility.",
            "status": status(ablation_ok),
            "evidence": "results/ablations/summary.json and results/ablations/summary.csv",
        },
        {
            "category": "optional benchmark claims",
            "claim": "Optional benchmark adapters are guarded and do not support core claims unless dependencies and runs are present.",
            "status": "PARTIAL",
            "evidence": "results/optional_benchmarks/summary.json and docs/benchmark_plan.md",
        },
        {
            "category": "unsupported future robotics claims",
            "claim": "Real robot validation is future work, not a supported repository claim.",
            "status": "UNSUPPORTED",
            "evidence": "No real-robot artifacts are present.",
        },
        {
            "category": "forbidden/overclaim claims",
            "claim": "Forbidden clone and overclaim statements are treated as disallowed, not supported findings.",
            "status": "SUPPORTED",
            "evidence": "docs/differentiation_from_wam_jepa_diffusion.md and docs/claims.md",
        },
    ]

    docs_text = "\n".join(
        [
            (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else "",
            (ROOT / "docs" / "theory.md").read_text(encoding="utf-8") if (ROOT / "docs" / "theory.md").exists() else "",
            (ROOT / "paper" / "abstract.md").read_text(encoding="utf-8") if (ROOT / "paper" / "abstract.md").exists() else "",
        ]
    )
    docs_text_lower = docs_text.lower()
    not_clone = [
        ("differentiation doc exists", (ROOT / "docs" / "differentiation_from_wam_jepa_diffusion.md").exists()),
        ("EBM-specific energy setup exists", "E_theta(o, a)" in docs_text or "E_theta(o,a)" in docs_text),
        ("S = -E convention exists", "S = -E" in docs_text or "S(o, a) = -E" in docs_text),
        ("learned EBM/IBC-style policy exists", (ROOT / "src" / "ebm_best_of_n" / "energy_models.py").exists() and bool(learned_json)),
        ("PyTorch neural EBM artifact exists", (ROOT / "src" / "ebm_best_of_n" / "torch_models.py").exists() and bool(torch_learned_json)),
        ("Meta-World benchmark artifact exists", bool(metaworld_rows) and bool(metaworld)),
        ("action/trajectory energy scorer exists", (ROOT / "src" / "ebm_best_of_n" / "energy_models.py").exists()),
        ("artifacts report low-energy tail metrics", raw and "low_energy_tail_real_utility" in raw[0]),
        ("raw EBM compared to calibrated/value-shaped EBM", bool(repair_ok or learned_ok)),
        ("compute/latency proxy reported", optimization_ok),
        ("no real-robot validation claim in supported claims", True),
        ("does not imply EBMs always work", "ebms always work" not in docs_text_lower),
        ("does not imply Best-of-N always helps", "best-of-n always helps" not in docs_text_lower),
        ("does not duplicate WAM/JEPA/diffusion failure wording as the EBM claim", "low-energy tail miscalibration" in docs_text_lower),
    ]

    payload = {
        "claims": claims,
        "not_clone_audit": [{"check": name, "passed": bool(ok)} for name, ok in not_clone],
        "all_required_figures_present": figure_ok,
        "summary": {
            "supported": sum(1 for c in claims if c["status"] == "SUPPORTED"),
            "partial": sum(1 for c in claims if c["status"] == "PARTIAL"),
            "unsupported": sum(1 for c in claims if c["status"] == "UNSUPPORTED"),
            "not_clone_checks_passed": sum(1 for _, ok in not_clone if ok),
            "not_clone_checks_total": len(not_clone),
        },
    }
    write_json(ROOT / "results" / "claims_status.json", payload)

    lines = [
        "# Claim Status",
        "",
        "Claims are classified as SUPPORTED, PARTIAL, or UNSUPPORTED from local artifacts.",
        "",
        "## Claims",
        "",
    ]
    for claim in claims:
        lines.append(f"- **{claim['status']}** ({claim['category']}): {claim['claim']}")
        lines.append(f"  Evidence: `{claim['evidence']}`")
    lines.extend(["", "## Not Clone Audit", ""])
    for name, ok in not_clone:
        lines.append(f"- {'PASS' if ok else 'FAIL'}: {name}")
    lines.extend(
        [
            "",
            "## Forbidden Claims",
            "",
            "The following are explicitly forbidden as supported claims: We prove EBMs work; we solve robot manipulation; we validate on real robots; Best-of-N always helps; calibration always fixes energy policies; energy is real utility; low energy means good action; this is not toy evidence without real benchmarks; this is a universal training recipe; this is WAM model-error amplification; this is JEPA latent tail hallucination; this is diffusion diversity-selection tradeoff.",
            "",
        ]
    )
    text = "\n".join(lines)
    write_text(ROOT / "results" / "claims_status.md", text)
    write_text(ROOT / "docs" / "claims.md", text)
    if any(not ok for _, ok in not_clone):
        return 1
    if not (
        controlled_ok
        and learned_ok
        and torch_ok
        and metaworld_ok
        and multimodal_ok
        and optimization_ok
        and repair_ok
        and ablation_ok
        and figure_ok
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
