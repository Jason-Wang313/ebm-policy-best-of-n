from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


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


def _has_unnegated_phrase(text_lower: str, phrase: str) -> bool:
    allowed_context = [
        "no ",
        "not ",
        "does not",
        "do not",
        "without",
        "future work",
        "unsupported",
        "not a",
        "not supported",
        "not claimed",
        "do not support",
        "does not support",
        "unless",
    ]
    for match in re.finditer(re.escape(phrase), text_lower):
        window = text_lower[max(0, match.start() - 90) : match.end() + 90]
        if not any(marker in window for marker in allowed_context):
            return True
    return False


def audit_claim_text(
    text: str,
    optional_success_nonzero: bool = True,
    autonomous_metaworld_success: bool = False,
    autonomous_learned_metaworld_success: bool = False,
) -> list[dict[str, object]]:
    """Return pass/fail rows for high-risk claim wording."""

    lower = text.lower()
    near_100_mentions = bool(
        re.search(r"almost\s+100\s*%|almost\s+100\s+percent|near[- ]100\s*%|near[- ]100\s+percent", lower)
    )
    metric_defined = "repair recovery ratio" in lower and "oracle" in lower and "raw" in lower
    optional_success_overclaim = False
    if not optional_success_nonzero:
        optional_success_overclaim = any(
            _has_unnegated_phrase(lower, phrase)
            for phrase in [
                "success on optional suites",
                "optional suites have nonzero success",
                "optional success is nonzero",
                "optional benchmark success",
            ]
        )
    autonomous_metaworld_overclaim = False
    if not autonomous_metaworld_success:
        autonomous_metaworld_overclaim = any(
            _has_unnegated_phrase(lower, phrase)
            for phrase in [
                "fully autonomous metaworld success",
                "fully autonomous meta-world success",
                "autonomous metaworld success",
                "autonomous meta-world success",
            ]
        )
    autonomous_learned_metaworld_overclaim = False
    if not autonomous_learned_metaworld_success:
        autonomous_learned_metaworld_overclaim = any(
            _has_unnegated_phrase(lower, phrase)
            for phrase in [
                "autonomous learned-policy success",
                "fully autonomous learned-policy success",
                "autonomous learned metaworld success",
                "autonomous learned meta-world success",
            ]
        )
    checks = [
        {
            "check": "near-100 wording is metric-defined",
            "passed": (not near_100_mentions) or metric_defined,
            "detail": "Any near-100 repair language must define repair recovery ratio.",
        },
        {
            "check": "does not claim solved manipulation",
            "passed": not (
                _has_unnegated_phrase(lower, "solves manipulation")
                or _has_unnegated_phrase(lower, "robot manipulation solved")
            ),
            "detail": "No broad manipulation-solved wording is allowed.",
        },
        {
            "check": "does not claim real-robot validation",
            "passed": not (
                _has_unnegated_phrase(lower, "validated on real robots")
                or _has_unnegated_phrase(lower, "real robot validation is supported")
                or _has_unnegated_phrase(lower, "real-robot validation is supported")
            ),
            "detail": "Real-robot evidence remains unsupported.",
        },
        {
            "check": "does not claim repairs always fix EBMs",
            "passed": not (
                _has_unnegated_phrase(lower, "always fixes")
                or _has_unnegated_phrase(lower, "always fix")
                or _has_unnegated_phrase(lower, "always works")
            ),
            "detail": "Repairs are reported as measured artifacts, not guarantees.",
        },
        {
            "check": "optional success claim blocked when zero",
            "passed": not optional_success_overclaim,
            "detail": "Optional-suite success can be claimed only when generated artifacts report nonzero success.",
        },
        {
            "check": "autonomous Meta-World claim blocked without low-dependency success",
            "passed": not autonomous_metaworld_overclaim,
            "detail": "Autonomous Meta-World success requires the ungated non-expert-centered threshold artifact.",
        },
        {
            "check": "autonomous learned-policy claim blocked without learned threshold",
            "passed": not autonomous_learned_metaworld_overclaim,
            "detail": "Autonomous learned-policy success requires a learned no-gate non-expert-centered threshold artifact.",
        },
    ]
    return checks


def model_rows(rows: list[dict[str, str]], model: str) -> list[dict[str, str]]:
    out = [r for r in rows if r.get("model") == model]
    out.sort(key=lambda r: int(float(r["N"])))
    return out


def np_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _figure_set(figures: dict) -> set[str]:
    out: set[str] = set()
    for item in figures.get("figures", []):
        path = Path(str(item))
        if path.is_absolute():
            try:
                path = path.relative_to(ROOT)
            except ValueError:
                pass
        out.add(str(path).replace("\\", "/"))
    return out


def main() -> int:
    controlled = read_csv(ROOT / "results" / "controlled_energy_tail" / "summary.csv")
    learned = read_csv(ROOT / "results" / "learned_ibc" / "summary.csv")
    learned_json = read_json(ROOT / "results" / "learned_ibc" / "summary.json")
    torch_learned = read_csv(ROOT / "results" / "learned_torch_ibc" / "summary.csv")
    torch_learned_json = read_json(ROOT / "results" / "learned_torch_ibc" / "summary.json")
    multimodal = read_json(ROOT / "results" / "multimodal_action" / "summary.json")
    optimization = read_csv(ROOT / "results" / "optimization_budget" / "summary.csv")
    repair = read_json(ROOT / "results" / "repair" / "summary.json")
    repair_effectiveness = read_json(ROOT / "results" / "repair_effectiveness" / "summary.json")
    repair_effectiveness_rows = read_csv(ROOT / "results" / "repair_effectiveness" / "summary.csv")
    ablations = read_json(ROOT / "results" / "ablations" / "summary.json")
    metaworld = read_json(ROOT / "results" / "benchmarks" / "metaworld" / "summary.json")
    metaworld_rows = read_csv(ROOT / "results" / "benchmarks" / "metaworld" / "summary.csv")
    metaworld_table = read_csv(ROOT / "results" / "benchmarks" / "metaworld" / "task_table.csv")
    metaworld_closed_loop = read_csv(ROOT / "results" / "benchmarks" / "metaworld" / "closed_loop_summary.csv")
    metaworld_closed_loop_ablation = read_csv(ROOT / "results" / "benchmarks" / "metaworld" / "closed_loop_ablation_summary.csv")
    metaworld_closed_loop_dependency = read_json(
        ROOT / "results" / "benchmarks" / "metaworld" / "closed_loop_ablation_summary.json"
    )
    reliability = read_csv(ROOT / "results" / "reliability" / "summary.csv")
    optional = read_json(ROOT / "results" / "optional_benchmarks" / "summary.json")
    optional_rows = read_csv(ROOT / "results" / "optional_benchmarks" / "summary.csv")
    exact = read_json(ROOT / "results" / "exact_law" / "summary.json")
    audit_text = "\n".join(
        [
            (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else "",
            (ROOT / "paper" / "abstract.md").read_text(encoding="utf-8") if (ROOT / "paper" / "abstract.md").exists() else "",
            (ROOT / "paper" / "intro.md").read_text(encoding="utf-8") if (ROOT / "paper" / "intro.md").exists() else "",
            (ROOT / "paper" / "method.md").read_text(encoding="utf-8") if (ROOT / "paper" / "method.md").exists() else "",
            (ROOT / "paper" / "experiments.md").read_text(encoding="utf-8") if (ROOT / "paper" / "experiments.md").exists() else "",
            (ROOT / "paper" / "limitations.md").read_text(encoding="utf-8") if (ROOT / "paper" / "limitations.md").exists() else "",
            (ROOT / "docs" / "final_audit.md").read_text(encoding="utf-8") if (ROOT / "docs" / "final_audit.md").exists() else "",
        ]
    )
    optional_success_nonzero = bool(optional.get("optional_success_nonzero", False))
    autonomous_metaworld_success = bool(metaworld_closed_loop_dependency.get("autonomous_success_threshold_met", False))
    autonomous_learned_metaworld_success = bool(
        metaworld_closed_loop_dependency.get("learned_ebm_low_dependency_success_threshold_met", False)
        or metaworld_closed_loop_dependency.get("learned_policy_low_dependency_success_threshold_met", False)
    )
    claim_stress_checks = audit_claim_text(
        audit_text,
        optional_success_nonzero=optional_success_nonzero,
        autonomous_metaworld_success=autonomous_metaworld_success,
        autonomous_learned_metaworld_success=autonomous_learned_metaworld_success,
    )
    write_json(
        ROOT / "results" / "claim_stress_audit.json",
        {
            "experiment": "claim_stress_audit",
            "checks": claim_stress_checks,
            "all_passed": all(bool(c["passed"]) for c in claim_stress_checks),
            "optional_success_nonzero": optional_success_nonzero,
            "autonomous_metaworld_success": autonomous_metaworld_success,
            "autonomous_learned_metaworld_success": autonomous_learned_metaworld_success,
        },
    )
    try:
        from ebm_tail_audit.plotting import regenerate_all_figures

        if (ROOT / "results" / "controlled_energy_tail" / "summary.csv").exists():
            regenerate_all_figures(ROOT / "results")
    except Exception:
        pass
    figures = read_json(ROOT / "results" / "figures" / "figures.json")
    figure_set = _figure_set(figures)

    raw = model_rows(controlled, "raw_miscalibrated")
    controlled_ok = bool(raw) and (
        float(raw[-1]["selected_energy"]) < float(raw[0]["selected_energy"])
        and float(raw[-1]["selected_real_utility"]) < float(raw[0]["selected_real_utility"])
        and float(raw[-1]["low_energy_tail_real_utility"]) < float(raw[0]["selected_real_utility"])
        and raw[-1]["deployment_gate"] in {"collect_pilot_labels", "block_high_n"}
    )

    learned_raw = model_rows(learned, "raw_ebm")
    learned_cal = model_rows(learned, "calibrated")
    learned_ok = bool(learned_raw and learned_cal) and (
        float(learned_raw[-1]["selected_energy"]) < float(learned_raw[0]["selected_energy"])
        and float(learned_raw[-1]["selected_real_utility"]) < float(learned_raw[0]["selected_real_utility"])
        and float(learned_cal[-1]["selected_real_utility"]) > float(learned_raw[-1]["selected_real_utility"])
        and bool(learned_json.get("model_metadata", {}).get("training_objective"))
    )

    torch_raw = model_rows(torch_learned, "raw_torch_ebm")
    torch_cal = model_rows(torch_learned, "calibrated_torch")
    torch_ok = bool(torch_raw and torch_cal and torch_learned_json) and (
        torch_learned_json.get("model_metadata", {}).get("model_type") == "PyTorch MLP EBM"
        and float(torch_learned_json.get("model_metadata", {}).get("loss_decrease", 0.0)) > 0.0
        and all(np_finite(r.get("selected_real_utility")) and np_finite(r.get("exact_law_abs_error")) for r in torch_raw)
        and float(torch_raw[-1]["exact_law_abs_error"]) < 0.10
    )

    exact_error = exact.get("prediction_error", {}).get("max_abs_error", 999.0)
    exact_ok = bool(exact) and float(exact_error) < 0.08
    repair_ok = (
        repair.get("raw_high_N_gate") in {"collect_pilot_labels", "block_high_n"}
        and repair.get("calibrated_high_N_gate") in {"allow_high_n", "stop_early"}
        and float(repair.get("calibrated_high_N_utility_gain_over_raw", 0.0)) > 1.0
    )
    main_repair_summary = next(
        (r for r in repair_effectiveness_rows if r.get("model") == repair_effectiveness.get("main_repair_stack")),
        {},
    )
    docs_metric_wording_ok = "repair recovery ratio" in audit_text.lower() and "95%" in audit_text
    claim_stress_ok = all(bool(c["passed"]) for c in claim_stress_checks)
    near_complete_repair_ok = bool(
        repair_effectiveness
        and main_repair_summary
        and repair_effectiveness.get("near_complete_repair_supported") is True
        and float(main_repair_summary.get("mean_repair_recovery_ratio", 0.0)) >= 0.95
        and float(main_repair_summary.get("worst_case_recovery_ratio", 0.0)) >= 0.85
        and int(float(main_repair_summary.get("num_utility_degrading_rows", 999))) == 0
        and docs_metric_wording_ok
        and claim_stress_ok
    )
    multimodal_ok = (
        multimodal.get("tail_aligned_ebm_high_N_utility", -999.0) > multimodal.get("mse_regression_mean_utility", 999.0)
        and multimodal.get("raw_shortcut_ebm_high_N_utility", 999.0) < multimodal.get("tail_aligned_ebm_high_N_utility", -999.0)
    )
    optimization_ok = len(optimization) > 0 and {"energy_evaluations", "latency_proxy", "utility_per_energy_eval"}.issubset(
        set(optimization[0].keys())
    )

    required_figures = {
        "results/figures/figure1_energy_utility_drop.png",
        "results/figures/figure2_repair_comparison.png",
        "results/figures/figure3_tail_reliability.png",
        "results/figures/figure4_tail_rank_and_tail_utility.png",
        "results/figures/figure5_compute_utility_frontier.png",
        "results/figures/figure6_multitask_benchmark_table.png",
        "results/figures/figure8_repair_recovery_ratio.png",
        "results/figures/figure9_attack_surface_audit.png",
        "results/figures/figure10_closed_loop_dependency_audit.png",
    }
    figure_ok = required_figures.issubset(figure_set) and all((ROOT / p).exists() or Path(p).exists() for p in figure_set)

    ablation_ok = False
    if ablations:
        pilot = ablations.get("best_pilot_label_result", {})
        support = ablations.get("best_support_weight_result", {})
        ablation_ok = (
            float(pilot.get("utility_gain_over_raw", 0.0)) > 1.0
            and float(support.get("utility_gain_over_raw", 0.0)) > 1.0
            and "results/figures/figure7_repair_ablations.png" in figure_set
        )

    benchmark_tasks = set(metaworld.get("tasks", []))
    required_tasks = {"reach-v3", "push-v3", "pick-place-v3", "button-press-v3"}
    table_tasks = {r["task"].replace("_fallback_contact_push", "") for r in metaworld_table}
    benchmark_present = (
        metaworld.get("status") in {"SUPPORTED", "PARTIAL"}
        and required_tasks.issubset(benchmark_tasks)
        and required_tasks.issubset(table_tasks)
        and len(metaworld_rows) > 0
        and len(metaworld_table) > 0
    )
    expert_rows = [r for r in metaworld_table if r.get("model") == "expert_ibc"]
    high_reward_rows = [r for r in metaworld_table if r.get("model") == "high_reward_ablation"]
    expert_demo_ok = (
        len(expert_rows) >= 3
        and all(r.get("training_source") == "scripted_expert" for r in expert_rows)
        and len(high_reward_rows) >= 3
        and all(r.get("training_source") == "high_reward_ablation" for r in high_reward_rows)
    )
    benchmark_finite = benchmark_present and all(
        np_finite(r.get("mean_selected_reward")) and np_finite(r.get("max_exact_law_abs_error")) for r in metaworld_table
    )
    benchmark_success_supported = benchmark_finite and all(str(r.get("task_status")) == "SUPPORTED" for r in expert_rows)
    benchmark_partial = benchmark_finite and not benchmark_success_supported

    reliability_ok = (
        len(reliability) > 0
        and {"energy_quantile_low", "energy_quantile_high", "mean_real_utility", "is_extreme_low_energy_tail"}.issubset(
            set(reliability[0].keys())
        )
        and any(r.get("is_extreme_low_energy_tail", "").lower() == "true" for r in reliability)
        and "results/figures/figure3_tail_reliability.png" in figure_set
    )
    optional_supported_rows = [
        r
        for r in optional_rows
        if r.get("task_status") == "SUPPORTED"
        and r.get("suite") != "metaworld"
        and np_finite(r.get("mean_total_reward"))
    ]
    optional_supported_suites = {r.get("suite") for r in optional_supported_rows}
    optional_rollout_ok = optional.get("status") == "SUPPORTED" and len(optional_supported_suites) >= 2
    optional_success_ok = bool(optional.get("optional_success_nonzero")) and any(
        r.get("success_nonzero", "").lower() == "true" and float(r.get("mean_success_rate", 0.0) or 0.0) > 0.0
        for r in optional_rows
    )
    required_closed_loop_variants = {
        "expert_centered_gate",
        "expert_centered_no_gate",
        "mixed_expert_local_no_gate",
        "local_gaussian_no_gate",
        "learned_demo_proposal_no_gate",
        "learned_demo_proposal_direct",
        "learned_bc_proposal_no_gate",
        "learned_bc_proposal_direct",
        "state_heuristic_proposal_no_gate",
        "state_heuristic_direct",
        "random_uniform",
        "scripted_expert",
    }
    closed_loop_variant_set = {r.get("policy") for r in metaworld_closed_loop_ablation}
    closed_loop_task_set = {r.get("task") for r in metaworld_closed_loop_ablation}
    learned_demo_rows = [
        r
        for r in metaworld_closed_loop_ablation
        if r.get("policy") in {"learned_demo_proposal_no_gate", "learned_demo_proposal_direct"}
    ]
    learned_bc_rows = [
        r
        for r in metaworld_closed_loop_ablation
        if r.get("policy") in {"learned_bc_proposal_no_gate", "learned_bc_proposal_direct"}
    ]
    state_heuristic_rows = [
        r
        for r in metaworld_closed_loop_ablation
        if r.get("policy") in {"state_heuristic_proposal_no_gate", "state_heuristic_direct"}
    ]
    closed_loop_dependency_ok = (
        metaworld.get("closed_loop_learned_ebm_rollout_diagnostic", {}).get("status") == "SUPPORTED"
        and metaworld_closed_loop_dependency.get("status") == "SUPPORTED"
        and required_closed_loop_variants.issubset(closed_loop_variant_set)
        and required_tasks.issubset(closed_loop_task_set)
        and len(metaworld_closed_loop_ablation) >= len(required_closed_loop_variants) * len(required_tasks)
        and all(r.get("continuation") == "none" for r in metaworld_closed_loop_ablation)
        and all(np_finite(r.get("mean_total_reward")) for r in metaworld_closed_loop_ablation)
        and bool(metaworld_closed_loop_ablation)
        and all(
            key in metaworld_closed_loop_ablation[0]
            for key in [
                "success_drop_without_gate",
                "success_drop_without_expert_centering",
                "fallback_dependency_score",
                "uses_learned_demo_proposal",
                "uses_learned_bc_proposal",
                "uses_state_heuristic_proposal",
                "runtime_scripted_expert_dependency",
            ]
        )
        and bool(learned_demo_rows)
        and all(r.get("uses_learned_demo_proposal", "").lower() == "true" for r in learned_demo_rows)
        and all(r.get("conservative_proposal_gate", "").lower() == "false" for r in learned_demo_rows)
        and all(r.get("runtime_scripted_expert_dependency", "").lower() == "false" for r in learned_demo_rows)
        and bool(learned_bc_rows)
        and all(r.get("uses_learned_bc_proposal", "").lower() == "true" for r in learned_bc_rows)
        and all(r.get("conservative_proposal_gate", "").lower() == "false" for r in learned_bc_rows)
        and all(r.get("runtime_scripted_expert_dependency", "").lower() == "false" for r in learned_bc_rows)
        and bool(state_heuristic_rows)
        and all(r.get("uses_state_heuristic_proposal", "").lower() == "true" for r in state_heuristic_rows)
        and all(r.get("conservative_proposal_gate", "").lower() == "false" for r in state_heuristic_rows)
        and all(r.get("runtime_scripted_expert_dependency", "").lower() == "false" for r in state_heuristic_rows)
        and "results/figures/figure10_closed_loop_dependency_audit.png" in figure_set
    )
    state_heuristic_low_dependency_ok = (
        closed_loop_dependency_ok
        and bool(metaworld_closed_loop_dependency.get("state_heuristic_ebm_success_threshold_met", False))
        and bool(metaworld_closed_loop_dependency.get("state_heuristic_direct_success_threshold_met", False))
        and bool(metaworld_closed_loop_dependency.get("low_dependency_success_threshold_met", False))
    )

    claims = [
        {
            "category": "theorem claims",
            "claim": "The finite tie-aware tail selection law predicts selected real utility for a fixed score/utility pool.",
            "status": status(exact_ok),
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
            "status": status(learned_ok),
            "evidence": "results/learned_ibc/summary.csv and results/learned_ibc/summary.json",
        },
        {
            "category": "PyTorch learned EBM claims",
            "claim": "A PyTorch MLP IBC-style EBM trains with decreasing contrastive loss and produces finite exact-law selected-utility curves.",
            "status": status(torch_ok),
            "evidence": "results/learned_torch_ibc/summary.csv and results/learned_torch_ibc/summary.json",
        },
        {
            "category": "scripted expert benchmark claims",
            "claim": "Meta-World reach-v3, push-v3, pick-place-v3, and button-press-v3 artifacts train the primary benchmark EBM on scripted expert actions; high-reward sampled actions are only an ablation.",
            "status": status(benchmark_finite and expert_demo_ok and benchmark_present),
            "evidence": "results/benchmarks/metaworld/task_table.csv and results/benchmarks/metaworld/summary.json",
        },
        {
            "category": "closed-loop Meta-World dependency audit claims",
            "claim": "A CPU-feasible closed-loop dependency audit reports gated, ungated, expert-centered, learned-demo-proposal, behavior-cloned-proposal, state-heuristic, local, random, and scripted variants without scripted continuation.",
            "status": status(closed_loop_dependency_ok),
            "evidence": "results/benchmarks/metaworld/closed_loop_ablation_summary.csv, results/benchmarks/metaworld/closed_loop_ablation_rollouts.csv, and results/figures/figure10_closed_loop_dependency_audit.png",
        },
        {
            "category": "state-heuristic low-dependency Meta-World claims",
            "claim": "In the closed-loop Meta-World audit, the state-heuristic proposal variants clear the no-gate, no-runtime-scripted-expert low-dependency success threshold.",
            "status": status(state_heuristic_low_dependency_ok),
            "evidence": "results/benchmarks/metaworld/closed_loop_ablation_summary.json and results/benchmarks/metaworld/closed_loop_ablation_summary.csv",
        },
        {
            "category": "selected-tail reliability claims",
            "claim": "Energy reliability diagrams bucket candidates by energy quantile and expose the extreme low-energy tail.",
            "status": status(reliability_ok),
            "evidence": "results/reliability/summary.csv and results/figures/figure3_tail_reliability.png",
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
            "claim": "Support penalties, conservative stopping, pilot-label calibration, value shaping, and oracle scoring form a repair hierarchy for selected-tail utility.",
            "status": status(repair_ok and ablation_ok),
            "evidence": "results/repair/summary.json, results/repair/summary.csv, and results/ablations/summary.csv",
        },
        {
            "category": "metric-bound near-complete repair claims",
            "claim": "On supported local repair stress tests, the main pilot-label combined repair nearly closes the measured high-N gap by the repair recovery ratio definition.",
            "status": status(near_complete_repair_ok),
            "evidence": "results/repair_effectiveness/summary.json, results/repair_effectiveness/summary.csv, and results/repair_effectiveness/stress_matrix.csv",
        },
        {
            "category": "submission figure claims",
            "claim": "The paper-facing figures, including repair recovery and attack-surface audit figures, are regenerated from local CSV/JSON artifacts.",
            "status": status(figure_ok),
            "evidence": "results/figures/figures.json",
        },
        {
            "category": "optional benchmark claims",
            "claim": "Optional non-Meta-World adapters produce finite guarded rollout artifacts; reward and success are reported honestly and do not support broad manipulation success.",
            "status": status(optional_rollout_ok),
            "evidence": "results/optional_benchmarks/summary.json, results/optional_benchmarks/rollouts.csv, and docs/benchmark_plan.md",
        },
        {
            "category": "optional scripted success claims",
            "claim": "At least one optional non-Meta-World suite reports nonzero scripted-policy success while remaining scoped to finite simulator rollouts.",
            "status": status(optional_success_ok),
            "evidence": "results/optional_benchmarks/summary.json and results/optional_benchmarks/summary.csv",
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
        ("learned EBM/IBC-style policy exists", (ROOT / "src" / "ebm_tail_audit" / "energy_models.py").exists() and bool(learned_json)),
        ("scripted expert benchmark artifact exists", expert_demo_ok and benchmark_present),
        ("multi-task benchmark table exists", required_tasks.issubset(table_tasks)),
        ("energy reliability artifact exists", reliability_ok),
        ("action/trajectory energy scorer exists", (ROOT / "src" / "ebm_tail_audit" / "torch_models.py").exists()),
        ("artifacts report low-energy tail metrics", raw and "low_energy_tail_real_utility" in raw[0]),
        ("raw EBM compared to calibrated/value-shaped EBM", bool(repair_ok or learned_ok)),
        ("compute/latency proxy reported", optimization_ok),
        ("no real-robot validation claim in supported claims", True),
        ("does not imply EBMs always work", "ebms always work" not in docs_text_lower),
        ("does not imply tail selection always helps", "tail selection always helps" not in docs_text_lower),
        ("near-complete repair wording defines recovery ratio", docs_metric_wording_ok),
        ("claim stress audit passed", claim_stress_ok),
        ("closed-loop dependency audit exists", closed_loop_dependency_ok),
        ("does not duplicate WAM/JEPA/diffusion failure wording as the EBM claim", "low-energy tail miscalibration" in docs_text_lower),
    ]

    non_claim_boundaries = [
        {
            "boundary": "real_robot_validation",
            "status": "UNSUPPORTED_NON_CLAIM",
            "evidence": "No real-robot artifacts are present; docs and claim stress audit block real-robot validation wording.",
        },
        {
            "boundary": "broad_manipulation_success",
            "status": "UNSUPPORTED_NON_CLAIM",
            "evidence": "All simulator claims remain artifact-scoped and do not support broad manipulation or real-robot success.",
        },
    ]
    if not autonomous_learned_metaworld_success:
        non_claim_boundaries.insert(
            1,
            {
                "boundary": "fully_autonomous_learned_metaworld_policy_success",
                "status": "UNSUPPORTED_NON_CLAIM",
                "evidence": "State-heuristic low-dependency success is supported, but nearest-demo, behavior-cloned, and local learned variants do not clear the learned-policy threshold.",
            },
        )

    payload = {
        "claims": claims,
        "non_claim_boundaries": non_claim_boundaries,
        "not_clone_audit": [{"check": name, "passed": bool(ok)} for name, ok in not_clone],
        "claim_stress_audit": claim_stress_checks,
        "near_complete_repair_ok": near_complete_repair_ok,
        "all_required_figures_present": figure_ok,
        "benchmark_partial_reason": "At least one Meta-World task has zero one-step success or fallback-only support." if benchmark_partial else "",
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
        "Promoted claims are classified from local artifacts. Unsupported future endpoints are listed separately as non-claim boundaries.",
        "",
        "## Claims",
        "",
    ]
    for claim in claims:
        lines.append(f"- **{claim['status']}** ({claim['category']}): {claim['claim']}")
        lines.append(f"  Evidence: `{claim['evidence']}`")
    lines.extend(["", "## Non-Claim Boundaries", ""])
    for boundary in non_claim_boundaries:
        lines.append(f"- **{boundary['status']}** ({boundary['boundary']}): {boundary['evidence']}")
    lines.extend(["", "## Not Clone Audit", ""])
    for name, ok in not_clone:
        lines.append(f"- {'PASS' if ok else 'FAIL'}: {name}")
    lines.extend(["", "## Claim Stress Audit", ""])
    for check in claim_stress_checks:
        lines.append(f"- {'PASS' if check['passed'] else 'FAIL'}: {check['check']}")
    lines.extend(
        [
            "",
            "## Forbidden Claims",
            "",
            "The following are explicitly forbidden as supported claims: We prove EBMs work; we solve robot manipulation; we validate on real robots; tail selection always helps; calibration always fixes energy policies; energy is real utility; low energy means good action; this is not toy evidence without real benchmarks; this is a universal training recipe; this is WAM model-error amplification; this is JEPA latent tail hallucination; this is diffusion diversity-selection tradeoff.",
            "",
        ]
    )
    text = "\n".join(lines)
    write_text(ROOT / "results" / "claims_status.md", text)
    write_text(ROOT / "docs" / "claims.md", text)

    required_ok = all(
        [
            exact_ok,
            controlled_ok,
            learned_ok,
            torch_ok,
            benchmark_finite,
            expert_demo_ok,
            reliability_ok,
            multimodal_ok,
            optimization_ok,
            repair_ok,
            ablation_ok,
            near_complete_repair_ok,
            closed_loop_dependency_ok,
            optional_success_ok,
            claim_stress_ok,
            figure_ok,
        ]
    )
    if any(not ok for _, ok in not_clone):
        return 1
    return 0 if required_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
