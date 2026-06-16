from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v4_frozen_evidence"
FIG_OUT = OUT / "figures"
PAPER_FIG_OUT = ROOT / "results" / "figures" / "v4"
MACROS = ROOT / "paper" / "iclr2026" / "v4_results_macros.tex"
PDF_METADATA = {
    "Creator": "experiments/v4_frozen_evidence.py",
    "CreationDate": None,
    "ModDate": None,
}


def read_csv(rel: str) -> list[dict[str, str]]:
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def num(value: Any, default: float = math.nan) -> float:
    try:
        if value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(np.mean(finite)) if finite else math.nan


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:.{digits}f}"
    if value is None:
        return "NA"
    return str(value)


def macro_line(name: str, value: Any, digits: int = 3) -> str:
    return f"\\newcommand{{\\{name}}}{{{fmt(value, digits)}}}\n"


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, metadata=PDF_METADATA)
    plt.close(fig)


def copy_figures() -> None:
    PAPER_FIG_OUT.mkdir(parents=True, exist_ok=True)
    for path in sorted(FIG_OUT.glob("*.pdf")):
        shutil.copyfile(path, PAPER_FIG_OUT / path.name)


def clean_stale_version_outputs() -> None:
    for directory in [OUT, FIG_OUT, PAPER_FIG_OUT]:
        if not directory.exists():
            continue
        for path in directory.glob("v3_*"):
            if path.is_file():
                path.unlink()


def artifact_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root_name in ["docs", "experiments", "paper", "results", "scripts", "src", "tests"]:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if OUT in path.parents:
                continue
            parts = set(path.parts)
            if "__pycache__" in parts or ".pytest_cache" in parts:
                continue
            if path.relative_to(ROOT).as_posix().startswith("paper/iclr2026/build/"):
                continue
            rows.append(
                {
                    "root": root_name,
                    "suffix": path.suffix.lower() or "none",
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                }
            )
    return rows


def group_rows(rows: list[dict[str, str]], *keys: str) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return dict(grouped)


def bool_from_json(payload: dict[str, Any], key: str) -> bool:
    return bool(payload.get(key, False))


def aggregate_closed_loop(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (policy,), policy_rows in sorted(group_rows(rows, "policy").items()):
        out.append(
            {
                "policy": policy,
                "tasks": len({r.get("task", "") for r in policy_rows}),
                "mean_success_rate": mean([num(r.get("mean_success_rate")) for r in policy_rows]),
                "mean_total_reward": mean([num(r.get("mean_total_reward")) for r in policy_rows]),
                "mean_fallback_rate": mean([num(r.get("mean_gate_fallback_rate")) for r in policy_rows]),
                "runtime_scripted_expert_dependency": any(
                    r.get("runtime_scripted_expert_dependency", "").lower() == "true" for r in policy_rows
                ),
                "uses_learned_demo_proposal": any(
                    r.get("uses_learned_demo_proposal", "").lower() == "true" for r in policy_rows
                ),
                "uses_learned_bc_proposal": any(
                    r.get("uses_learned_bc_proposal", "").lower() == "true" for r in policy_rows
                ),
                "uses_state_heuristic_proposal": any(
                    r.get("uses_state_heuristic_proposal", "").lower() == "true" for r in policy_rows
                ),
            }
        )
    return out


def aggregate_tail_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    high_n = [r for r in rows if int(num(r.get("N"), -1)) == 512]
    out: list[dict[str, Any]] = []
    for (model,), model_rows in sorted(group_rows(high_n, "model").items()):
        gates = Counter(r.get("deployment_gate", "") for r in model_rows)
        out.append(
            {
                "model": model,
                "rows": len(model_rows),
                "selected_energy": mean([num(r.get("selected_energy")) for r in model_rows]),
                "selected_real_utility": mean([num(r.get("selected_real_utility")) for r in model_rows]),
                "low_energy_tail_real_utility": mean([num(r.get("low_energy_tail_real_utility")) for r in model_rows]),
                "invalid_action_rate": mean([num(r.get("invalid_action_rate")) for r in model_rows]),
                "support_distance": mean([num(r.get("support_distance")) for r in model_rows]),
                "high_n_regret": mean([num(r.get("high_n_regret")) for r in model_rows]),
                "tail_rank_correlation": mean([num(r.get("tail_rank_correlation")) for r in model_rows]),
                "dominant_gate": gates.most_common(1)[0][0] if gates else "",
            }
        )
    return out


def metaworld_claim_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: (r.get("task", ""), r.get("model", ""))):
        model = row.get("model", "")
        if model == "high_reward_ablation":
            claim_role = "negative_control"
        elif "calibrated" in model or "support" in model:
            claim_role = "repair_or_sensitivity"
        else:
            claim_role = "primary_scripted_expert_baseline"
        out.append(
            {
                "task": row.get("task", ""),
                "model": model,
                "training_source": row.get("training_source", ""),
                "claim_role": claim_role,
                "task_status": row.get("task_status", ""),
                "mean_selected_reward": num(row.get("mean_selected_reward")),
                "selected_reward_ci_low": num(row.get("selected_reward_ci_low")),
                "selected_reward_ci_high": num(row.get("selected_reward_ci_high")),
                "mean_selected_success": num(row.get("mean_selected_success")),
                "mean_high_n_regret": num(row.get("mean_high_n_regret")),
                "mean_support_distance": num(row.get("mean_support_distance")),
                "max_exact_law_abs_error": num(row.get("max_exact_law_abs_error")),
                "num_seeds": int(num(row.get("num_seeds"), 0)),
            }
        )
    return out


def dependency_stress_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (policy,), policy_rows in sorted(group_rows(rows, "policy").items()):
        mean_success = mean([num(r.get("mean_success_rate")) for r in policy_rows])
        min_success = min([num(r.get("mean_success_rate")) for r in policy_rows])
        mean_reward = mean([num(r.get("mean_total_reward")) for r in policy_rows])
        fallback = mean([num(r.get("mean_gate_fallback_rate")) for r in policy_rows])
        expert_dep = any(r.get("runtime_scripted_expert_dependency", "").lower() == "true" for r in policy_rows)
        learned_demo = any(r.get("uses_learned_demo_proposal", "").lower() == "true" for r in policy_rows)
        learned_bc = any(r.get("uses_learned_bc_proposal", "").lower() == "true" for r in policy_rows)
        state = any(r.get("uses_state_heuristic_proposal", "").lower() == "true" for r in policy_rows)
        if policy == "random_uniform":
            claim_role = "negative_control"
        elif state and not expert_dep:
            claim_role = "supported_low_dependency_state_heuristic"
        elif learned_demo or learned_bc:
            claim_role = "learned_policy_boundary"
        elif expert_dep:
            claim_role = "expert_dependency_boundary"
        else:
            claim_role = "stress_baseline"
        out.append(
            {
                "policy": policy,
                "claim_role": claim_role,
                "tasks": len({r.get("task", "") for r in policy_rows}),
                "mean_success_rate": mean_success,
                "min_task_success_rate": min_success,
                "mean_total_reward": mean_reward,
                "mean_fallback_rate": fallback,
                "runtime_scripted_expert_dependency": expert_dep,
                "uses_learned_demo_proposal": learned_demo,
                "uses_learned_bc_proposal": learned_bc,
                "uses_state_heuristic_proposal": state,
            }
        )
    return out


def reliability_gap_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, grouped_rows in sorted(group_rows(rows, "benchmark", "task", "model").items()):
        extreme = [r for r in grouped_rows if r.get("is_extreme_low_energy_tail", "").lower() == "true"]
        rest = [r for r in grouped_rows if r.get("is_extreme_low_energy_tail", "").lower() != "true"]
        if not extreme or not rest:
            continue
        extreme_utility = mean([num(r.get("mean_real_utility")) for r in extreme])
        rest_utility = mean([num(r.get("mean_real_utility")) for r in rest])
        out.append(
            {
                "benchmark": key[0],
                "task": key[1],
                "model": key[2],
                "extreme_rows": len(extreme),
                "other_rows": len(rest),
                "extreme_tail_utility": extreme_utility,
                "other_tail_utility": rest_utility,
                "extreme_minus_other_utility": extreme_utility - rest_utility,
                "extreme_invalid_rate": mean([num(r.get("invalid_rate")) for r in extreme]),
                "extreme_support_distance": mean([num(r.get("support_distance")) for r in extreme]),
                "extreme_success_rate": mean([num(r.get("success_rate")) for r in extreme]),
            }
        )
    return out


def compute_stress_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (policy,), policy_rows in sorted(group_rows(rows, "policy").items()):
        drops = 0
        for (_,), seed_rows in group_rows(policy_rows, "seed").items():
            ordered = sorted(seed_rows, key=lambda r: num(r.get("energy_evaluations"), 0.0))
            previous = math.nan
            for row in ordered:
                utility = num(row.get("selected_real_utility"))
                if math.isfinite(previous) and math.isfinite(utility) and utility < previous - 0.05:
                    drops += 1
                previous = utility
        utilities = [num(r.get("selected_real_utility")) for r in policy_rows]
        out.append(
            {
                "policy": policy,
                "rows": len(policy_rows),
                "max_energy_evaluations": max([num(r.get("energy_evaluations"), 0.0) for r in policy_rows]),
                "best_selected_real_utility": max(utilities),
                "worst_selected_real_utility": min(utilities),
                "mean_utility_per_energy_eval": mean([num(r.get("utility_per_energy_eval")) for r in policy_rows]),
                "utility_drop_steps": drops,
            }
        )
    return out


def claim_gate_rows(summary: dict[str, Any], compute_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compute_drops = sum(int(r["utility_drop_steps"]) for r in compute_rows)
    gates = [
        (
            "claim_ledger_clean",
            summary["supported_claims"] >= 16 and summary["partial_claims"] == 0 and summary["unsupported_claims"] == 0,
            "supported>=16, partial=0, unsupported=0",
            f"{summary['supported_claims']}/{summary['partial_claims']}/{summary['unsupported_claims']}",
            "Allows promoted claims to stay in the main text.",
        ),
        (
            "exact_law_validated",
            summary["exact_law_max_abs_error"] <= 0.02,
            "max exact-law error <= 0.02",
            fmt(summary["exact_law_max_abs_error"], 4),
            "Allows finite-law accounting to be used as measurement.",
        ),
        (
            "low_energy_tail_failure_present",
            summary["raw_high_n_utility"] < -1.0 and summary["raw_high_n_invalid"] >= 0.99,
            "raw high-N utility < -1 and invalid >= .99",
            f"{fmt(summary['raw_high_n_utility'])}, {fmt(summary['raw_high_n_invalid'])}",
            "Allows the EBM-specific failure mechanism claim.",
        ),
        (
            "repair_metric_gate",
            summary["main_repair_mean_recovery"] >= 0.95
            and summary["main_repair_worst_recovery"] >= 0.95
            and summary["main_repair_utility_degrading_rows"] == 0,
            "mean/worst recovery >= .95 and no utility degradation",
            f"{fmt(summary['main_repair_mean_recovery'])}/{fmt(summary['main_repair_worst_recovery'])}",
            "Allows near-complete repair only under the recovery-ratio metric.",
        ),
        (
            "real_benchmark_ladder_present",
            summary["metaworld_tasks"] >= 4 and summary["closed_loop_tasks"] >= 4,
            ">=4 Meta-World tasks and closed-loop tasks",
            f"{summary['metaworld_tasks']}/{summary['closed_loop_tasks']}",
            "Allows simulator benchmark language, not robot language.",
        ),
        (
            "closed_loop_boundary_enforced",
            summary["state_heuristic_success"] >= 0.99
            and summary["learned_demo_success"] < 0.5
            and summary["learned_bc_success"] < 0.5
            and not summary["learned_policy_threshold_met"],
            "state heuristic succeeds; learned proposal threshold fails",
            f"{fmt(summary['state_heuristic_success'])}/{fmt(summary['learned_demo_success'])}/{fmt(summary['learned_bc_success'])}",
            "Blocks autonomous learned-policy Meta-World claims.",
        ),
        (
            "optional_adapter_boundary",
            summary["optional_supported_suites"] >= 3 and summary["optional_nonzero_success_rows"] >= 1,
            ">=3 optional suites and >=1 nonzero scripted success row",
            f"{summary['optional_supported_suites']}/{summary['optional_nonzero_success_rows']}",
            "Allows finite adapter-execution evidence only.",
        ),
        (
            "reliability_and_compute_stress_present",
            summary["reliability_rows"] >= 1600 and summary["optimization_rows"] >= 150 and compute_drops >= 5,
            "reliability>=1600, compute>=150, compute drops>=5",
            f"{summary['reliability_rows']}/{summary['optimization_rows']}/{compute_drops}",
            "Forces negative controls and compute-regression rows into the audit.",
        ),
        (
            "non_claim_boundaries_retained",
            summary["non_claim_boundaries"] >= 3,
            ">=3 unsupported non-claim boundaries",
            str(summary["non_claim_boundaries"]),
            "Keeps real-robot, broad-success, and learned-autonomy endpoints out of supported claims.",
        ),
    ]
    return [
        {
            "gate": gate,
            "status": "PASS" if passed else "FAIL",
            "threshold": threshold,
            "observed": observed,
            "claim_effect": claim_effect,
        }
        for gate, passed, threshold, observed, claim_effect in gates
    ]


def build_outputs() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    clean_stale_version_outputs()

    controlled = read_csv("results/controlled_energy_tail/summary.csv")
    repair_effectiveness = read_csv("results/repair_effectiveness/summary.csv")
    closed_loop = read_csv("results/benchmarks/metaworld/closed_loop_ablation_summary.csv")
    optional = read_csv("results/optional_benchmarks/summary.csv")
    reliability = read_csv("results/reliability/summary.csv")
    optimization = read_csv("results/optimization_budget/summary.csv")
    metaworld_tasks = read_csv("results/benchmarks/metaworld/task_table.csv")

    claims = read_json("results/claims_status.json")
    controlled_json = read_json("results/controlled_energy_tail/summary.json")
    exact_json = read_json("results/exact_law/summary.json")
    repair_json = read_json("results/repair_effectiveness/summary.json")
    torch_json = read_json("results/learned_torch_ibc/summary.json")
    optional_json = read_json("results/optional_benchmarks/summary.json")
    closed_loop_json = read_json("results/benchmarks/metaworld/closed_loop_ablation_summary.json")
    figures_json = read_json("results/figures/figures.json")

    claim_counts = Counter(c.get("status", "") for c in claims.get("claims", []))
    non_claim_counts = Counter(c.get("status", "") for c in claims.get("non_claim_boundaries", []))
    claim_inventory = [
        {"status": "SUPPORTED", "count": claim_counts.get("SUPPORTED", 0), "kind": "promoted_claim"},
        {"status": "PARTIAL", "count": claim_counts.get("PARTIAL", 0), "kind": "promoted_claim"},
        {"status": "UNSUPPORTED", "count": claim_counts.get("UNSUPPORTED", 0), "kind": "promoted_claim"},
        {
            "status": "UNSUPPORTED_NON_CLAIM",
            "count": non_claim_counts.get("UNSUPPORTED_NON_CLAIM", 0),
            "kind": "boundary",
        },
    ]
    write_csv(OUT / "v4_claim_inventory.csv", claim_inventory, ["kind", "status", "count"])

    tail_matrix = aggregate_tail_rows(controlled)
    write_csv(
        OUT / "v4_energy_tail_matrix.csv",
        tail_matrix,
        [
            "model",
            "rows",
            "selected_energy",
            "selected_real_utility",
            "low_energy_tail_real_utility",
            "invalid_action_rate",
            "support_distance",
            "high_n_regret",
            "tail_rank_correlation",
            "dominant_gate",
        ],
    )

    write_csv(
        OUT / "v4_repair_recovery.csv",
        repair_effectiveness,
        [
            "model",
            "N",
            "num_rows",
            "mean_repair_recovery_ratio",
            "worst_case_recovery_ratio",
            "num_repair_failures",
            "num_utility_degrading_rows",
            "recovery_ge_95_rate",
            "tail_rank_improved_rate",
            "support_distance_reduced_rate",
        ],
    )

    dependency_rows = aggregate_closed_loop(closed_loop)
    write_csv(
        OUT / "v4_closed_loop_dependency.csv",
        dependency_rows,
        [
            "policy",
            "tasks",
            "mean_success_rate",
            "mean_total_reward",
            "mean_fallback_rate",
            "runtime_scripted_expert_dependency",
            "uses_learned_demo_proposal",
            "uses_learned_bc_proposal",
            "uses_state_heuristic_proposal",
        ],
    )

    write_csv(
        OUT / "v4_optional_rollout_boundaries.csv",
        optional,
        [
            "suite",
            "task",
            "num_seeds",
            "task_status",
            "mean_total_reward",
            "mean_success_rate",
            "success_nonzero",
            "claim_boundary",
        ],
    )

    benchmark_claim = metaworld_claim_rows(metaworld_tasks)
    write_csv(
        OUT / "v4_metaworld_claim_matrix.csv",
        benchmark_claim,
        [
            "task",
            "model",
            "training_source",
            "claim_role",
            "task_status",
            "mean_selected_reward",
            "selected_reward_ci_low",
            "selected_reward_ci_high",
            "mean_selected_success",
            "mean_high_n_regret",
            "mean_support_distance",
            "max_exact_law_abs_error",
            "num_seeds",
        ],
    )

    dependency_stress = dependency_stress_rows(closed_loop)
    write_csv(
        OUT / "v4_policy_dependency_stress.csv",
        dependency_stress,
        [
            "policy",
            "claim_role",
            "tasks",
            "mean_success_rate",
            "min_task_success_rate",
            "mean_total_reward",
            "mean_fallback_rate",
            "runtime_scripted_expert_dependency",
            "uses_learned_demo_proposal",
            "uses_learned_bc_proposal",
            "uses_state_heuristic_proposal",
        ],
    )

    reliability_gaps = reliability_gap_rows(reliability)
    write_csv(
        OUT / "v4_reliability_tail_gaps.csv",
        reliability_gaps,
        [
            "benchmark",
            "task",
            "model",
            "extreme_rows",
            "other_rows",
            "extreme_tail_utility",
            "other_tail_utility",
            "extreme_minus_other_utility",
            "extreme_invalid_rate",
            "extreme_support_distance",
            "extreme_success_rate",
        ],
    )

    compute_stress = compute_stress_rows(optimization)
    write_csv(
        OUT / "v4_compute_frontier_stress.csv",
        compute_stress,
        [
            "policy",
            "rows",
            "max_energy_evaluations",
            "best_selected_real_utility",
            "worst_selected_real_utility",
            "mean_utility_per_energy_eval",
            "utility_drop_steps",
        ],
    )

    inventory = artifact_inventory()
    inv_summary: dict[tuple[str, str], dict[str, Any]] = {}
    for row in inventory:
        key = (str(row["root"]), str(row["suffix"]))
        if key not in inv_summary:
            inv_summary[key] = {"root": key[0], "suffix": key[1], "files": 0, "bytes": 0}
        inv_summary[key]["files"] += 1
        inv_summary[key]["bytes"] += int(row["bytes"])
    inventory_summary = sorted(inv_summary.values(), key=lambda r: (r["root"], r["suffix"]))
    write_csv(OUT / "v4_artifact_inventory.csv", inventory_summary, ["root", "suffix", "files", "bytes"])

    strongest = controlled_json["strongest_low_energy_tail_miscalibration"]
    main_repair = next(
        row for row in repair_effectiveness if row.get("model") == repair_json.get("main_repair_stack")
    )
    torch = torch_json["strongest_torch_ibc_artifact"]
    exact_error = float(exact_json["prediction_error"]["max_abs_error"])
    state_policy = next(row for row in dependency_rows if row["policy"] == "state_heuristic_proposal_no_gate")
    learned_demo = next(row for row in dependency_rows if row["policy"] == "learned_demo_proposal_no_gate")
    learned_bc = next(row for row in dependency_rows if row["policy"] == "learned_bc_proposal_no_gate")
    optional_nonzero = [row for row in optional if row.get("success_nonzero", "").lower() == "true"]
    reliability_extreme = [row for row in reliability if row.get("is_extreme_low_energy_tail", "").lower() == "true"]

    figure_paths = figures_json.get("figures", [])
    summary = {
        "artifact_files": len(inventory),
        "closed_loop_policies": len(dependency_rows),
        "closed_loop_rows": len(closed_loop),
        "closed_loop_tasks": len({r.get("task", "") for r in closed_loop}),
        "controlled_high_n_rows": sum(1 for r in controlled if int(num(r.get("N"), -1)) == 512),
        "controlled_models": len({r.get("model", "") for r in controlled}),
        "controlled_pool_n": int(controlled_json.get("pool_n", 0)),
        "controlled_seeds": len(controlled_json.get("seeds", [])),
        "exact_law_max_abs_error": exact_error,
        "figure_files": len(figure_paths),
        "learned_policy_threshold_met": bool_from_json(closed_loop_json, "learned_policy_low_dependency_success_threshold_met"),
        "main_repair_mean_recovery": float(main_repair["mean_repair_recovery_ratio"]),
        "main_repair_worst_recovery": float(main_repair["worst_case_recovery_ratio"]),
        "main_repair_failures": int(float(main_repair["num_repair_failures"])),
        "main_repair_utility_degrading_rows": int(float(main_repair["num_utility_degrading_rows"])),
        "metaworld_tasks": len({r.get("task", "") for r in metaworld_tasks}),
        "non_claim_boundaries": len(claims.get("non_claim_boundaries", [])),
        "not_clone_checks_passed": claims.get("summary", {}).get("not_clone_checks_passed", 0),
        "not_clone_checks_total": claims.get("summary", {}).get("not_clone_checks_total", 0),
        "optional_nonzero_success_rows": len(optional_nonzero),
        "optional_supported_suites": len(set(optional_json.get("supported_non_metaworld_suites", []))),
        "partial_claims": claim_counts.get("PARTIAL", 0),
        "promoted_claims": len(claims.get("claims", [])),
        "reliability_rows": len(reliability),
        "reliability_extreme_tail_rows": len(reliability_extreme),
        "state_heuristic_success": float(state_policy["mean_success_rate"]),
        "supported_claims": claim_counts.get("SUPPORTED", 0),
        "tail_matrix_rows": len(tail_matrix),
        "torch_loss_decrease": float(torch_json["model_metadata"]["loss_decrease"]),
        "torch_raw_high_n_utility": float(torch["selected_real_utility"]),
        "unsupported_claims": claim_counts.get("UNSUPPORTED", 0),
        "benchmark_claim_rows": len(benchmark_claim),
        "compute_stress_rows": len(compute_stress),
        "compute_utility_drop_steps": sum(int(r["utility_drop_steps"]) for r in compute_stress),
        "dependency_stress_rows": len(dependency_stress),
        "metaworld_ci_rows": sum(
            1
            for r in benchmark_claim
            if math.isfinite(float(r["selected_reward_ci_low"]))
            and math.isfinite(float(r["selected_reward_ci_high"]))
        ),
        "reliability_gap_rows": len(reliability_gaps),
        "v4_figure_files": 9,
        "raw_high_n_energy": float(strongest["selected_energy"]),
        "raw_high_n_utility": float(strongest["selected_real_utility"]),
        "raw_high_n_regret": float(strongest["high_n_regret"]),
        "raw_high_n_invalid": float(strongest["invalid_action_rate"]),
        "raw_high_n_support_distance": float(strongest["support_distance"]),
        "raw_tail_rank_correlation": float(strongest["tail_rank_correlation"]),
        "learned_demo_success": float(learned_demo["mean_success_rate"]),
        "learned_bc_success": float(learned_bc["mean_success_rate"]),
        "optimization_rows": len(optimization),
    }
    gate_rows = claim_gate_rows(summary, compute_stress)
    summary["claim_gate_rows"] = len(gate_rows)
    summary["claim_gates_passed"] = sum(1 for r in gate_rows if r["status"] == "PASS")
    write_csv(OUT / "v4_claim_gates.csv", gate_rows, ["gate", "status", "threshold", "observed", "claim_effect"])
    write_json(OUT / "summary.json", summary)

    # Figure 1: high-N tail matrix.
    sorted_tail = sorted(tail_matrix, key=lambda r: float(r["selected_real_utility"]))
    labels = [r["model"].replace("_", "\n") for r in sorted_tail]
    utilities = [float(r["selected_real_utility"]) for r in sorted_tail]
    colors = ["#b2182b" if v < 0 else "#2166ac" for v in utilities]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(labels, utilities, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("selected real utility at N=512")
    ax.set_title("Low-energy tail utility by EBM scorer")
    ax.tick_params(axis="x", labelrotation=45)
    savefig(fig, FIG_OUT / "v4_energy_tail_failure_matrix.pdf")

    # Figure 2: repair recovery stress.
    repair_sorted = sorted(repair_effectiveness, key=lambda r: float(r["mean_repair_recovery_ratio"]))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.barh(
        [r["model"].replace("_", " ") for r in repair_sorted],
        [float(r["mean_repair_recovery_ratio"]) for r in repair_sorted],
        color="#1b9e77",
    )
    ax.axvline(0.95, color="#d95f02", linestyle="--", linewidth=1.3)
    ax.set_xlabel("mean repair recovery ratio")
    ax.set_title("Repair stack stress test at N=512")
    savefig(fig, FIG_OUT / "v4_repair_recovery_stress.pdf")

    # Figure 3: closed-loop dependency audit.
    dep_sorted = sorted(dependency_rows, key=lambda r: float(r["mean_success_rate"]))
    dep_colors = [
        "#7570b3"
        if r["uses_state_heuristic_proposal"]
        else "#d95f02"
        if r["runtime_scripted_expert_dependency"]
        else "#1b9e77"
        if r["uses_learned_demo_proposal"] or r["uses_learned_bc_proposal"]
        else "#666666"
        for r in dep_sorted
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.barh([r["policy"].replace("_", " ") for r in dep_sorted], [float(r["mean_success_rate"]) for r in dep_sorted], color=dep_colors)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("mean closed-loop success")
    ax.set_title("Dependency audit: policy family versus success")
    savefig(fig, FIG_OUT / "v4_closed_loop_dependency_audit.pdf")

    # Figure 4: claim and boundary inventory.
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.bar([r["status"] for r in claim_inventory], [int(r["count"]) for r in claim_inventory], color=["#1b9e77", "#e6ab02", "#d95f02", "#7570b3"])
    ax.set_ylabel("count")
    ax.set_title("Promoted claims and explicit non-claim boundaries")
    ax.tick_params(axis="x", labelrotation=18)
    savefig(fig, FIG_OUT / "v4_claim_boundary_inventory.pdf")

    # Figure 5: optional rollout boundaries.
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    opt_labels = [f"{r['suite']}\n{r['task']}" for r in optional]
    rewards = [num(r.get("mean_total_reward"), 0.0) for r in optional]
    opt_colors = ["#2166ac" if r.get("success_nonzero", "").lower() == "true" else "#bdbdbd" for r in optional]
    ax.bar(opt_labels, rewards, color=opt_colors)
    ax.set_ylabel("mean total reward")
    ax.set_title("Optional adapters: finite rollout evidence only")
    ax.tick_params(axis="x", labelrotation=15)
    savefig(fig, FIG_OUT / "v4_optional_rollout_boundary.pdf")

    # Figure 6: artifact surface.
    by_root: dict[str, int] = defaultdict(int)
    for row in inventory:
        by_root[str(row["root"])] += 1
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    roots = sorted(by_root)
    ax.bar(roots, [by_root[root] for root in roots], color="#4daf4a")
    ax.set_ylabel("tracked audit files")
    ax.set_title("V4 artifact surface outside generated cache")
    savefig(fig, FIG_OUT / "v4_artifact_surface.pdf")

    # Figure 7: selected-action benchmark confidence rows.
    ci_models = {"expert_ibc", "expert_calibrated", "high_reward_ablation"}
    ci_rows = [r for r in benchmark_claim if r["model"] in ci_models]
    ci_rows = sorted(ci_rows, key=lambda r: (str(r["task"]), str(r["model"])))
    labels = [f"{r['task']}\n{r['model'].replace('_', ' ')}" for r in ci_rows]
    x = np.array([float(r["mean_selected_reward"]) for r in ci_rows])
    lows = np.array([float(r["selected_reward_ci_low"]) for r in ci_rows])
    highs = np.array([float(r["selected_reward_ci_high"]) for r in ci_rows])
    xerr = np.vstack([np.maximum(0.0, x - lows), np.maximum(0.0, highs - x)])
    colors = [
        "#2166ac" if r["model"] == "expert_ibc" else "#1b9e77" if r["model"] == "expert_calibrated" else "#bdbdbd"
        for r in ci_rows
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    y = np.arange(len(ci_rows))
    ax.barh(y, x, xerr=xerr, color=colors, alpha=0.9, ecolor="black", capsize=2)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("selected reward with five-seed CI")
    ax.set_title("Meta-World selected-action baselines and controls")
    savefig(fig, FIG_OUT / "v4_metaworld_selected_action_ci.pdf")

    # Figure 8: policy dependency stress rows.
    dep_claim_sorted = sorted(dependency_stress, key=lambda r: float(r["mean_success_rate"]))
    dep_role_colors = {
        "supported_low_dependency_state_heuristic": "#1b9e77",
        "learned_policy_boundary": "#d95f02",
        "expert_dependency_boundary": "#7570b3",
        "negative_control": "#969696",
        "stress_baseline": "#666666",
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.barh(
        [r["policy"].replace("_", " ") for r in dep_claim_sorted],
        [float(r["mean_success_rate"]) for r in dep_claim_sorted],
        color=[dep_role_colors.get(str(r["claim_role"]), "#666666") for r in dep_claim_sorted],
    )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("mean closed-loop success across tasks")
    ax.set_title("Claim gate: dependency and learned-policy boundaries")
    savefig(fig, FIG_OUT / "v4_dependency_claim_gate.pdf")

    # Figure 9: reliability gaps and compute regressions.
    gap_sorted = sorted(reliability_gaps, key=lambda r: float(r["extreme_minus_other_utility"]))[:8]
    compute_sorted = sorted(compute_stress, key=lambda r: int(r["utility_drop_steps"]), reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].barh(
        [f"{r['task']}\n{r['model']}".replace("_", " ") for r in gap_sorted],
        [float(r["extreme_minus_other_utility"]) for r in gap_sorted],
        color="#b2182b",
    )
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Extreme low-energy tail gap")
    axes[0].set_xlabel("extreme minus other utility")
    axes[1].bar(
        [r["policy"].replace("_", "\n") for r in compute_sorted],
        [int(r["utility_drop_steps"]) for r in compute_sorted],
        color="#d95f02",
    )
    axes[1].set_title("Compute steps with utility drops")
    axes[1].set_ylabel("drop count")
    axes[1].tick_params(axis="x", labelrotation=0, labelsize=8)
    savefig(fig, FIG_OUT / "v4_reliability_compute_stress.pdf")

    copy_figures()

    macros = [
        macro_line("VFourEBMSupportedClaims", summary["supported_claims"], 0),
        macro_line("VFourEBMPartialClaims", summary["partial_claims"], 0),
        macro_line("VFourEBMUnsupportedClaims", summary["unsupported_claims"], 0),
        macro_line("VFourEBMNonClaimBoundaries", summary["non_claim_boundaries"], 0),
        macro_line("VFourEBMNotClonePassed", summary["not_clone_checks_passed"], 0),
        macro_line("VFourEBMNotCloneTotal", summary["not_clone_checks_total"], 0),
        macro_line("VFourEBMExactMaxError", summary["exact_law_max_abs_error"], 4),
        macro_line("VFourEBMControlledSeeds", summary["controlled_seeds"], 0),
        macro_line("VFourEBMControlledModels", summary["controlled_models"], 0),
        macro_line("VFourEBMControlledPool", summary["controlled_pool_n"], 0),
        macro_line("VFourEBMHighNRows", summary["controlled_high_n_rows"], 0),
        macro_line("VFourEBMRawHighNEnergy", summary["raw_high_n_energy"], 3),
        macro_line("VFourEBMRawHighNUtility", summary["raw_high_n_utility"], 3),
        macro_line("VFourEBMRawHighNRegret", summary["raw_high_n_regret"], 3),
        macro_line("VFourEBMRawHighNInvalid", summary["raw_high_n_invalid"], 3),
        macro_line("VFourEBMRawSupportDistance", summary["raw_high_n_support_distance"], 3),
        macro_line("VFourEBMRawTailRankCorr", summary["raw_tail_rank_correlation"], 3),
        macro_line("VFourEBMRepairMeanRecovery", summary["main_repair_mean_recovery"], 3),
        macro_line("VFourEBMRepairWorstRecovery", summary["main_repair_worst_recovery"], 3),
        macro_line("VFourEBMRepairFailures", summary["main_repair_failures"], 0),
        macro_line("VFourEBMRepairDegrades", summary["main_repair_utility_degrading_rows"], 0),
        macro_line("VFourEBMTorchLossDecrease", summary["torch_loss_decrease"], 3),
        macro_line("VFourEBMTorchRawUtility", summary["torch_raw_high_n_utility"], 3),
        macro_line("VFourEBMMetaworldTasks", summary["metaworld_tasks"], 0),
        macro_line("VFourEBMClosedLoopRows", summary["closed_loop_rows"], 0),
        macro_line("VFourEBMClosedLoopPolicies", summary["closed_loop_policies"], 0),
        macro_line("VFourEBMClosedLoopTasks", summary["closed_loop_tasks"], 0),
        macro_line("VFourEBMStateHeuristicSuccess", summary["state_heuristic_success"], 3),
        macro_line("VFourEBMLearnedDemoSuccess", summary["learned_demo_success"], 3),
        macro_line("VFourEBMLearnedBCSuccess", summary["learned_bc_success"], 3),
        macro_line("VFourEBMOptionalSuites", summary["optional_supported_suites"], 0),
        macro_line("VFourEBMOptionalNonzero", summary["optional_nonzero_success_rows"], 0),
        macro_line("VFourEBMBenchmarkClaimRows", summary["benchmark_claim_rows"], 0),
        macro_line("VFourEBMClaimGatesPassed", summary["claim_gates_passed"], 0),
        macro_line("VFourEBMClaimGatesTotal", summary["claim_gate_rows"], 0),
        macro_line("VFourEBMComputeDropSteps", summary["compute_utility_drop_steps"], 0),
        macro_line("VFourEBMDependencyStressRows", summary["dependency_stress_rows"], 0),
        macro_line("VFourEBMMetaworldCIRows", summary["metaworld_ci_rows"], 0),
        macro_line("VFourEBMReliabilityGapRows", summary["reliability_gap_rows"], 0),
        macro_line("VFourEBMReliabilityRows", summary["reliability_rows"], 0),
        macro_line("VFourEBMReliabilityExtremeRows", summary["reliability_extreme_tail_rows"], 0),
        macro_line("VFourEBMOptimizationRows", summary["optimization_rows"], 0),
        macro_line("VFourEBMArtifactFiles", summary["artifact_files"], 0),
        macro_line("VFourEBMFigureFiles", summary["figure_files"], 0),
        macro_line("VFourEBMVFourFigures", summary["v4_figure_files"], 0),
    ]
    MACROS.write_text("".join(macros), encoding="utf-8")
    return summary


def main() -> int:
    summary = build_outputs()
    print(f"V4 EBM evidence complete: {OUT}")
    print(
        "claims={supported_claims} boundaries={non_claim_boundaries} "
        "artifacts={artifact_files} figures={v4_figure_files}".format(**summary)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
