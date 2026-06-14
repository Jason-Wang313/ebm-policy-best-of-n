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
OUT = ROOT / "results" / "v3_cached_evidence"
FIG_OUT = OUT / "figures"
PAPER_FIG_OUT = ROOT / "results" / "figures" / "v3"
MACROS = ROOT / "paper" / "iclr2026" / "v3_results_macros.tex"
PDF_METADATA = {
    "Creator": "experiments/v3_cached_evidence.py",
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


def build_outputs() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)

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
    write_csv(OUT / "v3_claim_inventory.csv", claim_inventory, ["kind", "status", "count"])

    tail_matrix = aggregate_tail_rows(controlled)
    write_csv(
        OUT / "v3_energy_tail_matrix.csv",
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
        OUT / "v3_repair_recovery.csv",
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
        OUT / "v3_closed_loop_dependency.csv",
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
        OUT / "v3_optional_rollout_boundaries.csv",
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

    inventory = artifact_inventory()
    inv_summary: dict[tuple[str, str], dict[str, Any]] = {}
    for row in inventory:
        key = (str(row["root"]), str(row["suffix"]))
        if key not in inv_summary:
            inv_summary[key] = {"root": key[0], "suffix": key[1], "files": 0, "bytes": 0}
        inv_summary[key]["files"] += 1
        inv_summary[key]["bytes"] += int(row["bytes"])
    inventory_summary = sorted(inv_summary.values(), key=lambda r: (r["root"], r["suffix"]))
    write_csv(OUT / "v3_artifact_inventory.csv", inventory_summary, ["root", "suffix", "files", "bytes"])

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
        "v3_figure_files": 6,
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
    savefig(fig, FIG_OUT / "v3_energy_tail_failure_matrix.pdf")

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
    savefig(fig, FIG_OUT / "v3_repair_recovery_stress.pdf")

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
    savefig(fig, FIG_OUT / "v3_closed_loop_dependency_audit.pdf")

    # Figure 4: claim and boundary inventory.
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.bar([r["status"] for r in claim_inventory], [int(r["count"]) for r in claim_inventory], color=["#1b9e77", "#e6ab02", "#d95f02", "#7570b3"])
    ax.set_ylabel("count")
    ax.set_title("Promoted claims and explicit non-claim boundaries")
    ax.tick_params(axis="x", labelrotation=18)
    savefig(fig, FIG_OUT / "v3_claim_boundary_inventory.pdf")

    # Figure 5: optional rollout boundaries.
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    opt_labels = [f"{r['suite']}\n{r['task']}" for r in optional]
    rewards = [num(r.get("mean_total_reward"), 0.0) for r in optional]
    opt_colors = ["#2166ac" if r.get("success_nonzero", "").lower() == "true" else "#bdbdbd" for r in optional]
    ax.bar(opt_labels, rewards, color=opt_colors)
    ax.set_ylabel("mean total reward")
    ax.set_title("Optional adapters: finite rollout evidence only")
    ax.tick_params(axis="x", labelrotation=15)
    savefig(fig, FIG_OUT / "v3_optional_rollout_boundary.pdf")

    # Figure 6: artifact surface.
    by_root: dict[str, int] = defaultdict(int)
    for row in inventory:
        by_root[str(row["root"])] += 1
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    roots = sorted(by_root)
    ax.bar(roots, [by_root[root] for root in roots], color="#4daf4a")
    ax.set_ylabel("tracked audit files")
    ax.set_title("V3 artifact surface outside generated cache")
    savefig(fig, FIG_OUT / "v3_artifact_surface.pdf")

    copy_figures()

    macros = [
        macro_line("VThreeEBMSupportedClaims", summary["supported_claims"], 0),
        macro_line("VThreeEBMPartialClaims", summary["partial_claims"], 0),
        macro_line("VThreeEBMUnsupportedClaims", summary["unsupported_claims"], 0),
        macro_line("VThreeEBMNonClaimBoundaries", summary["non_claim_boundaries"], 0),
        macro_line("VThreeEBMNotClonePassed", summary["not_clone_checks_passed"], 0),
        macro_line("VThreeEBMNotCloneTotal", summary["not_clone_checks_total"], 0),
        macro_line("VThreeEBMExactMaxError", summary["exact_law_max_abs_error"], 4),
        macro_line("VThreeEBMControlledSeeds", summary["controlled_seeds"], 0),
        macro_line("VThreeEBMControlledModels", summary["controlled_models"], 0),
        macro_line("VThreeEBMControlledPool", summary["controlled_pool_n"], 0),
        macro_line("VThreeEBMHighNRows", summary["controlled_high_n_rows"], 0),
        macro_line("VThreeEBMRawHighNEnergy", summary["raw_high_n_energy"], 3),
        macro_line("VThreeEBMRawHighNUtility", summary["raw_high_n_utility"], 3),
        macro_line("VThreeEBMRawHighNRegret", summary["raw_high_n_regret"], 3),
        macro_line("VThreeEBMRawHighNInvalid", summary["raw_high_n_invalid"], 3),
        macro_line("VThreeEBMRawSupportDistance", summary["raw_high_n_support_distance"], 3),
        macro_line("VThreeEBMRawTailRankCorr", summary["raw_tail_rank_correlation"], 3),
        macro_line("VThreeEBMRepairMeanRecovery", summary["main_repair_mean_recovery"], 3),
        macro_line("VThreeEBMRepairWorstRecovery", summary["main_repair_worst_recovery"], 3),
        macro_line("VThreeEBMRepairFailures", summary["main_repair_failures"], 0),
        macro_line("VThreeEBMRepairDegrades", summary["main_repair_utility_degrading_rows"], 0),
        macro_line("VThreeEBMTorchLossDecrease", summary["torch_loss_decrease"], 3),
        macro_line("VThreeEBMTorchRawUtility", summary["torch_raw_high_n_utility"], 3),
        macro_line("VThreeEBMMetaworldTasks", summary["metaworld_tasks"], 0),
        macro_line("VThreeEBMClosedLoopRows", summary["closed_loop_rows"], 0),
        macro_line("VThreeEBMClosedLoopPolicies", summary["closed_loop_policies"], 0),
        macro_line("VThreeEBMClosedLoopTasks", summary["closed_loop_tasks"], 0),
        macro_line("VThreeEBMStateHeuristicSuccess", summary["state_heuristic_success"], 3),
        macro_line("VThreeEBMLearnedDemoSuccess", summary["learned_demo_success"], 3),
        macro_line("VThreeEBMLearnedBCSuccess", summary["learned_bc_success"], 3),
        macro_line("VThreeEBMOptionalSuites", summary["optional_supported_suites"], 0),
        macro_line("VThreeEBMOptionalNonzero", summary["optional_nonzero_success_rows"], 0),
        macro_line("VThreeEBMReliabilityRows", summary["reliability_rows"], 0),
        macro_line("VThreeEBMReliabilityExtremeRows", summary["reliability_extreme_tail_rows"], 0),
        macro_line("VThreeEBMOptimizationRows", summary["optimization_rows"], 0),
        macro_line("VThreeEBMArtifactFiles", summary["artifact_files"], 0),
        macro_line("VThreeEBMFigureFiles", summary["figure_files"], 0),
        macro_line("VThreeEBMVThreeFigures", summary["v3_figure_files"], 0),
    ]
    MACROS.write_text("".join(macros), encoding="utf-8")
    return summary


def main() -> int:
    summary = build_outputs()
    print(f"v3 EBM evidence complete: {OUT}")
    print(
        "claims={supported_claims} boundaries={non_claim_boundaries} "
        "artifacts={artifact_files} figures={v3_figure_files}".format(**summary)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
