"""Regenerate paper-critical figures from CSV/JSON artifacts only."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _series(rows: list[dict[str, str]], model: str, metric: str) -> tuple[list[int], list[float]]:
    filt = [r for r in rows if r.get("model") == model]
    filt.sort(key=lambda r: int(float(r["N"])))
    return [int(float(r["N"])) for r in filt], [float(r[metric]) for r in filt]


def _save(fig, path: str | Path) -> str:
    p = Path(path)
    ensure_dir(p.parent)
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def plot_figure1_energy_utility_drop(controlled_csv: str | Path, out_dir: str | Path) -> str:
    rows = _read_rows(controlled_csv)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for model in ["raw_miscalibrated", "good", "oracle"]:
        n, energy = _series(rows, model, "selected_energy")
        _, utility = _series(rows, model, "selected_real_utility")
        if n:
            axes[0].plot(n, energy, marker="o", label=model)
            axes[1].plot(n, utility, marker="o", label=model)
    axes[0].set_title("Energy improves with N")
    axes[0].set_ylabel("Selected energy")
    axes[1].set_title("Utility can collapse")
    axes[1].set_ylabel("Selected real utility")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N candidates")
        ax.grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    return _save(fig, Path(out_dir) / "figure1_energy_utility_drop.png")


def plot_figure2_repair_comparison(repair_csv: str | Path, out_dir: str | Path) -> str:
    rows = _read_rows(repair_csv)
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    for model in ["raw_ebm", "support_penalized", "calibrated", "value_shaped", "oracle"]:
        n, y = _series(rows, model, "selected_real_utility")
        if n:
            ax.plot(n, y, marker="o", label=model)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N candidates")
    ax.set_ylabel("Selected real utility")
    ax.set_title("Repair hierarchy")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, Path(out_dir) / "figure2_repair_comparison.png")


def plot_figure3_reliability(reliability_csv: str | Path, out_dir: str | Path) -> str:
    rows = [r for r in _read_rows(reliability_csv) if r.get("model") == "expert_ibc"]
    tasks = sorted({r["task"] for r in rows}) or ["no-data"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(max(4.6 * len(tasks), 4.6), 3.8), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        task_rows = [r for r in rows if r["task"] == task]
        task_rows.sort(key=lambda r: float(r["energy_quantile_low"]))
        if task_rows:
            x = [0.5 * (float(r["energy_quantile_low"]) + float(r["energy_quantile_high"])) for r in task_rows]
            y = [float(r["mean_real_utility"]) for r in task_rows]
            ax.plot(x, y, marker="o", linewidth=1.5)
            tail = [r for r in task_rows if r["is_extreme_low_energy_tail"].lower() == "true"]
            if tail:
                tx = [0.5 * (float(r["energy_quantile_low"]) + float(r["energy_quantile_high"])) for r in tail]
                ty = [float(r["mean_real_utility"]) for r in tail]
                ax.scatter(tx, ty, color="crimson", s=55, label="lowest 5%")
        ax.set_title(task)
        ax.set_xlabel("Energy quantile, low to high")
        ax.grid(True, alpha=0.25)
    axes[0][0].set_ylabel("Mean real utility / reward")
    axes[0][-1].legend(frameon=False, fontsize=8)
    return _save(fig, Path(out_dir) / "figure3_tail_reliability.png")


def plot_figure4_tail_diagnostics(controlled_csv: str | Path, out_dir: str | Path) -> str:
    rows = [r for r in _read_rows(controlled_csv) if r.get("model") == "raw_miscalibrated"]
    rows.sort(key=lambda r: int(float(r["N"])))
    n = [int(float(r["N"])) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].plot(n, [float(r["low_energy_tail_real_utility"]) for r in rows], marker="o")
    axes[1].plot(n, [float(r["tail_rank_correlation"]) for r in rows], marker="o")
    axes[0].set_title("Low-energy tail utility")
    axes[1].set_title("Tail rank correlation")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N candidates")
        ax.grid(True, alpha=0.25)
    return _save(fig, Path(out_dir) / "figure4_tail_rank_and_tail_utility.png")


def plot_figure5_compute_frontier(budget_csv: str | Path, out_dir: str | Path) -> str:
    rows = _read_rows(budget_csv)
    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    for policy in sorted({r["policy"] for r in rows}):
        filt = [r for r in rows if r["policy"] == policy]
        filt.sort(key=lambda r: float(r["energy_evaluations"]))
        ax.plot(
            [float(r["energy_evaluations"]) for r in filt],
            [float(r["selected_real_utility"]) for r in filt],
            marker="o",
            label=policy,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Energy evaluations / latency proxy")
    ax.set_ylabel("Selected real utility")
    ax.set_title("Compute-utility frontier")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, Path(out_dir) / "figure5_compute_utility_frontier.png")


def plot_figure6_benchmark_table(task_table_csv: str | Path, out_dir: str | Path) -> str:
    rows = _read_rows(task_table_csv)
    tasks = sorted({r["task"] for r in rows})
    display_rows: list[list[str]] = []
    for task in tasks:
        by_model = {r["model"]: r for r in rows if r["task"] == task}
        expert = by_model.get("expert_ibc", {})
        calibrated = by_model.get("expert_calibrated", {})
        oracle = by_model.get("oracle", {})
        display_rows.append(
            [
                task,
                str(expert.get("task_status", "")),
                f"{float(expert.get('mean_selected_reward', 0.0)):.3f}",
                f"{float(calibrated.get('mean_selected_reward', 0.0)):.3f}",
                f"{float(oracle.get('mean_selected_reward', 0.0)):.3f}",
                f"{float(expert.get('mean_selected_success', 0.0)):.3f}",
            ]
        )
    fig, ax = plt.subplots(figsize=(9.0, 2.8 + 0.35 * max(len(display_rows), 1)))
    ax.axis("off")
    table = ax.table(
        cellText=display_rows,
        colLabels=["Task", "Claim", "Expert EBM", "Calibrated", "Oracle", "Success"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)
    ax.set_title("Multi-task benchmark ladder")
    return _save(fig, Path(out_dir) / "figure6_multitask_benchmark_table.png")


def plot_repair_ablations(ablation_csv: str | Path, out_dir: str | Path) -> str:
    rows = _read_rows(ablation_csv)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    for ablation, ax, xlabel in [
        ("pilot_labels", axes[0], "Pilot labels"),
        ("support_weight", axes[1], "Support penalty weight"),
    ]:
        filt = [r for r in rows if r["ablation"] == ablation]
        filt.sort(key=lambda r: float(r["value"]))
        ax.plot([float(r["value"]) for r in filt], [float(r["selected_real_utility"]) for r in filt], marker="o")
        ax.fill_between(
            [float(r["value"]) for r in filt],
            [float(r["selected_real_utility_ci_low"]) for r in filt],
            [float(r["selected_real_utility_ci_high"]) for r in filt],
            alpha=0.18,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("High-N selected utility")
        ax.grid(True, alpha=0.25)
    axes[0].set_xscale("log", base=2)
    axes[0].set_title("Pilot-label calibration")
    axes[1].set_title("Support repair")
    return _save(fig, Path(out_dir) / "figure7_repair_ablations.png")


def plot_figure8_repair_recovery_ratio(effectiveness_csv: str | Path, out_dir: str | Path) -> str:
    rows = _read_rows(effectiveness_csv)
    preferred = [
        "support_penalized_conservative_gate",
        "calibrated",
        "calibrated_support_penalized_conservative_gate",
        "value_shaped",
        "oracle",
    ]
    by_model = {r["model"]: r for r in rows}
    models = [m for m in preferred if m in by_model] or [r["model"] for r in rows]
    values = [float(by_model[m]["mean_repair_recovery_ratio"]) for m in models]
    lows = [float(by_model[m]["repair_recovery_ratio_ci_low"]) for m in models]
    highs = [float(by_model[m]["repair_recovery_ratio_ci_high"]) for m in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.bar(x, values, color="#3a7ca5", alpha=0.86)
    ax.errorbar(
        x,
        values,
        yerr=[np.asarray(values) - np.asarray(lows), np.asarray(highs) - np.asarray(values)],
        fmt="none",
        color="black",
        linewidth=1.0,
        capsize=3,
    )
    ax.axhline(0.95, color="crimson", linestyle="--", linewidth=1.2, label="0.95 target")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylim(0.0, max(1.08, max(values) * 1.08 if values else 1.0))
    ax.set_ylabel("Repair recovery ratio")
    ax.set_title("Metric-bound repair recovery")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, Path(out_dir) / "figure8_repair_recovery_ratio.png")


def plot_figure9_attack_surface_audit(audit_json: str | Path, out_dir: str | Path) -> str:
    with Path(audit_json).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    checks = payload.get("checks", [])
    labels = [str(c.get("check", "")) for c in checks]
    passed = [1.0 if bool(c.get("passed")) else 0.0 for c in checks]
    if not labels:
        labels = ["no audit rows"]
        passed = [0.0]
    fig, ax = plt.subplots(figsize=(8.2, max(3.2, 0.42 * len(labels) + 1.2)))
    y = np.arange(len(labels))
    colors = ["#2f8f5b" if v else "#b43b45" for v in passed]
    ax.barh(y, passed, color=colors, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["fail", "pass"])
    ax.invert_yaxis()
    ax.set_title("Claim attack-surface audit")
    ax.grid(True, axis="x", alpha=0.20)
    return _save(fig, Path(out_dir) / "figure9_attack_surface_audit.png")


def plot_figure10_closed_loop_dependency_audit(ablation_csv: str | Path, out_dir: str | Path) -> str:
    rows = _read_rows(ablation_csv)
    preferred = [
        "scripted_expert",
        "expert_centered_gate",
        "expert_centered_no_gate",
        "learned_demo_proposal_no_gate",
        "learned_demo_proposal_direct",
        "learned_bc_proposal_no_gate",
        "learned_bc_proposal_direct",
        "state_heuristic_proposal_no_gate",
        "state_heuristic_direct",
        "mixed_expert_local_no_gate",
        "local_gaussian_no_gate",
        "random_uniform",
    ]
    policies = [p for p in preferred if any(r.get("policy") == p for r in rows)]
    if not policies:
        policies = sorted({r.get("policy", "") for r in rows}) or ["no-data"]
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    fallback: list[float] = []
    distances: list[float] = []
    for policy in policies:
        group = [r for r in rows if r.get("policy") == policy]
        success = np.asarray([float(r.get("mean_success_rate", 0.0)) for r in group], dtype=float)
        mean = float(np.mean(success)) if len(success) else 0.0
        se = float(np.std(success, ddof=1) / np.sqrt(len(success))) if len(success) > 1 else 0.0
        means.append(mean)
        lows.append(mean - 1.96 * se)
        highs.append(mean + 1.96 * se)
        fallback.append(float(np.mean([float(r.get("mean_gate_fallback_rate", 0.0)) for r in group])) if group else 0.0)
        distances.append(
            float(np.mean([float(r.get("mean_selected_expert_action_distance", 0.0)) for r in group])) if group else 0.0
        )
    x = np.arange(len(policies))
    labels = [p.replace("_", "\n") for p in policies]
    fig, ax = plt.subplots(figsize=(12.2, 5.0))
    ax.bar(x, means, color="#406c9a", alpha=0.84, label="mean success")
    ax.errorbar(
        x,
        means,
        yerr=[np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)],
        fmt="none",
        color="black",
        linewidth=1.0,
        capsize=3,
    )
    ax.plot(x, fallback, color="#b43b45", marker="o", linewidth=1.4, label="fallback rate")
    ax.plot(x, distances, color="#2f8f5b", marker="s", linewidth=1.4, label="expert-action distance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Rate / normalized distance")
    ax.set_title("Closed-loop dependency audit")
    ax.grid(True, axis="y", alpha=0.24)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return _save(fig, Path(out_dir) / "figure10_closed_loop_dependency_audit.png")


def regenerate_all_figures(results_dir: str | Path = "results") -> list[str]:
    root = Path(results_dir)
    out_dir = root / "figures"
    paths = [
        plot_figure1_energy_utility_drop(root / "controlled_energy_tail" / "summary.csv", out_dir),
        plot_figure2_repair_comparison(root / "repair" / "summary.csv", out_dir),
        plot_figure3_reliability(root / "reliability" / "summary.csv", out_dir),
        plot_figure4_tail_diagnostics(root / "controlled_energy_tail" / "summary.csv", out_dir),
        plot_figure5_compute_frontier(root / "optimization_budget" / "summary.csv", out_dir),
        plot_figure6_benchmark_table(root / "benchmarks" / "metaworld" / "task_table.csv", out_dir),
    ]
    if (root / "ablations" / "summary.csv").exists():
        paths.append(plot_repair_ablations(root / "ablations" / "summary.csv", out_dir))
    if (root / "repair_effectiveness" / "summary.csv").exists():
        paths.append(plot_figure8_repair_recovery_ratio(root / "repair_effectiveness" / "summary.csv", out_dir))
    if (root / "claim_stress_audit.json").exists():
        paths.append(plot_figure9_attack_surface_audit(root / "claim_stress_audit.json", out_dir))
    if (root / "benchmarks" / "metaworld" / "closed_loop_ablation_summary.csv").exists():
        paths.append(
            plot_figure10_closed_loop_dependency_audit(
                root / "benchmarks" / "metaworld" / "closed_loop_ablation_summary.csv",
                out_dir,
            )
        )
    summary_path = out_dir / "figures.json"
    ensure_dir(summary_path.parent)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"figures": paths}, f, indent=2)
        f.write("\n")
    return paths
