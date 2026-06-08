"""Regenerate paper-critical figures from CSV/JSON artifacts only."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir, read_json


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _series(rows: list[dict[str, str]], model: str, metric: str) -> tuple[list[int], list[float]]:
    filt = [r for r in rows if r.get("model") == model]
    filt.sort(key=lambda r: int(r["N"]))
    return [int(r["N"]) for r in filt], [float(r[metric]) for r in filt]


def _save(fig, path: str | Path) -> str:
    p = Path(path)
    ensure_dir(p.parent)
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return str(p)


def plot_figure1(controlled_csv: str | Path, out_dir: str | Path) -> list[str]:
    rows = _read_rows(controlled_csv)
    paths: list[str] = []
    for metric, ylabel, fname in [
        ("selected_energy", "Selected energy", "figure1_low_energy_tail_energy.png"),
        ("selected_real_utility", "Selected real utility", "figure1_low_energy_tail_utility.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        for model in ["raw_miscalibrated", "good", "oracle"]:
            n, y = _series(rows, model, metric)
            if n:
                ax.plot(n, y, marker="o", label=model)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N candidates")
        ax.set_ylabel(ylabel)
        ax.set_title("Low-energy tail miscalibration")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        paths.append(_save(fig, Path(out_dir) / fname))
    return paths


def plot_figure2(repair_csv: str | Path, out_dir: str | Path) -> str:
    rows = _read_rows(repair_csv)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for model in ["raw_ebm", "calibrated", "value_shaped", "support_penalized", "random", "oracle"]:
        n, y = _series(rows, model, "selected_real_utility")
        if n:
            ax.plot(n, y, marker="o", label=model)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N candidates")
    ax.set_ylabel("Selected real utility")
    ax.set_title("Repair comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    return _save(fig, Path(out_dir) / "figure2_repair_comparison.png")


def plot_figure3(controlled_csv: str | Path, out_dir: str | Path) -> str:
    rows = [r for r in _read_rows(controlled_csv) if r.get("model") == "raw_miscalibrated"]
    rows.sort(key=lambda r: int(r["N"]))
    n = [int(r["N"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4))
    for ax, metric, title in [
        (axes[0], "low_energy_tail_real_utility", "Tail utility"),
        (axes[1], "tail_rank_correlation", "Tail rank correlation"),
        (axes[2], "invalid_action_rate", "Invalid selected rate"),
    ]:
        ax.plot(n, [float(r[metric]) for r in rows], marker="o")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    return _save(fig, Path(out_dir) / "figure3_tail_diagnostics.png")


def plot_figure4(budget_csv: str | Path, out_dir: str | Path) -> str:
    rows = _read_rows(budget_csv)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    policies = sorted({r["policy"] for r in rows})
    for policy in policies:
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
    ax.legend(frameon=False)
    return _save(fig, Path(out_dir) / "figure4_compute_utility_frontier.png")


def plot_figure5(exact_csv: str | Path, out_dir: str | Path) -> str:
    rows = _read_rows(exact_csv)
    pred = np.asarray([float(r["predicted_selected_utility"]) for r in rows])
    emp = np.asarray([float(r["empirical_selected_utility"]) for r in rows])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.scatter(pred, emp, s=42)
    lo = float(min(np.min(pred), np.min(emp)))
    hi = float(max(np.max(pred), np.max(emp)))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    for r in rows:
        ax.annotate(r["N"], (float(r["predicted_selected_utility"]), float(r["empirical_selected_utility"])), fontsize=8)
    ax.set_xlabel("Exact finite-law prediction")
    ax.set_ylabel("Monte Carlo empirical estimate")
    ax.set_title("Exact-law validation")
    ax.grid(True, alpha=0.25)
    return _save(fig, Path(out_dir) / "figure5_exact_law_validation.png")


def plot_figure6(out_dir: str | Path) -> str:
    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    ax.axis("off")
    rows = [
        ["WAM", "imagined rollouts", "imagined-vs-real dynamics mismatch"],
        ["JEPA", "latent predictor", "latent-real rank distortion"],
        ["Diffusion", "trajectory generator", "diversity-selection tradeoff"],
        ["EBM", "E(o,a) action energy", "low-energy tail miscalibration"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Project", "Object", "Failure mode"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    ax.set_title("Scientific distinction")
    return _save(fig, Path(out_dir) / "figure6_distinction_table.png")


def plot_figure7(torch_csv: str | Path, out_dir: str | Path) -> str:
    rows = _read_rows(torch_csv)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for model in ["raw_torch_ebm", "calibrated_torch", "value_shaped_torch", "support_penalized_torch", "raw_numpy_ebm", "oracle"]:
        n, y = _series(rows, model, "selected_real_utility")
        if n:
            ax.plot(n, y, marker="o", label=model)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N candidates")
    ax.set_ylabel("Selected real utility")
    ax.set_title("PyTorch neural IBC-style EBM")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, Path(out_dir) / "figure7_torch_ibc_repair.png")


def plot_figure8(metaworld_csv: str | Path, out_dir: str | Path) -> str:
    rows = _read_rows(metaworld_csv)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for metric, ax, title in [
        ("selected_energy", axes[0], "Selected energy"),
        ("selected_real_utility", axes[1], "Selected reward"),
    ]:
        for model in ["metaworld_raw_torch_ebm", "metaworld_calibrated", "metaworld_random", "metaworld_oracle"]:
            n, y = _series(rows, model, metric)
            if n:
                ax.plot(n, y, marker="o", label=model)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    return _save(fig, Path(out_dir) / "figure8_metaworld_benchmark.png")


def plot_figure9(ablation_csv: str | Path, out_dir: str | Path) -> str:
    rows = _read_rows(ablation_csv)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
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
    axes[0].set_title("Calibration label ablation")
    axes[1].set_title("Support repair ablation")
    return _save(fig, Path(out_dir) / "figure9_repair_ablations.png")


def plot_figure10(metaworld_csv: str | Path, out_dir: str | Path) -> str:
    rows = _read_rows(metaworld_csv)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for model in ["metaworld_raw_torch_ebm", "metaworld_calibrated", "metaworld_random", "metaworld_oracle"]:
        filt = [r for r in rows if r["model"] == model]
        filt.sort(key=lambda r: float(r["energy_evaluations"]))
        if filt:
            ax.plot(
                [float(r["energy_evaluations"]) for r in filt],
                [float(r["selected_real_utility"]) for r in filt],
                marker="o",
                label=model,
            )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Energy evaluations / latency proxy")
    ax.set_ylabel("Selected reward")
    ax.set_title("Meta-World compute-utility frontier")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, Path(out_dir) / "figure10_metaworld_compute_frontier.png")


def regenerate_all_figures(results_dir: str | Path = "results") -> list[str]:
    root = Path(results_dir)
    out_dir = root / "figures"
    paths: list[str] = []
    paths.extend(plot_figure1(root / "controlled_energy_tail" / "summary.csv", out_dir))
    paths.append(plot_figure2(root / "repair" / "summary.csv", out_dir))
    paths.append(plot_figure3(root / "controlled_energy_tail" / "summary.csv", out_dir))
    paths.append(plot_figure4(root / "optimization_budget" / "summary.csv", out_dir))
    paths.append(plot_figure5(root / "exact_law" / "validation.csv", out_dir))
    paths.append(plot_figure6(out_dir))
    if (root / "learned_torch_ibc" / "summary.csv").exists():
        paths.append(plot_figure7(root / "learned_torch_ibc" / "summary.csv", out_dir))
    if (root / "benchmarks" / "metaworld" / "summary.csv").exists():
        paths.append(plot_figure8(root / "benchmarks" / "metaworld" / "summary.csv", out_dir))
        paths.append(plot_figure10(root / "benchmarks" / "metaworld" / "summary.csv", out_dir))
    if (root / "ablations" / "summary.csv").exists():
        paths.append(plot_figure9(root / "ablations" / "summary.csv", out_dir))
    summary_path = out_dir / "figures.json"
    import json

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"figures": paths}, f, indent=2)
        f.write("\n")
    return paths
