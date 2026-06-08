"""Small filesystem, CSV, and reproducibility helpers."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


N_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: Mapping[str, Any] | list[Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str] | None = None) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    rows_list = list(rows)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows_list:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def set_reproducible_env(seed: int = 0) -> np.random.Generator:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    return np.random.default_rng(seed)


def mean_ci(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("values must be non-empty")
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"mean": mean, "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se}


def assert_file_exists(path: str | Path) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(str(path))


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out
