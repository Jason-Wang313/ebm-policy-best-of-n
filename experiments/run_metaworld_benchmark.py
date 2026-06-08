from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ebm_best_of_n.calibration import apply_energy_calibrator, fit_energy_calibrator
from ebm_best_of_n.diagnostics import aggregate_curve_rows, curve_rows_for_energy
from ebm_best_of_n.energy_models import oracle_energy, support_penalized_energy
from ebm_best_of_n.torch_models import TorchIBCEnergy
from ebm_best_of_n.utils import N_VALUES, write_csv, write_json


def _empty_pool() -> dict[str, np.ndarray]:
    return {
        "observations": np.zeros((0, 1), dtype=float),
        "actions": np.zeros((0, 1), dtype=float),
        "utility": np.zeros(0, dtype=float),
        "success": np.zeros(0, dtype=float),
        "invalid": np.zeros(0, dtype=float),
        "jerk_penalty": np.zeros(0, dtype=float),
        "contact_penalty": np.zeros(0, dtype=float),
        "support_penalty": np.zeros(0, dtype=float),
        "visible_score": np.zeros(0, dtype=float),
        "shortcut_score": np.zeros(0, dtype=float),
        "action_norm": np.zeros(0, dtype=float),
    }


def _collect_reach_pool(seed: int, num_states: int, candidates_per_state: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    import metaworld

    rng = np.random.default_rng(seed)
    mt1 = metaworld.MT1("reach-v3")
    env_cls = mt1.train_classes["reach-v3"]
    env = env_cls()
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    rewards: list[float] = []
    successes: list[float] = []
    obj_to_target: list[float] = []
    action_norms: list[float] = []
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)

    start = time.perf_counter()
    for state_idx in range(num_states):
        task = mt1.train_tasks[(seed + state_idx) % len(mt1.train_tasks)]
        env.set_task(task)
        obs, _info = env.reset(seed=seed + state_idx)
        state = env.get_env_state()
        for _candidate in range(candidates_per_state):
            action = rng.uniform(action_low, action_high).astype(np.float32)
            env.set_env_state(state)
            _next_obs, reward, terminated, truncated, info = env.step(action)
            obs_rows.append(np.asarray(obs, dtype=float))
            action_rows.append(np.asarray(action, dtype=float))
            rewards.append(float(reward))
            successes.append(float(info.get("success", 0.0)))
            obj_to_target.append(float(info.get("obj_to_target", max(0.0, 10.0 - float(reward)))))
            action_norms.append(float(np.linalg.norm(action)))
            if terminated or truncated:
                env.set_env_state(state)
    elapsed = time.perf_counter() - start
    rewards_arr = np.asarray(rewards, dtype=float)
    obj_arr = np.asarray(obj_to_target, dtype=float)
    action_norm_arr = np.asarray(action_norms, dtype=float)
    if rewards_arr.size == 0:
        pool = _empty_pool()
    else:
        reward_cutoff = float(np.quantile(rewards_arr, 0.25))
        obj_scale = float(np.std(obj_arr) + 1e-6)
        pool = {
            "observations": np.vstack(obs_rows).astype(float),
            "actions": np.vstack(action_rows).astype(float),
            "utility": rewards_arr,
            "success": np.asarray(successes, dtype=float),
            "invalid": (rewards_arr <= reward_cutoff).astype(float),
            "jerk_penalty": action_norm_arr * action_norm_arr,
            "contact_penalty": (1.0 - np.asarray(successes, dtype=float)),
            "support_penalty": (obj_arr - float(np.min(obj_arr))) / obj_scale,
            "visible_score": (rewards_arr - float(np.mean(rewards_arr))) / float(np.std(rewards_arr) + 1e-6),
            "shortcut_score": (rewards_arr <= reward_cutoff).astype(float),
            "action_norm": action_norm_arr,
        }
    metadata = {
        "task": "reach-v3",
        "num_states": int(num_states),
        "candidates_per_state": int(candidates_per_state),
        "num_candidates": int(len(rewards_arr)),
        "collection_runtime_seconds": float(elapsed),
        "action_low": action_low.astype(float).tolist(),
        "action_high": action_high.astype(float).tolist(),
        "mean_reward": float(np.mean(rewards_arr)) if rewards_arr.size else float("nan"),
        "max_reward": float(np.max(rewards_arr)) if rewards_arr.size else float("nan"),
    }
    return pool, metadata


def _positives_from_pool(pool: dict[str, np.ndarray], quantile: float = 0.82) -> tuple[np.ndarray, np.ndarray]:
    cutoff = float(np.quantile(pool["utility"], quantile))
    mask = pool["utility"] >= cutoff
    if int(np.sum(mask)) < 8:
        idx = np.argsort(pool["utility"])[-8:]
        mask = np.zeros(len(pool["utility"]), dtype=bool)
        mask[idx] = True
    return pool["observations"][mask], pool["actions"][mask]


def run(smoke: bool = False) -> dict[str, object]:
    out_dir = Path("results") / "benchmarks" / "metaworld"
    seeds = [0] if smoke else [0, 1, 2]
    states = 3 if smoke else 8
    train_candidates = 48 if smoke else 96
    eval_candidates = 64 if smoke else 128
    epochs = 35 if smoke else 90
    mc_trials = 200 if smoke else 700
    seed_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    try:
        for seed in seeds:
            train_pool, train_meta = _collect_reach_pool(seed + 400, states, train_candidates)
            eval_pool, eval_meta = _collect_reach_pool(seed + 800, states, eval_candidates)
            if len(train_pool["utility"]) == 0 or len(eval_pool["utility"]) == 0:
                raise RuntimeError("Meta-World pool collection produced no candidates")
            pos_obs, pos_actions = _positives_from_pool(train_pool, quantile=0.82)
            action_low = np.asarray(train_meta["action_low"], dtype=np.float32)
            action_high = np.asarray(train_meta["action_high"], dtype=np.float32)
            model = TorchIBCEnergy.fit(
                pos_obs,
                pos_actions,
                seed=seed,
                epochs=epochs,
                batch_size=64,
                negatives=32,
                lr=2e-3,
                hidden_dim=128,
                feature_mode="raw",
                action_low=action_low,
                action_high=action_high,
            )
            raw_energy = model.energy(eval_pool["observations"], eval_pool["actions"])
            calibrator = fit_energy_calibrator(eval_pool, raw_energy, pilot_size=min(192, len(raw_energy) // 2), seed=seed)
            energies = {
                "metaworld_raw_torch_ebm": raw_energy,
                "metaworld_calibrated": apply_energy_calibrator(eval_pool, raw_energy, calibrator),
                "metaworld_support_penalized": support_penalized_energy(raw_energy, eval_pool, weight=0.4),
                "metaworld_random": np.zeros_like(raw_energy),
                "metaworld_oracle": oracle_energy(eval_pool),
            }
            for model_name, energy in energies.items():
                rows, _summary = curve_rows_for_energy(eval_pool, energy, N_VALUES, model_name, seed, exact_mc_trials=mc_trials)
                for row in rows:
                    row["benchmark"] = "metaworld"
                    row["task"] = "reach-v3"
                    seed_rows.append(row)
            train_meta["seed"] = seed
            eval_meta["seed"] = seed
            eval_meta["model_metadata"] = model.metadata()
            metadata_rows.append({"seed": seed, "train": train_meta, "eval": eval_meta})
        summary_rows = aggregate_curve_rows(seed_rows)
        status = "SUPPORTED"
        error = ""
    except Exception as exc:  # guarded benchmark path
        summary_rows = []
        status = "UNSUPPORTED"
        error = f"{type(exc).__name__}: {exc}"

    write_csv(out_dir / "seed_level.csv", seed_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    high_n = max(N_VALUES)
    raw_high = [r for r in seed_rows if r.get("model") == "metaworld_raw_torch_ebm" and int(r["N"]) == high_n]
    payload = {
        "experiment": "metaworld_benchmark",
        "benchmark": "metaworld",
        "task": "reach-v3",
        "status": status,
        "error": error,
        "smoke": smoke,
        "n_values": N_VALUES,
        "seeds": seeds,
        "metadata": metadata_rows,
        "strongest_benchmark_artifact": raw_high[0] if raw_high else {},
        "artifacts": {
            "summary_csv": str(out_dir / "summary.csv"),
            "seed_level_csv": str(out_dir / "seed_level.csv"),
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
