from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ebm_tail_audit.utils import write_csv, write_json


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value: Any, default: float = 0.0) -> float:
    try:
        arr = _as_numpy(value).astype(float).reshape(-1)
        out = float(np.mean(arr))
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _done(value: Any) -> bool:
    try:
        return bool(np.asarray(_as_numpy(value), dtype=bool).reshape(-1).any())
    except Exception:
        return bool(value)


def _extract_success(info: Any) -> tuple[bool, float]:
    if not isinstance(info, dict):
        return False, 0.0
    for key in ("success", "is_success", "task_success", "success_rate"):
        if key in info:
            return True, _scalar(info[key], default=0.0)
    return False, 0.0


def _artifact_row(
    suite: str,
    task: str,
    seed: int,
    policy: str,
    horizon: int,
    rewards: list[float],
    successes: list[float],
    elapsed: float,
    failure: str = "",
) -> dict[str, object]:
    finite = bool(rewards) and all(np.isfinite(r) for r in rewards)
    success_available = bool(successes)
    success_rate = float(np.mean(successes)) if successes else ""
    return {
        "suite": suite,
        "task": task,
        "seed": seed,
        "policy": policy,
        "horizon": horizon,
        "steps": len(rewards),
        "total_reward": float(np.sum(rewards)) if finite else "",
        "mean_reward": float(np.mean(rewards)) if finite else "",
        "success_available": success_available,
        "success_rate": success_rate,
        "success_nonzero": bool(successes and np.max(successes) > 0.0),
        "runtime_seconds": float(elapsed),
        "task_status": "SUPPORTED" if finite else "UNSUPPORTED",
        "failure": failure,
        "claim_boundary": "Optional rollout artifact only; not a real-robot or broad manipulation-success claim.",
    }


def _failure_row(suite: str, task: str, seed: int, policy: str, horizon: int, exc: BaseException) -> dict[str, object]:
    return _artifact_row(
        suite=suite,
        task=task,
        seed=seed,
        policy=policy,
        horizon=horizon,
        rewards=[],
        successes=[],
        elapsed=0.0,
        failure=f"{type(exc).__name__}: {exc}",
    )


def _run_gymnasium(task: str, seed: int, horizon: int) -> dict[str, object]:
    import gymnasium as gym

    start = time.perf_counter()
    env = gym.make(task)
    try:
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        env.reset(seed=seed)
        rewards: list[float] = []
        successes: list[float] = []
        for _step in range(horizon):
            _obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            rewards.append(_scalar(reward))
            has_success, success = _extract_success(info)
            if has_success:
                successes.append(success)
            if bool(terminated) or bool(truncated):
                break
        return _artifact_row("gymnasium", task, seed, "seeded_random", horizon, rewards, successes, time.perf_counter() - start)
    finally:
        env.close()


def _run_maniskill(task: str, seed: int, horizon: int) -> dict[str, object]:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 - registers ManiSkill tasks

    start = time.perf_counter()
    env = gym.make(task, obs_mode="state", render_mode=None)
    try:
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        env.reset(seed=seed)
        rewards: list[float] = []
        successes: list[float] = []
        for _step in range(horizon):
            _obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            rewards.append(_scalar(reward))
            has_success, success = _extract_success(info)
            if has_success:
                successes.append(success)
            if _done(terminated) or _done(truncated):
                break
        return _artifact_row("mani_skill", task, seed, "seeded_random", horizon, rewards, successes, time.perf_counter() - start)
    finally:
        env.close()


def _run_robosuite(task: str, seed: int, horizon: int) -> dict[str, object]:
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    np.random.seed(seed)
    start = time.perf_counter()
    controller = load_composite_controller_config(controller="BASIC")
    env = suite.make(
        env_name=task,
        robots=["Panda"],
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        horizon=horizon,
    )
    try:
        env.reset()
        rewards: list[float] = []
        successes: list[float] = []
        for _step in range(horizon):
            _obs, reward, done, _info = env.step(np.zeros(env.action_dim, dtype=float))
            rewards.append(_scalar(reward))
            if hasattr(env, "_check_success"):
                successes.append(float(bool(env._check_success())))
            if bool(done):
                break
        return _artifact_row("robosuite", task, seed, "zero_action", horizon, rewards, successes, time.perf_counter() - start)
    finally:
        env.close()


def _run_robosuite_lift_scripted(task: str, seed: int, horizon: int) -> dict[str, object]:
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    if task != "Lift":
        raise ValueError("scripted Lift heuristic only supports robosuite Lift")
    np.random.seed(seed)
    scripted_horizon = max(int(horizon), 120)
    start = time.perf_counter()
    controller = load_composite_controller_config(controller="BASIC")
    env = suite.make(
        env_name=task,
        robots=["Panda"],
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        horizon=scripted_horizon,
    )
    try:
        obs = env.reset()
        rewards: list[float] = []
        successes: list[float] = []
        for step in range(scripted_horizon):
            eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
            cube = np.asarray(obs["cube_pos"], dtype=float)
            gripper = -1.0
            if step < 35:
                target = cube + np.array([0.0, 0.0, 0.08])
            elif step < 65:
                target = cube + np.array([0.0, 0.0, 0.015])
            elif step < 85:
                target = cube + np.array([0.0, 0.0, 0.015])
                gripper = 1.0
            else:
                target = cube + np.array([0.0, 0.0, 0.25])
                gripper = 1.0
            action = np.zeros(env.action_dim, dtype=float)
            action[:3] = np.clip((target - eef) * 8.0, -1.0, 1.0)
            action[-1] = gripper
            obs, reward, done, _info = env.step(action)
            rewards.append(_scalar(reward))
            if hasattr(env, "_check_success"):
                successes.append(float(bool(env._check_success())))
            if bool(done):
                break
        return _artifact_row(
            "robosuite",
            task,
            seed,
            "scripted_lift_state_feedback",
            scripted_horizon,
            rewards,
            successes,
            time.perf_counter() - start,
        )
    finally:
        env.close()


def _availability_rows() -> list[dict[str, object]]:
    candidates = ["gymnasium", "d4rl", "metaworld", "mani_skill", "robosuite"]
    rows = []
    for name in candidates:
        available = importlib.util.find_spec(name) is not None
        if name == "metaworld" and (Path("results") / "benchmarks" / "metaworld" / "summary.json").exists():
            status = "CORE_GUARDED_ARTIFACT_SEE_RESULTS_BENCHMARKS_METAWORLD"
        else:
            status = "UNSUPPORTED_FUTURE" if not available else "AVAILABLE_FOR_OPTIONAL_ROLLOUT"
        rows.append(
            {
                "benchmark": name,
                "available": available,
                "status": status,
                "claim_boundary": "Optional adapters are guarded and do not support real-robot claims.",
            }
        )
    return rows


def _rollout_specs(smoke: bool) -> list[tuple[str, str, Callable[[str, int, int], dict[str, object]], str]]:
    specs: list[tuple[str, str, Callable[[str, int, int], dict[str, object]], str]] = [
        ("gymnasium", "Pusher-v5", _run_gymnasium, "seeded_random"),
        ("mani_skill", "PushCube-v1", _run_maniskill, "seeded_random"),
        ("mani_skill", "PickCube-v1", _run_maniskill, "seeded_random"),
        ("robosuite", "Lift", _run_robosuite_lift_scripted, "scripted_lift_state_feedback"),
    ]
    if not smoke:
        specs.append(("robosuite", "Door", _run_robosuite, "zero_action"))
    return specs


def _summarize_rollouts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    keys = sorted({(str(r["suite"]), str(r["task"])) for r in rows})
    for suite, task in keys:
        group = [r for r in rows if str(r["suite"]) == suite and str(r["task"]) == task]
        supported = [r for r in group if str(r.get("task_status")) == "SUPPORTED"]
        rewards = [float(r["total_reward"]) for r in supported if r.get("total_reward") != ""]
        success_values = [float(r["success_rate"]) for r in supported if r.get("success_rate") != ""]
        out.append(
            {
                "suite": suite,
                "task": task,
                "num_seeds": len(group),
                "task_status": "SUPPORTED" if rewards else "UNSUPPORTED",
                "mean_total_reward": float(np.mean(rewards)) if rewards else "",
                "mean_reward_per_step": float(np.mean([float(r["mean_reward"]) for r in supported if r.get("mean_reward") != ""])) if rewards else "",
                "success_available": bool(success_values),
                "mean_success_rate": float(np.mean(success_values)) if success_values else "",
                "success_nonzero": bool(success_values and np.max(success_values) > 0.0),
                "failures": "; ".join(str(r.get("failure", "")) for r in group if r.get("failure")),
                "claim_boundary": "Finite optional rollout adapter evidence; success is reported separately and may be zero.",
            }
        )
    return out


def run(smoke: bool = False) -> dict[str, object]:
    out_dir = Path("results") / "optional_benchmarks"
    availability = _availability_rows()
    available = {str(r["benchmark"]) for r in availability if bool(r["available"])}
    seeds = [0] if smoke else [0, 1]
    horizon = 5 if smoke else 12
    rollout_rows: list[dict[str, object]] = []

    for suite, task, runner, policy in _rollout_specs(smoke):
        if suite not in available:
            continue
        for seed in seeds:
            print(f"optional_benchmark suite={suite} task={task} seed={seed}", flush=True)
            try:
                rollout_rows.append(runner(task, seed, horizon))
            except Exception as exc:
                traceback.print_exc(limit=2)
                rollout_rows.append(_failure_row(suite, task, seed, policy, horizon, exc))

    summary_rows = _summarize_rollouts(rollout_rows)
    supported_suites = sorted({str(r["suite"]) for r in summary_rows if str(r.get("task_status")) == "SUPPORTED" and str(r["suite"]) != "metaworld"})
    optional_success_nonzero = bool(any(bool(r.get("success_nonzero")) for r in summary_rows))
    status = "SUPPORTED" if len(supported_suites) >= 2 else ("PARTIAL" if rollout_rows else "UNSUPPORTED")

    write_csv(out_dir / "availability.csv", availability)
    write_csv(out_dir / "rollouts.csv", rollout_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    payload = {
        "experiment": "optional_benchmarks",
        "smoke": smoke,
        "status": status,
        "availability": availability,
        "supported_non_metaworld_suites": supported_suites,
        "optional_success_nonzero": optional_success_nonzero,
        "rollout_summary": summary_rows,
        "claim": "Optional non-Meta-World adapters produce finite guarded rollout artifacts only; manipulation success is reported separately and may be zero.",
        "artifacts": {
            "availability_csv": str(out_dir / "availability.csv"),
            "rollouts_csv": str(out_dir / "rollouts.csv"),
            "summary_csv": str(out_dir / "summary.csv"),
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
