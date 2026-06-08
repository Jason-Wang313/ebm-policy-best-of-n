"""Synthetic robot-action tasks used by the experiments.

The tasks are intentionally small and CPU-friendly. They are not real robot
benchmarks. Their purpose is to make the EBM-specific failure mode explicit:
the energy model can assign low energy to actions that look plausible in
observable coordinates but are physically invalid or outside support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ArrayDict = dict[str, np.ndarray]


@dataclass(frozen=True)
class ToyTask:
    name: str
    action_dim: int
    description: str
    pool_fn: Callable[[int, int], ArrayDict]


def _gaussian_bump(x: np.ndarray, scale: float) -> np.ndarray:
    return np.exp(-0.5 * (x / scale) ** 2)


def _support_distance_to_modes(x: np.ndarray, y: np.ndarray, target_y: np.ndarray) -> np.ndarray:
    left = (x + 1.0) ** 2 + (y - target_y) ** 2
    right = (x - 1.0) ** 2 + (y - target_y) ** 2
    return np.sqrt(np.minimum(left, right))


def evaluate_multimodal_actions(obs: np.ndarray, actions: np.ndarray) -> ArrayDict:
    """Evaluate a two-mode obstacle-avoidance action toy.

    Action columns: lateral mode x, target progress y, force, jerk. Valid
    actions go around an obstacle at x=-1 or x=+1. The shortcut x=0 looks cheap
    and goal-directed but collides with the obstacle.
    """

    target_y = obs[:, 0]
    x = actions[:, 0]
    y = actions[:, 1]
    force = actions[:, 2]
    jerk = actions[:, 3]

    support_distance = _support_distance_to_modes(x, y, target_y)
    mode_quality = np.maximum(
        _gaussian_bump(np.sqrt((x + 1.0) ** 2 + (y - target_y) ** 2), 0.22),
        _gaussian_bump(np.sqrt((x - 1.0) ** 2 + (y - target_y) ** 2), 0.22),
    )
    goal_quality = _gaussian_bump(y - target_y, 0.24)
    shortcut_collision = _gaussian_bump(x, 0.22) * goal_quality
    contact_penalty = np.clip(shortcut_collision + 0.25 * np.maximum(0.25 - force, 0.0), 0.0, 1.5)
    jerk_penalty = np.clip(jerk * jerk, 0.0, 3.0)
    force_penalty = 0.18 * np.maximum(force - 1.25, 0.0) ** 2
    invalid = (shortcut_collision > 0.45) | (support_distance > 1.8) | (force < 0.05) | (np.abs(jerk) > 2.3)

    utility = (
        1.35 * mode_quality
        + 0.35 * goal_quality
        - 1.85 * contact_penalty
        - 0.28 * jerk_penalty
        - force_penalty
        - 0.15 * np.maximum(support_distance - 0.7, 0.0)
    )
    success = (utility > 0.85) & (~invalid)
    visible_score = (
        0.95 * goal_quality
        + 0.55 * _gaussian_bump(force - 0.35, 0.40)
        + 0.55 * _gaussian_bump(jerk, 0.35)
        + 0.15 * _gaussian_bump(x, 1.2)
    )
    action_norm = np.sqrt(np.sum(actions * actions, axis=1))

    return {
        "utility": utility.astype(float),
        "success": success.astype(float),
        "invalid": invalid.astype(float),
        "jerk_penalty": jerk_penalty.astype(float),
        "contact_penalty": contact_penalty.astype(float),
        "support_penalty": support_distance.astype(float),
        "visible_score": visible_score.astype(float),
        "shortcut_score": shortcut_collision.astype(float),
        "action_norm": action_norm.astype(float),
    }


def generate_multimodal_pool(seed: int = 0, n: int = 4096) -> ArrayDict:
    rng = np.random.default_rng(seed)
    target_y = rng.uniform(0.75, 1.25, size=n)
    obs = target_y[:, None]
    mix = rng.choice(4, size=n, p=[0.34, 0.34, 0.22, 0.10])
    x = np.empty(n)
    y = target_y + rng.normal(0.0, 0.22, size=n)
    force = rng.normal(0.55, 0.22, size=n)
    jerk = rng.normal(0.0, 0.38, size=n)

    left = mix == 0
    right = mix == 1
    shortcut = mix == 2
    random = mix == 3
    x[left] = -1.0 + rng.normal(0.0, 0.18, size=int(np.sum(left)))
    x[right] = 1.0 + rng.normal(0.0, 0.18, size=int(np.sum(right)))
    x[shortcut] = rng.normal(0.0, 0.12, size=int(np.sum(shortcut)))
    y[shortcut] = target_y[shortcut] + rng.normal(0.0, 0.10, size=int(np.sum(shortcut)))
    force[shortcut] = rng.normal(0.18, 0.08, size=int(np.sum(shortcut)))
    jerk[shortcut] = rng.normal(0.0, 0.10, size=int(np.sum(shortcut)))
    x[random] = rng.uniform(-2.4, 2.4, size=int(np.sum(random)))
    y[random] = rng.uniform(0.1, 1.8, size=int(np.sum(random)))
    force[random] = rng.uniform(0.0, 1.6, size=int(np.sum(random)))
    jerk[random] = rng.normal(0.0, 1.0, size=int(np.sum(random)))

    actions = np.column_stack([x, y, np.clip(force, 0.0, 1.8), jerk])
    out = evaluate_multimodal_actions(obs, actions)
    out.update(
        {
            "observations": obs.astype(float),
            "actions": actions.astype(float),
            "candidate_type": mix.astype(float),
        }
    )
    return out


def evaluate_contact_push_actions(obs: np.ndarray, actions: np.ndarray) -> ArrayDict:
    """Evaluate a contact-rich push/insert toy.

    Action columns: approach angle, insertion depth, normal force, trajectory
    jerk. Valid actions use either of two approach corridors. Low-force central
    insertions look smooth and low-energy but lose contact or collide.
    """

    target_depth = obs[:, 0]
    angle = actions[:, 0]
    depth = actions[:, 1]
    force = actions[:, 2]
    jerk = actions[:, 3]

    corridor_distance = np.minimum(np.abs(angle - 0.85), np.abs(angle + 0.85))
    support_penalty = np.sqrt(corridor_distance**2 + (depth - target_depth) ** 2)
    corridor_quality = _gaussian_bump(corridor_distance, 0.16)
    depth_quality = _gaussian_bump(depth - target_depth, 0.18)
    force_quality = _gaussian_bump(force - 0.85, 0.24)
    center_bad = _gaussian_bump(angle, 0.20) * depth_quality
    slip_bad = depth_quality * np.maximum(0.38 - force, 0.0)
    contact_penalty = np.clip(0.9 * center_bad + 1.8 * slip_bad, 0.0, 2.0)
    jerk_penalty = np.clip(jerk * jerk, 0.0, 4.0)
    invalid = (center_bad > 0.40) | (force < 0.12) | (support_penalty > 1.6) | (np.abs(jerk) > 2.4)
    utility = (
        1.25 * corridor_quality
        + 0.55 * depth_quality
        + 0.35 * force_quality
        - 1.75 * contact_penalty
        - 0.22 * jerk_penalty
        - 0.12 * np.maximum(force - 1.3, 0.0) ** 2
    )
    success = (utility > 0.9) & (~invalid)
    visible_score = (
        0.95 * depth_quality
        + 0.48 * _gaussian_bump(force - 0.30, 0.35)
        + 0.65 * _gaussian_bump(jerk, 0.30)
        + 0.18 * _gaussian_bump(angle, 1.0)
    )
    action_norm = np.sqrt(np.sum(actions * actions, axis=1))
    return {
        "utility": utility.astype(float),
        "success": success.astype(float),
        "invalid": invalid.astype(float),
        "jerk_penalty": jerk_penalty.astype(float),
        "contact_penalty": contact_penalty.astype(float),
        "support_penalty": support_penalty.astype(float),
        "visible_score": visible_score.astype(float),
        "shortcut_score": center_bad.astype(float),
        "action_norm": action_norm.astype(float),
    }


def generate_contact_push_pool(seed: int = 0, n: int = 4096) -> ArrayDict:
    rng = np.random.default_rng(seed)
    target_depth = rng.uniform(0.82, 1.15, size=n)
    obs = target_depth[:, None]
    mix = rng.choice(4, size=n, p=[0.31, 0.31, 0.25, 0.13])
    angle = np.empty(n)
    depth = target_depth + rng.normal(0.0, 0.18, size=n)
    force = rng.normal(0.75, 0.25, size=n)
    jerk = rng.normal(0.0, 0.42, size=n)

    left = mix == 0
    right = mix == 1
    shortcut = mix == 2
    random = mix == 3
    angle[left] = -0.85 + rng.normal(0.0, 0.14, size=int(np.sum(left)))
    angle[right] = 0.85 + rng.normal(0.0, 0.14, size=int(np.sum(right)))
    angle[shortcut] = rng.normal(0.0, 0.10, size=int(np.sum(shortcut)))
    depth[shortcut] = target_depth[shortcut] + rng.normal(0.0, 0.08, size=int(np.sum(shortcut)))
    force[shortcut] = rng.normal(0.22, 0.08, size=int(np.sum(shortcut)))
    jerk[shortcut] = rng.normal(0.0, 0.08, size=int(np.sum(shortcut)))
    angle[random] = rng.uniform(-1.8, 1.8, size=int(np.sum(random)))
    depth[random] = rng.uniform(0.2, 1.7, size=int(np.sum(random)))
    force[random] = rng.uniform(0.0, 1.8, size=int(np.sum(random)))
    jerk[random] = rng.normal(0.0, 1.0, size=int(np.sum(random)))

    actions = np.column_stack([angle, depth, np.clip(force, 0.0, 1.8), jerk])
    out = evaluate_contact_push_actions(obs, actions)
    out.update(
        {
            "observations": obs.astype(float),
            "actions": actions.astype(float),
            "candidate_type": mix.astype(float),
        }
    )
    return out


def generate_demo_dataset(seed: int = 0, n: int = 512) -> ArrayDict:
    """Generate multimodal expert demonstrations for IBC-style training."""

    rng = np.random.default_rng(seed)
    target_y = rng.uniform(0.75, 1.25, size=n)
    obs = target_y[:, None]
    mode = rng.choice([-1.0, 1.0], size=n)
    x = mode + rng.normal(0.0, 0.08, size=n)
    y = target_y + rng.normal(0.0, 0.06, size=n)
    force = rng.normal(0.58, 0.08, size=n)
    jerk = rng.normal(0.0, 0.10, size=n)
    actions = np.column_stack([x, y, np.clip(force, 0.2, 1.1), jerk])
    evaluated = evaluate_multimodal_actions(obs, actions)
    evaluated.update({"observations": obs.astype(float), "actions": actions.astype(float), "mode": mode})
    return evaluated


def sample_ibc_negatives(obs: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Negative sampler used for contrastive IBC training.

    It intentionally samples broad wrong actions and misses the narrow
    low-effort obstacle shortcut, matching a common support-coverage problem.
    """

    batch = obs.shape[0]
    target_y = obs[:, 0][:, None]
    x = rng.uniform(-2.3, 2.3, size=(batch, k))
    y = target_y + rng.normal(0.0, 0.65, size=(batch, k))
    force = rng.uniform(0.0, 1.7, size=(batch, k))
    jerk = rng.normal(0.0, 0.95, size=(batch, k))
    return np.stack([x, y, force, jerk], axis=2).astype(float)


TASKS = {
    "contact_push": ToyTask(
        name="contact_push",
        action_dim=4,
        description="Contact-rich push/insert toy with low-force central shortcut failures.",
        pool_fn=generate_contact_push_pool,
    ),
    "multimodal": ToyTask(
        name="multimodal",
        action_dim=4,
        description="Two valid action modes around an obstacle plus an invalid low-effort shortcut.",
        pool_fn=generate_multimodal_pool,
    ),
}
