"""Candidate generator wrappers for toy EBM policy experiments."""

from __future__ import annotations

import numpy as np

from .toy_envs import ArrayDict, evaluate_contact_push_actions, evaluate_multimodal_actions, generate_contact_push_pool, generate_multimodal_pool


def gaussian_action_proposals(task: str, seed: int = 0, n: int = 4096) -> ArrayDict:
    if task == "contact_push":
        return generate_contact_push_pool(seed=seed, n=n)
    if task == "multimodal":
        return generate_multimodal_pool(seed=seed, n=n)
    raise ValueError(f"unknown task: {task}")


def mixture_expert_ood_proposals(task: str, seed: int = 0, n: int = 4096) -> ArrayDict:
    return gaussian_action_proposals(task, seed=seed + 101, n=n)


def cem_like_refinement(pool: ArrayDict, energy: np.ndarray, task: str, seed: int = 0, top_frac: float = 0.20, noise_scale: float = 0.10) -> ArrayDict:
    """One CEM-like refinement step around the lowest-energy candidates."""

    rng = np.random.default_rng(seed)
    obs = np.asarray(pool["observations"], dtype=float)
    actions = np.asarray(pool["actions"], dtype=float)
    n = len(actions)
    k = max(4, int(top_frac * n))
    elite_idx = np.argsort(np.asarray(energy, dtype=float), kind="mergesort")[:k]
    chosen = rng.choice(elite_idx, size=n, replace=True)
    refined_actions = actions[chosen] + rng.normal(0.0, noise_scale, size=actions.shape)
    refined_obs = obs[chosen]
    if task == "contact_push":
        evaluated = evaluate_contact_push_actions(refined_obs, refined_actions)
    elif task == "multimodal":
        evaluated = evaluate_multimodal_actions(refined_obs, refined_actions)
    else:
        raise ValueError(f"unknown task: {task}")
    evaluated.update(
        {
            "observations": refined_obs.astype(float),
            "actions": refined_actions.astype(float),
            "candidate_type": np.full(n, 4.0),
        }
    )
    return evaluated


def langevin_like_refinement(pool: ArrayDict, energy: np.ndarray, task: str, seed: int = 0, step_size: float = 0.06) -> ArrayDict:
    """Finite-difference-free toy refinement toward visible low-energy cues."""

    rng = np.random.default_rng(seed)
    obs = np.asarray(pool["observations"], dtype=float)
    actions = np.asarray(pool["actions"], dtype=float).copy()
    target = obs[:, 0]
    actions[:, 1] += 0.55 * step_size * (target - actions[:, 1])
    actions[:, 2] += 0.35 * step_size * (0.28 - actions[:, 2])
    actions[:, 3] += 0.70 * step_size * (0.0 - actions[:, 3])
    actions[:, 0] += rng.normal(0.0, 0.04, size=len(actions))
    if task == "contact_push":
        evaluated = evaluate_contact_push_actions(obs, actions)
    elif task == "multimodal":
        evaluated = evaluate_multimodal_actions(obs, actions)
    else:
        raise ValueError(f"unknown task: {task}")
    evaluated.update({"observations": obs.astype(float), "actions": actions.astype(float), "candidate_type": np.full(len(actions), 5.0)})
    return evaluated
