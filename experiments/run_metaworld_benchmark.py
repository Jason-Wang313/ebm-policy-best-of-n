from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ebm_tail_audit.calibration import apply_energy_calibrator, fit_energy_calibrator
from ebm_tail_audit.diagnostics import (
    aggregate_curve_rows,
    annotate_repair_effectiveness,
    curve_rows_for_energy,
    energy_reliability_rows,
    make_conservative_gated_rows,
)
from ebm_tail_audit.energy_models import oracle_energy, raw_miscalibrated_energy, support_penalized_energy, value_shaped_energy
from ebm_tail_audit.torch_models import TorchIBCEnergy, _stabilize_torch_runtime
from ebm_tail_audit.toy_envs import generate_contact_push_pool
from ebm_tail_audit.utils import METAWORLD_N_VALUES, write_csv, write_json


TASK_SPECS = {
    "reach-v3": ("metaworld.policies.sawyer_reach_v3_policy", "SawyerReachV3Policy"),
    "push-v3": ("metaworld.policies.sawyer_push_v3_policy", "SawyerPushV3Policy"),
    "pick-place-v3": ("metaworld.policies.sawyer_pick_place_v3_policy", "SawyerPickPlaceV3Policy"),
    "button-press-v3": ("metaworld.policies.sawyer_button_press_v3_policy", "SawyerButtonPressV3Policy"),
}


CLOSED_LOOP_VARIANTS = [
    {
        "policy": "expert_centered_gate",
        "proposal_source": "scripted_expert_centered_candidates",
        "uses_expert_centered_proposals": True,
        "conservative_proposal_gate": True,
        "uses_learned_ebm": True,
        "uses_learned_demo_proposal": False,
        "uses_learned_bc_proposal": False,
        "uses_state_heuristic_proposal": False,
    },
    {
        "policy": "expert_centered_no_gate",
        "proposal_source": "scripted_expert_centered_candidates",
        "uses_expert_centered_proposals": True,
        "conservative_proposal_gate": False,
        "uses_learned_ebm": True,
        "uses_learned_demo_proposal": False,
        "uses_learned_bc_proposal": False,
        "uses_state_heuristic_proposal": False,
    },
    {
        "policy": "mixed_expert_local_no_gate",
        "proposal_source": "mixed_expert_centered_and_local_gaussian_candidates",
        "uses_expert_centered_proposals": True,
        "conservative_proposal_gate": False,
        "uses_learned_ebm": True,
        "uses_learned_demo_proposal": False,
        "uses_learned_bc_proposal": False,
        "uses_state_heuristic_proposal": False,
    },
    {
        "policy": "local_gaussian_no_gate",
        "proposal_source": "local_gaussian_candidates",
        "uses_expert_centered_proposals": False,
        "conservative_proposal_gate": False,
        "uses_learned_ebm": True,
        "uses_learned_demo_proposal": False,
        "uses_learned_bc_proposal": False,
        "uses_state_heuristic_proposal": False,
    },
    {
        "policy": "learned_demo_proposal_no_gate",
        "proposal_source": "nearest_demo_behavior_proposal_candidates",
        "uses_expert_centered_proposals": False,
        "conservative_proposal_gate": False,
        "uses_learned_ebm": True,
        "uses_learned_demo_proposal": True,
        "uses_learned_bc_proposal": False,
        "uses_state_heuristic_proposal": False,
    },
    {
        "policy": "learned_demo_proposal_direct",
        "proposal_source": "nearest_demo_behavior_proposal_direct_action",
        "uses_expert_centered_proposals": False,
        "conservative_proposal_gate": False,
        "uses_learned_ebm": False,
        "uses_learned_demo_proposal": True,
        "uses_learned_bc_proposal": False,
        "uses_state_heuristic_proposal": False,
    },
    {
        "policy": "learned_bc_proposal_no_gate",
        "proposal_source": "behavior_cloned_proposal_candidates",
        "uses_expert_centered_proposals": False,
        "conservative_proposal_gate": False,
        "uses_learned_ebm": True,
        "uses_learned_demo_proposal": False,
        "uses_learned_bc_proposal": True,
        "uses_state_heuristic_proposal": False,
    },
    {
        "policy": "learned_bc_proposal_direct",
        "proposal_source": "behavior_cloned_direct_action",
        "uses_expert_centered_proposals": False,
        "conservative_proposal_gate": False,
        "uses_learned_ebm": False,
        "uses_learned_demo_proposal": False,
        "uses_learned_bc_proposal": True,
        "uses_state_heuristic_proposal": False,
    },
    {
        "policy": "state_heuristic_proposal_no_gate",
        "proposal_source": "state_heuristic_proposal_candidates",
        "uses_expert_centered_proposals": False,
        "conservative_proposal_gate": False,
        "uses_learned_ebm": True,
        "uses_learned_demo_proposal": False,
        "uses_learned_bc_proposal": False,
        "uses_state_heuristic_proposal": True,
    },
    {
        "policy": "state_heuristic_direct",
        "proposal_source": "state_heuristic_direct_action",
        "uses_expert_centered_proposals": False,
        "conservative_proposal_gate": False,
        "uses_learned_ebm": False,
        "uses_learned_demo_proposal": False,
        "uses_learned_bc_proposal": False,
        "uses_state_heuristic_proposal": True,
    },
    {
        "policy": "random_uniform",
        "proposal_source": "uniform_random_actions",
        "uses_expert_centered_proposals": False,
        "conservative_proposal_gate": False,
        "uses_learned_ebm": False,
        "uses_learned_demo_proposal": False,
        "uses_learned_bc_proposal": False,
        "uses_state_heuristic_proposal": False,
    },
    {
        "policy": "scripted_expert",
        "proposal_source": "scripted_expert_policy",
        "uses_expert_centered_proposals": True,
        "conservative_proposal_gate": False,
        "uses_learned_ebm": False,
        "uses_learned_demo_proposal": False,
        "uses_learned_bc_proposal": False,
        "uses_state_heuristic_proposal": False,
    },
]

CLOSED_LOOP_GATE_DISTANCE = 0.35
METAWORLD_RUNNER_VERSION = "metaworld_dependency_audit_v4"


def _metaworld_settings(smoke: bool) -> dict[str, int]:
    return {
        "demo_tasks": 2 if smoke else 5,
        "demo_horizon": 90 if smoke else 100,
        "train_states": 4 if smoke else 6,
        "eval_states": 2 if smoke else 3,
        "train_candidates": 18 if smoke else 20,
        "eval_candidates": 12 if smoke else 16,
        "max_advance": 8 if smoke else 10,
        "eval_continuation_horizon": 90 if smoke else 100,
        "epochs": 12 if smoke else 14,
        "negatives": 10 if smoke else 12,
        "mc_trials": 120 if smoke else 180,
        "fallback_pool_n": 512 if smoke else 1024,
        "closed_loop_horizon": 80 if smoke else 100,
        "closed_loop_candidates_per_step": 8 if smoke else 12,
        "behavior_proposal_epochs": 120 if smoke else 160,
    }


def _policy_class(task_name: str):
    module_name, class_name = TASK_SPECS[task_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _clip_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(action, dtype=np.float32), low, high).astype(np.float32)


def _contact_failure(info: dict[str, Any], success: float) -> float:
    in_place = float(info.get("in_place_reward", 0.0))
    near_object = float(info.get("near_object", 0.0))
    grasp = float(info.get("grasp_success", 0.0))
    quality = max(success, min(max(in_place, near_object, grasp), 1.0))
    return float(np.clip(1.0 - quality, 0.0, 1.0))


def _make_pool(
    observations: list[np.ndarray],
    actions: list[np.ndarray],
    rewards: list[float],
    successes: list[float],
    contact_failures: list[float],
    support_distances: list[float],
    one_step_rewards: list[float] | None = None,
) -> dict[str, np.ndarray]:
    rewards_arr = np.asarray(rewards, dtype=float)
    success_arr = np.asarray(successes, dtype=float)
    contact_arr = np.asarray(contact_failures, dtype=float)
    support_raw = np.asarray(support_distances, dtype=float)
    action_arr = np.vstack(actions).astype(float)
    action_norm = np.linalg.norm(action_arr, axis=1)
    if rewards_arr.size == 0:
        raise RuntimeError("candidate pool contains no actions")
    reward_cutoff = float(np.quantile(rewards_arr, 0.25))
    support_scale = float(np.std(support_raw) + 1e-6)
    out = {
        "observations": np.vstack(observations).astype(float),
        "actions": action_arr,
        "utility": rewards_arr,
        "success": (success_arr > 0.5).astype(float),
        "invalid": ((rewards_arr <= reward_cutoff) | (contact_arr > 0.85)).astype(float),
        "jerk_penalty": action_norm * action_norm,
        "contact_penalty": contact_arr,
        "support_penalty": (support_raw - float(np.min(support_raw))) / support_scale,
        "support_distance": support_raw,
        "visible_score": (rewards_arr - float(np.mean(rewards_arr))) / float(np.std(rewards_arr) + 1e-6),
        "shortcut_score": (rewards_arr <= reward_cutoff).astype(float),
        "action_norm": action_norm,
    }
    if one_step_rewards is not None:
        out["one_step_reward"] = np.asarray(one_step_rewards, dtype=float)
    return out


def _collect_scripted_demos(task_name: str, seed: int, num_tasks: int, horizon: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    import metaworld

    rng = np.random.default_rng(seed)
    mt1 = metaworld.MT1(task_name)
    env_cls = mt1.train_classes[task_name]
    env = env_cls()
    policy_cls = _policy_class(task_name)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    rewards: list[float] = []
    successes: list[float] = []
    start = time.perf_counter()
    for task_idx in range(num_tasks):
        task = mt1.train_tasks[(seed + task_idx) % len(mt1.train_tasks)]
        env.set_task(task)
        obs, _info = env.reset(seed=seed + task_idx)
        policy = policy_cls()
        for _step in range(horizon):
            action = _clip_action(policy.get_action(obs), action_low, action_high)
            obs_rows.append(np.asarray(obs, dtype=float))
            action_rows.append(action.astype(float))
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            successes.append(float(info.get("success", 0.0)))
            if terminated or truncated:
                break
        # Add a small number of expert-near positives so the contrastive model
        # sees local support, not only the exact scripted action.
        if obs_rows:
            for _ in range(2):
                action_rows.append(_clip_action(action_rows[-1] + rng.normal(0.0, 0.03, size=len(action_low)), action_low, action_high))
                obs_rows.append(obs_rows[-1])
    elapsed = time.perf_counter() - start
    metadata = {
        "demo_rollout_tasks": int(num_tasks),
        "demo_horizon": int(horizon),
        "num_demo_actions": int(len(action_rows)),
        "demo_runtime_seconds": float(elapsed),
        "demo_mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "demo_success_rate": float(np.mean(successes)) if successes else float("nan"),
        "action_low": action_low.astype(float).tolist(),
        "action_high": action_high.astype(float).tolist(),
    }
    if not obs_rows:
        raise RuntimeError(f"scripted policy produced no demos for {task_name}")
    return np.vstack(obs_rows).astype(np.float32), np.vstack(action_rows).astype(np.float32), metadata


class _NearestDemoProposal:
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
        model_type: str = "nearest-demo behavior proposal",
        training_source: str = "scripted_demo_observation_action_pairs",
        feature_mode: str = "raw_observation",
    ):
        obs = np.asarray(observations, dtype=np.float32)
        acts = np.asarray(actions, dtype=np.float32)
        if len(obs) == 0 or len(acts) == 0:
            raise ValueError("nearest-demo proposal requires non-empty demos")
        self.observations = obs
        self.actions = acts
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.feature_mode = str(feature_mode)
        features = self._features(obs)
        self.obs_mean = np.mean(features, axis=0).astype(np.float32)
        self.obs_scale = (np.std(features, axis=0) + 1e-4).astype(np.float32)
        self.normalized_observations = ((features - self.obs_mean) / self.obs_scale).astype(np.float32)
        self.model_type = str(model_type)
        self.training_source = str(training_source)

    def _features(self, observations: np.ndarray) -> np.ndarray:
        if self.feature_mode == "raw_observation_plus_hand_object_goal_relations":
            return _behavior_observation_features(observations)
        if self.feature_mode == "raw_observation":
            obs = np.asarray(observations, dtype=np.float32)
            return obs[None, :] if obs.ndim == 1 else obs
        raise ValueError(f"unknown nearest proposal feature mode: {self.feature_mode}")

    def predict(self, obs: np.ndarray) -> np.ndarray:
        phi = self._features(np.asarray(obs, dtype=np.float32))[0]
        z = ((phi - self.obs_mean) / self.obs_scale).astype(np.float32)
        dist = np.sum((self.normalized_observations - z[None, :]) ** 2, axis=1)
        idx = int(np.argmin(dist))
        return _clip_action(self.actions[idx], self.action_low, self.action_high)

    def metadata(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "training_source": self.training_source,
            "feature_mode": self.feature_mode,
            "num_demo_pairs": int(len(self.actions)),
            "num_training_pairs": int(len(self.actions)),
            "runtime_scripted_expert_dependency": False,
        }


def _behavior_observation_features(observations: np.ndarray) -> np.ndarray:
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    if obs.shape[1] < 7:
        return obs.astype(np.float32)
    hand = obs[:, :3]
    gripper = obs[:, 3:4]
    obj = obs[:, 4:7]
    goal = obs[:, -3:]
    rel = np.concatenate(
        [
            goal - hand,
            obj - hand,
            goal - obj,
            np.linalg.norm(goal - hand, axis=1, keepdims=True),
            np.linalg.norm(obj - hand, axis=1, keepdims=True),
            np.linalg.norm(goal - obj, axis=1, keepdims=True),
            gripper,
        ],
        axis=1,
    )
    return np.concatenate([obs, rel], axis=1).astype(np.float32)


class _TorchBehaviorProposal:
    def __init__(
        self,
        state_dict: dict[str, object],
        obs_mean: np.ndarray,
        obs_scale: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        train_loss: list[float],
        train_size: int,
        model_type: str,
        training_source: str,
        training_objective: str,
        feature_mode: str,
    ):
        self.state_dict = state_dict
        self.obs_mean = np.asarray(obs_mean, dtype=np.float32)
        self.obs_scale = np.asarray(obs_scale, dtype=np.float32)
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.train_loss = [float(x) for x in train_loss]
        self.train_size = int(train_size)
        self.model_type = str(model_type)
        self.training_source = str(training_source)
        self.training_objective = str(training_objective)
        self.feature_mode = str(feature_mode)
        self._cached_model = None

    @classmethod
    def fit(
        cls,
        observations: np.ndarray,
        actions: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
        seed: int,
        epochs: int,
        batch_size: int = 128,
        hidden_dim: int = 128,
        lr: float = 2e-3,
        weight_decay: float = 0.0,
        model_type: str = "PyTorch MLP behavior-cloned action proposal",
        training_source: str = "scripted_demo_and_high_reward_candidate_actions",
        training_objective: str = "MSE behavior cloning from observations to normalized actions",
        feature_mode: str = "raw_observation_plus_hand_object_goal_relations",
    ) -> "_TorchBehaviorProposal":
        import torch

        _stabilize_torch_runtime(torch)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        obs = np.asarray(observations, dtype=np.float32)
        acts = np.asarray(actions, dtype=np.float32)
        if len(obs) == 0 or len(acts) == 0:
            raise ValueError("behavior-cloned proposal requires non-empty training data")
        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)
        phi = _behavior_observation_features(obs)
        obs_mean = np.mean(phi, axis=0).astype(np.float32)
        obs_scale = (np.std(phi, axis=0) + 1e-4).astype(np.float32)
        action_mid = ((action_high + action_low) / 2.0).astype(np.float32)
        action_half_range = np.maximum((action_high - action_low) / 2.0, 1e-4).astype(np.float32)
        targets = np.clip((acts - action_mid) / action_half_range, -1.0, 1.0).astype(np.float32)
        x = ((phi - obs_mean) / obs_scale).astype(np.float32)

        model = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, targets.shape[1]),
            torch.nn.Tanh(),
        )
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        losses: list[float] = []
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(targets, dtype=torch.float32)
        n = len(x)
        for _epoch in range(int(epochs)):
            order = rng.permutation(n)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, n, int(batch_size)):
                idx = order[start : start + int(batch_size)]
                pred = model(x_t[idx])
                loss = torch.nn.functional.mse_loss(pred, y_t[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.detach().cpu())
                batches += 1
            losses.append(epoch_loss / max(1, batches))

        return cls(
            state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            obs_mean=obs_mean,
            obs_scale=obs_scale,
            action_low=action_low,
            action_high=action_high,
            input_dim=x.shape[1],
            hidden_dim=int(hidden_dim),
            action_dim=targets.shape[1],
            train_loss=losses,
            train_size=n,
            model_type=model_type,
            training_source=training_source,
            training_objective=training_objective,
            feature_mode=feature_mode,
        )

    def _model(self):
        import torch

        if self._cached_model is not None:
            return self._cached_model
        model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, self.action_dim),
            torch.nn.Tanh(),
        )
        model.load_state_dict(self.state_dict)
        model.eval()
        self._cached_model = model
        return model

    def predict(self, obs: np.ndarray) -> np.ndarray:
        import torch

        _stabilize_torch_runtime(torch)
        phi = _behavior_observation_features(np.asarray(obs, dtype=np.float32))[0]
        x = ((phi - self.obs_mean) / self.obs_scale).astype(np.float32)
        model = self._model()
        with torch.no_grad():
            scaled = model(torch.tensor(x[None, :], dtype=torch.float32)).reshape(-1).cpu().numpy()
        mid = (self.action_high + self.action_low) / 2.0
        half_range = np.maximum((self.action_high - self.action_low) / 2.0, 1e-4)
        return _clip_action(mid + scaled.astype(np.float32) * half_range, self.action_low, self.action_high)

    def metadata(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "training_source": self.training_source,
            "training_objective": self.training_objective,
            "feature_mode": self.feature_mode,
            "num_training_pairs": self.train_size,
            "hidden_dim": self.hidden_dim,
            "initial_train_loss": float(self.train_loss[0]) if self.train_loss else float("nan"),
            "final_train_loss": float(self.train_loss[-1]) if self.train_loss else float("nan"),
            "loss_decrease": float(self.train_loss[0] - self.train_loss[-1]) if len(self.train_loss) >= 2 else 0.0,
            "runtime_scripted_expert_dependency": False,
        }


def _candidate_actions(
    expert_action: np.ndarray,
    rng: np.random.Generator,
    action_low: np.ndarray,
    action_high: np.ndarray,
    candidates_per_state: int,
) -> list[np.ndarray]:
    actions = [_clip_action(expert_action, action_low, action_high)]
    local_count = min(8, max(2, candidates_per_state // 5))
    for _ in range(local_count):
        scale = rng.choice([0.05, 0.12, 0.25])
        actions.append(_clip_action(expert_action + rng.normal(0.0, scale, size=len(action_low)), action_low, action_high))
    while len(actions) < candidates_per_state:
        if rng.random() < 0.35:
            # Smooth low-effort proposals are plausible EBM false positives.
            proposal = np.zeros(len(action_low), dtype=np.float32)
            proposal[: min(3, len(proposal))] = rng.normal(0.0, 0.18, size=min(3, len(proposal)))
            proposal[-1] = rng.normal(0.0, 0.08)
            actions.append(_clip_action(proposal, action_low, action_high))
        else:
            actions.append(rng.uniform(action_low, action_high).astype(np.float32))
    return actions


def _local_gaussian_actions(
    previous_action: np.ndarray | None,
    rng: np.random.Generator,
    action_low: np.ndarray,
    action_high: np.ndarray,
    candidates_per_state: int,
) -> list[np.ndarray]:
    center = np.zeros_like(action_low, dtype=np.float32) if previous_action is None else _clip_action(previous_action, action_low, action_high)
    actions = [_clip_action(center, action_low, action_high)]
    while len(actions) < candidates_per_state:
        scale = rng.choice([0.08, 0.18, 0.35, 0.65])
        actions.append(_clip_action(center + rng.normal(0.0, scale, size=len(action_low)), action_low, action_high))
    return actions


def _learned_demo_proposal_actions(
    learned_action: np.ndarray,
    previous_action: np.ndarray | None,
    rng: np.random.Generator,
    action_low: np.ndarray,
    action_high: np.ndarray,
    candidates_per_state: int,
) -> list[np.ndarray]:
    center = _clip_action(learned_action, action_low, action_high)
    actions = [center]
    if previous_action is not None:
        prev = _clip_action(previous_action, action_low, action_high)
        actions.append(_clip_action(0.8 * center + 0.2 * prev, action_low, action_high))
    while len(actions) < candidates_per_state:
        scale = rng.choice([0.01, 0.025, 0.05, 0.10, 0.18])
        actions.append(_clip_action(center + rng.normal(0.0, scale, size=len(action_low)), action_low, action_high))
    return actions


def _move_toward(current: np.ndarray, target: np.ndarray, gain: float) -> np.ndarray:
    return (float(gain) * (np.asarray(target, dtype=np.float32) - np.asarray(current, dtype=np.float32))).astype(np.float32)


def _state_heuristic_action(task_name: str, obs: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    hand = obs[:3]
    obj = obs[4:7]
    goal = obs[-3:]
    action = np.zeros(4, dtype=np.float32)
    if task_name == "reach-v3":
        action[:3] = _move_toward(hand, goal, 5.0)
        action[3] = 0.0
    elif task_name == "push-v3":
        puck = obj + np.asarray([-0.005, 0.0, 0.0], dtype=np.float32)
        if np.linalg.norm(hand[:2] - puck[:2]) > 0.02:
            desired = puck + np.asarray([0.0, 0.0, 0.2], dtype=np.float32)
        elif abs(float(hand[2] - puck[2])) > 0.04:
            desired = puck + np.asarray([0.0, 0.0, 0.03], dtype=np.float32)
        else:
            desired = goal
        grab = 0.0 if np.linalg.norm(hand[:2] - obj[:2]) > 0.02 or abs(float(hand[2] - obj[2])) > 0.10 else 0.6
        action[:3] = _move_toward(hand, desired, 10.0)
        action[3] = grab
    elif task_name == "pick-place-v3":
        puck = obj + np.asarray([-0.005, 0.0, 0.0], dtype=np.float32)
        gripper_separation = float(obs[3])
        if np.linalg.norm(hand[:2] - puck[:2]) > 0.02:
            desired = puck + np.asarray([0.0, 0.0, 0.1], dtype=np.float32)
        elif abs(float(hand[2] - puck[2])) > 0.05 and float(puck[2]) < 0.04:
            desired = puck + np.asarray([0.0, 0.0, 0.03], dtype=np.float32)
        elif gripper_separation > 0.73:
            desired = hand
        else:
            desired = goal
        action[:3] = _move_toward(hand, desired, 10.0)
        action[3] = 1.0 if np.linalg.norm(hand - obj) < 0.07 else 0.0
    elif task_name == "button-press-v3":
        button = obj + np.asarray([0.0, 0.0, -0.07], dtype=np.float32)
        if not np.all(np.isclose(np.asarray([hand[0], hand[2]]), np.asarray([button[0], button[2]]), atol=0.02)):
            desired = button.copy()
            desired[1] = hand[1] - 0.1
        else:
            desired = button.copy()
            desired[1] += 0.02
        action[:3] = _move_toward(hand, desired, 25.0)
        action[3] = 0.0
    else:
        raise ValueError(f"unsupported state heuristic task: {task_name}")
    return _clip_action(action, action_low, action_high)


def _state_heuristic_proposal_actions(
    heuristic_action: np.ndarray,
    previous_action: np.ndarray | None,
    rng: np.random.Generator,
    action_low: np.ndarray,
    action_high: np.ndarray,
    candidates_per_state: int,
) -> list[np.ndarray]:
    center = _clip_action(heuristic_action, action_low, action_high)
    actions = [center]
    if previous_action is not None:
        prev = _clip_action(previous_action, action_low, action_high)
        actions.append(_clip_action(0.85 * center + 0.15 * prev, action_low, action_high))
    while len(actions) < candidates_per_state:
        scale = rng.choice([0.005, 0.015, 0.035, 0.075])
        actions.append(_clip_action(center + rng.normal(0.0, scale, size=len(action_low)), action_low, action_high))
    return actions


def _closed_loop_candidate_actions(
    variant: dict[str, object],
    task_name: str,
    expert_action: np.ndarray,
    previous_action: np.ndarray | None,
    learned_action: np.ndarray | None,
    behavior_action: np.ndarray | None,
    heuristic_action: np.ndarray | None,
    rng: np.random.Generator,
    action_low: np.ndarray,
    action_high: np.ndarray,
    candidates_per_step: int,
) -> list[np.ndarray]:
    policy = str(variant["policy"])
    if policy in {"expert_centered_gate", "expert_centered_no_gate"}:
        return _candidate_actions(expert_action, rng, action_low, action_high, candidates_per_step)
    if policy == "mixed_expert_local_no_gate":
        expert_count = max(1, candidates_per_step // 2)
        local_count = max(1, candidates_per_step - expert_count)
        actions = _candidate_actions(expert_action, rng, action_low, action_high, expert_count)
        actions.extend(_local_gaussian_actions(previous_action, rng, action_low, action_high, local_count))
        return actions[:candidates_per_step]
    if policy == "local_gaussian_no_gate":
        return _local_gaussian_actions(previous_action, rng, action_low, action_high, candidates_per_step)
    if policy == "learned_demo_proposal_no_gate":
        if learned_action is None:
            raise ValueError("learned-demo proposal variant requires a learned action")
        return _learned_demo_proposal_actions(learned_action, previous_action, rng, action_low, action_high, candidates_per_step)
    if policy == "learned_bc_proposal_no_gate":
        if behavior_action is None:
            raise ValueError("behavior-cloned proposal variant requires a learned action")
        return _learned_demo_proposal_actions(behavior_action, previous_action, rng, action_low, action_high, candidates_per_step)
    if policy == "state_heuristic_proposal_no_gate":
        if heuristic_action is None:
            raise ValueError("state-heuristic proposal variant requires a heuristic action")
        return _state_heuristic_proposal_actions(heuristic_action, previous_action, rng, action_low, action_high, candidates_per_step)
    if policy == "random_uniform":
        return [rng.uniform(action_low, action_high).astype(np.float32) for _ in range(candidates_per_step)]
    raise ValueError(f"unknown closed-loop policy variant: {policy}")


def _collect_candidate_pool(
    task_name: str,
    seed: int,
    num_states: int,
    candidates_per_state: int,
    max_advance: int,
    continuation_horizon: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    import metaworld

    rng = np.random.default_rng(seed)
    mt1 = metaworld.MT1(task_name)
    env_cls = mt1.train_classes[task_name]
    env = env_cls()
    policy_cls = _policy_class(task_name)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    one_step_rewards: list[float] = []
    successes: list[float] = []
    contact_failures: list[float] = []
    support_distances: list[float] = []
    start = time.perf_counter()

    for state_idx in range(num_states):
        task = mt1.train_tasks[(seed + state_idx) % len(mt1.train_tasks)]
        env.set_task(task)
        obs, _info = env.reset(seed=seed + state_idx)
        policy = policy_cls()
        advance = state_idx % max(1, max_advance)
        for _ in range(advance):
            action = _clip_action(policy.get_action(obs), action_low, action_high)
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
        state = env.get_env_state()
        expert_action = _clip_action(policy.get_action(obs), action_low, action_high)
        for candidate in _candidate_actions(expert_action, rng, action_low, action_high, candidates_per_state):
            env.set_task(task)
            env.reset(seed=seed + state_idx)
            env.set_env_state(state)
            next_obs, reward, terminated, truncated, info = env.step(candidate)
            one_step_reward = float(reward)
            rollout_rewards = [float(reward)]
            max_reward = float(reward)
            max_success = float(info.get("success", 0.0))
            min_contact_failure = _contact_failure(info, max_success)
            min_support_distance = float(info.get("obj_to_target", max(0.0, 10.0 - float(reward))))
            rollout_obs = next_obs
            for _ in range(max(0, int(continuation_horizon) - 1)):
                if terminated or truncated:
                    break
                follow_action = _clip_action(policy.get_action(rollout_obs), action_low, action_high)
                rollout_obs, follow_reward, terminated, truncated, follow_info = env.step(follow_action)
                follow_success = float(follow_info.get("success", 0.0))
                rollout_rewards.append(float(follow_reward))
                max_reward = max(max_reward, float(follow_reward))
                max_success = max(max_success, follow_success)
                min_contact_failure = min(min_contact_failure, _contact_failure(follow_info, follow_success))
                min_support_distance = min(
                    min_support_distance,
                    float(follow_info.get("obj_to_target", max(0.0, 10.0 - float(follow_reward)))),
                )
            observations.append(np.asarray(obs, dtype=float))
            actions.append(np.asarray(candidate, dtype=float))
            rewards.append(float(np.mean(rollout_rewards)))
            one_step_rewards.append(one_step_reward)
            successes.append(max_success)
            contact_failures.append(min_contact_failure)
            support_distances.append(min_support_distance)

    elapsed = time.perf_counter() - start
    pool = _make_pool(observations, actions, rewards, successes, contact_failures, support_distances, one_step_rewards=one_step_rewards)
    metadata = {
        "task": task_name,
        "num_states": int(num_states),
        "candidates_per_state": int(candidates_per_state),
        "num_candidates": int(len(rewards)),
        "collection_runtime_seconds": float(elapsed),
        "utility_mode": "mean_reward_over_selected_action_then_scripted_continuation",
        "continuation_horizon": int(continuation_horizon),
        "action_low": action_low.astype(float).tolist(),
        "action_high": action_high.astype(float).tolist(),
        "mean_one_step_reward": float(np.mean(one_step_rewards)),
        "mean_reward": float(np.mean(rewards)),
        "max_reward": float(np.max(rewards)),
        "success_available": True,
        "success_nonzero": bool(np.max(successes) > 0.0),
        "mean_success_rate": float(np.mean(successes)),
        "mean_contact_failure_rate": float(np.mean(contact_failures)),
        "mean_support_distance": float(np.mean(support_distances)),
    }
    return pool, metadata


def _positives_from_pool(pool: dict[str, np.ndarray], quantile: float = 0.82) -> tuple[np.ndarray, np.ndarray]:
    cutoff = float(np.quantile(pool["utility"], quantile))
    mask = pool["utility"] >= cutoff
    if int(np.sum(mask)) < 8:
        idx = np.argsort(pool["utility"])[-8:]
        mask = np.zeros(len(pool["utility"]), dtype=bool)
        mask[idx] = True
    return pool["observations"][mask].astype(np.float32), pool["actions"][mask].astype(np.float32)


def _best_actions_from_pool(pool: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(pool["observations"], dtype=np.float32)
    actions = np.asarray(pool["actions"], dtype=np.float32)
    utility = np.asarray(pool.get("one_step_reward", pool["utility"]), dtype=float)
    best_by_obs: dict[bytes, tuple[float, int]] = {}
    for idx, obs in enumerate(observations):
        key = np.round(obs, 6).tobytes()
        score = float(utility[idx])
        if key not in best_by_obs or score > best_by_obs[key][0]:
            best_by_obs[key] = (score, idx)
    best_idx = [idx for _score, idx in best_by_obs.values()]
    if not best_idx:
        return observations[:0], actions[:0]
    return observations[best_idx].astype(np.float32), actions[best_idx].astype(np.float32)


def _behavior_proposal_training_set(
    demo_obs: np.ndarray,
    demo_actions: np.ndarray,
    train_pool: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    pos_obs, pos_actions = _positives_from_pool(train_pool, quantile=0.70)
    best_obs, best_actions = _best_actions_from_pool(train_pool)
    obs = np.vstack([demo_obs, pos_obs, best_obs, best_obs]).astype(np.float32)
    actions = np.vstack([demo_actions, pos_actions, best_actions, best_actions]).astype(np.float32)
    return obs, actions


def _add_curve_rows(
    seed_rows: list[dict[str, object]],
    reliability_rows: list[dict[str, object]],
    pool: dict[str, np.ndarray],
    energies: dict[str, tuple[np.ndarray, str]],
    task_name: str,
    seed: int,
    task_status: str,
    task_error: str,
    success_available: bool,
    success_nonzero: bool,
    mc_trials: int,
    benchmark: str = "metaworld",
) -> None:
    for model_name, (energy, training_source) in energies.items():
        rows, _summary = curve_rows_for_energy(pool, energy, METAWORLD_N_VALUES, model_name, seed, exact_mc_trials=mc_trials)
        for row in rows:
            row["benchmark"] = benchmark
            row["task"] = task_name
            row["training_source"] = training_source
            row["task_status"] = task_status
            row["task_error"] = task_error
            row["success_available"] = success_available
            row["success_nonzero"] = success_nonzero
            seed_rows.append(row)
        if model_name in {"expert_ibc", "expert_calibrated", "high_reward_ablation", "oracle"}:
            reliability_rows.extend(
                energy_reliability_rows(
                    pool,
                    energy,
                    model=model_name,
                    seed=seed,
                    task=task_name,
                    benchmark=benchmark,
                    training_source=training_source,
                )
            )


def _fallback_rows_for_task(
    task_name: str,
    seed: int,
    pool_n: int,
    mc_trials: int,
    seed_rows: list[dict[str, object]],
    reliability_rows: list[dict[str, object]],
    error: str,
) -> None:
    pool = generate_contact_push_pool(seed + 1200, pool_n)
    raw = raw_miscalibrated_energy(pool, seed)
    energies = {
        "fallback_raw_ebm": (raw, "fallback_contact_push"),
        "fallback_support_penalized": (support_penalized_energy(raw, pool, weight=2.0), "fallback_contact_push"),
        "fallback_value_shaped": (value_shaped_energy(raw, pool), "upper_bound"),
        "fallback_oracle": (oracle_energy(pool), "upper_bound"),
    }
    _add_curve_rows(
        seed_rows,
        reliability_rows,
        pool,
        energies,
        task_name=f"{task_name}_fallback_contact_push",
        seed=seed,
        task_status="PARTIAL",
        task_error=error,
        success_available=True,
        success_nonzero=bool(np.max(pool["success"]) > 0.0),
        mc_trials=mc_trials,
        benchmark="fallback_contact_push",
    )


def _aggregate_task_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    tasks = sorted({str(r["task"]) for r in rows})
    for task in tasks:
        task_rows = [r for r in rows if str(r["task"]) == task]
        for row in aggregate_curve_rows(task_rows):
            row["task"] = task
            row["benchmark"] = task_rows[0].get("benchmark", "metaworld")
            row["task_status"] = max(set(str(r["task_status"]) for r in task_rows), key=[str(r["task_status"]) for r in task_rows].count)
            out.append(row)
    return out


def _benchmark_table(seed_rows: list[dict[str, object]], high_n: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in sorted({str(r["task"]) for r in seed_rows}):
        for model in [
            "expert_ibc",
            "expert_calibrated",
            "expert_support_penalized",
            "expert_support_penalized_conservative_gate",
            "expert_calibrated_support_penalized",
            "expert_calibrated_support_penalized_conservative_gate",
            "high_reward_ablation",
            "random",
            "oracle",
            "fallback_raw_ebm",
            "fallback_oracle",
        ]:
            group = [r for r in seed_rows if str(r["task"]) == task and str(r["model"]) == model and int(r["N"]) == high_n]
            if not group:
                continue
            vals = np.asarray([float(r["selected_real_utility"]) for r in group], dtype=float)
            success = np.asarray([float(r["success_rate"]) for r in group], dtype=float)
            exact = np.asarray([float(r["exact_law_abs_error"]) for r in group], dtype=float)
            rows.append(
                {
                    "benchmark": group[0].get("benchmark", "metaworld"),
                    "task": task,
                    "model": model,
                    "training_source": group[0].get("training_source", ""),
                    "N": high_n,
                    "task_status": group[0].get("task_status", ""),
                    "success_available": group[0].get("success_available", True),
                    "success_nonzero": group[0].get("success_nonzero", False),
                    "mean_selected_reward": float(np.mean(vals)),
                    "selected_reward_ci_low": float(np.mean(vals) - 1.96 * (np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)),
                    "selected_reward_ci_high": float(np.mean(vals) + 1.96 * (np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)),
                    "mean_selected_success": float(np.mean(success)),
                    "mean_high_n_regret": float(np.mean([float(r["high_n_regret"]) for r in group])),
                    "mean_contact_failure_rate": float(np.mean([float(r["contact_failure_rate"]) for r in group])),
                    "mean_support_distance": float(np.mean([float(r["support_distance"]) for r in group])),
                    "max_exact_law_abs_error": float(np.max(exact)),
                    "num_seeds": len(group),
                }
            )
    return rows


def _mean_ci(values: list[float] | np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, mean - 1.96 * se, mean + 1.96 * se


def _closed_loop_ablation_rollout(
    task_name: str,
    seed: int,
    expert_model: TorchIBCEnergy,
    proposal_model: _NearestDemoProposal,
    behavior_model: _TorchBehaviorProposal,
    num_episodes: int,
    horizon: int,
    candidates_per_step: int,
) -> list[dict[str, object]]:
    import metaworld

    rng = np.random.default_rng(seed + 17_000)
    mt1 = metaworld.MT1(task_name)
    env_cls = mt1.train_classes[task_name]
    env = env_cls()
    policy_cls = _policy_class(task_name)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    for variant in CLOSED_LOOP_VARIANTS:
        for episode in range(num_episodes):
            task = mt1.train_tasks[(seed + episode) % len(mt1.train_tasks)]
            env.set_task(task)
            obs, _info = env.reset(seed=seed + episode)
            policy = policy_cls()
            rewards: list[float] = []
            successes: list[float] = []
            selected_energy: list[float] = []
            selected_expert_distance: list[float] = []
            max_expert_distance = 0.0
            gate_fallbacks = 0
            previous_action: np.ndarray | None = None
            for _step in range(horizon):
                expert_action = _clip_action(policy.get_action(obs), action_low, action_high)
                learned_action = proposal_model.predict(obs) if bool(variant.get("uses_learned_demo_proposal", False)) else None
                behavior_action = behavior_model.predict(obs) if bool(variant.get("uses_learned_bc_proposal", False)) else None
                heuristic_action = (
                    _state_heuristic_action(task_name, obs, action_low, action_high)
                    if bool(variant.get("uses_state_heuristic_proposal", False))
                    else None
                )
                if str(variant["policy"]) == "scripted_expert":
                    action = expert_action
                    selected_value = 0.0
                    expert_distance = 0.0
                elif str(variant["policy"]) == "learned_demo_proposal_direct":
                    if learned_action is None:
                        raise ValueError("learned-demo direct variant requires a learned action")
                    action = learned_action
                    selected_value = 0.0
                    expert_distance = float(np.linalg.norm(action - expert_action))
                elif str(variant["policy"]) == "learned_bc_proposal_direct":
                    if behavior_action is None:
                        raise ValueError("behavior-cloned direct variant requires a learned action")
                    action = behavior_action
                    selected_value = 0.0
                    expert_distance = float(np.linalg.norm(action - expert_action))
                elif str(variant["policy"]) == "state_heuristic_direct":
                    if heuristic_action is None:
                        raise ValueError("state-heuristic direct variant requires a heuristic action")
                    action = heuristic_action
                    selected_value = 0.0
                    expert_distance = float(np.linalg.norm(action - expert_action))
                else:
                    candidates = _closed_loop_candidate_actions(
                        variant,
                        task_name,
                        expert_action,
                        previous_action,
                        learned_action,
                        behavior_action,
                        heuristic_action,
                        rng,
                        action_low,
                        action_high,
                        candidates_per_step,
                    )
                    actions = np.vstack(candidates).astype(np.float32)
                    if bool(variant["uses_learned_ebm"]):
                        obs_batch = np.repeat(np.asarray(obs, dtype=np.float32)[None, :], len(actions), axis=0)
                        energy = expert_model.energy(obs_batch, actions)
                        best = int(np.argmin(energy))
                        action = _clip_action(actions[best], action_low, action_high)
                        selected_value = float(energy[best])
                    else:
                        best = int(rng.integers(0, len(actions)))
                        action = _clip_action(actions[best], action_low, action_high)
                        selected_value = 0.0
                    expert_distance = float(np.linalg.norm(action - expert_action))
                    if bool(variant["conservative_proposal_gate"]) and expert_distance > CLOSED_LOOP_GATE_DISTANCE:
                        action = expert_action
                        selected_value = float(expert_model.energy(np.asarray(obs, dtype=np.float32)[None, :], action[None, :])[0])
                        expert_distance = 0.0
                        gate_fallbacks += 1
                obs, reward, terminated, truncated, info = env.step(action)
                previous_action = action
                rewards.append(float(reward))
                success = float(info.get("success", 0.0))
                successes.append(success)
                selected_energy.append(selected_value)
                selected_expert_distance.append(expert_distance)
                max_expert_distance = max(max_expert_distance, expert_distance)
                if terminated or truncated or success > 0.5:
                    break
            rows.append(
                {
                    "benchmark": "metaworld",
                    "task": task_name,
                    "seed": seed,
                    "episode": episode,
                    "policy": str(variant["policy"]),
                    "proposal_source": str(variant["proposal_source"]),
                    "uses_expert_centered_proposals": bool(variant["uses_expert_centered_proposals"]),
                    "conservative_proposal_gate": bool(variant["conservative_proposal_gate"]),
                    "uses_learned_ebm": bool(variant["uses_learned_ebm"]),
                    "uses_learned_demo_proposal": bool(variant.get("uses_learned_demo_proposal", False)),
                    "uses_learned_bc_proposal": bool(variant.get("uses_learned_bc_proposal", False)),
                    "uses_state_heuristic_proposal": bool(variant.get("uses_state_heuristic_proposal", False)),
                    "expert_action_available": True,
                    "runtime_scripted_expert_dependency": bool(
                        str(variant["policy"]) == "scripted_expert"
                        or bool(variant["uses_expert_centered_proposals"])
                        or bool(variant["conservative_proposal_gate"])
                    ),
                    "gate_fallback_rate": float(gate_fallbacks / max(len(rewards), 1)),
                    "continuation": "none",
                    "horizon": horizon,
                    "steps": len(rewards),
                    "num_candidates_per_step": candidates_per_step,
                    "total_reward": float(np.sum(rewards)),
                    "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                    "success_available": True,
                    "success_rate": float(np.max(successes)) if successes else 0.0,
                    "success_nonzero": bool(successes and np.max(successes) > 0.0),
                    "mean_selected_energy": float(np.mean(selected_energy)) if selected_energy else 0.0,
                    "mean_selected_expert_action_distance": float(np.mean(selected_expert_distance)) if selected_expert_distance else 0.0,
                    "max_selected_expert_action_distance": float(max_expert_distance),
                    "runtime_seconds": float(time.perf_counter() - start),
                    "claim_boundary": "Closed-loop dependency audit; variants report proposal centering and gate dependence and do not support real-robot claims.",
                }
            )
    return rows


def _closed_loop_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for task in sorted({str(r["task"]) for r in rows}):
        task_rows = [r for r in rows if str(r["task"]) == task]
        by_policy = {str(r["policy"]): [x for x in task_rows if str(x["policy"]) == str(r["policy"])] for r in task_rows}
        gate_group = by_policy.get("expert_centered_gate", [])
        no_gate_group = by_policy.get("expert_centered_no_gate", [])
        local_group = by_policy.get("local_gaussian_no_gate", [])
        gate_success = float(np.mean([float(r["success_rate"]) for r in gate_group])) if gate_group else 0.0
        no_gate_success = float(np.mean([float(r["success_rate"]) for r in no_gate_group])) if no_gate_group else 0.0
        local_success = float(np.mean([float(r["success_rate"]) for r in local_group])) if local_group else 0.0
        gate_reward = float(np.mean([float(r["total_reward"]) for r in gate_group])) if gate_group else 0.0
        no_gate_reward = float(np.mean([float(r["total_reward"]) for r in no_gate_group])) if no_gate_group else 0.0
        local_reward = float(np.mean([float(r["total_reward"]) for r in local_group])) if local_group else 0.0
        fallback_dependency = float(np.mean([float(r["gate_fallback_rate"]) for r in gate_group])) if gate_group else 0.0
        dependency = {
            "success_drop_without_gate": float(gate_success - no_gate_success),
            "success_drop_without_expert_centering": float(gate_success - local_success),
            "reward_drop_without_gate": float(gate_reward - no_gate_reward),
            "reward_drop_without_expert_centering": float(gate_reward - local_reward),
            "fallback_dependency_score": fallback_dependency,
        }
        for policy_name in [str(v["policy"]) for v in CLOSED_LOOP_VARIANTS]:
            group = by_policy.get(policy_name, [])
            if not group:
                continue
            successes = [float(r["success_rate"]) for r in group]
            rewards = [float(r["total_reward"]) for r in group]
            success_mean, success_low, success_high = _mean_ci(successes)
            reward_mean, reward_low, reward_high = _mean_ci(rewards)
            fallback_values = [float(r["gate_fallback_rate"]) for r in group]
            distance_values = [float(r["mean_selected_expert_action_distance"]) for r in group]
            max_distance_values = [float(r["max_selected_expert_action_distance"]) for r in group]
            out.append(
                {
                    "benchmark": "metaworld",
                    "task": task,
                    "policy": policy_name,
                    "proposal_source": group[0].get("proposal_source", ""),
                    "uses_expert_centered_proposals": group[0].get("uses_expert_centered_proposals", False),
                    "conservative_proposal_gate": group[0].get("conservative_proposal_gate", False),
                    "uses_learned_ebm": group[0].get("uses_learned_ebm", False),
                    "uses_learned_demo_proposal": group[0].get("uses_learned_demo_proposal", False),
                    "uses_learned_bc_proposal": group[0].get("uses_learned_bc_proposal", False),
                    "uses_state_heuristic_proposal": group[0].get("uses_state_heuristic_proposal", False),
                    "expert_action_available": group[0].get("expert_action_available", True),
                    "runtime_scripted_expert_dependency": group[0].get("runtime_scripted_expert_dependency", True),
                    "mean_gate_fallback_rate": float(np.mean(fallback_values)) if fallback_values else 0.0,
                    "continuation": group[0].get("continuation", ""),
                    "num_rollouts": len(group),
                    "num_seeds": len({int(r["seed"]) for r in group}),
                    "mean_total_reward": reward_mean,
                    "total_reward_ci_low": reward_low,
                    "total_reward_ci_high": reward_high,
                    "mean_success_rate": success_mean,
                    "success_rate_ci_low": success_low,
                    "success_rate_ci_high": success_high,
                    "success_nonzero": bool(successes and np.max(successes) > 0.0),
                    "mean_selected_expert_action_distance": float(np.mean(distance_values)) if distance_values else 0.0,
                    "max_selected_expert_action_distance": float(np.max(max_distance_values)) if max_distance_values else 0.0,
                    **dependency,
                }
            )
    return out


def _closed_loop_policy_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for policy_name in [str(v["policy"]) for v in CLOSED_LOOP_VARIANTS]:
        group = [r for r in rows if str(r.get("policy")) == policy_name]
        if not group:
            continue
        successes = np.asarray([float(r["success_rate"]) for r in group], dtype=float)
        rewards = np.asarray([float(r["total_reward"]) for r in group], dtype=float)
        success_mean, success_low, success_high = _mean_ci(successes)
        reward_mean, reward_low, reward_high = _mean_ci(rewards)
        out.append(
            {
                "benchmark": "metaworld",
                "policy": policy_name,
                "proposal_source": group[0].get("proposal_source", ""),
                "uses_expert_centered_proposals": group[0].get("uses_expert_centered_proposals", False),
                "conservative_proposal_gate": group[0].get("conservative_proposal_gate", False),
                "uses_learned_ebm": group[0].get("uses_learned_ebm", False),
                "uses_learned_demo_proposal": group[0].get("uses_learned_demo_proposal", False),
                "uses_learned_bc_proposal": group[0].get("uses_learned_bc_proposal", False),
                "uses_state_heuristic_proposal": group[0].get("uses_state_heuristic_proposal", False),
                "runtime_scripted_expert_dependency": group[0].get("runtime_scripted_expert_dependency", True),
                "mean_gate_fallback_rate": float(np.mean([float(r["gate_fallback_rate"]) for r in group])),
                "continuation": group[0].get("continuation", ""),
                "num_rollouts": len(group),
                "num_seeds": len({int(r["seed"]) for r in group}),
                "num_tasks": len({str(r["task"]) for r in group}),
                "mean_total_reward": reward_mean,
                "total_reward_ci_low": reward_low,
                "total_reward_ci_high": reward_high,
                "mean_success_rate": success_mean,
                "success_rate_ci_low": success_low,
                "success_rate_ci_high": success_high,
                "success_nonzero": bool(len(successes) and np.max(successes) > 0.0),
                "mean_selected_expert_action_distance": float(
                    np.mean([float(r["mean_selected_expert_action_distance"]) for r in group])
                ),
            }
        )
    return out


def _closed_loop_dependency_payload(summary_rows: list[dict[str, object]], task_names: list[str]) -> dict[str, object]:
    required_variants = {str(v["policy"]) for v in CLOSED_LOOP_VARIANTS}
    present_variants = {str(r.get("policy")) for r in summary_rows}
    present_tasks = {str(r.get("task")) for r in summary_rows}

    def _all_tasks_meet(
        policy: str,
        threshold: float,
        require_no_runtime_scripted_expert: bool = False,
    ) -> bool:
        rows = [r for r in summary_rows if str(r.get("policy")) == policy]
        if {str(r.get("task")) for r in rows} != set(task_names):
            return False
        return all(
            float(r.get("mean_success_rate", 0.0)) >= threshold
            and str(r.get("success_nonzero")).lower() == "true"
            and (not require_no_runtime_scripted_expert or str(r.get("runtime_scripted_expert_dependency")).lower() == "false")
            for r in rows
        )

    learned_demo_ebm_success_threshold_met = _all_tasks_meet(
        "learned_demo_proposal_no_gate", 0.5, require_no_runtime_scripted_expert=True
    )
    learned_demo_direct_success_threshold_met = _all_tasks_meet(
        "learned_demo_proposal_direct", 0.5, require_no_runtime_scripted_expert=True
    )
    learned_bc_ebm_success_threshold_met = _all_tasks_meet(
        "learned_bc_proposal_no_gate", 0.5, require_no_runtime_scripted_expert=True
    )
    learned_bc_direct_success_threshold_met = _all_tasks_meet(
        "learned_bc_proposal_direct", 0.5, require_no_runtime_scripted_expert=True
    )
    learned_ebm_low_dependency_success_threshold_met = (
        learned_demo_ebm_success_threshold_met or learned_bc_ebm_success_threshold_met
    )
    learned_policy_low_dependency_success_threshold_met = (
        learned_demo_direct_success_threshold_met or learned_bc_direct_success_threshold_met
    )
    state_heuristic_ebm_success_threshold_met = _all_tasks_meet(
        "state_heuristic_proposal_no_gate", 0.5, require_no_runtime_scripted_expert=True
    )
    state_heuristic_direct_success_threshold_met = _all_tasks_meet(
        "state_heuristic_direct", 0.5, require_no_runtime_scripted_expert=True
    )
    local_gaussian_success_threshold_met = _all_tasks_meet("local_gaussian_no_gate", 0.5, require_no_runtime_scripted_expert=True)
    low_dependency_success_threshold_met = (
        learned_ebm_low_dependency_success_threshold_met
        or learned_policy_low_dependency_success_threshold_met
        or state_heuristic_ebm_success_threshold_met
        or state_heuristic_direct_success_threshold_met
        or local_gaussian_success_threshold_met
    )
    autonomous_success_threshold_met = low_dependency_success_threshold_met
    return {
        "experiment": "closed_loop_dependency_audit",
        "status": "SUPPORTED" if required_variants.issubset(present_variants) and set(task_names).issubset(present_tasks) else "UNSUPPORTED",
        "required_variants": sorted(required_variants),
        "present_variants": sorted(present_variants),
        "tasks": task_names,
        "low_dependency_success_threshold": 0.5,
        "low_dependency_success_threshold_met": bool(low_dependency_success_threshold_met),
        "learned_demo_ebm_success_threshold_met": bool(learned_demo_ebm_success_threshold_met),
        "learned_demo_direct_success_threshold_met": bool(learned_demo_direct_success_threshold_met),
        "learned_bc_ebm_success_threshold_met": bool(learned_bc_ebm_success_threshold_met),
        "learned_bc_direct_success_threshold_met": bool(learned_bc_direct_success_threshold_met),
        "learned_ebm_low_dependency_success_threshold_met": bool(learned_ebm_low_dependency_success_threshold_met),
        "learned_policy_low_dependency_success_threshold_met": bool(learned_policy_low_dependency_success_threshold_met),
        "state_heuristic_ebm_success_threshold_met": bool(state_heuristic_ebm_success_threshold_met),
        "state_heuristic_direct_success_threshold_met": bool(state_heuristic_direct_success_threshold_met),
        "local_gaussian_success_threshold_met": bool(local_gaussian_success_threshold_met),
        "autonomous_success_threshold_met": bool(autonomous_success_threshold_met),
        "claim_boundary": "This audit measures dependence on expert-centered proposals, nearest-demo learned proposals, behavior-cloned learned proposals, state heuristics, and fallback gates; it does not by itself claim real-robot or broad manipulation success.",
        "summary_rows": summary_rows,
    }


def _partial_task_seed_result(task_name: str, seed: int, smoke: bool, error: str) -> dict[str, object]:
    settings = _metaworld_settings(smoke)
    seed_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    _fallback_rows_for_task(
        task_name,
        seed,
        settings["fallback_pool_n"],
        settings["mc_trials"],
        seed_rows,
        reliability_rows,
        error,
    )
    return {
        "runner_version": METAWORLD_RUNNER_VERSION,
        "runner_settings": settings,
        "task": task_name,
        "seed": seed,
        "task_status": "PARTIAL",
        "task_error": error,
        "seed_rows": seed_rows,
        "reliability_rows": reliability_rows,
        "closed_loop_rows": [],
        "metadata": {"task": task_name, "seed": seed, "task_status": "PARTIAL", "error": error, "fallback": "contact_push"},
    }


def _run_task_seed(task_name: str, seed: int, smoke: bool) -> dict[str, object]:
    settings = _metaworld_settings(smoke)
    seed_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    closed_loop_rows: list[dict[str, object]] = []
    try:
        print(f"metaworld_benchmark task={task_name} seed={seed} demos", flush=True)
        demo_obs, demo_actions, demo_meta = _collect_scripted_demos(
            task_name,
            seed + 300,
            settings["demo_tasks"],
            settings["demo_horizon"],
        )
        print(f"metaworld_benchmark task={task_name} seed={seed} pools", flush=True)
        train_pool, train_meta = _collect_candidate_pool(
            task_name,
            seed + 500,
            settings["train_states"],
            settings["train_candidates"],
            settings["max_advance"],
            continuation_horizon=1,
        )
        eval_pool, eval_meta = _collect_candidate_pool(
            task_name,
            seed + 800,
            settings["eval_states"],
            settings["eval_candidates"],
            settings["max_advance"],
            continuation_horizon=settings["eval_continuation_horizon"],
        )
        action_low = np.asarray(train_meta["action_low"], dtype=np.float32)
        action_high = np.asarray(train_meta["action_high"], dtype=np.float32)
        print(f"metaworld_benchmark task={task_name} seed={seed} expert_model", flush=True)
        expert_model = TorchIBCEnergy.fit(
            demo_obs,
            demo_actions,
            seed=seed,
            epochs=settings["epochs"],
            batch_size=64,
            negatives=settings["negatives"],
            lr=2e-3,
            hidden_dim=96 if smoke else 128,
            feature_mode="raw",
            action_low=action_low,
            action_high=action_high,
        )
        proposal_model = _NearestDemoProposal(demo_obs, demo_actions, action_low, action_high)
        behavior_obs, behavior_actions = _behavior_proposal_training_set(demo_obs, demo_actions, train_pool)
        behavior_model = _TorchBehaviorProposal.fit(
            behavior_obs,
            behavior_actions,
            action_low,
            action_high,
            seed=seed + 47,
            epochs=settings["behavior_proposal_epochs"],
            batch_size=96,
            hidden_dim=96 if smoke else 128,
            lr=2e-3,
        )
        pos_obs, pos_actions = _positives_from_pool(train_pool, quantile=0.82)
        print(f"metaworld_benchmark task={task_name} seed={seed} high_reward_ablation", flush=True)
        high_reward_model = TorchIBCEnergy.fit(
            pos_obs,
            pos_actions,
            seed=seed + 31,
            epochs=max(8, settings["epochs"] // 2),
            batch_size=64,
            negatives=settings["negatives"],
            lr=2e-3,
            hidden_dim=64 if smoke else 96,
            feature_mode="raw",
            action_low=action_low,
            action_high=action_high,
        )
        print(f"metaworld_benchmark task={task_name} seed={seed} curves", flush=True)
        expert_energy = expert_model.energy(eval_pool["observations"], eval_pool["actions"])
        high_reward_energy = high_reward_model.energy(eval_pool["observations"], eval_pool["actions"])
        calibrator = fit_energy_calibrator(
            eval_pool,
            expert_energy,
            pilot_size=min(192 if not smoke else 64, max(8, len(expert_energy) // 2)),
            seed=seed,
        )
        expert_calibrated = apply_energy_calibrator(eval_pool, expert_energy, calibrator)
        oracle = oracle_energy(eval_pool)
        success_nonzero = bool(eval_meta["success_nonzero"])
        status = "SUPPORTED" if success_nonzero else "PARTIAL"
        energies = {
            "expert_ibc": (expert_energy, "scripted_expert"),
            "high_reward_ablation": (high_reward_energy, "high_reward_ablation"),
            "expert_calibrated": (expert_calibrated, "small_pilot_labels"),
            "expert_support_penalized": (support_penalized_energy(expert_energy, eval_pool, weight=0.4), "no_utility_labels"),
            "expert_calibrated_support_penalized": (
                support_penalized_energy(expert_calibrated, eval_pool, weight=0.4),
                "small_pilot_labels_plus_support",
            ),
            "value_shaped_upper": (value_shaped_energy(expert_energy, eval_pool), "upper_bound"),
            "oracle_value_shaped_upper": (value_shaped_energy(oracle, eval_pool), "upper_bound"),
            "random": (np.zeros_like(expert_energy), "none"),
            "oracle": (oracle, "upper_bound"),
        }
        _add_curve_rows(
            seed_rows,
            reliability_rows,
            eval_pool,
            energies,
            task_name,
            seed,
            status,
            "" if success_nonzero else "success metric was present but zero after scripted continuation",
            success_available=bool(eval_meta["success_available"]),
            success_nonzero=success_nonzero,
            mc_trials=settings["mc_trials"],
        )
        print(f"metaworld_benchmark task={task_name} seed={seed} closed_loop_ablations", flush=True)
        closed_loop_rows.extend(
            _closed_loop_ablation_rollout(
                task_name,
                seed,
                expert_model,
                proposal_model,
                behavior_model,
                num_episodes=1,
                horizon=settings["closed_loop_horizon"],
                candidates_per_step=settings["closed_loop_candidates_per_step"],
            )
        )
        metadata = {
            "task": task_name,
            "seed": seed,
            "task_status": status,
            "demo": {**demo_meta, "model_metadata": expert_model.metadata()},
            "learned_demo_proposal_metadata": proposal_model.metadata(),
            "behavior_cloned_proposal_metadata": behavior_model.metadata(),
            "high_reward_ablation_metadata": high_reward_model.metadata(),
            "train": train_meta,
            "eval": eval_meta,
        }
        return {
            "runner_version": METAWORLD_RUNNER_VERSION,
            "runner_settings": settings,
            "task": task_name,
            "seed": seed,
            "task_status": status,
            "task_error": "" if success_nonzero else "success metric was present but zero after scripted continuation",
            "seed_rows": seed_rows,
            "reliability_rows": reliability_rows,
            "closed_loop_rows": closed_loop_rows,
            "metadata": metadata,
        }
    except Exception as exc:
        return _partial_task_seed_result(task_name, seed, smoke, f"{type(exc).__name__}: {exc}")
    finally:
        gc.collect()


def _run_task_seed_subprocess(task_name: str, seed: int, smoke: bool, output_path: Path) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    settings = _metaworld_settings(smoke)
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            if (
                cached.get("runner_version") == METAWORLD_RUNNER_VERSION
                and cached.get("runner_settings") == settings
                and cached.get("task") == task_name
                and int(cached.get("seed", -1)) == seed
                and cached.get("seed_rows")
            ):
                print(f"metaworld_benchmark task={task_name} seed={seed} resume_existing", flush=True)
                return cached
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        output_path.unlink()
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-task",
        task_name,
        "--single-seed",
        str(seed),
        "--single-task-seed-output",
        str(output_path),
    ]
    if smoke:
        args.append("--smoke")
    last_result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, 4):
        print(f"metaworld_benchmark task={task_name} seed={seed} subprocess attempt={attempt}", flush=True)
        result = subprocess.run(args, env=env, text=True, capture_output=True)
        last_result = result
        if result.stdout:
            print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr, flush=True)
        if result.returncode == 0 and output_path.exists():
            break
        time.sleep(8.0 * attempt)
    else:
        assert last_result is not None
        error = (
            f"SubprocessFailed: returncode={last_result.returncode}; "
            f"stderr={last_result.stderr.strip()[:500]}"
        )
        return _partial_task_seed_result(task_name, seed, smoke, error)
    time.sleep(2.0)
    with output_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _consume_task_seed_result(
    result: dict[str, object],
    task_seed_status: dict[str, list[str]],
    task_seed_errors: dict[str, list[str]],
    seed_rows: list[dict[str, object]],
    reliability_rows: list[dict[str, object]],
    closed_loop_rows: list[dict[str, object]],
    metadata_rows: list[dict[str, object]],
) -> None:
    task_name = str(result.get("task", "unknown"))
    seed = int(result.get("seed", -1))
    task_seed_status.setdefault(task_name, []).append(str(result.get("task_status", "PARTIAL")))
    error = str(result.get("task_error", ""))
    if error:
        task_seed_errors.setdefault(task_name, []).append(f"seed={seed}: {error}")
    seed_rows.extend(list(result.get("seed_rows", [])))
    reliability_rows.extend(list(result.get("reliability_rows", [])))
    closed_loop_rows.extend(list(result.get("closed_loop_rows", [])))
    metadata_rows.append(dict(result.get("metadata", {})))


def run(smoke: bool = False) -> dict[str, object]:
    out_dir = Path("results") / "benchmarks" / "metaworld"
    reliability_dir = Path("results") / "reliability"
    seeds = [0] if smoke else [0, 1, 2, 3, 4]
    task_names = ["reach-v3", "push-v3", "pick-place-v3", "button-press-v3"]
    seed_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    closed_loop_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    task_status: dict[str, str] = {}
    task_errors: dict[str, str] = {}
    task_seed_status: dict[str, list[str]] = {task: [] for task in task_names}
    task_seed_errors: dict[str, list[str]] = {task: [] for task in task_names}

    if smoke:
        for task_name in task_names:
            for seed in seeds:
                _consume_task_seed_result(
                    _run_task_seed(task_name, seed, smoke=True),
                    task_seed_status,
                    task_seed_errors,
                    seed_rows,
                    reliability_rows,
                    closed_loop_rows,
                    metadata_rows,
                )
    else:
        max_workers = max(1, int(os.environ.get("METAWORLD_PARALLEL_JOBS", "2")))
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for task_name in task_names:
                for seed in seeds:
                    output_path = out_dir / "tmp" / f"{task_name.replace('-', '_')}_seed_{seed}.json"
                    futures[
                        executor.submit(
                            _run_task_seed_subprocess,
                            task_name,
                            seed,
                            False,
                            output_path,
                        )
                    ] = (task_name, seed)
            for future in as_completed(futures):
                task_name, seed = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = _partial_task_seed_result(task_name, seed, smoke=False, error=f"{type(exc).__name__}: {exc}")
                _consume_task_seed_result(
                    result,
                    task_seed_status,
                    task_seed_errors,
                    seed_rows,
                    reliability_rows,
                    closed_loop_rows,
                    metadata_rows,
                )

    for task_name in task_names:
        statuses = task_seed_status[task_name]
        task_status[task_name] = "SUPPORTED" if statuses and all(s == "SUPPORTED" for s in statuses) else "PARTIAL"
        task_errors[task_name] = "; ".join(task_seed_errors[task_name])

    seed_rows.extend(
        make_conservative_gated_rows(
            seed_rows,
            "expert_support_penalized",
            "expert_support_penalized_conservative_gate",
        )
    )
    seed_rows.extend(
        make_conservative_gated_rows(
            seed_rows,
            "expert_calibrated_support_penalized",
            "expert_calibrated_support_penalized_conservative_gate",
        )
    )
    annotate_repair_effectiveness(
        seed_rows,
        "expert_ibc",
        "oracle",
        {
            "expert_calibrated",
            "expert_support_penalized",
            "expert_support_penalized_conservative_gate",
            "expert_calibrated_support_penalized",
            "expert_calibrated_support_penalized_conservative_gate",
            "value_shaped_upper",
            "oracle_value_shaped_upper",
            "oracle",
        },
    )
    annotate_repair_effectiveness(
        seed_rows,
        "fallback_raw_ebm",
        "fallback_oracle",
        {"fallback_support_penalized", "fallback_value_shaped", "fallback_oracle"},
    )
    summary_rows = aggregate_curve_rows(seed_rows)
    task_summary_rows = _aggregate_task_rows(seed_rows)
    table_rows = _benchmark_table(seed_rows, high_n=max(METAWORLD_N_VALUES))
    closed_loop_summary_rows = _closed_loop_summary(closed_loop_rows)
    closed_loop_policy_summary_rows = _closed_loop_policy_summary(closed_loop_rows)
    closed_loop_dependency = _closed_loop_dependency_payload(closed_loop_summary_rows, task_names)
    write_csv(out_dir / "seed_level.csv", seed_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "task_summary.csv", task_summary_rows)
    write_csv(out_dir / "task_table.csv", table_rows)
    write_csv(out_dir / "closed_loop_rollouts.csv", closed_loop_rows)
    write_csv(out_dir / "closed_loop_summary.csv", closed_loop_summary_rows)
    write_csv(out_dir / "closed_loop_ablation_rollouts.csv", closed_loop_rows)
    write_csv(out_dir / "closed_loop_ablation_summary.csv", closed_loop_summary_rows)
    write_csv(out_dir / "closed_loop_ablation_policy_summary.csv", closed_loop_policy_summary_rows)
    write_csv(reliability_dir / "summary.csv", reliability_rows)

    statuses = set(task_status.values())
    overall_status = "SUPPORTED" if statuses == {"SUPPORTED"} else ("PARTIAL" if seed_rows else "UNSUPPORTED")
    expert_high = [r for r in seed_rows if r.get("model") == "expert_ibc" and int(r["N"]) == max(METAWORLD_N_VALUES)]
    failure_candidates = [r for r in seed_rows if int(r["N"]) == max(METAWORLD_N_VALUES) and str(r["model"]) in {"expert_ibc", "high_reward_ablation"}]
    strongest_failure = max(failure_candidates, key=lambda r: float(r["high_n_regret"])) if failure_candidates else {}
    closed_loop_success_nonzero = bool(any(bool(r.get("success_nonzero")) for r in closed_loop_summary_rows))
    payload = {
        "experiment": "metaworld_benchmark",
        "benchmark": "metaworld",
        "tasks": task_names,
        "status": overall_status,
        "task_status": task_status,
        "task_errors": task_errors,
        "smoke": smoke,
        "n_values": METAWORLD_N_VALUES,
        "seeds": seeds,
        "metadata": metadata_rows,
        "primary_training_source": "scripted_expert",
        "high_reward_sampled_actions_role": "ablation_only",
        "success_claim_note": "Benchmark claims use selected-action plus scripted-continuation reward and success; zero-success tasks are downgraded to PARTIAL.",
        "closed_loop_learned_ebm_rollout_diagnostic": {
            "status": "SUPPORTED" if closed_loop_rows else "UNSUPPORTED",
            "policy": "closed_loop_dependency_ablation_ladder",
            "proposal_source": "expert_centered_learned_demo_behavior_cloned_state_heuristic_local_random_and_scripted_variants",
            "conservative_proposal_gate": "variant_specific",
            "continuation": "none",
            "num_rollouts": len(closed_loop_rows),
            "success_nonzero": closed_loop_success_nonzero,
            "summary": closed_loop_summary_rows,
            "policy_summary": closed_loop_policy_summary_rows,
            "dependency_audit": closed_loop_dependency,
            "claim_boundary": "Diagnostic only; variants expose expert-centering, learned-demo proposal, behavior-cloned proposal, state-heuristic proposal, and fallback-gate dependence and do not support real-robot claims.",
        },
        "closed_loop_dependency_audit": closed_loop_dependency,
        "strongest_benchmark_failure_artifact": strongest_failure,
        "strongest_expert_artifact": expert_high[0] if expert_high else {},
        "artifacts": {
            "summary_csv": str(out_dir / "summary.csv"),
            "task_summary_csv": str(out_dir / "task_summary.csv"),
            "task_table_csv": str(out_dir / "task_table.csv"),
            "closed_loop_rollouts_csv": str(out_dir / "closed_loop_rollouts.csv"),
            "closed_loop_summary_csv": str(out_dir / "closed_loop_summary.csv"),
            "closed_loop_ablation_rollouts_csv": str(out_dir / "closed_loop_ablation_rollouts.csv"),
            "closed_loop_ablation_summary_csv": str(out_dir / "closed_loop_ablation_summary.csv"),
            "closed_loop_ablation_policy_summary_csv": str(out_dir / "closed_loop_ablation_policy_summary.csv"),
            "closed_loop_ablation_summary_json": str(out_dir / "closed_loop_ablation_summary.json"),
            "seed_level_csv": str(out_dir / "seed_level.csv"),
            "reliability_csv": str(reliability_dir / "summary.csv"),
            "summary_json": str(out_dir / "summary.json"),
        },
    }
    write_json(out_dir / "summary.json", payload)
    write_json(out_dir / "closed_loop_ablation_summary.json", closed_loop_dependency)
    write_json(reliability_dir / "summary.json", {"experiment": "energy_reliability", "rows": len(reliability_rows), "artifacts": {"summary_csv": str(reliability_dir / "summary.csv")}})
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--single-task", type=str, default=None)
    parser.add_argument("--single-seed", type=int, default=None)
    parser.add_argument("--single-task-seed-output", type=Path, default=None)
    args = parser.parse_args()
    if args.single_task is not None or args.single_seed is not None:
        if args.single_task is None or args.single_seed is None or args.single_task_seed_output is None:
            raise SystemExit("--single-task, --single-seed, and --single-task-seed-output are required together")
        write_json(args.single_task_seed_output, _run_task_seed(args.single_task, args.single_seed, smoke=args.smoke))
    else:
        run(smoke=args.smoke)
