"""Energy functions and a tiny NumPy IBC-style EBM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .toy_envs import ArrayDict, sample_ibc_negatives


def oracle_energy(pool: ArrayDict) -> np.ndarray:
    """Energy that is exactly negative real utility."""

    return -np.asarray(pool["utility"], dtype=float)


def good_tail_aligned_energy(pool: ArrayDict, seed: int = 0, noise_scale: float = 0.08) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_scale, size=len(pool["utility"]))
    return -pool["utility"] + noise


def raw_miscalibrated_energy(pool: ArrayDict, seed: int = 0) -> np.ndarray:
    """Energy with plausible average signal but bad low-energy tail.

    It rewards visible task progress, low force, and smoothness while ignoring
    hidden contact and support validity. The shortcut term makes invalid actions
    spuriously low energy in the selected tail.
    """

    rng = np.random.default_rng(seed)
    visible = np.asarray(pool["visible_score"], dtype=float)
    support = np.asarray(pool["support_penalty"], dtype=float)
    shortcut = np.asarray(pool["shortcut_score"], dtype=float)
    action_norm = np.asarray(pool["action_norm"], dtype=float)
    noise = rng.normal(0.0, 0.025, size=len(visible))
    return -1.10 * visible + 0.10 * support - 0.72 * shortcut + 0.025 * action_norm + noise


def smoothness_blind_energy(pool: ArrayDict, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return -1.05 * pool["visible_score"] + 0.04 * pool["support_penalty"] + rng.normal(
        0.0, 0.035, size=len(pool["utility"])
    )


def ood_low_energy(pool: ArrayDict, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return raw_miscalibrated_energy(pool, seed=seed) - 0.48 * pool["support_penalty"] + rng.normal(
        0.0, 0.02, size=len(pool["utility"])
    )


def shortcut_energy(pool: ArrayDict, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return -1.0 * pool["visible_score"] - 1.05 * pool["shortcut_score"] + rng.normal(
        0.0, 0.02, size=len(pool["utility"])
    )


def support_penalized_energy(raw_energy: np.ndarray, pool: ArrayDict, weight: float = 0.85) -> np.ndarray:
    return np.asarray(raw_energy, dtype=float) + weight * np.asarray(pool["support_penalty"], dtype=float)


def value_shaped_energy(raw_energy: np.ndarray, pool: ArrayDict) -> np.ndarray:
    """Add explicit toy validity penalties to the raw energy."""

    return (
        np.asarray(raw_energy, dtype=float)
        + 1.35 * np.asarray(pool["contact_penalty"], dtype=float)
        + 0.42 * np.asarray(pool["jerk_penalty"], dtype=float)
        + 0.70 * np.asarray(pool["support_penalty"], dtype=float)
    )


def energy_summary(name: str, energy: np.ndarray) -> dict[str, float | str]:
    arr = np.asarray(energy, dtype=float)
    return {
        "name": name,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def ibc_features(obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Feature map used by the small contrastive EBM.

    The feature map has enough structure to learn multimodal action support, but
    it does not observe the hidden obstacle/contact mode. That omission is the
    honest shortcut that the learned energy can over-optimize at high N.
    """

    obs = np.asarray(obs, dtype=float)
    actions = np.asarray(actions, dtype=float)
    target_y = obs[:, 0]
    x = actions[:, 0]
    y = actions[:, 1]
    force = actions[:, 2]
    jerk = actions[:, 3]
    broad_goal = np.exp(-0.5 * ((y - target_y) / 0.35) ** 2)
    low_effort = np.exp(-0.5 * (x / 1.0) ** 2)
    smooth = np.exp(-0.5 * (jerk / 0.35) ** 2)
    mild_force = np.exp(-0.5 * ((force - 0.35) / 0.45) ** 2)
    return np.column_stack(
        [
            np.ones(len(actions)),
            broad_goal,
            low_effort,
            smooth,
            mild_force,
            -(y - target_y) ** 2,
            -jerk * jerk,
            -force * force,
        ]
    ).astype(float)


@dataclass
class NumpyIBCEnergy:
    """Tiny linear-feature EBM trained with an InfoNCE-style contrastive loss."""

    weights: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    train_loss: list[float]

    @classmethod
    def fit(
        cls,
        obs: np.ndarray,
        actions: np.ndarray,
        seed: int = 0,
        epochs: int = 180,
        batch_size: int = 64,
        negatives: int = 24,
        lr: float = 0.09,
        l2: float = 0.002,
    ) -> "NumpyIBCEnergy":
        rng = np.random.default_rng(seed)
        base_features = ibc_features(obs, actions)
        mean = np.mean(base_features, axis=0)
        scale = np.std(base_features, axis=0) + 1e-6
        mean[0] = 0.0
        scale[0] = 1.0
        weights = rng.normal(0.0, 0.03, size=base_features.shape[1])
        losses: list[float] = []
        n = len(obs)

        for _ in range(epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                o_batch = obs[idx]
                pos = actions[idx]
                neg = sample_ibc_negatives(o_batch, negatives, rng)
                all_actions = np.concatenate([pos[:, None, :], neg], axis=1)
                flat_obs = np.repeat(o_batch, negatives + 1, axis=0)
                flat_actions = all_actions.reshape(-1, actions.shape[1])
                phi = (ibc_features(flat_obs, flat_actions) - mean) / scale
                phi = phi.reshape(len(idx), negatives + 1, -1)
                logits = phi @ weights
                logits = logits - np.max(logits, axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                loss = -np.mean(np.log(probs[:, 0] + 1e-12)) + 0.5 * l2 * float(np.sum(weights * weights))
                grad = np.mean(np.sum(probs[:, :, None] * phi, axis=1) - phi[:, 0, :], axis=0)
                grad += l2 * weights
                weights -= lr * grad
                epoch_loss += float(loss)
                batches += 1
            losses.append(epoch_loss / max(batches, 1))
        return cls(weights=weights, feature_mean=mean, feature_scale=scale, train_loss=losses)

    def score(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        phi = (ibc_features(obs, actions) - self.feature_mean) / self.feature_scale
        return phi @ self.weights

    def energy(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return -self.score(obs, actions)

    def metadata(self) -> dict[str, object]:
        return {
            "model_type": "NumPy linear-feature EBM",
            "training_objective": "InfoNCE-style contrastive classification over one expert action and K negatives",
            "negative_sampling": "Broad continuous action negatives; narrow obstacle-shortcut negatives intentionally undercovered",
            "num_features": int(len(self.weights)),
            "final_train_loss": float(self.train_loss[-1]),
        }


def mse_regression_policy_utility(obs: np.ndarray, demo_actions: np.ndarray) -> dict[str, float]:
    """Evaluate a simple explicit regression baseline for multimodal demos."""

    mean_offsets = np.mean(demo_actions - np.column_stack([np.zeros(len(demo_actions)), obs[:, 0], np.zeros(len(demo_actions)), np.zeros(len(demo_actions))]), axis=0)
    pred = np.column_stack(
        [
            np.full(len(obs), mean_offsets[0]),
            obs[:, 0] + mean_offsets[1],
            np.full(len(obs), np.mean(demo_actions[:, 2])),
            np.full(len(obs), np.mean(demo_actions[:, 3])),
        ]
    )
    return {"mean_predicted_x": float(np.mean(pred[:, 0])), "mean_predicted_force": float(np.mean(pred[:, 2]))}
