"""PyTorch EBM models used by the stronger learned-policy experiments.

The base package intentionally keeps PyTorch out of import paths that do not
need it. This module is imported only by the neural EBM and benchmark scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .toy_envs import sample_ibc_negatives


FeatureMode = str
NegativeSampler = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]


def _stabilize_torch_runtime(torch_module) -> None:
    torch_module.set_num_threads(1)
    try:
        torch_module.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch allows interop threads to be set only before parallel work
        # starts; repeated experiment calls can safely keep the existing value.
        pass


def _raw_features(obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    return np.concatenate([obs, actions], axis=1).astype(np.float32)


def _toy_visible_features(obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Observation-limited feature map for the neural toy EBM.

    It deliberately mirrors the deployment-visible cues used by the NumPy IBC
    model. The hidden obstacle/contact validity is not observed, so a neural
    scorer can still over-optimize the low-energy shortcut tail.
    """

    obs = np.asarray(obs, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
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
            broad_goal,
            low_effort,
            smooth,
            mild_force,
            -(y - target_y) ** 2,
            -jerk * jerk,
            -force * force,
            x,
            y - target_y,
        ]
    ).astype(np.float32)


def _features(obs: np.ndarray, actions: np.ndarray, mode: FeatureMode) -> np.ndarray:
    if mode == "raw":
        return _raw_features(obs, actions)
    if mode == "toy_visible":
        return _toy_visible_features(obs, actions)
    raise ValueError(f"unknown feature mode: {mode}")


def uniform_box_negatives(
    obs: np.ndarray,
    k: int,
    rng: np.random.Generator,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    low = np.asarray(action_low, dtype=np.float32)
    high = np.asarray(action_high, dtype=np.float32)
    return rng.uniform(low=low, high=high, size=(len(obs), int(k), len(low))).astype(np.float32)


@dataclass
class TorchIBCEnergy:
    """Small MLP EBM trained by contrastive classification."""

    state_dict: dict[str, object]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_mode: FeatureMode
    input_dim: int
    hidden_dim: int
    train_loss: list[float]
    action_low: np.ndarray | None = None
    action_high: np.ndarray | None = None

    @classmethod
    def fit(
        cls,
        obs: np.ndarray,
        actions: np.ndarray,
        seed: int = 0,
        epochs: int = 120,
        batch_size: int = 128,
        negatives: int = 32,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        hidden_dim: int = 96,
        feature_mode: FeatureMode = "raw",
        negative_sampler: NegativeSampler | None = None,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ) -> "TorchIBCEnergy":
        import torch

        _stabilize_torch_runtime(torch)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        base_features = _features(obs, actions, feature_mode)
        mean = np.mean(base_features, axis=0).astype(np.float32)
        scale = (np.std(base_features, axis=0) + 1e-6).astype(np.float32)
        input_dim = int(base_features.shape[1])

        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        losses: list[float] = []
        n = len(obs)
        if negative_sampler is None:
            if action_low is None or action_high is None:
                raise ValueError("action_low/action_high are required without a negative_sampler")

            def sampler(o: np.ndarray, k: int, local_rng: np.random.Generator) -> np.ndarray:
                return uniform_box_negatives(o, k, local_rng, np.asarray(action_low), np.asarray(action_high))

            negative_sampler = sampler

        mean_t = torch.tensor(mean, dtype=torch.float32)
        scale_t = torch.tensor(scale, dtype=torch.float32)

        for _epoch in range(int(epochs)):
            order = rng.permutation(n)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, n, int(batch_size)):
                idx = order[start : start + int(batch_size)]
                o_batch = obs[idx]
                pos = actions[idx]
                neg = negative_sampler(o_batch, int(negatives), rng).astype(np.float32)
                all_actions = np.concatenate([pos[:, None, :], neg], axis=1)
                flat_obs = np.repeat(o_batch, int(negatives) + 1, axis=0)
                flat_actions = all_actions.reshape(-1, actions.shape[1])
                phi = _features(flat_obs, flat_actions, feature_mode)
                phi_t = (torch.tensor(phi, dtype=torch.float32) - mean_t) / scale_t
                logits = -model(phi_t).reshape(len(idx), int(negatives) + 1)
                labels = torch.zeros(len(idx), dtype=torch.long)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.detach().cpu())
                batches += 1
            losses.append(epoch_loss / max(1, batches))

        return cls(
            state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            feature_mean=mean,
            feature_scale=scale,
            feature_mode=feature_mode,
            input_dim=input_dim,
            hidden_dim=int(hidden_dim),
            train_loss=losses,
            action_low=None if action_low is None else np.asarray(action_low, dtype=np.float32),
            action_high=None if action_high is None else np.asarray(action_high, dtype=np.float32),
        )

    def _model(self):
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )
        model.load_state_dict(self.state_dict)
        model.eval()
        return model

    def energy(self, obs: np.ndarray, actions: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        import torch

        _stabilize_torch_runtime(torch)
        phi = _features(obs, actions, self.feature_mode)
        out: list[np.ndarray] = []
        model = self._model()
        mean_t = torch.tensor(self.feature_mean, dtype=torch.float32)
        scale_t = torch.tensor(self.feature_scale, dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, len(phi), int(batch_size)):
                x = torch.tensor(phi[start : start + int(batch_size)], dtype=torch.float32)
                x = (x - mean_t) / scale_t
                out.append(model(x).reshape(-1).cpu().numpy())
        return np.concatenate(out).astype(float)

    def metadata(self) -> dict[str, object]:
        return {
            "model_type": "PyTorch MLP EBM",
            "training_objective": "InfoNCE-style contrastive classification over one expert/high-value action and K negatives",
            "feature_mode": self.feature_mode,
            "hidden_dim": self.hidden_dim,
            "num_features": self.input_dim,
            "final_train_loss": float(self.train_loss[-1]),
            "initial_train_loss": float(self.train_loss[0]),
            "loss_decrease": float(self.train_loss[0] - self.train_loss[-1]),
        }


def toy_negative_sampler(obs: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    return sample_ibc_negatives(obs, k, rng).astype(np.float32)
