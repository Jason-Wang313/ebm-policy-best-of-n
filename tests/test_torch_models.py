from __future__ import annotations

import importlib.util

import numpy as np
import pytest

torch_available = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not torch_available, reason="PyTorch is not installed")
def test_torch_ibc_energy_trains_and_scores_finite():
    from ebm_best_of_n.torch_models import TorchIBCEnergy, uniform_box_negatives

    rng = np.random.default_rng(0)
    obs = rng.normal(size=(24, 2)).astype(np.float32)
    actions = np.tanh(obs @ np.array([[0.4, -0.2], [0.1, 0.3]], dtype=np.float32))
    low = np.array([-1.0, -1.0], dtype=np.float32)
    high = np.array([1.0, 1.0], dtype=np.float32)
    model = TorchIBCEnergy.fit(
        obs,
        actions,
        seed=0,
        epochs=4,
        batch_size=12,
        negatives=4,
        hidden_dim=16,
        feature_mode="raw",
        action_low=low,
        action_high=high,
    )
    energy = model.energy(obs, actions)
    assert energy.shape == (24,)
    assert np.all(np.isfinite(energy))
    assert model.metadata()["loss_decrease"] > -0.5
    neg = uniform_box_negatives(obs[:2], 3, rng, low, high)
    assert neg.shape == (2, 3, 2)
