from __future__ import annotations

import numpy as np

from ebm_tail_audit.energy_models import raw_miscalibrated_energy
from ebm_tail_audit.toy_envs import generate_multimodal_pool


def test_seeded_pool_and_energy_are_reproducible():
    pool_a = generate_multimodal_pool(seed=11, n=128)
    pool_b = generate_multimodal_pool(seed=11, n=128)
    assert np.allclose(pool_a["actions"], pool_b["actions"])
    assert np.allclose(raw_miscalibrated_energy(pool_a, seed=3), raw_miscalibrated_energy(pool_b, seed=3))
