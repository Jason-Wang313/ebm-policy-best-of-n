from __future__ import annotations

from ebm_best_of_n.best_of_n import score_from_energy, utility_best_of_n_finite
from ebm_best_of_n.calibration import apply_energy_calibrator, fit_energy_calibrator
from ebm_best_of_n.energy_models import raw_miscalibrated_energy, value_shaped_energy
from ebm_best_of_n.toy_envs import generate_contact_push_pool


def test_calibration_and_value_shaping_improve_high_n_tail_utility():
    pool = generate_contact_push_pool(seed=1, n=512)
    raw = raw_miscalibrated_energy(pool, seed=1)
    calibrated = apply_energy_calibrator(pool, raw, fit_energy_calibrator(pool, raw, pilot_size=96, seed=1))
    shaped = value_shaped_energy(raw, pool)
    raw_u = utility_best_of_n_finite(score_from_energy(raw), pool["utility"], [64])[64]
    cal_u = utility_best_of_n_finite(score_from_energy(calibrated), pool["utility"], [64])[64]
    shaped_u = utility_best_of_n_finite(score_from_energy(shaped), pool["utility"], [64])[64]
    assert cal_u > raw_u + 1.0
    assert shaped_u > raw_u + 1.0
