from __future__ import annotations

from ebm_best_of_n.diagnostics import curve_rows_for_energy, tail_rank_correlation
from ebm_best_of_n.energy_models import raw_miscalibrated_energy
from ebm_best_of_n.toy_envs import generate_contact_push_pool


def test_tail_rank_correlation_detects_anti_aligned_tail():
    energy = [0.0, 0.1, 0.2, 5.0, 6.0]
    utility = [-1.0, 1.0, 3.0, 0.0, 0.0]
    assert tail_rank_correlation(energy, utility, tail_fraction=0.6) < 0.0


def test_curve_rows_include_required_ebm_metrics():
    pool = generate_contact_push_pool(seed=0, n=256)
    energy = raw_miscalibrated_energy(pool, seed=0)
    rows, summary = curve_rows_for_energy(pool, energy, [1, 8], "raw", seed=0, exact_mc_trials=100)
    assert rows[0]["selected_energy"] > rows[-1]["selected_energy"]
    for key in [
        "low_energy_tail_real_utility",
        "tail_rank_correlation",
        "invalid_action_rate",
        "jerk_smoothness_penalty",
        "contact_feasibility_penalty",
        "compute_latency_proxy",
        "deployment_gate",
    ]:
        assert key in rows[-1]
    assert "prediction_error" in summary
