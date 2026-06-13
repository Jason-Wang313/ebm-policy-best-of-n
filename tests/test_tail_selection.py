from __future__ import annotations

import numpy as np
import pytest

from ebm_tail_audit.tail_selection import (
    binary_success_tail_selection,
    monte_carlo_tail_selection,
    score_from_energy,
    utility_tail_selection_finite,
)


def test_tied_score_groups_use_mean_utility():
    scores = np.array([0.0, 1.0, 1.0])
    utility = np.array([0.0, 1.0, 3.0])
    out = utility_tail_selection_finite(scores, utility, [1, 2])
    assert out[1] == pytest.approx(np.mean(utility))
    assert out[2] == pytest.approx(2.0 * (1.0 - (1.0 / 3.0) ** 2))


def test_binary_utility_equals_real_valued_binary_case():
    scores = np.array([0.0, 0.5, 0.5, 1.0])
    success = np.array([0, 1, 0, 1])
    assert utility_tail_selection_finite(scores, success, [1, 2, 8]) == binary_success_tail_selection(scores, success, [1, 2, 8])


def test_constant_utility_remains_constant():
    scores = np.array([0.0, 1.0, 2.0, 3.0])
    utility = np.full(4, 7.25)
    out = utility_tail_selection_finite(scores, utility, [1, 4, 16])
    assert all(v == pytest.approx(7.25) for v in out.values())


def test_oracle_score_monotone_and_anti_aligned_degrades():
    utility = np.array([-1.0, 0.0, 0.5, 2.0, 4.0])
    oracle = utility_tail_selection_finite(utility, utility, [1, 2, 4, 8])
    anti = utility_tail_selection_finite(-utility, utility, [1, 2, 4, 8])
    assert [oracle[n] for n in [1, 2, 4, 8]] == sorted(oracle.values())
    assert anti[8] < anti[1]


def test_exact_law_matches_monte_carlo_within_tolerance():
    rng = np.random.default_rng(7)
    scores = rng.normal(size=200)
    utility = 0.4 * scores + rng.normal(scale=0.5, size=200)
    exact = utility_tail_selection_finite(scores, utility, [1, 4, 16])
    mc = monte_carlo_tail_selection(scores, utility, [1, 4, 16], n_trials=12_000, seed=9)
    for n in exact:
        assert mc[n]["mean"] == pytest.approx(exact[n], abs=0.04)


def test_energy_to_score_convention():
    energy = np.array([3.0, 1.0, 2.0])
    assert np.all(score_from_energy(energy) == np.array([-3.0, -1.0, -2.0]))


@pytest.mark.parametrize(
    "scores,utility,n_values",
    [
        ([[1.0]], [1.0], [1]),
        ([1.0, float("nan")], [1.0, 2.0], [1]),
        ([1.0, 2.0], [1.0], [1]),
        ([1.0, 2.0], [1.0, 2.0], [0]),
    ],
)
def test_invalid_input_raises_clean_errors(scores, utility, n_values):
    with pytest.raises(ValueError):
        utility_tail_selection_finite(scores, utility, n_values)
