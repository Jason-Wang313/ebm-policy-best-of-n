from __future__ import annotations

import numpy as np

from ebm_tail_audit.tail_selection import score_from_energy


def test_low_energy_selection_equals_max_score_selection():
    energy = np.array([2.0, -1.0, 0.5, -3.0])
    score = score_from_energy(energy)
    assert int(np.argmin(energy)) == int(np.argmax(score))


def test_score_from_energy_validates_nonfinite():
    try:
        score_from_energy([1.0, float("inf")])
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("expected ValueError")
