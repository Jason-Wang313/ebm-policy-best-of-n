"""Deployment gate decisions for Best-of-N EBM inference."""

from __future__ import annotations

from typing import Literal

GateDecision = Literal["allow_high_n", "stop_early", "collect_pilot_labels", "block_high_n"]


def deployment_gate(
    tail_rank_correlation: float,
    high_n_regret: float,
    invalid_action_rate: float,
    marginal_utility_high_n: float,
    calibration_available: bool = False,
) -> GateDecision:
    """Return exactly one conservative deployment decision."""

    if tail_rank_correlation < -0.05 or invalid_action_rate > 0.35 or high_n_regret > 0.45:
        return "block_high_n" if calibration_available else "collect_pilot_labels"
    if marginal_utility_high_n < 0.002:
        return "stop_early"
    return "allow_high_n"
