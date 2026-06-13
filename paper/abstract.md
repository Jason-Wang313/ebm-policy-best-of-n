# Abstract

Energy-based robot policies select low-energy actions from candidate action or
trajectory sets. tail selection inference, where more candidates are sampled and the
minimum-energy candidate is executed, stresses the low-energy tail of the
energy model. We give exact finite tail selection laws that characterize selected
real utility for a fixed energy stack, using the score convention `S = -E`.
Low-energy tail miscalibration occurs when energy improves with `N` but real
utility does not. Controlled and learned IBC-style toy experiments show this
failure: raw energies can assign spuriously low energy to invalid, shortcut, or
contact-bad actions. A guarded Meta-World ladder over reach, push, pick-place,
and button-press trains its primary benchmark EBM on scripted expert actions and
keeps high-reward sampled positives as an ablation. Energy reliability diagrams
show how real utility varies across energy quantiles and highlight the extreme
low-energy tail. Calibration, value-shaped energy, and support-aware energy
repair selected-tail utility in controlled settings; the strongest repair claim
is limited to a 95% repair recovery ratio target on supported local stress
artifacts. No real-robot validation is claimed.
