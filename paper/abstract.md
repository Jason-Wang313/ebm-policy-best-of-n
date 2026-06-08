# Abstract

Energy-based robot policies select low-energy actions from candidate action or
trajectory sets. Best-of-N inference, where more candidates are sampled and the
minimum-energy candidate is executed, stresses the low-energy tail of the
energy model. We give exact finite Best-of-N laws that characterize selected
real utility for a fixed energy stack, using the score convention `S = -E`.
Low-energy tail miscalibration occurs when energy improves with `N` but real
utility does not. Controlled and learned IBC-style toy experiments show this
failure: raw energies can assign spuriously low energy to invalid, shortcut, or
contact-bad actions. Calibration, value-shaped energy, and support-aware energy
repair selected-tail utility in these toy settings. The experiments also show
that compute and latency matter because more energy evaluations can
over-optimize the wrong tail. No real-robot validation is claimed unless such
artifacts are added.
