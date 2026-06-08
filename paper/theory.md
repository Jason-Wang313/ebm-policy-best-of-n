# Theory Summary

The exact law is a finite empirical identity for Best-of-N selection. It is not
an EBM training theorem. It says that for a fixed finite pool of scores and real
utilities, the expected utility of the maximum-score selected item is the sum
over tied score groups of group mean utility times the probability that the
top sampled score falls in that group.

For EBMs, the score is `S = -E`. The theory therefore audits the selected
minimum-energy tail. If that tail has high real utility, larger `N` can help or
saturate. If that tail is noisy, invalid, or anti-aligned with real utility,
larger `N` can reduce energy while reducing selected utility.

The theorem is conditional on the candidate generator, energy scorer, and
utility measurement. It does not claim a general EBM guarantee.
