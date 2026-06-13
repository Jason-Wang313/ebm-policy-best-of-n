# Theory

## Formal EBM Policy Setup

Let `o` be an observation or state and let `a` be an action or short trajectory.
An energy-based policy has a conditional energy function:

```text
E_theta(o, a)
```

Lower energy is preferred. The theorem code in this repository uses scores and
always maximizes:

```text
S(o, a) = -E_theta(o, a)
```

A candidate generator `q(a | o)` may be Gaussian proposals,
replay-near-demonstration proposals, Langevin/refinement proposals, CEM-like
proposals, or a learned proposal. Real utility `U(o, a)` is measured separately
from energy. It can be task reward, success, contact feasibility, smoothness,
safety, or a weighted utility in a toy environment.

The tail selection EBM policy is:

```text
sample a_1, ..., a_N ~ q(a | o)
choose argmin_i E_theta(o, a_i)
equivalently choose argmax_i S(o, a_i)
evaluate U(o, a_selected)
```

## Exact Finite Tie-Aware Tail-Selection Law

For a finite pool of `m` candidate actions with scores `S_i` and real utilities
`U_i`, sort candidates by score in ascending order. Group tied scores. For a tie
group `g`, let `r_min(g)` and `r_max(g)` be its one-indexed rank interval in the
ascending ordering. Under sampling with replacement, the probability that the
maximum-score draw lands in group `g` is:

```text
(r_max(g) / m)^N - ((r_min(g) - 1) / m)^N
```

With deterministic mean tie handling, the expected selected real utility is:

```text
sum_g mean_U(g) * [(r_max(g) / m)^N - ((r_min(g) - 1) / m)^N]
```

Binary success is the special case `U in {0, 1}`. Monte Carlo estimates in the
repository are sanity checks; the finite tie-aware law is the source of truth
for fixed finite pools. Exact-law prediction error is reported as mean absolute
error, max absolute error, and RMSE between the finite-law prediction and Monte
Carlo estimates.

Important edge cases:

- Tied score groups use the tied group's mean utility.
- Constant utility remains constant for every `N`.
- Oracle score `S = U` gives monotone nondecreasing expected utility.
- Anti-aligned score can make selected utility decrease as `N` grows.
- The EBM energy convention is handled by `S = -E`; minimum energy selection is
  maximum score selection.

## EBM-Specific Definitions

- Low-energy tail: the subset of candidates with smallest `E_theta(o, a)`.
- Energy-tail calibration: alignment between low energy and real utility in the
  selected tail, not merely average prediction accuracy.
- Energy-utility rank alignment: rank correlation between `-E` and `U`.
- Tail rank correlation: rank correlation between `-E` and `U` inside a
  low-energy quantile relevant to `N`.
- Low-energy tail real utility: average `U` among low-energy tail candidates.
- Energy over-optimization: selected energy improves with more compute while
  selected utility does not improve or decreases.
- High-N energy regret: oracle-selected utility minus energy-selected utility
  at high `N`.
- Oracle-minus-energy gap: same finite-pool gap for a specified `N`.
- Action-validity gap: utility loss caused by selecting invalid actions.
- Physical-validity false positives: candidates with low energy but invalid
  contact, safety, support, or task outcome.
- Jerk/smoothness penalty: utility penalty from nonsmooth actions.
- Contact-feasibility penalty: utility penalty from contact loss, collision, or
  unsafe contact.
- Compute-utility frontier: selected utility as a function of energy
  evaluations or latency proxy.

## Central Corollaries

1. If the low-energy tail is aligned with real utility, increasing `N` helps or
   saturates.
2. If the low-energy tail is noisy or anti-aligned with real utility, increasing
   `N` can reduce energy while hurting or saturating real utility.
3. Average energy-utility correlation is insufficient; the lower energy tail is
   the deployment-relevant region.
4. Calibration helps only if it repairs selected-tail ranking, not merely
   average prediction error.
5. tail selection inference trades sampling or optimization compute and latency
   against real utility.

## EBM Tail Propositions

**Proposition 1: high-N inference selects the lower-energy tail.** For a fixed
candidate generator and energy function, the probability that tail selection selects
from any low-energy quantile increases with `N`. Equivalently under the score
convention `S = -E`, larger `N` concentrates selection on higher score ranks.

**Proposition 2: tail selection helps only when low-energy quantiles improve real
utility.** For a fixed finite pool, increasing `N` improves expected selected
utility exactly when the additional selection mass placed on lower-energy
candidates has higher real utility than the mass it replaces. Average
energy-utility correlation is not sufficient; the relevant object is the
ordered sequence of energy quantiles.

**Proposition 3: low-energy false positives can improve energy while degrading
utility.** If the extreme low-energy tail contains invalid, contact-bad, or
out-of-support actions whose real utility is below the current selected
expectation, then larger `N` can reduce selected energy and simultaneously
reduce selected real utility.

**Proposition 4: calibration repairs high-N only if it repairs tail ranking.**
Calibration that improves average prediction error but leaves false positives
ranked in the extreme low-energy tail does not fix tail selection deployment. A
repair helps high-N inference only to the extent that it moves real high-utility
actions above low-energy false positives in the selected tail.

## Boundary

The theorem audits a fixed generator/scorer/energy stack. It does not prove a
general property of EBMs, solve robot planning, establish real-robot validation,
or guarantee that calibration fixes every energy policy.
