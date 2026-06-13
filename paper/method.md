# Method

## EBM Policy Setup

For observation `o` and action or trajectory `a`, the policy has conditional
energy `E_theta(o, a)`. The theorem code maximizes scores, so EBM inference is
written as `S(o, a) = -E_theta(o, a)`.

Candidate actions are sampled from `q(a | o)`. The selected action is the
minimum-energy candidate among `N` samples. Real utility `U(o, a)` is measured
separately.

In simulator benchmarks, scripted expert policies provide the primary positive
actions for IBC-style EBM training. High-reward sampled actions remain an
ablation, not the main learned-policy evidence.

## Finite Tail-Selection Law

For a finite candidate pool, tied score groups define exact selection
probabilities under sampling with replacement. The law predicts expected
selected utility for every `N` and is used to audit each fixed
generator/scorer/utility stack.

## Tail Diagnostics

The method reports selected energy, selected score, selected real utility,
success rate, low-energy tail real utility, tail rank correlation,
energy-utility rank correlation, invalid action rate, jerk penalty, contact
penalty, support distance, high-N regret, oracle gap, and marginal utility per
candidate. Reliability diagrams bucket candidates by energy quantile and report
mean real utility, success, contact failure, and support distance per bucket,
with the extreme low-energy tail highlighted.

## Repairs

The repository separates repairs by deployment assumptions: support penalties
and conservative stopping need no utility labels; calibrated energy uses a
small pilot set of utility labels; value-shaped and oracle energy are
upper-bound repairs. A repair is counted as useful only if selected-tail utility
improves. Repair effectiveness is summarized by
`repair_recovery_ratio = (repair_utility - raw_utility) /
(oracle_utility - raw_utility)`. Near-complete recovery means at least 95%
mean recovery of the measured oracle-minus-raw gap on supported stress tests,
with utility-degrading rows reported as failures rather than hidden.

## Compute Frontier

Sampling and refinement are treated as energy-evaluation budgets. The frontier
plots real utility against energy evaluations or latency proxy.
