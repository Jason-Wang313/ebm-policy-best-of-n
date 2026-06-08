# Method

## EBM Policy Setup

For observation `o` and action or trajectory `a`, the policy has conditional
energy `E_theta(o, a)`. The theorem code maximizes scores, so EBM inference is
written as `S(o, a) = -E_theta(o, a)`.

Candidate actions are sampled from `q(a | o)`. The selected action is the
minimum-energy candidate among `N` samples. Real utility `U(o, a)` is measured
separately.

## Finite Best-of-N Law

For a finite candidate pool, tied score groups define exact selection
probabilities under sampling with replacement. The law predicts expected
selected utility for every `N` and is used to audit each fixed
generator/scorer/utility stack.

## Tail Diagnostics

The method reports selected energy, selected score, selected real utility,
success rate, low-energy tail real utility, tail rank correlation,
energy-utility rank correlation, invalid action rate, jerk penalty, contact
penalty, high-N regret, oracle gap, and marginal utility per candidate.

## Repairs

The repository tests pilot-label calibration, value-shaped energy, support
penalties, tail-aware calibration, and conservative deployment gates. A repair
is counted as useful only if selected-tail utility improves.

## Compute Frontier

Sampling and refinement are treated as energy-evaluation budgets. The frontier
plots real utility against energy evaluations or latency proxy.
