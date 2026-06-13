# Experiments

## Controlled Energy-Tail Miscalibration

Two toy robot-action tasks are used: contact-rich push/insert and multimodal
obstacle action selection. Energy variants include oracle, good tail-aligned,
raw miscalibrated, shortcut, smoothness-blind, OOD-low-energy, support-blind,
noisy-energy, and pilot-label-noise energies. Controlled stress tests extend
the candidate-count axis through `N=512`; Meta-World remains capped at
`N=128` for CPU runtime.

## Learned IBC-Style EBM Policy

A small NumPy linear-feature EBM is trained with an InfoNCE-style contrastive
classification loss over one expert action and negative action samples. The
dataset is multimodal. The learned model is intentionally observation-limited:
it sees goal, force, smoothness, and low-effort cues, but not hidden contact or
obstacle validity.

The v2 artifact also trains a PyTorch MLP EBM with the same contrastive policy
interface. This gives a stronger learned-policy artifact and shows the
aligned-tail regime alongside the failure case.

## Multimodal Action Support

The experiment compares explicit MSE regression, a tail-aligned EBM, and a raw
shortcut EBM. The point is fair: EBMs can represent multimodal action choices,
but high-N low-energy selection can still choose invalid shortcuts if the tail
is miscalibrated.

## Optimization Budget and Latency

The repository compares sample-only scoring, CEM-like top-k refinement, and a
Langevin-like refinement heuristic under energy-evaluation budgets.

## Selected-Tail Reliability

Candidate actions are bucketed by energy quantile. Each bucket reports mean real
utility, success, contact failure, support distance, and whether it is the
extreme low-energy tail. This directly tests whether low energy means useful
action in the deployment-relevant tail.

## Repair and Gates

Repairs are separated by deployment assumptions. Support penalties and
conservative stopping require no utility labels. Calibrated energy uses a small
pilot set of real-utility labels. Value-shaped and oracle energies are
upper-bound repairs. Gate outputs are exactly one of `allow_high_n`,
`stop_early`, `collect_pilot_labels`, or `block_high_n`.
The main reported repair stack is calibrated plus support-penalized scoring
with a conservative high-`N` gate. The support-only gated stack is kept as a
deployable ablation. Repair recovery is reported in
`results/repair_effectiveness/summary.csv` and must reach the 95% mean
recovery target before the paper uses near-complete recovery language.

## Exact-Law Validation

Finite-law predictions are compared to Monte Carlo selected utility estimates
with confidence intervals.

## Meta-World Benchmark Ladder

The benchmark path attempts Meta-World `reach-v3`, `push-v3`,
`pick-place-v3`, and `button-press-v3`. It collects scripted expert actions
from Meta-World policies as primary positives, trains an IBC-style EBM against
sampled negatives, and keeps high-reward sampled positives as an ablation.
Candidate pools are audited under tail selection for average closed-loop reward after
the selected action, success under scripted continuation, contact failure,
support distance, high-N regret, and exact-law error.

The full run also writes a closed-loop learned-EBM dependency audit under
`results/benchmarks/metaworld/closed_loop_ablation_summary.csv`. This audit
executes policies at every step without scripted continuation and compares
`expert_centered_gate`, `expert_centered_no_gate`,
`mixed_expert_local_no_gate`, `local_gaussian_no_gate`,
`learned_demo_proposal_no_gate`, `learned_demo_proposal_direct`,
`learned_bc_proposal_no_gate`, `learned_bc_proposal_direct`,
`state_heuristic_proposal_no_gate`, `state_heuristic_direct`,
`random_uniform`, and `scripted_expert`. The table reports fallback rate,
expert-action distance, nearest-demo, behavior-cloned, and state-heuristic
proposal labels, success drops, and reward drops. In the current full
artifacts, the state-heuristic no-gate variants clear the low-dependency
threshold, while the nearest-demo, behavior-cloned, and local Gaussian learned
variants do not clear the learned-policy threshold. We therefore report this as
a proposal-dependency result, not as autonomous learned-policy success.

Optional Gymnasium, ManiSkill, and robosuite adapters are reported separately.
They support finite guarded rollout execution only. In the current artifacts,
robosuite Lift has nonzero scripted-policy success, while other optional tasks
remain reward-only or zero-success diagnostics.

These artifacts support only guarded simulator-benchmark claims. They do not
claim real-robot validation.
