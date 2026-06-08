# Experiments

## Controlled Energy-Tail Miscalibration

Two toy robot-action tasks are used: contact-rich push/insert and multimodal
obstacle action selection. Energy variants include oracle, good tail-aligned,
raw miscalibrated, shortcut, smoothness-blind, and OOD-low-energy energies.

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

## Repair and Gates

Pilot-label calibration, value-shaped energy, support penalties, and deployment
gates are evaluated. Gate outputs are exactly one of `allow_high_n`,
`stop_early`, `collect_pilot_labels`, or `block_high_n`.

## Exact-Law Validation

Finite-law predictions are compared to Monte Carlo selected utility estimates
with confidence intervals.

## Meta-World Benchmark Probe

The benchmark path attempts Meta-World reach-v3. It collects candidate actions
from simulator states, trains a contrastive EBM on high-reward sampled actions,
and audits selected reward under Best-of-N. These artifacts support only a
guarded simulator-benchmark claim, not real-robot validation.
