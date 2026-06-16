# Low Energy Is Not Enough

Inference-value laws for energy-based robot policies.

## Thesis

Energy-based robot policies score candidate actions or trajectories with a
conditional energy `E_theta(o, a)` and execute the minimum-energy candidate.
Sampling more candidates at test time is tempting: draw `N` actions, score each
one, and run the lowest-energy action. This repository tests the narrower claim
that this helps only when the low-energy selected tail is aligned with real
task utility.

When the energy tail is miscalibrated, selected energy can improve with `N`
while real utility saturates or drops. The experiments here diagnose that
failure with exact finite tail selection laws, low-energy tail metrics, learned
IBC-style toy EBMs, repair methods, and compute-utility frontiers.

## Why EBM Policies

EBMs are a natural object because robot action distributions can be multimodal,
discontinuous, and awkward for direct MSE regression. But EBM inference is also
an action optimization problem: the deployed policy repeatedly evaluates
`E_theta(o, a)` and selects low energy. That makes the lower energy tail, not
only average imitation likelihood, the relevant deployment object.

## Distinction From Nearby Projects

- WAM: explicit imagined rollouts and imagined-vs-real dynamics mismatch.
- JEPA: latent encoder/predictor scores and latent-real rank distortion.
- Diffusion: stochastic trajectory generation and diversity-selection tradeoff.
- This repo: conditional action energy `E_theta(o, a)`, candidates from
  `q(a | o)`, low-energy selection, real utility after execution, tail
  calibration, and compute/latency tradeoffs.

See `docs/differentiation_from_wam_jepa_diffusion.md`.

## Quickstart

```bash
pip install -r requirements.txt
bash scripts/run_smoke.sh
bash scripts/run_all.sh
bash scripts/run_claim_audit.sh
pytest
```

## Final v4 Submission Package

The submission artifact is built from the RAM-light frozen evidence path:

```bash
python scripts/build_v4_paper.py
python scripts/run_v4_claim_audit.py
pytest
```

The v4 cache does not rerun Meta-World or retrain EBMs. It audits the committed
CSV/JSON artifacts, writes `results/v4_frozen_evidence/`, exports
`paper/iclr2026/v4_results_macros.tex`, creates nine v4 figures under
`results/figures/v4/`, and copies `paper/final/ebm-policy-best-of-n-v4.pdf`
to the visible Desktop. The v4 claim gates currently report 9/9 passing gates,
36 Meta-World selected-action confidence rows, 12 closed-loop dependency stress
rows, 16 reliability tail-gap rows, and 51 compute-frontier utility-drop steps.

The full run is CPU-only and includes learned NumPy/PyTorch IBC-style EBMs plus
a guarded Meta-World benchmark ladder over `reach-v3`, `push-v3`,
`pick-place-v3`, and `button-press-v3`, when the local simulator stack is
available. Benchmark EBMs use scripted expert actions as the primary positives;
high-reward sampled actions are reported only as an ablation. Benchmark utility
is measured as average reward after executing the selected action and then
letting the scripted policy continue for a short horizon, with success reported
separately. A separate CPU closed-loop dependency audit executes policies
without scripted continuation across gated, ungated, expert-centered,
learned-demo-proposal, behavior-cloned-proposal, state-heuristic, local,
random, and scripted variants.
It reports fallback, proposal-prior, and expert-centering dependence
explicitly. In the current full artifacts, the state-heuristic no-gate variants
clear the low-dependency threshold, while nearest-demo, behavior-cloned, and
local Gaussian learned variants do not clear the learned-policy threshold; this
supports a state-heuristic proposal audit, not real-robot or broad manipulation
success.
External robotics suites remain guarded and do not support claims unless they
produce artifacts. Optional Gymnasium/ManiSkill/robosuite rollouts are
adapter-execution evidence only, with reward and success reported honestly; the
current artifacts include nonzero scripted robosuite Lift success.

Repair effectiveness is reported with a metric-bound definition:
`repair_recovery_ratio = (repair_utility - raw_utility) /
(oracle_utility - raw_utility)`. The near-complete repair claim is supported
only when the main supported repair stack reaches at least 95% mean recovery,
keeps worst-case supported-stress recovery above the audit threshold, and does
not degrade utility in the generated local artifacts.

## Key Artifacts

- Controlled tail failure: `results/controlled_energy_tail/summary.csv`
- Learned IBC-style EBM: `results/learned_ibc/summary.json`
- PyTorch neural IBC-style EBM: `results/learned_torch_ibc/summary.json`
- Meta-World benchmark ladder: `results/benchmarks/metaworld/task_table.csv`
- Meta-World closed-loop dependency audit:
  `results/benchmarks/metaworld/closed_loop_ablation_summary.csv`
- Optional non-Meta-World rollouts:
  `results/optional_benchmarks/summary.csv`
- Energy reliability buckets: `results/reliability/summary.csv`
- Repair ablations: `results/ablations/summary.json`
- Repair and gates: `results/repair/summary.json`
- Repair effectiveness: `results/repair_effectiveness/summary.json`
- Compute frontier: `results/optimization_budget/summary.csv`
- Exact law validation: `results/exact_law/validation.csv`
- Claim audit: `results/claims_status.md`
- Final audit: `docs/final_audit.md`
- V4 claim gates: `results/v4_frozen_evidence/v4_claim_gates.csv`
- V4 Meta-World claim matrix:
  `results/v4_frozen_evidence/v4_metaworld_claim_matrix.csv`
- V4 dependency stress:
  `results/v4_frozen_evidence/v4_policy_dependency_stress.csv`
- V4 reliability/compute stress:
  `results/v4_frozen_evidence/v4_reliability_tail_gaps.csv` and
  `results/v4_frozen_evidence/v4_compute_frontier_stress.csv`

## Key Figures

After `bash scripts/run_all.sh`, submission figures are regenerated from
CSV/JSON only:

- `results/figures/figure1_energy_utility_drop.png`
- `results/figures/figure2_repair_comparison.png`
- `results/figures/figure3_tail_reliability.png`
- `results/figures/figure4_tail_rank_and_tail_utility.png`
- `results/figures/figure5_compute_utility_frontier.png`
- `results/figures/figure6_multitask_benchmark_table.png`
- `results/figures/figure7_repair_ablations.png`
- `results/figures/figure8_repair_recovery_ratio.png`
- `results/figures/figure9_attack_surface_audit.png`
- `results/figures/figure10_closed_loop_dependency_audit.png`
- `results/figures/v4/v4_metaworld_selected_action_ci.pdf`
- `results/figures/v4/v4_dependency_claim_gate.pdf`
- `results/figures/v4/v4_reliability_compute_stress.pdf`

## Claim Boundaries

Supported claims are local to fixed generator/scorer/energy stacks. Evidence is
split into controlled toy artifacts, learned NumPy/PyTorch IBC-style EBM
artifacts, selected-tail reliability diagrams, guarded multi-task Meta-World
artifacts, and optional finite simulator-adapter rollouts. The repository does
not claim real-robot validation, robot manipulation solved, fully autonomous
Meta-World learned-policy success, or a universal EBM training recipe unless
the generated dependency audit satisfies its strict threshold.
Calibration and support penalties are repairs demonstrated in generated
artifacts, not a general guarantee. The strongest repair wording is limited to
recovery of the measured oracle-minus-raw gap in local controlled and simulator
artifacts. Optional non-Meta-World rollouts support finite adapter execution;
scripted success is claimed only for task rows that report nonzero success.

For the current evidence ledger, inspect `docs/claims.md` or
`results/claims_status.json`.
