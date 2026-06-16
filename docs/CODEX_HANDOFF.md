# Codex Handoff

## Current State

- Repo path: `C:\Users\wangz\ebm-policy-best-of-n`.
- Branch: `main`; remote `origin` is configured.
- Current objective: paper-credible EBM tail selection repository about
  extreme-tail inference in energy-based robot policies.
- Primary claim boundary: no real-robot validation and no broad manipulation
  success claim.
- Current final artifact target: `ebm-policy-best-of-n-v4.pdf`.

## Implemented Upgrade

- Added a guarded Meta-World benchmark ladder for `reach-v3`, `push-v3`,
  `pick-place-v3`, and `button-press-v3`.
- Benchmark EBMs now use scripted expert actions as primary positives.
- Benchmark utility is average closed-loop reward after the selected action is
  followed by scripted continuation; success is reported separately.
- High-reward sampled actions remain as `high_reward_ablation`, not main
  evidence.
- Added a CPU-feasible closed-loop Meta-World dependency audit with gated,
  ungated, expert-centered, learned-demo-proposal, behavior-cloned-proposal,
  state-heuristic, local, random, and scripted variants and no scripted
  continuation.
- Added optional guarded rollout artifacts for Gymnasium, ManiSkill, and
  robosuite, including nonzero scripted robosuite Lift success.
- Added candidate energy reliability buckets under `results/reliability/`.
- Added submission figures:
  - `figure1_energy_utility_drop.png`
  - `figure2_repair_comparison.png`
  - `figure3_tail_reliability.png`
  - `figure4_tail_rank_and_tail_utility.png`
  - `figure5_compute_utility_frontier.png`
  - `figure6_multitask_benchmark_table.png`
  - `figure7_repair_ablations.png`
- Updated `scripts/claim_audit.py` so promoted claims are all supported,
  real-robot and broad manipulation success remain non-claim boundaries, and
  state-heuristic low-dependency success is separated from autonomous
  learned-policy success even after behavior-cloned learned proposal rows are
  added.
- Updated README, theory docs, paper docs, claim docs, tests, and final audit.
- Added v4 frozen evidence under `results/v4_frozen_evidence/`.
- Added v4 claim gates, Meta-World selected-action CI rows, dependency stress
  rows, reliability tail-gap rows, compute-frontier stress rows, and nine v4
  PDF figures under `results/figures/v4/`.
- Updated the anonymous paper to use visible green citation links and the v4
  claim-gated benchmark protocol.

## Final Command Results

- V4 submission commands:
  - `python scripts/build_v4_paper.py`
  - `python scripts/run_v4_claim_audit.py`
  - `pytest`
- `python experiments/run_metaworld_benchmark.py`: PASS, 1334.1 s for the full
  twelve-variant, five-seed, four-task Meta-World child cache refresh.
- `bash scripts/run_smoke.sh`: PASS, 556.7 s.
- `bash scripts/run_all.sh`: PASS, 574.2 s from the latest recorded full run;
  the Meta-World stage reused versioned five-seed child artifacts.
- `bash scripts/run_claim_audit.sh`: PASS, 22.2 s; current ledger has 16
  SUPPORTED promoted claims, 0 PARTIAL, and 0 UNSUPPORTED promoted claims.
- `pytest`: PASS, latest recorded suite had `31 passed in 20.04 s`.

## Key Artifact Paths

- Benchmark table: `results/benchmarks/metaworld/task_table.csv`
- Closed-loop Meta-World dependency audit:
  `results/benchmarks/metaworld/closed_loop_ablation_summary.csv`
- Reliability buckets: `results/reliability/summary.csv`
- Reliability figure: `results/figures/figure3_tail_reliability.png`
- Claim ledger: `results/claims_status.json`
- Optional non-Meta-World rollouts:
  `results/optional_benchmarks/summary.csv`
- V4 claim gates: `results/v4_frozen_evidence/v4_claim_gates.csv`
- V4 benchmark claim matrix:
  `results/v4_frozen_evidence/v4_metaworld_claim_matrix.csv`
- V4 final PDF: `paper/final/ebm-policy-best-of-n-v4.pdf`
- Final audit: `docs/final_audit.md`

## Weakest Remaining Claim

Real-robot validation and broad manipulation success remain
UNSUPPORTED_NON_CLAIM boundaries. State-heuristic no-gate Meta-World variants
now clear the low-dependency threshold, but autonomous learned-policy success
remains unsupported because nearest-demo, behavior-cloned, and local learned
variants do not clear their learned threshold across all tasks. Optional
non-Meta-World adapter claims are supported only for finite guarded rollouts;
scripted success is currently nonzero for robosuite Lift and does not support
broad manipulation success.

## Exact Next Patch

Train or distill a learned proposal that clears the per-task learned threshold
on push, pick-place, reach, and button-press without a runtime scripted expert
or runtime state heuristic, then rerun the dependency audit before making any
autonomous learned-policy claim.
