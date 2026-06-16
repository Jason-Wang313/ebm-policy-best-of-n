# Final Audit

## V4 Submission Package

V4 adds a frozen adversarial evidence layer without rerunning the heavy
Meta-World stack. The build path is:

- `python scripts/build_v4_paper.py`
- `python scripts/run_v4_claim_audit.py`
- `pytest`

Current v4 frozen evidence is under `results/v4_frozen_evidence/`. It reports
9/9 passing claim gates, 36 Meta-World selected-action confidence rows, 12
closed-loop dependency stress rows, 16 reliability tail-gap rows, 51
compute-frontier utility-drop steps, 151 audited non-cache files, and 9 v4
figures. The final versioned PDF is
`paper/final/ebm-policy-best-of-n-v4.pdf` and the visible Desktop copy must
have the same SHA-256 hash.

## 1. Required Command Results

- `python experiments/run_metaworld_benchmark.py`: PASS, 1334.1 s for the full twelve-variant, five-seed, four-task Meta-World child cache refresh; later aggregate refreshes reused the versioned child artifacts.
- `bash scripts/run_smoke.sh`: PASS, 556.7 s.
- `bash scripts/run_all.sh`: PASS, 978.2 s observed wall time on the final full run. The Meta-World stage used versioned task/seed child artifacts and rewrote the aggregate CSV/JSON outputs.
- `bash scripts/run_claim_audit.sh`: PASS after final doc refresh, 47.6 s.
- `pytest`: PASS after final doc refresh, `31 passed in 30.75 s`.

## 2. Key Artifacts

- Controlled tail failure: `results/controlled_energy_tail/summary.json`
- Learned NumPy IBC artifact: `results/learned_ibc/summary.json`
- Learned PyTorch IBC artifact: `results/learned_torch_ibc/summary.json`
- Multi-task Meta-World table: `results/benchmarks/metaworld/task_table.csv`
- Meta-World closed-loop dependency audit: `results/benchmarks/metaworld/closed_loop_ablation_summary.csv`
- Meta-World seed curves: `results/benchmarks/metaworld/seed_level.csv`
- Optional non-Meta-World rollouts: `results/optional_benchmarks/summary.csv`
- Energy reliability buckets: `results/reliability/summary.csv`
- Repair hierarchy: `results/repair/summary.json`
- Repair effectiveness: `results/repair_effectiveness/summary.json`
- Repair ablations: `results/ablations/summary.json`
- Claim audit: `results/claims_status.json`
- Submission figures: `results/figures/figures.json`

## 3. Strongest Failure Artifact

Path: `results/controlled_energy_tail/summary.json` and `results/controlled_energy_tail/seed_level.csv`

Strongest current high-N row: `model=raw_miscalibrated`, `N=512`, `seed=2`.

- Selected energy: `-3.1041`
- Selected real utility: `-1.3797`
- Low-energy tail utility: `-1.4401`
- Tail rank correlation: `0.4033`
- Energy/utility rank correlation: `-0.3605`
- Contact failure rate diagnostic: `1.1099`
- Support distance: `0.8210`
- High-N regret: `3.4925`
- Deployment gate: `collect_pilot_labels`

This remains the cleanest extreme-tail failure: energy improves under high-N selection while real utility collapses because the low-energy selected tail is misaligned with the utility metric.

## 4. Strongest Repair Artifact

Path: `results/repair_effectiveness/summary.json`

At high N, repair gains over raw are measured by `repair_recovery_ratio = (repair_utility - raw_utility) / (oracle_utility - raw_utility)`. The near-complete repair wording is allowed only under this metric-bound definition.

At `N=512`, the main stack `calibrated_support_penalized_conservative_gate` reports:

- Mean repair recovery ratio: `0.9719`
- Worst-case recovery ratio: `0.9624`
- Recovery >= 95% rate: `1.0`
- Utility-degrading rows: `0`
- Explicit repair failure rows: `1`, from the tail-rank-improvement diagnostic

Support-only repairs are weaker and are kept as deployable ablations: `support_penalized` mean recovery is `0.8859`, and `support_penalized_conservative_gate` mean recovery is `0.8696`.

## 5. Benchmark Ladder

Path: `results/benchmarks/metaworld/task_table.csv`

Tasks:

- `reach-v3`
- `push-v3`
- `pick-place-v3`
- `button-press-v3`

Primary benchmark model: `expert_ibc`, trained on scripted expert actions. High-reward sampled positives are present only as `high_reward_ablation`.

Current benchmark claim status: `SUPPORTED` for guarded simulator artifacts. The benchmark evaluates the selected first action followed by scripted continuation. All four tasks have selected success `1.0` in the generated five-seed artifacts. Average selected reward for `expert_ibc` is `8.3263` on `reach-v3`, `5.7263` on `push-v3`, `5.3309` on `pick-place-v3`, and `0.8929` on `button-press-v3`; calibrated scoring reports `8.5808`, `6.2151`, `5.5665`, and `0.9086`.

## 6. Closed-Loop Dependency Audit

Path: `results/benchmarks/metaworld/closed_loop_ablation_summary.csv`

The closed-loop audit executes policies without scripted continuation. It reports twelve variants: gated expert-centered, ungated expert-centered, mixed expert/local, local Gaussian, learned-demo proposal with EBM selection, learned-demo direct, behavior-cloned proposal with EBM selection, behavior-cloned direct, state-heuristic proposal with EBM selection, state-heuristic direct, random uniform, and scripted expert.

Five-seed policy summary:

- `expert_centered_gate`: success `1.0`, mean reward `145.0970`, fallback rate `0.1426`.
- `expert_centered_no_gate`: success `0.65`, mean reward `117.2150`.
- `mixed_expert_local_no_gate`: success `0.15`, mean reward `108.6346`.
- `local_gaussian_no_gate`: success `0.05`, mean reward `117.9830`.
- `learned_demo_proposal_no_gate`: success `0.30`, mean reward `166.5241`.
- `learned_demo_proposal_direct`: success `0.35`, mean reward `166.0971`.
- `learned_bc_proposal_no_gate`: success `0.40`, mean reward `138.3411`.
- `learned_bc_proposal_direct`: success `0.50`, mean reward `109.0708`.
- `state_heuristic_proposal_no_gate`: success `1.0`, mean reward `143.2824`.
- `state_heuristic_direct`: success `1.0`, mean reward `143.1380`.
- `random_uniform`: success `0.0`, mean reward `55.3964`.
- `scripted_expert`: success `1.0`, mean reward `143.1379`.

The low-dependency success threshold is met by the state-heuristic no-gate variants, including the learned-EBM-selected state-heuristic proposal variant. The nearest-demo, behavior-cloned, and local Gaussian learned no-gate variants do not clear the learned-policy threshold across all tasks. This is a stronger result than the earlier dependency-only audit because it includes a trained behavior-cloned proposal, but it remains a proposal-prior result rather than a broad learned manipulation claim.

## 7. Optional Rollouts

Path: `results/optional_benchmarks/summary.json` and `results/optional_benchmarks/summary.csv`

Optional Gymnasium, ManiSkill, and robosuite adapters produce finite guarded rollout artifacts. This supports adapter execution only, not broad manipulation success. The current optional success flag is nonzero because robosuite Lift reports scripted success:

- Gymnasium Pusher-v5: reward-only artifact, success unavailable, mean total reward `-18.4175`.
- ManiSkill PickCube-v1: success `0.0`, mean total reward `0.9875`.
- ManiSkill PushCube-v1: success `0.0`, mean total reward `1.2973`.
- robosuite Door: success `0.0`, mean total reward `0.0273`.
- robosuite Lift: success `0.275`, mean total reward `55.8737`.

## 8. Reliability Diagram

Path: `results/reliability/summary.csv`

Figure path: `results/figures/figure3_tail_reliability.png`

Reliability rows bucket candidate actions from low to high energy and report mean reward/utility, success, invalid rate, contact failure, support distance, and an extreme-low-energy-tail flag.

## 9. Submission Figures

Regenerated figure set:

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

## 10. Claim Ledger

`results/claims_status.json` reports:

- 16 SUPPORTED promoted claim categories.
- 0 PARTIAL promoted claim categories.
- 0 UNSUPPORTED promoted claim categories.
- 18/18 not-clone checks passed.

Unsupported endpoints are listed separately as non-claim boundaries:

- `real_robot_validation`: no real-robot artifacts are present.
- `fully_autonomous_learned_metaworld_policy_success`: the nearest-demo, behavior-cloned, and local learned variants do not meet the learned-policy threshold.
- `broad_manipulation_success`: simulator artifacts remain scoped and do not imply broad manipulation success.

## 11. Remaining Weaknesses

- No real-robot validation.
- Meta-World benchmark utility uses selected-action plus scripted continuation.
- Closed-loop Meta-World success is still proposal-dependent: state-heuristic proposals clear the low-dependency threshold, but nearest-demo, behavior-cloned, and local learned proposals do not clear their learned threshold.
- Optional non-Meta-World rollouts are finite adapter checks; only robosuite Lift currently has nonzero scripted success.
- Full-run statistical budgets are still local-CPU sized, despite moving the main experiments to five seeds.

## 12. Exact Next Patch

The next meaningful improvement is to train or distill a learned proposal that clears the per-task learned threshold on push, pick-place, reach, and button-press without a runtime scripted expert or runtime state heuristic, then rerun the closed-loop dependency audit before making any autonomous learned-policy claim.
