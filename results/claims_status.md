# Claim Status

Promoted claims are classified from local artifacts. Unsupported future endpoints are listed separately as non-claim boundaries.

## Claims

- **SUPPORTED** (theorem claims): The finite tie-aware tail selection law predicts selected real utility for a fixed score/utility pool.
  Evidence: `results/exact_law/summary.json and results/exact_law/validation.csv`
- **SUPPORTED** (controlled EBM toy claims): Controlled contact and multimodal toys exhibit low-energy tail miscalibration: selected energy decreases while real utility falls.
  Evidence: `results/controlled_energy_tail/summary.csv`
- **SUPPORTED** (learned IBC-style EBM claims): A small learned contrastive EBM can over-optimize an observation-limited low-energy tail, and calibrated/value-shaped scoring improves high-N utility.
  Evidence: `results/learned_ibc/summary.csv and results/learned_ibc/summary.json`
- **SUPPORTED** (PyTorch learned EBM claims): A PyTorch MLP IBC-style EBM trains with decreasing contrastive loss and produces finite exact-law selected-utility curves.
  Evidence: `results/learned_torch_ibc/summary.csv and results/learned_torch_ibc/summary.json`
- **SUPPORTED** (scripted expert benchmark claims): Meta-World reach-v3, push-v3, pick-place-v3, and button-press-v3 artifacts train the primary benchmark EBM on scripted expert actions; high-reward sampled actions are only an ablation.
  Evidence: `results/benchmarks/metaworld/task_table.csv and results/benchmarks/metaworld/summary.json`
- **SUPPORTED** (closed-loop Meta-World dependency audit claims): A CPU-feasible closed-loop dependency audit reports gated, ungated, expert-centered, learned-demo-proposal, behavior-cloned-proposal, state-heuristic, local, random, and scripted variants without scripted continuation.
  Evidence: `results/benchmarks/metaworld/closed_loop_ablation_summary.csv, results/benchmarks/metaworld/closed_loop_ablation_rollouts.csv, and results/figures/figure10_closed_loop_dependency_audit.png`
- **SUPPORTED** (state-heuristic low-dependency Meta-World claims): In the closed-loop Meta-World audit, the state-heuristic proposal variants clear the no-gate, no-runtime-scripted-expert low-dependency success threshold.
  Evidence: `results/benchmarks/metaworld/closed_loop_ablation_summary.json and results/benchmarks/metaworld/closed_loop_ablation_summary.csv`
- **SUPPORTED** (selected-tail reliability claims): Energy reliability diagrams bucket candidates by energy quantile and expose the extreme low-energy tail.
  Evidence: `results/reliability/summary.csv and results/figures/figure3_tail_reliability.png`
- **SUPPORTED** (multimodal action claims): A tail-aligned EBM can represent two valid modes better than explicit MSE regression, while a shortcut energy can still fail at high N.
  Evidence: `results/multimodal_action/summary.json`
- **SUPPORTED** (optimization/latency claims): More EBM inference compute trades energy evaluations and latency proxy against real utility, and added candidates can over-optimize energy.
  Evidence: `results/optimization_budget/summary.csv`
- **SUPPORTED** (repair/calibration claims): Support penalties, conservative stopping, pilot-label calibration, value shaping, and oracle scoring form a repair hierarchy for selected-tail utility.
  Evidence: `results/repair/summary.json, results/repair/summary.csv, and results/ablations/summary.csv`
- **SUPPORTED** (metric-bound near-complete repair claims): On supported local repair stress tests, the main pilot-label combined repair nearly closes the measured high-N gap by the repair recovery ratio definition.
  Evidence: `results/repair_effectiveness/summary.json, results/repair_effectiveness/summary.csv, and results/repair_effectiveness/stress_matrix.csv`
- **SUPPORTED** (submission figure claims): The paper-facing figures, including repair recovery and attack-surface audit figures, are regenerated from local CSV/JSON artifacts.
  Evidence: `results/figures/figures.json`
- **SUPPORTED** (optional benchmark claims): Optional non-Meta-World adapters produce finite guarded rollout artifacts; reward and success are reported honestly and do not support broad manipulation success.
  Evidence: `results/optional_benchmarks/summary.json, results/optional_benchmarks/rollouts.csv, and docs/benchmark_plan.md`
- **SUPPORTED** (optional scripted success claims): At least one optional non-Meta-World suite reports nonzero scripted-policy success while remaining scoped to finite simulator rollouts.
  Evidence: `results/optional_benchmarks/summary.json and results/optional_benchmarks/summary.csv`
- **SUPPORTED** (forbidden/overclaim claims): Forbidden clone and overclaim statements are treated as disallowed, not supported findings.
  Evidence: `docs/differentiation_from_wam_jepa_diffusion.md and docs/claims.md`

## Non-Claim Boundaries

- **UNSUPPORTED_NON_CLAIM** (real_robot_validation): No real-robot artifacts are present; docs and claim stress audit block real-robot validation wording.
- **UNSUPPORTED_NON_CLAIM** (fully_autonomous_learned_metaworld_policy_success): State-heuristic low-dependency success is supported, but nearest-demo, behavior-cloned, and local learned variants do not clear the learned-policy threshold.
- **UNSUPPORTED_NON_CLAIM** (broad_manipulation_success): All simulator claims remain artifact-scoped and do not support broad manipulation or real-robot success.

## Not Clone Audit

- PASS: differentiation doc exists
- PASS: EBM-specific energy setup exists
- PASS: S = -E convention exists
- PASS: learned EBM/IBC-style policy exists
- PASS: scripted expert benchmark artifact exists
- PASS: multi-task benchmark table exists
- PASS: energy reliability artifact exists
- PASS: action/trajectory energy scorer exists
- PASS: artifacts report low-energy tail metrics
- PASS: raw EBM compared to calibrated/value-shaped EBM
- PASS: compute/latency proxy reported
- PASS: no real-robot validation claim in supported claims
- PASS: does not imply EBMs always work
- PASS: does not imply tail selection always helps
- PASS: near-complete repair wording defines recovery ratio
- PASS: claim stress audit passed
- PASS: closed-loop dependency audit exists
- PASS: does not duplicate WAM/JEPA/diffusion failure wording as the EBM claim

## Claim Stress Audit

- PASS: near-100 wording is metric-defined
- PASS: does not claim solved manipulation
- PASS: does not claim real-robot validation
- PASS: does not claim repairs always fix EBMs
- PASS: optional success claim blocked when zero
- PASS: autonomous Meta-World claim blocked without low-dependency success
- PASS: autonomous learned-policy claim blocked without learned threshold

## Forbidden Claims

The following are explicitly forbidden as supported claims: We prove EBMs work; we solve robot manipulation; we validate on real robots; tail selection always helps; calibration always fixes energy policies; energy is real utility; low energy means good action; this is not toy evidence without real benchmarks; this is a universal training recipe; this is WAM model-error amplification; this is JEPA latent tail hallucination; this is diffusion diversity-selection tradeoff.
