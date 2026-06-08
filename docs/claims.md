# Claim Status

Claims are classified as SUPPORTED, PARTIAL, or UNSUPPORTED from local artifacts.

## Claims

- **SUPPORTED** (theorem claims): The finite tie-aware Best-of-N law predicts selected real utility for a fixed score/utility pool.
  Evidence: `results/exact_law/summary.json and results/exact_law/validation.csv`
- **SUPPORTED** (controlled EBM toy claims): Controlled contact and multimodal toys exhibit low-energy tail miscalibration: selected energy decreases while real utility falls.
  Evidence: `results/controlled_energy_tail/summary.csv`
- **SUPPORTED** (learned IBC-style EBM claims): A small learned contrastive EBM can over-optimize an observation-limited low-energy tail, and calibrated/value-shaped scoring improves high-N utility.
  Evidence: `results/learned_ibc/summary.csv and results/learned_ibc/summary.json`
- **SUPPORTED** (PyTorch learned EBM claims): A PyTorch MLP IBC-style EBM trains with decreasing contrastive loss and produces finite exact-law selected-utility curves, showing the aligned-tail regime alongside the failure regime.
  Evidence: `results/learned_torch_ibc/summary.csv and results/learned_torch_ibc/summary.json`
- **SUPPORTED** (Meta-World benchmark claims): The repository produces a guarded Meta-World reach-v3 benchmark artifact with finite reward, energy, compute, and exact-law diagnostics.
  Evidence: `results/benchmarks/metaworld/summary.json and results/benchmarks/metaworld/summary.csv`
- **SUPPORTED** (multimodal action claims): A tail-aligned EBM can represent two valid modes better than explicit MSE regression, while a shortcut energy can still fail at high N.
  Evidence: `results/multimodal_action/summary.json`
- **SUPPORTED** (optimization/latency claims): More EBM inference compute trades energy evaluations and latency proxy against real utility, and added candidates can over-optimize energy.
  Evidence: `results/optimization_budget/summary.csv`
- **SUPPORTED** (repair/calibration claims): Pilot-label calibration, value shaping, and support penalties repair selected-tail utility in controlled toy settings.
  Evidence: `results/repair/summary.json and results/repair/summary.csv`
- **SUPPORTED** (repair ablation claims): Pilot-label count and support-penalty sweeps quantify how much repair signal is needed to recover high-N selected-tail utility.
  Evidence: `results/ablations/summary.json and results/ablations/summary.csv`
- **PARTIAL** (optional benchmark claims): Optional benchmark adapters are guarded and do not support core claims unless dependencies and runs are present.
  Evidence: `results/optional_benchmarks/summary.json and docs/benchmark_plan.md`
- **UNSUPPORTED** (unsupported future robotics claims): Real robot validation is future work, not a supported repository claim.
  Evidence: `No real-robot artifacts are present.`
- **SUPPORTED** (forbidden/overclaim claims): Forbidden clone and overclaim statements are treated as disallowed, not supported findings.
  Evidence: `docs/differentiation_from_wam_jepa_diffusion.md and docs/claims.md`

## Not Clone Audit

- PASS: differentiation doc exists
- PASS: EBM-specific energy setup exists
- PASS: S = -E convention exists
- PASS: learned EBM/IBC-style policy exists
- PASS: PyTorch neural EBM artifact exists
- PASS: Meta-World benchmark artifact exists
- PASS: action/trajectory energy scorer exists
- PASS: artifacts report low-energy tail metrics
- PASS: raw EBM compared to calibrated/value-shaped EBM
- PASS: compute/latency proxy reported
- PASS: no real-robot validation claim in supported claims
- PASS: does not imply EBMs always work
- PASS: does not imply Best-of-N always helps
- PASS: does not duplicate WAM/JEPA/diffusion failure wording as the EBM claim

## Forbidden Claims

The following are explicitly forbidden as supported claims: We prove EBMs work; we solve robot manipulation; we validate on real robots; Best-of-N always helps; calibration always fixes energy policies; energy is real utility; low energy means good action; this is not toy evidence without real benchmarks; this is a universal training recipe; this is WAM model-error amplification; this is JEPA latent tail hallucination; this is diffusion diversity-selection tradeoff.
