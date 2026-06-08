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
failure with exact finite Best-of-N laws, low-energy tail metrics, learned
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

The full run is CPU-only and now includes a PyTorch neural IBC-style EBM plus a
guarded Meta-World reach-v3 benchmark artifact when the local simulator stack is
available. External robotics suites remain guarded and do not support claims
unless they produce artifacts.

## Key Artifacts

- Controlled tail failure: `results/controlled_energy_tail/summary.csv`
- Learned IBC-style EBM: `results/learned_ibc/summary.json`
- PyTorch neural IBC-style EBM: `results/learned_torch_ibc/summary.json`
- Meta-World benchmark: `results/benchmarks/metaworld/summary.json`
- Repair ablations: `results/ablations/summary.json`
- Repair and gates: `results/repair/summary.json`
- Compute frontier: `results/optimization_budget/summary.csv`
- Exact law validation: `results/exact_law/validation.csv`
- Claim audit: `results/claims_status.md`
- Final audit: `docs/final_audit.md`

## Key Figures

After `bash scripts/run_all.sh`, figures are regenerated from CSV/JSON only:

- `results/figures/figure1_low_energy_tail_energy.png`
- `results/figures/figure1_low_energy_tail_utility.png`
- `results/figures/figure2_repair_comparison.png`
- `results/figures/figure3_tail_diagnostics.png`
- `results/figures/figure4_compute_utility_frontier.png`
- `results/figures/figure5_exact_law_validation.png`
- `results/figures/figure6_distinction_table.png`
- `results/figures/figure7_torch_ibc_repair.png`
- `results/figures/figure8_metaworld_benchmark.png`
- `results/figures/figure9_repair_ablations.png`
- `results/figures/figure10_metaworld_compute_frontier.png`

## Claim Boundaries

Supported claims are local to fixed generator/scorer/energy stacks. Evidence is
split into controlled toy artifacts, learned NumPy/PyTorch IBC-style EBM
artifacts, and guarded Meta-World reach-v3 benchmark artifacts. The repository
does not claim real-robot validation, robot manipulation solved, or a universal
EBM training recipe. Calibration and support penalties are repairs demonstrated
in generated artifacts, not a general guarantee.

For the current evidence ledger, inspect `docs/claims.md` or
`results/claims_status.json`.
