# Codex Handoff

## Current Goal

Preserve and strengthen the EBM Best-of-N repository so a new thread can resume
from the worktree and continue improving paper-quality evidence.

## Repo Facts Verified From Files

- Repo path: `C:\Users\wangz\ebm-policy-best-of-n`.
- Local git repository initialized during the v2 pass; no remote was verified.
- Package code lives in `src/ebm_best_of_n/`.
- Experiment entrypoints live in `experiments/`.
- Shell runners live in `scripts/`.
- Paper and audit docs live in `paper/` and `docs/`.
- Results are stored under `results/` as CSV/JSON/PNG artifacts.
- `README.md` describes the thesis, claim boundaries, quickstart, key
  artifacts, and figures.
- `docs/differentiation_from_wam_jepa_diffusion.md` distinguishes this EBM
  project from WAM, JEPA, and diffusion Best-of-N projects.
- `docs/final_audit.md` is the authoritative v2 audit.
- `results/claims_status.json` currently reports 10 SUPPORTED claim categories,
  1 PARTIAL category, 1 UNSUPPORTED category, and 14/14 not-clone checks passed.

## Files Changed and Why

- Added `src/ebm_best_of_n/torch_models.py`: PyTorch MLP EBM trained with
  InfoNCE-style contrastive classification.
- Added `experiments/run_learned_torch_ibc.py`: neural IBC-style EBM artifact.
- Added `experiments/run_metaworld_benchmark.py`: guarded Meta-World reach-v3
  benchmark artifact.
- Added `experiments/run_calibration_ablation.py`: pilot-label and
  support-weight repair ablations.
- Updated shell runners to execute the new experiments.
- Updated `src/ebm_best_of_n/plotting.py` to regenerate Figures 7-10.
- Updated `scripts/claim_audit.py` to audit PyTorch, Meta-World, and ablation
  claims.
- Added tests for PyTorch model behavior and artifact schema categories.
- Updated `README.md`, `docs/benchmark_plan.md`, `paper/experiments.md`,
  `paper/limitations.md`, and `docs/final_audit.md`.
- Added this handoff file for fresh-thread continuity.

## Commands/Tests Run

- `bash scripts/run_smoke.sh`: PASS, 212.2 s on final v2 smoke.
- `bash scripts/run_all.sh`: PASS, 1006.3 s on final v2 full run.
- `bash scripts/run_claim_audit.sh`: PASS, 6.5 s.
- `pytest`: PASS, 46.9 s, `19 passed in 32.85 s`.

## Experiment Artifacts and Results

- Controlled tail failure:
  - `results/controlled_energy_tail/summary.csv`
  - Strongest raw miscalibration at `task=multimodal`, `N=128`: selected energy
    `-2.9748`, selected utility `-1.5131`, invalid rate `1.0`.
- Learned NumPy IBC failure:
  - `results/learned_ibc/summary.json`
  - Raw utility drops from `0.3627` at N=1 to `-1.1628` at N=128.
- PyTorch neural IBC:
  - `results/learned_torch_ibc/summary.json`
  - Loss decreased from `1.6582` to `0.0123`.
  - Strongest raw torch row: selected utility `1.5847`, exact-law error
    `0.0043`, invalid rate approximately `0`.
- Meta-World benchmark:
  - `results/benchmarks/metaworld/summary.json`
  - Status `SUPPORTED`.
  - Task `reach-v3`, 3 seeds, 8 states x 128 actions per seed.
  - Strongest raw benchmark row: selected reward `1.6220`, exact-law error
    `0.0005`, high-N regret `0.2990`.
- Repair and ablations:
  - `results/repair/summary.json`
  - `results/ablations/summary.json`
  - Best pilot-label ablation: 512 labels, high-N utility `2.0318`, gain
    `3.3843`.
- Figures:
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

## Known Failures or Bugs

- No real-robot validation exists.
- Meta-World benchmark coverage is currently only `reach-v3`.
- ManiSkill and robosuite import locally but do not yet have supported rollout
  artifacts in this repo.
- Meta-World one-step `reach-v3` success is zero in the current artifact; the
  supported benchmark claim is reward/diagnostic based, not success based.
- PyTorch and simulator imports are slow on this machine.

## Open Questions

- UNKNOWN whether a remote repository should be configured.
- UNKNOWN whether the user wants a committed local git history or remote PR.
- UNKNOWN whether Meta-World `push-v3`, ManiSkill, or robosuite should be the
  next benchmark priority.

## Next Recommended Steps

1. Add a contact-rich Meta-World task such as `push-v3`.
2. Add finite rollout artifacts for ManiSkill or robosuite if stable.
3. Add a multi-task benchmark table.
4. Add bibliography and turn the paper skeleton into a draft.
5. Add real-robot or high-fidelity policy evidence before any real-deployment
   claim.

Safe to clear after handoff is updated.
