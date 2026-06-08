# Final Audit

## 1. Required Command Results

- `bash scripts/run_smoke.sh`: PASS, 212.2 s on the final v2 smoke run.
- `bash scripts/run_all.sh`: PASS, 1006.3 s on the final v2 full run.
- `bash scripts/run_claim_audit.sh`: PASS, 6.5 s after regenerating `docs/claims.md` and `results/claims_status.*`.
- `pytest`: PASS, 46.9 s, `19 passed in 32.85 s`.

The project is now initialized as a local git repository. There is no remote
configured at audit time.

## 2. Artifact Inventory

Key JSONs:

- `results/controlled_energy_tail/summary.json`
- `results/learned_ibc/summary.json`
- `results/learned_torch_ibc/summary.json`
- `results/benchmarks/metaworld/summary.json`
- `results/multimodal_action/summary.json`
- `results/optimization_budget/summary.json`
- `results/repair/summary.json`
- `results/ablations/summary.json`
- `results/exact_law/summary.json`
- `results/optional_benchmarks/summary.json`
- `results/claims_status.json`
- `results/figures/figures.json`

Key CSVs:

- `results/controlled_energy_tail/summary.csv`
- `results/learned_ibc/summary.csv`
- `results/learned_torch_ibc/summary.csv`
- `results/benchmarks/metaworld/summary.csv`
- `results/multimodal_action/summary.csv`
- `results/optimization_budget/summary.csv`
- `results/repair/summary.csv`
- `results/ablations/summary.csv`
- `results/exact_law/validation.csv`
- `results/optional_benchmarks/summary.csv`

Seed-level CSVs:

- `results/controlled_energy_tail/seed_level.csv`
- `results/learned_ibc/seed_level.csv`
- `results/learned_torch_ibc/seed_level.csv`
- `results/benchmarks/metaworld/seed_level.csv`
- `results/multimodal_action/seed_level.csv`
- `results/optimization_budget/seed_level.csv`
- `results/repair/seed_level.csv`
- `results/ablations/seed_level.csv`

Figures:

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

## 3. Strongest Low-Energy Tail Miscalibration Artifact

Path: `results/controlled_energy_tail/seed_level.csv`

Strongest row: `task=multimodal`, `model=raw_miscalibrated`, `seed=0`,
`N=128`.

- N values: `1, 2, 4, 8, 16, 32, 64, 128`
- Selected energy at N=128: `-2.9748`
- Selected real utility at N=128: `-1.5131`
- Low-energy tail real utility at N=128: `-1.4994`
- Tail rank correlation at N=128: `-0.2008`
- High-N regret: `3.1830`
- Invalid action rate: `1.0000`
- Contact feasibility penalty: `0.9840`
- Deployment gate: `collect_pilot_labels`

This remains the cleanest proof-of-phenomenon: selected energy improves while
selected utility collapses because the low-energy tail is invalid.

## 4. Strongest Learned IBC-Style Artifacts

NumPy failure artifact:

- Path: `results/learned_ibc/summary.json` and `results/learned_ibc/summary.csv`
- Model type: NumPy linear-feature EBM.
- Objective: InfoNCE-style contrastive classification over one expert action
  and negative action samples.
- Raw EBM selected energy by N: `141.1534, 35.4146, 6.7477, -3.4590, -7.5547, -9.4334, -10.4285, -11.0067`.
- Raw EBM selected real utility by N: `0.3627, 0.2044, -0.1251, -0.4753, -0.7424, -0.9296, -1.0671, -1.1628`.
- Calibrated high-N utility gain over raw: `2.8207`.

PyTorch aligned-tail artifact:

- Path: `results/learned_torch_ibc/summary.json` and `results/learned_torch_ibc/summary.csv`
- Model type: PyTorch MLP EBM.
- Objective: InfoNCE-style contrastive classification.
- Training loss decreased from `1.6582` to `0.0123`.
- Strongest raw torch row: `seed=2`, `N=128`, selected utility `1.5847`,
  exact-law error `0.0043`, invalid rate approximately `0`, tail rank
  correlation `0.4082`, gate `stop_early`.

Together these artifacts are stronger than v1: one learned EBM exhibits the
failure mode, while the richer neural EBM shows the aligned-tail regime.

## 5. Strongest Benchmark Artifact

Path: `results/benchmarks/metaworld/summary.json` and
`results/benchmarks/metaworld/summary.csv`

- Benchmark: Meta-World `reach-v3`.
- Status: `SUPPORTED` by claim audit.
- Seeds: `0, 1, 2`.
- Candidates: 8 states x 128 actions per seed for evaluation.
- Model: PyTorch MLP EBM trained on high-reward sampled actions.
- Strongest raw benchmark row: `seed=0`, `N=128`, selected reward `1.6220`,
  selected energy `-8.0680`, exact-law error `0.0005`, high-N regret `0.2990`,
  tail rank correlation `0.1766`, gate `stop_early`.

Boundary: this is a simulator benchmark artifact, not real-robot validation.

## 6. Strongest Repair and Ablation Artifacts

Repair path: `results/repair/summary.json` and `results/repair/summary.csv`

At `N=128`:

- Raw EBM utility: `-1.3545`, CI `[-1.3710, -1.3381]`, gate `collect_pilot_labels`.
- Calibrated utility: `1.9986`, CI `[1.9653, 2.0319]`, gate `stop_early`.
- Value-shaped utility: `1.7511`, CI `[1.7188, 1.7834]`, gate `stop_early`.
- Support-penalized utility: `1.7742`, CI `[1.7173, 1.8312]`, gate `stop_early`.
- Oracle utility: `2.0909`, CI `[2.0846, 2.0971]`, gate `stop_early`.

Ablation path: `results/ablations/summary.json` and
`results/ablations/summary.csv`

- Best pilot-label result: 512 labels, high-N utility `2.0318`, CI
  `[2.0215, 2.0421]`, utility gain over raw `3.3843`, gate `allow_high_n`.
- Best support-weight result: weight `4.0`, high-N utility `1.7793`, CI
  `[1.7680, 1.7907]`, utility gain over raw `3.1318`.

## 7. Compute/Latency Artifact

Path: `results/optimization_budget/summary.csv`

- Energy evaluations are reported as `N * (refinement_steps + 1)`.
- Runtime proxy is the same deterministic evaluation count.
- The raw controlled compute frontier shows that additional evaluations can
  lower energy while decreasing real utility.
- Meta-World compute frontier is plotted in
  `results/figures/figure10_metaworld_compute_frontier.png`.

## 8. EBM Validity Checklist

- Learned energy function exists: yes, NumPy and PyTorch EBMs.
- Trained on observation-action expert/high-value data: yes.
- Negative samples used: yes.
- Inference selects low-energy actions: yes, all theorem code uses `S = -E`.
- Real utility measured separately: yes, toy utilities and Meta-World rewards.
- Low-energy tail diagnostics exist: yes.
- Direct MSE regression is not the main EBM policy: yes, MSE is only a baseline.

## 9. Not-Clone Checklist

- Reused math: finite tie-aware Best-of-N selection law only.
- New scientific object: conditional action/trajectory energy `E_theta(o, a)`.
- New failure mode: low-energy tail miscalibration.
- New experiments: contact/action validity, learned IBC EBM, PyTorch EBM,
  Meta-World reach-v3 benchmark, calibration/support ablations, compute
  frontier.
- Forbidden clone patterns absent from supported claims: yes, claim audit
  passes all 14 not-clone checks.

## 10. Paper-Readiness Judgment

Judgment: stronger paper-quality v2. The repo now supports a mechanistic paper
with toy, learned EBM, neural EBM, and guarded simulator-benchmark artifacts.

Still not a real-robot paper. A CoRL/robotics submission would need broader
benchmark coverage and ideally a real manipulation policy, but the repository is
now substantially stronger than the original toy-only v1.

## 11. Top Remaining Weaknesses

- No real-robot validation.
- Meta-World coverage is currently `reach-v3` only.
- ManiSkill and robosuite are import-probed but not promoted to supported
  rollout artifacts.
- PyTorch benchmark training uses high-reward sampled actions, not expert human
  demonstrations.
- Simulator success is zero for one-step reach-v3, so benchmark support is
  reward-based rather than success-based.

## 12. Exact Next Steps

1. Add Meta-World `push-v3` or another contact-rich task.
2. Promote ManiSkill or robosuite from availability probe to finite rollout
   artifact if stable.
3. Add a multi-task benchmark table across at least three manipulation tasks.
4. Add a bibliography and convert the paper skeleton into a draft.
5. Add real-robot or high-fidelity policy evidence before claiming robot
   deployment validation.
