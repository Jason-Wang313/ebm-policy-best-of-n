# Optional Benchmark Plan

Core runs include CPU toy experiments plus a guarded Meta-World benchmark
ladder over `reach-v3`, `push-v3`, `pick-place-v3`, and `button-press-v3`.
Optional non-Meta-World suites remain guarded so missing or unstable
dependencies do not break smoke, full, audit, or tests.

Possible extensions:

- Meta-World `reach-v3`, `push-v3`, `pick-place-v3`, and `button-press-v3` are
  the primary benchmark ladder when artifacts are generated under
  `results/benchmarks/metaworld/`.
- The primary benchmark EBM is trained on scripted expert actions from
  Meta-World policies. High-reward sampled positives are an ablation only.
- If a Meta-World task is unstable, a deterministic contact/push fallback may
  be emitted for that task and the task claim must be PARTIAL.
- Gymnasium continuous-control tasks if installed.
- Simple navigation tasks with continuous action energies.
- FetchReach or FetchPush if Gymnasium robotics dependencies are already
  available.
- D4RL-style generated toy datasets without requiring the real D4RL package.
- Gymnasium Pusher-v5, ManiSkill PushCube/PickCube, and robosuite Lift/Door
  only when they can produce stable finite rollout artifacts in the local
  headless environment.
- Closed-loop Meta-World dependency audits may be reported only when all twelve
  policy variants, proposal source, conservative gate, continuation mode,
  reward, success, learned-demo/behavior-cloned/state-heuristic proposal labels,
  runtime scripted-expert dependency labels, and dependency-drop fields are
  written to `closed_loop_ablation_summary.csv`.

Claim boundary:

- Meta-World claims are SUPPORTED only when finite CSV/JSON/figure artifacts
  exist, success is available and nonzero for the relevant task artifacts, and
  `scripts/run_claim_audit.sh` passes.
- Reward-only or zero-success benchmark artifacts are useful diagnostics but
  must not be used as broad manipulation success evidence.
- Optional non-Meta-World adapter claims are SUPPORTED only for the narrow
  statement that finite guarded rollout artifacts were produced. They do not
  support broad manipulation success; task-level scripted success can be
  mentioned only for generated rows where success is available and nonzero.
- Closed-loop Meta-World dependency audits may support low-dependency
  simulator success only for the proposal family that clears the ungated
  non-expert-centered threshold in `closed_loop_ablation_summary.json`.
  Autonomous learned-policy success remains unsupported unless the nearest-demo,
  behavior-cloned, or other learned no-gate rows clear their separate learned
  threshold.
- No real-robot validation is claimed.
- Benchmark adapters must write JSON plus CSV summaries and seed-level rollout
  CSVs when relevant.
