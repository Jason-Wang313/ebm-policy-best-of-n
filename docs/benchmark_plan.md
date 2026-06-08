# Optional Benchmark Plan

Core runs include CPU toy experiments plus a guarded Meta-World reach-v3
benchmark attempt. Optional non-Meta-World suites remain guarded so missing or
unstable dependencies do not break smoke, full, audit, or tests.

Possible extensions:

- Meta-World reach-v3 is the primary supported benchmark target when artifacts
  are generated under `results/benchmarks/metaworld/`.
- Gymnasium continuous-control tasks if installed.
- Simple navigation tasks with continuous action energies.
- FetchReach or FetchPush if Gymnasium robotics dependencies are already
  available.
- D4RL-style generated toy datasets without requiring the real D4RL package.
- ManiSkill and robosuite only if they can produce stable finite artifacts.

Claim boundary:

- Meta-World claims are SUPPORTED only when finite CSV/JSON/figure artifacts
  exist and pass `scripts/run_claim_audit.sh`.
- Optional non-Meta-World benchmark claims are UNSUPPORTED until artifacts are
  present.
- No real-robot validation is claimed.
- Benchmark adapters must write JSON, CSV, seed-level CSV when relevant, and
  figures regenerated from CSV/JSON.
