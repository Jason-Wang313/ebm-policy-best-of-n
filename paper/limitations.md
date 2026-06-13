# Limitations

- The core evidence includes controlled toys and guarded Meta-World
  `reach-v3`, `push-v3`, `pick-place-v3`, and `button-press-v3` simulator
  artifacts when the local benchmark run succeeds.
- No real-robot validation is claimed.
- Benchmark success is measured after executing the selected action followed by
  scripted continuation, not as autonomous long-horizon policy deployment.
- The separate closed-loop Meta-World dependency audit includes ungated local,
  nearest-demo, behavior-cloned, and state-heuristic proposal variants.
  State-heuristic no-gate variants clear the current low-dependency threshold,
  but autonomous learned-policy success remains unsupported because the
  nearest-demo, behavior-cloned, and local learned variants do not clear their
  learned threshold.
- Reward-only or zero-success simulator artifacts are diagnostic only, not broad
  manipulation success.
- The learned EBMs are still small relative to modern robot policies.
- Scripted expert actions are used for benchmark positives; this is stronger
  than high-reward sampling but still not human demonstrations or real robot
  teleoperation.
- Some failure modes are constructed to isolate the mechanism.
- The theorem is conditional on a fixed generator/scorer stack.
- Calibration needs pilot real-utility labels.
- The toy utility function is known by the experimenter.
- Real manipulation validation is future work.
