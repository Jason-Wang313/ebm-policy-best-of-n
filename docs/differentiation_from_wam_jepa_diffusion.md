# Differentiation from WAM, JEPA, and Diffusion tail selection Projects

## 1. Reused Theorem

This repository reuses only the finite, tie-aware tail-selection law as a
mathematical selection identity. Given a fixed finite pool of candidate actions
or trajectories with scores \(S_i\) and measured utilities \(U_i\), the law gives
the exact expected utility of selecting the maximum-score item after drawing
\(N\) candidates with replacement from that pool. Ties are handled by the mean
utility of the tied top-score group.

The theorem is deliberately object-agnostic. It does not mention imagined
rollouts, latent predictors, diffusion denoisers, robot dynamics models, or EBM
training. In this project it is used only to audit a fixed EBM inference stack:
candidate generator \(q(a \mid o)\), conditional energy \(E_\theta(o,a)\),
score \(S(o,a)=-E_\theta(o,a)\), and separately measured real utility \(U(o,a)\).

## 2. New Scientific Object

The scientific object here is a conditional energy function over robot actions
or trajectories:

\[
E_\theta(o,a), \quad S(o,a)=-E_\theta(o,a).
\]

The policy samples or optimizes candidate actions \(a_1,\ldots,a_N\) from a
proposal \(q(a\mid o)\), then executes the minimum-energy action. The central
question is not whether the model can imagine good futures, recover latent
semantics, or generate diverse trajectories. The question is whether the
low-energy selected tail is aligned with real robot utility after execution.

## 3. Why This Is Action-Energy Tail Calibration

EBM policy inference is an extreme-tail operation. Increasing \(N\) does not
merely average over more actions; it searches harder for unusually low energy.
If the low-energy tail contains out-of-distribution, physically invalid,
unsafe, jerky, contact-bad, or task-wrong actions, then selected energy can
improve while real utility stagnates or falls.

This is not latent planning or rollout hallucination:

- WAM-style failure concerns explicit imagined rollouts and imagined-vs-real
  dynamics mismatch.
- JEPA-style failure concerns latent encoder/predictor scores and latent-real
  rank distortion.
- Diffusion-style failure concerns stochastic trajectory generation,
  diversity, and selection among generated samples.
- This project concerns action or trajectory energies, low-energy tail quality,
  action validity, contact feasibility, smoothness, and compute-utility
  tradeoffs for fixed EBM policy stacks.

## 4. Experiments Natural Here but Unnatural Elsewhere

The natural experiments in this repository directly manipulate EBM action
energies and action-validity tails:

- Score candidate actions by \(E_\theta(o,a)\), select by \(\arg\min E\), and
  measure a separate real utility \(U(o,a)\).
- Construct low-energy false positives: actions that look kinematically
  plausible or imitation-like but are physically invalid, contact-bad, too
  jerky, or outside demonstration support.
- Train a small IBC-style energy model using contrastive positives and negative
  actions, then test whether high-\(N\) inference over-optimizes its learned
  energy tail.
- Compare raw EBM, calibrated energy, value-shaped energy, support-penalized
  energy, conservative stopping, and optional short-horizon verification.
- Plot compute-utility frontiers for sampling and refinement budgets, because
  EBM action selection often spends test-time energy evaluations.

These experiments would be awkward as the primary evidence for WAM, JEPA, or
diffusion projects because they do not require imagined rollout dynamics,
latent predictive targets, or diffusion sample diversity as the failure object.

## 5. Forbidden Clone Claims

The repository must not make claims that collapse this project into the earlier
tail selection repos. Forbidden claims include:

- "We prove EBMs work."
- "We solve robot manipulation."
- "We validate on real robots." unless a real-robot experiment is actually
  implemented and audited.
- "tail selection always helps."
- "Calibration always fixes energy policies."
- "Energy is real utility."
- "Low energy means good action."
- "This is not toy evidence." unless real benchmarks exist.
- "This is a universal training recipe."
- "This is the same as WAM model-error amplification."
- "This is the same as JEPA latent tail hallucination."
- "This is the same as diffusion diversity-selection tradeoff."

The allowed central claim is narrower: tail selection inference for energy-based
robot policies helps only when the low-energy selected tail is aligned with
real utility; otherwise energy can improve while utility saturates or degrades,
and tail calibration, support awareness, or verification can repair the selected
tail in controlled and learned toy settings.
