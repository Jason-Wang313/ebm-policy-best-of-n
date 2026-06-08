# Reviewer Attacks and Honest Responses

1. **This is only a toy.**  
   Correct. The core evidence is controlled and learned toy evidence. The value
   is isolating a deployment-tail failure mode before claiming robot-scale
   validation.

2. **The theorem is reused from the Best-of-N/WAM paper.**  
   The finite selection identity is reused as math. The scientific object here
   is a conditional action energy and its low-energy tail, not imagined rollout
   dynamics.

3. **This is just reward-model misalignment.**  
   It is related, but the operational object is EBM inference: selecting
   minimum-energy actions from candidate sets. The diagnostics target the
   selected low-energy tail.

4. **This is just distribution shift.**  
   Distribution shift is one cause. The paper isolates the selection mechanism:
   larger `N` puts more pressure on the low-energy tail and can amplify hidden
   false positives.

5. **Why is this specifically EBM?**  
   Because inference directly optimizes `E_theta(o, a)` over sampled or refined
   actions. The failure is not only training error; it is the deployment tail of
   an energy scorer.

6. **Low energy was never supposed to equal real utility.**  
   Agreed. The paper does not require equality. It requires that the selected
   low-energy tail be sufficiently aligned with real utility for Best-of-N to
   help deployment.

7. **Why not just train a better EBM?**  
   Better training can help. The repository tests repairs that use pilot
   utility labels, value shaping, and support penalties. The point is to audit
   whether the deployed tail improved.

8. **Does calibration solve everything?**  
   No. Calibration helps in the toy artifacts when it changes selected-tail
   ranking. It is not claimed as a universal fix.

9. **Does high N always hurt?**  
   No. Oracle and tail-aligned energies improve or saturate. The problem is
   miscalibrated low-energy tails.

10. **Where is robotics?**  
    The repository uses robot-action toy tasks with contact, support, jerk, and
    multimodal action structure. Real robot validation is future work.

11. **Is this different from JEPA latent tail hallucination?**  
    Yes. JEPA concerns latent predictive scores. This project concerns action
    energies and real utility after selecting low-energy actions.

12. **Is this different from diffusion reranking?**  
    Yes. Diffusion work centers on generated trajectory diversity and sampling.
    Here the scorer is a conditional energy over actions and the tail is a
    low-energy action tail.

13. **Is this different from WAM model-error amplification?**  
    Yes. There is no imagined rollout model in the core theorem or toy failure.
    The failure is an energy scorer assigning low energy to bad actions.

14. **Is the learned EBM real or hand-designed?**  
    The learned artifact trains a small NumPy IBC-style EBM with contrastive
    positives and negative action samples. The toy utility is constructed, but
    the raw energy used in that experiment is learned.

15. **What would make this publishable at ICLR/CoRL?**  
    Stronger benchmark or real-robot evidence, richer learned EBMs, and
    ablations showing the diagnostics predict when high-N inference is safe.

16. **What is the actual scientific contribution?**  
    Reframing inference-time scaling for EBM robot policies as a low-energy
    tail calibration problem, with exact finite laws, diagnostics, repairs, and
    compute-utility frontiers.

17. **What experiments would be needed for real robot credibility?**  
    A real or high-fidelity manipulation EBM policy, pilot utility labels,
    high-N inference sweeps, contact/safety metrics, and before/after repair
    validation with deployment gates fixed before evaluation.
