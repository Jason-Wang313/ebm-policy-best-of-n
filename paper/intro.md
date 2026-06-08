# Introduction

Energy-based policies are attractive for robot control because they can model
multimodal and discontinuous action distributions. A single observation may
permit two grasps, two insertion approaches, or two routes around an obstacle,
and a direct MSE action regressor can average these modes into a poor action.

The price of this flexibility is inference. An EBM policy must select or
optimize actions by energy. The natural inference-time scaling knob is to
sample more candidates, score them, and choose the lowest-energy one.

But high `N` selection is extreme-tail optimization. The selected action is not
a typical sample from the policy distribution; it is the action that most
exploits the learned energy surface among the candidates. Therefore low energy
must be evaluated by selected-tail real utility, not only by average energy,
demo likelihood, or global rank correlation.

This repository supports a paper skeleton with four ingredients: an exact
finite Best-of-N law, EBM-specific tail diagnostics, controlled and learned toy
evidence, and repair plus compute-frontier experiments. The central claim is
conditional: Best-of-N inference helps when the low-energy tail is aligned with
real utility and can fail when that tail contains low-energy false positives.
