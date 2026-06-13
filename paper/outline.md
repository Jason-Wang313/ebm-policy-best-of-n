# Outline

Title: Low Energy Is Not Enough: Tail Audits for Energy-Based Robot Policies

Subtitle: Tail Selection, Tail Calibration, and Compute-Utility Tradeoffs in Implicit Robot Policies

1. Motivation: EBMs can represent multimodal robot actions, but inference
   selects low-energy actions.
2. Problem: tail selection inference is extreme-tail optimization over energy.
3. Theory: finite tie-aware tail selection law for fixed score/utility pools.
4. Diagnostics: low-energy tail utility, tail rank correlation, invalid rate,
   oracle gap, high-N regret, and compute frontier.
5. Evidence: controlled toy tasks, learned IBC-style EBM, multimodal support,
   repair methods, exact-law validation.
6. Boundaries: toy evidence, no real-robot claim, theorem conditional on fixed
   generator/scorer stack.
