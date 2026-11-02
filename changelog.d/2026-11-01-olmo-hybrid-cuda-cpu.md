- **Olmo Hybrid runs on the CPU when CUDA is asked for.** CUDA's Gated DeltaNet did not apply two of Olmo Hybrid's settings,
  `linear_allow_neg_eigval` (the widened write gate) and its own output-norm epsilon, so the resident computed the family
  differently from the CPU. CUDA now declines it to the CPU, naming the missing parameter, as Metal does.
