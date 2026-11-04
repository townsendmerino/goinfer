- **Olmo Hybrid runs resident on CUDA and Metal (`int4` only on Metal).** CUDA's and Metal's Gated DeltaNet kernels now apply
  Olmo Hybrid's `linear_allow_neg_eigval` (the write gate widened to `[0, 2)`) and its own output-norm epsilon (`1e-5` beside a
  `1e-6` layer norm), gated against the CPU on `olmo_hybrid-tiny` and the real 7B checkpoint on CUDA. Metal runs Olmo Hybrid at
  `int4` only (an `int8int8` or `int8` load declines to the CPU), and WebGPU has been read only on the M1 Pro's adapter.
