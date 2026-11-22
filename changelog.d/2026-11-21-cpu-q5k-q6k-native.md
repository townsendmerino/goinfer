- **Faster: a Q4_K_M file's Q5_K and Q6_K tensors run as the file stores them on the CPU.** At `--quant q4k`, including
  the default sidecar, they were widened to int8; they now run on new kernels from aikit v1.78.0. Phi-3 mini decodes
  1.10x as fast on a Ryzen (10 of 10 timed pairs), inside its registered fidelity gate. Each token allocates more than
  before: about 266 KB against 201 KB (#1170).
