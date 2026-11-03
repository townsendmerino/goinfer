- **Olmo 3, Olmo Hybrid and SmolLM3 run resident on WebGPU.** WebGPU now implements post-only norms, whole-vector QK-norm, layers
  without position encoding, and Gated-DeltaNet's widened write gate and its own output-norm epsilon. Each holds against the CPU on tiny
  fixtures at int4, int8 and f32. Olmo's prompts run the per-token path. Olmo Hybrid at `--kv f16` or `--kv i8` stays on the CPU.
  No real checkpoint of these families has been measured on WebGPU.
