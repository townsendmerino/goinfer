- **Olmo 3, Olmo Hybrid and SmolLM3 run resident on WebGPU.** WebGPU now implements post-only norms, whole-vector QK-norm, layers
  without position encoding, and Gated-DeltaNet's widened write gate and its own output-norm epsilon. Each holds against the CPU on tiny
  fixtures at int4, int8 and f32, and at `--kv f16` and `--kv i8`. Olmo's prompts run the per-token path. For Olmo Hybrid, Metal is int4
  only, and WebGPU has been read only on the M1 Pro's adapter (not on Vulkan), with no real-checkpoint reading on WebGPU yet.
