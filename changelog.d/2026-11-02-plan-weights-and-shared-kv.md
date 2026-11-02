- **The plan prices three more things as the backends allocate them.** A Gated DeltaNet layer's small f32 weights count, a Gemma 4
  E-model's KV-shared layers no longer count as owning a cache on the CPU and WebGPU, and Mamba-2 projections held as f32 count as the
  int8 a resident backend uploads. A planned context can move slightly either way on those families.
