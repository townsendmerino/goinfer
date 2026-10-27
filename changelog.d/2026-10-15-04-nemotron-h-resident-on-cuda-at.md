- **Nemotron-H resident on CUDA, at int4.** The same operations on the 8 GB card: the 4B whole (min logit cosine 0.994,
  argmax 30 of 32 against the CPU), a three-layer slice of the 9B, and the whole 9B (0.954, 28 of 32; 47 tokens a second,
  at a 4,096-position context). An int8 Nemotron-H and Nemotron-3-Nano's MoE blocks stay on the CPU; Granite-4.0-H does too.
