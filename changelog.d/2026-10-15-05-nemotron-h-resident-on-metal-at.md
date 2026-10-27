- **Nemotron-H resident on Metal, at int4.** The Mamba-2 mixer and the squared-ReLU MLP run on the GPU; an int8
  Nemotron-H stays on the CPU rather than re-quantize. Gated on tiny models, a three-layer slice of the real
  Nemotron-Nano-9B-v2, and the whole 9B against the CPU (min logit cosine 0.929, argmax 27 of 32).
