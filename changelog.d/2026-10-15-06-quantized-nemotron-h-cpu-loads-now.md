- **Quantized Nemotron-H CPU loads now hold the Mamba projections at int8** (about 12 GB less on the 9B), from a .gguf or
  a .giw; an f32 load keeps them f32. The GPU backends upload the same codes as before.
