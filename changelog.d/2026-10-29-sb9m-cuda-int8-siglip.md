- **CUDA's int8 SigLIP tower: exact attention fixed, and the tower walks SigLIP's program** (SB9 step M). The int8 tower
  (`-vision-quant int8`, or the fallback when the float32 tower does not attach) failed to launch its exact attention
  kernel. That path is taken for any SigLIP whose head_dim is not 72, and under `GOINFER_CUDA_VISION_ATTN=exact`; it now
  runs. The tower now walks `vision.SiglipProgram` like every other SigLIP tower. Gemma 3's output is bit-identical on
  every attention arm, and the tower has a golden of its own on CUDA.
