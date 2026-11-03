- **Metal frees the batched step's startup probes.** The check that `simd_sum` matches the batched kernels' addition tree left its
  544 KiB of buffers allocated for the life of every resident with a batched step. They are now freed when the check ends.
