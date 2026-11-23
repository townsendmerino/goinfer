- **CUDA: a `--quant q4k` load prefills in batched passes, bit-identical to the one-token-at-a-time route it replaces.** Phi-3 and
  Phi-4's default CUDA load used to read every prompt token through a full decode forward ("prefill path: sequential"). Row-batched
  twins of the per-32 kernels now run the q4k lane's `Q4_K`, `Q5_K`, `Q6_K` and `int8` projections over a pass of up to 2,048 rows,
  with output unchanged to the bit. On Phi-3 mini (RTX 2070 SUPER) time to first token falls 2.81× at 128 prompt tokens (953 to
  340 ms) and 1.57× at 3,900 (44.7 to 28.5 s). A pass holds up to 290 MiB of scratch on the card while it runs; the standing memory
  and the longest context that fits are unchanged.
