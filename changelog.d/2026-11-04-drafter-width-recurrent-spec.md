- **Block-drafter speculation on recurrent Qwen3.5 hybrids on CUDA and `--drafter-width`.** `--drafter` on CUDA now admits recurrent
  Qwen3.5 hybrids by checkpointing and restoring the target's `DeltaNet` conv and recurrent state around each verify step (`1.61×`
  decode on Qwen3.5-4B with a DFlash block drafter, `1.32×` on Qwen3.6-35B-A3B under expert caching, greedy bit-identical).
  `--drafter-width` sets the verify width for `--drafter` (default `0`, meaning 8; values below 4 are refused).
