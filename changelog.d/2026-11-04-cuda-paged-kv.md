- **Paged KV on CUDA.** CUDA's resident KV cache is now a pool of 64-position blocks per layer group (full-attention and sliding-window)
  behind per-slot block tables, bit-identical to contiguous slot storage. Each slot keeps its guaranteed share while one long
  conversation can borrow free and idle blocks across the pool (2.71× the single-conversation context on Qwen2.5-7B on an 8 GB card at
  the default 4 slots). Prefix sharing (`--kv-share`) shares full-group blocks by reference with copy on write instead of copying rows
  between slots, `/v1/models` and the startup banner report the pool (`kv_share`), responses add `usage.prefill_reuse` (`own`, `shared`,
  `host`, `disk`, `promoted_blocks`), and `goinfer-serve check` verifies shared-preamble reuse. Four-client batched decode reads `0.973×`
  against contiguous slots (about 2% slower, accepted by Francis).
