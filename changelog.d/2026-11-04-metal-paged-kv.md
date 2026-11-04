- **Paged KV on Metal.** Metal's resident KV cache now uses a 64-position block pool with per-slot block tables and copy-on-write prefix
  sharing (`--kv-share`), bit-identical to contiguous slot storage across all 29 gated families. Decode is within 1% on the 1.5B
  (`0.990`–`1.009×` across depths 128 to 8,000) and the 7B (`0.997`–`1.018×`), and `gemma3-1b` is about 3–4% slower at long context
  (`0.975` / `0.970×` at depth 2,048 and `0.963` / `0.956×` at depth 3,900, accepted by Francis).
