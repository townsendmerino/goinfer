# fused_rms_qkv / fused_rms_gu: fewer redundant prologues, bit-identically (+2-3.5% decode; a 4x-unroll null result)

Continues `d7-decode-breakdown-2026-09-21.md`. That profile showed D7 decode GPU-bound with `fused_rms_qkv` at ~163 GB/s (1.39 ms/token) and `fused_rms_gu` at ~316 GB/s (5.9 ms/token, 48%).
Instrument: `ncu --metrics gpu__time_duration.sum` on kernel launches over random int4 weights of each real geometry (`cuda/fused_qkv_rows_test.go`: `TestFusedQKVRowsBench`,
`TestFusedGUBench`), RTX 2070 SUPER, driver `595.91.07`, idle box. Raw CSVs in `fused-rms-qkv-2026-09-21/`.

## The mechanism, and a change that cannot alter a logit

Both kernels give every block a redundant prologue: recompute the layer's rmsnorm and int8 quantisation of x[H] (three block-wide reductions, ~16 barriers) into shared memory before reading
any weight. `fused_rms_qkv` gave each warp ONE row, so D7 ran 576 blocks paying that prologue. Letting each warp walk `rowsPerWarp` rows off one shared activation divides the redundant
prologues by `rowsPerWarp` (`fused_rms_gu` already did it, hard-coded to 8). Two new kernels in their OWN modules (`fused_qkv_rows.cu`, `fused_gu_rows.cu`; the audited `fused_qkv.ptx` is not
regenerated) take it as a runtime parameter. **Bit-identical by construction** — the prologue is a verbatim copy and every output row's arithmetic is the original per-warp sequence; only which
warp computes which row changes — and checked, not assumed:

- `TestFusedQKVRowsBitIdentical`: 128,016 output elements, 4 geometries x with/without bias x rows-per-warp {1,2,3,4,6,8,16}: 0 differing bits.
- `TestFusedGURowsBitIdentical`: 1,247,232 output elements, 9 geometries x rows-per-warp {1,2,3,4,8,16,32}: 0 differing bits.
- **Decode logits, end to end:** SHA-256 over a 300-token prefill + 24 decode steps' full logits on the 1.5B, gemma3-1b, the 0.5B and D7, before and after wiring BOTH kernels in:
  **identical on all four** (D7 takes both new kernels; gemma3-1b takes the gate/up one; the 0.5B correctly keeps both originals). A scratch test (a golden hash would break on any legitimate numerics change).

## fused_rms_qkv (kernel time, us; median of 3 launches)

| geometry (rows) | original | best rows-per-warp | best | rule's pick |
|---|---:|---|---:|---|
| D7 qwen2.5-7b (4608) | 50.5 | 8 | 32.5 (0.64x) | 8 |
| qwen2.5-3b (2560) | 18.0 | 4 | 13.6 (0.76x) | 4 |
| 1.5B (2048) | 15.9 | 2-4 | 11.5 (0.72x) | 4 |
| qwen3-4b (6144) | 46.2 | 8 | 30.3 (0.66x) | 8 |
| qwen3-1.7b (4096) | 27.6 | 8 | 18.4 (0.67x) | 8 |
| llama3-8b/mistral-7b (6144) | 69.1 | 16 (flat 4-16) | 47.2 (0.68x) | 8 |
| gemma3-1b (1536) | 12.1 | 2 | 9.2 (0.76x) | 2 |
| phi3-mini (9216) | 71.2 | 8 | 46.1 (0.65x) | 16 |
| olmo/llama-1b (3072) | 21.2 | 4 | 15.4 (0.72x) | 4 |
| 0.5B (1152) | 8.4 | 1 | no gain (rows-per-warp 2 is 6% worse) | 1 (floor) |

The best setting lands where the grid is ~64-100 blocks, so the rule is: the largest power of two <= 16 that still leaves >= 64 blocks, for projections of >= 1536 rows, else 1
(`fusedQKVRowsPerWarp`). A wrong pick can only cost speed.

**Served, same session, three engines** (`bench_peer.py`; goinfer new / previous build, both with the lane on / Ollama v0.32.5; loadavg per cell 0.50-1.00, one sweep):

| model | depth | new | previous | new / previous | new / Ollama |
|---|---:|---:|---:|---:|---:|
| 0.5B | 128 / 2048 / 3900 | 336.3 / 309.4 / 303.8 | 341.3 / 310.5 / 310.5 | 0.985 / 0.996 / 0.978 | 1.26 / 1.15 / 1.17 |
| 1.5B | 128 / 2048 / 3900 | 245.5 / 221.4 / 208.7 | 241.0 / 217.5 / 204.9 | **1.019 / 1.018 / 1.019** | 1.26 / 1.23 / 1.19 |
| D7 | 128 / 2048 / 3900 | 78.9 / 74.1 / 70.7 | 76.2 / 71.7 / 68.9 | **1.035 / 1.033 / 1.026** | 1.06 / 1.04 / 1.01 |

The 0.5B cells read -0.4 to -2.2% on a path this change leaves on the ORIGINAL kernel (rows-per-warp 1), so that is run-to-run noise (the 0.5B has the widest spread of the three), not an effect. D7 goes from
0.99-1.03x to **1.01-1.06x of Ollama**. In situ (ncu, D7, ~130-token prompt) `fused_rms_qkv` is 50.7 -> **33.1 us/layer** and GPU time per token 12.78 -> **12.31 ms (-3.7%)**.

## fused_rms_gu: a null result, then a smaller table

- **4x load-unroll ladder (4x -> 2x -> 1x), bit-identical, NO gain.** Hypothesis from the GEMV study (`docs/completed/task-cuda-cgofree-spike.md`: COAL4 lost on the 1.5B only to a large 1x
  remainder): more loads in flight. Result: ratio 1.000 on every large geometry (D7 215.0 vs 215.1 us; llama3-8b 180.0 vs 179.7), 1.014-1.065 (WORSE) on the small/odd ones. The kernel
  is not limited by loads in flight. **Dropped; nothing of it ships.**
- **Rows-per-warp as a runtime parameter** (the QKV idea), 15 launches per cell, spreads 1-10 us. The win is real but irregular — it jumps at grid-size thresholds (wave quantisation on 40 SMs), so no
  simple rule fits (rows-per-warp 16 is -13% on qwen3-1.7b and -8% on gemma3-1b but +33% on the 0.5B and +4% on qwen2.5-3b, which prefers 32 at -11%). Shipped as a **table of measured
  geometries** (`fusedGURowsPerWarp`); every other geometry keeps the original kernel (8). Medians: D7 214.9 -> 208.0 (16), qwen2.5-3b 76.9 -> 68.1 (32), qwen3-4b 79.6 -> 76.2 (16), qwen3-1.7b
  46.3 -> 40.3 (16), llama3-8b 179.6 -> 164.2 (32), gemma3-1b 35.5 -> 32.7 (16), phi3-mini 79.1 -> 75.7 (16); the 1.5B and 0.5B stay original.
- **In situ on D7:** `fused_rms_gu` 215.6 -> **210.9 us/layer**, GPU time per token 12.31 -> **12.20 ms (-0.9%)**. Not measured separately in a served run: ~1% is inside the served noise (a table entry
  is a bet on the microbenchmark, guarded by bit-identity).

## Cumulative, D7 GPU time per token (ncu, ~130-token prompt, lane on)

13.39 ms (start of this thread) -> 12.78 (`glu_quant` 1024 threads) -> 12.31 (`fused_rms_qkv` rows-per-warp) -> **12.20 (`fused_rms_gu` table)**: **-8.9%**, every step bit-identical.

## Left on the table, and what this does not establish

`gemv_w4a8_fwd` (down/o projections, 3.16 ms/token, ~57 us avg, ~350 GB/s), the LM head (1.3 ms at ~420 GB/s, near the roof), `glu_quant` (13.6 us/layer, 0.37 ms/token), the lane's `fa_combine`, the wall-vs-GPU
gap (~3%). The rows-per-warp tables are per geometry measured on THIS card with random weights (real weights have the same access pattern, but nothing was checked on real ones beyond the in-situ D7
profile). Geometries outside the tables run the original kernels. Nothing was re-measured on the CPU/Metal/WebGPU backends (CUDA-only change).
