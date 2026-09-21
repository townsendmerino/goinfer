# Why fused_rms_gu is 10% slower than the GEMV: the redundant prologue, and wave quantisation

The question from `gemv-w4a8-2026-09-21.md`: `fused_rms_gu` (47% of a D7 token) runs at ~316 GB/s while the plain down-projection GEMV, with the same inner loop, runs at ~362. Instrument: `ncu gpu__time_duration.sum`,
random int4 weights, 15 launches per cell, RTX 2070 SUPER (40 SMs), idle box. Raw data and the scratch ablation kernel source are in `fused-rms-gu-diagnosis-2026-09-21/`.

## 1. Is it the shape or the fusion? Equal-bytes reference (67.9 MB per launch)

The plain GEMV at equal bytes but different row lengths: 37888x3584 (the gate/up shape: short rows, 448 words) **352 GB/s** (192.6 us); 7168x18944 (the down shape, 2368 words) 370 GB/s; 18944x7168 371; 14336x9472 365;
3584x37888 370; 75776x1792 (very short rows) 328. So row length costs a little (352 vs 370, ~5%), but the plain GEMV on the EXACT gate/up shape is still 192.6 us against the fused kernel's **214.6 us: ~22 us (11%) belongs to the fusion**, not the shape.

## 2. Ablation: which part of the fusion (D7 gate/up, 592 blocks of rows-per-warp 8)

| variant | us | GB/s |
|---|---:|---:|
| full fused kernel (prologue + shared-memory activation) | 215.1 | 316 |
| **no prologue math**: activation copied from a precomputed global buffer into the same shared array | 190.9 | 356 |
| no prologue, no shared memory: activation read from global (L1-cached), like the plain GEMV | 189.1 | 359 |
| prologue only (rmsnorm + int8 quantisation, no rows) | 27.8 | — |

The whole gap is the **prologue**: 215.1 - 190.9 = **24 us**. The shared-memory read path costs only 1.8 us (1%), so the GEMV study's shared-staging slowdown was a different effect (per-block staging), not what limits this kernel.
Every block first recomputes the layer's rmsnorm + int8 quantisation (dependent global loads, two block-wide reductions, ~20 barriers) before it streams a single weight, and that work does not overlap the block's own DRAM traffic.

## 3. What did NOT help: cheaper barriers

A bit-identical prologue with the last five steps of the sum ladder as warp shuffles (identical pairing) and the max reduction fully in shuffles (exact), 20 barriers down to 9: 0 of 18,944 rows differ, **215.1 -> 211.1 us**
(-1.9%). Barriers are not the main cost; the serial dependency chain and the lack of overlap are. Not adopted (no numerics risk taken for 2%).

## 4. What did help: size the grid to ONE resident wave

Sweeping rows-per-warp beyond powers of two on the ablation kernel (D7 gate/up, us): rpw 8 (592 blocks) 214.6 | 16 (296) 208.9 | 24 (198) 199.8 | 32 (148) **224.1** | 36 (132) **238.5** | **40 (119) 196.9** | 44 (108) 200.2 | 48 (99) 200.0 | 64 (74) 204.2.
Each SM holds 3 of these blocks at once (256 threads; 18.9 KB shared each of 64 KB), so 40 SMs x 3 = **120 resident blocks; 119 is one full wave**. Grids that leave a partly-empty second wave (132, 148) are up to 11% SLOWER than 592;
one full wave with 5x fewer prologues is 8.2% faster. Same signature on the no-prologue kernels (which sit near 190-196 regardless), i.e. the effect is the prologue count and the wave shape, not the streaming loop.

So the earlier table (powers of two, per geometry) was fitting wave quantisation without knowing it, which is why it was irregular (rpw 16 great on one geometry, +33% on another). **The rule** (`waveRowsPerWarpFor`): blocks = (resident blocks per SM, limited by
threads/256 and by the block's shared memory) x (SM count), rows-per-warp = ceil(rows / (8 x blocks)); the device shape is read from the driver attributes at build. Against the original kernel and the shipped table, same random weights (us):

| kernel, geometry | original | shipped pick | **wave rule** | rule vs original |
|---|---:|---:|---:|---:|
| gate/up D7 (rpw 40) | 214.8 | 208.3 | **197.6** | 0.920 |
| gate/up qwen2.5-3b (18) | 77.3 | 68.2 | 68.4 | 0.885 |
| gate/up 1.5B (14) | 48.4 | 48.4 | **44.1** | 0.911 |
| gate/up 0.5B (8 = original) | 24.2 | 24.8 | 24.7 | — |
| gate/up qwen3-4b (16) | 79.4 | 75.3 | 76.9 | 0.969 |
| gate/up qwen3-1.7b (10) | 46.3 | 40.2 | 40.4 | 0.873 |
| gate/up llama3-8b (30) | 179.7 | 164.2 | 167.5 | 0.932 |
| gate/up gemma3-1b (11) | 35.6 | 31.5 | **29.9** | 0.840 |
| gate/up phi3-mini (13) | 79.5 | 75.9 | 75.6 | 0.951 |
| qkv D7 (5) | 50.6 | 32.7 | **31.0** | 0.613 |
| qkv 1.5B (2) | 15.7 | 11.6 | 11.2 | 0.711 |
| qkv qwen3-4b (5) | 46.5 | 30.2 | 29.3 | 0.631 |
| qkv gemma3-1b (2) | 11.9 | 9.3 | 9.2 | 0.771 |
| qkv 0.5B (1 = original) | 8.3 | 8.3 | 8.3 | — |

It is within ~2% of the table where the table was tuned (qwen3-4b and llama3-8b gate/up are ~2% worse), better where the table was not (1.5B and gemma3-1b gate/up, D7), and needs no table: a new geometry gets a sensible pick for free.
The rule assumes registers do not limit occupancy (true for these kernels on this card); if they did, the grid would just be bigger than one wave. **Measured on one card (RTX 2070 SUPER) only**; the kernels are bit-identical for any rows-per-warp, so a bad pick on another device can cost speed, never a logit.

## 5. Bit-identity, and the effect in situ

Decode-logit SHA-256 (300-token prefill + 24 decode steps) with the wave rule active: **identical to the original baselines on the 1.5B, gemma3-1b, the 0.5B and D7.** `TestWaveRowsPerWarpFor` pins the rule's values on this card. In situ on D7 (ncu, ~130-token prompt, lane on):
`fused_rms_gu_rows` 210.9 -> **201.2 us/layer** (-4.6%), `fused_rms_qkv_rows` 33.0 -> 31.4, GPU time per token **12.20 -> 11.87 ms (-2.7%)**. Cumulative over this thread: **13.39 -> 11.87 ms (-11.3%)**, every step bit-identical. Served result: section 6.

## 6. Served, same session (`bench_peer.py`; goinfer with the wave rule / previous build, both lane on / Ollama v0.32.5; per-cell loadavg 0.79-1.00, one sweep)

| model | depth | new | previous | new / previous | new / Ollama |
|---|---:|---:|---:|---:|---:|
| 0.5B | 128 / 2048 / 3900 | 340.7 / 304.8 / 301.2 | 342.0 / 311.3 / 307.6 | 0.996 / 0.979 / 0.979 | 1.27 / 1.13 / 1.16 |
| 1.5B | 128 / 2048 / 3900 | 253.2 / 228.1 / 214.8 | 245.8 / 221.4 / 208.5 | **1.030 / 1.030 / 1.030** | 1.30 / 1.27 / 1.23 |
| D7 | 128 / 2048 / 3900 | 81.5 / 76.3 / 73.2 | 79.9 / 75.0 / 72.0 | **1.020 / 1.017 / 1.017** | 1.10 / 1.08 / 1.05 |

The 1.5B gains a steady +3.0% (its gate/up now takes rows-per-warp 14, which the table missed), D7 +1.7-2.0% (its GPU-time gain in situ was 2.7%; end to end it shows smaller). **The 0.5B reads -0.4 to -2.1% on a path that runs the
IDENTICAL original kernels under both builds** (the rule picks rows-per-warp 8 for its gate/up and 1 for its QKV): two sweeps in a row lean slightly negative there and I cannot explain it; it is inside that model's
run-to-run spread (up to ~3%) and is recorded, not explained. D7 vs Ollama: 1.05-1.10x. One sweep, no repeat; cross-session drift (~3.5% on this box) is not bounded.

## What this does not establish

Why the prologue does not overlap (not profiled below "it is dependent global loads plus reductions ahead of the block's first weight load"); a persistent-block scheme or a prologue-once kernel could remove more of the remaining ~10 us (201 vs the ~190 us no-prologue floor) but neither was built. The
`gemv_w4a8_fwd` down/o projections are untouched (at their bandwidth bound, `gemv-w4a8-2026-09-21.md`).
