# Perf review finding 6: CUDA Phi-3 decode at 128, the launch census and the per-32 activation kernels (2026-11-23)

## Census (tier 0 counters, then ncu)

`backend/cuda/kv_slots_test.go:TestCost_cudaDecodeLaunchCensus` on `~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf` at
`q4k` (the default), depth 128, nobara (RTX 2070 SUPER, driver 595.112.07), base `1e69062e9`: **418 launches a token, 13.06 a layer**
(`fuseQKV` and `fuseG32` off, per-32 activations, hidden 3,072). The review's 419 holds.

| kernel | launches a token | a layer |
|---|---|---|
| `gemvQ4K` | 112 | 3.50 |
| `gemvQ5K` | 96 | 3.00 (q, k, v) |
| `fRms` (`rmsnorm_quant_g32`) | 65 | 2.03 |
| `bAttn`, `fQ` (`quant_vec_g32`), `fSw` (`glu_quant_g32`), `ropeKV` | 32 each | 1.00 |
| `gemvQ6K` | 17 | 0.53 |

GPU time of the same token (ncu `gpu__time_duration.sum`, depth 2, the last token's 418 launches; `census/ncu-d2.log.gz`): 7.56 ms,
against about 7.9 ms of wall at 126.6 tok/s, so decode is GPU-bound and a launch costs only what its kernel takes.

| kernel | launches | mean | share |
|---|---|---|---|
| `gemv_q4k_g32` | 112 | 33.2 µs | 49.1% |
| `gemv_q5k_g32` | 96 | 17.9 µs | 22.7% |
| `gemv_q6k_g32` | 17 | 61.4 µs | 13.8% |
| `rmsnorm_quant_g32` (1 block) | 65 | 5.4 µs | 4.6% |
| `glu_quant_g32` (1 block) | 32 | 10.8 µs | 4.6% |
| `attn_batched` | 32 | 6.2 µs | 2.6% |
| `quant_vec_g32` (1 block) | 32 | 3.7 µs | 1.6% |
| `rope_kv` | 32 | 2.3 µs | 1.0% |

**Reading:** the three single-block per-32 kernels are 10.8% of the token, past the finding's 3% kill bar, and they are the lever.
The q5k GEMVs already run near bandwidth (6.75 MB in 17.9 µs, about 377 GB/s), so concatenating q‖k‖v (and gate‖up) at upload would
save only launch tails, and the norm-folded fusion already measured slower at 3,072 (`backend/cuda/resident.go:fusedG32MaxHidden`).
Neither is pursued here.

## The change (`25c7ad334`)

The per-32 quantizers ran one block, a thread a group, each thread looping over its group's 32 elements. `quantG32Lane` quantizes a
group across a warp instead, and `rmsnorm_quant_g32`, `quant_vec_g32` and `glu_quant_g32` take a warp a group at any grid; `actCfg` gives
them a block for every blockDim/32 groups (12 blocks for the 3,072-wide norm and quant, 8 for the 8,192-wide GLU). Bit-identical by
construction: the max is exact and order-free, the code sum an exact integer sum, every element's arithmetic unchanged, and the norm's
sum-of-squares ladder the same in every block.

- Tier 0: `TestActGroupKernels_warpQuantizersEqualSerial` against the previous bodies (kept as `*_serial`, test-only), grids 1, 3 and
  full, a zero group and a NaN, both GLU activations: bit-identical; red with the lane's rounding planted to round-toward-zero. Every
  other kernel of `actgroup.ptx` is unchanged (entry-by-entry, labels normalized). `TestActCfg_spreadsOnlyPer32` pins the grids.
- Real model: `logits_scratch_test.go.txt` built into both arms, 128 positions then 32 greedy decode steps on Phi-3 mini q4k: SHA-256
  over every decode logit `3d770fdb20bcaa75c4b9af8ffbb5cbe476aac10c8c1a2ac7e21508f4269ee5e6` on both, the same 32 ids.
