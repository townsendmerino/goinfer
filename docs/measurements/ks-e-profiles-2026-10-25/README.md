# KS-E1, E4, E5, E6: WebGPU profiles on the Mac (2026-10-25)

Francis's order of 12:30 (relayed): the kernel survey's profiling items that either box can run, measured on the Mac, by day,
under the timing lock held exclusively. Measurement only: nothing is built, and what the readings suggest building is the output.
The items are in `docs/tasks/task-kernel-survey-2026-10.md` §8. MacBook Pro M1 Pro (16 GB), WebGPU (wgpu on Metal), fresh
`origin/main`, checkpoints from `~/models`.

## Registration (before any reading)

- **KS-E1 (int4 decode GEMV).**
  - **Instrument:** `TestW4A8_parity_and_bandwidth`, the 8960 × 1536 gate/up GEMV, parity, then its weight-bytes GB/s from the
    slope of 1 and 200 dispatches. Beside it, `TestGEMV_microbench`'s int8 GEMV GB/s on the same Mac.
  - **Reading:** int4's GB/s ÷ int8's.
  - **Meaning:** under 0.80, the int4 GEMV is compute-bound here and KS-E1's `nib4` + `dot4I8Packed` change has room to build.
    At 0.90 or above it is bandwidth-bound and the change cannot pay. Between them, ambiguous.
  - **Not done:** KS-E1's A/B needs the change built, so it is not part of this measurement.
- **KS-E4 (int4 prefill GEMM) and KS-E6 (prefill attention, several rows per workgroup).**
  - **Instrument:** `TestPrefillInt4_profile` on Qwen3-VL-2B (`~/models/qwen3-vl-2b-instruct`, WebGPU, int4) at 256, 1,024 and
    4,096 rows. It reports the pass's wall time unprofiled, and its split by category: projections, attention, norms and rotary,
    K/V write.
  - **KS-E4's reading:** the projections' share and ms per row. If projections are at least 40% of the pass at 1,024 rows, E4's
    estimated 1.4–2× on that class is a candidate to build.
  - **KS-E6's reading:** the attention share at 4,096 rows. At 30% or more, E6's 22–30% estimate is plausible and E6 is a candidate.
    Under 15% it cannot reach its estimate.
- **KS-E5 (decode attention at depth).**
  - **Instrument:** `TestDecode_dispatchProfile` on the 1.5B (`.gguf`, int8int8, WebGPU) at depths 2,048, 8,192 and 16,384.
    `GOINFER_DECODE_POS` sets the depths and the resident context is sized to the deepest. It ablates each pipeline class from
    the token plan, R = 8 recordings, minimum of 5.
  - **Reading:** the attention classes' ms a token, its share of the token, and µs per position of context.
  - **Meaning:** at 30% or more of a 16k token, (b) split-KV (fidelity-gated) is worth building. (a), the bit-identical V-load
    unroll, is the cheap first step in any case.
- **Cross-box caution.** The survey's figures (250–280 GB/s, 1.18–1.29 ms a row, 1.14 µs a position) come from other sessions, and
  some from nobara. A ratio across boxes or sessions is not a ratio (`docs/measurements/noise-registry.md`), so they are context only.
- **Void.** A run started above a 1-minute load of 4.0, or under 2 GB of reclaimable memory beyond the load's need, is void and is
  re-run.

## Readings (2026-10-25, 12:33–13:19; the lock held exclusively; every run started at a 1-minute load of 2.65 or below, with 7.5 GB or more reclaimable)

Log: `~/goinfer-logs/ks-e-profiles-2026-10-25.log`.

### KS-E1: int4 decode GEMV
- **The int4 GEMV:** `TestW4A8_parity_and_bandwidth`, 8960 × 1536: parity cosine 1.000000, **144.1 µs a dispatch, 7.74 MB, 53.7 GB/s.**
- **The registered baseline did not measure what it was meant to.** `TestGEMV_microbench`'s int8 figure (4096², "GPU GEMV runner"
  1.692 ms, 9.9 GB/s) times a standalone kernel, not the production decode GEMV, so the registered ratio (int4 ÷ int8 = 5.4) is not
  read.
- **The production int8 GEMV's rate, from the E5 profile:** `gemvPipeline` plus `gemvBiasPipeline` take 13.6 ms of a token at depth
  2,048 and 10.7 ms at 8,192. The 1.5B's int8 decode weights are about 1.55 GB, so that is about **113–145 GB/s**.
- **Comparison:** int8 at those rates moves the same 8960 × 1536 matrix (13.8 MB) in about 95–120 µs. The int4 kernel takes 144 µs
  for half the bytes. **int4 is slower per matrix than int8 here: compute-bound, int4 ÷ int8 about 0.37–0.48 in GB/s.**
- **Reading:** below the registered 0.80 line, on the corrected baseline. This is an estimate from the ablation, not a same-kernel
  A/B.

### KS-E4 and KS-E6: the batched prefill pass, Qwen3-VL-2B, int4 (`TestPrefillInt4_profile`)

| rows | wall unprofiled | projections | attention | norms and rotary | K/V write | row-by-row walk, for scale |
|---|---|---|---|---|---|---|
| 256 | 1.95 s | 1.93 s (84%) | 0.12 s (5%) | 0.14 s (6%) | 0.10 s (4%) | 4.30 s |
| 1,024 | 8.21 s | 8.03 s (80%) | 1.64 s (16%) | 0.27 s (3%) | 0.15 s (1%) | 23.85 s |
| 4,096 | 49.98 s | 28.16 s (54%) | 22.42 s (43%) | 1.05 s (2%) | 0.71 s (1%) | 167.19 s |

- **KS-E4:** projections are 80% of the pass at 1,024 rows, over the registered 40%. It is a candidate. At 4,096 rows they still
  take 28 s.
- **KS-E6:** attention is 43% of the pass at 4,096 rows, over the registered 30%. It is a candidate, and its share grows with the
  prompt: 5% at 256, 16% at 1,024, 43% at 4,096.

### KS-E5: decode attention at depth, the 1.5B, int8int8 (`TestDecode_dispatchProfile`; 366 dispatches a token, 12 heads, 2 KV heads)

| depth | full token | `attnKeysPipeline` (ablation) | share | per position of context | `gemvPipeline` |
|---|---|---|---|---|---|
| 2,048 | 47.6 ms | 29.7 ms | 62.3% | 14.5 µs | 11.9 ms |
| 8,192 | 135.3 ms | 117.6 ms | 86.9% | 14.4 µs | 10.1 ms |
| 16,384 | 270.0 ms | 251.7 ms | 93.2% | 15.4 µs | 16.3 ms |

- **KS-E5:** attention is 93% of a 16k token, over the registered 30%, at about 15 µs a position.
- **On scale:** the survey's 1.14 µs came from another session and box. Even so, about 13× suggests the kernel's one workgroup
  per head (12 here) leaves most of the M1 Pro's GPU idle at depth.
- **Ablation noise:** the 16k table's parts sum to 118%, and some classes read negative at 8k.

### What the numbers suggest building (for Francis; nothing is built here)

On the Mac's WebGPU path, in order of what each would save:
1. **KS-E5 (split-KV decode attention, fidelity-gated; (a) the bit-identical V-load unroll first).** Attention is 62–93% of a
   decode token from 2k to 16k positions. More workgroups per head is where the depth cost is.
2. **KS-E6 (prefill attention, several rows a workgroup).** It is 43% of a 4,096-row pass and rising with length.
3. **KS-E4 (the int4 prefill GEMM).** It is 54–84% of the pass.
4. **KS-E1 (`nib4` with `dot4I8Packed`).** The int4 GEMV is compute-bound, slower per matrix than int8 here. A same-kernel A/B is
   needed before building.

**Caveat:** on the Mac, production GPU inference is Metal; WebGPU is the portable path (Linux without NVIDIA, Windows). These
shares should be confirmed on nobara's WebGPU before ranking work on them. The attention kernel's shape, one workgroup per head with
a serial key loop, is the same on every adapter.
