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

## Readings

(added when the runs land)
