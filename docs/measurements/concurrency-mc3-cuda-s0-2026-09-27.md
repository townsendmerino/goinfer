# MC3 on CUDA, S0 — a batched step through the exact kernels is bit-identical to decode and 1.75–2.04× cheaper at B = 4 (2026-09-27)

An exploratory S0 for porting MC3 (batched multi-request decode) to CUDA, per
[`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md) ("CUDA, only on its own measurement"). Its
prerequisite, MC1 on CUDA (multi-slot resident KV), shipped the same day
([`concurrency-mc1-cuda-2026-09-27.md`](concurrency-mc1-cuda-2026-09-27.md)). This is measurement only; no production
code changed.

**Result: both S0 questions come out yes.**
1. **Identity.** A batched M-row pass through the exact kernels (`gemv_w4a8_rn` below 16 rows, `attn_batched`) returns
   every row's logits **bit-identical** to a sequential single-token `Forward` at the same position: 0 differing
   logits of 0.3–1.2M, on the 1.5B and the 7B, at M = 2, 4 and 8. A CUDA step can meet MC3's identity gate on kernels
   that already exist.
2. **Amortisation.** At depth 1024 the batched pass with every row's exact logits is **1.75×** (1.5B) and **2.04×**
   (7B) cheaper than 4 single-token decodes, and 2.22× / 2.41× at 8. Metal's MC3 S1 read 1.73–1.84× at B = 4 and
   became 1.59× end to end.

## Setup

- Machine: `nobara`, RTX 2070 SUPER 8 GB, NVIDIA driver 595.91.07.
- Models: qwen2.5-coder-1.5b and qwen2.5-7b instruct q4_k_m from `~/models`, int4, with the tree at `5f911d0b` plus
  the probe file.
- Both probes warm the resident KV to depth 1024, then take the best of 5 per cell.
- Idle-gated (load1 ≤ 1.0); 2026-09-27 18:36–18:38 PDT.
- Raw: [`concurrency-mc3-cuda-s0-2026-09-27/`](concurrency-mc3-cuda-s0-2026-09-27/).

## Results

`TestMC3CUDAS0_batchedRowsVsDecode` (`cuda/mc3_cuda_s0_test.go`). The unit is M sequential `Forward` calls; the table
shows how many times cheaper each batched variant is.

| model | M | M × decode | `PrefillLastN` (every row's logits, head per row) | `PrefillLastNArgmax` (one batched head, argmax only) | logits differing from `Forward` |
|---|---:|---:|---:|---:|---:|
| 1.5B | 2 | 9.71 ms | 7.48 ms = 1.30× | 6.69 ms = 1.45× | 0 of 303,872 |
| 1.5B | 4 | 18.58 ms | 10.61 ms = **1.75×** | 8.44 ms = 2.20× | 0 of 607,744 |
| 1.5B | 8 | 37.24 ms | 16.76 ms = 2.22× | 11.61 ms = 3.21× | 0 of 1,215,488 |
| 7B | 2 | 27.53 ms | 18.73 ms = 1.47× | 17.52 ms = 1.57× | 0 of 304,128 |
| 7B | 4 | 55.13 ms | 27.00 ms = **2.04×** | 22.81 ms = 2.42× | 0 of 608,256 |
| 7B | 8 | 110.27 ms | 45.78 ms = 2.41× | 37.31 ms = 2.96× | 0 of 1,216,512 |

The existing `TestSpecVerifyCeiling` (`PrefillLast`, whose head runs on the last row only) reads 2.18× (1.5B) and 2.49×
(7B) at M = 4. That flatters a step, which needs logits for every sequence.

## What it does not show, and what the build needs

- **One sequence's rows, not B sequences.** The rows here share one KV cache at consecutive positions. A step needs
  per-sequence positions (RoPE) and each sequence's own KV slot.
  - The identity-safe way is to run each sequence's rope_kv and attention through decode's own path, per sequence,
    with that sequence's slot bound. That includes the flash-decode lane past 2048 keys, which is not bit-identical to
    the exact path; running decode's own path keeps that choice identical by construction.
  - The GEMVs, norms and MLP stay batched over the B rows.
- **The head.** The exact-logits variant re-reads the head per row: 4.2 ms of the 7B's 27.0 ms at M = 4.
  - A batched head that is proven bit-identical (the int8 batched GEMV is documented as bit-identical by construction;
    `PrefillLastNArgmax` claims only argmax) would move M = 4 toward the argmax column's 2.2–2.4×.
  - That is the first lever to test.
- **Scope, as Metal's MC3 had it at first:** dense families only; temperature-sampled tokens run solo (Metal added
  on-device draws in steps later, S2); no speculation, adapters, vision or C′ expert streaming in a step.
- **The decoder side is already generic.** `ResidentBatchStepper` and MC3's batcher, `EnableResidentConcurrency`,
  serve's `setConcurrency`, and the banner all work unchanged. CUDA implements `BatchStepRange` / `StepBatch`.

## Reading

It is worth building. By Metal's ratio of step to end to end, a CUDA step at 1.75–2.04× suggests roughly **1.5–1.8×**
at 4 clients end to end, with a lone request unchanged. That is before any head lever. The build gets its own
pre-registration: correctness gates first (through the production path, against each conversation served alone), then
a W7 grading shaped like MC3's.
