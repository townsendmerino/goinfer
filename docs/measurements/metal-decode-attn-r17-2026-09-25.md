# Metal decode attention at depth — R17 (2026-09-25)

R17 (`docs/tasks/red-october.md`) is pre-registered: ship ≥ 2.5× / park 1.5–2.5× / kill < 1.5× on the in-sequence
attention work at 3900 keys on the 1.5B (the no-op method of `metal-decode-decomp-2026-09-25.md`), with the
teacher-forced fidelity gate, ≤ 3% at 128 keys, sustained timing and a confirmation run as preconditions.

## Step 0 — a larger `attention_fa` split count: 1.20× at best, KILL band (2026-09-25)

**What was tested.** Whether `attention_fa` is latency-bound on too few threadgroups in flight: at 3900 keys it runs
28 threadgroups on the 1.5B (`attnFASplitFor`: S = 14 for nKV = 2, from a 2×-core-count rule). A test-only override
(`resident.attnFASplitOverride`, zero in production, read inside `attnFASplitFor` so the dispatch grid and the
per-step uniform agree) swept S; the partial buffer was re-allocated for S ≤ 64. Kernels take nSplit at runtime.

**How.** `TestMetalDecodeDecomp` with `GOINFER_METAL_DDECOMP_SPLITS=0,20,28,32,48,64` (0 = production's rule):
per split count, the production decode token with and without the attention pipelines no-op'd, arms interleaved rep
by rep, 5 reps × 20 tokens. M1 Pro, 1.5B q4_k_m, goinfer `143314c1` + the uncommitted override and sweep;
16:59:15–17:00:43 local, idle at start (load1 1.98). Raw: [`step0-split-sweep.log`](metal-decode-attn-r17-2026-09-25/step0-split-sweep.log).

| S | attention @ 2048 keys | @ 3900 keys | vs S=14 @ 3900 | full token @ 3900 |
|---:|---:|---:|---:|---:|
| 14 (production) | 4.984 ms | 8.649 ms | 1.00× | 20.982 ms |
| 20 | 5.462 | 9.221 | 0.94× | 21.560 |
| 28 | 4.918 | 8.076 | 1.07× | 20.419 |
| **32** | **4.598** | **7.235** | **1.20×** | **19.588** |
| 48 | 4.855 | 7.277 | 1.19× | 19.622 |
| 64 | 5.554 | 8.006 | 1.08× | 20.330 |

Per-rep spreads are small (e.g. S=32 at 3900: 7.2–7.6 ms).

**Outcome: KILL band as an R17 candidate** (best 1.20× < 1.5×; not taken to a confirmation run). What it says:

- **The "too few threadgroups in flight" reading is largely refuted.** More than doubling the threadgroups buys at
  most 1.2×, and the response is not monotonic (S=20 is slower than 14; S=64 gives back most of S=32's gain). The
  per-key cost is inside each simdgroup's dependent chain — consistent with the other two reasons the kernel read
  ranked: the online-softmax work is repeated for every key (a `simd_sum`, two `exp`s and a rescale per key per head,
  on every lane) rather than amortized over a block, and few loads are in flight per iteration. Both are what the
  prototype's block-of-32, lane-per-key-softmax shape changes; neither is touched by more splits.
- **A small real gain exists outside R17's band:** S=32 is 1.20× on attention and ~7% on the whole 1.5B token at
  3900. It changes the split-merge order, so taking it on its own would need the teacher-forced fidelity gate; it is
  recorded, not proposed, since R17's decision is about the kernel shape.
