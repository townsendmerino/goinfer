# MC3 S3 — step polish: a batched step's per-row work in one dispatch; 4 clients at 1.26× S2, p99 turn 0.78×, bit-identical (2026-09-27)

S3 of MC3 in [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md), the owner's second item of
2026-09-27. After MC3 and S2 ([`concurrency-mc3-2026-09-26.md`](concurrency-mc3-2026-09-26.md),
[`concurrency-mc3-s2-2026-09-27.md`](concurrency-mc3-s2-2026-09-27.md)), a batched step's time grew with the batch in
a way its matmuls did not explain.

**Result: all five pre-registered gates pass.**
- 4 greedy clients read **153.1 tok/s against S2's 117.8–121.6** (1.261×); p99 per turn falls from 4.66–5.09 s to
  3.66–3.69 s (0.784×).
- 2 clients: 1.08–1.11×. A lone request: p50 0.999×, p99 1.015×.
- Every reply identical.
- Under 4-client load a turn now takes 1.92× a lone request's, where MC3 alone took 2.47×.

## Where the step's time went (exploratory, before the build)

`TestMC3StepBreakdown` (`metal/mc3_step_test.go`) times the batched step in sequence. It then replaces one category's
kernels at a time with an empty kernel, so the dispatches still launch but do no work, and finally every kernel.
Median of 15 steps per arm, arms interleaved; 1.5B, M1 Pro.

| depth, B | step | batched matmuls | attention | per-row kernels | dispatch overhead |
|---|---:|---:|---:|---:|---:|
| 128, 4 | 22.59 ms | 14.42 | 2.56 | **4.21** | 1.59 |
| 128, 8 | 30.06 | 14.52 | 4.99 | **8.09** | 2.96 |
| 512, 4 | 26.58 | 14.23 | **6.47** | 4.41 | 1.48 |
| 512, 8 | 37.75 | 14.23 | **12.68** | 8.04 | 2.92 |

- **The matmuls were already amortised.** They cost 14.4 ms at any B; this is S0's kernels doing their job.
- **What grew was the per-row work, run one sequence after another.** The small per-row kernels cost ~1 ms per
  sequence: norm + quantisation ×2, RoPE, KV store, ctx quantisation, SwiGLU. Each is a single-threadgroup dispatch,
  28 layers deep. Attention ran each sequence's heads as its own dispatch.

Logs: [`s3-breakdown.log`](concurrency-mc3-s3-2026-09-27/s3-breakdown.log),
[`s3a-breakdown.log`](concurrency-mc3-s3-2026-09-27/s3a-breakdown.log),
[`s3b-breakdown.log`](concurrency-mc3-s3-2026-09-27/s3b-breakdown.log).

## The build (`731f4f4e`)

- **S3a: multi-row forms of rmsnorm_quant, quant_vec, swiglu_quant and rope2** (`metal/batch_rows.go`), one
  threadgroup or thread block per row, all rows in one dispatch.
  - Their bodies are production's byte for byte. Each variant is derived at init from `allKernels`' own source, by
    renaming the pointer and per-row parameters in the signature and adding a prologue that points them at the row.
  - Nothing is retyped, because these kernels round differently when their code shape changes (rmsnorm_quant's own
    notes). A production signature change panics at init instead of silently diverging.
  - They compile into the resident's main library.
- **S3b: multi-row kv_store and attention** (the per-head kernel). A row map selects the rows below
  `attnFADepthFloor`; attention_fa rows keep their own dispatches, so one step mixes the two.
  - One dispatch must reach every row's KV slot, and aikit binds at most 16 buffers per dispatch. So with several
    slots a layer's KV slots are now **one allocation**, slot views at fixed offsets, each with its own padded region.
  - aikit's host views (`Floats`, `U16s`) ignore a view's offset, so the two host KV writers (`UploadKV`, the
    attn-confirm test hook) add the bound slot's offset (`kvHostOff`).

GPU step time after each part:

| depth, B | before | S3a | **S3a + S3b** |
|---|---:|---:|---:|
| 128, 2 | 18.85 ms | 17.73 | **17.20** |
| 128, 4 | 22.59 | 19.26 | **17.69** |
| 128, 8 | 30.06 | 22.47 | **18.72** |
| 512, 4 | 26.58 | 23.14 | **20.24** |
| 512, 8 | 37.75 | 30.20 | **23.36** |

After S3, per-row work is ~1.3–1.4 ms and dispatch overhead ~0.56 ms at any B. At depth 512, attention (4.0 ms at
B = 4) is what still grows.

## Correctness, run before any timing

- `TestMC3Step_bitIdentical`, `TestMC3Step_bitIdenticalDeep` and `TestMC3Step_drawsMatchForwardSample`: 0 differ.
  The deep test runs attention_fa rows and per-head rows in one step, one row crossing the floor mid-run.
- Both concurrent-vs-alone tests through `Model.Generate` pass, greedy and sampled.
- MC1's slot tests pass on the new allocation. The tagged Metal suite ran 182 pass, 0 fail
  ([`metal-suite-tagged-s3.log`](concurrency-mc3-s3-2026-09-27/metal-suite-tagged-s3.log)).

## The W7 grading

Pre-registered in `767cffac`, committed before the S3 code:
- *old* = `serve-metal` at `4954f978` (S2), *new* = at `731f4f4e`.
- The MC3 W7 workload, greedy, `--fixed-nonce`, serve defaults; qwen2.5-coder-1.5b int4 from `~/models`.
- 4, 1 and 2 clients, 3 interleaved pairs each; idle-gated per cell; 2026-09-27 04:28–04:33 PDT.
- All 18 servers logged the batched-decode banner.

Raw: [`w7-s3.json`](concurrency-mc3-s3-2026-09-27/w7-s3.json), [`w7-s3-run.log`](concurrency-mc3-s3-2026-09-27/w7-s3-run.log),
[`w7-s3-servers.log`](concurrency-mc3-s3-2026-09-27/w7-s3-servers.log), [`run-w7.sh`](concurrency-mc3-s3-2026-09-27/run-w7.sh);
gates → [`gates-output.txt`](concurrency-mc3-s3-2026-09-27/gates-output.txt).

| gate (pre-registered) | result |
|---|---|
| 1. identity, every turn, every cell | **pass** (18 cells) |
| 2. reuse equal | **pass** |
| 3. 4-client aggregate new ÷ old, median ≥ 1.03× | **pass: 1.261×** (1.259, 1.261, 1.300) |
| 4. 4-client p99 turn new ÷ old ≤ 1.0 | **pass: 0.784×** (0.724, 0.784, 0.791) |
| 5. lone request p50 / p99 ≤ 1.05× | **pass: 0.999× / 1.015×** |
| reported: 2 clients | aggregate 1.079 / 1.086 / 1.107×; p99 3.34–3.40 → 3.07–3.08 s |

The lone request's aggregate reads 74.0–75.0 against 74.6–75.5 tok/s, about 1% lower. Its p50 turn ratios are
0.998 / 0.999 / 1.021, so this is within the noise the guard allows. A lone request's tokens run production's own
per-sequence path, which S3 does not touch. The only changes on that path are the KV slots' allocation and the extra
kernels compiled into its library.
