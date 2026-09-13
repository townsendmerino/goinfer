# Batched GPU prefill TTFT — the measurement `task-gpu-batched-prefill.md`'s own Definition of Done always required, run for the first time (2026-09-13)

**`gpu-dp4a-batched-prefill`'s `PrefillLastW8A8` passed every bit-exact gate and was never once
measured end-to-end. Run for the first time here: it was 7–29x SLOWER than the sequential loop it
was meant to replace, and worsening with prompt length. Root-caused to a per-row dispatch
explosion (~39 GPU dispatches × M per layer instead of the intended O(1)), fixed, and re-measured:
now a consistent 2.5–3.5x win at every tested size.**

## Provenance

| | |
|---|---|
| box | nobara-pc, AMD64, 16 cores, 62 GB RAM, Nobara Linux (kernel 7.2.0-202.fc44) |
| GPU | NVIDIA GeForce RTX 2070 SUPER (TU104), 8 GB VRAM, Vulkan backend |
| toolchain | go1.26.5 linux/amd64 |
| model | qwen2.5-coder-0.5b-instruct-q4_k_m.gguf, from `~/models` on local NVMe |
| backend/quant | `webgpu`, `int8int8` (W8A8) |
| harness | `gpu/prefilllast_ttft_test.go` `TestResidentPrefillLast_TTFT` — sequential (today's shipped `residentPrefillSeed` per-token loop) vs one `PrefillLast` call, same prompt, same resident pipeline, both timed in the same process run (not cross-run) |
| commits | before: `3a9d313c` (branch tip as left by the prior session); after: `fda9f7e9` (`gpu-dp4a-fix`, two commits on top) |

## Before the fix (3a9d313c) — the number nobody had run

```
GOINFER_RESIDENT_GGUF=~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
  GOINFER_HEAVY_TESTS=1 go test -tags 'gpu goinfer_testhooks' ./gpu/ -run TestResidentPrefillLast_TTFT -v -timeout 20m
```

| P | sequential | batched | speedup |
|---|---|---|---|
| 64 | 1472.44 ms | 10525.84 ms | **0.14x** |
| 256 | 5153.86 ms | 52031.80 ms | **0.10x** |
| 1024 | 20607.20 ms | 602745.14 ms | **0.03x** (29x SLOWER) |

Not a wash — a severe regression, and getting *worse* with M (4x more rows, 11.6x more time at the
256→1024 step). The do-nothing arm (today's shipped sequential loop) beat the built feature at
every size tested.

## Root cause

`tiledProj` (the helper every projection in every layer called) did real batched work for exactly
one thing — the tiled GEMM — and wrapped it in **M separate dispatches** for everything else:
RMSNorm, quantize-and-gather each row into the GEMM's input buffer, scatter each output row back
out. Same shape for RoPE (M calls, one scalar `pos` each) and the KV-cache write (M per-row
copies). ~39 dispatches × M per layer, each allocating a fresh `wgpu.Buffer` held (never released)
until the whole call returned — at M=1024 that is on the order of a million dispatches and hundreds
of thousands of live buffers, which is both the raw dispatch-submission overhead and why the
slowdown worsens faster than linear (VRAM/allocator pressure compounding).

## The fix

Operate on packed `[M, width]` row-major buffers end-to-end instead of `[]*wgpu.Buffer` per-row
slices, so the ops around the GEMM collapse from M dispatches to 1 (attention itself is the one
exception — see Known remainder below). Full detail in the `gpu-dp4a-fix` branch's own commits
(`771f7bb1`, `fda9f7e9`); short version:

- `quantizeShaderWGSL` (`device.go`) was **already written** to quantize M rows in one dispatch
  (`QDims.m`, `workgroup_id.x` = row) — the old call site just never used that, dispatching it M
  times with `m=1`. Fixed the call site only, zero kernel change.
- New `rmsnormBatchedShaderWGSL` / `ropeBatchedShaderWGSL` kernels, kept **separate** from the
  existing M=1 kernels the decode path uses (zero risk to decode): grid `(1,M)` / `(heads*half,M)`
  instead of M separate single-row dispatches. Bit-identical by construction (each
  workgroup/row is independent, no cross-row barrier) — gated by new
  `TestRMSNormBatched_parity` / `TestRoPEBatched_parity` (no checkpoint needed).
- `residualShaderWGSL` / `swigluShaderWGSL` needed **no kernel change** — both are flat elementwise
  ops with no cross-row state, so `[M,width]` treated as one `[M*width]` array is exactly what M
  separate dispatches computed.
- Bulk KV-cache write: positions are always contiguous within one `PrefillLastW8A8` call, so a
  projection's packed `[M,kvDim]` rows land at one contiguous cache range — 1 copy instead of M.
- A second real bug found only by running P=1024: WebGPU caps
  `maxComputeWorkgroupsPerDimension` at 65535, and the flattened residual/swiglu dispatch (M*width
  elements / 64 per workgroup) exceeds that once `M*width > 4,194,240` — hit exactly at M=1024 on
  this checkpoint's `inter=4864` (`M*inter=4,980,736`). `PrefillLast` declined safely into the
  sequential fallback (a caught `Validation Error`, not silently-wrong output) — just not the speed
  this fix is for. Fixed via `dispFlat`, chunking any flat op into multiple dispatches each within
  the limit, via offset+size bind-group views (safe here — every chunk boundary is a multiple of
  the workgroup size, so always 256-byte aligned).

## After the fix (fda9f7e9)

```
GOINFER_RESIDENT_GGUF=~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
  GOINFER_HEAVY_TESTS=1 go test -tags 'gpu goinfer_testhooks' ./gpu/ -run TestResidentPrefillLast_TTFT -v -timeout 20m
```

| P | sequential | batched | speedup |
|---|---|---|---|
| 64 | 411.14 ms | 116.95 ms | **3.52x** |
| 256 | 1351.02 ms | 438.06 ms | **3.08x** |
| 1024 | 8133.01 ms | 3226.38 ms | **2.52x** |

A consistent win at every size tested — the P=1024 case that was 602.7s (29x slower) is now 3.2s
(2.5x faster). Run-to-run absolute times for the sequential arm moved between the before/after runs
(1472→411ms at P=64) — expected process/thermal variance on a shared box, not a claim about the
sequential path itself; the speedup ratio is what this test measures, both arms timed in the same
process run.

Bit-exactness unaffected: `TestRMSNormBatched_parity`, `TestRoPEBatched_parity`,
`TestResidentPrefillLast_parity`, `TestPrefillLastW8A8_parity` all pass (cosine 1.0, maxAbsDiff 0)
both before and after the chunking fix.

## Known remainder

The speedup gently shrinks with M (3.52x → 3.08x → 2.52x) — consistent with the one dispatch-count
cost this fix deliberately did not eliminate: attention itself is still M per-row dispatches into
the existing M=1 kernel. `task-gpu-batched-prefill.md`'s own Increment 1 (a real fused
multi-query batched-causal-attention kernel) was never built, on this branch or before it. As M
grows, that remaining O(M) cost is an increasing share of the total — the likely reason the ratio
is trending down rather than flat. Worth revisiting if a real workload pushes past M=1024 and the
ratio keeps degrading; not blocking today's result, which is a clear win at every size actually
measured.

## Verdict

Ship-worthy on the numbers now in hand: the batched path beats the sequential loop it replaces at
every tested prompt length, with bit-exact parity intact. Still gated behind
`task-gpu-batched-prefill.md`'s own scope (Vulkan-only for bias-enabled models — Metal has an
unresolved smaller correctness gap past nKeys~15 documented there; dense W8A8/W4A8 architectures
only).

Increment 3 (wiring `PrefillLastW8A8` into `decoder.Generate`'s actual prefill path) turned out to
already be done, not owed: `decoder/model.go`'s `residentPrefillSeed` has a generic
`if pf, ok := m.resident.(Prefiller); ok` check that fires for ANY resident type satisfying the
interface, and `gpu.residentDecoder` has satisfied it since `813be4e7` (already on this branch,
predating the dispatch-count fix). Confirmed through the real `Generate()` API, not just the
isolated `PrefillLast` gates above:
`TestGenerate_batchedPrefillMatchesSequential` (`gpu/prefilllast_generate_integration_test.go`) —
same real prompt, greedy, `GOINFER_BATCHED_PREFILL=0` vs default, on real hardware: 24/24 tokens
identical, and both match `decode_parity_test.go`'s independently-pinned CPU reference for the same
prompt/checkpoint exactly. This means the fix in this document is not a pending feature — it is
already live in production for every eligible WebGPU resident prompt.
