# gpu: notes moved out of code comments

History, measurements and open work that used to sit in the comments of package `gpu`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## ensurePrefillBatched

Moved from `gpu/prefillrunner.go` (the comment above `ensurePrefillBatched`) on 2026-10-10.

```text
ensurePrefillBatched lazily compiles the two batched-row kernels PrefillLastW8A8
needs beyond its existing ensure-list (M-08's dispatch-count fix, audit-metal
class finding but on WebGPU: the original implementation issued one dispatch PER
ROW for RMSNorm/RoPE/quantize-gather/GEMM-scatter, ~39 dispatches × M per layer,
measured 0.03-0.14x SLOWER than the sequential loop it was meant to replace at
P=64..1024 — see docs/measurements/ and TestResidentPrefillLast_TTFT). Kept
separate from ensureLayer/ensureAttn's M=1 pipelines: the decode path must not
pay for or risk these at all.
```

## runModelToModelW

Moved from `gpu/prefillrunner.go` (the comment above `runModelToModelW`) on 2026-10-10.

```text
runModelToModelW narrows a resident runModel (the general polymorphic
representation DecodeRunner uses — W8A8/W4A8, MoE, MLA, Mamba, DeltaNet,
sliding window, QK-norm, bias, per-layer geometry overrides, …) down to the
plain dense-W8A8 ModelW shape PrefillLastW8A8 accepts, or reports ok=false when
this model uses ANY feature outside that shape. The same scope
DecodeTokenFusedBatched declares ("no MoE/MLA/SSM/bias/QK-norm").

q/k/v bias (Qwen2) IS accepted here (unlike DecodeTokenFusedBatched) — see
AttnWeights.QBias/KBias/VBias and tiledProj's bias parameter — but callers must
additionally check ModelW.hasBias() against the backend before trusting the
result; see residency.go's prefillLast closure for that gate and the full
history below. This function's own scope guard is architecture-only.

HISTORY (2026-09-12, real-checkpoint debugging, qwen2.5-coder-0.5b): enabling
bias originally diverged from sequential Forward() at nKeys>=3 (cosine ~0.99).
Two real, independent causes were found and fixed:

 1. Kernel mismatch: this function unconditionally dispatched c.attnPipeline
    (the plain f32 attention kernel), while decoderunner.go's sequential decode
    picks per-geometry via attnKernel (attention.go) — for this checkpoint's
    geometry (hd=64, kvDim=128, f32 KV) that's the key-split kernel. Two
    different kernels computing the same attention is not guaranteed
    bit-identical. Fix: call the same c.attnKernel(hd, kvDim, false)
    decoderunner.go uses (both now share one implementation).

 2. Epilogue fusion: production's gemvBias computes
    `f32(acc)*aScale*bScale + bias[n]` in ONE expression inside ONE dispatch;
    this function did a plain GEMM into a fresh buffer, scattered it out via a
    byte-exact copy, THEN a separate residualShaderWGSL dispatch added the bias
    — mathematically the same formula, but forced through an f32 round-trip
    between the multiply and the add that a single WGSL expression may
    evaluate with a different (FMA-contracted) rounding. Measured directly
    (TestLocalize_BiasEpilogue, layer 0's real Q weight+bias): 98 of 896
    elements differed by up to 2.4e-7 between the two forms — tiny in f32
    terms, but enough that a subsequent int8 requantize can flip a rounding
    bucket for an element sitting on the boundary, and 24 layers of that
    compounds into the observed divergence. Fix: matmulTiledW8A8BiasKernelWGSL
    (gemm.go) — the SAME tiled GEMM with a fused per-column bias epilogue,
    textually matching gemvBias's expression.

Together these two fixes take real-checkpoint bias-enabled parity to BIT-EXACT
(cosine 1.0, maxAbsDiff LITERALLY 0, not float noise) on Vulkan/RTX 2070 SUPER
at every nKeys measured from 1 to 50.

STILL OPEN, Vulkan-vs-Metal only: on Metal/M1 Pro, the SAME fixes leave a
smaller but real residual divergence past nKeys~15 (cosine ~0.997-0.999,
maxAbs ~0.3-0.65 — not float noise, and NOT monotonic with nKeys). A same
analogy — production fuses the O-proj/down-proj GEMV WITH the residual add
(decoderunner.go's gemvAdd / gemvW8A8ShaderWGSL's addResidual epilogue), while
this function does a plain GEMM then a separate residualShaderWGSL add, same
forced-round-trip shape as the bias case — was the obvious next suspect and is
almost certainly PART of the real mechanism, but a first attempt at a fused
residual-epilogue kernel (mirroring matmulTiledW8A8BiasKernelWGSL) introduced a
NEW regression on Vulkan too (removed rather than shipped broken — see git
history around 2026-09-12 for the attempt, which gathered the residual stream
into a contiguous buffer, ran a read-write accumulate kernel, then scattered it
back — the bug was not found before time ran out on that investigation).
Whoever picks this up next: rebuild that attempt carefully (gather/accumulate/
scatter, matching tiledProj's bias-parameter shape but for O-proj/down-proj),
verify it against TestLocalize_BiasEpilogue-style isolated tests BEFORE wiring
it into the main loop, and re-measure both backends. Also worth checking: this
repo's real ship gate for a fast-prefill path is the §3.2 pooled fidelity gate
against the CPU-f32 reference (docs/completed/task-prefill-gap.md), not bit-exactness
against sequential GPU decode — Metal's current gap might already clear that
bar even before a further fix, which would change the urgency here.

Until Metal is resolved or independently cleared, residency.go's prefillLast
gates bias to backends where it's actually proven (Vulkan only) — a decline
there falls back to the slower-but-correct sequential loop, never serving a
silently-wrong result.

It is a zero-copy view: every *wgpu.Buffer is wrapped, not duplicated, and the
caller must not Close() the resulting ModelW (ModelW.Release would double-free
buffers rd.rm still owns) — the wrapper DeviceBuffers exist only so ModelW's
field types match; PrefillLastW8A8 never calls Close on them either.
hd is the model's head dimension — needed only to tell genuine partial RoPE
(ropeHalf set to something less than hd/2) apart from ropeHalf simply being SET
to the full-rotation value instead of left at its 0 "use hd/2" sentinel; both
PrefillLastW8A8's and DecodeTokenFusedBatched's rope() closures hardcode
half:=hd/2 (full rotation, no partial-RoPE support), so the latter is fine and
only the former must decline.
```

## maxChunkElems

Moved from `gpu/prefillrunner.go` (the comment above `maxChunkElems`) on 2026-10-10.

```text
maxChunkElems: WebGPU's per-dimension workgroup-COUNT limit is 65535
(maxComputeWorkgroupsPerDimension); each workgroup here covers 64 elements, so a
single 1-D dispatch tops out at 65535*64 elements. Hit for real at M=1024 on
qwen2.5-coder-0.5b (inter=4864: M*inter=4,980,736 > 4,194,240) — surfaced as a
WebGPU validation error PrefillLast would otherwise decline into (safe, just
slow), not a silent wrong-output risk, but worth actually fixing.
```

## quantPackedM

Moved from `gpu/prefillrunner.go` (the comment above `quantPackedM`) on 2026-10-10.

```text
quantPackedM quantizes ALL M rows of a packed [M, K] buffer in ONE dispatch,
straight into the [M, kp/4] packed-int8 / [M] scales layout tiledProjB's GEMM
wants — quantizeShaderWGSL (device.go) was ALREADY written for M rows in one
dispatch (QDims.m, workgroup_id.x = row); the old per-row call site (quant1,
dispatched with m=1, M times) never used that capability. No gather needed:
the GEMM's aq/aScale inputs ARE this call's direct output.
```

## tiledProjB

Moved from `gpu/prefillrunner.go` (the comment above `tiledProjB`) on 2026-10-10.

```text
tiledProjB runs a projection over all M rows: quantPackedM (one dispatch, no
gather) → ONE unbounded-M tiled GEMM (weight streamed once, DP4A-accelerated
when available) → the packed [M, N] GEMM output IS the return value, no
scatter into per-row buffers. rowsM lets the LM head reuse this at M=1 (a
one-row "batch" is just the M=1 case of the same dispatch shape).

bias (nil for every projection except Qwen2's q/k/v) selects the bias-epilogue
tiled kernel instead of a separate post-hoc residual-kernel add — see the
original tiledProj's doc comment (git history) for the measured reason this
matters for bit-exactness (TestLocalize_BiasEpilogue).

R-25 (docs/tasks/task-recompute-audit.md): it is quantB then projQB, so the projections that read the same input
(q, k, v of xn; gate, up of xn2) quantize it once and share the result. Quantization is deterministic, so each GEMM
reads the same bytes it did when it quantized its own copy.
```

## PrefillLastW8A8: profBoundary

Moved from `gpu/prefillrunner.go` (the comment above `pt` and `profBoundary` inside `PrefillLastW8A8`) on 2026-10-10.

```text
R10 prefill decomposition (gpu/prefill_prof.go): pt is a no-op (time.Time{}) whenever
c.prefillProf is nil, so every call below costs one nil-check when profiling is off —
mirroring cudaResident's own profTic/profToc exactly, including the accepted trade-off
that category boundaries are syncs, so the category sum runs a bit over pipelined wall
time (docs/measurements/webgpu-prefill-decomp-2026-09-22.md has the numbers). residualB
and swigluB (elementwise, not GEMM/attn/norm-rope/kv-write) fold into normsRope, the
same "glue" role cuda/prefill.go's glueCat plays.
profBoundary closes one category: it FLUSHES the open encoder first (Poll only waits for
submitted work — dispatches still in the encoder are invisible to it, so without this the
split follows the 32-dispatch flush cadence, not the categories; the first run of this
profiler read attention at 544 ms for P=256 and 55 ms for P=512, physically impossible for an
O(P^2) kernel, which is how the bug was found), then polls and books the elapsed time. The
extra flushes happen ONLY when profiling is on; production cadence is untouched.
```

## PrefillLastW8A8: attention step

Moved from `gpu/prefillrunner.go` (the comment above the attention dispatch inside `PrefillLastW8A8`) on 2026-10-10.

```text
All rows' K/V are now in the cache; each row attends to its causal prefix
(including earlier rows of this same prefill block) — the same ordering
DecodeTokenFusedBatched's parity gate already proves correct.

docs/completed/task-gpu-batched-prefill.md Increment 1: ONE dispatch, grid (nH, M), against
attnBatchedKernel's chosen kernel (mirrors attnKernel's own tiled-vs-plain
preference, attention.go) — replaces what used to be M per-row dispatches
into the M=1 kernel (the one dispatch-count cost the earlier fix in this
function's history did not eliminate). q/ctxv are read/written directly at
each row's own offset inside the shader now, so the qRow-extract/cv-scatter
copy pair this loop used to need is gone entirely, not just the M-1 spare
allocations of it.
```
