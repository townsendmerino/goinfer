# gpu: notes moved out of code comments

History, measurements and open work that used to sit in the comments of package `gpu`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## ensurePrefillBatched

Moved from `gpu/prefillrunner.go` (the comment above `ensurePrefillBatched`) on 2026-10-09.

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

Moved from `gpu/prefillrunner.go` (the comment above `runModelToModelW`) on 2026-10-09.

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

Moved from `gpu/prefillrunner.go` (the comment above `maxChunkElems`) on 2026-10-09.

```text
maxChunkElems: WebGPU's per-dimension workgroup-COUNT limit is 65535
(maxComputeWorkgroupsPerDimension); each workgroup here covers 64 elements, so a
single 1-D dispatch tops out at 65535*64 elements. Hit for real at M=1024 on
qwen2.5-coder-0.5b (inter=4864: M*inter=4,980,736 > 4,194,240) — surfaced as a
WebGPU validation error PrefillLast would otherwise decline into (safe, just
slow), not a silent wrong-output risk, but worth actually fixing.
```

## quantPackedM

Moved from `gpu/prefillrunner.go` (the comment above `quantPackedM`) on 2026-10-09.

```text
quantPackedM quantizes ALL M rows of a packed [M, K] buffer in ONE dispatch,
straight into the [M, kp/4] packed-int8 / [M] scales layout tiledProjB's GEMM
wants — quantizeShaderWGSL (device.go) was ALREADY written for M rows in one
dispatch (QDims.m, workgroup_id.x = row); the old per-row call site (quant1,
dispatched with m=1, M times) never used that capability. No gather needed:
the GEMM's aq/aScale inputs ARE this call's direct output.
```

## tiledProjB

Moved from `gpu/prefillrunner.go` (the comment above `tiledProjB`) on 2026-10-09.

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

Moved from `gpu/prefillrunner.go` (the comment above `pt` and `profBoundary` inside `PrefillLastW8A8`) on 2026-10-09.

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

Moved from `gpu/prefillrunner.go` (the comment above the attention dispatch inside `PrefillLastW8A8`) on 2026-10-09.

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

## matmulShaderWGSL

Moved from `gpu/gpu.go` (the comment above `matmulShaderWGSL`) on 2026-10-09.

```text
matmulShaderWGSL computes dst[m,n] = Σ_k a[m,k]·b[n,k], i.e. dst =
a·bᵀ — the encoder's matmulBT contract: a is [M,K] row-major, b is
[N,K] row-major (the PyTorch [out,in] weight layout, so no transpose
is needed), dst is [M,N] row-major.

One invocation per output element, 16×16 workgroup. This is the
NAIVE kernel — no shared-memory tiling, every invocation streams a
full a-row and b-row from global memory. It is correct and proves
the whole upload/dispatch/readback pipeline; a tiled kernel that
stages K-strips into workgroup memory is the throughput follow-up.
```

## Context.releases

Moved from `gpu/gpu.go` (the comment above `Context.releases`) on 2026-10-09.

```text
releases holds one closure per lazily-created device object (shader module + compute
pipeline), registered AT CREATION by mkPipeline/track and drained LIFO by Close (audit C-26).

It replaces the hand-maintained per-field release list Close used to carry, which had drifted
to 14 of ~40 pipelines — every ensure* added since simply leaked, and ensureVision's shader
modules were dropped on the floor entirely (never stored, so unreleasable at any later point).
A hand list cannot stay correct: it is edited in a different file from the code that allocates.
Registering at the allocation site makes the default behaviour correct for pipelines that do
not exist yet.
```

## Context.closed

Moved from `gpu/gpu.go` (the comment above `Context.closed`) on 2026-10-09.

```text
closed makes Close IDEMPOTENT. `defer m.Close()` alongside an explicit m.Close() is the
ordinary Go shape, and decoder.Model.Close calls m.be.Close() unconditionally — so a second
Close used to double-release the wgpu handles, a use-after-free inside the native layer. The
cpu and cuda backends were already idempotent (cuda guards on r.reqCh == nil); only WebGPU
crashed, and only on a machine with a real GPU.
```

## Context.hasDP4A

Moved from `gpu/gpu.go` (the comment above `Context.hasDP4A`) on 2026-10-09.

```text
hasDP4A records whether this adapter's WGSL compiler accepts dot4I8Packed
(probed once in New(), never re-checked). gfx-rs/wgpu merged the builtin in
April 2025; whether it's actually reachable here depends on the wgpu-native
build this binding vendors, not on anything goinfer controls, so it must be
probed live rather than assumed from the backend/OS. ensureTiled uses it to
pick the DP4A kernel (native hardware dot-product instruction on backends
that lower it, e.g. Vulkan's VK_KHR_shader_integer_dot_product on the DP4A-
capable TU10x+) over the scalar-unpack fallback every backend accepts.
```

## Context.attnBatchedShader

Moved from `gpu/gpu.go` (the comment above `Context.attnBatchedShader`) on 2026-10-09.

```text
docs/completed/task-gpu-batched-prefill.md Increment 1: batched causal attention (PrefillLastW8A8
only, lazy via ensurePrefillBatched, prefillrunner.go) — grid (nH, M), one dispatch
for ALL M query rows against the shared resident K/V cache, each row's causal bound
computed in-kernel as basePos+row+1. Two variants matching attnKernel's own
selection (attention.go): attnKeysBatched mirrors attnKeysShaderWGSL's tiled
key-split decomposition (used whenever attnKeysEligible — most real dense
architectures), attnBatched mirrors the plain per-key attnShaderWGSL (the
fallback attnKernel itself falls back to). Neither is bit-identical to the
single-query kernel it replaces in general — same house rule as
attnKeysShaderWGSL vs attnShaderWGSL, a reduction-order difference — see
TestAttnKeysBatched_parity / TestAttnBatched_parity's cosine/maxAbs gates rather
than a bit-exact one.
```

## liveContexts

Moved from `gpu/gpu.go` (the comment above `liveContexts`) on 2026-10-09.

```text
liveContexts counts Contexts that have been created and not yet Closed. A WebGPU device is
a scarce driver resource — measured on this box's NVIDIA/Vulkan stack, exactly 63 can be
LIVE at once, while create/destroy churn is free (200 cycles with no trouble). So a leaked
Context is not a slow drain, it is a hard cliff: past 63 every later New() fails with
"failed to request device", and in a test binary that silently converts gates into skips.
Exposed to tests through liveContexts.Load(); see TestDeviceExhaustion_repro.
```

## Context.mkPipeline

Moved from `gpu/gpu.go` (the comment above `Context.mkPipeline`) on 2026-10-09.

```text
mkPipeline compiles one WGSL shader into a compute pipeline, registers BOTH objects for release,
and returns them plus the auto bind-group layout. It is the single tracked constructor the
ensure* builders share; it was four byte-identical `mk` closures (attention.go, decodefuse.go,
layer.go, vision.go), one of which — vision's — discarded its *wgpu.ShaderModule so it could
never be released at all (audit C-26a).

On pipeline-creation failure the shader is released immediately and NOTHING is registered, so a
failed ensure* leaves the Context exactly as it found it.
```

## finalizerSerial

Moved from `gpu/gpu.go` (the comment above `finalizerSerial`) on 2026-10-09.

```text
finalizerSerial runs f on the runtime's finalizer goroutine and waits for it.

Why: the wgpu bindings give every wrapper (Buffer, CommandEncoder, ...) its own reference on the device and release it from a finalizer. A buffer a caller leaks after its Context was
closed therefore holds the LAST reference on that device, and the garbage collector drops it on the finalizer goroutine at an arbitrary moment. If that lands while another Context.Close
is releasing its own device on a test goroutine, two devices are destroyed at once inside wgpu-native and both threads park in wgpuDeviceRelease for good (the first heavy-tier gate's
webgpu-parity hang, 1 run in 8, a different test each time; gpu.TestContextClose_finalizerRace reproduces it). Every finalizer runs on one goroutine, so running Close's releases there
serializes them with all of those. It does not serialize an explicit Release from another goroutine (a caller closing a matrix after its Context), which stays the caller's ordering.

If the finalizer goroutine does not take the job within about a second (it is blocked), f runs in place, which is what Close did before.
```

## Stage 3 attention primitives: long-context RoPE

Moved from `gpu/attention.go` (the comment at the top of the file) on 2026-10-09.

```text
Stage 3 attention primitives — RoPE and single-query attention on the GPU, so a
decode token's full forward records into one command buffer (no CPU interleave).
Both match the CPU (decoder.applyRoPE / attendQuery) to f32 tolerance (the CPU
uses f64 accumulation; the GPU f32 — cosine ~1.0, not bit-exact).

N-85 (docs/audit-2026-09-10.md, measured 2026-09-16): every `theta := f32(pos) * invFreq[d]`
below (this kernel and its siblings further down this file) feeds WGSL's `sin`/`cos` with an
angle that grows with position — at a long context's far end that argument is tens of
thousands of radians, and how a GPU's `sin`/`cos` range-reduces an argument that large before
evaluating it is implementation-defined (vendor/driver-specific), unlike the CPU reference's
f64 accumulation. Was flagged UNMEASURED; TestRoPE_parityAtLongContextCeiling (this package's
own test file) now measures it directly at pos=65535 (decoder/fitplan.go's int8-KV
fit-by-default context ceiling, a position a real served request can actually reach) on real
hardware: cosine 0.99999978, maxAbs 2.9e-3 — about 1000x worse than TestRoPE_parity's pos=37
baseline (maxAbs 3.1e-06), a real and measured effect, not just a theoretical one. Whether that
magnitude matters for real model quality is a product decision this file does not make; the
test records the number rather than asserting a pass/fail bar this session has no basis to
pick. `metal/kernels.go`'s identical `float(pos)*invf[dd]` pattern (the "09-02" Metal note
this cross-references) remains unmeasured — this file's measurement does not carry over to
Metal, since GPU vendors' sin/cos range reduction is independently implementation-defined.
```

## attnWG

Moved from `gpu/attention.go` (the comment above `attnWG`) on 2026-10-09.

```text
attnWG is the single-query attention kernels' NARROW workgroup width, and attnWGWide the wide
one. The kernels put one lane on each head dim, so the workgroup width IS the largest head_dim
they can dot: above it the tail dims go un-dotted and the o-projection consumes half-zero
context — plausible-looking WRONG output, no error (audit M-12).

128 covered every family until the Gated-DeltaNet hybrids, whose RELEASED checkpoints all use
head_dim 256 (Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3-Next-80B — verified from their configs, not
assumed). Their tiny fixtures use 32, so the limit was invisible until a real-width fixture met
it, and Gemma 4's global layers at head_dim 512 would have met it again.

The wide kernel (attnWideTemplateWGSL, below) gives each lane a STRIDE of dims rather than one
dim, so head_dim stopped being a reason to decline at all. The narrow kernels are untouched and
still serve head_dim <= attnWG: striding costs a per-lane array and two extra loops, and every
ordinary model would pay that for nothing.

attnWG MUST stay equal to the @workgroup_size(128) and `red: array<f32, 128>` in the three
narrow single-query kernels below.
```

## attnShaderWGSL

Moved from `gpu/attention.go` (the comment above `attnShaderWGSL`) on 2026-10-09.

```text
Single-query attention (decode): one workgroup per query head, an online
(FlashAttention-style) softmax over keys [start, nKeys) so it is numerically
stable and needs no scratch for the full score row. GQA maps kvh = qh/group.
Parallel over the head dimension: workgroup_size = WG (one lane per dim, hd ≤
WG), so each key's score is a workgroup tree-reduce and the value accumulate is
one lane per dim — replacing the original single-thread-per-head kernel (the §5
finding's largest remaining glue kernel, ~5.8 ms). acc/m/l for the online
softmax: acc is per-dim (per lane); m and l are replicated identically across
lanes (every lane sees the same score x), so no extra reduction is needed.
```

## attnKeysShaderWGSL

Moved from `gpu/attention.go` (the comment above `attnKeysShaderWGSL`) on 2026-10-09.

```text
attnKeys — decode attention that splits the workgroup over KEYS, not over the head
dimension. Same math as attnShaderWGSL, different decomposition, and the decomposition
is the whole point.

attnShaderWGSL puts one lane on each of the hd dimensions, so the q·k dot for EVERY key
is a cross-lane reduction: red[d]=prod, barrier, 7 barrier'd tree levels, trailing
barrier = 9 workgroupBarrier() PER KEY. At nKeys=513 that is 4,617 barriers per layer,
and measured 13.69 ms/token at pos 512 on the 1.5B — 28 layers reading 29.4 MB of KV,
which at 448 GB/s is a 0.066 ms job. ~0.5% of streaming roofline, against the GEMV path's
~83%: a latency wall, not a bandwidth one. (TestDecode_dispatchProfile, G35.)

Here each lane owns a disjoint set of KEYS and computes its own dots with no cross-lane
traffic; only the softmax max and denominator reduce, ONCE per tile rather than once per
key. Barriers per layer fall ~250×. This is the shape cuda/attn_block.cu already uses —
WebGPU's kernel was a generation behind it, not blocked by WGSL.

TILED, because WGSL has no dynamic workgroup storage. CUDA sizes sc[nWin] per launch via
extern __shared__; a WGSL var<workgroup> is fixed at compile time, so scores are processed
in TILE-key tiles with the online-softmax state (m, l, acc) carried across them. Storage is
2048*4 + 128*4 = 8.5 KB, inside WebGPU's guaranteed 16 KB — no limit raise, no portability
cost, and any context length works.

vec4 K/q loads are load-bearing, not a micro-optimization. Splitting over keys makes the K
read stride kvDim across the warp; ncu measured that pattern using only ~22% of each 32-byte
L1TEX sector on the CUDA twin, which is why attn_block.cu reads float4. Same fix here, and
it is why the eligibility guard requires hd%4==0 and kvDim%4==0.

NOT bit-identical to attnShaderWGSL: the denominator sums in a different order and the tiled
rescale reassociates. Attention was never bit-exact anyway — it runs f32 against the CPU
oracle's f64 (see the note at the top of this file), so the standing gate is TestAttention_parity's
cosine/maxAbs against that f64 reference, plus argmax through TestWebGPU_forwardParity.
```

## attnKernel

Moved from `gpu/attention.go` (the comment above `attnKernel`) on 2026-10-09.

```text
attnKernel picks the non-int8-KV attention pipeline+layout for a given geometry —
factored out of decoderunner.go's per-token dispatch (the newDecodeRunner build
loop) so every caller that dispatches attention against a plain-f32 or f16 KV
cache picks the SAME kernel for the SAME geometry. This one call site existing
twice (once inline in decoderunner.go, once copy-pasted into prefillrunner.go)
is exactly how PrefillLastW8A8 and the sequential resident DecodeRunner ended up
dispatching DIFFERENT kernels for the identical hd=64/kvDim=128/f32-KV geometry
(qwen2.5-coder-0.5b is attnKeysEligible, so decode used the key-split kernel
while the batched path always used the plain one) — found 2026-09-12 via a
real-checkpoint parity test that diverged (cosine ~0.99 at nKeys=3, ~0.62 at
nKeys=6-7) despite the M=20 synthetic-weight gate (which never varies the
kernel choice, since ITS reference also goes through the plain kernel) staying
bit-exact throughout. Does not cover kvI8 — that path binds a different
(9-argument) bind group shape entirely; callers that might see kvI8 must keep
handling it separately, as decoderunner.go already does.
```

## ensureAttn: guard

Moved from `gpu/attention.go` (the comment inside `ensureAttn`) on 2026-10-09.

```text
Guard on the LAST pipeline built, not the first: these are created in order, so a non-nil last
field means every earlier one succeeded too. Guarding on ropePipeline (the first) let a
mid-build failure leave the guard satisfied with later pipelines nil, and the next call
dispatched a nil pipeline (audit R-30). On a retry after a partial build the earlier fields are
rebuilt (the old ones stay tracked for release at Close — bounded, not leaked).
```

## attnWideTemplateWGSL

Moved from `gpu/attention.go` (the comment above `attnWideTemplateWGSL`) on 2026-10-09.

```text
The WIDE single-query attention kernel: one template, STRIDED lanes.

The shipped narrow kernels put one lane on each head dim, which makes the workgroup width a
hard ceiling on head_dim — 128, and 256 is as far as that idea can go because 256 is WebGPU's
guaranteed maxComputeInvocationsPerWorkgroup. That is a higher wall, not the absence of one,
and the wall is real: this family's released checkpoints are head_dim 256 and Gemma 4's global
layers are 512.

So the wide variant gives each lane a STRIDE of dims — d0, d0+WG, d0+2·WG, … — and any
head_dim up to attnWGWide·attnMaxPerLane works with a fixed workgroup. The narrow kernels are
left untouched and still serve head_dim ≤ attnWG, because striding costs a per-lane array and
two extra loops that every ordinary model would pay for nothing.

One template rather than three near-identical copies: the algorithm (online softmax, the
tree-reduce, the rescale) is the part that is easy to get subtly wrong and hard to notice, and
three copies of it is three places for a fix to land in two. The three variants differ ONLY in
how a K/V element is fetched, which is the part that is obvious on sight.
```

## ensureAttnWide: guard

Moved from `gpu/attention.go` (the comment inside `ensureAttnWide`) on 2026-10-09.

```text
N-14: guard on the LAST pipeline, not the first. This compiles THREE variants (f32, f16
KV, int8 KV) and returned early when the FIRST existed — so a failure on the second or
third left those nil, and the next call reported success while a kvF16/kvI8 plan went on
to bind a nil pipeline. That is the R-30 class, in the file R-30 fixed: a
partially-completed lazy init that looks complete.

The three are all-or-nothing (any error returns before assigning the rest), so the last
being non-nil implies all three are.
```

## webgpuBackend

Moved from `gpu/backend.go` (the comment above `webgpuBackend`) on 2026-10-09.

```text
webgpuBackend runs MatmulBT on a WebGPU adapter (Vulkan / Metal / D3D12) via
the Context foundation in this package. Built only under -tags gpu.

Resident weights: a model weight matrix is constant across every token, so
the first MatmulBT for a given weight uploads it to a GPU storage buffer and
caches the handle (keyed by the slice's backing pointer); every later call
only uploads the (small) activation and reads the result back. This removes
the catastrophic per-token re-upload of the weights (the LM head alone is
the ~671 MB embedding). On any per-call GPU error it falls back to the CPU
matmul, so results are always correct.

Still naive beyond that: each matmul is its own synchronous dispatch +
readback, so decode is latency-bound on per-matmul round-trips. Keeping the
activations resident on-device across a layer's matmuls (and porting
norms/rope/softmax to WGSL) is the remaining work for a GPU-fast forward.
```

## webgpuBackend.MatmulW4A8

Moved from `gpu/backend.go` (the comment above `webgpuBackend.MatmulW4A8`) on 2026-10-09.

```text
MatmulW4A8 is MatmulW8A8's int4 (W4A8) twin — G6 (docs/tasks/task-gpu-paths-2026-09.md), the "staged
int4" item: decoder/weightmat.go's matmulInto never consulted a backend for int4 before this
(its int8 branch already did, via MatmulW8A8/QuantBackend), so an int4-quantized model on the
STAGED (non-resident) path ran every projection on the CPU regardless of which backend was
active — gpu/gemv_w4a8.go's kernel, upload paths and decodeWeight interface already existed,
but only gpu/residency.go's RESIDENT uploadProj used them.

M=1 only (decode): the M>1 (prefill/tiled) case has no int4 GEMM kernel on this backend yet —
declines, so matmulInto's caller falls back to the CPU W4A8 kernel exactly as it did before
this method existed. bQ4 is decoder's native on-disk packed layout (2 nibbles/byte); group is
always w4a8GroupSize (32) for goinfer's models (matches uploadProj's own assumption).
```

## webgpuBackend.MatmulW4A8Batch

Moved from `gpu/backend.go` (the comment above `webgpuBackend.MatmulW4A8Batch`) on 2026-10-09.

```text
MatmulW4A8Batch runs several W4A8 GEMVs that share one activation (fused q/k/v or gate/up,
M=1 decode) as ONE GPU submit — quantize once, dispatch all, sync once. P-16's int4 twin of
MatmulW8A8Batch: staged int4 previously had no batch dispatch on ANY GPU backend, so a fused
call on an int4 model paid one sync PER PROJECTION (three for q/k/v, two for gate/up) instead
of one for the whole group — exactly the per-dispatch overhead MatmulW8A8Batch exists to
remove for int8, never extended to int4. Falls back (returns false) for M>1, a group other
than w4a8GroupSize, or any GPU error — matmulW4A8Batch's caller then uses the CPU batch kernel,
the same decline contract MatmulW4A8/MatmulW8A8Batch already use.
```

## liveBufferBytes

Moved from `gpu/bufaccount.go` (the comment above `liveBufferBytes`) on 2026-10-09.

```text
Live device-buffer accounting.

A full ./gpu/ run used to climb to 7,782 MiB of 8,192 and then fail every later test with
"failed to request device" — an out-of-memory wearing an unrelated message. Context.Close
releases the DEVICE, but buffers uploaded through UploadF32/UploadW8A8/UploadW4A8Packed are
caller-owned, and each live one pins that memory.

nvidia-smi can show the total but cannot say WHICH test left it behind. This counter can:
it is exact, in-process, and read at each test's start, so a full run attributes the growth
to the test that caused it. See TestNoBufferLeak.
```

## §2 fused decode kernels

Moved from `gpu/decodefuse.go` (the comment at the top of the file) on 2026-10-09.

```text
§2 fused decode kernels. The §5 finding: decode is glue-serialization-bound —
the per-token cost is a deep RAW dependency chain of ~535 small dispatches,
each forcing a barrier the GPU can't hide. The lever is critical-PATH length,
so every standalone glue op that can be folded into the kernel it borders
removes a link from the serialized spine. These kernels do that folding; each
is bit-exact with the unfused pair it replaces (same f32 math, same int8 pack).
```

## gegluQuant

Moved from `gpu/decodefuse.go` (the comment above `gegluQuantWGSL`) on 2026-10-09.

```text
gegluQuant is swigluQuant's GELU-tanh-gated twin — Gemma's FeatGatedGELU (G6,
docs/tasks/task-gpu-paths-2026-09.md). Same fused shape (product recomputed in the pack pass, no
inter-wide global-memory round-trip); only the activation differs. gelu_tanh's argument is
CLAMPED to ±15 before calling tanh — see gegluShaderWGSL's own comment (layer.go) for why:
unclamped, it overflows f32 before saturating, which cost Metal's own port a real cosine
regression (0.818→0.994 after the fix) the first time it shipped this exact math.
```

## DecodeTokenFusedBatched

Moved from `gpu/decodetoken_batched.go` (the comment above `DecodeTokenFusedBatched`) on 2026-10-09.

```text
DecodeTokenFusedBatched is the Stage-B (docs/spec/07) batched verify forward: it
runs M token rows (a speculative block at consecutive positions) through the dense
W8A8 layers, with the weight-heavy PROJECTIONS done as M=K tiled GEMMs — each
weight streamed once across all M rows — and the cheap per-row ops (rmsnorm, RoPE,
attention, SwiGLU, residual, KV-store) looped over the rows. This is the
projection-side weight-stream collapse that the per-row resident verify (runBatch:
M separate M=1 GEMVs) cannot do.

It is bit-equivalent to M sequential DecodeTokenFused calls at positions[0..M-1]
(same int8 inputs, same int32 accumulation; the tiled GEMM equals the GEMV): all
rows' K/V are written to the shared cache before any row's attention reads it, so
row i attends to rows 0..i-1 exactly as sequential decode would. Gated by
TestDecodeTokenFusedBatched_parity. Dense W8A8 only (no MoE/MLA/SSM/bias/QK-norm)
— the first arch of the Stage-B rollout.

NOT one command buffer end to end, despite the name's original intent: cogentcore/webgpu's
Metal backend allocates a fresh native MTLCommandBuffer inside every ComputePassEncoder.TryEnd()
(wgpuComputePassEncoderEnd -> wgpu_hal Metal begin_encoding -> -[MTLCommandQueue
commandBufferWithUnretainedReferences]), and MTLCommandQueue caps how many can exist
uncommitted at once — block on that cap and Submit() is unreachable because nothing already
queued will ever be committed to free a slot. A single encoder spanning all M rows x all
layers hit that cap even at the smallest tested dims (qwen0.5b, M=8/L=8, ~1000+ passes) and
deadlocked forever (confirmed by sampling the live process: near-zero CPU, parked on
semaphore_wait_trap under Metal's command-buffer allocator — TestDecodeTokenFusedBatched_microbench
and _largedim both hung this way, unrelated to any dependency version). flushPasses below
commits periodically (Submit with no intervening Poll — same-queue submissions are ordered by
WebGPU/Metal automatically, so KV-cache writes still happen-before later attention reads) to
stay under the cap while keeping the weight-stream-once-per-layer win this function exists to
measure: the amortization is across M rows within a layer, not across the whole block.
```

## deltaRuleShaderWGSL

Moved from `gpu/deltanet.go` (the comment above `deltaRuleShaderWGSL`) on 2026-10-09.

```text
Resident Gated-DeltaNet decode step (Qwen3.5/3.6-MoE, Qwen3-Next, Qwen3.8).

N-34 (09-02): this comment used to say the mixer below "makes every DeltaNet hybrid CPU-only
on every backend today" — true when it was written, false once this file shipped. Status per
decoder/features.go's FeatDeltaNet declarations: webgpu (here) and metal (metal/deltanet.go,
metal/deltanet_kernels.go) both implement the full family, dense and MoE siblings alike; cuda
(cuda/deltanet.ptx) implements it too, gated end-to-end by TestQwen35ResidentParityCUDA. See
docs/deltanet-residency-plan.md (DONE AND MEASURED 2026-08-19 for this backend, 11.4-12.2x CPU
decode) for the webgpu measurement this file is the result of.

It slots onto the resident DecodeRunner the same way the Mamba-2 engine does — the conv window
and the gated norm are that engine's (mambaConv / mambaGNorm, same shapes) and the persistent
per-layer state is a storage buffer updated in place per token. Only the delta rule below is new.

THE STATE IS STORED TRANSPOSED RELATIVE TO THE CPU, and that is the point of the port rather
than an implementation detail. decoder/deltanet.go holds S as [hk, hv] and walks it COLUMN-wise:

	for vd:  for kd: kv += S[kd*hv+vd]*k[kd]          // stride hv — a cache line per access
	         for kd: S[kd*hv+vd] += ...; o += ...     // second pass over the same memory

Here S is [hv, hk], so thread (headV, vd) OWNS the contiguous row S[headV][vd][0:hk] and reads it
with stride 1. That gives the same thread-owns-its-row race-freedom mambaSSM relies on (no scan,
no atomics, no position loop) and fixes the CPU's strided access and double pass at the same time.

Per thread, for its own output element vd:

	S[kd] *= gt                        // decay (gt = exp(negExpA·softplus(a+dt_bias)))
	kv     = Σ_kd S[kd]·k[kd]
	delta  = (v[vd] − kv)·beta         // beta = sigmoid(b)
	S[kd] += k[kd]·delta ; o += S[kd]·q[kd]
	out[headV*hv+vd] = o

vBase lets the caller bind the WHOLE post-conv [q|k|v] buffer and point at its v slice, the
alignment-free trick mambaSSMOp uses for its in_proj slices — a byte offset of 2*keyDim*4 is not
guaranteed to satisfy minStorageBufferOffsetAlignment at every geometry.

q and k arrive ALREADY l2-normalized (deltaNorm below): the norms are per KEY head, so computing
them inside this kernel would repeat each one hv times — the same work as the recurrence itself.
```

## deltaQSplitShaderWGSL

Moved from `gpu/deltanet.go` (the comment above `deltaQSplitShaderWGSL`) on 2026-10-09.

```text
The SOFTMAX layers of this family are not ordinary GQA either, and that is easy to miss: with
attn_output_gate, q_proj emits [query ‖ gate] PER HEAD at double width, and the attention
context is scaled by sigmoid(gate) before o_proj. Two small kernels rather than a load-time
weight split, because the weight is quantized — slicing rows out of an int4 WeightMat with its
per-group scales is real surgery, while splitting the [nH*2*hd] activation is 6144 threads of
copy.

deltaQSplit: qg[head*2*hd .. ] → q[head*hd ..] and gate[head*hd ..]. Interleaved PER HEAD, not
two concatenated blocks; reading it as two blocks measures cosine 0.90 with a DRIFTING
signature (TestQwen35ResidentParity mutation W1b) — plausible logits from the wrong tensor.
```

## slotsBeforeContext

Moved from `gpu/kv_slots.go` (the comment above `slotsBeforeContext`) on 2026-10-09.

```text
slotsBeforeContext is MC1's "slots before context" on WebGPU (owner decision 2026-09-27, the rule CUDA's
ctxForSlots applies): when more than one KV slot is requested and the caller did not choose the context
(decoder.Model.ResidentContextPinned — a fit-guard auto-pin is a one-slot ceiling, not a choice), give up context,
down to webgpuSlotCtxFloor, until every requested slot fits; below the floor the slot count is clamped instead, by
buildKVSlots as always. An explicit -ctx is never shrunk.

darwin only, for now: there the device's memory is host RAM and darwinKVSlots can price a context before anything
is allocated. A discrete GPU has no free-memory query on this backend, so what fits is learned only by allocating;
shrinking there waits on a measurement of how a failed allocation behaves on real Vulkan hardware (the MC1-WebGPU
nobara prompt). KV is linear in the context, so "fits" is monotone and a binary search finds the edge.
```

## Stage 2: device-resident activations

Moved from `gpu/device.go` (the comment at the top of the file) on 2026-10-09.

```text
Stage 2 — device-resident activations. At M=1 decode the activation is a few
KB, so the per-token cost isn't data movement, it's LATENCY: the Stage-1
MatmulW8A8 calls Poll(true) to map its readback, and ~7 matmuls × N layers =
hundreds of synchronous round-trips per token. Keeping activations in device
buffers lets a chain of matmuls submit back-to-back and sync ONCE.

Chaining W8A8→W8A8 needs the one glue op that was on the CPU — int8
re-quantization of a matmul's f32 output — moved onto the GPU. quantizeShader
does it: one workgroup per row computes the row max-abs (a 64-lane tree reduce),
then all lanes quantize+pack to the same 4×int8/u32 layout MatmulW8A8 consumes.

The reduce was a serial scan on lane 0 until 2026-09-02, justified as "trivial at
decode; the rows run in parallel". That is exactly backwards: decode is M=1, so
there is only ever ONE row and nothing to run in parallel — one lane scanned the
whole row while 63 idled at the barrier, in a single-workgroup dispatch. Measured
37 µs/dispatch against rmsnormQuantWGSL's 7.9 µs for strictly more work; see the
ablation profile in TestDecode_dispatchProfile and G35 in docs/QUEUE.md.
```

## readbackRaw

Moved from `gpu/device.go` (the comment above `readbackRaw`) on 2026-10-09.

```text
readbackRaw reads n f32 elements from a raw *wgpu.Buffer whose REAL release stays exactly
where it already was — a `defer buf.Release()` in the caller. This function only fixes the
ACCOUNTING: newDeviceBuffer's accountAlloc is paired with accountFree before returning,
WITHOUT calling Close/Release on buf itself.

V-22 (docs/review-2026-09-04.md), and a mutation-testing catch on the fix: the first version
of this helper called db.Close() to balance the accounting — which also runs buf.Release(),
double-releasing a buffer the caller's OWN defer buf.Release() already owns. That corrupted
the allocator: the accounting-only symptom of the ORIGINAL bug (LiveBufferBytes growing
unbounded, easy to see) was replaced by a SIGTRAP inside wgpu-native on a LATER, unrelated
CreateBuffer call — reproduced on TestZZRepeatLayerNorm's 3rd iteration, harder to see and
worse than the bug being fixed. Confirmed against the ORIGINAL (pre-V-22) code with the same
repeated-call test: no crash, LiveBufferBytes growing every call — the leak, not a corruption.

The original bug this whole helper exists for: every Readback(newDeviceBuffer(buf, n)) call
site built a throwaway wrapper, read it, and threw it away — accountAlloc ran, nothing ever
ran accountFree, so LiveBufferBytes grew by that buffer's size on every call and never came
back down, for a real GPU allocation that WAS correctly released by the caller's own defer.
```

## matmulTiledW8A8BiasKernelWGSL

Moved from `gpu/gemm.go` (the comment above `matmulTiledW8A8BiasKernelWGSL`) on 2026-10-09.

```text
matmulTiledW8A8BiasKernelWGSL is matmulTiledW8A8KernelWGSL with one addition: a
per-output-column bias, added in the SAME expression as the dequant multiply
(`f32(acc) * aScales[row] * bScales[col] + bias[col]`) — textually identical
operand order to gemvW8A8BiasShaderWGSL's epilogue (gemv.go), which is the
point. PrefillLastW8A8 used to do this as TWO dispatches (this kernel without
bias, then a separate residualShaderWGSL add) — mathematically the same
formula, but forced through an f32 round-trip through memory between the
multiply and the add, which a single WGSL expression may evaluate with an FMA
contraction the compiler cannot apply across two separate dispatches. Measured
real effect (TestLocalize_BiasEpilogue, qwen2.5-coder-0.5b layer 0, real
weights): 98 of 896 elements differed between the two forms, all within 2.4e-7
absolute — tiny in f32 terms, but large enough that a subsequent int8 quantize
of an element sitting near a rounding boundary can flip which bucket it lands
in, and 24 layers of that compounds into the observed cosine ~0.99x /
maxAbs ~1 divergence in final logits. This kernel exists so PrefillLastW8A8
can dispatch the identical single-expression epilogue gemvBias does, for M
rows at once instead of M separate GEMV calls.
```

## Register-blocked tiled W8A8 GEMM

Moved from `gpu/gemm_rb.go` (the comment at the top of the file) on 2026-10-09.

```text
Register-blocked tiled W8A8 GEMM (R10, docs/measurements/webgpu-prefill-profile-2026-09-22.md).

The 16×16 kernel (matmulTiledW8A8KernelWGSL) gives each thread ONE output: per packed
word of K it does two shared-memory loads for one dot4, and profiled at a flat ~1 TFLOPS
— 81–94% of batched prefill at ~11% of this card's f32 peak. This kernel keeps the same
staging idea but gives each thread a 4×4 block of outputs in a 64×64 workgroup tile: per
word of K, eight shared loads feed sixteen dot4s, four times the arithmetic per byte
moved through shared memory, with the accumulators in registers (four vec4<i32>).

The output assignment is STRIDED, not contiguous — thread (lid.x, lid.y) owns rows
lid.y + {0,16,32,48} and columns lid.x + {0,16,32,48} — so that a warp's shared reads
(Bs[w][lid.x + 16j]) and its dst writes are consecutive across lanes, and the shared
tiles are stored word-major with a padded row stride (LD = 65) so the staging stores
(consecutive lanes write consecutive w for one r) do not bank-conflict.

BIT-IDENTICAL TO THE 16×16 KERNEL BY CONSTRUCTION. The K reduction is an exact i32 sum
(|acc| ≤ K·127² ≈ 1.4e8 at K = 8960, far under 2³¹), summed in the same ascending-K
order, and the dequant epilogue is the identical single expression
(`f32(acc) * aScales[row] * bScales[col]`, `+ bias[col]` in the bias form). Pinned by
TestTiledRB64_bitIdentical across aligned and ragged shapes.
```

## f32to16

Moved from `gpu/gemv_w4a8.go` (the comment above `f32to16`) on 2026-10-09.

```text
f32to16 is an alias for the ONE converter this package uses; see f32ToF16 in f16.go.

N-04: there used to be two different float32→half converters in this package, and this was the
load-bearing one — every W4A8 group-scale upload and NewKVCacheF16 go through it. It FLUSHED
THE ENTIRE SUBNORMAL RANGE (`exp <= 0 → sign`), so an int4 group whose scale is below 2^-14
read as all-zero on WebGPU and nowhere else. The 2026-08-05 audit's C-15 fixed exactly this in
cuda/ and gpu/ was not in the disposition.

It also made GOINFER_INT4_F16_SCALES lie: that diagnostic claims to reproduce WebGPU's unpack
on the CPU, and for those groups it did not.
```

## gegluShaderWGSL

Moved from `gpu/layer.go` (the comment above `gegluShaderWGSL`) on 2026-10-09.

```text
gegluShaderWGSL is swigluShaderWGSL's GELU-tanh-gated twin — Gemma's FeatGatedGELU (G6,
docs/tasks/task-gpu-paths-2026-09.md). The tanh argument is CLAMPED to ±15 before calling tanh:
unclamped, tanh's argument overflows f32 before saturating at Gemma's activation magnitudes,
producing NaN — the exact defect Metal's own port hit (metal/kernels.go's glu_act, logit
cosine 0.818→0.994 after the clamp fix, decoder/features.go's FeatGatedGELU-adjacent note).
tanh itself saturates to ±1 by |arg|~9, so the clamp is a correctness fix at f32 overflow, not
an approximation — every other family (SwiGLU) never reaches this branch at all.
```

## SetAdapter: partial-bind cleanup

Moved from `gpu/lora_resident.go` (the comment inside `SetAdapter`) on 2026-10-09.

```text
N-82 (docs/audit-2026-09-10.md): an mk error partway through used to return immediately,
leaking every already-built projection in built (metal/lora.go's own C-04, audit-metal-
2026-09-12.md, already fixed the identical shape there — same defer+bound idiom, applied
here too). Worse than a leak alone: r.loraLayers was already released and nilled above, but
r.steps was left pointing at the PREVIOUS adapter's now-released bind groups — this file's
own doc comment on SetAdapter promises "r.steps restored to r.baseSteps" on any error, which
rebuildSteps() (keyed on r.loraLayers, already nil) is what actually delivers; without
calling it here, a Run() after a failed rebind would dispatch against freed WebGPU
resources. bound latches true only once every projection in every layer has converted
cleanly; the deferred release fires on any earlier return (explicit or panic) or the
rebuildSteps() call afterward, undoing exactly the partial work this call itself allocated
and keeping r.steps consistent with the nil r.loraLayers either way.
```

## testhooks.go

Moved from `gpu/testhooks.go` (the comment at the top of the file) on 2026-10-09.

```text
Code relocated by the B-08 build-tag pass: these are test-only hooks, compiled
only under -tags goinfer_testhooks so they are NOT part of the public API
(audit B-08). See RELEASING.md. Imports are added to satisfy the moved bodies.

HAND-MAINTAINED, NOT MACHINE-GENERATED — despite this file's former name
(testhooks_gen.go, renamed 2026-09-11, audit-2026-09-02.md N-41; a separate
same-day fix gave it back the missing `gpu` tag this build line requires —
see git history). There is no //go:generate directive and nothing
regenerates it; edit it directly like any other file. Mirrors
decoder/testhooks.go and cuda/testhooks.go.
```

## Resident SigLIP vision encoder

Moved from `gpu/vision.go` (the comment at the top of the file) on 2026-10-09.

```text
Resident SigLIP vision encoder — the path to a GPU-fast image prefill
(docs/completed/task-gpu-vision-tower.md). Per-call matmul offload was a measured dead
end (WebGPU's ~1s submit+readback overhead × ~162 matmuls/forward ≈ the whole
runtime). The fix is residency: keep the [np, hidden] activation in device
buffers through all layers, chaining each op as a Submit with NO Poll (queue
order guarantees the dependency), so the forward syncs once. This file builds
the batched (M = np patches) kernels the encoder needs that the decode path
lacks: standard LayerNorm (mean/var, vs RMSNorm), gelu-tanh (vs silu-gated),
and a bidirectional softmax — composed with the existing device matmul /
quantize / residual primitives.
```

## VisionEncoder

Moved from `gpu/vision_encoder.go` (the comment above `VisionEncoder`) on 2026-10-09.

```text
VisionEncoder is the resident GPU SigLIP forward. It uploads the tower once
(int8 matmul weights as ResidentW8A8, f32 norms/biases as device buffers) and
runs ForwardPatches entirely on the device: the [np, hidden] activation stays
resident, every op chains as a Submit, and there is ONE Poll per layer (which
just waits for that layer's compute and bounds memory before the next layer's
scratch is allocated — 8 GB can't hold 27 layers of intermediates at once).
This pays WebGPU's submit/sync cost ~27× instead of the per-op-offload's ~162×
(a measured dead end). vision.Encoder delegates here when a resident backend is
attached (-tags gpu); the default build is pure-Go CPU.
```

## prefillProf

Moved from `gpu/prefill_prof.go` (the comment above `prefillProf`) on 2026-10-09.

```text
R10's prefill-profile step (docs/tasks/red-october.md, docs/measurements/webgpu-prefill-decomp-2026-09-22.md):
per-category wall time for the batched prefill path, mirroring cuda/prefill.go's own profTic/profToc
exactly (same shape, same accepted trade-off — "category boundaries are syncs, so the category sum
runs a bit over the pipelined wall time... the price of per-kernel attribution", cuda/prefill_decomp_test.go's
own doc comment). Four categories, matching R10's own ask ("the GEMM, the attention, the batched
norms/rope and the KV write each carry a number") — one more than CUDA's three (gemv/attn/glue),
because R10 asked norms/rope to be its own class rather than folded into a catch-all.
```

## f32ToF16

Moved from `gpu/f16.go` (the comment above `f32ToF16`) on 2026-10-09.

```text
f32ToF16 is THE float32 → IEEE-754 half converter for this package, byte-for-byte identical to
decoder.f32ToF16bits — the canonical resident-backend representation that cuda/kernels.go's
f32tof16, metal/pack.go and the GOINFER_INT4_F16_SCALES CPU diagnostic all replicate.
Round-half-up plus gradual underflow to subnormals.

N-04: this package had TWO converters and neither matched. gemv_w4a8.go's flushed the whole
subnormal range (so an int4 group scale below 2^-14 read as all-zero on WebGPU only), and this
one used round-to-nearest-EVEN in the normal range where every other backend rounds half up —
so identical inputs could produce different halves on exact ties. One converter now, and
TestF32ToF16_N04 pins it against a local copy of the canonical algorithm, the same way
cuda/f16_convert_test.go does for C-15.

NOT RNE: a lone RNE here would re-introduce the divergence C-15 closed.
```

## Package gpu: status paragraph

Moved from `gpu/doc.go` (the comment in the package doc) on 2026-10-09.

```text
Package gpu is the OPTIONAL WebGPU (Metal / Vulkan / DX12) compute backend
for goinfer's decoder and aikit's encoder matmuls, compiled only under the
`gpu` build tag.

It is the ONE place github.com/oliverbestmann/webgpu (cgo, bundling the
wgpu-native Rust library) is allowed to appear. Every file except this doc
carries `//go:build gpu`, and the backends register themselves through the
decoder/encoder Backend registries on init — so the aikit and goinfer core
modules never import webgpu, preserving their pure-Go / no-cgo promise.

Usage: add a blank import of this package and build with `-tags gpu`:

	import _ "github.com/townsendmerino/goinfer/gpu"

then select the backend (decoder.Options{Backend: "webgpu"} /
encoder NewBackend("webgpu")). Without the tag this module's
implementation is absent and "webgpu" falls back to CPU with a note.

Status (corrected N-34 (09-02) — this said "FOUNDATION cut: a single dst = a·bᵀ GEMM
offloaded" long after that stopped being true; restating a snapshot here just invites the
same drift again, so this points at what stays current instead). This is a full resident
decode runner, not a single offloaded matmul: MoE (routed + gated-shared expert), MLA,
Mamba-2, Gated-DeltaNet (dense and MoE siblings), LoRA, and vision encoding are all
implemented — see decoder/features.go's "webgpu" ResidentBackendFeatures entry for the exact,
enforced capability set (the map a model is admitted against, so it cannot drift from what
actually runs the way a status paragraph can) and the gpu/ test suite (-tags gpu) for
end-to-end parity coverage per family.
```

## Releasing a resident ModelW

Moved from `gpu/modelw_release.go` (the comment at the top of the file) on 2026-10-09.

```text
Releasing a resident ModelW.

Context.Close() releases the DEVICE, but not the buffers a caller uploaded through
UploadF32/UploadW8A8 — those are caller-owned, and each live one holds a reference that
keeps the device's memory alive. DecodeRunner.Release says so explicitly for the scratch
("frees the runner's scratch, not the resident model"); nothing said it for the model.

The consequence was a test-suite failure that read as something else entirely: a full
./gpu/ run climbed to 7,782 MiB of 8,192 and then every later test failed with
"gpu: request device: failed to request device" — an out-of-memory wearing an
unrelated error message. Because the tests that lost their device then SKIP rather than
fail, TestWebGPU_forwardParity and TestResidentForwardN_parity quietly stopped being gates.

Measured while diagnosing it, and worth recording because two plausible causes were
refuted before the real one: 200 Context create/destroy cycles in one process are fine
(churn is not the problem), 63 Contexts can be LIVE at once (a real limit, but the counter
read ~0 at the failure), and file descriptors plateau at 128 against a 524,288 limit.
The only thing that actually ran out was VRAM.
```

## swigluQuant

Moved from `gpu/decodefuse.go` (the comment above `swigluQuantWGSL`) on 2026-10-09.

```text
swigluQuant fuses SwiGLU → activation-quantize: silu(gate)·up is computed,
max-abs'd, and packed to int8 in ONE dispatch, so the inter-wide (8960, ~36 KB)
SwiGLU output never materializes in global memory or crosses a barrier — a
double win under the §5 "per-link drain scales with the bordering kernel's
data" model. The product is recomputed in the pack pass (cheap arithmetic)
rather than staged, so no inter-wide workgroup array is needed.
```

## DecodeTokenFused

Moved from `gpu/decodetoken_fused.go` (the comment above `DecodeTokenFused`) on 2026-10-09.

```text
copies between passes — then ONE Submit + ONE Poll. The per-op-submit DecodeToken
was correct but issued ~500 submits; this collapses them to one, which is what
the §0.5 probe measured as the win. Bit-exact-identical to DecodeToken
(TestDecodeTokenFused_parity).
```

## attnKeysEligible

Moved from `gpu/attention.go` (the comment above `attnKeysEligible` (it sat above `attnKeysDisabled`)) on 2026-10-09.

```text
attnKeysEligible reports whether the key-split attention kernel can serve this geometry.
f32 KV only (the f16/int8 caches have their own packed kernels), hd within the narrow
128-lane kernel, and hd/kvDim both multiples of 4 so the vec4 K/q loads are in bounds and
aligned — without those loads the key-split read pattern wastes ~78% of each L1TEX sector,
which is the whole reason cuda/attn_block.cu reads float4.
attnKeysDisabled force-disables the key-split kernel (GOINFER_ATTN_KEYS=0), so the old
dim-split kernel can be A/B'd in the same binary. Read once: the plan is recorded per
runner, and a mid-run flip would leave a half-converted plan.
```

## moeRouteWGSL

Moved from `gpu/moe.go` (the comment above `moeRouteWGSL`) on 2026-10-09.

```text
MoE residency (Lever C3) — sparse expert routing + dispatch on the GPU so the MoE
families (Mixtral / Qwen2-MoE / GLM / DeepSeek) run on the resident DecodeRunner
instead of the staged path. This file is C3a: the router top-k SELECTION kernel.
C3b adds the indexed sparse-expert GEMV; C3c wires both into the runner.

The selection mirrors the CPU routeExperts (decoder/mlp.go): score the router logits
(softmax for Mixtral/Qwen2-MoE, or per-expert sigmoid for DeepSeek/GLM), optionally add
a per-expert selection bias, take the top-k by selection score, set each chosen
expert's WEIGHT to its un-biased score, optionally renormalize the k weights to sum 1
(Mixtral norm_topk_prob) and scale them (DeepSeek routed_scaling_factor). The
group-limited variant (DeepSeek nGroup>1) is deferred to C3d; this kernel is the
nGroup==1 path. nE is tiny (8–512) so one single-lane workgroup is plenty — selection
is not the cost, the expert GEMVs are.
```

## matmulW8A8ShaderWGSL

Moved from `gpu/quant.go` (the comment above `matmulW8A8ShaderWGSL`) on 2026-10-09.

```text
W8A8 (int8×int8) matmul on the GPU — the Stage-1 "quantized matmul" unlock.
Decode is memory-bandwidth-bound (every token reads all resident weights), so
the win is storing weights as 1-byte int8 PACKED 4-per-u32 (≈4× less traffic
than f32), not the multiply itself. WebGPU has no i8 type and (in this binding)
no dot4I8Packed builtin, so the kernel unpacks four sign-extended int8 per u32
and accumulates in i32 — a few extra ALU ops that don't matter when the kernel
is bandwidth-bound. Math matches linalg.MatmulBTW8A8 exactly:
```

## attnMaxHeadDim

Moved from `gpu/attention.go` (the comment inside the `attnWG` constant block) on 2026-10-09.

```text
attnMaxHeadDim is therefore 2048, which no model is near. That is the point: head_dim
stopped being a reason to decline, rather than the wall moving up one notch to 256.
```

## New: storage-buffer binding limit

Moved from `gpu/gpu.go` (the comment inside `New`) on 2026-10-09.

```text
Raise the storage-buffer binding limit: the DEFAULT device caps it at 128 MB,
smaller than a real model's LM head / embedding at int8 (e.g. 152k vocab ×
hidden ≈ 233 MB). Start from the valid default limit set and bump only the two
size limits to the adapter's max (requiring the adapter's full limit set
verbatim fails — some advertised limits, e.g. maxBufferSize, aren't valid as
required limits). maxBufferSize must be ≥ the binding size.
```

## New: MaxBufferSize

Moved from `gpu/gpu.go` (the comment inside `New`) on 2026-10-09.

```text
Raise MaxBufferSize to the binding max (2 GB on this card) so large single
weights fit — a 7B's LM head is ~272 MB int4 / ~545 MB int8, past the 256 MB
WebGPU default. Set unconditionally: DefaultLimits() leaves MaxBufferSize at
the u64-max "unset" sentinel, so a `<` guard never fires and the device
silently keeps the 256 MB default (it must be a concrete value to take).
```

## W4A8 decode GEMV

Moved from `gpu/gemv_w4a8.go` (the comment above `w4a8GroupSize`) on 2026-10-09.

```text
W4A8 decode GEMV: int4 group-wise weights × int8 activation. The decode
roofline is weight bytes/token; int8 streams ~1.55 GB, int4 ~0.97 GB (½ the
nibbles + ⅛ the f32 group scales), so this is the lever on the 4.3 ms gemv
floor (docs/gpu-assessment.md §0.0, W4A8). Format matches aikit's int4-resident
(`internal/linalg`, group=32, GGUF Q4_K granularity): per row, K/32 groups each
of 32 nibbles + one f32 scale; element i's nibble = (word>>4i)&0xF, value
(nibble−8). group=32 nibbles = 16 bytes = exactly one vec4<u32>, so one group
is one coalesced load. Activations are int8 (per-row aScale, same as W8A8).
```

## UploadW4A8Packed

Moved from `gpu/gemv_w4a8.go` (the comment above `UploadW4A8Packed`) on 2026-10-09.

```text
UploadW4A8Packed is the fast path of UploadW4A8: it uploads int4 weights whose bytes are
ALREADY in the GPU packed layout. The decoder's int4 storage (2 nibbles/byte) is
byte-identical to packNibbles' output when K is a multiple of the 32-wide group (no row
padding — proven by TestInt4LayoutMatch), so the resident upload is a straight
CreateBufferInit of the decoder bytes — skipping the per-element unpack + packNibbles
re-pack that costs ~30 s on a 12 B model (docs/task-mellum2-fast-load.md). q4 is the
decoder int4 bytes (≥ N*K/2); scales the per-group f32 (≥ N*K/32). Requires K%32==0;
callers fall back to UploadW4A8 otherwise. Same nibble value convention (value+8).
```

## gemmRowMaxM

Moved from `gpu/gemm_rows.go` (the comment above `gemmRowMaxM`) on 2026-10-09.

```text
Thin-M W8A8 GEMM for the Stage-B verify (docs/spec/07). The 16×16 tiled GEMM
(gemm.go) is the PREFILL kernel; at the small M of a speculative block (M≈K+1≈8)
it wastes half its 16-row tile and loses to per-row GEMV (measured 0.88×). This
kernel keeps the GEMV's structure — one WORKGROUP per output column n, 64 lanes
coalesce-stride the contiguous weight row — but loads each weight vec4 ONCE and
accumulates it against ALL M activation rows (M register accumulators). So the
weight matrix streams once for the whole block (the Stage-B win) with NO wasted
tile dimension. Arithmetic is identical to M separate GEMVs ⇒ bit-parity.
```

## deltaNormShaderWGSL

Moved from `gpu/deltanet.go` (the comment above `deltaNormShaderWGSL`) on 2026-10-09.

```text
deltaNorm l2-normalizes the per-head q and k slices of the conv output, and applies the query
scale. One thread per KEY head: the work is nk·hk (2048 elements on the real 27.8B geometry),
far too small to be worth splitting further, and doing it here instead of inside deltaRule keeps
the recurrence from recomputing each norm hv times.

The conv output is [q(keyDim) ‖ k(keyDim) ‖ v(valueDim)] contiguous — the same packing the CPU
reference slices, so the layouts agree by construction rather than by comment.
```

## deltaGatesShaderWGSL

Moved from `gpu/deltanet.go` (the comment above `deltaGatesShaderWGSL`) on 2026-10-09.

```text
deltaGates turns the two small per-value-head projections into the pair the delta rule consumes:
beta = sigmoid(b) and the decay gt = exp(negExpA·softplus(a + dt_bias)). nv threads (48 on the
real 27.8B) — trivial work, but it has to happen ON DEVICE, because the alternative is a
round-trip per layer per token.

softplus carries torch's threshold=20 linear branch. Without it exp(a) overflows to +inf for
large a and the decay becomes NaN; with the CPU reference doing the same thing, matching it is
not optional.
```

## darwinKVSlots

Moved from `gpu/kv_slots.go` (the comment above `darwinKVSlots`) on 2026-10-09.

```text
darwinKVSlots prices slots where the device's buffers ARE host RAM (darwin), against the same two ceilings Metal's
guard takes the smaller of (metalMemoryCeiling):
  - the fit guard's share of physical RAM, against everything the build holds: the device weights, the host copy a
    unified-memory backend keeps beside them (decoder.Model.ResidentHostCopyBytes), and one slot of KV each;
  - the memory available when the build STARTED (avail0, read before its first upload, 0 when unknown), against
    only what the build then allocated: the device weights and the slots. The host copy was already resident when
    avail0 was read, so it is not counted again — and a live figure read after the upload would count the weights
    twice, the bug MC1's Metal clamp had (docs/tasks/task-concurrency-2026-09.md, "priced after the build").

An unreadable RAM size grants one, as metalKVSlots does.
```

# Test files

Moved from the comments of the package's `_test.go` files (CC5); same conventions as above.

## TestAdapterProbe

Moved from `gpu/adapter_probe_test.go` (the comment above `TestAdapterProbe`) on 2026-10-09.

```text
TestAdapterProbe is cmd/gate's WebGPU adapter-detection vehicle (audit G-09): a fast,
checkpoint-free check for a real adapter, run as a subprocess (`go test -tags gpu ./gpu/ -run
TestAdapterProbe -v`) exactly the way the CUDA/Metal detection paths shell out to
nvidia-smi/uname — cmd/gate stays free of the gpu build tag and its cgo dependency. Prints one
parseable line to stdout; never itself a correctness gate (see TestMatmulBT_matchesNaive and the
resident-parity tests for that).
```

## TestRoPE_parityAtLongContextCeiling

Moved from `gpu/attention_test.go` (the comment above `TestRoPE_parityAtLongContextCeiling`) on 2026-10-09.

```text
TestRoPE_parityAtLongContextCeiling is N-85 (docs/audit-2026-09-10.md): TestRoPE_parity above
only exercises pos=37, so the concern this measures — WGSL sin/cos range-reducing a large
angle differently from the CPU's f64 reference — was flagged but never actually checked at a
position a real served request can reach. decoder/fitplan.go's fit-by-default heuristic can
grow context capacity to 65536 positions at int8 KV quant (the "i8 ceiling" the finding cites),
so pos=65535 here is that real ceiling, not an arbitrary large number. theta at d=0 (invFreq≈1)
lands at ~65535 rad, matching the finding's own "~6.5e4 rad" figure.
```

## TestRoPE_parityAtLongContextCeiling.measurement

Moved from `gpu/attention_test.go` (the comment at the end of `TestRoPE_parityAtLongContextCeiling`) on 2026-10-09.

```text
MEASUREMENT, not a pass/fail gate: N-85's own point is that this was UNMEASURED, and
reusing TestRoPE_parity's pos=37 tolerance here would silently assert a quality bar this
batch has no basis to pick — whether ~2.9e-3 maxAbs at the realistic long-context ceiling
(decoder/fitplan.go's int8-KV fit-by-default can reach 65536 positions) is ACCEPTABLE for
real model quality is a product decision, not something to decide by copy-pasting a
threshold calibrated for a 1700x shorter position. Recorded here so the real number is on
record instead of "unmeasured" — see docs/audit-2026-09-10.md's N-85 closure note.
```

## TestAttnHeadDimSupported_M12

Moved from `gpu/attn_headdim_guard_test.go` (the comment above `TestAttnHeadDimSupported_M12`) on 2026-10-09.

```text
TestAttnHeadDimSupported_M12 gates M-12: newDecodeRunner must decline a resident plan whose
head_dim exceeds what the single-query attention kernels can dot, at both the model level and
any per-layer geometry override (Gemma 4). Before the guard an admitted arch with an oversized
head_dim silently dotted only the first workgroup's worth of dims and the o-projection consumed
half-zero context.

The bound used to be the 128-wide workgroup. It is now the WIDE kernel's stride reach, which is
2048 — head_dim is no longer a practical reason to decline. The guard stays because the failure
mode it protects against (silent truncation, no error) has not changed, only moved.
```

## TestWebGPUBackend_MatmulW4A8Batch_matchesCPU

Moved from `gpu/backend_w4a8_batch_test.go` (the comment above `TestWebGPUBackend_MatmulW4A8Batch_matchesCPU`) on 2026-10-09.

```text
TestWebGPUBackend_MatmulW4A8Batch_matchesCPU is P-16 (audit-2026-09-10): the int4 twin of
TestWebGPUBackend_MatmulW4A8_matchesCPU, but batched — mimics a fused q/k/v projection (three
ops sharing one activation) the way the qkv/gate-up batch call sites in
decoder/attention.go and decoder/mlp.go actually build them. Compares the ONE-SUBMIT GPU batch
against linalg.MatmulBTW4A8Batch, the CPU reference matmulW4A8Batch falls back to when no GPU
backend implements QuantBatchBackend4 — the exact bar P-16 closes.
```

## TestWebGPUBackend_MatmulW4A8Batch_oneSubmit

Moved from `gpu/backend_w4a8_batch_test.go` (the comment above `TestWebGPUBackend_MatmulW4A8Batch_oneSubmit`) on 2026-10-09.

```text
TestWebGPUBackend_MatmulW4A8Batch_oneSubmit is the actual claim P-16 makes: the batch call
dispatches all ops in ONE GPU submit (one increment to b.fallbacks-free success), not one
submit per op — proven indirectly by asserting the SAME cached *ResidentW4A8 (by identity via
residentW4A8For's key map) is reused across repeated batch calls with the SAME weight bytes,
which only holds if MatmulW4A8Batch and MatmulW4A8 share the upload/cache path. A prior
MatmulW4A8 call on op 0's bytes must make the batch call for the SAME bytes a cache hit (no
re-upload), proving both paths route through the one shared cache this fix introduced.
```

## TestWebGPUBackend_MatmulW4A8_matchesCPU

Moved from `gpu/backend_w4a8_test.go` (the comment above `TestWebGPUBackend_MatmulW4A8_matchesCPU`) on 2026-10-09.

```text
TestWebGPUBackend_MatmulW4A8_matchesCPU is G6's staged-int4 gate (docs/tasks/task-gpu-paths-2026-09.md):
webgpuBackend.MatmulW4A8, the actual decoder.QuantBackend4 entry point matmulInto/matmul now
call, against the CPU reference (linalg.MatmulBTW4A8Into) — the same comparison
TestWebGPUBackend_matchesCPU does for the f32 path, but nothing previously did this directly
for MatmulW8A8 either (worth noting: this establishes the pattern, not just mirrors it).

K=517 is deliberately NOT a multiple of 32 (w4a8GroupSize): it exercises both the fallback
unpack-and-repack upload path (K%32==0 is the fast path) and GEMVRunner's zero-tail-padding
invariant (aBuf is sized to kPad() bytes, and CreateBuffer zero-inits the untouched tail —
see GEMVRunner.Run's own comment).

N-88 (docs/audit-2026-09-10.md): K=517 alone only ever drove the fallback path — the
production K%32==0 fast path (residentW4A8For's UploadW4A8Packed branch, the byte-identical
straight-upload one real checkpoints with 32-aligned dims actually take) was covered only
transitively by whatever else happened to call it, not by this dedicated correctness gate.
K=512 below is a real, common 32-aligned dim (checked directly against the same CPU
reference) added alongside 517 so this gate exercises both upload paths itself.
```

## TestNoBufferLeak

Moved from `gpu/bufaccount_test.go` (the comment above `TestNoBufferLeak`) on 2026-10-09.

```text
TestNoBufferLeak pins V-22 (docs/review-2026-09-04.md): bufaccount.go's own comment has
pointed callers here since before this test existed ("See TestNoBufferLeak") — a doc comment
claiming coverage that isn't there is worse than no comment, because it reads as verified.

Two real accounting bugs let LiveBufferBytes grow without bound even though the underlying GPU
memory WAS correctly released:
  - Readback(newDeviceBuffer(buf, n)) built a throwaway *DeviceBuffer (accountAlloc), read it,
    and threw it away — nothing ever called Close (accountFree). Fixed via readbackRaw, which
    owns the wrapper for exactly the call and closes it before returning.
  - FusedMLP wrapped the SAME xn/mid buffer in newDeviceBuffer TWICE — once kept in `keep`
    (released at the end) and once more, locally, for quantizeDevice — double-accounting one
    real allocation. Fixed by reusing the single kept wrapper instead of building a second.

This drives each fixed path several times and asserts LiveBufferBytes returns to its PRE-CALL
baseline every time — not just "does not exceed some threshold," which a slow leak could still
pass for a while. A resident weight fixture (rmsWDev/gateRM/upRM/downRM) is created once and
deliberately stays live across iterations — that is real, intended residency, not a leak — so
the baseline is taken AFTER the fixture, not before it.
```

## TestClose_drainsTrackedReleasesLIFO.filecomment

Moved from `gpu/close_idempotent_test.go` (the comment at the top of the file) on 2026-10-09.

```text
Audit C-26 — Context.Close must release everything, exactly once, however many times it is called.

WHY THIS EXISTS. Close carried a hand-maintained list of per-field Release calls that had drifted
to 14 of ~40 pipelines: every ensure* builder added after the list was written simply leaked, and
ensureVision's five shader modules were never even stored, so no later code could release them.
Worse, Close was not idempotent — and `defer m.Close()` next to an explicit `m.Close()` is the
ordinary Go shape, while decoder.Model.Close calls m.be.Close() unconditionally. The second call
re-released live wgpu handles: a use-after-free inside the native layer, on GPU machines only.

The fix is structural rather than a longer list: objects register a release closure AT CREATION
(mkPipeline / track), Close drains that list LIFO, and a `closed` flag makes the whole thing a
no-op the second time. These tests pin the two properties a hand-list could not guarantee.

NO DEVICE NEEDED: track takes plain closures, and Close nil-checks the base handles, so a
zero-value Context exercises the drain and the idempotency guard.
```

## BenchmarkDecodeStaged_prize

Moved from `gpu/decode_staged_prize_test.go` (the comment above `BenchmarkDecodeStaged_prize`) on 2026-10-09.

```text
TestDecodeStaged_prize sizes the §2 (widen resident-runner eligibility) lever: the
per-token cost of the STAGED fallback path (recreates ~330 bind groups + uniforms +
scratch buffers per token) vs the resident DecodeRunner. It measures the SAME model
both ways (resident, then GOINFER_NO_RESIDENCY=1 → staged) so the delta is purely the
path, no size-normalization. A genuinely-ineligible family (Gemma-4 softcap) is also
measured to confirm the staged cost on weights that REQUIRE the fallback.

Pure measurement; reports best (throttle-free) inter-token rate via decoder.Generate.
G-10: a Benchmark, not a Test. It reports numbers and asserts nothing, so as a Test*
its green said only that the harness ran — not that the effect it maps is there.
Go runs a benchmark this slow exactly once (N=1 already exceeds benchtime).
```

## gpuTimePlanTWE

Moved from `gpu/decode_twe_split_test.go` (the comment above `gpuTimePlanTWE`) on 2026-10-09.

```text
gpuTimePlanTWE returns the min/30 on-GPU ms for the runner's dispatch plan via a
timestamp query, or -1 if the device lacks the timestamp feature (so the harness
degrades to TSync wall-clock). The old cogentcore/webgpu binding lacked
pass-descriptor TimestampWrites, so this used to bracket the pass with a
standalone encoder WriteTimestamp instead; oliverbestmann/webgpu removed that
method entirely (it never survived into the current WebGPU spec — native
backends couldn't implement it reliably outside a pass) and only exposes
TimestampWrites on the pass descriptor, so this now uses that directly.
```

## TestDecodeRunnerW4A8_7B_fit.owned

Moved from `gpu/decoderunner_w4a8_bench_test.go` (the comment inside `TestDecodeRunnerW4A8_7B_fit`) on 2026-10-09.

```text
Everything below is CALLER-owned: ctx.Close() releases the device handle but not these
buffers, and runner.Release frees only scratch ("not the resident model"). Unreleased,
this test's ~5.6 GB stayed live for the rest of the process and every later test failed
with "failed to request device" — an out-of-memory wearing an unrelated message.
```

## TestDecodeToken_throughput

Moved from `gpu/decodetoken_bench_test.go` (the comment above `TestDecodeToken_throughput`) on 2026-10-09.

```text
TestDecodeToken_throughput times the one-fence DecodeToken on the real Qwen-1.5B
shapes (28 layers, hidden 1536, GQA 12/2, FFN 8960, vocab 151936) with resident
random weights — the shapes drive the timing. This is the practical payoff: the
per-token decode rate of the one-command-buffer forward, to compare against the
staged W8A8 path (25.6 tok/s E2E, 3.07× CPU). Logs; run -v.
```

## TestDecodeToken_throughput.attribution

Moved from `gpu/decodetoken_bench_test.go` (the comment inside `TestDecodeToken_throughput`) on 2026-10-09.

```text
§5 attribution: time the built plan with only the gemv dispatches vs only
the non-gemv (glue) dispatches, each one pass + submit + blocking Poll, min
over reps. Correctness is irrelevant here — this splits the 42 ms of GPU
execution into matmul-kernel time vs glue-kernel time directly.
```

## TestDecodeToken_throughput.roofline

Moved from `gpu/decodetoken_bench_test.go` (the comment inside `TestDecodeToken_throughput`) on 2026-10-09.

```text
Resident int8 weight bytes streamed once per token (the decode roofline
denominator): per-layer qkv+o+gate+up+down, plus the LM head, plus the f32
per-row scales. effective GB/s = weightBytes × tok/s, vs ~448 peak / ~350
streaming on this card.
```

## TestDecodeTokenFusedBatched_largedim

Moved from `gpu/decodetoken_largedim_bench_test.go` (the comment above `TestDecodeTokenFusedBatched_largedim`) on 2026-10-09.

```text
TestDecodeTokenFusedBatched_largedim is the docs/spec/07 follow-up kill-gate: the
Stage-B batched verify is ~0.98× (no win) at qwen2.5-0.5b dims because at M=8 the
resident decode is COMPUTE-bound, so batching's weight-reuse saving recovers nothing.
The open hypothesis: on LARGE models (big hidden/inter) decode becomes
BANDWIDTH-bound on streaming the projection weights, so batching M rows reads each
weight once for all M → real amortization → batched beats per-row×M. This sweeps a
dim ladder and prints the batched-vs-sequential speedup at each. ≥~1.3× at large dims
would justify the (high-risk) Stage-B runner surgery; ~1× confirms it's dead. Run -v.
```

## TestDecodeTokenFusedBatched_largedim.mcap

Moved from `gpu/decodetoken_largedim_bench_test.go` (the comment inside the M-sweep of `TestDecodeTokenFusedBatched_largedim`) on 2026-10-09.

```text
gemmRowMaxM is a HARD kernel limit, not a tuning knob: the thin-M gemmRow kernel
accumulates into a private array<i32, gemmRowMaxM> (gpu/gemm_rows.go, gemmRowMaxM), and
DecodeTokenFusedBatched returns a clean error above it. This sweep used to request
M=32/48 anyway and t.Fatalf on that error, so the whole test reported FAIL for asking
the API to do something it documents it cannot — charging every unrelated leg an
investigation. (Proven unrelated to the MoE cap work in 0018114 by stashing that
commit's only gpu/ change and re-running: identical failure.) The measurable part of
the sweep is M <= gemmRowMaxM; beyond it the answer is "chunk the block", which is a
Stage-B design question, not a measurement.
```

## TestE2E_webgpu

Moved from `gpu/e2e_test.go` (the comment above `TestE2E_webgpu`) on 2026-10-09.

```text
TestE2E_webgpu loads a real model on both the CPU and WebGPU backends and
compares (1) greedy output — the GPU W8A8 path must track the CPU's tokens —
and (2) prefill + decode timing. Gated on GOINFER_E2E_MODEL (a .gguf).

Honest scope: only the NON-fused W8A8 matmuls (o_proj, down, lm_head, router)
route to the GPU today; the fused qkv / gate-up batch dispatches stay on CPU
(QuantBackend doesn't cover MatmulBTW8A8Batch yet). And each GPU matmul still
pays one sync, which the Stage-2 finding showed is the decode floor. So this
measures partial offload — the number that scopes the remaining work, not a
finished decode win.
```

## TestEmbedInt4_residentDecisionDoesNotChange

Moved from `gpu/embed_int4_resident_test.go` (the comment above `TestEmbedInt4_residentDecisionDoesNotChange`) on 2026-10-09.

```text
The WebGPU twin of cuda's TestEmbedInt4_residentDecisionDoesNotChange. Options.EmbedInt4 reaches every family loader since 2026-10-09, so the families that
used to be handed an int8 head by default are now handed an int4 one. The resident must take it without changing its decision: the same decode path with
either table, and the head reading int8 then int4.
```

## TestContextClose_finalizerRace

Moved from `gpu/context_close_race_test.go` (the comment above `TestContextClose_finalizerRace`) on 2026-10-09.

```text
TestContextClose_finalizerRace reproduces the intermittent hang the first heavy-tier gate hit in the webgpu-parity cell (1 run in 8 locally; a different test each time): a test
leaks a buffer, its Context is closed, and the buffer's finalizer later drops the LAST reference on that device on the finalizer goroutine -- concurrently with another
Context's Close releasing its own device on the test goroutine. wgpu-native deadlocks when two devices are dropped at once (the hung run's stacks: one thread in
wgpuDeviceRelease from runtime.runFinalizers -> Buffer.release, one in wgpuDeviceRelease from Context.Close, both parked for 100 s).
The loop leaks one matrix per round, closes its context, opens a second, forces the collection that queues the finalizer, and closes the second at once. Under the
deadline: a hang is a failure that names the cause, not a stuck package.
```

## TestDecodeArgmaxHeadroom

Moved from `gpu/decode_argmax_headroom_test.go` (the comment above `TestDecodeArgmaxHeadroom`) on 2026-10-09.

```text
TestDecodeArgmaxHeadroom is R10's "measure before building" step for the greedy-decode item named
in docs/tasks/red-october.md R10 ("on-device argmax for the greedy path with a K-entry MapAsync ...
so the 608 KB readback goes"): ForwardSample's greedy branch (gpu/resident_sample.go) still calls
Run() — full LM-head GEMV + a vocab*4-byte staging copy + MapAsync + a host linear scan — where a
device-side argmax kernel would need only the GEMV (unavoidable either way) plus a tiny readback.
Isolates the piece a device-argmax kernel would remove: RunNoLogits skips the LM-head GEMV
dispatch, the staging copy, AND the MapAsync/readback entirely, so Run-vs-RunNoLogits bounds what
ANY logits-avoiding scheme (device argmax included) could recover.

RETRACTION, 2026-09-21 (docs/QUEUE.md G38-CORRECTED): the first version of this test read the
delta at 81-85% of the token and reported an "LM-head GEMV 10-17x over roofline" finding — WRONG,
and wrong for a reason worth stating plainly: RunNoLogits calls c.device.Poll(false, nil) —
NON-BLOCKING — while Run() calls Poll(true, nil). Timing RunNoLogits back-to-back without an
explicit blocking poll measures CPU-side submission only, not GPU completion, so the very first
delta compared "wait for everything" against "don't wait at all" — an apples-to-oranges bug in
THIS test, not in RunNoLogits itself (whose real callers do not need synchronous timing). Fixed
by adding an explicit c.device.Poll(true, nil) after RunNoLogits and alternating small blocks of
each arm (not two long back-to-back runs) so drift cannot bias one side. The corrected delta is
4.7-4.8% of the token, reproduced across two independent runs — the on-device-argmax item really
is low-value, but not for the reason first reported, and the "LM-head GEMV" finding is retracted
in full; it does not exist.
```

## TestDecode_dispatchProfile

Moved from `gpu/decode_dispatch_profile_test.go` (the comment above `TestDecode_dispatchProfile`) on 2026-10-09.

```text
TestDecode_dispatchProfile answers "which of the ~13 dispatches/layer actually
pins the decode critical path NOW" — the question every fusion proposal has to
clear first, and the one this repo has already mispredicted twice (§0.0's
link-counting missed by ~3×; the Increment-3 qk-norm fold shipped a −2.7%
regression that the dependent-fold heuristic predicted as a win).

Method: ABLATION, not attribution. For each distinct compute pipeline the
resident plan uses, re-record the whole token plan R times into ONE pass with
that pipeline's dispatches OMITTED, submit once, poll once (blocking → GPU
done), and difference against the unablated plan. The delta / R is what
deleting that class would save — barrier effects included, which is exactly the
quantity a fusion buys. Attribution by per-kernel timing would NOT answer this:
a dispatch that fully overlaps its neighbours costs wall-clock nothing to run
and nothing to remove.

Values go garbage after the first repetition (residual epilogues accumulate R
times). That is deliberate and harmless — both arms are equally garbage, and on
NVIDIA f32 there is no denormal/NaN timing cliff to bias the comparison. This
is a timing harness; TestResidentForwardN_parity is the correctness gate.
```

## TestDecode_dispatchProfile.close

Moved from `gpu/decode_dispatch_profile_test.go` (the comment inside `TestDecode_dispatchProfile`) on 2026-10-09.

```text
Close the RESIDENT model, not just the backend: its weight buffers are caller-owned and
b.Close() does not free them. Unreleased, this profiler held ~2.4 GB for the rest of the
process — the first of the two leaks that were emptying the GPU mid-suite.
```

## TestDecode_instrument

Moved from `gpu/decode_instrument_test.go` (the comment above `TestDecode_instrument`) on 2026-10-09.

```text
TestDecode_instrument decomposes the ~44.7 ms/token that the §1 single-pass
DecodeRunner leaves on the table (34.7 GB/s = ~10% of roofline). Two
hypotheses about this code have already missed by ~4× (the §0.5 1.48× probe,
then §1's "93% pass overhead" → actually ~24 µs/pass), so instead of guessing
where the time goes a third time we measure it:

	(a) one resident gate GEMV (13.76 MB) standalone → its real GB/s. The fork:
	    bandwidth-correct kernel ⇒ the cost is dispatch count → §2/§3 pays;
	    slow kernel ⇒ fusion won't save it.
	(b) a 1-workgroup dependent no-op dispatch → the per-dispatch launch+barrier
	    floor, isolated from kernel work.
	(c) reconstruct the token: fixed + 197 gemv-dispatches + 338 glue-dispatches,
	    and check it sums to the measured 44.7 ms → matmul-time vs glue-time.

Method: record K identical dispatches into ONE compute pass, ONE submit, ONE
blocking Poll; the slope (t(Khi)-t(Klo))/(Khi-Klo) is the per-dispatch GPU
cost with the fixed submit/poll/encode overhead differenced out; the intercept
is that fixed overhead. Logs; run -v.
```

## TestDecode_instrument.census

Moved from `gpu/decode_instrument_test.go` (the comment inside `TestDecode_instrument`) on 2026-10-09.

```text
---- (c) reconstruct the 44.7 ms token from the measured pieces ----
```

## TestF32ToF16_N04

Moved from `gpu/f16_convert_test.go` (the comment above `TestF32ToF16_N04`) on 2026-10-09.

```text
N-04: gpu/ carried TWO float32→half converters and neither matched the canonical one.

	gemv_w4a8.go's f32to16   flushed the ENTIRE subnormal range (exp <= 0 → sign), and it is the
	                         load-bearing one — every W4A8 group-scale upload and NewKVCacheF16.
	                         An int4 group with scale < 2^-14 therefore read as all-zero on
	                         WebGPU and nowhere else.
	mamba_f16.go's f32ToF16  handled subnormals but rounded to nearest EVEN in the normal range,
	                         where every other backend rounds half up.

C-15 fixed this class in cuda/ and gpu/ was not in that disposition. This is the same gate.
```

## TestKVI8ScaleWrittenAtTruePosition

Moved from `gpu/forwardmrope_kvi8_test.go` (the comment above `TestKVI8ScaleWrittenAtTruePosition`) on 2026-10-09.

```text
TestKVI8ScaleWrittenAtTruePosition gates audit-2026-09-10 C-09. The int8 rope-store kernel wrote
each key's per-(position, KV-head) scale at the ROPE position instead of the true sequential one.
So any decode with ropePos != pos (Qwen2.5-VL past an image) overwrote an earlier position's scale
and left its own slot unset. The reader indexes scales by the true position.

The check is the mechanism itself, not a logit cosine. On this model, int8 KV already sits at
~0.96 cosine against f32 on a random-token prompt, which is too noisy to hold a shift bar. Instead:
a fresh model (so the scale buffer starts zeroed) decodes N steps with every rope angle shifted by
delta, and every layer's scale slots [0, N) must then be written, with nothing written past N.
```

## TestForwardMRoPE_constantShiftInvariance

Moved from `gpu/forwardmrope_parity_test.go` (the comment above `TestForwardMRoPE_constantShiftInvariance`) on 2026-10-09.

```text
TestForwardMRoPE_constantShiftInvariance is ForwardMRoPE's FIRST real correctness test on the
WebGPU backend — the CUDA twin lives in cuda/forwardmrope_parity_test.go; see that file's doc
comment for the full rationale (RoPE's relative-angle invariance: a decode run where every
step's rope angle is shifted by the same constant delta must match delta=0 exactly, a property
independent of any specific family's m-RoPE usage).
```

## TestTiledDP4A_parity

Moved from `gpu/gemm_dp4a_test.go` (the comment above `TestTiledDP4A_parity`) on 2026-10-09.

```text
TestTiledDP4A_parity is the Increment-1 gate (docs/completed/task-gpu-batched-prefill.md's
prerequisite): the dot4I8Packed tiled-GEMM kernel must match BOTH the int32 CPU
reference AND the scalar-unpack kernel, at odd dims that exercise every tile
tail. Runs both variants explicitly via runTiledKernel regardless of what this
machine's Context.hasDP4A actually probed, so the DP4A path stays covered on a
non-DP4A CI/test box and the fallback path stays covered on a DP4A box (today
both dev machines probe DP4A-capable, so relying on ensureTiled's live selection
alone would silently stop exercising the fallback kernel).
```

## TestTiledRB64_microbench

Moved from `gpu/gemm_rb_test.go` (the comment above `TestTiledRB64_microbench`) on 2026-10-09.

```text
TestTiledRB64_microbench is the kernel-level signal, at the 1.5B shapes the profile put
the class at ~975 GFLOPS: the two kernels back to back, same buffers, alternating.
```

## TestGemma3ResidentParityWebGPU

Moved from `gpu/gemma3_resident_parity_test.go` (the comment above `TestGemma3ResidentParityWebGPU`) on 2026-10-09.

```text
TestGemma3ResidentParityWebGPU is G6's Gemma set gate (docs/tasks/task-gpu-paths-2026-09.md):
FeatEmbedScale + FeatSandwichNorm + FeatGatedGELU (+ the already-declared FeatQKNorm/
FeatSlidingWindow/FeatRMSAddOne) against a REAL, non-seeded checkpoint
(testdata/gemma3-vl-tiny's text tower — a real small Gemma3 VL model, not a tiny-random
fixture), unlike every G5 smoke test this session wrote. That makes this a genuine numeric
floor, not just an admission-and-no-NaN check: resident vs CPU, both int4, at every position.

FeatFinalLogitSoftcap is NOT exercised here (this fixture has no final_logit_softcapping) —
its own correctness rests on TestApplySoftcap_bitIdentical (softcap_test.go), a direct port of
cuda/metal's own gate, since gpu/softcap.go is a byte-identical copy of their applySoftcap.

Gemma 4's dense-twogeom/dense-scaled fixtures are NOT used here: both have a per-layer
head_dim that differs between local and global attention layers (head_dim 256 vs
global_head_dim 512), and gpu/residency.go's per-layer geometry seam (runLayer.ghd/gnKV) is
never actually set anywhere in this backend — confirmed by grep, and by BuildResident failing
on gemma4-dense-twogeom-tiny with "unsupported projection precision \"\"" even after this row's
four features are declared. That is a SEPARATE, pre-existing gap (per-layer attention geometry
on WebGPU) this row does not touch — CUDA/Metal both implement it, WebGPU does not yet.
```

## TestGemma3ResidentParityWebGPU.stat

Moved from `gpu/gemma3_resident_parity_test.go` (the comment inside `TestGemma3ResidentParityWebGPU`) on 2026-10-09.

```text
Stat the WEIGHTS, not the directory (audit-2026-09-10 G-13(h)). The dir and its config.json
are tracked while the weights are gitignored, so a dir stat passes on every clone. Then Load
failed on the missing weights, and the test skipped saying "no webgpu device".
```

## TestGemma3ResidentParityWebGPU.floor

Moved from `gpu/gemma3_resident_parity_test.go` (the comment at the end of `TestGemma3ResidentParityWebGPU`) on 2026-10-09.

```text
A REAL numeric floor, not a smoke-test bar: 0.999, matched against a genuine checkpoint,
not a seeded/synthetic one — measured 0.9998 on this fixture at implementation time.
```

## TestGemma4DenseTwoGeom_residentParity.bar

Moved from `gpu/gemma4_twogeom_test.go` (the comment inside `TestGemma4DenseTwoGeom_residentParity`) on 2026-10-09.

```text
S1.0 amendment 2026-10-07 (docs/tasks/task-multimodal-support-2026-10.md): the bar sits between the before-v_norm-fix and after readings; the fix is the mechanism. Never loosened. minCosine 0.977930 -> 1.000000.
```

## TestGemma4DenseScaled_webgpuParity.bar

Moved from `gpu/gemma4_twogeom_test.go` (the comment inside `TestGemma4DenseScaled_webgpuParity`) on 2026-10-09.

```text
S1.0 amendment 2026-10-07 (docs/tasks/task-multimodal-support-2026-10.md): the bar sits between the before-v_norm-fix and after readings; the fix is the mechanism. Never loosened. pos0 0.996182 -> 1.000000, minCosine 0.800099 -> 0.975300 (exact-argmax 15/16 -> 16/16 on nobara's RTX 2070 SUPER only; the Apple GPU reads 15/16 either way, so the argmax bar below stays at 15/16).
```

## TestGemma4DenseScaled_webgpuParity.argmax

Moved from `gpu/gemma4_twogeom_test.go` (the comment inside `TestGemma4DenseScaled_webgpuParity`) on 2026-10-09.

```text
15 of 16 is the pre-S1.0 bar, restored 2026-10-07: the 16/16 this line asked for was one device's reading (RTX 2070 SUPER, Vulkan). The
Apple GPU's WebGPU reads 15/16 with the fix, and 15/16 without it, so an argmax count is a near-tie flip and not the discriminating metric.
pos0 and minCosine above carry the fix's signature: with v_norm re-dropped they fail on both devices.
```

## TestGEMVLMHeadIsolation

Moved from `gpu/gemv_lmhead_isolation_test.go` (the comment above `TestGEMVLMHeadIsolation`) on 2026-10-09.

```text
TestGEMVLMHeadIsolation is G38's decisive step: TestGEMVNSweep found the GEMV KERNEL clean and
on-roofline (874us at N=151936, ~267 GB/s) when isolated to its own process with a synthetic
buffer -- yet the SAME shape, SAME N, inside a loaded real model (TestDecodeArgmaxHeadroom) costs
8512us, a ~10x gap the kernel itself does not explain. This isolates the remaining variable: is
it the REAL model's OWN resident lm-head buffer specifically (co-residency/allocator state from
the rest of the model's weights and KV cache), or is it something about running the GEMV as the
LAST dispatch of a big multi-layer pass (Run()'s own recording shape) rather than standalone?

Calls the model's OWN r.lmHead buffer directly via ctx.MatmulW8A8GEMV, in its own
encoder/pass/submit, with a synthetic activation -- BEFORE any Forward() has run (so nothing else
in the process has touched the allocator except the load itself), and again AFTER 200 real decode
steps (so the KV cache and everything else is now also resident, matching TestDecodeArgmaxHeadroom's
own state) -- same buffer, same call shape, two points in the model's own lifetime.
```

## TestGEMVNSweep

Moved from `gpu/gemv_n_sweep_test.go` (the comment above `TestGEMVNSweep`) on 2026-10-09.

```text
TestGEMVNSweep is R10/G38's root-cause step. First run (in-process, many N in one binary) found a
wild-looking cliff between N=65536 and N=100000 with the SAME N reading up to 15x apart depending
on what ran before it -- which turned out to be an ARTIFACT of the sweep's own methodology
(allocating and releasing a differently-sized weight buffer for every N in one process, unlike a
real model, which allocates its LM-head buffer ONCE at load and never touches the allocator again).
This variant runs ONE N per PROCESS (GOINFER_GEMV_SWEEP_N), matching this repo's own
separate-process-per-arm convention for exactly this contamination reason, so each reading reflects
a clean, isolated allocation the way a real model's own load does.
```

## TestW4A8_parity_and_bandwidth

Moved from `gpu/gemv_w4a8_test.go` (the comment above `TestW4A8_parity_and_bandwidth`) on 2026-10-09.

```text
TestW4A8_parity_and_bandwidth is the W4A8 probe: it validates the int4
group-wise GEMV kernel against a CPU reference (the exact same grouped
int8×int4 math) and measures its standalone bandwidth, then projects the
decode token. The question: does int4 actually cut the 4.3 ms gemv floor, and
by how much once the per-group f32 scales (~⅛ extra bytes) are counted?
```

## TestW4A8_parity_and_bandwidth.project

Moved from `gpu/gemv_w4a8_test.go` (the comment inside `TestW4A8_parity_and_bandwidth`) on 2026-10-09.

```text
--- project the decode token vs the measured int8 baseline ---
int8 token = 11.15 ms = gemv 4.3 (1.55 GB @ ~360 GB/s) + (glue+attn+
barriers+host) 6.85 (fixed — W4A8 doesn't touch it). The W4A8 token gemv
streams the whole model at the measured per-matrix byte ratio AND the
measured W4A8 GB/s (int4 unpack is more ALU-bound than int8).
```

## TestGIWInt4_loadtime

Moved from `gpu/giw_resident_test.go` (the comment above `TestGIWInt4_loadtime`) on 2026-10-09.

```text
TestGIWInt4_loadtime times JUST the int4 .giw resident load (no direct cross-load), isolating
the upload from the bf16-read / quantize noise of a direct load. (Its old comparison arm,
GOINFER_INT4_SLOWPATH, was retired 2026-09-24; the fast-vs-slow result is in
docs/completed/mellum2-resident.md.)
```

## TestGIWInt4_resident

Moved from `gpu/giw_resident_test.go` (the comment above `TestGIWInt4_resident`) on 2026-10-09.

```text
TestGIWInt4_resident gates the int4 `.giw` → GPU-resident seam (docs/task-mellum2-fast-load.md
acceptance #2): a prequant int4 bundle must deserialize into the int4 linalg.WeightMat that the
resident builders (dense uploadProj + the stacked-MoE buildStacked "int4" path) consume — i.e.
loading the bundle goes RESIDENT at int4 with NO requant, and decodes token-IDENTICALLY to a
direct int4 load of the source model (the bundle is just those same int4 weights serialized).
This is the one previously-untested step in the fast-load path.
```

## TestGLMResidency_matchesCPU

Moved from `gpu/glm_resident_test.go` (the comment above `TestGLMResidency_matchesCPU`) on 2026-10-09.

```text
TestGLMResidency_matchesCPU is the end-to-end gate for Lever C5 (partial RoPE): the
tiny GLM-4.5 checkpoints (glm4_moe — sigmoid routing + selection bias + ungated shared
expert + dense prefix + partial_rotary_factor 0.5) must go GPU-resident — proving the
partial-RoPE eligibility relaxation + the rotaryDim/2 rope dispatch — and decode
greedily in agreement with the CPU f32 forward. Two fixtures: plain, and with q/k/v
bias + qk-norm. The resident path runs int8, the CPU path f32, so a tiny-random model's
logits eventually flip an argmax under int8 rounding; the gate is the FIRST generated
token, with full sequences logged. Before C5, GLM was off the resident path (partial
rotary tripped the RotaryDim==HeadDim check) — stranding the C3d shared-expert support.
```

## TestGptOssResidentParityWebGPU

Moved from `gpu/gptoss_resident_test.go` (the comment above `TestGptOssResidentParityWebGPU`) on 2026-10-09.

```text
TestGptOssResidentParityWebGPU is G6's gpt-oss gate (docs/tasks/task-gpu-paths-2026-09.md):
FeatAttnSink + FeatOutBias against the REAL, committed decoder/testdata/gptoss_tiny.gguf —
not a seeded/synthetic fixture, so this is a genuine numeric floor, the same class of evidence
gpu/gemma3_resident_parity_test.go established for the Gemma set.

This is the hardest of G6's six features: it needed a per-head softmax sink threaded through
every attention kernel (attn, attn-keys, attn-f16, attn-i8, and all three wide variants — 7
pipelines), plus three brand-new MoE kernels (gpt-oss disagrees with the generic MoE path on
what the router bias means and what the activation clamps — see routeGptOssWGSL/
gptossGluQuantWGSL/moeExpertGptOssDownGEMVWGSL's own comments in gpu/moe.go).

The floor was 0.95 (Metal's own tiny-fixture bar), and this backend measured 0.9943-0.9967
against it. That margin was the defect. The fixture is bias-dominated (audit-2026-09-10 G-07),
so C-06, where every int4 routed expert's down matmul collapsed to its bias, moved the logits only
a few percent and passed. The bar is now cosine >= 0.998 with argmax matching at every position.
It was registered before C-06's fix, from Metal's 0.9989 on the same fixture, and the 0.9943
baseline fails it.
```

## gpuWasAvailable

Moved from `gpu/gpu_test.go` (the comment above `gpuWasAvailable`) on 2026-10-09.

```text
gpuWasAvailable records that some earlier test in THIS process successfully got a device.

It is the difference between the two reasons a device request fails, which used to be
indistinguishable and were both reported as a skip:

	never worked  → this machine has no usable GPU. Skipping is right.
	worked, then stopped → the process EXHAUSTED it (measured: VRAM climbing to 7,782 MiB
	                       of 8,192 as tests retain resident weights). Skipping is wrong:
	                       it silently converts every later gate into a no-op, which is how
	                       TestWebGPU_forwardParity and TestResidentForwardN_parity stopped
	                       verifying anything without anyone noticing.

So the second case now FAILS. A red suite that names the cause is worth more than a green
one that ran nothing — the same reason this project treats a skip as not-a-pass.
```

## TestGraniteResidentParity

Moved from `gpu/granite_resident_parity_test.go` (the comment above `TestGraniteResidentParity`) on 2026-10-09.

```text
TestGraniteResidentParity is the whole-model integration gate (P5b.3 short / P5b.4 long):
resident granite-4.0-h-tiny (Mamba SSM mixer + attention + MoE-every-layer + the 4 Granite
multipliers) vs the CPU runLayersGranite, feeding the SAME fixed token sequence to both and
comparing logits at 1/16/256/1k/2k tokens. State (conv ring + ssm + KV) COMPOUNDS, so an
integration error — a wrong multiplier fold, mixer-kind misroute, or pos off-by-one — shows
as a cosine that DRIFTS long. GOINFER_SSM_NTOK bounds the run (P5b.3 a few; P5b.4 the ladder).
```

## TestGraniteResidentSpeedup

Moved from `gpu/granite_resident_speedup_test.go` (the comment above `TestGraniteResidentSpeedup`) on 2026-10-09.

```text
TestGraniteResidentSpeedup is the P7 deliverable: resident granite-4.0-h-tiny generates
coherently on the resident SSM DecodeRunner, and the measured decode rate beats the CPU
(300 ms/tok) and staged-webgpu (498 ms/tok) baselines. The resident path runs W8A8 (int8);
its logits diverge from the f32 CPU reference (cosine ~0.37 — granite's 64-expert MoE
selection + SSM are int8-sensitive, see docs/ssm-residency-build.md), but the engine is
computationally correct (matches an int8 reference at cosine ~0.99) and the int8 output is
coherent + factual. A tight f32 gate would need f16 weights (follow-up).
```

## TestGraph_oneFencePerToken

Moved from `gpu/graph_test.go` (the comment above `TestGraph_oneFencePerToken`) on 2026-10-09.

```text
TestGraph_oneFencePerToken is the §0.5 probe: does recording a whole token's
dispatches into ONE command buffer with ONE fence beat the current staged path
(113 submit+poll syncs/token for the dense 1.5B)? It records the real per-token
dispatch sequence — 28 layers × [rmsnorm×2, quantize×3, 7 gemv, swiglu] + final
rmsnorm/quantize/lm_head = 367 dispatches at real shapes (H=1536, FFN=8960,
vocab=151936) against resident weights — then times two replays of the IDENTICAL
work: (A) one submit + one Poll; (B) flushed at 113 points (one Poll each). Data
is not a correct forward pass (no attention compute); this measures the fence
cost structure only. Logs; run -v.
```

## requireHeavyModel

Moved from `gpu/heavytest_test.go` (the comment above `requireHeavyModel`) on 2026-10-09.

```text
requireHeavyModel gates a test that loads a multi-GB checkpoint from ~/models behind an explicit
env opt-in — the package-local twin of decoder/heavytest_test.go's helper, same GOINFER_HEAVY_TESTS
key, so `GOINFER_HEAVY_TESTS=1 go test ./...` runs every backend's heavy tests uniformly.

The bug this closes: the gpu (WebGPU) real-model tests decided whether to run by PATH EXISTENCE
alone (os.ExpandEnv("$HOME/models/...") → skip if absent, else Load()). On the Ryzen box with the
model zoo present, `go test ./gpu/` fired the whole zoo opportunistically — climbing to ~75 GB RSS
and thrashing, never finishing. The asset happening to be on disk is not a request to run a
multi-GB test. The per-test os.Stat skip stays as a second guard, so opting in on a bare box is
still harmless.
requireHeavyModel takes testing.TB so the measurement harnesses can be Benchmarks (G-10):
they report numbers and assert nothing, and shipping them as Test* made a green suite include
a PASS that proves nothing about the win they map.
```

## TestInt4LayoutMatch

Moved from `gpu/int4_layout_test.go` (the comment above `TestInt4LayoutMatch`) on 2026-10-09.

```text
TestInt4LayoutMatch checks the hypothesis behind the fast int4 resident upload: the
decoder's int4 storage (2 nibbles/byte, elem k → byte k>>1, low nibble if even) is
BYTE-IDENTICAL to the GPU packNibbles layout (8 nibbles/u32, elem k at nibble k%8 of word
k/8) when K is a multiple of 32 (so kp==K, no row padding). If so, the resident upload can
CreateBufferInit the decoder bytes directly — skipping the unpack + packNibbles that cost
~30 s on a 12 B model — with no new .giw format. Same nibble value convention (value+8)
on both sides, so values are preserved too.
```

## TestKVCacheI8_parity

Moved from `gpu/kv_i8_parity_test.go` (the comment above `TestKVCacheI8_parity`) on 2026-10-09.

```text
int8 rounding acts over thousands of keys and a long-range near-tie can flip. int8
is coarser than f16 (8-bit vs 10-bit), so this lands a touch below f16's measured
0.99868 but comfortably ≥ 0.99 (GPU residency = full-attention Qwen2/Llama, the
least outlier-prone post-RoPE case). The f32 default stays bit-exact elsewhere.
```

## dispatchI8

Moved from `gpu/kv_i8_test.go` (the comment above `dispatchI8`) on 2026-10-09.

```text
dispatch binds storage buffers (binding 0..k-1) + one or more trailing uniforms (binding
k, k+1, …) and runs. G6 (docs/tasks/task-gpu-paths-2026-09.md) widened this from a single trailing
uniform to a slice: attnI8ShaderWGSL now carries TWO (the shared geometry P, and the
genuinely-per-layer HS attention-sink flag — see attnShaderWGSL's own comment for why they
can't be merged into one).
```

## BenchmarkKVLongCtx

Moved from `gpu/kv_longctx_test.go` (the comment above `BenchmarkKVLongCtx`) on 2026-10-09.

```text
TestKVLongCtx sweeps decode tok/s vs context depth for int8 / f16 / f32 KV on the
residency path — the residual the GPU int8-KV + f16-KV tasks left open: the
attention kernel is KV-read-bound, so at long context fewer KV bytes (f32 > f16 >
i8) should make decode FASTER, but the shipped fit test only measured ~1k (0.96×,
weight-stream-bound). This maps the curve at 1k/4k/8k to locate any crossover —
far cheaper than a single 16k point (prefill is O(L²), no batched prefill). One
model load per precision, reused across depths.
```

## TestResidency_kvPrecisionDeclineCoversTheNonGenericBranches

Moved from `gpu/kvprec_decline_test.go` (the comment above `TestResidency_kvPrecisionDeclineCoversTheNonGenericBranches`) on 2026-10-09.

```text
M-32: only the generic GQA branch honoured --kv i8 / --kv f16. The Nemotron, Qwen3.5 and MLA
branches always allocated f32 NewKVCache, while ctxCap was raised by the flag and the kernel
selection was model-wide. Two shapes, both bad in a way the operator cannot diagnose:

	--kv i8  rl.kScale stays nil → bind reports "nil buffer for binding 3 (allocation failed)"
	         → "device allocation failed (VRAM exhausted?)" → the whole model runs on CPU for
	         a reason that is not true.
	--kv f16 each cache is ctxCap×kvDim×4 at the RAISED cap: 2x the intended f16 footprint and
	         2x the f32 default, inverting the "f16 halves KV bytes so 32k fits" premise.

The fix declines instead of implementing quantized KV for MLA's rank-space latent unvalidated.
This asserts the SHAPE of that decision on the source, because reaching the branch needs one
of those three model families resident on a GPU — which no fixture here provides, and a test
that silently skipped would be exactly the "a skip is not a pass" trap.
```

## TestLocalize_BiasEpilogue

Moved from `gpu/localize_bias_epilogue_test.go` (the comment above `TestLocalize_BiasEpilogue`) on 2026-10-09.

```text
TestLocalize_BiasEpilogue is the other Claude session's step-1 localization,
narrowed to the single most likely stage: does production's FUSED gemvBias
kernel (dst = f32(acc)*aScale*bScale + bias, one dispatch, computed in
decoderunner.go for every biased W8A8 projection) produce a BIT-DIFFERENT
result than PrefillLastW8A8's UNFUSED approach (tiled GEMM -> store -> separate
residual-kernel bias add) given the IDENTICAL quantized activation and the
SAME real resident Q weight + bias from layer 0 of a real checkpoint?

If these two disagree even in isolation (no attention, no multi-layer
compounding — just one projection), that's the first differing stage and the
bug is the epilogue fusion (FMA contraction inside one shader vs a forced f32
round-trip through memory between two shaders). If they agree exactly, the
bug is further downstream (rope, the KV cache write, or compounding through
24 layers of an even-smaller difference this test can't see with one call).
```

## TestSetAdapter_partialBindErrorRestoresSteps

Moved from `gpu/lora_resident_bind_leak_test.go` (the comment above `TestSetAdapter_partialBindErrorRestoresSteps`) on 2026-10-09.

```text
TestSetAdapter_partialBindErrorRestoresSteps is N-82 (docs/audit-2026-09-10.md): SetAdapter's
conversion loop used to return immediately on the first mk error, leaking every already-built
projection in built (metal/lora.go's own C-04, audit-metal-2026-09-12.md, already fixed the
identical shape there — TestSetAdapter_partialBindErrorReleasesBuffers is its regression gate).

Worse than a leak alone here: this backend's SetAdapter clears and releases r.loraLayers
UNCONDITIONALLY at the top of the call (correct), but on the old code a mid-loop error left
r.steps — the flat, Go-side dispatch-step list Run actually walks — pointing at whatever
adapter was bound BEFORE this call, which just had its bind groups released one line above by
that same top-of-function release. A Run() after a failed rebind would then dispatch against
freed WebGPU resources. This backend has no buffer ledger to assert a leak count directly
(unlike Metal's d.LedgerLen()), so this test pins the more severe, directly observable half:
r.steps must be reset to r.baseSteps' length on a failed rebind, not left at a stale adapter's
larger spliced-in step count.
```

## TestMambaResidentLayerSweep

Moved from `gpu/mamba_layersweep_test.go` (the comment above `TestMambaResidentLayerSweep`) on 2026-10-09.

```text
Phase-A localization: the mamba path is exonerated (in_proj + mixer correct). The 93.6%→66%
gap is the resident attn/MoE (W8A8 GEMVs) vs D3's staged matmuls. This sweeps
the layer-sweep seam (ssmStopLayerForTest / decoder.SetSSMStopLayerForTest) — rebuilding the resident plan to emit logits from the hidden after
each layer L — and compares the resident's token-0 logits to the f32 CPU reference per L. The
first L where cosine drops names the buggy layer; a gradual drop from every layer ⇒ MoE
(every layer), a jump at 5/15/25/35 ⇒ attention.
```

## TestMambaResidentLayerSweep.reference

Moved from `gpu/mamba_layersweep_test.go` (the comment inside `TestMambaResidentLayerSweep`) on 2026-10-09.

```text
f32 CPU reference, truncated per layer through the test hook (this used to os.Setenv
GOINFER_SSM_STOP_LAYER, which the decoder read once at init, so every "reference" was the full model).
```

## TestMambaRealInputParity

Moved from `gpu/mamba_realinput_test.go` (the comment above `TestMambaRealInputParity`) on 2026-10-09.

```text
Phase-A reconciliation (docs/ssm-int8-quality.md): the GPU conv/ssm/gatedNorm kernels pass
isolation parity on RANDOM inputs over 300 tokens (TestMambaLayer_compose) yet the resident
granite drops to 66%. This test feeds the kernels REAL granite layer-0 inputs — captured from
an actual model run — over a multi-token sequence and diffs the GPU mixer output against
mamba2Step's. Diverge → kernel-math bug in the real-input regime; match → the bug is wiring
(the resident hands the kernels different inputs/state than mamba2Step gets).
```

## TestMambaResidentCapture

Moved from `gpu/mamba_resident_capture_test.go` (the comment above `TestMambaResidentCapture`) on 2026-10-09.

```text
Phase-A wiring diff (docs/ssm-int8-quality.md). The GPU mamba kernels are proven correct on
real inputs IN ISOLATION (TestMambaRealInputParity). This test captures the resident's ACTUAL
per-token layer-0 kernel I/O from the full plan (in_proj output proj, conv, ssm-out y, mixer
gated) and replays the resident's OWN proj through the CPU reference mixer (mamba2.go steps
2-5) with fresh evolving state. Per stage (conv/ssm/gatedNorm):
  - resident matches CPU-on-resident-proj  → the mixer-in-plan is correct on its proj; the bug
    is upstream (proj: in_proj GEMV / activation quant) or in attn/MoE.
  - resident DIVERGES                       → a PLAN/STATE bug (the kernels run differently in
    the full command buffer than in isolation — barrier/race/state corruption).
```

## TestMLAResidency_realDeepSeek_R05

Moved from `gpu/mla_real_resident_test.go` (the comment above `TestMLAResidency_realDeepSeek_R05`) on 2026-10-09.

```text
TestMLAResidency_realDeepSeek_R05 is the regression gate for review-finding R-05: a REAL
DeepSeek MLA base (qk head dim = qk_nope 128 + qk_rope 64 = 192) must NOT be rejected by the
resident decode path's 128-wide single-query-attention head-dim guard (attnHeadDimSupported).
MLA runs its own mlaAttn kernels, so it is exempt from that guard; before the R-05 fix the guard
declined every DeepSeek/Kimi base and residency silently fell back to CPU.

The tiny fixture TestMLAResidency_matchesCPU uses has qkHead=48, so it does NOT exercise the >128
path — this test needs a real DeepSeek. It is size-agnostic about the card: a 16B MoE won't fit an
8GB card fully resident (and MLA can't use the .giw streaming path), so residency may legitimately
decline for VRAM. The gate is the DECLINE REASON, not fit: a head_dim decline means R-05 regressed;
an OOM decline means the build got PAST the guard (the first check in newDecodeRunner) and R-05
holds. Point GOINFER_MLA_MODEL at a DeepSeek/Kimi checkpoint (default: the box's V2-Lite GGUF).
```

## TestMLAResidency_realDeepSeek_R05.heavy

Moved from `gpu/mla_real_resident_test.go` (the comment inside `TestMLAResidency_realDeepSeek_R05`) on 2026-10-09.

```text
The asset happening to be on disk is not a request to run a multi-GB test
(gpu/heavytest_test.go's requireHeavyModel, which this file — package gpu_test — cannot call
directly). Missing this gate let a checkpoint left over from earlier work make an ordinary
`-short` suite run try to load a real DeepSeek-V2-Lite and get killed (G-09's webgpu gate
group ran ./gpu/ automatically for the first time and caught it).
```

## TestMLAResidency_realDeepSeek_R05.runs

Moved from `gpu/mla_real_resident_test.go` (the comment inside `TestMLAResidency_realDeepSeek_R05`) on 2026-10-09.

```text
Resident is the R-05 gate; also decode a few tokens to confirm the resident MLA runner
actually RUNS (a model can report resident yet silently decode on CPU — the 7557723 seam).
```

## TestNemotronBlockKind_hasADefault

Moved from `gpu/n_batch_test.go` (the comment above `TestNemotronBlockKind_hasADefault`) on 2026-10-09.

```text
N-12: BuildResident admits Nemotron regardless of DecodeRunnerEligible, and the block-kind
switch had NO default — so an unhandled kind (nemoMoE) appended a layer with nil weights and
decoderunner.go's gemv nil-dereferenced on the first token. decoder/residency.go calls
DecodeRunnerEligible "the one predicate every backend's admission funnels through" and
documents declining there; this bypass skips exactly that, which makes the switch the last
place the shape is checked. Reachable through the exported ResidencyBackend.BuildResident.
```

## TestEnsureAttnWide_guardsOnTheLastPipeline

Moved from `gpu/n_batch_test.go` (the comment above `TestEnsureAttnWide_guardsOnTheLastPipeline`) on 2026-10-09.

```text
N-14: ensureAttnWide compiles THREE variants (f32, f16 KV, int8 KV) and returned early when
the FIRST existed — so a failure on the second or third left those nil while the next call
reported success, and a kvF16/kvI8 plan then bound a nil pipeline. The R-30 class, in the file
R-30 fixed.
```

## TestGPU_deviceBoundaryErrorsAreNotDropped

Moved from `gpu/n_batch_test.go` (the comment above `TestGPU_deviceBoundaryErrorsAreNotDropped`) on 2026-10-09.

```text
N-15: DecodeRunner.ReadMambaCap is EXPORTED and panicked on a failed buffer map — "test-only"
is a comment, not a compiler constraint. And residentDecoder.Reset dropped every WriteBuffer
error, so a failed C-01 re-zero left the previous sequence's recurrent state in place and the
next generation continued from it silently.
```

## TestNemotronBenignHarmful

Moved from `gpu/nemotron_benign_test.go` (the comment above `TestNemotronBenignHarmful`) on 2026-10-09.

```text
Benign-vs-harmful characterization of int4-resident Nemotron-H's ~7.5% argmax disagreements vs
f32. Near-lossless perplexity/KL with sub-95% agreement should mean the disagreements are
coin-flips on near-tied tokens (int4 picked f32's #2/#3 where f32 was itself indifferent), not
confident-token mistakes. The robust signal is the CORRELATION: benign ⇒ disagreements
concentrate at SMALL f32 top1–top2 margins; harmful ⇒ at LARGE margins (f32 confident, int4 wrong).
```

## TestNemotronMoEResidentParityWebGPU

Moved from `gpu/nemotron_moe_resident_test.go` (the comment above `TestNemotronMoEResidentParityWebGPU`) on 2026-10-09.

```text
TestNemotronMoEResidentParityWebGPU is the G7 part 2 gate (docs/tasks/task-gpu-paths-2026-09.md):
Nemotron-H's fourth block kind (MoE FFN — routed NON-GATED relu² experts plus an always-on
ungated shared expert of the same shape, NOT moeMLP's gated SwiGLU) against
testdata/nemotron3nano-tiny, whose 6-layer block pattern (linear_attention, moe,
linear_attention, full_attention, moe, linear_attention) exercises mamba, attention, AND moe
block kinds in one fixture.

Reuses moeRouteWGSL/moeExpertWGSL/relu2QuantWGSL wholesale — no new kernels were needed. The
router is DeepSeek/GLM's own sigmoid+bias+group-limited-top-k shape (moeRouteWGSL already
implements this; Nemotron's n_group=1 degenerates it to plain top-k, the kernel's own nGroup==1
path), and moeExpert is already generic per single projection (called here for up/down only,
with no expGate at all) — the gap was purely gpu/residency.go's per-layer switch never building
case 3's weights and gpu/decoderunner.go never dispatching nemoKMoE, not a missing primitive.
```

## TestNemotronMoEResidentParityWebGPU.stat

Moved from `gpu/nemotron_moe_resident_test.go` (the comment inside `TestNemotronMoEResidentParityWebGPU`) on 2026-10-09.

```text
Stat the WEIGHTS, not the directory (audit-2026-09-10 G-13(h)). The dir and its config.json
are tracked while the weights are gitignored, so a dir stat passes on every clone. Then Load
failed on the missing weights, and the test skipped saying "no webgpu device".
```

## TestNemotronMoEResident_noSharedExpertBuilds

Moved from `gpu/nemotron_moe_resident_test.go` (the comment above `TestNemotronMoEResident_noSharedExpertBuilds`) on 2026-10-09.

```text
TestNemotronMoEResident_noSharedExpertBuilds is M-35's gate (audit-2026-09-10): a Nemotron-H
MoE block with n_shared_experts=0 (a real, validator-accepted config shape — decoder/config.go's
validateNemotron only errors on the opposite combination, n_shared_experts>0 with the
intermediate size unset) used to fail BuildResident with "unsupported projection precision
\"\"" — the case-3 branch of the per-layer switch projected SharedExpert.Up/.Down
unconditionally, so a zero-value SharedExpert (no tensors for it exist when
moe_shared_expert_intermediate_size is also 0) hit proj() with nothing to wrap. Neither real
Nemotron-H checkpoint has this shape (both ship a shared expert), so this is a config-space
hole, not something the existing parity test above ever exercised.

Derives its fixture from testdata/nemotron3nano-tiny by copying config.json with
n_shared_experts and moe_shared_expert_intermediate_size zeroed (which makes
arch.MoE.SharedIntermediateDim resolve to 0 — decoder/registry.go's nemotron branch reads it
directly from moe_shared_expert_intermediate_size, NOT derived from n_shared_experts, so both
must be zeroed) and symlinking the same real model.safetensors — its now-unreferenced
mixer.shared_experts.* tensors are simply never looked up by either the CPU or GPU loader once
SharedIntermediateDim is 0, so no fresh weights need to be generated.
```

## TestNemotronResidentQuality

Moved from `gpu/nemotron_quality_test.go` (the comment above `TestNemotronResidentQuality`) on 2026-10-09.

```text
P5/P6: the headline. Nemotron-H is the SAME deep hybrid + recurrent-mamba architecture whose
int8 resident gap granite proved FUNDAMENTAL and precision-invariant — but it has NO MoE router,
so the verdict is OPEN. Measure it: resident int4 (int8 doesn't fit 8 GB — see
docs/completed/nemotron-resident.md; granite proved the gap is precision-invariant so int4≈int8) vs the
f32 CPU reference (R1), teacher-forced agreement/KL/top-5/perplexity, plus free-running coherence
and decode speed. The AGREEMENT NUMBER drives the default-vs-opt-in flip — no pre-set bar.
```

## TestPrefill_dispatchProfile

Moved from `gpu/prefill_dispatch_profile_test.go` (the comment above `TestPrefill_dispatchProfile`) on 2026-10-09.

```text
TestPrefill_dispatchProfile is R10's prefill investigation (docs/tasks/red-october.md): "extend
the ablation harness to the batched path at P in {256, 1024}: per-class time with one class
omitted, so the GEMM, the attention, the batched norms/rope and the KV write each carry a
number. Then one arithmetic line: GFLOPS achieved by the GEMM class against the card's
naive-f32 and its shared-memory-tiled expectations."

TestDecode_dispatchProfile's own ABLATION method (re-record R times into one pass, omit one
pipeline class, difference against the full plan) does not transfer: PrefillLastW8A8 is not a
recorded/replayable step list (it builds fresh buffers and issues M-row-scaled dispatches
inline, flushing every 32 passes for Metal's uncommitted-command-buffer cap) — re-running it
with a class's dispatches skipped would cascade into shape/buffer errors for every downstream
stage in the same layer, not a clean subtraction. Instead this uses PER-CATEGORY WALL-CLOCK
ACCUMULATION (gpu/prefill_prof.go's profTic/profToc, wired into PrefillLastW8A8 itself, nil by
default so it costs nothing when off) — the SAME method cuda/prefill_decomp_test.go already
uses for the CUDA side, including its own accepted trade-off: category boundaries are syncs
(c.device.Poll(true, nil)), so the category sum runs a bit over the pipelined wall time — "the
price of per-kernel attribution", reported as such, not hidden.
```

## TestPrefill_dispatchProfile.depths

Moved from `gpu/prefill_dispatch_profile_test.go` (the comment inside `TestPrefill_dispatchProfile`) on 2026-10-09.

```text
GOINFER_PREFILL_PROF_P overrides the depth list. The 1.5B int8int8 OOMs at P=1024 on this 8 GB
card (the WebGPU path does not chunk the way cuda's prefillChunked does; O(M*inter) scratch), so
its 1024 cell comes from the 0.5B — which is also the model the doc's own prior P=1024 reference
(prefill-batched-ttft-2026-09-13.md) used. Recorded, not silently narrowed.
```

## TestPrefill_dispatchProfile.warmup

Moved from `gpu/prefill_dispatch_profile_test.go` (the comment inside `TestPrefill_dispatchProfile`) on 2026-10-09.

```text
WARM-UP, discarded — the decode profiler warms explicitly for the same reason: the first
prefill in a process pays one-time driver JIT for every WGSL pipeline it touches, booked into
whichever category dispatches each one first. Unwarmed, the first P cell read normsRope at
496 ms (P=256) against 50 ms at P=512 for the 1.5B — O(P) work cannot shrink 10x as P doubles.
A single warm call at a small P touches every pipeline the timed cells will use.
```

## TestPrefill_dispatchProfile.selfcheck

Moved from `gpu/prefill_dispatch_profile_test.go` (the comment inside `TestPrefill_dispatchProfile`) on 2026-10-09.

```text
Instrument self-check: causal attention is O(P^2), so its booked time MUST grow with P. The
first version of this profiler booked 544 ms at P=256 and 55 ms at P=512 — the boundary
poll was waiting only for SUBMITTED work while dispatches still sat in the open encoder —
and this assertion is what would have caught it on the spot.
```

## TestGenerate_batchedPrefillMatchesSequential

Moved from `gpu/prefilllast_generate_integration_test.go` (the comment above `TestGenerate_batchedPrefillMatchesSequential`) on 2026-10-09.

```text
TestGenerate_batchedPrefillMatchesSequential is the integration gate docs/completed/task-gpu-batched-prefill.md's
Increment 3 calls for ("TestDecodeParity-class greedy continuation unchanged") but through the
REAL production entry point, not an isolated PrefillLast call.

Increment 3 ("wire into decoder.Generate") reads as not-yet-started in this doc's own Definition
of Done, but it already is: decoder/model.go's residentPrefillSeed (shared by every resident
generation path) has a backend-agnostic `if pf, ok := m.resident.(Prefiller); ok` check that
fires automatically the moment a resident type satisfies decoder.Prefiller — and
gpu.residentDecoder has satisfied it since 813be4e7 (Increment 3's own commit, predating this
branch's dispatch-count fix). So the moment PrefillLastW8A8 works AND is fast, every WebGPU
resident Generate() call on an eligible prompt (>=8 tokens, no adapter) already takes the batched
path — nobody had confirmed that through Generate() itself, only through PrefillLast in
isolation (TestResidentPrefillLast_parity/_TTFT). This test closes that gap: same real prompt,
same greedy sampling, GOINFER_BATCHED_PREFILL=0 (forced sequential, the trusted baseline) vs the
default (batched prefill on — automatically hits PrefillLastW8A8 for this checkpoint's dense
W8A8 Qwen2 shape) must produce the IDENTICAL token stream.
```

## TestResidentPrefillLast_parity.decline

Moved from `gpu/prefilllast_resident_parity_test.go` (the comment inside `TestResidentPrefillLast_parity`) on 2026-10-09.

```text
A decline (this model uses a feature outside PrefillLastW8A8's scope —
see runModelToModelW's doc comment, e.g. Qwen2 bias is deliberately
declined pending a real bug fix) is the SAFE, EXPECTED outcome for many
checkpoints, not a test failure — decoder/model.go's residentPrefillSeed
treats any PrefillLast error identically, by falling back to the
sequential loop. Skip rather than fail; this gate only asserts
correctness for models PrefillLast actually accepts.
```

## TestResidentPrefillLast_TTFT

Moved from `gpu/prefilllast_ttft_test.go` (the comment above `TestResidentPrefillLast_TTFT`) on 2026-10-09.

```text
TestResidentPrefillLast_TTFT is the real end-to-end number the whole
docs/completed/task-gpu-batched-prefill.md build was for: sequential per-token Forward (today's
shipped residentPrefillSeed loop) vs one PrefillLast call, at realistic prompt
lengths, on the actual resident decode pipeline (not an isolated matmul
microbenchmark like TestTiledDP4A_microbench — that one only measured the
projection GEMM in isolation and found DP4A's own contribution modest; this is
the trustworthy comparison for whether batching prefill is worth shipping).
```

## TestResidentQKNorm_parity

Moved from `gpu/qknorm_parity_test.go` (the comment above `TestResidentQKNorm_parity`) on 2026-10-09.

```text
TestResidentQKNorm_parity is the gate for Lever C increment 1: per-head QK-norm in the
resident DecodeRunner (gpu/qknorm.go), which makes Qwen3 (and any dense QK-norm arch)
GPU-resident — previously decodeRunnerEligible excluded QKNorm outright. Two checks:
```

## TestQwen35ResidentParity.siblings

Moved from `gpu/qwen35_resident_parity_test.go` (the comment inside `TestQwen35ResidentParity`) on 2026-10-09.

```text
BOTH siblings, because the resident bridge composes differently for each and "two proven
halves" is the argument that has been wrong here before. The dense one pairs the DeltaNet
mixer with a plain SwiGLU; the MoE one pairs it with the sparse router + stacked experts +
shared expert in the same layer. Mixer+MoE is gated for Mamba-2 (Granite) and the mixer
alone is gated by the dense fixture — neither gates THIS pairing.

V-06 (docs/review-2026-09-04.md): BOTH fixtures' model.safetensors are gitignored
(qwen35ResidentParity's own comment says so — only config.json is tracked), so on a fresh
clone every t.Run below skips. Go reports a parent whose subtests all skipped as a top-level
PASS, not SKIP -- and cmd/gate/gpu.go's webgpu-parity cell counts top-level results only
(TopLevelOnly: true), so that vacuous pass registered as real coverage and made
cr.vacuous() return false. subTs captures each subtest's *testing.T so the parent can check
Skipped() itself after every t.Run returns (checking it INSIDE the subtest closure, after
calling qwen35ResidentParity, would never run: Skip/Skipf call runtime.Goexit, which unwinds
the rest of that closure) and skip itself when nothing actually ran.
```

## TestQwen35ResidentParity.replay

Moved from `gpu/qwen35_resident_parity_test.go` (the comment inside `qwen35ResidentParity`) on 2026-10-09.

```text
SECOND GENERATION on the same model — a REPLAY, not another CPU comparison.

The recurrent state compounds and is NOT positional, so unlike a KV cache the next sequence
cannot simply overwrite it; it has to be zeroed (audit C-01). The first loop cannot see
that: the state buffers are allocated zeroed, so a runner that never resets still passes
its first generation.

Comparing generation 2 against the CPU does not see it either — measured. Deleting the
DeltaNet arm of residentDecoder.Reset and re-running left the cosine above 0.95, because
the decay gate shrinks the stale state faster than 16 tokens of comparison can notice. So
this replays the FIRST run's opening tokens and requires the resident to reproduce ITS OWN
logits. Same inputs + same (reset) state ⇒ the same output, to f32 determinism — a bound
nothing but leftover state can break, and one no tolerance argument can absorb.
```

## TestParentSelfSkipsWhenNoSubtestRan

Moved from `gpu/qwen35_resident_parity_test.go` (the comment above `TestParentSelfSkipsWhenNoSubtestRan`) on 2026-10-09.

```text
TestParentSelfSkipsWhenNoSubtestRan pins the anti-vacuity mechanism V-06 (docs/review-2026-09-04.md)
added to TestQwen35ResidentParity, portably: a stub in place of qwen35ResidentParity so this runs
on any machine regardless of whether the real fixtures are present, and asserts on go test's own
JSON action rather than hardware behaviour.

The bug: Go reports a parent test whose every t.Run subtest skipped as a top-level PASS, not
SKIP — and cmd/gate's webgpu-parity cell counts top-level results only, so that vacuous pass
registered as real coverage and cr.vacuous() stayed false. The fix tracks each subtest's own
t.Skipped() (checked from the OUTER test after t.Run returns — checking it inside the subtest
closure, after a call that itself calls Skip, would never run: Skip calls runtime.Goexit, which
unwinds the rest of that closure) and has the parent skip itself when nothing ran.
```

## TestQwen35ResidentDecodeRate

Moved from `gpu/qwen35_resident_speed_test.go` (the comment above `TestQwen35ResidentDecodeRate`) on 2026-10-09.

```text
Decode rate, resident vs CPU, for the Gated-DeltaNet hybrid.

This exists because residency was landed as a CAPABILITY and the plan's second kill criterion
is still unanswered: "residency is admitted but not faster" is a real outcome, and the
CUDA-graphs precedent says a 1.01× is a safety improvement mislabelled as a speed one. A
capability with no measurement attached tends to get quoted as a speedup by whoever reads the
hardware matrix next.

READ THE FIXTURE SIZE BEFORE READING THE RATIO. On the tiny fixtures (hidden 64, 4 layers) the
per-dispatch overhead dominates completely — roughly 20 GPU dispatches per DeltaNet layer
against a few microseconds of actual arithmetic — so a ratio below 1 here says nothing about the
27B and everything about dispatch cost. The number that matters comes from a real-WIDTH fixture
(GOINFER_DNET_SPEED_CKPT), where the arithmetic is large enough to pay for the dispatches. Both
are reported rather than only the flattering one.
```

## TestQwen35ResidentDecodeRate.extrapolation

Moved from `gpu/qwen35_resident_speed_test.go` (the comment inside `TestQwen35ResidentDecodeRate`) on 2026-10-09.

```text
ms/LAYER is the transferable quantity and the ratio is the robust one; the absolute
per-token figure is not, because this fixture omits the released vocab (248320, whose LM
head is ~5% of the real per-token MACs) and runs a short context. Extrapolating
ms/layer × real layer count overstates the CPU side by roughly 2× against the 0.656 tok/s
actually measured on the real 27B — so quote the RATIO, and treat any absolute
extrapolation as indicative only.
```

## TestResident_C01_pos0ResetsRecurrent

Moved from `gpu/residency_c01_reset_test.go` (the comment above `TestResident_C01_pos0ResetsRecurrent`) on 2026-10-09.

```text
TestResident_C01_pos0ResetsRecurrent is the resident-half gate for audit C-01: the compounding
Mamba-2 {win,ssm} state must be re-zeroed at the start of a fresh sequence, or a second Generate
on the same *Model decodes its first token from the PRIOR sequence's state (cross-conversation
leak, silently wrong output). residentDecoder.Reset() existed but "was called by nothing"; the fix
wired it into Forward at pos==0. This reproduces the leak directly: run token T at pos 0 from a
fresh resident, COMPOUND the state by decoding several more tokens, then run token T at pos 0
AGAIN — with the fix the second pos-0 logits are identical to the first; without it they diverge.

Requires an SSM (Mamba-2 hybrid) resident, which only exists as a real model — this Mac carries no
such checkpoint, so it runs on the CUDA/webgpu box (set GOINFER_HEAVY_TESTS=1; override the path
with GOINFER_SSM_MODEL). The CPU rewind half is covered by decoder.TestTruncateTo_resetsRecurrent.
```

## TestForwardN_declinesRecurrent.filecomment

Moved from `gpu/residency_forwardn_recurrent_test.go` (the comment at the top of the file) on 2026-10-09.

```text
Audit C-17 — batched verify must DECLINE on a recurrent model.

WHY THIS EXISTS. residentDecoder.TruncateTo is a no-op, documented on the premise that "the
resident cache is positional" — true of the KV, false of the Mamba-2 {win,ssm} state the same
runner owns. mamba2Step mutates that state in place, so ForwardN(K) advances it K times and a
partial accept cannot undo the rejected rows: the next round decodes from over-advanced state.
Silently wrong output, not a crash.

WHY IT WAS UNREACHABLE AND STILL NEEDS A GATE. decoder.specRollbackSafe refuses the recurrent
families at the four speculative entry points, so nothing calls this today with mamba set. That
protection lives in ANOTHER PACKAGE, one indirection away from the state it protects — a future
"re-enable recurrent speculation" change relaxes that check and never reads this function. The
guard therefore belongs on the runner that owns the state. This test pins it there.

NO DEVICE NEEDED: the guard runs before any runner/queue is touched, and newRunner is a func
field, so the control case is stubbable.
```

## BenchmarkResidentPackCost

Moved from `gpu/resident_pack_bench_test.go` (the comment above `BenchmarkResidentPackCost`) on 2026-10-09.

```text
TestResidentPackCost measures the CPU byte-shuffle that dominates an int4 resident load
(docs/task-mellum2-fast-load.md): for every projection the bridge unpacks the decoder's
2-nibble/byte int4 into one-nibble-per-element, then packNibbles re-packs to the GPU u32
layout, then packF16Pairs converts the scales — two+ full passes over every param, all
CPU, paid each launch. A GPU-layout .giw would store the packNibbles/packF16Pairs output
directly so the resident load is a straight CreateBufferInit (PCIe ~<1 s for 7 GB). This
extrapolates the per-pass cost to Mellum2's ~12 B params to confirm the lever before
building it. Not an assertion — it logs ms; run with -v.
G-10: a Benchmark, not a Test. It reports numbers and asserts nothing, so as a Test*
its green said only that the harness ran — not that the effect it maps is there.
Go runs a benchmark this slow exactly once (N=1 already exceeds benchtime).
```

## TestResidentPrefixReuse_tokenIdentical.diverge

Moved from `gpu/resident_reuse_parity_test.go` (the comment inside `TestResidentPrefixReuse_tokenIdentical`) on 2026-10-09.

```text
Turn 4 DIVERGES: it shares only the opening words with what the cache holds. That
turn is what makes this gate real. A first version used three strict prefix
extensions only, and a deliberately broken matcher — one that claimed the prefix
WITHOUT comparing ids — passed it, because on an ever-extending transcript the
wrong answer and the right answer coincide. The bug was invisible in exactly the
dimension being tested (CLAUDE.md's minimal-repro trap). Divergence is the only
thing that distinguishes "compared the ids" from "assumed they matched".
```

## TestZZ_decodeRunAllocs

Moved from `gpu/run_alloc_test.go` (the comment above `TestZZ_decodeRunAllocs`) on 2026-10-09.

```text
TestZZ_decodeRunAllocs reports allocs/op and B/op for DecodeRunner.Run via testing.Benchmark,
isolating the per-token host logits allocation. Uses realistic qwen3-class vocab (151936) with
only 2 layers so setup is fast and Run is cheap — the vocab-sized readback slice dominates the
alloc delta this change targets (make([]float32, vocab) → reused r.logitsHost). Compare the
numbers on main (before) vs this branch (after): B/op should drop by ~vocab*4 = 607744 and
allocs/op by ~1. Opt-in diagnostic, not a gate.
```

## TestSampledGumbelStreamIdentity

Moved from `gpu/sampled_gumbel_identity_test.go` (the comment above `TestSampledGumbelStreamIdentity`) on 2026-10-09.

```text
TestSampledGumbelStreamIdentity is the WebGPU end-to-end gate for the device Gumbel-max sampler (R7b): with a
fixed seed, the tokens drawn on-device are IDENTICAL to the host's (GOINFER_NO_SAMPLE_FASTPATH=1) on real
checkpoints. Same PRE-REGISTERED rule as the CUDA test: any divergence fails and is investigated, and the test
fails if the device path never engaged (DeviceSampled == 0), so it cannot pass vacuously. WGSL's log/mulhi are
weaker than CUDA's, so a rounding-level divergence is somewhat likelier here (~1e-5 per token at worst); at 2,000
tokens that is still a low-percent event, so a failure is more likely a bug than rounding.
```

## softcapSerial

Moved from `gpu/softcap_test.go` (the comment above `softcapSerial`) on 2026-10-09.

```text
softcapSerial is the reference: the exact loop that shipped at cuda/resident.go and
cuda/prefill.go before applySoftcap, and that still ships at decoder/forwardn.go,
decoder/model.go and metal/model.go. Kept verbatim so the gate compares against the thing the
other four siblings still do, not against a re-derivation of it. This is the WebGPU twin of
cuda/softcap_test.go — gpu/softcap.go is a byte-identical port of cuda/softcap.go's
applySoftcap (G6, docs/tasks/task-gpu-paths-2026-09.md), so this test is a direct port too.
```

## TestSpeculative_C03_concurrentResidentClaim

Moved from `gpu/spec_c03_concurrent_test.go` (the comment above `TestSpeculative_C03_concurrentResidentClaim`) on 2026-10-09.

```text
TestSpeculative_C03_concurrentResidentClaim is the gate for audit C-03: GenerateSpeculative must
claim the shared resident KV (resBusy CAS) before any device write, exactly as generateInto and
the n-gram path do. Before the fix it skipped the claim, so a second concurrent generation on the
same *Model — which the Model doc explicitly permits for distinct sequences — prefilled into the
SAME positional device KV, interleaving writes and silently corrupting both streams.

The bug is a missing mutual-exclusion claim, so it only manifests under real concurrency: this
runs a plain resident Generate and a GenerateSpeculative CONCURRENTLY on one *Model. Whichever
loses the CAS falls back to the staged CPU cache — separate state — so each stream is a VALID
greedy decode on EITHER the resident or the CPU backend (the two can differ by a token at a
near-tie: the documented resident-vs-CPU parity gap, NOT a bug). The test asserts each output
equals one of those two references; a C-03 corruption (interleaved resident KV) matches neither.
Deterministically passes on the fixed code regardless of which side wins the race.

Target = 1.5B (resident), draft = 0.5B (CPU), both dense Qwen2 and vocab-matched — the same models
as TestSpeculativeResident_parity. Heavy + webgpu gated.
```

## TestSpeculativeResident_decomp.ceiling

Moved from `gpu/spec_decomp_test.go` (the comment inside `TestSpeculativeResident_decomp`) on 2026-10-09.

```text
Spec ceiling: per round = K·draft + verify; ideal Stage-B verify ≈ 1 target
token. speedup ≈ (tokens/round) / (round_ms / target_ms). Use a representative
tokens/round ≈ 3.3 at K=4 (measured acceptance 0.58).
```

## TestSpeculativeResident_e2eThroughput

Moved from `gpu/spec_e2e_bench_test.go` (the comment above `TestSpeculativeResident_e2eThroughput`) on 2026-10-09.

```text
TestSpeculativeResident_e2eThroughput is the Lever 2 KILL-GATE measurement: real
end-to-end tokens/sec for resident GPU speculative decoding (1.5B target on webgpu,
0.5B CPU draft) vs plain resident greedy on the SAME target, on code prompts where
acceptance is high. It also re-asserts the parity invariant (spec output ==
greedy output) so a throughput regression can't hide behind a correctness break.

This measures Stage A (the batched ForwardN amortizes only the (K-1) Submit/Poll
syncs, not the K×~420 dispatch records — see gpu-next-levers-assessment.md §7), so a
speedup BELOW ~1.3× here is the expected, documented signal that Stage B (the true
M=K GEMM verify) is what unlocks the predicted 1.4–2.0×. The number is logged, not
asserted as a floor (Stage A is not where the win lives); only parity is hard-gated.
```

## TestSpeculativeResident_e2eThroughput.draft

Moved from `gpu/spec_e2e_bench_test.go` (the comment inside `TestSpeculativeResident_e2eThroughput`) on 2026-10-09.

```text
Draft is ALSO GPU-resident: a CPU draft is slower per token than the GPU target
(measured), which makes speculation a net loss regardless of verify cost. Two
resident models = two Contexts/devices sharing the 8 GB pool.
```

## TestSSMKernelControlD3

Moved from `gpu/ssm_kernel_control_test.go` (the comment above `TestSSMKernelControlD3`) on 2026-10-09.

```text
D3 — the last localization control (docs/ssm-int8-quality.md). The CPU experiments
(decoder/ssm_precision_localize_test.go) showed every precision/quant change lands at
95-100% vs the f32 reference: E1 f32-SSM=100%, D1 int8-mamba=97%, R2 int8-attn/MoE=95%.
None reproduce the resident's 66%. D3 stacks ALL of them on the staged path — int8
mamba (ssmQ8CPU) + int8 attn/MoE (staged webgpu) + (f32 SSM is equivalent to f64 per
E1, so left f64) — with the mamba COMPUTE on the CPU (mamba2Step), not the GPU kernels.
If D3 ≈ 95% while the int8 GPU-resident is 66%, the only remaining variable is the GPU
mamba kernels (conv/ssm/gatedNorm) — i.e. the gap is a kernel discrepancy, not precision.
```

## TestStagedGPU_nonDense

Moved from `gpu/staged_bench_test.go` (the comment above `TestStagedGPU_nonDense`) on 2026-10-09.

```text
TestStagedGPU_nonDense is Lever A / B14 (docs/task-benchmark-refresh.md): the honest
staged-path GPU decode number for residency-INELIGIBLE families (MoE / Mamba-2 hybrid
/ MLA). These never enter the resident DecodeRunner, but under -tags gpu their int8
(W8A8) matmuls still dispatch to the GPU backend (decoder/weightmat.go matmul →
QuantBackend.MatmulW8A8), CPU glue between — so a real "staged GPU" tok/s exists today
with no code. It is glue/dispatch-bound, NOT the resident megakernel path; report it
in its own table, NEVER as the GPU headline (which is dense-only residency).

int8 ONLY: the int4 (W4A8) matmul branch dispatches straight to linalg.MatmulBTW4A8
with no backend routing, so -tags gpu does nothing for an int4 model (that gap is
Lever B). Loads here are Quant "int8int8" so the matmuls actually reach the GPU.
```

## TestStagedLayerSweep.hook

Moved from `gpu/staged_layersweep_test.go` (the comment inside `TestStagedLayerSweep`) on 2026-10-09.

```text
Both loops run the DECODER's granite forward; the truncation goes through the test hook (this
used to os.Setenv GOINFER_SSM_STOP_LAYER, which the decoder read once at init, so both sides ran
every layer and each row compared two full-model forwards).
```

## TestWebGPUThetaAB

Moved from `gpu/theta_ab_test.go` (the comment above `TestWebGPUThetaAB`) on 2026-10-09.

```text
TestWebGPUThetaAB — does wiring the MEASURED Theta actually make WebGPU speculative decode
faster (or at least no worse)?

Theta was reachable only as 0.5 on this backend (P22, docs/queue-performance.md): WebGPU's
residentDecoder implemented neither VerifyPathReporter nor PrefillPathReporter, so
decoder.verifyTheta() fell through to the unmeasured default. Theta actually measures
0.978-1.028 (gpu/theta_probe_test.go) — ForwardN's single-submit structure does not make the
marginal row cheap, it just removes Go-side dispatch overhead between rows. Under 0.5 the
controller drafts several nodes deep; under the measured value it should decline almost
entirely, because a WebGPU verify node costs very close to a full target step.

That predicts the speculative path under Theta=0.5 is SLOWER than not speculating at all on
this backend, and that VerifyPath's fix recovers it by declining. This measures that rather
than asserting it. Three arms, one prompt, interleaved (mirrors metal/theta_ab_test.go exactly,
same reasoning, same backend-agnostic decoder API):
```

## TestThetaProbe_WebGPU

Moved from `gpu/theta_probe_test.go` (the comment above `TestThetaProbe_WebGPU`) on 2026-10-09.

```text
TestThetaProbe_WebGPU measures Theta — the marginal cost of one extra verify node, in units of
one single-token target step — on the cgo WebGPU resident path.

WHY THIS EXISTS. decoder/spec_adaptive.go says Theta "is the relative cost of one extra verify
node on *this backend* — measure it", ships 0.5 as the batched-CPU value, and CPU/CUDA/Metal
have each had a real probe since (decoder/theta_probe_test.go, cuda/theta_probe_test.go,
metal/theta_probe_test.go — docs/measurements/theta-per-backend-2026-09-01.md,
theta-cuda-ab-2026-09-01.md) — P22 (docs/queue-performance.md) named WebGPU as the one backend
still falling through to the unmeasured 0.5 default, filed open rather than assumed either way.
The Metal probe found the domain check itself excluded Metal's real value (1.00–1.05) and
silently substituted 0.5 — the WORST available choice, since a smaller Theta drafts deeper, not
shallower (spec_adaptive.go's `Depth()` is monotone-decreasing in Theta). Whether WebGPU has the
same shape of defect is exactly what this measures.

METHOD, identical to the CPU control and the CUDA/Metal probes so all four numbers are directly
comparable: seed a context of `depth` positions, then time ForwardN over n tokens for a ladder
of n, truncating back to `depth` between every call. Theta = (least-squares slope of T(n)) /
T(1). residentDecoder.TruncateTo is a documented no-op on this backend (gpu/residency.go: the
cache is positional, Forward sets nKeys=pos+1, so entries past pos are simply never read and get
overwritten next round) — safe for this probe specifically because every call passes the SAME
startPos, so nothing past `depth` is ever read regardless of what a prior wider call left there.
```

## TestUploadKV_matchesSequentialForward

Moved from `gpu/uploadkv_parity_test.go` (the comment above `TestUploadKV_matchesSequentialForward`) on 2026-10-09.

```text
TestUploadKV_matchesSequentialForward is UploadKV's FIRST real correctness test on the WebGPU
backend (P6b / gap-0, docs/multimodal.md) — the CUDA twin of this test lives in
cuda/uploadkv_parity_test.go; see that file's doc comment for the full rationale. Before this,
UploadKV had zero non-test call sites and every fake stub ignored its arguments and returned
nil.

Two arms on the SAME resident handle (Reset between them):
```

## TestUploadKV_matchesSequentialForward.model

Moved from `gpu/uploadkv_parity_test.go` (the comment inside the doc comment of `TestUploadKV_matchesSequentialForward`) on 2026-10-09.

```text
Two cases: base=0 (no sliding-window ring wrap) and base>0 (prompt long enough to wrap the
ring). Model: tinymistral-248m (Mistral, ALL layers sliding-window, sliding_window=32 per its
own config.json) rather than Gemma 3 — confirmed by running this suite for real: gemma3
declines WebGPU residency on this build ("arch needs unimplemented feature(s) [embed-scale
gated-gelu sandwich-norm]", a pre-existing WebGPU feature gap, unrelated to this change), so it
cannot exercise this test at all on WebGPU. tinymistral's tiny window (32, vs gemma3's 1024)
also makes the ring-wrap case cheap to trigger — no need for a >1000-token prefill. The WebGPU
UploadKV implementation additionally has real per-precision byte-offset arithmetic (f32/f16/
int8, and int8's separate scale buffers) that the base=0 case alone cannot exercise, since an
all-zero offset is degenerate for every precision.
```

## TestWebGPUKVSlots_realClamp

Moved from `gpu/kv_slots_test.go` (the comment above `TestWebGPUKVSlots_realClamp`) on 2026-10-09.

```text
TestWebGPUKVSlots_realClamp is item 26 of docs/prompts/nobara-mc1-webgpu-2026-09.md, §2: the discrete-GPU clamp
(a slot whose allocation fails, or whose headroom probe after it fails, stops buildKVSlots) has only run under an
injected failure (TestWebGPUKVSlots_clampedBuild) — nobody has seen what wgpu-native on Vulkan does when a real
buffer allocation runs out of VRAM. This is that measurement, on real hardware and a real checkpoint, not a
synthetic one: load with 4 slots requested, run every granted slot, and check the device comes back clean.

It records what happened rather than asserting an expected count (the brief's own instruction) — but DOES fail
hard on any device-lost or validation error, on a slot that isn't bit-identical to a fresh one-slot load, or on
memory that does not return to baseline. A skip here is not a pass; only a missing checkpoint file skips.
```

## TestWebGPUKVSlots_realClamp.cases

Moved from `gpu/kv_slots_test.go` (the comment inside the case table of `TestWebGPUKVSlots_realClamp`) on 2026-10-09.

```text
~4.4 GB of int4 weights + ~1.88 GB/slot at the default 16k f32 KV on an 8 GB card: the brief
expects 1 slot, with slot 2 failing — CLAUDE.md's own model-storage rule applies (~/models
NVMe only; a path under /srv/models or /Volumes/ would be measuring a 5400 rpm SMR disk).
```

## TestWebGPUKVSlots_realClamp.small

Moved from `gpu/kv_slots_test.go` (the comment inside the case table of `TestWebGPUKVSlots_realClamp`) on 2026-10-09.

```text
~0.94 GB/slot: the brief expects all 4 to fit.
```

## TestMellum2_decodeThroughput

Moved from `gpu/mellum2_decode_bench_test.go` (the comment above `TestMellum2_decodeThroughput`) on 2026-10-09.

```text
TestMellum2_decodeThroughput measures decode tok/s of the real Mellum2 (12B-class,
64-expert top-8 MoE, 3:1 sliding/full attention, sliding window 1024) on the webgpu
backend, and reports whether it runs RESIDENT (full-residency DecodeRunner) or STAGED
(per-matmul backend) + the path reason.

On an 8 GB card Mellum2 stages: int8 MoE experts (the only kind the resident builder
stacks) make the model ~12 GB > VRAM, while int4 fits VRAM but the resident MoE
stacking is int8-only (gpu/residency.go). So this measures the STAGED number here;
set GOINFER_MELLUM_QUANT + a bigger card to get the resident one.
```

## attn_headdim_guard_test.L20

Moved from `gpu/attn_headdim_guard_test.go` (the comment at old line 20, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
The invariant is no longer "== the workgroup width". The wide kernel strides, so its reach
is width × dims-per-lane, and attnMaxHeadDim must track THAT product — if a future change
shrinks attnMaxPerLane without shrinking the limit, the predicate would admit a head_dim
the kernel's per-lane array cannot hold.
```

## attnbatched_test.L12

Moved from `gpu/attnbatched_test.go` (the comment at old line 12, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
TestAttnBatched_parity gates docs/completed/task-gpu-batched-prefill.md's Increment 1: does ONE
grid-(nH,M) dispatch of attnBatchedKernel's chosen kernel match M separate
dispatches of attnKernel's chosen kernel (the shape PrefillLastW8A8 used before
this fix, and what decode still uses today) over the SAME fully-pre-populated
K/V cache? No checkpoint needed — synthetic Q/K/V, a fresh cache (basePos=0), M
rows each attending to keys [0, row].

Two geometries, one per attnKernel/attnBatchedKernel branch (attention.go's
attnKeysEligible): hd=64/kvDim=128 (both %4==0, both KEYS-eligible — the common
dense-architecture shape, e.g. qwen2.5-coder-0.5b) and hd=48/kvDim=48 (kvDim not
a multiple of 4 via nKV=1 group — falls to the plain kernel).

NOT bit-exact, unlike TestRMSNormBatched_parity/TestRoPEBatched_parity: attention
kernels in this file are never bit-identical to each other even for the SAME
math (attnKeysShaderWGSL's own comment: "the denominator sums in a different
```

## bufaccount_test.L51

Moved from `gpu/bufaccount_test.go` (the comment at old line 51, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
Same shape TestVisionLayerNorm_parity already exercises — a smaller, ad hoc size here
once triggered a SIGTRAP inside the wgpu-native driver on CreateBuffer, unrelated to
the accounting this test is actually about; reusing a proven-safe size avoids
introducing a second, unrelated flake into a test meant to pin V-22.
```

## decode_instrument_test.L78

Moved from `gpu/decode_instrument_test.go` (the comment at old line 78, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
---- (a) resident gate GEMV [inter=8960, hidden=1536], ~13.76 MB int8 ----
```

## decode_instrument_test.L195

Moved from `gpu/decode_instrument_test.go` (the comment at old line 195, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
gemv dispatches stream weights of many sizes; scale the measured 13.76 MB
gemv cost by total weight bytes so size mix is accounted for, not dispatch
count alone. glue dispatches touch only a few KB → ≈ the no-op floor.
```

## decode_realmodel_bench_test.L21

Moved from `gpu/decode_realmodel_bench_test.go` (the comment at old line 21, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
Default model is the Qwen2.5-coder-1.5B GGUF (has qkv bias → the target for the
Increment-2 bias→GEMV-epilogue fusion). Override with GOINFER_DECODE_GGUF.
```

## TestDecodeRunnerW4A8_7B_fit.prior

Moved from `gpu/decoderunner_w4a8_bench_test.go` (the comment at old line 12, reworded in place; the comment above `TestDecodeRunnerW4A8_7B_fit` keeps the contract) on 2026-10-09.

```text
TestDecodeRunnerW4A8_7B_fit is the W4A8 footprint gate: it builds the FULL
Qwen2.5-7B shape as a resident int4 model on the GPU with a 16k-context f32 KV
cache (F2), confirms it FITS the 8 GB card (no OOM, resident bytes logged vs
budget), and measures decode throughput. The footprint is shape-determined
(weight values don't affect resident bytes or bandwidth), so synthetic int4
weights of the exact 7B shape give the real fit + tok/s — the same way the
1.5B int8 throughput bench works; bit-exact correctness is gated separately on
the 1.5B (TestDecodeRunnerW4A8_parity). The claim under test: a model that does
NOT fit at int8 (7.07 GB weights) runs at int4.
```

## TestDecodeTokenFusedBatched_parity.prior

Moved from `gpu/decodetoken_batched_test.go` (the comment at old line 11, reworded in place; the comment above `TestDecodeTokenFusedBatched_parity` keeps the contract) on 2026-10-09.

```text
TestDecodeTokenFusedBatched_parity is the Stage-B (docs/spec/07) increment-1 gate:
the batched M=K verify forward (projections as one tiled GEMM, attention per-row)
must be BIT-IDENTICAL to M sequential DecodeTokenFused calls at consecutive
positions sharing one KV cache — the per-row path it will replace. Same int8
inputs + int32 accumulation ⇒ exact equality, including each row attending to the
earlier rows of the block via the shared cache.
```

## decodetoken_batched_test.L93

Moved from `gpu/decodetoken_batched_test.go` (the comment at old line 93, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
Release at TEST scope, not inside buildMW. A `defer mw.Release()` in the builder fires
when the BUILDER returns — and since `return mw` copies to the return slot before defers
run, the caller receives a struct whose buffers are all already closed, then nil-derefs
on the first use. Resident weights are caller-owned; Context.Close does not free them.
```

## TestDecodeTokenFusedBatched_microbench.prior

Moved from `gpu/decodetoken_batched_test.go` (the comment at old line 129, reworded in place; the comment above `TestDecodeTokenFusedBatched_microbench` keeps the contract) on 2026-10-09.

```text
TestDecodeTokenFusedBatched_microbench is the Stage-B increment-2 go/no-go: does
the batched M=K verify forward (one tiled GEMM per projection, weights streamed
once) actually beat M sequential per-row forwards (the runBatch model, weights
re-streamed per row) on the GPU? Realistic qwen2.5-coder-0.5b dims. Logs the
per-block wall and the speedup; a ratio ≤ ~1 means the thin-M tile wastes the win
and production wiring isn't worth it (the docs/spec/07 kill gate). Run -v.
```

## TestDecodeTokenFusedBatched_Mbound_C13.prior

Moved from `gpu/decodetoken_bound_test.go` (the comment at old line 10, reworded in place; the comment above `TestDecodeTokenFusedBatched_Mbound_C13` keeps the contract) on 2026-10-09.

```text
TestDecodeTokenFusedBatched_Mbound_C13 gates C-13: the thin-M gemmRow kernel accumulates into a
private array<i32, gemmRowMaxM>, so a block of M > gemmRowMaxM rows would silently alias the last
accumulator under WGSL robustness clamping — wrong logits, no error. The entry point must reject
it. The bound check runs before any device call, so no adapter is needed.
```

## decodetoken_largedim_bench_test.L31

Moved from `gpu/decodetoken_largedim_bench_test.go` (the comment at old line 31, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
vocab fixed small (32k) so the LM head doesn't dominate/OOM — we are measuring the
PROJECTION amortization (q/k/v/o + gate/up/down), which is where a big model's
decode bandwidth goes. L kept small at the biggest dims to fit 8 GB.
```

## decodetoken_largedim_bench_test.L52

Moved from `gpu/decodetoken_largedim_bench_test.go` (the comment at old line 52, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
gemmRowMaxM is a HARD kernel limit, not a tuning knob: the thin-M gemmRow kernel
accumulates into a private array<i32, gemmRowMaxM> (gpu/gemm_rows.go, gemmRowMaxM), and
DecodeTokenFusedBatched returns a clean error above it. This sweep used to request
M=32/48 anyway and t.Fatalf on that error, so the whole test reported FAIL for asking
the API to do something it documents it cannot — charging every unrelated leg an
investigation. (Proven unrelated to the MoE cap work in 0018114 by stashing that
commit's only gpu/ change and re-running: identical failure.) The measurable part of
the sweep is M <= gemmRowMaxM; beyond it the answer is "chunk the block", which is a
Stage-B design question, not a measurement.
```

## decodetoken_test.L155

Moved from `gpu/decodetoken_test.go` (the comment at old line 155, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
P1 (own-forward residency bridge): both layers share one attention geometry, so the
value-keyed geomFor must dedup them to a single attnGeom. A regression that allocated
one uniform per layer would leave these logits byte-identical while inflating this to
the layer count — the assertion catches what the parity check cannot see.
```

## deltanet_test.L14

Moved from `gpu/deltanet_test.go` (the comment at old line 14, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
TestDeltaRule_cpuParity — the whole Gated-DeltaNet mixer chain on the GPU vs the CPU reference,
at REAL head geometry, driven for enough tokens that a drifting state shows.

Four kernels, run CHAINED (each consumes the previous one's real GPU output, not the CPU's), so
this gates the composition the resident runner will execute rather than four isolated
primitives — the A′ zero-copy post-mortem's lesson, recorded as "isolation proves the primitive,
```

## deltanet_test.L24

Moved from `gpu/deltanet_test.go` (the comment at old line 24, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
The two GEMVs and the causal conv are deliberately outside: they are ordinary matmuls and
mambaConv, both already gated.

WHY REAL GEOMETRY AND NOT THE TINY GOLDEN. testdata/qwen35_deltanet_golden.json pins a real HF
layer, but at hk=hv=8, nv=4: 32 threads, and a state row shorter than a cache line. Qwen3.8 runs
hk=hv=128, nk=16, nv=48 — 6144 threads each owning a 128-float row. Those are the numbers the
kernel has to be right at, and the scaled-fixture discipline exists here
(TestGemma4MoEScaled_residentParity) precisely because toy widths hide this class of bug.

WHY COMPARE TO THE CPU AND NOT TO HF. The CPU recurrence is already gated against transformers'
torch_recurrent_gated_delta_rule (decoder's DeltaNet golden), so this makes the chain
kernel ≡ CPU ≡ HF. A reference written inside this package would be a second unvalidated
implementation of the thing under test.
```

## deltanet_test.L178

Moved from `gpu/deltanet_test.go` (the comment at old line 178, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
Worst-over-steps per stage, so a failure names WHICH kernel drifted rather than only that
the chain did. The stages run chained (deltaRule consumes deltaGates' and deltaNorm's real
GPU output, not the CPU's), so an error in an early kernel reaches the late ones — which is
the composition the resident runner will actually execute.
```

## f16_convert_test.L10

Moved from `gpu/f16_convert_test.go` (the comment at old line 10, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
canonF32ToF16 is a local copy of decoder.f32ToF16bits (unexported, another module) — THE
```

## canonF32ToF16.prior

Moved from `gpu/f16_convert_test.go` (the comment at old line 13, reworded in place; the comment above `canonF32ToF16` keeps the contract) on 2026-10-09.

```text
cuda/f16_convert_test.go's copy, which gates the identical property for C-15.
```

## TestTiledDP4A_microbench.prior

Moved from `gpu/gemm_dp4a_test.go` (the comment at old line 193, reworded in place; the comment above `TestTiledDP4A_microbench` keeps the contract) on 2026-10-09.

```text
TestTiledDP4A_microbench compares the DP4A kernel's throughput against the
scalar-unpack kernel and the M=1 GEMV, at realistic prefill M — this is the
number that proves or disproves docs/completed/task-gpu-batched-prefill.md's whole premise
(the tiled GEMM must clear the bandwidth-bound M=1 GEMV to make batching worth
it at all). Skips (not fails) when this adapter doesn't accept dot4I8Packed —
informational, not a correctness gate.
```

## gemm_dp4a_test.L219

Moved from `gpu/gemm_dp4a_test.go` (the comment at old line 219, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
sequentialGEMV runs M *real* single-token GEMV dispatches (MatmulW8A8GEMV,
the actual decode-path kernel) — this, not the naive M-row matmul, is what
today's option-(a) O(prompt-len) prefill loop actually costs per layer
projection. That's the baseline the tiled GEMM must beat for batching to
pay off at all (docs/completed/task-gpu-batched-prefill.md).
```

## geom_variant_test.L10

Moved from `gpu/geom_variant_test.go` (the comment at old line 10, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
TestGeomVariants_dedup is the P1 (own-forward residency bridge) unit gate for the
value-keyed attention-geometry dedup. TestDecodeToken_parity already proves the
*collapse* direction (a uniform multi-layer model dedups to one attnGeom, so
non-Gemma models stay byte-identical). This proves the *expand* direction and that
the key actually depends on every field: two layers whose {hd, nKV, half} tuples
differ in ANY one component must produce two distinct geoms, not one. If the key
silently dropped a field, the matching case below would collapse to a single variant.

It asserts GeomVariantCount only — the plan is built, never run — so the second
layer's tuple may diverge from its (uniform) weight shapes without consequence; the
```

## gptoss_down_w4_test.L11

Moved from `gpu/gptoss_down_w4_test.go` (the comment at old line 11, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
TestGptOssDownW4_parity gates gpt-oss's down-projection combine on an INT4 stacked expert set.
That is the layout Quant "int4" uploads, and the only one that fits gpt-oss-20b on an 8 GB card
(audit-2026-09-10 C-06). It runs through gptOssDownPipelineFor, the resident builder's own kernel
choice, and checks it against the CPU reference: dst + wgt·(W4A8 matmul + down bias).

The matmul term is made to dominate the bias ON PURPOSE. G-07 is the record of a bias-dominated
```

## benchGPU.prior

Moved from `gpu/gpu_test.go` (the comment at old line 201, reworded in place; the comment above `benchGPU` keeps the contract) on 2026-10-09.

```text
BenchmarkMatmulBT_GPU measures the full offload (upload + dispatch +
readback) at forward-pass shapes. Compare the reported GFLOP/s
(MB/s column ÷ 1000, since SetBytes = 2*M*K*N FLOPs) against the
encoder package's BenchmarkMatmul{Serial,Parallel}_* CPU numbers.
EXPECT the GPU to trail the CPU here at these shapes — per-call
transfer dominates; this benchmark exists to quantify that gap and
track it as the resident-buffer follow-up lands.
```

## gumbel_test.L14

Moved from `gpu/gumbel_test.go` (the comment at old line 14, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
TestGumbelDeviceAgreesWithHost is the WebGPU kernel gate for the Gumbel-max sampler (R7b). Same PRE-REGISTERED
rule as cuda/gumbel_test.go, written before the first run:
```

## kv_i8_parity_test.L188

Moved from `gpu/kv_i8_parity_test.go` (the comment at old line 188, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
doc's "≥0.99" prediction was for a real Qwen-7B distribution; the kernels' bit-
exactness (TestKVI8Kernels/Attn, cosine 1.000000) is the exactness proof.
```

## kvprec_decline_test.L61

Moved from `gpu/kvprec_decline_test.go` (the comment at old line 61, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
And the -ctx request must be consulted. It was read nowhere under gpu/, so -ctx 32768
kept 16k and -ctx 2048 still allocated 16k per layer.
```

## leak_report_test.L21

Moved from `gpu/leak_report_test.go` (the comment at old line 21, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
The report is printed at the end of the run and any entry FAILS the package (the suite was made leak-free on 2026-10-08, so a new leak is a regression); GOINFER_GPU_LEAKS_FATAL=0 turns the failure off (a test-only switch,
not a production read).
```

## lora_resident_parity_test.L17

Moved from `gpu/lora_resident_parity_test.go` (the comment at old line 17, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
TestLoRAResidentParityWebGPU is the G3 numeric gate (docs/tasks/task-gpu-paths-2026-09.md): compute-
time LoRA applied on this backend's resident decode path must match the CPU reference (the
same adapter applied via decoder's generic gatedMLP/causalAttention forward). Mirrors
metal/lora_resident_parity_test.go — same fixture (testdata/llama-tiny, GQA 4-heads/2-kv-heads
so o-proj and the differently-widthed q/k/v sites are all genuinely exercised), same
vacuousness check, same reasoning for the 0.95 floor (gpt2_resident_parity_test.go's
established resident-vs-CPU decode-logit bar on the Metal side; this backend has no
resident-vs-CPU whole-model floor of its own narrower than that, so it is the right anchor
here too) — but drives decoder.ResidentAdapter directly via ResidentForwardForTest/
ResidentAdapterLayersForTest to isolate the KERNEL correctness question, same as every other
resident parity test in this file's package.

Architecturally this backend's SetAdapter (gpu/lora_resident.go) is a genuinely different
mechanism from Metal's — a flat, Go-side dispatch-step list rebuilt from a saved pristine plan
plus spliced-in LoRA steps, not a per-token re-encode with a plain `if` — so this test is the
```

## mla_real_resident_test.L70

Moved from `gpu/mla_real_resident_test.go` (the comment at old line 70, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
R-05 regression: the 128-wide head-dim guard rejected MLA's 192-wide qk head.
```

## mla_real_resident_test.L75

Moved from `gpu/mla_real_resident_test.go` (the comment at old line 75, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
guard — R-05 holds; the model just doesn't fit fully resident here (MLA can't stream).
```

## TestMLAResidency_matchesCPU.prior

Moved from `gpu/mla_resident_test.go` (the comment at old line 14, reworded in place; the comment above `TestMLAResidency_matchesCPU` keeps the contract) on 2026-10-09.

```text
TestMLAResidency_matchesCPU is the end-to-end bridge gate for Lever C4d: the tiny
DeepSeek-V3 checkpoint (q-LoRA + compressed-KV latent attention + group-limited
DeepSeekMoE + dense prefix + ungated shared expert) loaded on the webgpu backend must
go GPU-resident (proving decodeRunnerEligible + BuildResident accept MLA) and decode
greedily in agreement with the CPU f32 forward. The resident path runs int8, the CPU
path f32, so a tiny-random model's logits can eventually flip an argmax under int8
rounding — the gate is the FIRST generated token (single forward, minimal quant
accumulation), with the full sequences logged. Numerical correctness of the MLA kernels
themselves is pinned bit-identically by TestDecodeRunnerMLA_parity.
```

## TestDecodeRunnerMLA_parity.prior

Moved from `gpu/mla_test.go` (the comment at old line 35, reworded in place; the comment above `TestDecodeRunnerMLA_parity` keeps the contract) on 2026-10-09.

```text
TestDecodeRunnerMLA_parity gates Lever C4c: the full MLA latent-attention forward on
the resident runner, end-to-end against a CPU int8 oracle that mirrors the same absorb
math. The tiny model is DeepSeek-V3-shaped — q-LoRA bottleneck, compressed-KV latent
cache, decoupled interleaved RoPE, and a DeepSeekMoE FFN (sigmoid routing + selection
bias + group-limited top-k + ungated shared expert). Every int8 GEMV runs identical
math both sides (W8A8); the absorb/lift/attention are f32. Cosine must be ~1.0, proving
the latent store, W_UK absorb, qRope, rank-space attend, W_UV lift, group routing, and
the shared-expert combine all land correctly on top of the C3 MoE machinery.
```

## mla_test.L374

Moved from `gpu/mla_test.go` (the comment at old line 374, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
TestMLAHeadMatvec_parity gates Lever C4b's per-head block-diagonal matvec — the W_UK
```

## TestMLALatentStore_parity.prior

Moved from `gpu/mla_test.go` (the comment at old line 376, reworded in place; the comment above `TestMLALatentStore_parity` keeps the contract) on 2026-10-09.

```text
TestMLALatentStore_parity gates Lever C4b's latent append: kvA-norm the rank latent +
decoupled-RoPE the key, mirroring decoder.cache.AppendLatent's normalized/roped form
(cn ‖ krj). Both V3 GPT-J interleave and plain NeoX are covered against a CPU f64
reference (rmsNorm + mlaRope) at a nonzero position so the rope-at-pos path is real.
```

## TestMLAHeadMatvec_parity.prior

Moved from `gpu/mla_test.go` (the comment at old line 440, reworded in place; the comment above `TestMLAHeadMatvec_parity` keeps the contract) on 2026-10-09.

```text
TestMLAHeadMatvec_parity gates Lever C4b's per-head block-diagonal matvec — the W_UK
absorb (a=q with a wider stride than K, so the qk_rope tail is skipped) and the W_UV
lift (aStride==K). A CPU f64 reference over random per-head a/w must match (cosine ~1.0).
```

## TestMLAAttn_parity.prior

Moved from `gpu/mla_test.go` (the comment at old line 481, reworded in place; the comment above `TestMLAAttn_parity` keeps the contract) on 2026-10-09.

```text
TestMLAAttn_parity gates Lever C4a: the absorb-path MLA rank-space attention kernel.
It mirrors decoder.mlaAttentionAbsorb steps 4b+5 — score each cached latent by the
full latDim dot (qNopeAbs·cn + qRope·krj), per-head softmax, then collapse V to the
rank-space weighted latent sum wsum[h] = Σ_j p_j·cn_j. A CPU f64 reference over the
same random qAbs/latent must match the GPU online-softmax kernel (cosine ~1.0). Uses
DeepSeek-V3-ish dims (rank 512 > the 128-lane width) so the strided score/value paths
are exercised, plus a small-rank case so the rank ≤ WG path is covered too.
```

## TestMoEExpertGEMV_indexing.prior

Moved from `gpu/moe_expert_test.go` (the comment at old line 12, reworded in place; the comment above `TestMoEExpertGEMV_indexing` keeps the contract) on 2026-10-09.

```text
TestMoEExpertGEMV_indexing gates C3b: the indexed sparse-expert GEMV must read the
RIGHT expert's row out of the stacked buffer (the dynamic index comes from the routing
buffer, not a record-time constant). For each expert e it runs the indexed GEMV with
idx[0]=e and checks the result equals a direct int8 GEMV of expert e's weight — proving
the stacked addressing + W8A8 math. (The full gate→up→SwiGLU→down combine is gated
end-to-end on mixtral-tiny in C3c.)
```

## TestDecodeRunnerMoE_parity.prior

Moved from `gpu/moe_runner_test.go` (the comment at old line 12, reworded in place; the comment above `TestDecodeRunnerMoE_parity` keeps the contract) on 2026-10-09.

```text
TestDecodeRunnerMoE_parity gates Lever C3c: the integrated resident MoE forward
(router top-k → indexed gate/up GEMVs → SwiGLU → weighted down-combine, all on the
device in one command buffer) must match a CPU oracle running the SAME math. Both
sides share the int8 expert/router weights and the int8 activation quantization, so
this is a wiring/kernel gate (cosine ~1.0): it proves the on-GPU router selects the
right experts and the stacked indexed GEMVs combine them into the residual exactly
as moeMLP does. Mixtral-class shape: 8 experts, top-2, softmax + NormTopKProb, no
shared expert, every layer MoE.
```

## TestDecodeRunnerMoEShared_parity.prior

Moved from `gpu/moe_shared_runner_test.go` (the comment at old line 12, reworded in place; the comment above `TestDecodeRunnerMoEShared_parity` keeps the contract) on 2026-10-09.

```text
TestDecodeRunnerMoEShared_parity gates Lever C3d: the resident MoE forward with an
always-on shared expert. Two flavors run through the integrated runner — GLM-style
(sigmoid routing + selection bias, UNGATED shared expert) and qwen2_moe-style
(softmax routing, sigmoid-GATED shared expert). Each must match a CPU oracle running
the identical int8 math (cosine ~1.0): proves the shared-expert add (gated/ungated)
and the sigmoid+bias routing land correctly on top of the C3c routed-expert combine.
```

## TestMoERoute_parity.prior

Moved from `gpu/moe_test.go` (the comment at old line 69, reworded in place; the comment above `TestMoERoute_parity` keeps the contract) on 2026-10-09.

```text
TestMoERoute_parity gates the C3a GPU router top-k kernel against the reference for
the routing flavors the MoE families use (nGroup==1): Mixtral (softmax + renorm),
Qwen2-MoE (softmax, no renorm), and the DeepSeek/GLM sigmoid + selection-bias + scale
path (group-limit deferred to C3d). Same chosen experts (as a set; the kernel emits
them in descending selection order, matching the reference sort) and weights to f32 tol.
```

## moe_test.L113

Moved from `gpu/moe_test.go` (the comment at old line 113, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
TestMoERoute_pastOldCap is the gate for raising the router cap 256 -> 512 (MAXE in gpu/moe.go,
MOE_MAX_E in cuda/moe.cu). The cap bounds the ROUTER's per-invocation scratch and nothing else —
the expert GEMVs select by index arithmetic into row-stacked weights, so they are expert-count
agnostic. That makes "does moe_route still agree with the CPU reference at nE > 256" the whole
correctness question, and this is it.

384 is Kimi-K2's real routed-expert count, which is the shipped family the old cap declined.
```

## nemotron_moe_resident_test.L92

Moved from `gpu/nemotron_moe_resident_test.go` (the comment at old line 92, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
0.95 is the established resident-vs-CPU floor this session's other new-family gates use
(gpt2_resident_parity_test.go's own precedent), not a bar invented for this test.
```

## nemotron_moe_resident_test.L174

Moved from `gpu/nemotron_moe_resident_test.go` (the comment at old line 174, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
A forward pass must actually run without panicking on the nil shUp/shDown fields the
fixed residency.go now leaves unbuilt — the decode-time `if lw.shUp != nil` guard
(gpu/decoderunner.go) is what this exercises end to end, not just that Load succeeds.
```

## TestNemotronResidentParity.prior

Moved from `gpu/nemotron_resident_parity_test.go` (the comment at old line 13, reworded in place; the comment above `TestNemotronResidentParity` keeps the contract) on 2026-10-09.

```text
P3/P4 whole-model parity: resident Nemotron-H (single-op-per-block Mamba-2 / NoPE-GQA / relu²
MLP) vs the CPU runLayersNemotron, feeding the SAME fixed token sequence to both and comparing
logits at 1/16/256/1k/2k. State (conv ring + ssm + KV) COMPOUNDS, so a wiring error — a
block-kind misroute, NoPE slip, single-op pairing bug, or pos off-by-one — shows as a cosine
that DRIFTS long. Matched precision (resident int8 vs CPU int8 via GOINFER_SSM_CPUQ8) isolates
WIRING from int8 quality: the engine should track ~flat near the int8-activation floor (the
resident quantizes the relu²/conv/proj activations the CPU keeps f32). Opt-in (tiny fixture).
```

## TestDecodeRunnerPerLayerRoPE_parity.prior

Moved from `gpu/perlayer_rope_runner_test.go` (the comment at old line 12, reworded in place; the comment above `TestDecodeRunnerPerLayerRoPE_parity` keeps the contract) on 2026-10-09.

```text
TestDecodeRunnerPerLayerRoPE_parity gates Lever C7: the resident runner binding a
DIFFERENT RoPE table + cos/sin scale (mscale) per layer — Mellum's YaRN-on-global vs
default-local interleave. Two dense layers use distinct (invFreq, ropeScale) pairs; a
CPU int8 oracle ropes each layer with its own table+scale. A bug that shared one rope
across layers (the pre-C7 behavior) would diverge here. Cosine must be ~1.0.
```

## prefilllast_ttft_test.L89

Moved from `gpu/prefilllast_ttft_test.go` (the comment at old line 89, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
SEQUENTIAL — today's shipped default (decoder/model.go's residentPrefillSeed
per-token loop): one Forward call per prompt token, from an empty cache.
```

## TestPrefillLastW8A8_parity.prior

Moved from `gpu/prefillrunner_test.go` (the comment at old line 12, reworded in place; the comment above `TestPrefillLastW8A8_parity` keeps the contract) on 2026-10-09.

```text
TestPrefillLastW8A8_parity is the Increment-2 gate (docs/task-gpu-batched-
prefill.md): PrefillLastW8A8's single returned row (the last position's logits)
must be BIT-IDENTICAL to the last of M sequential DecodeToken calls over one
shared KV cache. M=20 deliberately exceeds gemmRowMaxM=16 — the whole point of
this function over DecodeTokenFusedBatched is that it is NOT capped, since real
prompts run far past 16 tokens.
```

## qknorm_parity_test.L29

Moved from `gpu/qknorm_parity_test.go` (the comment at old line 29, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
(decoder.TestQwen3_forwardParity); here it's the resident path.
```

## TestNemotronRelu2FFN_parity.prior

Moved from `gpu/relu2_test.go` (the comment at old line 14, reworded in place; the comment above `TestNemotronRelu2FFN_parity` keeps the contract) on 2026-10-09.

```text
P2 parity: the resident non-gated squared-ReLU FFN — up(W8A8) → relu²→int8 (relu2Quant) →
down(W8A8) — vs the CPU reference down(relu²(up(x))). int8 weights, real-ish f32 activations
over many tokens (state-free FFN, so per-token), like the mamba-kernel isolation parities.
```

## TestResidentDecoderCheckCap.prior

Moved from `gpu/residency_cap_test.go` (the comment at old line 10, reworded in place; the comment above `TestResidentDecoderCheckCap` keeps the contract) on 2026-10-09.

```text
TestResidentDecoderCheckCap gates M20/C3: writes at/past the resident KV cap are refused
(a real Forward there is a silent out-of-bounds device write, WGSL-clamped → garbage reads).
Pure logic — no device.
```

## TestRopeStorePartialRotary_tailStored.prior

Moved from `gpu/rope_partial_test.go` (the comment at old line 12, reworded in place; the comment above `TestRopeStorePartialRotary_tailStored` keeps the contract) on 2026-10-09.

```text
TestRopeStorePartialRotary_tailStored gates C4: for partial rotary (rotaryDim < headDim — GLM,
some Phi), the K-store kernels must store the un-rotated pass-through tail [2*half, headDim), not
just the rotated span. Dropping it makes attention read zeros for those key dims at every cached
position — plausible-looking, silently wrong logits. Compares the resident f32 ropeStore + the
fused qkvFinalize against a CPU reference that rotates [0,2*half) and passes the tail through.
```

## rope_partial_test.L53

Moved from `gpu/rope_partial_test.go` (the comment at old line 53, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
assertTail fails if any tail dim was dropped (the C4 bug wrote zeros there).
```

## TestDecodeRunnerSlidingWindow_parity.prior

Moved from `gpu/sliding_runner_test.go` (the comment at old line 12, reworded in place; the comment above `TestDecodeRunnerSlidingWindow_parity` keeps the contract) on 2026-10-09.

```text
TestDecodeRunnerSlidingWindow_parity gates Lever C6: the resident dense forward with
sliding-window (local) attention. All layers are local with a window W small enough to
BITE at the test position (start = max(0, pos+1-W) > 0), so the runner must attend only
the last W cached keys — not the full history. A CPU int8 oracle that windows the same
way must match (cosine ~1.0). Mirrors Mistral's all-local attention; full-attention
layers (isLocal=false) keep the unwindowed start and are covered by the other runner
gates. The window biting is the point: at W≥nKeys it would be a no-op.
```

## spec_c03_concurrent_test.L90

Moved from `gpu/spec_c03_concurrent_test.go` (the comment at old line 90, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
CPU-path greedy references. The CAS loser falls back to the STAGED CPU path (model.go),
whose greedy output can differ from the resident's at a near-tied token (documented
resident-vs-CPU parity gap) — so a contended stream may legitimately land on EITHER sequence.
A C-03 corruption (interleaved resident KV) matches NEITHER. We assert membership in the two
valid backends; that distinguishes a backend switch (fine) from corruption (the bug).
```

## theta_ab_test.L36

Moved from `gpu/theta_ab_test.go` (the comment at old line 36, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
	theta=0.5  the shipped-until-now behaviour, forced explicitly
```

## TestWebGPUThetaAB.prior

Moved from `gpu/theta_ab_test.go` (the comment at old line 40, reworded in place; the comment above `TestWebGPUThetaAB` keeps the contract) on 2026-10-09.

```text
be materially slower than `off` (it should be within noise of it, since it declines to draft),
and it must beat `theta=0.5`.
```

## theta_ab_test.L58

Moved from `gpu/theta_ab_test.go` (the comment at old line 58, reworded in place; the comment above `the code below` keeps the contract) on 2026-10-09.

```text
Quant int8int8, not int4: M-09 (decoder/spec_verify_guard.go) refuses ALL speculative
decoding on a staged webgpu-int4 model outright (M=1 decode and M>1 verify are two
different kernels with no measured tolerance) — a correctness guard this test must not
route around. int8int8 is not covered by that guard and still resides on WebGPU.
```

## TestKVCacheF16_fit

Moved from `gpu/kv_f16_fit_test.go` (the comment above `TestKVCacheF16_fit`) on 2026-10-09.

```text
TestKVCacheF16_fit is the Increment-2 measurement gate (task-gpu-f16-kv.md +
task-gpu-kv-i8.md): a 7B int4 model with a 64k int8 / 32k f16 KV cache must fit
8 GB (real allocation, no OOM), and lossy-KV decode should be no slower than f32
at equal context (the attention kernel is KV-read-bound → fewer bytes). Loading
the real model with KVPrecision="i8" also exercises the full residency int8 path
(NewKVCacheI8 + the on-device WRITE/READ kernels over real RoPE'd K/V), the real-
distribution complement to the synthetic TestKVCacheI8_parity. Measures peak VRAM
(nvidia-smi) and steady-state decode tok/s per precision. Asset-gated on a 7B
GGUF; skips if the model isn't residency-eligible in int4.
```
