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

## matmulShaderWGSL

Moved from `gpu/gpu.go` (the comment above `matmulShaderWGSL`) on 2026-10-10.

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

Moved from `gpu/gpu.go` (the comment above `Context.releases`) on 2026-10-10.

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

Moved from `gpu/gpu.go` (the comment above `Context.closed`) on 2026-10-10.

```text
closed makes Close IDEMPOTENT. `defer m.Close()` alongside an explicit m.Close() is the
ordinary Go shape, and decoder.Model.Close calls m.be.Close() unconditionally — so a second
Close used to double-release the wgpu handles, a use-after-free inside the native layer. The
cpu and cuda backends were already idempotent (cuda guards on r.reqCh == nil); only WebGPU
crashed, and only on a machine with a real GPU.
```

## Context.hasDP4A

Moved from `gpu/gpu.go` (the comment above `Context.hasDP4A`) on 2026-10-10.

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

Moved from `gpu/gpu.go` (the comment above `Context.attnBatchedShader`) on 2026-10-10.

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

Moved from `gpu/gpu.go` (the comment above `liveContexts`) on 2026-10-10.

```text
liveContexts counts Contexts that have been created and not yet Closed. A WebGPU device is
a scarce driver resource — measured on this box's NVIDIA/Vulkan stack, exactly 63 can be
LIVE at once, while create/destroy churn is free (200 cycles with no trouble). So a leaked
Context is not a slow drain, it is a hard cliff: past 63 every later New() fails with
"failed to request device", and in a test binary that silently converts gates into skips.
Exposed to tests through liveContexts.Load(); see TestDeviceExhaustion_repro.
```

## Context.mkPipeline

Moved from `gpu/gpu.go` (the comment above `Context.mkPipeline`) on 2026-10-10.

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

Moved from `gpu/gpu.go` (the comment above `finalizerSerial`) on 2026-10-10.

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

Moved from `gpu/attention.go` (the comment at the top of the file) on 2026-10-10.

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

Moved from `gpu/attention.go` (the comment above `attnWG`) on 2026-10-10.

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

Moved from `gpu/attention.go` (the comment above `attnShaderWGSL`) on 2026-10-10.

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

Moved from `gpu/attention.go` (the comment above `attnKeysShaderWGSL`) on 2026-10-10.

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

Moved from `gpu/attention.go` (the comment above `attnKernel`) on 2026-10-10.

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

Moved from `gpu/attention.go` (the comment inside `ensureAttn`) on 2026-10-10.

```text
Guard on the LAST pipeline built, not the first: these are created in order, so a non-nil last
field means every earlier one succeeded too. Guarding on ropePipeline (the first) let a
mid-build failure leave the guard satisfied with later pipelines nil, and the next call
dispatched a nil pipeline (audit R-30). On a retry after a partial build the earlier fields are
rebuilt (the old ones stay tracked for release at Close — bounded, not leaked).
```

## attnWideTemplateWGSL

Moved from `gpu/attention.go` (the comment above `attnWideTemplateWGSL`) on 2026-10-10.

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

Moved from `gpu/attention.go` (the comment inside `ensureAttnWide`) on 2026-10-10.

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

Moved from `gpu/backend.go` (the comment above `webgpuBackend`) on 2026-10-10.

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

Moved from `gpu/backend.go` (the comment above `webgpuBackend.MatmulW4A8`) on 2026-10-10.

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

Moved from `gpu/backend.go` (the comment above `webgpuBackend.MatmulW4A8Batch`) on 2026-10-10.

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

Moved from `gpu/bufaccount.go` (the comment above `liveBufferBytes`) on 2026-10-10.

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

Moved from `gpu/decodefuse.go` (the comment at the top of the file) on 2026-10-10.

```text
§2 fused decode kernels. The §5 finding: decode is glue-serialization-bound —
the per-token cost is a deep RAW dependency chain of ~535 small dispatches,
each forcing a barrier the GPU can't hide. The lever is critical-PATH length,
so every standalone glue op that can be folded into the kernel it borders
removes a link from the serialized spine. These kernels do that folding; each
is bit-exact with the unfused pair it replaces (same f32 math, same int8 pack).
```

## gegluQuant

Moved from `gpu/decodefuse.go` (the comment above `gegluQuantWGSL`) on 2026-10-10.

```text
gegluQuant is swigluQuant's GELU-tanh-gated twin — Gemma's FeatGatedGELU (G6,
docs/tasks/task-gpu-paths-2026-09.md). Same fused shape (product recomputed in the pack pass, no
inter-wide global-memory round-trip); only the activation differs. gelu_tanh's argument is
CLAMPED to ±15 before calling tanh — see gegluShaderWGSL's own comment (layer.go) for why:
unclamped, it overflows f32 before saturating, which cost Metal's own port a real cosine
regression (0.818→0.994 after the fix) the first time it shipped this exact math.
```

## DecodeTokenFusedBatched

Moved from `gpu/decodetoken_batched.go` (the comment above `DecodeTokenFusedBatched`) on 2026-10-10.

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

Moved from `gpu/deltanet.go` (the comment above `deltaRuleShaderWGSL`) on 2026-10-10.

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

Moved from `gpu/deltanet.go` (the comment above `deltaQSplitShaderWGSL`) on 2026-10-10.

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

Moved from `gpu/kv_slots.go` (the comment above `slotsBeforeContext`) on 2026-10-10.

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

Moved from `gpu/device.go` (the comment at the top of the file) on 2026-10-10.

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

Moved from `gpu/device.go` (the comment above `readbackRaw`) on 2026-10-10.

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

Moved from `gpu/gemm.go` (the comment above `matmulTiledW8A8BiasKernelWGSL`) on 2026-10-10.

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

Moved from `gpu/gemm_rb.go` (the comment at the top of the file) on 2026-10-10.

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

Moved from `gpu/gemv_w4a8.go` (the comment above `f32to16`) on 2026-10-10.

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

Moved from `gpu/layer.go` (the comment above `gegluShaderWGSL`) on 2026-10-10.

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

Moved from `gpu/lora_resident.go` (the comment inside `SetAdapter`) on 2026-10-10.

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

Moved from `gpu/testhooks.go` (the comment at the top of the file) on 2026-10-10.

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

Moved from `gpu/vision.go` (the comment at the top of the file) on 2026-10-10.

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

Moved from `gpu/vision_encoder.go` (the comment above `VisionEncoder`) on 2026-10-10.

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

Moved from `gpu/prefill_prof.go` (the comment above `prefillProf`) on 2026-10-10.

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

Moved from `gpu/f16.go` (the comment above `f32ToF16`) on 2026-10-10.

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

Moved from `gpu/doc.go` (the comment in the package doc) on 2026-10-10.

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

Moved from `gpu/modelw_release.go` (the comment at the top of the file) on 2026-10-10.

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

Moved from `gpu/decodefuse.go` (the comment above `swigluQuantWGSL`) on 2026-10-10.

```text
swigluQuant fuses SwiGLU → activation-quantize: silu(gate)·up is computed,
max-abs'd, and packed to int8 in ONE dispatch, so the inter-wide (8960, ~36 KB)
SwiGLU output never materializes in global memory or crosses a barrier — a
double win under the §5 "per-link drain scales with the bordering kernel's
data" model. The product is recomputed in the pack pass (cheap arithmetic)
rather than staged, so no inter-wide workgroup array is needed.
```

## DecodeTokenFused

Moved from `gpu/decodetoken_fused.go` (the comment above `DecodeTokenFused`) on 2026-10-10.

```text
copies between passes — then ONE Submit + ONE Poll. The per-op-submit DecodeToken
was correct but issued ~500 submits; this collapses them to one, which is what
the §0.5 probe measured as the win. Bit-exact-identical to DecodeToken
(TestDecodeTokenFused_parity).
```

## attnKeysEligible

Moved from `gpu/attention.go` (the comment above `attnKeysEligible` (it sat above `attnKeysDisabled`)) on 2026-10-10.

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

Moved from `gpu/moe.go` (the comment above `moeRouteWGSL`) on 2026-10-10.

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

Moved from `gpu/quant.go` (the comment above `matmulW8A8ShaderWGSL`) on 2026-10-10.

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

Moved from `gpu/attention.go` (the comment inside the `attnWG` constant block) on 2026-10-10.

```text
attnMaxHeadDim is therefore 2048, which no model is near. That is the point: head_dim
stopped being a reason to decline, rather than the wall moving up one notch to 256.
```

## New: storage-buffer binding limit

Moved from `gpu/gpu.go` (the comment inside `New`) on 2026-10-10.

```text
Raise the storage-buffer binding limit: the DEFAULT device caps it at 128 MB,
smaller than a real model's LM head / embedding at int8 (e.g. 152k vocab ×
hidden ≈ 233 MB). Start from the valid default limit set and bump only the two
size limits to the adapter's max (requiring the adapter's full limit set
verbatim fails — some advertised limits, e.g. maxBufferSize, aren't valid as
required limits). maxBufferSize must be ≥ the binding size.
```

## New: MaxBufferSize

Moved from `gpu/gpu.go` (the comment inside `New`) on 2026-10-10.

```text
Raise MaxBufferSize to the binding max (2 GB on this card) so large single
weights fit — a 7B's LM head is ~272 MB int4 / ~545 MB int8, past the 256 MB
WebGPU default. Set unconditionally: DefaultLimits() leaves MaxBufferSize at
the u64-max "unset" sentinel, so a `<` guard never fires and the device
silently keeps the 256 MB default (it must be a concrete value to take).
```

## W4A8 decode GEMV

Moved from `gpu/gemv_w4a8.go` (the comment above `w4a8GroupSize`) on 2026-10-10.

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

Moved from `gpu/gemv_w4a8.go` (the comment above `UploadW4A8Packed`) on 2026-10-10.

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

Moved from `gpu/gemm_rows.go` (the comment above `gemmRowMaxM`) on 2026-10-10.

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

Moved from `gpu/deltanet.go` (the comment above `deltaNormShaderWGSL`) on 2026-10-10.

```text
deltaNorm l2-normalizes the per-head q and k slices of the conv output, and applies the query
scale. One thread per KEY head: the work is nk·hk (2048 elements on the real 27.8B geometry),
far too small to be worth splitting further, and doing it here instead of inside deltaRule keeps
the recurrence from recomputing each norm hv times.

The conv output is [q(keyDim) ‖ k(keyDim) ‖ v(valueDim)] contiguous — the same packing the CPU
reference slices, so the layouts agree by construction rather than by comment.
```

## deltaGatesShaderWGSL

Moved from `gpu/deltanet.go` (the comment above `deltaGatesShaderWGSL`) on 2026-10-10.

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

Moved from `gpu/kv_slots.go` (the comment above `darwinKVSlots`) on 2026-10-10.

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
