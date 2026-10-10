# metal: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `metal`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## loadEmbedRow

Moved from `metal/model.go` (the comment above `loadEmbedRow`) on 2026-10-09.

```text
loadEmbedRow dequantizes token `id`'s embedding into the shared input buffer and applies the
arch's embedding scale — the complete "token id → layer-0 input" step.

AUDIT G-02. This exists so the two id-taking entry points cannot disagree with production. The
scale (Gemma's √hidden, FeatEmbedScale) was applied ONLY by decoder.embedResident, on the
ForwardEmb path; Forward and ForwardArgmax took a raw dequantized row straight into layer 0.
Metal declares FeatEmbedScale: true and admits gemma3/gemma4, so both methods returned wrong
logits for those families for any direct caller — and the snapshot golden, which drives exactly
these two methods, pinned that wrong computation as its stored reference. A regression confined
to the embed→layer-0 seam was therefore invisible to the one absolute gate in the Metal suite.

The scale is a no-op (≤1) for every non-Gemma family, so this is inert on the dense archs.
```

## resident.ForwardEmb

Moved from `metal/model.go` (the comment above `ForwardEmb`) on 2026-10-09.

```text
ForwardEmb is Forward given a precomputed embedding[H] (the decoder.ResidentForward shape)
instead of a token id — it copies the embedding into the shared input buffer and skips the
internal embedding lookup. This is the PRODUCTION path: decoder.embedResident does the lookup
and applies any embed scale before calling in.

Numerically identical to Forward(id,pos) when emb is the SCALED embedding row. The previous
wording — "when emb = Embed.Row(id) (the eligible dense archs have no embed scale)" — was the
premise audit G-02 falsified: it stopped being true once Metal declared FeatEmbedScale and
admitted gemma3/gemma4, and Forward carried on ignoring the scale. Forward now applies it
(loadEmbedRow), so the two are equivalent again on every admitted family.
```

## softcapParallel

Moved from `metal/model.go` (the comment above `softcapParallel`) on 2026-10-09.

```text
softcapParallel applies Gemma's final-logit softcap sc·tanh(x/sc) in place, across cores. Every
element is independent and math.Tanh is deterministic, so splitting the loop is BYTE-IDENTICAL to
the serial form (same per-element sc*float32(math.Tanh(float64(v/sc))); disjoint writes, no
reduction/ordering) — gated by TestMetalSoftcapParallel_bitIdentical.

N-09 (audit-metal-2026-09-12.md): this does NOT skip on greedy decode in production, despite
argmax being softcap-invariant (the softcap is monotonic). finalizeLogits calls this whenever
r.finalSoftcap > 0 and runs on EVERY execLoop token (model.go's own call site,
unconditional on sampling mode) — the production decode path. The only site that genuinely
skips it is ForwardArgmax's on-device fused argmax dispatch (which never reads r.logitsHost at
all), and production greedy does not call that (N-03: ResidentGreedy is absent on Metal, so
greedy runs the same full-logits ForwardEmbPipe as sampling and argmaxes host-side). So a Gemma
family pays this 262k-wide tanh loop on every decode token regardless of temperature. At
Gemma's 256k vocab the serial float64 tanh loop is a real per-token tax this fans out over
GOMAXPROCS. Serial below the goroutine-spawn threshold (mirrors parallelF32ToF16).
```

## resident.execLoop

Moved from `metal/model.go` (the comment above `execLoop`) on 2026-10-09.

```text
execLoop is the pinned executor: pipeline commit(t) → pre-encode(t+1) → wait(t). One shared
autorelease pool, drained every drainEvery tokens (with a one-token non-overlapped hiccup so
no un-committed command buffer is live across the drain — keeps the pool LIFO-safe).

A buffer is encoded for a KEY COUNT, not just a head mode: the attention plan (attnPlan) is baked in at
encode time. Each buffer is encoded for the job it will run — the current job's own key count when
encoded fresh, pos+2 when pre-encoded for the predicted next job (the same request, one position on) —
and a pre-encoded buffer whose plan is not the arriving job's is dropped uncommitted and re-encoded. So a
new request, the attnFADepthFloor crossing, or a toggle can never run another position's plan (found
2026-09-25, R17: docs/measurements/metal-decode-attn-r17-2026-09-25.md; TestExecutorAttnPlan).
```

## resident.stopExec

Moved from `metal/model.go` (the comment above `stopExec`) on 2026-10-09.

```text
stopExec shuts down the executor goroutine (if started) and BLOCKS until it has returned —
which is what makes freeing safe. Closing the channel only signals; the loop may still be
waiting on an in-flight command buffer, and releasing a buffer it references is a
use-after-free. (CUDA hit the mirror-image ordering constraint in d8e81cb.)
```

## resident.slotBuffers

Moved from `metal/model.go` (the comment above `slotBuffers`) on 2026-10-09.

```text
slotBuffers returns the paged MoE slot-pool buffers — GPU-READ-ONLY (phase 2 reads them; the pread
stage CPU-writes their contents). Safe to pin resident.

P-22: previously enumerated only g4moe's pool, leaving every OTHER paged-MoE family (Mixtral/
Qwen/GLM/gpt-oss/qwen3_5_moe/qwen3_next — moe.go's generic path, "same mechanism, same
expertPool" per its own doc comment) unpinned and paying the full per-commit re-validation cost
the residency set exists to remove. Both pools hold the identical buffer category the five-arm
bisect (buildResident, above) found the win in — pread-invalidated slot buffers — so this
extends the SAME category to a second architecture family rather than adding a new one.
```

## resident.Close

Moved from `metal/model.go` (the comment above `Close` (it sat above `kvSlotBuf`)) on 2026-10-09.

```text
Close stops the executor, waits for it, then releases every MTLBuffer this resident allocated.
Returns error to satisfy io.Closer and match cuda/gpu's Close() error (audit B-12); teardown
itself can't fail (best-effort native releases), so it always returns nil.

It used to do only the first of those, with the comment "Metal buffers are freed at process
exit (single-model lifetime)". That assumption was false and had teeth: cmd/serve is
multi-model (--model name=path is repeatable) with /admin/models/{load,unload}, so every
load+unload leaked the whole model — weights + per-layer KV + the MoE stacked experts, i.e.
GIGABYTES of unified (system) memory, until the process exited. purego has no ARC and Metal
has no context-destroy to reclaim in bulk, so each buffer must be released explicitly.
Idempotent: ReleaseAll empties the ledger, so a second Close is a no-op (N-11).

⚠ THIS COMMENT IS WHY SOMEONE WILL ADD THE UNSAFE CALL. Read in isolation it says "unload leaks
gigabytes; Close reclaims them", which makes calling Close from handleAdminUnload look obviously
correct. It is not safe TODAY, and the reason is not in this file: serve's unload has a window
between pick() and enter() in which a request still holds lm.model with no lock held, so
TryLock grants the unload while a handler is mid-tokenize against those weights. Adding the call
turns a bounded leak into a use-after-free — on CUDA, a driver SIGSEGV that kills the server.
The blocker is serve's drain, not this teardown. Full account: internal/serveapp/admin.go,
handleAdminUnload. Close itself is correct and stays correct; it just has no safe caller yet.
kvSlotBuf is one resident KV slot's per-layer buffers (MC1).
```

## resident.ForwardArgmax

Moved from `metal/model.go` (the comment above `ForwardArgmax`) on 2026-10-09.

```text
ForwardArgmax runs the identical trunk but replaces the full lm head + 608KB readback with
Fable's fused block-argmax (per-tile (maxLogit,rowIdx) → argmax_finish → 4-byte token). It
returns argmax(Forward's logits) — same values, tie-broken first-max-wins — without ever
materializing the logit vector.

N-10 (audit-metal-2026-09-12.md): NOT actually production's greedy decode path — Metal has no
ResidentGreedy implementation, so generateInto's greedy case runs the same full-logits
ForwardEmbPipe every other sampling mode uses and argmaxes host-side (recorded speed-neutral on
UMA: the zero-copy logits view makes the host argmax ~30 µs, not a real cost). This method is
exercised only by tests/gates today; it is the fastest AVAILABLE greedy path, kept as API for a
future wiring, not a claim about what production calls.
```

## attnFACoreCount

Moved from `metal/model.go` (the comment above `attnFACoreCount`) on 2026-10-09.

```text
attnFACoreCount is the M1 Pro's GPU core count attention_fa's split count targets ("kvHead x S
>= 2x the core count", R2's own registered rule): 14 on the M1 Pro R2 was tuned on, which is this
repo's Mac (system_profiler, 2026-10-01; C-D01 in audit-metal-2026-09-30.md read it as 16). It sets
only the legacy kernel's split: the block kernel, which serves G = 6 and 7, uses attnFABlkSplit
(E-D01); changing it changes that split and so the legacy kernel's bits. It is hardcoded, not device-queried: aikit's Device
has no core-count accessor, and this kernel is default-on on every chip via metalAttnFAEnabled()
(2026-09-21) despite being tuned and measured on the M1 Pro alone; a wider port (other Apple
GPU core counts) would need this read from the device, not assumed.
```

## attnFABlkSplit

Moved from `metal/model.go` (the comment above `attnFABlkSplit`) on 2026-10-09.

```text
attnFABlkSplit is the split count the R17 block kernel (attention_fa_blk) runs at. Its split response is not
monotone and not the legacy kernel's: measured 2026-09-25 on the 1.5B at 3900 keys, S = 8/14/16/24/32/48 gave
2.61/2.82/3.37/2.15/2.90/2.36x in-sequence attention, and S = 16 was also best or near-best at 2048 keys and on the
7B. The confirmation run and the fidelity decision were both at S = 16
(docs/measurements/metal-decode-attn-r17-2026-09-25.md). The nKeys/32 cap never binds above attnFADepthFloor.
```

## attnFADepthFloor

Moved from `metal/model.go` (the comment above `attnFADepthFloor`) on 2026-10-09.

```text
attnFADepthFloor is where attention_fa takes over from the shipped kernel. It is 1024 since B-P03
(docs/tasks/task-metal-audit-2026-10.md, night of 2026-10-03): T1.2 measured the block kernel at 1.167x the legacy
kernel at 1024 keys on the 1.5B and 1.056x on the 7B, and the fidelity gate at the new depths passed. P1 on both
models put the block kernel's median and p99 relative L2 against float64 below the exact kernel's at 1024, 1280 and
1535 keys; P2 PASSES on the 1.5B at 1024 (KL ratio 0.9987). Below 1024 the two kernels were level (0.97-1.03x), so
the floor stays there. A same-day revert to 1536 was a test artifact: TestSpecNgram_copyOnStepVerify compared spec,
whose prompt reused the cache and re-decoded its last position, against a cold plain arm that had prefilled it, and the
f16 prefill is not bit-identical to decode; spec matches a plain arm on the same reuse exactly, tokens and K/V. The
history below is the original per-query-head kernel's, which set the old 1536 floor.

The original floor, 1536: where attention_fa (at a properly-sized split count) started beating the
shipped kernel — measured directly (TestAttentionFA_speedProbe, since deleted; tight-interleaved min-of-40,
S sized to attnFACoreCount*2): 0.98x at K=1024 (not yet a win), 1.07x at K=1536, climbing to
1.26x at K=3900. S=1 (no split) is NOT a shallow-depth fallback within this kernel — it measured
UNIFORMLY worse than shipped at every depth tried (0.38-0.53x even at K=1024), so below this
floor canUseAttnFA declines entirely and the shipped kernel runs, rather than this kernel at
S=1 as R2's own Build text first proposed ("S=1 below a measured crossover") — a correction the
speed probe surfaced, recorded here rather than silently overriding the brief's own text.
```

## gemvExtOn

Moved from `metal/model.go` (the comment above `gemvExtOn`) on 2026-10-09.

```text
gemvExtOn routes the int4 GEMVs R18 never reached through its rows-per-simdgroup kernels (D-B04): DeltaNet's qkv and
z projections, and a shared expert's gate|up and down. Each rows kernel is bit-identical to the one it replaces
(same lane-strided words, per-word sum and simd_sum; TestGemvExt_bitIdentical). ON since its grade passed
(docs/tasks/task-metal-audit-2026-10.md, "D-B04", night of 2026-10-03): the 9B decodes 1.064x faster at depth 128 and
1.061x at 1024, 7 of 7 reps above 1, logits equal in every rep, against the pre-registered >= 1.02.
```

## attnFABlkAnyG

Moved from `metal/model.go` (the comment above `attnFABlkAnyG`) on 2026-10-09.

```text
attnFABlkAnyG selects the block kernel for the dense group sizes other than the graded 6 and 7 (G = 2, 3, 4, 5 and
8; B-P02). ON since its grade passed (docs/tasks/task-metal-audit-2026-10.md, "B-P02", night of 2026-10-03): on the
two G = 2 models, legacy / block attention measured 2.37x and 2.32x (internlm2-1.8b) and 2.47x and 2.30x
(qwen3-0.6b) at 2048 and 3900 keys, 5 of 5 reps above 1, against the pre-registered >= 1.5. The kernel agrees with
g7 head for head (TestAttnFABlk_anyGMatchesG7).
```

## attnFABlk64On

Moved from `metal/model.go` (the comment above `attnFABlk64On`) on 2026-10-09.

```text
attnFABlk64On admits hd = 64 layers to attention_fa through the block kernel's hd = 64 twin (B-P01). ON since its
pre-registered grade passed (docs/tasks/task-metal-audit-2026-10.md, "B-P01", night of 2026-10-03), on the 0.5B:
P1, the twin's median and p99 relative L2 against float64 below the exact kernel's; P2 at 3900 keys, critA, critB and
the ceiling hold with a KL ratio of 1.0098 (<= 1.05); speed, legacy / block attention 3.17x at 2048 keys and 3.63x
at 3900 (>= 1.5). The twin is not bit-identical to the per-query-head kernel, which is why it was graded on fidelity.
```

## canUseAttnFA

Moved from `metal/model.go` (the comment above `canUseAttnFA`) on 2026-10-09.

```text
canUseAttnFA reports whether attention_fa may replace the shipped kernel for layer l in the command
buffer being encoded (at planNKeys keys). R2 (docs/tasks/red-october.md): dense-GQA only, hd==128
only (the kernel's own cooperative-load tiling is fixed to 32 lanes x half4) — sinks, windows,
and the f32-KV twin are explicitly out of scope until the dense-GQA kernel clears the band (the
brief's own text), same reasoning as canUseF16Lane's family exclusions above it.
```

## buildResident.qGateLayer

Moved from `metal/model.go` (the comment in `buildResident`, above the gated softmax layer branch) on 2026-10-09.

```text
The same family's GATED softmax layer (qwen3_5/qwen3_5_moe/qwen3_next — NOT every
dnetOK family: Olmo Hybrid's full-attention layer is olmo3's plain scheme instead,
AttnGate=false, and falls through to the ordinary branch below — G5,
docs/tasks/task-gpu-paths-2026-09.md. This used to be a bare `dnetOK` check, silently
wrong the moment a non-gated hybrid family reached residency, since
Qwen35ResidentParams hardcoded attnGate=true). Its q/k/v/o live off qattn, not
lw.QProj/KProj/VProj/OProj,
and q_proj is DOUBLE WIDTH ([query ‖ gate] per head, interleaved — NOT two
concatenated blocks; treating it as an ordinary q_proj yields the first nH/2
heads' query+gate as "queries", plausible logits from the wrong tensor, measured
cosine 0.90 on the WebGPU side). So the fused QKV path doesn't apply here: K‖V
still fuse into L.qkvW (2-way, not 3-way); Q is a separate double-width projection
(L.dnQw/dnQs), split on the ACTIVATION at encode time (delta_qsplit) because the
weight is quantized and slicing rows out of an int4 bundle with per-group scales
is real surgery.
```

## buildResident.lmHead

Moved from `metal/model.go` (the comment in `buildResident`, above the LM head upload) on 2026-10-09.

```text
The LM head is LOGIT-CRITICAL and must stay int8. decoder/weightmat.go: "at int4 they flip
the argmax and tank the cosine (the tied head dots every logit against them)" — which is
why the decoder PINS the embedding/LM-head at int8 even in int4 mode. Metal was
int4-quantizing it anyway, violating that pin. Worst for Gemma: a TIED head, 262k x 2560,
so every one of 262k logits is dotted against int4-mangled embedding rows.
```

## buildResident.sa8

Moved from `metal/model.go` (the comment in `buildResident`, above `bad8`) on 2026-10-09.

```text
C-10: the SA-GEMV decode kernels (gemv_w4a8_sa / _sa_bias / _sa_resid and the MoE variants)
derive the output row from the runtime threadgroup size and — unlike gemv_w4a8_sa_bk — carry
no `row >= N` guard. So an output width N%8 != 0 makes the tail threadgroup rewrite an
already-written row while the true tail rows stay uninitialised scratch: plausible-looking
wrong logits, no error. The attention widths (qDim = nH·hd, kvDim = nKV·hd) are structurally
%8 (hd is 64/128), so the risk is the model-level FFN widths below. Decline any that isn't a
multiple of 8 → the correct CPU path. Every shipped metal-eligible arch is %8 today, so this
declines nothing now; it guards a future odd-width model from the silent corruption.
(The deeper fix — an N param + `row >= N` guard in the SA-family/MoE kernels — touches every
dispatch binding and is deferred as device-validation-gated; this build-time decline is the
safe half for THOSE kernels.)

Vocab is NOT checked here (2026-08-18): it never routes through an SA-family kernel — the LM
head is pinned int8 (line ~624) and only ever dispatches gemv_w8a8_coal (forwardLogits) or
gemv_w8a8_amax (ForwardArgmax). gemv_w8a8_coal addresses its row directly via
threadgroup_position_in_grid, which Metal guarantees correct regardless of a threadgroup's
uniformity — no hazard, %8 or not. gemv_w8a8_amax uses the SAME hazardous tgs-derived
formula as the SA family, but ForwardArgmax now routes a non-%8 vocab around it (full logits
+ host argmax) rather than declining the whole family — see ForwardArgmax. GPT-2 (50257) is
the first family this reaches.
```

## buildResident.attnFABlk

Moved from `metal/model.go` (the comment in `buildResident`, above the `attnFANKV` block-kernel selection) on 2026-10-09.

```text
R17: for the GQA group sizes it was graded at (G = 6, Qwen2.5-1.5B; G = 7, Qwen2.5-7B), attention_fa's first
pass is the block-of-32 kernel attention_fa_blk at a fixed split count of 16 (same grid shape, partial layout
and combine). Every other group size keeps attention_fa and its core-count rule — the block kernel is
instantiated, measured and fidelity-gated only for these two.
```

## buildResident.residencySet

Moved from `metal/model.go` (the comment in `buildResident`, above `paged :=`) on 2026-10-09.

```text
Residency set (default ON when supported + paged; GOINFER_MOE_RESIDENCY=0 opts out). The paged
path submits per-layer, and the pread stage CPU-writes the slot buffers each token — dirtying
their residency so the driver re-validates them every phase-2 commit (~9 ms/CB of GPU-idle-in-
wait). Pinning the SLOT POOL resident holds it across those writes → ~0.44 ms/CB (measured
−11%: 0.61→0.68 tok/s at N=32 on an idle 16 GB box). SLOTS ONLY: a five-arm bisect showed
pinning anything more (weights/KV/scratch) regresses phase 1 in proportion to pin-set size,
read/write-agnostic — pinning helps only the pread-INVALIDATED buffers. FOOTPRINT: the slot pool
is N × MoE-layers × per-expert-bytes (≈3 GB at N=32), PERMANENTLY requested resident; re-measure
the win if N grows or the box is under other load. Capability-gated (macOS 15+); older OSes keep
the correct, slower per-submit path.

P-22: this used to gate on g4moe alone, so a generic paged-MoE model (moe.go's path — same
per-commit re-validation cost, same pread-invalidated slot buffers) never got a residency set
built at all, regardless of what slotBuffers() enumerated.
```

## buildResident.residencyAttach

Moved from `metal/model.go` (the comment in `buildResident`, above `r.residency = rs`) on 2026-10-09.

```text
M-14 (audit-metal-2026-09-12.md): attaching the set at the QUEUE (r.q.AddResidencySet)
used to ride it on EVERY command buffer, so phase 1 carried the ~3 GB of pinned slots in
its referenced set even though it never touches them — +2.07 ms/CB → +62 ms/tok measured
cost. r.residency is instead attached PER-ENCODER, only on phase 2's command buffers
(encodeG4Phase2Paged's / encodeMoEExpertsPaged's callers, via Encoder.UseResidencySet —
aikit gpu/v0.33.1+), which is the buffer category that actually reads it.
```

## Package.model

Moved from `metal/model.go` (the comment opening the file (the package comment)) on 2026-10-09.

```text
Real-model resident Metal decoder — the final GO/NO-GO piece of the spike. Loads a
dense Qwen2/Llama model's int8 weights out of the goinfer decoder, uploads them to
Metal once, and runs the full layer stack per token in ONE command buffer (the tax
requirement). cgo-free throughout (purego-objc + dlopen Metal).
```

## resolveMetalCtxCap

Moved from `metal/model.go` (the comment above `resolveMetalCtxCap`) on 2026-10-09.

```text
resolveMetalCtxCap turns a request into the effective resident KV capacity, mirroring
cuda/resident.go's resolveCtxCap[Fit] in SHAPE: an unpinned load (req <= 0) gets metalCtxCapDefault (4096);
an explicit request up to metalCtxCapMax (32768) is honored (optionally clamped to the model's window);
and a request ABOVE metalCtxCapMax is REFUSED with the numbers. A context the load-time fit guard auto-pinned
(R13; the caller did not choose it) is a ceiling, not a request: it may lower the default, never raise it.
Read as a request it allocated KV several times the default on a tight machine and, above metalCtxCapMax, was
refused, which moved the whole forward to the CPU (C-C01, docs/audit-metal-2026-09-30.md).
```

## tgReduceNorm

Moved from `metal/model.go` (the comment above the `tgReduceNorm`/`tgReduceAttn` constants) on 2026-10-09.

```text
Threadgroup widths for the kernels that contain a CROSS-THREAD FLOAT SUM reduction (a
`red[tid]+=red[tid+st]` tree): rmsnorm sum-of-squares, softmax denominator, qk-norm.

THESE ARE BIT-IDENTITY-LOAD-BEARING, NOT PERFORMANCE KNOBS. Float add is non-associative, so a
tree reduction's result depends on its WIDTH: T threads sum N/T strided partials, then a T-wide
tree — change T and the last bits of the sum move. On CUDA the warp reduce is a fixed 32, so the
coupling can't exist; on Metal the threadgroup width is exactly what you'd sweep for a 5% win, so
every such kernel's numerics are wired to its launch configuration.

The existing gates will NOT catch a change here: paged≡non-paged compares the SAME kernel at the
SAME width (self-consistent — both move together), and GPU-vs-CPU parity is cosine/tolerance. The
coupling only surfaces when a NEW path computes the same reduction at a DIFFERENT width and is gated
byte-exact — which is exactly how the split/staged attention rewrites diverged (a 256-wide softmax
denom vs the shipped 128-wide tree; and it only appeared past nKeys>256, below a short fixture).
So: pin the width here, keep every dispatch of these kernels bound to it, and make any alternate
same-op kernel inherit it. A byte-exact fixture for such an op MUST use context > the width. Max
reductions and simd_sum (32, hardware-fixed) are exempt — associative+commutative, order-exact.
See docs/ollama-chase.md §2 (ground rules) and §A2-Metal.
```

## prefillFeatures

Moved from `metal/model.go` (the comment above `prefillFeatures`) on 2026-10-09.

```text
prefillFeatures is what the f16 MMA prefill kernels (prefill.go) actually implement: a dense
gated FFN (SiLU or GeGLU), per-head QK-norm, a PER-LAYER rope table and window, Gemma's
sandwich norms and (1+w) RMS offset, the embed-scale/final-logit-softcap pair (both applied
OUTSIDE this file — embedResident scales the input embeddings before PrefillLast ever runs,
and PrefillLast's own final step applies softcap — so declaring them here is a pure capability
statement, no kernel change), and MoE (G8's second half, while the attention half batches
normally: by default the FFN half runs expert-major — one batched router GEMM and top-k over all
rows, host grouping, then each active expert over its rows — and GOINFER_MOE_EXPERT_MAJOR=0 keeps
the original row-by-row loop through the per-token decode MoE chain; PrefillLast's L.moe != nil
branch. A paged MoE declines batched prefill altogether; D-D01, audit-metal-2026-09-30.md).
FeatMoEGatedShared (Qwen2-MoE's sigmoid-gated shared expert) comes along for free: it's the
SAME encodeMoESharedExpert dispatch decode already uses, gated/ungated branch and all.
```

## residLayer.delta

Moved from `metal/model.go` (the comment in `residLayer`, above `delta`) on 2026-10-09.

```text
Gated-DeltaNet mixer layer (Qwen3.5/3.6-MoE, Qwen3-Next, Qwen3.8; see metal/deltanet.go).
delta non-nil marks this layer's sequence mixer as the recurrent delta rule instead of
attention: no KV cache, no q/k/v/o, no geom (a DeltaNet layer has no attention geometry —
r.kc[l]/r.vc[l] are left as their zero Buffer for these layers, saving real memory on 3 of
every 4 layers this family has, and encodeLayer never dispatches pKv/pAttn for them).
```

## resident.decodeLaneW4F16

Moved from `metal/model.go` (the comment in `resident`, above `pRmsF16`) on 2026-10-09.

```text
R1 (docs/tasks/red-october.md): the W4F16 decode lane — f16 activations, no int8
quantization, gated by GOINFER_METAL_DECODE_LANE=w4f16 (decodeLaneW4F16). Pipelines are
always built ("one binary carries both arms", per the brief); only DISPATCH is
conditional. First slice only: plain dense QKV/o-proj/gate-up (see canUseF16Lane) — down-
proj (the "coal" family) and every special-case path (MoE, paged, sandwich, postOnly,
parallelBlock, qGate, outBias, DeltaNet, nonGatedMLP) stay on the shipped W4A8 kernels.
```

## resident.pAttnFA

Moved from `metal/model.go` (the comment in `resident`, above `pAttnFA`) on 2026-10-09.

```text
R2 (docs/tasks/red-october.md): the split-KV decode-attention lane, gridded by (kvHead,
split) instead of by query head, DEFAULT ON since 2026-09-21 (decodeAttnFA, set from
metalAttnFAEnabled(); GOINFER_METAL_ATTN_FA=0 opts out) — see that function's own doc
comment for the gate/speed/verify-oracle history. Pipelines always built; dispatch is
conditional (canUseAttnFA), same "one binary carries both arms" shape as R1's f16 lane
above. hd==128 (or hd==64 through B-P01's block twin, attnFAHeadDimOK), dense-GQA-only (no window/sinks/f32-KV) — see attention_fa's own doc
comment in kernels.go for why, and §2.2/R2's own speed-probe record for why S must be sized
for real occupancy (S=1 is UNIFORMLY worse than the shipped kernel at every depth measured,
not just below some crossover — a correction to this brief's own original "S=1 below a
measured crossover" text).
```

## resident.attnFASplitOverride

Moved from `metal/model.go` (the comment in `resident`, above `attnFASplitOverride`) on 2026-10-09.

```text
attnFASplitOverride, when > 0, replaces attnFASplitFor's split-count rule (the nKeys/32 and
attnFAMaxSplit caps still apply). ZERO in production — set only by tests (R17 step 0 sweeps S to test
whether attention_fa is latency-bound on too few simdgroups in flight). Read inside attnFASplitFor, the
one function both the dispatch grid and setPos's uAttnFANSplit use — and both for the same key count
(the grid's comes from planNKeys) — so the two cannot disagree.
```

## resident.attnFAFloorOverride

Moved from `metal/model.go` (the comment in `resident`, above `attnFAFloorOverride`) on 2026-10-09.

```text
attnFAFloorOverride, when > 0, replaces attnFADepthFloor as the key count at which attention_fa takes over.
ZERO in production — set only by tests (T1.2 of docs/tasks/task-metal-audit-2026-10.md: legacy-against-blk arms
below the floor). Read through attnFAFloor by attnPlanFor, canUseAttnFA and canUseAttnFAAt, so the single-token
step and the batched step plan alike.
```

## resident.uAttnFAG

Moved from `metal/model.go` (the comment in `resident`, continuing the trailing comment of `uAttnFAG`) on 2026-10-09.

```text
per-layer dispatch site: a prior version SetU32'd these once per LAYER, i.e. during encodeTrunkCB's
encoding of the NEXT command buffer while the CURRENT one was still executing on the GPU (the
pipelined executor's own "encode t+1 while t runs" design, execLoop). That is a raw CPU write to
shared memory racing a concurrently-running GPU kernel's read of the SAME buffer — Metal's automatic
hazard tracking covers GPU-encoded command dependencies, not this. Reproduced in isolation
(TestAttentionFA_pipelinedEncodeRace, varying nSplit per iteration to make it observable) with
EXACTLY R2's own real-generation signature: iterations 0-1 correct, iteration 2 wrong, an irregular
pass/fail pattern across later iterations — the signature of a genuine race, not a deterministic
logic bug. Root cause of the divergence r2-attn-fa-2026-09-19.md left unexplained.
```

## resident.pLoraDelta

Moved from `metal/model.go` (the comment in `resident`, above `pLoraDelta`) on 2026-10-09.

```text
Compute-time LoRA (G3, docs/tasks/task-gpu-paths-2026-09.md — see lora.go). pLoraDelta is
allocated once in BuildResident unconditionally (cheap; same "always create, gate on the
per-model state" shape every other optional pipeline in this struct already uses). Fused
down+up into one kernel/dispatch (P-11, audit-2026-09-10) — t[R] lives in the kernel's own
threadgroup memory now, no device-side scratch buffer to hold between dispatches.
loraLayers is nil until SetAdapter binds one; the dispatch sites (encodeAttention/encodeLayer)
no-op per projection when it is nil or the targeted projection's delta is nil.
```

## resident.loraCacheSrc

Moved from `metal/model.go` (the comment in `resident`, above `loraCacheSrc`) on 2026-10-09.

```text
loraCacheSrc/loraCached are P-10's single-adapter device cache (audit-2026-09-10): the
source layers a bind's device buffers were built from, and those buffers themselves,
KEPT ALIVE across a SetAdapter(nil) clear rather than released — a rebind of the SAME
adapter (the common "one chat session, many turns" shape internal/serveapp/main.go's
N-adapters-one-resident design produces) then skips the re-upload entirely instead of
paying it on every generation. See SetAdapter's own comment for the identity check.
```

## resident.embedScale

Moved from `metal/model.go` (the comment in `resident`, above `embedScale`) on 2026-10-09.

```text
embedScale is Gemma's √hidden token-embedding multiplier (FeatEmbedScale); 0/1 ⇒ none.
Applied by the id-taking entry points (Forward / ForwardArgmax) right after the embedding
lookup — the ONE place they differ from ForwardEmb, whose caller has already scaled
(decoder.embedResident). Before audit G-02 this field did not exist and those two methods
silently ran Gemma unscaled; see Forward.
```

## resident.execReq

Moved from `metal/model.go` (the comment in `resident`, above `execReq`) on 2026-10-09.

```text
pipelined logits executor (encode-ahead): a persistent OS-thread-pinned goroutine that
commits token t, pre-encodes t+1 while the GPU runs t, then waits — hiding the ~0.9ms
host encode bubble. Lazily (re-)started on the first ForwardEmbPipe after execReq is nil —
C-07: SetAdapter tears the executor down (stopExec) rather than a sync.Once, because its
pre-encoded t+1 buffer bakes in r.loraLayers AT ENCODE TIME; a bind/switch/clear between
two ForwardEmbPipe calls left that stale buffer to be committed under the NEW adapter
state. ForwardEmbPipe and SetAdapter are never concurrent (the resBusy winner's own
sequential SetAdapter → Forward* → SetAdapter(nil)), so the plain nil-check below needs no
extra lock.
```

## resident.pfErr

Moved from `metal/model.go` (the comment in `resident`, above `pfErr`) on 2026-10-09.

```text
pfErr latches a compile/pipeline-creation failure from ensurePrefill (N-47,
audit-2026-09-10.md): without this, r.pf stays nil after a failure and every later
PrefillLast call re-attempts the full MSL compile from scratch and re-panics identically —
wasted work on every request for the rest of the process's life, not just the first.
```

## int4DirectWords

Moved from `metal/model.go` (the comment above `int4DirectWords`) on 2026-10-09.

```text
int4DirectWords converts a decoder int4 WeightMat's packed nibbles + f32 group scales straight
into Metal's W4A8 buffers (uint32 words + f16 scales) — NO int8 intermediate. aikit's group=32
packing (nib = q+8, byte k/2 low/high) and Metal's packW4A8Row (element k → word k/8, bit
4·(k%8)) are the SAME bytes on little-endian, so the nibbles copy verbatim; only the group
scales narrow f32→f16. Returns ok=false if the weight is not group-32 int4.

This is the fix for Gemma's dormant residual: BuildResident's default path double-quantizes
(f32→int8→int4), and Gemma's low-magnitude attention contexts amplify that int8-intermediate
drift into a catastrophic context error (metal/gemma_sublayer_test.go: L1 cosine craters to
0.649 vs the direct-int4 reference's ~0.93). Consuming the decoder's int4 directly — exactly
what CUDA/WebGPU do — removes the int8 step. Qwen is insensitive to it (ships clean either way).
```

## bytesToU32

Moved from `metal/model.go` (the comment above `bytesToU32`) on 2026-10-09.

```text
bytesToU32 reinterprets a little-endian byte slice as uint32 words (len must be a multiple of 4).
N-29 (audit-metal-2026-09-12.md): b is already the target little-endian word bytes (nothing to
reconstruct arithmetically) — a per-element shift-and-mask loop was doing a byte copy the slow
way. w is freshly allocated (always 4-aligned, unlike the mmap-backed sources int4DirectBytes
exists to avoid this same reinterpret on), so a single bulk copy into its own []byte view is
exactly as safe as manual shifts and orders of magnitude fewer instructions.
```

## int4DirectBytesOnly

Moved from `metal/model.go` (the comment above `int4DirectBytesOnly`) on 2026-10-09.

```text
int4DirectBytesOnly returns a canonical group-32 int4 WeightMat's packed nibble bytes ALIASED straight from the
mmap: no bytesToU32 reconstruction and no per-stage []uint32 allocation (measured on the 26B: ~215 ms and 1.9 GB of
reconstruction per run, both removed). They are byte-for-byte the words int4DirectWords builds (little-endian), so a
byte-copy into a uint32 slot buffer reproduces them exactly (expertpool.copyBytesToU32Buf). The paged MoE stage
functions (buildMoELayer, buildGemma4MoELayer) pair it with the WeightMat's own Int4ScalesF16: re-deriving f16 scales
from an f32 copy on every page-in was ~2.85 GB/token of transient allocation on the 26B (N-20,
audit-metal-2026-09-12.md), and the build-time cache that replaced it duplicated 1361 MB of scales the mapping
already holds (C-P01, audit-metal-2026-09-30.md).
```

## parallelF32ToF16

Moved from `metal/model.go` (the comment above `parallelF32ToF16`) on 2026-10-09.

```text
parallelF32ToF16 converts src (f32 group scales) to dst (f16 bits) across up to 8 workers. In the
gemma4-26b expert-paging path this f32→f16 conversion runs once per expert PER STAGE (~600 stages/
token × ~186K scales) and was ~228 ms/token of staging, arithmetic-dominated (alloc ~27 ms, copy
~5 ms) — but every element is independent and f32ToF16 is deterministic, so splitting it across
cores is a free (no-memory) staging win that is BYTE-IDENTICAL to the serial loop. Serial for small
inputs (the non-paged one-time build, where goroutine spawn would not pay).
```

## maxThreadgroupStageBytes

Moved from `metal/model.go` (the comment above `maxThreadgroupStageBytes` (the comment also held `int4Buf`'s first lines)) on 2026-10-09.

```text
int4Buf uploads a WeightMat as W4A8 (int4, group=32) + f16 group scales. If the weight is
ALREADY int4 (a Quant:"int4" load), it consumes the nibbles directly (int4-direct, no int8
step); otherwise it re-quantizes the int8 weight through the validated packer. One-time at build.
maxThreadgroupStageBytes returns the largest threadgroup staging allocation (bytes) any resident
dispatch will request. The SA-GEMV / MoE kernels stage the GEMV's contraction row into threadgroup
memory at 2 bytes/element (DispatchTG tgBytes = 2·K); the widest staged K is `hidden` (qkv/gate-up),
the q-width `qWidth` = nH·hd (o-proj), or — for MoE — the expert intermediate `moeInter`/`g4moeInter`
(the expert down-proj stages `inter`). The dense down-proj uses the non-staging pGemv, so the dense
intermediate is deliberately NOT counted. Split out so the M-11 budget arithmetic is unit-testable.
N-32: dnValueDim is DeltaNet's out-projection staging width. deltanet.go dispatches pSAResid
with `dp.valueDim*2` threadgroup bytes, and that term was missing here — so on a DeltaNet model
whose value dim exceeds every other staged width, the M-11 budget under-counts and the check
passes on a configuration that then exceeds the device limit at dispatch.
```

## int4BufA.K32

Moved from `metal/model.go` (the comment in `int4BufA`) on 2026-10-09.

```text
The W4A8 layout and every GEMV kernel hard-assume K is a multiple of the group (32): rows are
packed K/8 words + K/32 scales with no partial-group handling. A K%32 != 0 weight would pack a
truncated last group (trailing nibbles decode as −8) with a per-row stride the kernel disagrees
with — silently wrong, or a panic at K<32. Decline so BuildResident falls back to CPU (M-10).
```

## int4Concat.presize

Moved from `metal/model.go` (the comment in `int4Concat`) on 2026-10-09.

```text
N-29 (audit-metal-2026-09-12.md): pre-size from the known final shape (N*K/8 words, N*K/32
f16 scales per group-32 W4A8 tensor — the same formula int4Buf's own int8-fallback branch
already allocates by) instead of growing two nil slices by append, which reallocates+copies
on every capacity doubling across a fused tensor's weights.
```

## int4Concat.noCanonical

Moved from `metal/model.go` (the comment in `int4Concat`) on 2026-10-09.

```text
int4DirectWords already declined above (its own w.Int4() check failed), and this is
not int8 either. The common real cause is the repacked-only int4 policy
(wantsCanonicalInt4, aikit audit M-22): a tensor built with Options.Backend != "cpu"
stays canonical, but one built WITH Backend:"cpu" (or loaded generically and handed
to Metal's own BuildResident out of band — decoder.Load has no way to see that
coming) may have no canonical bytes left at all. Name that condition specifically —
the previous message ("weight kind %q not int8 or int4") both mis-stated the
check (it read Int4()'s narrower "canonical present" ok, not Kind()) and gave no
actionable next step. A tensor that is neither int8 nor int4 at all (a real
programming error reaching this function) still gets a message, just the older,
more generic one.
```

## nativeInt8

Moved from `metal/model.go` (the comment above `nativeInt8`) on 2026-10-09.

```text
nativeInt8 turns the native int8 path on (w8Eligible). ON since 2026-10-04: F3′, F2 (read as hard flips, owner
decision), P2, S and S-auto passed (docs/tasks/task-metal-int8-2026-10.md). Tests turn it off for the int4
re-quant arm; it is not an option or an environment variable.
```

## nativeInt4Mix

Moved from `metal/model.go` (the comment above `nativeInt4Mix`) on 2026-10-09.

```text
nativeInt4Mix turns on int4mix's native path (w8AttnEligible: attention W8A8, FFN int4). ON since 2026-10-04: M1 and
M3 passed, and M2 is read as hard flips (owner decision; it passes, 1 against the re-quant's 3), with M4 the night's
prefill confirmation (docs/tasks/task-metal-int8-2026-10.md, "Slice 4: int4mix"). Tests turn it off for the re-quant arm.
```

## nativeInt8MoE

Moved from `metal/model.go` (the comment above `nativeInt8MoE`) on 2026-10-09.

```text
nativeInt8MoE admits a generic resident MoE to the native int8 path (w8Eligible; int8 expert GEMVs gemv_w8a8_moe*).
ON since 2026-10-04: X1-X4 passed (docs/tasks/task-metal-int8-2026-10.md, "Slice 4: MoE int8"); tests turn it off.
```

## w8PreciseMath

Moved from `metal/model.go` (the comment above `w8PreciseMath`) on 2026-10-09.

```text
w8PreciseMath compiles the library of a model headed for the native int8 path without fast math (owner decision,
2026-10-04, docs/tasks/task-metal-int8-2026-10.md): at int8int8 any difference from the CPU is amplified by the
activation quantization, and fast math was about 40% of gate F3's gap (KL(f32 ‖ Metal int8) 0.027917 fast, 0.025440
precise, against the f16-KV CPU's 0.024045). It is decided before the build from the model alone, so a model that
passes w8Weights but is MoE (Gemma 4's parallel dense‖MoE) is excluded here too.
```

## buildResident

Moved from `metal/model.go` (the comment above `buildResident`) on 2026-10-09.

```text
BuildResident builds a Metal resident decoder from an int8-loaded dense Qwen2/Llama
Model. Handles Qwen2 q/k/v bias; assumes no QK-norm / sliding-window / embed-scale
(the DecodeRunnerEligible dense shape), full RoPE via the model's own inv-freq table.
```

## buildResident.kvSlots

Moved from `metal/model.go` (the comment in `buildResident`, above `kvSlots`) on 2026-10-09.

```text
MC1's slot count is priced BEFORE any buffer below exists. Its budget, metalMemoryCeiling, reads the live
available memory; asked after the weights were allocated (as it once was, at the slot-allocation site), that
reading already excluded the resident's own buffers while the base it is compared with (residentNeedBytes)
still counts them — the weights counted twice. Measured 2026-09-26 on the 1.5B at a 1024 context: 2 slots of 8
granted (28 MB each) with a base of 4,044 MB against 5,531 MB live before the build, where 8 fit.
```

## buildResident.preciseMath

Moved from `metal/model.go` (the comment in `buildResident`, above `compile`) on 2026-10-09.

```text
preciseMathCompile (test/measurement toggle, Task 3): fast-math OFF removes the compiler's
contraction/reassociation/transcendental discretion — the axis that makes within-machine
bit-identity fragile to an OS toolchain update. Measure its decode-tok/s cost before adopting.
Default fast-math (measured 2026-08-04: precise costs ~4% @2048 / ~7% shallow and does NOT
improve CPU parity — §A2-Metal). GOINFER_PRECISE_MATH is a documented opt-in for anyone who wants
bits robust to an OS-toolchain update at that cost; the snapshot golden otherwise DETECTS such
drift, which is the cheaper path we chose.
```

## buildResident.kvF32

Moved from `metal/model.go` (the comment in `buildResident`, above `r.kvF32 = false`) on 2026-10-09.

```text
f32 KV path (kv_store_f32/attention_f32) exists but is DISABLED: the matched-input confirmer
proved f16-vs-f32 KV storage is NOT the Gemma crater. The whole crater is Metal's BOS
(position-0) K/V being computed wrong (cos 0.40 vs goinfer); overwriting just that recovers
the context to 0.999. So precision was a red herring (mine and the CUDA box's) — the real
bug is the position-0/attention-sink K/V compute. Kept off until that's fixed.
```

## buildResident.prefillOK

Moved from `metal/model.go` (the comment in `buildResident`, above `missing := m.MissingResidentFeatures`) on 2026-10-09.

```text
prefillOK, derived rather than hand-listed: the f16 prefill kernels implement exactly the
features below, so ANY model needing more (MoE — never packs the dense FFN buffers at all)
declines prefill and falls back to the sequential Forward loop.

G8's Gemma-set admission (prefillFeatures above) is NOT enough on its own for dense Gemma 4:
PrefillLast reads g0 := r.layers[0].geom ONCE and reuses it for every layer's rope/attention
dims — correct for every uniform-geometry family (including Gemma 3's dual-base RoPE, which
keeps head_dim uniform), but Gemma 4's local/global split genuinely varies head_dim per
layer (256 vs 512). decoder.Model.PerLayerGeomOK(backend) answers "does this arch need
per-layer geometry, and does BACKEND declare support" — Metal's DECODE path does (it has
its own per-layer geom seam, encodeAttention's geomFor), so calling it with "metal" would
wrongly clear Gemma 4 here too. Checking !m.HasPerLayerGeometry() directly answers
"does this arch vary per layer at all", the thing this uniform-g0 fast path can't handle
regardless of what Metal's decode path separately supports.
G8 MoE (docs/tasks/task-gpu-paths-2026-09.md): Gemma-4's enable_moe_block variant (parallel
dense‖MoE FFN, residLayer.g4moe, encodeGemma4MoEFFN) is a THIRD FFN shape this row's
L.moe != nil branch in PrefillLast does not cover at all — declaring FeatMoE above would
otherwise admit it if it happens to have uniform per-layer geometry (HasPerLayerGeometry alone
only catches the local/global head_dim variance dense Gemma 4 has; nothing about g4moe's
FFN shape is geometry). Explicit, checked directly rather than assumed caught by the other
guard — the exact class of blind spot this doc's own G6 WebGPU incident already burned once.
C-08 (audit-2026-09-10): a PAGED generic MoE's per-layer expGuW/expGuS/expDW/expDS buffers
are zero-value (moe.go — the real weights live in the slot pool instead), but PrefillLast's
row loop calls the same non-paged encodeMoERoute/encodeMoEExperts pair unconditionally.
Same predicate as the dense Gemma-4 MoE guard above, generalized to the generic twin.
A-C01 (docs/audit-metal-2026-09-30.md): the prefill kernels write K/V with kv_store_f16, half per element at
pos*kvDim, but an int8 KV cache (-kv i8) is allocated at one byte per element with separate scale buffers, and
PrefillLast's attention reads the cache as half too. So with -kv i8 every prompt position landed in the wrong
layout and positions at or past ctxCap/2 were written past the buffer. Such a model takes the sequential path,
whose decode kernels write and read the int8 cache.
D-B01: a Gated-DeltaNet hybrid takes the pass only with dnetPrefillOn (on since its grade, 2026-10-04), and only in the shape
prefill_deltanet.go implements: Qwen3.5's pre-norm layers, no LayerNorm bias. Olmo Hybrid (postOnly) stays
sequential.
```

## prefillKernels

Moved from `metal/prefill.go` (the comment above `prefillKernels`) on 2026-10-09.

```text
Prefill kernels — the f16 simdgroup_matrix (MMA) path for fast prompt ingestion. Unlike the
int4/scalar-MAC decode path (which can't amortize batching), an MMA GEMM reuses each weight
across all M prompt rows → ~2.5× the per-token GEMV, flat with M. Activations flow in f16
(no int8 quant); weights stay int4 and are dequanted to f16 in-kernel (no extra RAM). Kept in
a SEPARATE library from allKernels so the decode path is unchanged and prefill is opt-in.
```

## gemmTile

Moved from `metal/prefill.go` (the comment above `gemmTile`) on 2026-10-09.

```text
gemmTile is the tile selector for a rows × N prefill GEMM. Every choice is bit-identical (gemm_w4f16_tile's comment,
TestGemmTile_bitIdentical).

The default (D-B02, docs/tasks/task-metal-audit-2026-10.md "D-B02: the 16-row tile"): a threadgroup costs the same
whatever share of its rows is real, so a smaller token tile is taken when it pads the pass to strictly fewer rows:
32 tokens up to 128 rows, 16 up to 16 rows and, when N <= 8192, up to 48. Past those, more row tiles cost more than
the padding they save (TestGemmTile16_probe: 16-token tiles lose from 96 rows, and a 100-token pass ran 0.847x on
all-16-row tiles); a gate|up GEMM as wide as the 1.5B's or 7B's already fills the cores at 64 (the 7B's 40-token pass
ran 0.984x with it on 16-token tiles). A
16-token tile always takes 32 features; otherwise A-P01's feature rule, 32 when N <= 2048 and rows <= 64, so the
narrow GEMMs fill more cores.

A-P01's rule ("a01"): 32 tokens when the pass has at most 32 rows, 32 features when N <= 2048 and rows <= 64.
```

## prefillState.pGemm

Moved from `metal/prefill.go` (the comment in `prefillState`, above `pGemmStore`) on 2026-10-09.

```text
pGemm (gemm_w4f16, no store epilogue) was created but never dispatched — the prefill LM head
moved to pRmsQ + pGemvW8, and every GEMM here uses pGemmStore. Removed (audit R-22 / N-09 class).
```

## parallelEmbedsF32ToF16

Moved from `metal/prefill.go` (the comment above `parallelEmbedsF32ToF16`) on 2026-10-09.

```text
parallelEmbedsF32ToF16 converts M rows of embs (each H wide) into dst[m*H:(m+1)*H] as f16 bits,
splitting across up to 8 workers by ROW — P-14 (audit-2026-09-10): the serial scalar loop this
replaces was 7.3M f32ToF16 calls at M=2048, H=3584 on the TTFT path, and model.go's own
parallelF32ToF16 (built for exactly this conversion in the gemma4-26b expert-paging path)
already proved the parallel split is byte-identical to serial — every element is independent
and f32ToF16 is a pure function of its one input. Not reused directly: embs is [][]float32 (one
slice per row, not necessarily contiguous), where parallelF32ToF16 wants one flat []float32; a
flatten-then-call would pay its own copy, so this splits by row directly instead, over M×H
rather than a flat index range, but is otherwise the same threshold/worker shape.
```

## prefillExactAttnMaxKeys

Moved from `metal/prefill.go` (the comment above `prefillExactAttnMaxKeys` (the comment also held `PrefillLast`'s opening)) on 2026-10-09.

```text
PrefillLast ingests M prompt embeddings at positions startPos..startPos+M-1 in ONE command
buffer via the f16 MMA path (weights read once, amortized across M — unlike the token-by-token
decode loop), populating the resident KV cache, and returns the LAST token's logits[V] (what a
generator needs to sample the first output token). Correctness-gated vs the sequential path.
prefillExactAttnMaxKeys is the exact attention_prefill kernel's bound: it keeps one score per key in
`threadgroup float sc[4096]`, indexed by the key's absolute position, with no tiling (the decode kernels tile theirs,
attnScoreTileBound). F-C02 (docs/audit-metal-2026-09-30.md): above 4096 keys it wrote past threadgroup memory. The
exact kernel runs whenever the fused one cannot (head dim above 128 or not a multiple of 8, or
GOINFER_METAL_FUSED_ATTENTION off), and a resident context reaches 32768 for an explicit -ctx or a guard-pinned load,
so metalResident.PrefillLast declines such a pass to the sequential path. TestPrefillExactAttnBound ties this to the
kernel source.
```

## gemmaPrefillResidScale

Moved from `metal/prefill.go` (the comment above `gemmaPrefillResidScale`) on 2026-10-09.

```text
gemmaPrefillResidScale is s for the families prefillResidScale applies to: the smallest power of two leaving 4x headroom
under f16's 65,504 over Gemma 3 4B's measured residual peak, 295,936 (its <bos> row after layer 31, the same on prompts
of 31 to 3,094 tokens and an image turn; S17's G-RS2 record).
```

## prefillLastImg.release

Moved from `metal/prefill.go` (the comment in `prefillLastImg`, above `scratch :=`) on 2026-10-09.

```text
C5: every buffer above is per-call scratch/uniform allocated onto the device ledger, which
ReleaseAll frees only at Close — so before this fix each PrefillLast leaked ~24 buffers
(~100–150 MB for a 7B; guF alone is Mpad*2I*2), ratcheting until the mustBuf OOM panic killed
serve (that panic is recovered only on the BuildResident path, not here). e.End() below
commits AND waits, so the GPU is finished with them by the time this returns — release each
at end of call. (r.uH / r.uKvDim / r.uHd are resident-owned and reused — deliberately NOT in
this list; releasing them would corrupt the decode path.)
```

## prefillLastImg.qkNormShared

Moved from `metal/prefill.go` (the comment in `prefillLastImg`, the qk_norm block) on 2026-10-09.

```text
Q heads only: a shared layer projects no K. The kernel maps a threadgroup to (row, head) by nH+nKV, so
nKV must be 0 here too, or rows shift and the "K" head normalizes the next row's Q (found by G-S9d).
```

## prefillLastImg.rope

Moved from `metal/prefill.go` (the comment in `prefillLastImg`, above the rope dispatches) on 2026-10-09.

```text
rope q, k (per-row positions) — bind the PER-LAYER RoPE table and window, exactly as decode
does (encodeTrunkInto), not the model-level r.invf/r.uWindow. For a mixed local/global-window
arch the global layers must see window=0, and each layer its own RoPE base; the model-level
bindings applied the local window (and one RoPE table) to every layer (audit M-09). Admitted
prefill archs have a uniform RoPE table (FeatPerLayerRoPE is not claimed), so L.invf equals
r.invf there — this is behaviour-neutral for them and correct for the mixed-window case.
```

## prefillScratchU16

Moved from `metal/prefill.go` (the comment above `prefillScratchU16`) on 2026-10-09.

```text
prefillScratchU16 allocates one of a pass's f16 scratch buffers, n halves, zero-filled. Metal fills a new buffer with
zeros itself (newBufferWithLength), so the pass no longer builds a zeroed Go slice of the same size and copies it in:
on the Qwen1.5-MoE slice that host work was 14.6 ms before a 512-token pass reached the GPU and 77.7 ms before a
2048-token one (TestDB02_expertMajorProbe). prefillScratchCopy restores the copy, for the test that compares the two.

THIS RELIES ON THE ZERO-FILL, and the zero-fill is a Metal property only: aikit's gpu.NewBufferLen* contract says Metal returns zeroed memory and CUDA returns
uninitialized memory, and that code shared across backends must not rely on zeros. These scratch buffers are sized to Mpad rows and the pass writes M of them, so the pad rows hold
the zeros Metal gave them. Metal-only code may do that; a CUDA port of the same pattern must zero the buffer explicitly (Queue.ZeroAsync) or it reads garbage.
TestPrefillScratch_zeroFilled pins the property on this backend.
```

## prefillScratch

Moved from `metal/prefill.go` (the comment above `prefillScratch`) on 2026-10-09.

```text
prefillScratch is prefillScratchU16 for any element type (R-20, docs/tasks/task-recompute-audit.md): the pass's f32
and u32 buffers that the router GEMM, the route kernels or the host fill before they are read (the expert-major MoE
branch's moeLogits, moeIdx, moeWgt, rowIdxBuf, rowWgtBuf), and xF, which the embeddings are converted straight into.
Same zero-fill reliance and the same copy arm, plus prefillR20Copy, which copies these alone so a test can time R-20
apart from the scratch ecafa0ae already moved.
```

## moeLayer.pool

Moved from `metal/moe.go` (the comment in `moeLayer`, above `pool`) on 2026-10-09.

```text
pool is non-nil when mo.paged: a bounded LRU slot pool + on-demand staging (expertpool.go),
generalized from the gemma4-only path (gemma4_moe.go) — same mechanism, no gemma4-specific
assumptions in expertPool itself, so this file supplies the stageFn instead of inventing one.
expGuW/expGuS/expDW/expDS stay zero-value when paged (the stacked all-E buffer is exactly the
11.96 GB/19.2 GB this exists to avoid).
```

## moeResident.paged

Moved from `metal/moe.go` (the comment in `moeResident`, above `paged`) on 2026-10-09.

```text
Synchronous paging (GOINFER_METAL_MOE_SLOTS=N>0): generalizes gemma4_moe.go's paging to this
generic MoE shape (Mixtral/Qwen/GLM/gpt-oss/qwen3_5_moe/qwen3_next) — same env var, same
mechanism, same expertPool. N must be >= k (a token's own top-k must fit); N==0/unset ⇒ all
experts resident (today's behavior, unchanged). slotIdx (M-11, audit-metal-2026-09-12.md) is
host-written each token with the pool ROW holding each routed expert, so the paged expert
GEMVs read the right row of the pool's contiguous buffer while rWgt is still indexed by the
selection slot — the reused gemv_w4a8_moe(_wacc[_bias]) kernels compute byte-identically to
the stacked path. Was idxZeros (always 0) back when each slot was its own single-expert
Buffer object; see gemma4_moe.go's gemma4MoeResident.slotIdx doc comment for the full "why".
```

## moeResident.indexSpaces

Moved from `metal/moe.go` (the comment in `moeResident`, above the two-index-spaces table) on 2026-10-09.

```text
TWO INDEX SPACES, DIVERGING ONLY WHEN PAGING IS ON — the same doctrine as cuda/resident.go's
expIdx / expertBiasIdx pair, and adopted here because Metal had the defect that pair exists
to prevent (C-09).
```

## moeResident.indexSpaces.shipped

Moved from `metal/moe.go` (the comment in `moeResident`, below the two-index-spaces table) on 2026-10-09.

```text
With paging off the two are the same buffer, so a site that binds the wrong one is correct in
every configuration anyone has run and wrong — silently, with plausible logits — in the one
configuration that needs it. That is exactly what shipped: the paged encoder passed idxZeros
to both roles, so gpt-oss's stacked expGuBias/expDBias tables were addressed by a constant
zero and every routed expert got expert 0's bias.
```

## errPagedExpertsNotInt4

Moved from `metal/moe.go` (the comment above `errPagedExpertsNotInt4` (the comment also held `buildMoE`'s opening)) on 2026-10-09.

```text
buildMoE builds the resident-level MoE state (pipelines, config, uniforms, scratch) from a
dense-shaped model that actually declares an MoE FFN. Returns nil when the model is not MoE,
and an error for an MoE variant this path does not implement (so BuildResident declines →
CPU fallback rather than running wrong).
errPagedExpertsNotInt4 is the expert cache's decline for experts it cannot stage (the option-path admission doc's
§4.3 finding 2, G-2). Paging copies each expert's int4 bytes into a slot as they are stored; a non-paged resident
re-quantizes int8 experts to int4 while it stacks them, which a per-token stage cannot afford. Checked before the
build, so it is a named decline rather than the panic buildMoELayer and buildGemma4MoELayer used to raise (a
recovered "metal build panicked").
```

## buildMoE.gemma4

Moved from `metal/moe.go` (the comment in `buildMoE`, above the `HasGemma4MoEResident` check) on 2026-10-09.

```text
Gemma 4's enable_moe_block (gemma4_text, 26B-A4B) sets arch.MoE, so MoEResidentParams reports
ok — but it is NOT the generic Mixtral/Qwen shape this path implements. It is a parallel
dense‖MoE FFN with gelu-tanh experts (gemv_w4a8_moe's epilogue is SiLU), a weightless router
pre-norm + learned [hidden] scale + per_expert_scale, and seven norms. Building it here would
run a wrong (SiLU, plain-router) MoE. Decline until the Metal gemma4MoeMLP path lands (9c
Step 5) so gemma4_text falls back to CPU rather than mis-running — same discipline as CUDA's
isG4MoE split (cuda/backend.go). Dense Gemma 4 has no MoE, so it returns nil,nil above and
admits normally; this only gates the MoE variant.
```

## buildMoE.pread

Moved from `metal/moe.go` (the comment in `buildMoE`, above the `GOINFER_MOE_PREAD` branch) on 2026-10-09.

```text
Stage experts by pread'ing their nibbles straight into the slot buffers instead of byte-copying
off the mmap — the refinement gemma4_moe.go measured (cold A/B: 1892→1488 ms/tok, 1.26×; major
faults 92.8→0.0/stage, the demand-fault page-in gone) applied to this shape. DEFAULT ON; needs a
.giw-mmap'd model, since the offsets are into that file. Re-opened once, shared across layers.
GOINFER_MOE_PREAD=0 opts out (the mmap byte-copy baseline, kept for the A/B). If the open fails
or the model isn't .giw-backed, buildMoELayer falls back to the byte-copy.

GOINFER_MOE_NOCACHE is deliberately NOT wired here: gemma4 measured it and DECLINED (no effect,
and its motivating evidence was an ordering confound — see buildGemma4MoEResident). Porting a
declined flag would re-open a settled question.
```

## buildMoELayer.paged

Moved from `metal/moe.go` (the comment in `buildMoELayer`, above the paged branch's `gw0`) on 2026-10-09.

```text
Paged: don't stack the experts (that is the multi-GB set this exists to avoid). Build a
bounded LRU slot pool + a stage fn reading expert e's W4A8 bytes from the layer's own
WeightMats on demand. Per-expert buffer sizes come from expert 0. Mirrors
buildGemma4MoELayer's paged branch exactly (byte-copy staging; no pread fast path here —
that was a later, separately-measured refinement on top of a working paged baseline).
```

## buildMoELayer.scales

Moved from `metal/moe.go` (the comment in `buildMoELayer`, above the paged `stage` func) on 2026-10-09.

```text
C-P01 (audit-metal-2026-09-30.md): each expert's f16 scales are read from its own WeightMats
(Int4ScalesF16), which a v14 metal or v15 .giw aliases from the mapping. This used to be a
build-time heap cache (N-20, which replaced a per-page-in f32→f16 conversion), holding the
same bits again (1361 MB on the Gemma 4 26B, measured 2026-10-02, T1.8).
```

## encodeMoEFFNWithX

Moved from `metal/moe.go` (the comment above `encodeMoEFFNWithX`) on 2026-10-09.

```text
encodeMoEFFNWithX records the MoE FFN for one layer, replacing the dense gate/up/swiglu/down
dispatches. post-attn norm → router logits → on-GPU top-k → per-selected-expert
gate|up/swiglu/weighted-down → optional shared expert. Value-independent (idx/wgt read at
kernel-execution time), so the encode-ahead executor still pre-encodes it. x is the
post-attention hidden state to route from (r.x for the decode path; a caller-owned buffer for
batched prefill, hence the WithX form rather than a fixed r.x read).

This is the NON-PAGED path: it reads the stacked all-E buffers (ml.expGuW/expGuS/expDW/expDS),
which stay zero-value once the layer is paged (see moeLayer's own comment on the pool field).
forwardLogitsMoEPaged never reaches here for a paged layer (it tears the layer into
encodeMoERouter + encodeMoEExpertsPaged around a host readback instead) — the panic below is a
chokepoint against every OTHER caller of encodeLayer (Forward, ForwardArgmax,
forwardHiddenNoHead's encodeTrunkInto) reaching a paged layer through the non-paged encoder and
silently computing off zero-value weights instead of failing (audit-metal-2026-09-12.md C-02).

FinishEncoding before the panic: e already has this layer's attention/mixer dispatches recorded
(encodeLayer calls this after encodeAttention/encodeDeltaNetMixer), and Metal asserts if a command
encoder is released without endEncoding — an uncommitted, never-`.End()`'d Encoder left for the Go
GC to finalize hits exactly that assertion (observed: "[_MTLCommandEncoder dealloc]: failed
assertion" on the first version of this guard). Ending encoding without committing discards the
partial work cleanly with no GPU execution and no side effect on r.x.
```

## moeKSlotsOn

Moved from `metal/moe.go` (the comment above `moeKSlotsOn` (the comment also held `encodeMoEExperts`'s doc)) on 2026-10-09.

```text
encodeMoEExperts runs the k selected experts out of the STACKED all-E buffer (ml.expGuW/expDW,
indexed by rIdx at kernel-execution time) plus the optional shared expert. The non-paged path.

dst is the down-projection's accumulate target — decode's own caller (encodeMoEFFN) always
passes r.x (the shared F32 residual stream), but G8's batched-prefill MoE row loop
(metal/prefill.go, docs/tasks/task-gpu-paths-2026-09.md) needs the SAME expert-loop math to
accumulate into an isolated F32 scratch buffer instead, since prefill's own residual (xF) is
F16 and these kernels are F32-only — added into xF's row by a SEPARATE small kernel afterward,
not by pointing these dispatches at xF directly (a type mismatch: `device float*` vs `half*`).
moeKSlotsOn selects D-P03's k-slot dispatches where the shape admits them. Off: PARKED 2026-10-04, bit-identical
but 1.019x on the Qwen1.5-MoE slice's token (21.5 us per MoE layer at k = 4), the owner's park zone for speed
(docs/tasks/task-metal-audit-2026-10.md, "D-P03"). Tests turn it on for the k-slot arm.
```
