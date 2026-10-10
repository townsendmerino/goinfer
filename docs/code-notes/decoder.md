# decoder: notes moved out of code comments

History, measurements and open work that used to sit in the comments of package `decoder`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## isWebGPUBackend

Moved from `decoder/residency.go` (the comment above `isWebGPUBackend`) on 2026-10-10.

```text
isWebGPUBackend reports whether a Backend.Name() string names the webgpu backend. NOT a plain
equality check: gpu.webgpuBackend's real Name() is "webgpu:" + the underlying GPU surface
(e.g. "webgpu:metal" on macOS, "webgpu:vulkan" on Linux) — every OTHER backend's Name() is a
bare string ("cpu", "cuda", "metal"), so this asymmetry is specific to webgpu.

Found while wiring G7 part 2 (docs/tasks/task-gpu-paths-2026-09.md): a NEW `== "webgpu"` check
written for that row silently never matched on this exact Mac, which led straight to a
PRE-EXISTING bug this asymmetry already caused — DecodePath()'s own `case be != "webgpu":`
(below) has never matched a real webgpu backend on any platform, so a declined webgpu load has
always been reported through the cuda/metal branch's wording ("has no staged decode path" —
false; webgpu does have one) instead of the webgpu-staged branch meant for it. Fixed here, not
papered over with another exact-match string.
```

## ResidentForward.ForwardN

Moved from `decoder/residency.go` (the comment above `ForwardN` in `ResidentForward`) on 2026-10-10.

```text
ForwardN runs K tokens at consecutive positions startPos..startPos+K-1,
appending K KV positions and returning K logit rows — the batched verify for
speculative decoding. Causal: row i attends to positions [0, startPos+i].
Bit-identical to K sequential Forward calls (TestResidentForwardN_parity), so
no implementation may change numerics to batch. nil/empty embeddings ⇒ no-op.

"Batched" is an amortization OPPORTUNITY, not a structural guarantee every
backend takes; each backend reports at load which it took (VerifyPathReporter).
CUDA's prefillReady path runs the whole batch in one weight-stationary pass (one
command submission) when prefillCore admits the arch, falling back to a per-token
sequential loop otherwise (DeltaNet, non-int4 projections, and the geometries
prefillCore declines). Metal's ForwardBatch encodes the whole batch layer-major into
ONE command buffer (since a1640a6a, 2026-09-16 — it was the per-token loop before,
N-26), falling back to the per-token loop only for paged MoE. webgpu runs the K steps
in one submit. Callers should not assume "K rows in" implies "one command buffer
out" for every backend.
```

## ResidentGreedy

Moved from `decoder/residency.go` (the comment above `ResidentGreedy`) on 2026-10-10.

```text
ResidentGreedy is an optional capability on a ResidentForward: compute the token's greedy
argmax on-device and hand back just the id, skipping the full-logits readback (594 KB per
token at a 151936 vocab — ~1.7% of a 1.5B token, ~6.5% of a 0.5B one, measured on CUDA).
The decode loop uses it ONLY when Sampler.ArgmaxEquivalent() OR Sampler.GreedyEquivalent()
(the `top_k=1` shape, P1) holds and no LogitProcessor is set,
so the emitted tokens are identical to the logits path. Backends may skip implementing it.
```

## Prefiller.PrefillLast

Moved from `decoder/residency.go` (the comment above `PrefillLast` in `Prefiller`) on 2026-10-10.

```text
ctx is checked at whatever granularity the backend can honour — between passes, and inside
them where the backend's own loop allows. It exists because a batched prefill is the ONE
place where a cancelled request keeps a GPU busy with nothing watching: the sequential
fallback checks ctx.Err() per token (G18, "an abandoned client leaves the whole prompt
streaming through the device"), and a backend that ingests the prompt in one pass has no
equivalent unless it is handed the context. Cancellation returns ctx.Err(); the caller
treats that like any other decline and stops, rather than falling back to the sequential
loop it was also cancelling.
```

## ResidentHiddenLast

Moved from `decoder/residency.go` (the comment above `ResidentHiddenLast`) on 2026-10-10.

```text
ResidentHiddenLast is an OPTIONAL ResidentForward extension: ingest a whole sequence at
positions startPos..startPos+len-1 and return the LAST position's hidden state AFTER the
model's final norm — the resident twin of Prefiller, but stopping before the LM head instead
of after it (decoder/embed.go's HiddenLast never needs logits, and the head is the single
most expensive matmul in a forward). Used by HiddenLast/G4 (docs/tasks/task-gpu-paths-2026-09.md):
an embedding request on a GPU box otherwise runs the whole text decoder on the CPU even when
the same arch decodes resident, the exact defect G2 documents for image turns.

startPos is always 0 for HiddenLast's callers today (a fresh KV per sequence, no prefix
reuse), but the parameter mirrors Prefiller's so a backend can share its prefill-chunking
scaffolding. Bit-identical to the CPU path is the bar (embed.go's own doc comment); a backend
whose batched forward is NOT bit-identical to its own sequential one (Metal was, pre-gate —
see metal/backend.go's metalFastPrefillEnabled; default-on since 2026-09-09, above
metalFastPrefillFloor, 16 tokens since A-P02) must implement this some other way (a
per-token sequential forward that stops before the head) rather than reuse a declining
Prefiller, or must not implement this interface at all.
```

## PrefillPathReporter

Moved from `decoder/residency.go` (the comment above `PrefillPathReporter`) on 2026-10-10.

```text
PrefillPathReporter is an OPTIONAL Prefiller extension: report at LOAD time whether the batched
prefill will actually be taken for THIS model, and when it won't, why and what that costs. The
Prefiller contract declines per call (arch/geometry/quant), and generateInto's fallback is silent
by design — correct, but a decline is a large, invisible TTFT regression: a cuda int8int8 model
takes the sequential per-token prefill (the batched GEMV is int4-only), measured 9× slower TTFT
and 20× the CPU on a 300-token prompt. Backends implement this so serve can state the resolved
prefill path at startup instead of leaving it to be discovered under load.

reason is a human-readable phrase naming the ACTUAL condition and its cost, not "declined" —
e.g. "batched prefill requires int4 projections (int8int8 at layer 0) — ~9× slower TTFT".
```

## PrefillTailExact

Moved from `decoder/residency.go` (the comment above `PrefillTailExact`) on 2026-10-10.

```text
PrefillTailExact is an OPTIONAL Prefiller extension: true when the backend's PrefillLast over a SHORT suffix that
continues a prefix it prefilled computes, bit for bit, what one PrefillLast over the whole prompt computes for those
positions, and its routing between its prefill lanes depends on the whole prompt's length, not the suffix's. Then a
reused prompt (resident_reuse.go) can hand the batched path a suffix of any length and get the cold run's bits.

Without it residentPrefillSeed offers only suffixes of 8 tokens or more, so a repeated prompt (a 1-token suffix) ran
its last position through decode while the cold run had computed it in the batched pass, which is not bit-identical
to decode: on Metal the warm reply differed from the cold one some tens of tokens in (2026-10-04, found as the
"MC3 pass identity defect" in docs/tasks/task-metal-audit-2026-10.md's A-P02 result).
```

## ResidentPrefillKernelFloor

Moved from `decoder/residency.go` (the comment above `ResidentPrefillKernelFloor`) on 2026-10-10.

```text
ResidentPrefillKernelFloor is the optional capability of a resident whose batched prefill runs one class of
kernels below a whole-prompt length and another, not bit-identical, from it (Metal: decode rows below its
floor, f16 MMA from it). PrefillKernelFloor is that length, or 0 when one class runs at every length.
Chunked prefill (mc3Prefill) reads it so no chunk straddles the boundary the whole prompt would not: a first
chunk below the floor ran on the other class, and the reply then depended on whether another generation was
decoding (option-path admission finding 3, 2026-10-08). A narrower contract than ResidentFastPrefill's,
which prefix reuse also reads; a backend can report this one without changing what it reuses.
```

## Model.decodeRunnerDecline: the webgpu Nemotron MoE override

Moved from `decoder/residency.go` (the comment above `eligible = true` in `decodeRunnerDecline`) on 2026-10-10.

```text
G7 part 2 (docs/tasks/task-gpu-paths-2026-09.md): decodeRunnerEligible's own Nemotron branch
hard-declines ANY MoE-block variant for every backend — a real crash-risk guard, since
cuda/metal's own per-layer block-kind switches have no MoE case at all and would
otherwise nil-dereference on the first token. webgpu now implements it for real
(gpu/residency.go's switch has a case 3, gpu/decoderunner.go a real nemoKMoE dispatch),
so this is a narrow, backend-specific override — NOT a general "trust the backend"
escape hatch. cuda/metal still decline exactly as before.
```

## Model.decodeRunnerDecline: the Nemotron-H precision policy

Moved from `decoder/residency.go` (the comment above the Nemotron-H check in `decodeRunnerDecline`) on 2026-10-10.

```text
Nemotron-H resident is DEFAULT-on at int4 — characterized benign vs f32 (92.5% greedy /
99.6% top-2 agreement, perplexity 1.677≈1.695, KL 0.058; the ~7.5% disagreements are all
at near-tied positions, int4 picking f32's #2, zero confident-token errors — see
docs/completed/nemotron-resident.md). int8 (unmeasured on 8 GB; fits ≥12 GB) stays OPT-IN behind
GOINFER_SSM_RESIDENT. Other resident families are unchanged.
```

## Architecture.decodeRunnerEligible: gemma4

Moved from `decoder/residency.go` (the comment above the `gemma4` case in `decodeRunnerEligible`) on 2026-10-10.

```text
Gemma 4 is admitted UNCONDITIONALLY as of the Check-A backfill. It sat behind
GOINFER_GEMMA4_RESIDENT through the Split-A/B bring-up (granite's GOINFER_SSM_RESIDENT
precedent), and that flag outlived its purpose in a way worth recording: it had become
load-bearing by accident. Every real Gemma-4 checkpoint reaching the resident path was
compared only against ITSELF (cache on/off, graphs on/off), so the flag was the only thing
standing between users and a forward whose numerics no gate asserted at real width.

It comes off because that gate now exists and passes, not because the flag was tidy to
remove: cuda.TestGemma4MoEScaled_residentParity (MoE, real hidden/moe_inter/head geometry,
scales transplanted from the real 26B) and TestGemma4DenseScaled_residentParity (dense,
real head dims) both assert the calibrated CPU curve. Do not re-gate a family on an env
var without a gate behind it; the flag reads as caution and functions as a coverage hole.

Split B lands the enable_moe_block parallel dense‖MoE FFN (arch.MoE != nil) on its OWN cuda
path (gemma4MoeMLP) — the CUDA build routes it around the generic MoE checks via
HasGemma4MoEResident. Both dense and MoE fall through to the common checks below (softcap
is handled by the per-backend feature gate, so WebGPU still declines on its own terms).
E-models (PLE) decline via the feature gate regardless — TestGemma4EModel_realDeclinesResident.
```

## Architecture.decodeRunnerEligible: lfm2

Moved from `decoder/residency.go` (the comment above `return false` in the `lfm2` case of `decodeRunnerEligible`) on 2026-10-10.

```text
LFM2/LFM2.5: own forward (runLayersLFM2), not bridged. Declining HERE and not only
via FeatShortConv is deliberate — the two gates answer different questions and this
one is the honest answer to THIS question. decodeRunnerEligible is an ARCH-SHAPE
predicate, and 22 of 30 layers run a gated short convolution with a rolling window
that no uniform-layer runner can express at all; that is a property of the family,
not of any backend's kernel coverage.

It also stops the capability matrix publishing a falsehood: the GPU-resident column
is generated from exactly this predicate, and without this case LFM2 falls through
every switch arm to ropeResidentCompatible(), whose local/global tables are equal
length (one base), so it returned TRUE. The matrix then read "GPU-resident: yes" for
a family no backend can run — caught 2026-08-31 by reading the generated row instead
of trusting the column name, which is the G10 question about this column's semantics
arriving as a concrete case.
```

## Architecture.decodeRunnerEligible: nemotron

Moved from `decoder/residency.go` (the comment above the `nemotron` check in `decodeRunnerEligible`) on 2026-10-10.

```text
Nemotron-H resident (dense squared-ReLU hybrid): single-op-per-block Mamba-2 / NoPE-GQA /
relu² MLP, reusing the granite SSM engine. Arch-eligible; the int4-default-vs-int8-opt-in
precision gate is applied at the Model level (DecodeRunnerEligible).

Nemotron 3 Nano adds a FOURTH block kind (MoE FFN, arch.MoE != nil — plain Nemotron-H never
sets it) that no GPU resident builder knows about yet: gpu/residency.go's per-layer switch on
NemotronBlockKind has cases 0/1/2 (mamba/attn/mlp) and NO default — an unhandled kind 3 would
silently leave that layer's op buffers nil rather than erroring, a crash-or-garbage risk, not
a clean decline. Decline here instead, at the one predicate every backend's admission funnels
through, until a GPU backend actually implements the MoE block (verified against its
dispatch, not assumed from "it's just another case").
```

## Architecture.decodeRunnerEligible: feature gates and MLA

Moved from `decoder/residency.go` (the comment above the `mla` check in `decodeRunnerEligible`) on 2026-10-10.

```text
FFN / norm / head constraints common to both attention paths. Sandwich norms (Gemma's
4-norm block), logit softcap, NonGatedMLP, LearnedPosEmbed and OutBias are all
representable now — a backend that hasn't implemented one declines via the feature gate
(FeatSandwichNorm / FeatAttnLogitSoftcap / FeatFinalLogitSoftcap / FeatNonGatedMLP /
FeatLearnedPos / FeatOutBias), not here, so this stays an ARCH-shape predicate and the
per-backend answer lives in one place (decoder/features.go). Softcap was moved OUT of this
decline in 9a-P2 for the same reason Gemma 4 needed it: a per-backend capability blocked
here for everyone is a family blocked for a backend that already implements it.

NonGatedMLP/LearnedPosEmbed/OutBias's hard decline came off 2026-08-18 once Metal built the
three matching kernels (layernorm_quant, act_quant, gemv_w4a8_sa_bias_resid/
gemv_w4a8_resid_bias) for GPT-2 — the only family this reaches in practice: Nemotron-H's
NonGatedMLP and gpt-oss's OutBias both short-circuit at the switch above before ever
reaching here, and no arch sets LearnedPosEmbed but GPT-2. GPT-2 has no RoPE at all
(finalizeRoPE returns early on LearnedPosEmbed, leaving BOTH ropeInvFreqLocal/Global nil),
so it falls through to ropeResidentCompatible() below and passes vacuously (0==0) — no
extra bypass needed for that check either.
MLA (DeepSeek/Kimi) runs its own latent-attention path on the resident runner
(Lever C4 — gpu/mla.go), so the standard GQA/RoPE/QK-norm/sliding-window checks
don't apply; its decoupled RoPE rides a separate qk_rope slice, not HeadDim.
```

## Model.VFromKResident

Moved from `decoder/residency.go` (the comment above `VFromKResident`) on 2026-10-10.

```text
VFromKResident reports whether layer i is a K=V layer (attention_k_eq_v): it carries NO
v_proj — V is v_norm(the raw k_proj output). Gemma 4 sets this on its GLOBAL layers (12B/26B;
off for E2B). This is arch.KVShared (the per-layer attention_k_eq_v), deliberately distinct
from SharedKVLayers (the unrelated cross-layer KV reuse the E-models use).

Two sources, OR'd. The config flag (attention_k_eq_v, from config.json) and the per-layer
LayerWeights.VFromK the loader sets. The GGUF loader has no config flag to read — a GGUF carries
no attention_k_eq_v — so it marks a layer by the ABSENCE of attn_v.weight (gguf.go), leaving the
config flag false. Reading only the config flag made every GGUF-derived Gemma 4 K=V layer look like
an ordinary QKV layer with an empty V, so a resident backend fused a zero-row matrix and declined
(Metal: `int4Concat weight kind "" not int8 or int4`), while a config.json/safetensors-derived
bundle of the same model went resident. The per-layer flag is also what a .giw stores.
```

## Model.RopeInvFreqLayer

Moved from `decoder/residency.go` (the comment above `RopeInvFreqLayer`) on 2026-10-10.

```text
RopeInvFreqLayer returns layer i's RoPE inverse-frequency table as float32 — the global
or local table per the layer's attention type (Mellum YaRN-on-global vs default-local).

G5 (docs/tasks/task-gpu-paths-2026-09.md), FeatNoPE: a NoPE layer (SmolLM3's no_rope_layers,
Cohere2's every-Nth global layer — arch.isNoPELayer) gets an all-ZERO table instead of the
real one, same length. The resident rope kernels compute cos=cos(pos·invFreq)·mscale,
sin=sin(pos·invFreq)·mscale (cuda/gemv_fwd.cu, metal/kernels.go); at invFreq==0 that is
cos=mscale, sin=0 — the identity rotation ONLY when mscale==1 (RopeMscaleLayer). Every family
that sets layerNoPE today (SmolLM3) also has no YaRN scaling on those layers, so this holds;
it is NOT a general "any NoPE layer is safe" claim, and a future family combining YaRN with
NoPE would need this revisited (a parity test would catch it — a wrong scale is not a
discrete failure like a MoE routing bug, but a real forward divergence at every position).
This is a RESIDENT-ONLY accessor, so the CPU forward path (attention.go/forwardn.go, which
skip the ropeAt call entirely for a NoPE layer) is untouched by this.

G5, Olmo Hybrid (found bringing up FeatPostOnlyNorm/FeatQKNormWhole): a WHOLE-MODEL no-RoPE
arch (rope_theta absent — arch.NoPositionEncoding, layerNoPE unconditionally true on every
layer) never populates ropeInvFreqGlobal/Local at all — finalizeRoPE's own early return, since
there is no base to compute a table from — so ropeInvFreq(i) is empty for EVERY layer, not
just the ones isNoPELayer would call NoPE anyway. The len(inv)-sized zero table above degrades
to a ZERO-LENGTH one in that case, not a same-width all-zero one, and a resident backend that
uploads it unconditionally (cuda/backend.go's L.invF = r.up32(h.invFreq), no length guard)
hits a 0-byte device allocation, which this driver treats as a hard error — measured directly:
"cuda: device allocation failed (typed-len, 0 bytes): cuda: invalid length" bringing up Olmo
Hybrid's full-attention layer, the first family combining whole-model NoPE with reaching this
resident code path at all. Metal happened not to crash on this (its build code already has an
unrelated len(invf)>0 guard, originally for GPT-2's LearnedPosEmbed case, that incidentally
also skips building L.invf here) — but that leaves Metal's rope dispatch reading a zero-VALUE,
never-written Buffer rather than a deliberately-zero one, which is the kind of "works by
accident" this fix removes for both backends: rotaryDim()/2 is the width every resident rope
kernel launch actually expects (rhalf), independent of whether this model has any real RoPE
table to inherit a width from.
```

## Model.AttnScale

Moved from `decoder/residency.go` (the comment above `AttnScale`) on 2026-10-10.

```text
AttnScale is the attention softmax q·k multiplier the resident backends (CUDA/Metal/WebGPU)
apply. It returns the registry-RESOLVED arch.AttnScale, which is 1/√headDim for the common case
but deliberately not for several archs: Gemma-3 uses query_pre_attn_scalar^-0.5 (== headDim for
1B/4B/12B but hidden/heads for 27B), Granite uses attention_multiplier, and MLA uses the qk
head-dim. Hardcoding 1/√headDim here (the old behaviour) silently mis-scaled every attention
logit on those — invisible on the small parity fixtures (where the scalar == headDim) but wrong
on 27B/Granite. Falls back to 1/√headDim only if an arch left AttnScale unset (M19).
```

## Model.withResidency: the Nemotron MoE decline

Moved from `decoder/residency.go` (the comment above the Nemotron MoE decline in `withResidency`) on 2026-10-10.

```text
G7 (docs/tasks/task-gpu-paths-2026-09.md): Nemotron 3 Nano / 3.5 Lightning's fourth block
kind (MoE FFN) has no GPU resident implementation on cuda/metal — decodeRunnerEligible's
own nemotron branch already declines this for them (`a.MoE == nil`), but that predicate
returns a bare bool, so the generic "arch is not eligible" message below would otherwise
hide WHY. Checked here, ahead of the generic call, so DecodePath/`serve check` name the
actual gap instead of a family-agnostic decline. webgpu is EXCLUDED from this decline
(part 2, same doc): it implements the MoE block for real now
(gpu/residency.go/gpu/decoderunner.go's nemoKMoE case) — DecodeRunnerEligible's own
backend-aware override admits it, so falling into this specific-reason branch would be
wrong for webgpu (it isn't declining here at all).
```

## Model.withResidency: a declined BuildResident

Moved from `decoder/residency.go` (the comment above the decline switch in `withResidency`) on 2026-10-10.

```text
Falls back silently for correctness (a decline must never be fatal mid-load), but the
REASON is kept: on a GPU backend this is the whole forward moving to CPU, which is the
same silent-regression class as the prefill decline one layer down. A decline carries its
reason as a *ResidentDeclineError; any other non-nil err is a driver or device failure —
`--backend cuda` on a box with no usable GPU lands here, since the cuda factory always
succeeds and the device is only touched at build time.
```

## Model.withResidency: releaseHostMemory

Moved from `decoder/residency.go` (the comment above `releaseHostMemory()` in `withResidency`) on 2026-10-10.

```text
BuildResident's host-side work — the CUDA backend packs every layer's weights on the Go heap before it
uploads them, and stages the expert stacks in pinned memory — is garbage the moment it returns, but Go hands
freed heap back to the OS only when its background scavenger gets round to it. MEASURED 2026-09-20 on the
real gpt-oss-20b (--backend cuda --moe-cache-experts): RSS sat at ~39 GB for ~5 minutes after the load
finished, then fell to ~22 GB. Release it now, once, at load time: a resident model's whole point is that the
host copy is no longer needed.
```

## Model.DecodePath: the cuda and metal staged case

Moved from `decoder/residency.go` (the comment above `req := be` in `DecodePath`) on 2026-10-10.

```text
R9 (docs/measurements/cold-user-2026-09-06-nobara-pc.md, corrected on review):
"cuda-staged (int4)" on an 8 GB card sat at the idle VRAM baseline (464 MiB,
unchanged) for a full request sampled at 1 Hz. The first pass here found the int4
half (matmul()/weightmat.go's int4 branch calls the CPU integer W4A8 kernel
unconditionally, no backend parameter at all) and wrongly generalized that
int8/int8int8/f32 "reach the backend" on every backend. They do not on cuda or
metal: both backends' own Backend.MatmulBT implementations are themselves bare CPU
calls (cuda/backend.go, metal/backend.go — no device dispatch), and NEITHER
implements QuantBackend at all, so the int8 branch's `be.(QuantBackend)` assertion
fails for them too and falls to the same CPU kernel int4 uses. There is no quant at
which cuda's or metal's staged path ever reaches the device — "cuda-staged"/
"metal-staged" were never a real path, so this reports it the same shape
BackendSummary already uses for a backend that failed to build at all: it is the
same kind of claim (requested vs. actually executing), and hardware-matrix.md
already only ever says "resident" or "CPU" per cell — this makes the banner agree
with that page instead of inventing a third state it doesn't have.
```

## residentQuantLabel

Moved from `decoder/residency.go` (the comment above `residentQuantLabel`) on 2026-10-10.

```text
residentQuantLabel is DecodePath's resident-quant string, split out so it is testable without a
live Model (see decoder/staged_device_note_test.go) — same shape as declinedToCPUReason below.

G10 (docs/tasks/task-gpu-paths-2026-09.md), widened by M-15... M-25 (audit-2026-09-10): Metal has
no int8 GEMV kernel at all — ANY Int8()-kind weight is silently re-quantized to W4A8 (int4) at
resident-build time (metal/model.go's int4Buf falls back to w.Int8() whenever int4DirectWords
finds no already-int4 nibbles), not just under `--quant int8int8`. `--quant int8` produces the
same Int8-kind WeightMat and hits the identical fallback; `--quant int4mix`'s int8-kind tensors
(the ones it deliberately did NOT keep at int4) do too, so even a "mix" ends up effectively
uniform int4 on this backend. f32 is NOT affected: an f32-kind WeightMat has neither int4Buf
fallback available (int4DirectWords finds no nibbles, w.Int8() finds no q8) and int4Buf's own
call sites panic rather than silently proceed, so this backend never actually produces a
resident build to mislabel in the first place — it errors/declines well before DecodePath would
print anything. Every (backend, quant) pair outside these three metal cases is unaffected and
echoes the requested quant string exactly as before.
```

## autoMetalPrecision

Moved from `decoder/residency.go` (the comment above `autoMetalPrecision`) on 2026-10-10.

```text
autoMetalPrecision is why a model stays on the CPU when -backend auto chose metal, "" when it may go resident. A
backend the user did not name must not change the precision they loaded at, such as the int8int8 bundle a
model-included goinfer-chat carries (R17, docs/tasks/task-first-hour.md). Metal runs int4 resident, and since slice 1
of docs/tasks/task-metal-int8-2026-10.md dense int8int8 natively; it re-quantizes int8, int4mix and the int8int8
models native int8 does not cover (MoE, DeltaNet) to int4, and takes no f32. So int8int8 may try the resident here
(slice 5: gate S-auto passed, 2026-10-04), and autoMetalKeepsPrecision checks afterwards that it ran natively; the
others stay on the CPU. A named -backend metal still runs them re-quantized.
```

## stagedDeviceNote

Moved from `decoder/residency.go` (the comment above `stagedDeviceNote`) on 2026-10-10.

```text
stagedDeviceNote names when webgpu's own staged path — the only one that has one, see
DecodePath above — still reaches no device for a given quant, so the path name alone does not
overclaim device use.

WebGPU's Backend.MatmulBT and QuantBackend.MatmulW8A8 (gpu/backend.go) do real device work in
the staged path, unlike cuda/metal. Its int4 dispatch, QuantBackend4.MatmulW4A8 (G6, db81b08),
covers M=1 only: decode runs int4 on the device, and a multi-token prefill declines to the host
W4A8 kernel because this backend has no int4 GEMM. So int4's note splits decode from prefill,
and so does int4mix's, whose FFN tensors are int4 (its attention tensors are int8 and reach
QuantBackend). f32/int8/int8int8 DO reach webgpu's device and get no note. Until
audit-2026-09-10 G-13(a) the int4 note said no GPU dispatch existed at all, which db81b08 had
made false, and TestStagedDeviceNote pinned the false text.
```

## Model.PrefillPath

Moved from `decoder/residency.go` (the comment above `PrefillPath`) on 2026-10-10.

```text
PrefillPath reports how a long prompt will actually be ingested — one batched pass, or the
sequential per-token loop — plus the reason when it is sequential. It is the load-time answer to
a decline that generateInto otherwise takes silently: the batched prefill is an OPTIONAL backend
capability that falls back per call, so a model can lose a 9× TTFT (measured: cuda int8int8,
300-token prompt, 1.73 s vs 0.19 s) without emitting anything. serve prints this at startup and
-require-backend refuses to start on a decline.

Both halves are covered: the resident path asks the backend (PrefillPathReporter), and the
staged/CPU path reports canBatchN, which excludes the families with their own sequential forward
(Gemma 4, qwen3_5, Mamba-2 hybrids, MLA, Llama 4, gpt-oss).
```

## Model.PrefillPath: the CPU "batched" wording

Moved from `decoder/residency.go` (the comment above the CPU "batched shape" return in `PrefillPath`) on 2026-10-10.

```text
The word "batched" here names the SHAPE (weights stream once per prompt at
M=K instead of once per token), not a throughput claim. Measured on an M1
Pro, dense 1.5B: the CPU prefill path is single-threaded and its best
observed rate is the same order as the model's DECODE rate, so a reader who
hears "batched" as "fast" is being misled by this string (queue G17; the
missing parallelism is G16). Say what it is and what it is not.
```
