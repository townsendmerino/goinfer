# metal: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `metal`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## TestDumpLogitsForBisect

Moved from `metal/logit_dump_test.go` (the comment above `TestDumpLogitsForBisect`) on 2026-10-09.

```text
TestDumpLogitsForBisect writes the resident GPU logits for a fixed 4-token sequence to
$GOINFER_DUMP_LOGITS as raw little-endian float32. It exists to back the 9c Step-1
"byte-identical" claim with an actual BITWISE diff (parent commit vs HEAD), not just a
cosine-vs-CPU threshold: run it on the parent, run it on HEAD, `cmp` the two files. It uses
only BuildResident + ForwardEmb + EmbedResidentForTest — API identical across the refactor —
so the same source compiles and runs on both commits. Skips unless the env var is set.
```

## TestSetAdapter_partialBindErrorReleasesBuffers

Moved from `metal/lora_bind_leak_test.go` (the comment above `TestSetAdapter_partialBindErrorReleasesBuffers`) on 2026-10-09.

```text
TestSetAdapter_partialBindErrorReleasesBuffers gates C-04 (audit-metal-2026-09-12.md):
SetAdapter's conversion loop used to return immediately on the first bad projection, leaving
every earlier layer's already-converted device buffers (A/B plus their uniform buffers) on the
device ledger — referenced by nothing (r.loraLayers/r.loraCached are never set on an error
return), so they leaked until Close. Every failed bind attempt — a bad rank is the easy way to
trigger it, but any mid-loop error does — added to the leak.

llama-tiny (4 layers) lets layer 0 convert cleanly before layer 1's invalid rank fails the bind,
exercising the actual partial-progress path rather than failing on the very first projection.
```

## TestLoRAResidentParityMetal.floor

Moved from `metal/lora_resident_parity_test.go` (the comment above `TestLoRAResidentParityMetal.floor`) on 2026-10-09.

```text
N-14/N-52 (audit-metal-2026-09-12.md, audit-2026-09-10.md): this floor is far looser than
what a correct bind actually measures on this machine — 0.999969 here, 0.998835 on the
armed-executor variant below (both this session's real runs, not a single cherry-picked
number) — so a bug that drops one of the seven per-layer projections could plausibly still
clear 0.95 if that projection's contribution is small relative to total variance; the
vacuousness check below only proves "some effect survives", not "every targeted projection
fired". NOT tightened here: two single-machine runs a few thousandths apart is not enough to
pick a real "measured floor minus noise" number with confidence, and a mis-set tight floor
risks flaking CI on legitimate cross-machine/quantization variance — parked per N-52's own
recommendation, not rejected.
```

## TestMC3Chain_newcomerJoinsLongPrompts

Moved from `metal/mc3_chain_long_test.go` (the comment above `TestMC3Chain_newcomerJoinsLongPrompts`) on 2026-10-09.

```text
TestMC3Chain_newcomerJoinsLongPrompts and TestMC3Chain_stalledConsumerLongPrompts are the two TestMC3Chain_ scenarios
with prompts long enough for the batched pass (100/80 and 90/70 tokens) instead of the step. Their alone runs are
cold and their joint runs reuse the slot the alone run left, so they also pin warm-against-cold identity: before
decoder.PrefillTailExact a warm prompt re-ran its last position through decode where the cold one had used the pass,
and both generations diverged tens of tokens in (A at token 42 of 160, B at 47 of 64). That was first read as an MC3
batching defect; batching is bit-identical (the joint run on a fresh model matches), the cold/warm route was not.
```

## mc3ChainFixture

Moved from `metal/mc3_chain_test.go` (the comment above `mc3ChainFixture`) on 2026-10-09.

```text
The greedy and sampled chains (C-B01 / C-P02) under MC3: a generation alone on the batcher runs the chain with the
resident held across its tokens (decoder's holdSolo), and hands the resident back the first token anyone else wants
it. Before this, every generation on a model with more than one KV slot took an MC3 place, the chains required
mc3 == nil, and serve (2 slots by default on Metal) never ran them (the post-merge peer read, 2026-10-03). These
tests drive production's Generate with the batcher on, which is where that gap hid: the chains' own A/Bs ran with
no batcher at all.
```

## TestMC3Step_qwen3BitIdentical

Moved from `metal/mc3_step_test.go` (the comment above `TestMC3Step_qwen3BitIdentical`) on 2026-10-09.

```text
TestMC3Step_qwen3BitIdentical (E-P07): a Qwen3-shaped model (per-head QK-norm) runs the batched step, and its rows
equal production's single-token forward bit for bit, as TestMC3Step_bitIdentical checks for the qwen2 shape. On the
generated qwen3 fixture by default; under GOINFER_METAL_MC3=1 the identity suite runs on whatever checkpoint
GOINFER_METAL_MC3_MODEL names (the real Qwen3-0.6B read 0 differing values, 2026-10-04).
```

## TestMC5_prefillChunkInvariance.sizes

Moved from `metal/mc5_chunk_invariance_test.go` (the comment above `TestMC5_prefillChunkInvariance.sizes`) on 2026-10-09.

```text
100 and 77 start chunks off the steel kernel's 32-row tiles; a prefix-reuse turn's startPos is arbitrary too. The
aligned sizes cannot see a bug at a tile edge, which shows the same way whole and chunked: with the kernel's causal
limit moved one key, every aligned size still matched bit for bit, and 100 and 77 differed from the first chunk
boundary on (2026-10-01, F-G02). 512 is serve's default chunk. 81 leaves a 28-token tail at position 972: the
step route (E-P01, promptStepOK) must not take a short chunk above the floor, or the tail runs decode's numerics
and differs from the whole pass (measured with the route open there: all 20480 logits).

16, 32 and 48 cross A-P01's GEMM tile selector (gemmTile: 32-token tiles at <= 32 rows, 32-feature tiles for the
narrow GEMMs at <= 64 rows), so their chunks run the smaller tiles where the whole pass runs 64 × 64. Their first
chunk is 64 tokens: a chunk that ends below the fast-prefill floor never reaches the batched pass (it ran
sequentially before E-P01 and on the step kernels since, decode's numerics either way), so it cannot equal it.
```

## TestMellumResidentParity

Moved from `metal/mellum_real_test.go` (the comment above `TestMellumResidentParity`) on 2026-10-09.

```text
TestMellumResidentParity is G11's Metal half — the real-weight successor to the abandoned
synthetic-random-weight attempt (G10, docs/queue-correctness.md). G10 declared FeatRopeMscale
for Metal to unblock gpt-oss's YaRN, which as a documented side effect also admits Mellum onto
the Metal resident path with zero end-to-end validation there. decoder/mellum_slice_test.go
(tag realckpt) already proved goinfer's own CPU forward matches the real HF reference on a
REAL 4-layer weight slice (argmax 417, cosine 1.00000000) — same discipline as
TestGPT2ResidentParity/TestGptOssResidentParity, just pointed at the sliced checkpoint
directory instead of a small full model. Regenerate the slice per docs/queue-correctness.md
G11 if it's absent (4 GB, gitignored — only the golden ships).
```

## TestMinistral3ResidentSmokeMetal

Moved from `metal/ministral3_resident_parity_test.go` (the comment above `TestMinistral3ResidentSmokeMetal`) on 2026-10-09.

```text
TestMinistral3ResidentSmokeMetal is G5's FeatAttnTemp row (docs/tasks/task-gpu-paths-2026-09.md)
smoke gate: the model actually goes resident and produces finite, non-degenerate output when
the post-RoPE query scale is genuinely exercised (testdata/ministral3-tiny's
AttnTempOrigMaxPos=8, so a 32-token run steps through four distinct scale values, past the
identity-at-short-prompts trap AttnTempBeta's own comment warns about).

This is DELIBERATELY NOT a resident-vs-CPU cosine floor — same finding as G5 row 1's
smollm3-tiny (metal/smollm3_resident_parity_test.go's TestSmolLM3ResidentSmokeMetal has the
full writeup). MEASURED here too, not assumed: with the real fix and with it force-disabled
(qTempScale always 1, i.e. the attn-temp scale silently dropped on every position), the worst
cosine against the CPU reference over 32 tokens was 0.96446 and 0.96525 respectively — a
~0.0008 spread, indistinguishable from noise. testdata/ministral3-tiny is ALSO seeded/synthetic
(scripts/pin_ministral3_tiny.py), so despite its own test file's comment about deliberately
exercising AttnTempOrigMaxPos and a real YaRN mscale ratio nontrivially, int8-on-random-weights
noise still dominates a whole-model resident-vs-CPU comparison at this scale.

The actual correctness gate is decoder.TestAttnTempScale_matchesSequentialFormula — a pure,
backend-agnostic unit test of the exact formula both Model.AttnTempScale (decode) and
Model.AttnTempParams (CUDA's batched prefill, which must recompute it per row device-side)
expose, no GPU, no quantization noise. This test's only job: does declaring FeatAttnTemp let
the model go resident and run without error/NaN.
```

## TestMoEExpertReuseProbe

Moved from `metal/moe_expert_reuse_probe_test.go` (the comment above `TestMoEExpertReuseProbe`) on 2026-10-09.

```text
TestMoEExpertReuseProbe investigates M-05 (audit-metal-2026-09-12.md) cheaply, BEFORE building
the expert-major prefill restructuring it proposes: on a REAL paged MoE resident, run genuine
text through the EXACT path paged MoE prefill already uses today — prefillOK is hard-false for
a paged generic MoE (metal/model.go), so every prompt token already goes through the sequential
per-token decode loop, one router readback + stage per token per layer — and read off the
expert pool's own telemetry (stages / distinctExperts / nE) to see whether an expert-major
regroup (stage each routed expert ONCE per prefill instead of once per token that routes to it)
would converge close to nE (little win left — routing already touches most experts anyway) or
stay far below today's stage count (real cache churn from LRU thrashing, so M-05 is worth
building).

Manual/one-off by design (a real multi-GB checkpoint, several minutes of GPU time) — gated
behind GOINFER_MOE_REUSE_PROBE_CKPT rather than GOINFER_HEAVY_TESTS' usual asset registry, same
convention as TestMoEPrefillMeasure_batchedVsSequential (moe_prefill_measure_test.go). Unlike
that test, this one REQUESTS Metal's own GPU expert-cache paging (Options.MoECacheExperts) with
slots left at 0 (auto-sized, M-13) rather than requiring the whole expert set resident — the
qwen15-moe-a27b shape this repo already has on disk needs ~16.7 GB resident non-paged, which
does not fit a 16 GB Mac at all (confirmed 2026-09-10, ~/models/moe_prefill_measure.log), but
paged only holds N << nE experts per layer.

	GOINFER_MOE_REUSE_PROBE_CKPT=~/models/qwen15-moe-a27b GOINFER_MOE_REUSE_PROBE_M=512 \
	  go test -tags metal ./metal/ -run TestMoEExpertReuseProbe -v -timeout 30m
```

## TestMoE_declinesPrefill

Moved from `metal/moe_model_test.go` (the comment above `TestMoE_declinesPrefill`) on 2026-10-09.

```text
TestMoE_declinesPrefill: MoE moved from the decline side to the admit side on 2026-09-08 (G8
MoE half, docs/tasks/task-gpu-paths-2026-09.md) — the f16 MMA prefill path now runs a MoE layer's FFN
row by row off the batched residual, reusing the unchanged per-token decode MoE dispatch chain
(encodeMoERoute/encodeMoEExperts/encodeMoESharedExpert, metal/moe.go); the name is historical
(kept so `git log -p` on it still tells the right story — decoder/gptoss_decline_test.go
precedent). Both MoE and its dense twin must now accept.

Reuses the same identical-experts trick as TestMoE_assemblyVsDense above (8 identical experts +
zeroed shared expert ⇒ MoE FFN is mathematically equal to the dense FFN regardless of routing),
so this also gets a free numeric check that the dst-redirection (moeDst, docs/task-gpu-paths-
2026-09.md) didn't drop or double-count anything: MoE prefill logits must match dense prefill
logits closely, not just both "succeed".
```

## TestMoE_declinesPrefill.gate

Moved from `metal/moe_model_test.go` (the comment above `TestMoE_declinesPrefill.gate`) on 2026-10-09.

```text
This test exercises the ARCH admission (MoE now has a row-by-row FFN path) — not the
bit-identity decline that gates the Metal backend by default (54% divergence, §A2-Metal).
Opt past that outer gate so the arch logic is what's under test.
```

## TestMoEPrefillMeasure_batchedVsSequential

Moved from `metal/moe_prefill_measure_test.go` (the comment above `TestMoEPrefillMeasure_batchedVsSequential`) on 2026-10-09.

```text
TestMoEPrefillMeasure_batchedVsSequential is P-15's real-checkpoint measurement
(audit-2026-09-10): Metal's batched f16-MMA prefill (metal/prefill.go's PrefillLast) has been
default-ON above the 512-token floor for MoE since 306b16b, but §3.2's decision set that
justified default-ON was dense-only — nobody has actually timed the batched path against the
sequential per-token loop on a real, generic (non-Gemma-4) MoE. This times both, directly on
the raw resident (bypassing metalResident's floor/env-var wrapper, which only decides which
path a caller reaches — it does not change either kernel path's own cost), at K ∈ {512, 1024,
2048}.

Manual/one-off by design (real 28.6GB checkpoint, several minutes of GPU time) — gated behind
GOINFER_MOE_PREFILL_CKPT rather than GOINFER_HEAVY_TESTS' usual asset registry, since this is a
measurement script (docs/measurements/), not a correctness gate:

	GOINFER_MOE_PREFILL_CKPT=~/models/qwen15-moe-a27b \
	  go test -tags metal ./metal/ -run TestMoEPrefillMeasure_batchedVsSequential -v -timeout 30m
```

## TestMoECacheExperts_bitExactMetal

Moved from `metal/moecache_bitexact_test.go` (the comment above `TestMoECacheExperts_bitExactMetal`) on 2026-10-09.

```text
TestMoECacheExperts_bitExactMetal is G-1 (docs/tasks/task-option-path-admission-2026-10.md §4.3, finding 2, registered
before the code and amended with step 0's mechanism): Metal's expert cache (MoECacheExperts with fewer slots than
experts) is bit-identical to the fully resident load, through the real caller (decoder.Load, then Generate): the
generated tokens, their top-5 log-probabilities, and every layer's K/V.

The reference is the fully resident load with ExactPrefill. A paged model never takes the batched f16 prefill
(paging is never used by prefill), so against the DEFAULT resident load a prompt over the batched-prefill floor
differs by the batched prefill's own tolerance (0.0044 here), which is the gap finding 2 measured; step 0 showed
--exact-prefill alone reproduces it to the last bit and paging adds nothing. Two prompts, under and over the floor;
at top-k, top-k+1 and top-k+2 slots and at all but one, each asserted to have evicted. The generic MoE path
(mixtral-tiny: 8 experts, top-2) and Gemma 4's (gemma4-moe-tiny: 4 experts, top-2) page separately. Each planted
defect (expertPoolDefect) must go red: an evicted slot not restaged, the GPU told the next slot, a neighbour's scales.
```

## TestOptionPathMetal_kvPrecision

Moved from `metal/optiongrid_metal_test.go` (the comment above `TestOptionPathMetal_kvPrecision`) on 2026-10-09.

```text
TestOptionPathMetal_kvPrecision: KVPrecision's resident decode, speculative verify and session
cells. Metal keeps its resident KV at f16 whatever is asked (ResidentKVPrecision says so), so "f16"
must match the baseline exactly; "i8" is lossy, so its decode is held to a cosine floor against the
f16 baseline, its speculative path to exact agreement with its own plain decoding, and its session
reuse to agreement with a cold run of the same turn. Not exactness there: resident prefix reuse on
Metal is not bit-identical to a cold prefill even at f16 (about 0.004 in log-probability on this
model, kernels differing between decode and prefill), and resident_reuse.go does not claim it is.
```

## TestOptionPathMetal_prefillChunk

Moved from `metal/optiongrid_metal_test.go` (the comment above `TestOptionPathMetal_prefillChunk`) on 2026-10-09.

```text
TestOptionPathMetal_prefillChunk: ResidentPrefillChunk under MC3 prefills a long prompt in chunks while
another generation is decoding, and the chunked generation must match the same generation prefilled whole,
with the batcher's prefill-pass count showing the chunks ran.

Chunk 8 and 16 are the regression case (option-path admission finding 3, 2026-10-08): with two KV slots
Metal's batched prefill runs decode rows below a 32-token whole-prompt floor and f16 MMA from it, so a first
chunk under the floor ran on the other kernel class, and the reply differed from the whole prefill by
0.003-0.004 in log-probability: it depended on whether another generation was decoding. mc3Prefill now
raises a chunk below the resident's PrefillKernelFloor to the floor.
```

## TestPageCost_submissionStructure

Moved from `metal/pagecost_measure_test.go` (the comment above `TestPageCost_submissionStructure`) on 2026-10-09.

```text
TestPageCost_submissionStructure is Step-6 Step-0: PRICE what losing the value-independent
pre-encode costs, to decide whether a synchronous Metal expert-paging path is viable or whether
the speculative (prefetch-last-token's-experts) design is mandatory.

HONEST SCOPE (read this before trusting the number): the ask was to measure the GENERIC Mixtral-
class MoE path on a fitting checkpoint. There is NO such checkpoint on this Mac (only gemma4-26b,
which doesn't fit and is the new path), so this measures the SUBMISSION-STRUCTURE cost on a DENSE
model (qwen2.5-1.5b, 28 layers) — the dominant, architecture-independent term (1 command buffer/
token vs ~nL/token). It is a faithful LOWER BOUND on the per-layer MoE regime: MoE adds the router
readback + expert-stage host work at each boundary, which overlaps the GPU-idle gap this already
pays for. It is NOT the MoE-path number and is not reported as one. Option 3 (MTLSharedEvent
handshake — one submit/token + per-layer CPU↔GPU events) needs aikit bindings that don't exist yet;
it is measured only if regime (2) here is expensive enough to matter.

Reports baseline (pipelined pre-encode) and per-layer-submit as tok/s + ms/token, best of 3 warm
runs (first discarded), greedy, same model + prompt. Heavy-gated (loads a ~1 GB checkpoint).
```

## TestPageCost_submissionStructure.baseline

Moved from `metal/pagecost_measure_test.go` (the comment above `TestPageCost_submissionStructure.baseline`) on 2026-10-09.

```text
Baseline is ForwardEmb (INLINE single command buffer/token), NOT ForwardEmbPipe: the pipelined
path runs a persistent executor GOROUTINE, and measuring the inline per-layer regime while that
goroutine is alive contends the single Metal GPU and inflates per-layer (an earlier revision
read +106% that way; clean it is ~+43%). Both regimes measured inline, same conditions. The
encode-ahead OVERLAP that Pipe adds is small here anyway (~1-2%) — decode is GPU-bound, so the
dominant cost is the SUBMISSION STRUCTURE (1 command buffer/token vs ~nL), which this isolates.
```

## TestPageCost_sharedEventReal

Moved from `metal/pagecost_sharedevent_test.go` (the comment above `TestPageCost_sharedEventReal`) on 2026-10-09.

```text
TestPageCost_sharedEventReal is Step-6 Step-0 regime (3) on the REAL forward — the authoritative
measurement that the synthetic aikit probe (gpu/metal_sharedevent_test.go) understates. It reprices
all three regimes inline on qwen2.5-1.5b so shared-event is directly comparable to baseline and
per-layer-submit. FINDING (2026-08): shared-event handshake (~0.26 ms/boundary) is ≈ per-layer
submit (~0.23 ms/boundary) — recovers ~0%; both synchronous shapes cost ~+45%. Conclusion:
synchronous Metal MoE paging is not viable by either shape → speculative prefetch is the path.

This is the committed, repo-reproducible form of the regime-3 result (was run under a local go.work

	override at Step-0). Run: GOINFER_HEAVY_TESTS=1 GOINFER_HANDSHAKE_PROBE=1 go test ./metal/ \
	  -run TestPageCost_sharedEventReal -v  (loads a ~1 GB checkpoint; needs aikit/gpu >= v0.23.0).
```

## TestPairwiseRoPERealMetal.meanbar

Moved from `metal/pairwise_rope_real_test.go` (the comment above `TestPairwiseRoPERealMetal.meanbar`) on 2026-10-09.

```text
The mean bar is 0.99 for every model but Command-R7B, whose 0.989242 the owner accepted on 2026-10-09 ("close
enough, so it's a go"; docs/tasks/task-metal-pairwise-rope-2026-10.md G-PR5). Part C of
docs/tasks/task-metal-pairwise-followups-2026-10.md localizes that gap; this floor records the decision, it is
not a measured tolerance.
```

## TestMetalPrefillDecomp.reps

Moved from `metal/prefill_decomp_test.go` (the comment above `TestMetalPrefillDecomp.reps`) on 2026-10-09.

```text
Every repeat of every category that matters, so a wide spread can be traced to its reps
(S0's first run printed medians only, and its two >50% spreads could not be read back).
```

## TestMetalPrefillDecomp.loo

Moved from `metal/prefill_decomp_test.go` (the comment above `TestMetalPrefillDecomp.loo`) on 2026-10-09.

```text
Leave-one-out (GOINFER_METAL_DECOMP_LOO=1): a category timed in its own command buffer ran in two
modes ~2x apart on the MLP GEMMs at K=3900 (S0's first run, 52-55% spreads), and in a stable
all-fast run the categories summed to only 64.6% of the full replay — so an isolated time is not
that kernel's cost inside production. Here each large category's IN-SEQUENCE cost is the full
replay's time minus the time of the same replay with that category's dispatches removed, rep by
rep interleaved. Kernel timing does not depend on the values it reads (no data-dependent
branches), so the stale inputs a removed category leaves behind change no downstream kernel's
work — only the cache and memory state it sees, which is the effect being measured.
```

## TestMetalPrefillDecomp.state

Moved from `metal/prefill_decomp_test.go` (the comment above `TestMetalPrefillDecomp.state`) on 2026-10-09.

```text
Prior-state test (GOINFER_METAL_DECOMP_STATE=1): the MLP GEMMs run ~1.9x slower inside the sequence
than alone, at both K, and alone they flip between two levels ~2x apart. If that tracks the GPU's
RECENT WORKLOAD (a clock/power state) rather than the kernel's inputs, gate/up timed alone should
be slow straight after heavy work and fast after an idle gap. Four conditions, interleaved per rep.
```

## TestPrefillGateVsReference.ctxpin

Moved from `metal/prefill_gate_ref_test.go` (the comment above `TestPrefillGateVsReference.ctxpin`) on 2026-10-09.

```text
Pin ResidentContext to metalCtxCapDefault (4096): the 0 (backend-default) auto-cap sizes ctx off
AVAILABLE MEMORY, not this backend's fixed kernel-score-buffer ceiling, so on a box
with generous free RAM it picks something above metalCtxCapMax and BuildResident
declines outright ("resident context ... exceeds this backend's hard ceiling") —
falling back to CPU/staged, which fails this test's *metalResident type assertion.
4096 comfortably covers every decision/confirm cell here (2026-09-25: this pin used to be metalCtxCapMax,
written when that was 4096; it has been 32768 since 26f64807, so the fit guard priced a 32k KV — ~1.8 GB on
the 1.5B — and refused loads that fit): the
widest, K=3900 + continuationN(64) teacher-forced steps, tops out at pos 3962.
GOINFER_METAL_GATE_QUANT=int8int8 grades the native int8 path (docs/tasks/task-metal-int8-2026-10.md, slice
2): an int8int8 load with nativeInt8 on, both arms on its W8 kernels.
```

## runPrefillGateSet.g02

Moved from `metal/prefill_gate_ref_test.go` (the comment above `runPrefillGateSet.g02`) on 2026-10-09.

```text
G-02 (audit-metal-2026-09-12.md): a missing reference file used to silently drop that K
from the pool (runPrefillRefGateCellK returns nil, the loop above just skips appending it) —
so a pooled verdict that says "SHIPS" could rest on fewer decision cells than decisionKs
names, with nothing in the output calling that out unless a reader compares the header's K
list against decisionCells' own count by hand. Observed for real, repeatedly, this same
audit: S set "a" K=512 has no reference file and was pooled over silently in every prior
M-03/M-04 oracle run. A partial decision set is fine to REPORT (re-scoring, deciding=false,
is explicitly "never fails the test on its own" per this function's own doc comment above)
but must not silently DECIDE — Fatalf here, not there, so the message names exactly which K
is missing rather than requiring a reader to diff the header against the cell count.
```

## TestPrefillGate.floor

Moved from `metal/prefill_gate_test.go` (the comment above `TestPrefillGate.floor`) on 2026-10-09.

```text
G-07 (audit-metal-2026-09-12.md): this test's K=256 decision cell used to Fatalf outright —
metalFastPrefillFloor was 512 and nothing here overrode it, so PrefillLast declined before
any comparison ran. The floor has since dropped to 256 (M-02) and then 64 (R3), which clears
K=256 on its own, but disable it explicitly anyway so this test does not silently break
again the next time the floor default moves.
```

## TestPrefillGate.divergence

Moved from `metal/prefill_gate_test.go` (the comment above `TestPrefillGate.divergence`) on 2026-10-09.

```text
G-07 (audit-metal-2026-09-12.md): TestMetalPrefillDivergenceRate, the test that originally
measured this 54%, no longer exists in the tree (superseded by TestPrefillGateVsReference's
pooled §3.2 criteria, which withdrew the exact-as-oracle scoring that number came from) —
the figure itself is historical record, not a live test (docs/ollama-chase.md).
```

## runPrefillGateCell.exact

Moved from `metal/prefill_gate_test.go` (the comment above `runPrefillGateCell.exact`) on 2026-10-09.

```text
EXACT — sequential Forward per token, today's shipped default.

Forward's return is a REUSED buffer (metal/model.go ForwardEmbPipe: "Returns logits[V]
(reused buffer; consume before the next call)") — every capture below is cloned
immediately. Storing the raw slice instead silently aliases whatever the LAST Forward call
in the whole cell wrote, which is exactly the bug this comment is here to stop someone
from reintroducing: it first shipped that way, and the seed and every continuation position
all came back reading the same final buffer, producing a self-contradictory result (a
"42% seed gap" while position 0 of the continuation — which SHOULD be the same comparison —
quietly agreed).
```

## gemmS2Kernels

Moved from `metal/prefill_gemm_s2_test.go` (the comment above `gemmS2Kernels`) on 2026-10-09.

```text
gemmS2Kernels holds R16's prototypes (1-4; prototype 4 became production 2026-09-25) and the retired production
kernel gemm_w4f16_store_r15. It began as R16's first prototype (docs/tasks/red-october.md; design in
docs/measurements/metal-prefill-gemm-s2-2026-09-25.md): the int4 prefill GEMM restructured in the shape of
llama.cpp's classic kernel_mul_mm, which S1b measured at >= 2.96 TFLOPS on this machine against the current
gemm_w4f16_store's ~0.75 under sustained load. TEST-ONLY until R16's band is met — nothing in production
compiles it.
```

## TestPrefillParityMoEGatedShared

Moved from `metal/prefill_moe_parity_test.go` (the comment above `TestPrefillParityMoEGatedShared`) on 2026-10-09.

```text
TestPrefillParityMoEGatedShared exercises the OTHER branch encodeMoESharedExpert's
parameterization (this row) must get right: a sigmoid-GATED always-on shared expert
(FeatMoEGatedShared, Qwen2-MoE), not just the ungated GLM/DeepSeek shape the plain mixtral
fixture above never touches at all (mixtral has no shared expert).

The fixtures are the two committed gated-shared tiny checkpoints, both DeltaNet hybrids whose MoE layers take the same
batched shared-expert dispatch (since D-B01). It used to be testdata/tiny-qwen2-moe, a per-machine download whose
moe_intermediate_size of 44 the int4 group of 32 rejects, so BuildResident declined it and the test could only fail.
The cosine bar is 0.999, not the plain MoE test's 0.95: on these fixtures the gate is a small term, and dropping the
sigmoid (gate fixed at 1) still clears 0.95 on qwen35-tiny (0.99813) while reading 0.9998 / 0.9994 unmutated
(measured 2026-10-05). At 0.999 that mutation fails both.
```

## TestPrefillNoNaN

Moved from `metal/prefill_nan_test.go` (the comment above `TestPrefillNoNaN`) on 2026-10-09.

```text
TestPrefillNoNaN asserts the f16-MMA prefill produces finite logits across a range of prompt
lengths — the direct regression gate for the bug where the int8-pinned LM head was run through
the int4 gemm_w4f16 kernel (weights misread as packed nibbles), yielding NaN logits at EVERY M
including the minimal single-tile M=8. Complements TestPrefillParity (which pins the value); a
NaN here is the specific shipped-path failure that a hand-run caught only because no CI ran
Metal against a checkpoint.
```

## TestPrefillRefIdentity

Moved from `metal/prefill_ref_identity_test.go` (the comment above `TestPrefillRefIdentity`) on 2026-10-09.

```text
resident's. A reference generated from the same text lands at the W4A8-vs-CPU level (measured 0.0015–0.072 on
set B's S-K3900); one generated from different text lands orders of magnitude higher (3–18 on set A's). Found
2026-09-25: set A's 2026-09-05 files predate the 2026-09-09 snapshot (metal-decode-attn-r17-2026-09-25.md).
```

## TestPromptHiddenResidentMetal.int4

Moved from `metal/prompthidden_resident_parity_test.go` (the comment above `TestPromptHiddenResidentMetal.int4`) on 2026-10-09.

```text
Both sides run int4. Metal has no int8 GEMV (its decode path reports int8int8 -> int4), so an int8int8 CPU side compares
two quantizations, not two executions: measured 2026-09-30 on this fixture, CPU int8int8 vs Metal read cosine 0.977 with
the adapter merged, while CPU int4 vs Metal int4 reads 0.99985 (0.99998 without it). The adapter-merged tiny weights are
unusually int4-sensitive (CPU f32 vs CPU int4 is 0.980), which is a property of the fixture, not of the resident path.
```

## TestQuantBar_CPUInt4VsInt8

Moved from `metal/quantbar_test.go` (the comment above `TestQuantBar_CPUInt4VsInt8`) on 2026-10-09.

```text
TestQuantBar_CPUInt4VsInt8 answers the question every Metal Gemma parity number has been
unable to: how much of the gap is QUANTIZATION and how much is METAL?

The problem it solves: Metal has no like-for-like CPU reference by design. BuildResident
requires an int8 load and re-quantizes to its own W4A8, so residentParity is always
int4-GPU vs int8-CPU — a comparison that mixes two independent effects. Gemma measures
0.818 there and the control 0.990, and for months that delta was read as "Metal has a Gemma
bug". It might instead be "int4 costs more at Gemma's shape", and the two are indistinguishable
from a single number.

This fork separates them on the CPU alone, no GPU involved. decoder's int4 mode is the near-
exact quantization twin of Metal's resident W4A8: group-32 symmetric weights at scale=maxabs/7
(int4GroupSize, weightmat.go) and an int8 LM head + embedding (quantMode.embedding() pins them —
the same pin Metal ships). Same weights, same activations, same arithmetic class — the ONLY
thing that changes between the two runs is int4-vs-int8. So:

	cpu-int4 vs cpu-int8  =  the cost of the quantization class, at THIS model's shape
	metal    vs cpu-int8  =  that cost + whatever Metal adds

If Gemma's CPU fork lands near its Metal number (~0.82), then 0.818 IS the int4 bar at Gemma's
shape and there is nothing left to fix — the model ships. If the fork comes back ~0.99 while
Metal stays at 0.82, the quantization is exonerated and Metal really does have a Gemma-specific
bug, which reopens the hunt with the suspect list cut in half either way.

Both models run so the control calibrates the subject, per the lesson that a bar must be
measured on this box rather than assumed.
```

## TestQwen35_35B_pagedRuns

Moved from `metal/qwen35_35b_paged_test.go` (the comment above `TestQwen35_35B_pagedRuns`) on 2026-10-09.

```text
WHAT THIS MEASURES AND WHAT IT DOES NOT. Steady-state decode rate through the pager, plus the
staging decomposition (how much of a token is expert I/O). It is NOT a peer comparison: no
Ollama/llama.cpp arm runs here, and the CPU-paged 1.3-1.4 tok/s figure this lane has quoted was
measured in another session on another path, so no ratio against it is computed. See
~/goinfer-bench-logs/PREREGISTERED-qwen35-metal-paging.md for the rule this run was written
against, BEFORE it produced a number.
```

## TestQwen35_35B_cpuPagedBaseline

Moved from `metal/qwen35_35b_paged_test.go` (the comment above `TestQwen35_35B_cpuPagedBaseline`) on 2026-10-09.

```text
TestQwen35_35B_cpuPagedBaseline is the DO-NOTHING ARM for the run above: the same checkpoint,
same box, same session, decoded through the CPU expert pager (StreamWeights, decoder/moepaging.go)
instead of the Metal one. Without it, "Metal paging is faster" would be a cross-session ratio
against a number measured on another day — exactly the comparison this repo's own
docs/completed/task-zeno-compare.md shows can drift 2.6x from machine load alone (kind-3 gemma4 at one identical
config read 1.128 and then 2.917 tok/s).

IT LIVES IN metal/ AND USES NO METAL. That is deliberate: its only reason to exist is to be the
baseline for the Metal number, and a baseline that drifts away from its comparand into another
package (and another run) is how the 1.3-1.4 figure went stale in the first place. Same helpers,
same seed token, same greedy self-feeding trajectory, same token count.
```

## TestR1_laneVsCPU

Moved from `metal/r1_lane_vs_cpu_test.go` (the comment above `TestR1_laneVsCPU`) on 2026-10-09.

```text
TestR1_laneVsCPU is experiment X2 of the R1 (W4F16 decode lane) re-investigation. X1
(r1_gu_reference_test.go) showed that, within the resident int4 model, the f16 lane's gate/up
GEMV sits on the f64 reference and the shipped W4A8 lane is the coarse arm (its single
per-tensor int8 activation scale zeroes 97% of the FFN-26 input at the attention-sink
position). That leaves the one question that decides R1's fate: against an EXTERNAL
reference, is the f16 lane's full-model output better, equal, or worse than W4A8's?

Reference: the CPU backend, Options{Backend:"cpu", Quant:"int8"} — weight-only per-row int8,
f32 activations, exact f64-accumulating attention (GOINFER_CPU_FAST_ATTENTION=0, as
decoder/prefill_ref_gen_test.go forces it). The repo's own S reference is Quant:"" (f32 weights,
~6 GB) and does not fit beside anything on this 16 GB machine today (~5 GB free); "int8" is the
same choice prefill_ref_gen_test.go's d7RefQuant documents for the same reason. It is an
external reference: its weight requantisation differs from Metal's int4-g32, so BOTH Metal arms
carry the same weight-quant noise floor against it and the PAIRED comparison between arms is
what carries information. The CPU int4 path is deliberately NOT used (it quantizes activations).

Teacher-forced, no generation: the same N prompt tokens are fed one per step to the CPU
reference and to each Metal arm; per position we score cosine(logits), KL(softmax(cpu) ||
softmax(arm)) in float64, top-1 agreement and hard flips (decoder.NearTieArgmaxForTest).
Both Metal arms come from ONE model load: the lane is toggled at runtime via
r.decodeLaneW4F16 (canUseF16Lane reads the field on every call); Forward(id,pos) calls
setPos(pos), which rewrites curNKeys=pos+1 and the KV slot for pos, so restarting the arm at
pos 0 is clean (no other per-sequence state accumulates in forwardLogits).

Memory: ONE checkpoint resident at a time — the CPU model is Closed before the Metal load.

	GOINFER_HEAVY_TESTS=1 go test -tags "darwin goinfer_testhooks" ./metal/ -run 'TestR1_laneVsCPU$' -v
```

## TestR2_perLayerCtxDiff

Moved from `metal/r2_ctx_diff_test.go` (the comment above `TestR2_perLayerCtxDiff`) on 2026-10-09.

```text
TestR2_perLayerCtxDiff is the measurement R2's two prior investigations never took
(docs/measurements/r2-attn-fa-2026-09-19.md, r2-attn-fa-followup-2026-09-20.md): they compared
FINAL LOGITS of an attention_fa generation against a shipped-kernel generation, 28 layers and an
int8-activation-quantized pipeline downstream of the kernel under test, on Gaussian-noise
embeddings. The follow-up's own ULP control showed that pipeline is hypersensitive on that input
(a 1-ULP scale nudge flips logits by ~0.9 immediately), so "two clean steps, then a stable
plateau" is consistent with BOTH a real position-linked kernel defect AND a single int8 rounding
flip whose timing happened to line up. This test separates them by comparing the kernels'
OWN outputs (r.ctx) on IDENTICAL inputs, per layer, per decode step:

  - trajectory 1: the shipped kernel end to end via ForwardEmb (the record's reference arm);
  - trajectory 2: a manual per-layer harness with attention_fa OFF on both arms — must match
    trajectory 1 bit-for-bit, or the harness itself is wrong (checked, fatal if not);
  - trajectory 3: attention_fa's trajectory, where at every (step, layer) the shipped kernel is
    run from the same pre-layer residual and KV state first, its ctx and layer output captured,
    the residual restored, then attention_fa run — so |ctxFA - ctxShipped| is a pure kernel
    comparison on identical (q, K, V, nKeys), and |xFA - xShipped| shows where any downstream
    rounding flip lands. The trajectory then continues from attention_fa's own output, so its
    per-step logits reproduce the record's arm exactly.

The runtime toggle is r.decodeAttnFA (canUseAttnFA reads it on every dispatch; pipelines and the
partial buffer are always built) — the same technique R1's root-cause tests used for its lane.
One model load. Prefill length from GOINFER_R2_PREFILL (default 1600, the record's), steps from
GOINFER_R2_STEPS (default 5). If a layer's ctx diverges locally (maxRel > 1e-3, three orders
above f32 reduction noise), that layer's exact inputs and both outputs are dumped to
GOINFER_R2_DUMP_DIR for TestR2_replayDump below.

	GOINFER_HEAVY_TESTS=1 go test -tags "darwin goinfer_testhooks" ./metal/ -run 'TestR2_perLayerCtxDiff$' -v -timeout 20m
```

## runDecodeFidelityGate.identity

Moved from `metal/r2_gate_test.go` (the comment above `runDecodeFidelityGate.identity`) on 2026-10-09.

```text
Prompt identity: the reference files carry no prompt ids, so check the one row both arms share — the
prompt-final logits (reference row 0 vs the batched prefill's seed). A reference generated from different
text lands far above the W4A8 level here. Found 2026-09-25: set A's S-K3900 files (generated 2026-09-05 from
the live docs) predate the 2026-09-09 snapshot the prompts now come from, and 4 of 10 prompts changed inside
the 3900-token window (metal-decode-attn-r17-2026-09-25.md).
```

## TestZZ_residencyProbe

Moved from `metal/residency_probe_test.go` (the comment above `TestZZ_residencyProbe`) on 2026-10-09.

```text
TestZZ_residencyProbe isolates the per-submit RESIDENCY cost that dominates the paged-MoE decode
(~15 ms/boundary of GPU-idle-in-wait, 72× Step-0's 0.213 ms). Three arms separate per-buffer from
per-byte from re-validation, with trivial GPU work (1 thread) and many buffers REFERENCED (bound +
read) per command buffer. Load-bearing result (Arm A): a REPEATED identical set caches — submit[0]
~70 ms, submit[1..] ~0.4 ms — so the cost is the referenced set CHANGING per submit, and pinning
the working set resident once (MTLResidencySet / heap useHeap) collapses it. Arms B/C show the
uncached cost has both a per-buffer term (tracked-allocation count) and a per-byte term. Diagnostic,
not a gate; documents why the residency-set fix is the lever and would flag an OS residency change.
```

## TestS16MRoPEPrefill_real.sidecar

Moved from `metal/s16_mrope_prefill_test.go` (the comment above `TestS16MRoPEPrefill_real.sidecar`) on 2026-10-09.

```text
Through the directory's sidecar when it exists (S18: `<dir>.int4.metal.giw`, what serve's Metal load reads):
its int4 weights are mapped, not held twice, which is what the 2026-10-08 night's live budget (2.80 GB
against 3.77 GB for Qwen2.5-VL) lacked. It generates exactly what the direct load does
(prequant.TestDirSidecar_matchesDirectLoad).
```

## TestS18ShrinkCtxToFit

Moved from `metal/s18_ctx_shrink_test.go` (the comment above `TestS18ShrinkCtxToFit`) on 2026-10-09.

```text
The probe is a Metal load too: a CPU load prices more (it keeps arm64-repacked weights a Metal load does not), which put the budget above the
Metal build's 4096 need on the first try.
```

## TestS9EModelBatched_tiny

Moved from `metal/s9_emodel_batched_test.go` (the comment above `TestS9EModelBatched_tiny`) on 2026-10-09.

```text
TestS9EModelBatched_tiny is G-S9c of S9 step 2 (docs/tasks/task-multimodal-support-2026-10.md, registered before any
code): on gemma4-emodel-tiny (6 layers; head dims 32 and 64; 2 KV-shared layers with their own sources and double FFN
width; PLE P=32; layer scalars; v_norm), the f16 batched pass against the S9 layer-major pass (the validated path,
bit-identical to sequential) on one Metal resident. Compared: the last prompt row and 8 decode steps teacher-forced
along the layer-major pass's own greedy tokens; the bar is cosine >= 0.98 with every argmax equal or an R10 near-tie
(amended 2026-10-08 by the owner from the registered 0.9999, which the pass missed at 0.9907/0.9916 with no non-tie
argmax difference: the task doc's G-S9c record has the mechanism; 0.98 sits between that and the nearest defect, 0.94).
Two prompts: the golden 12-token prompt and a synthetic 96-token one, which runs every local layer past its window of 4
and both shared layers over long sources. The batched pass is called directly, so a silent fallback cannot stand in for
it. Each planted defect (emodelBatchDefect) must go red on at least one prompt.
```

## s3Sharpen

Moved from `metal/s3_towers_test.go` (the comment above `s3Sharpen`) on 2026-10-09.

```text
The tiny towers are degenerate along several of the registered defects' axes, and G-S3a's first tiny run showed it:
their biases are all zero (a dropped patch bias cannot show) and Qwen2.5-VL's q/k weights are at init scale (0.02), so
attention is nearly uniform and the scale, RoPE and windowing barely matter (each defect read 0.99999 or better). So,
before comparing, the check sharpens the fixture through the exports' aliasing slices, which the CPU tower reads too:
random biases (s3Sharpen) and q/k projections scaled up (s3Scale), so the attention logits are not near zero; for
Qwen2.5-VL also v and the output projection, whose init scale left attention a negligible share of the residual stream
(defect (3) still read 0.99992 with q/k alone: the tiny tower has one windowed block, then a full one).
```

## TestSAGemv_OutlierRegime

Moved from `metal/sa_outlier_test.go` (the comment above `TestSAGemv_OutlierRegime`) on 2026-10-09.

```text
TestSAGemv_OutlierRegime is the decisive experiment for the Gemma o-proj amplitude bug
(docs/prompts/gemma-metal-signflip-bisect.md, Fork 2). The CUDA box showed Metal's o-proj
contribution inflates 2–6× and flips on the secondary channels while CUDA's dp4a stays clean,
and named it a Metal W4A8 GEMV scale bug. TestSAGemvLargeK already proves the kernel correct at
K=4096 — but with a BENIGN random activation. This drives the same production kernel in the
regime that actually triggers the bug: at Gemma's o-proj K=2048, with an activation carrying
ONE massive outlier channel that sets the int8 scale (~558) and crushes every other channel to
near-zero int8 — exactly the post-quant state of the attention context feeding the o-proj.

The CPU reference computes the kernel's OWN formula bit-for-bit (dequant nibble × int8 act ×
f16 group scale × asc). So this isolates the KERNEL from the quant scheme:
  - Metal == CPU here ⇒ the o-proj GEMV arithmetic is FAITHFUL even in the outlier regime, and
    Metal's divergence-vs-CUDA lives upstream (a different attention context) or in the quant
    policy, NOT this kernel — which would redirect the fix.
  - Metal ≠ CPU here ⇒ the kernel mis-handles the outlier regime (group-scale accumulation or
    asc application under extreme dynamic range) — the bug, localized.

Reported per-channel on the crushed rows, because a whole-vector cosine is dominated by the one
massive output and would hide a 6× error on the secondary rows (the sink lesson).
```

## TestSAQVFusion_correctnessAndThroughput

Moved from `metal/sa_qv_fusion_test.go` (the comment above `TestSAQVFusion_correctnessAndThroughput`) on 2026-10-09.

```text
TestSAQVFusion_correctnessAndThroughput is the measurement item #5 of the 9-finding audit
needs before it goes near a production dispatch site: does fusing quant_vec into
gemv_w4a8_sa (gemv_w4a8_sa_qv) actually win, given every threadgroup the fused kernel
launches redoes quant_vec's O(K) amax reduction independently (no cheap way for one
threadgroup to hand a computed scale to another within one Metal dispatch)? Unlike item #4
(the rope2 merge, a genuine reduction in total work), this trades one dispatch launch +
one K-element device-memory round-trip against (N/8 - 1) redundant K-element reductions.

VERDICT (measured, not inspected): roughly NEUTRAL, leaning slightly negative — NOT the win
the audit's dispatch-count estimate implied. Four interleaved runs (see the interleaving
comment below for why non-interleaved gave a false 1.28x win) at real dims: 0.974x, 0.949x,
0.992x, 0.972x speedup — a tight cluster around ~0.97x, i.e. the fused kernel is a few
percent SLOWER, not faster. The redundant per-threadgroup reduction cost roughly cancels the
removed-dispatch savings at this K/N. Kept in the tree as a correctness-proven (bit-identical
to the two-dispatch path) but NOT-production-worthwhile experiment — do not wire this into
model.go on the strength of the dispatch-count argument alone; the wall-clock number doesn't
back it up here.
```

## TestSAQVFusion_correctnessAndThroughput.reps

Moved from `metal/sa_qv_fusion_test.go` (the comment above `TestSAQVFusion_correctnessAndThroughput.reps`) on 2026-10-09.

```text
reps=20: NOT the real per-token count (a real decode token issues far fewer than this per
layer) — chosen to stay clear of a real, separate, pre-existing issue: repeatedly calling
Encoder.Dispatch on the SAME reused buffers hundreds of times in one encoder (via the
general Dispatch path, which rebinds every buffer each call, unlike Run1DBatch's
bind-once-dispatch-many) hits a probabilistic crash unrelated to gemv_w4a8_sa_qv's
correctness (reproduces with the plain two-dispatch pattern alone, no fused kernel
involved, and isn't a hard threshold — it can still fire occasionally even at reps=20).
Worth its own investigation separately; out of scope here.
```

## TestSAQVFusion_correctnessAndThroughput.interleave

Moved from `metal/sa_qv_fusion_test.go` (the comment above `TestSAQVFusion_correctnessAndThroughput.interleave`) on 2026-10-09.

```text
INTERLEAVED, not two separate blocks: measuring "all of A then all of B" confounds the
comparison with whatever changes between the two blocks (thermal ramp, GPU contention
drift) — measured directly here: a first pass with A-then-B block order showed fused
WINNING 1.28x; two immediate re-runs of the same block order showed fused LOSING ~0.47x,
consistently with each other but not with the first run. That is a confound, not a real
effect, and it wouldn't have been visible without deliberately re-running. Alternating A/B
every sample makes drift affect both roughly equally instead of favoring whichever block
happens to run when conditions are better.
```

## TestSink_NormsNotCosines

Moved from `metal/sink_test.go` (the comment above `TestSink_NormsNotCosines`) on 2026-10-09.

```text
Fable's hypothesis: the probe token is <bos> (id 2), Gemma's ATTENTION SINK. Sink V vectors
are trained near-zero (sink K is a strong direction; sink V is a no-op). A cosine between two
near-zero vectors is rounding noise — which would make "layer-1 V cos = -0.047 ⇒ x after
layer 0 is orthogonal" a measurement artifact, not a bug. Every number in the debug report was
a cosine; nobody measured a NORM. So measure norms — and probe a NON-sink token too.
```

## TestZZ_metalSoftcapTokenShare

Moved from `metal/softcap_ab_test.go` (the comment above `TestZZ_metalSoftcapTokenShare`) on 2026-10-09.

```text
TestZZ_metalSoftcapTokenShare loads a real dense Gemma (final-logit softcap, 262k vocab) and times
the full-logits sampling path (ForwardEmb → forwardLogits → finalizeLogits) per token, so the
isolated softcap A/B (BenchmarkSoftcap_gemmaVocab_*: serial ~3.4ms, parallel ~0.86ms) can be
expressed as a share of the token. Opt-in timing diagnostic, not a gate.
```

## TestSpecNgram_copyOnStepVerify

Moved from `metal/spec_multiturn_test.go` (the comment above `TestSpecNgram_copyOnStepVerify`) on 2026-10-09.

```text
Both compared arms reuse the cache's prompt prefix, so both re-decode the prompt's last position. A cold arm would
have prefilled it, and the f16 prefill is not bit-identical to decode: comparing spec against the cold arm is a
cold-against-warm comparison, which on 2026-10-03 diverged at token 105 with attnFADepthFloor at 1024 and was taken
for a spec defect (B-P03's revert, since undone). The cold run is a warm-up only.
```

## TestMetalSnapshotGolden

Moved from `metal/snapshot_golden_test.go` (the comment above `TestMetalSnapshotGolden`) on 2026-10-09.

```text
TestMetalSnapshotGolden is the ABSOLUTE STORED REFERENCE the Metal gate suite otherwise lacks.

Every other Metal gate is self-consistent or tolerance-based: `paged ≡ non-paged` compares one
kernel against itself under different residency (any change to the kernel moves BOTH arms
identically → passes), and `Metal-vs-CPU` is cosine/tolerance (small movements pass by
construction, unavoidable given the f16 scale gap). That whole class — a reduction-WIDTH change
(float sum is non-associative, wired to threadgroup width; see tgReduce* in model.go), a
fused-kernel rewrite, a different accumulation order, a moved scale-application point — is
invisible to those gates. Only a reference that does NOT move when the code moves catches it.

This decodes a FIXED token sequence through the Metal resident path on tiny committed models, to
depths PAST the reduction widths (128 and 256), and byte-compares the logits (sha256) to a committed
golden. It is self-referential: it detects that something moved, not which side is correct — exactly
what's missing. It is MACHINE-PINNED (Metal float results are deterministic run-to-run and across
code versions on a given GPU, but not guaranteed identical across chip families). It WILL go red on
a legitimate improvement — that's the point; regenerate with the refresh flag after verifying the
change is intentional (the same goldens-refresh discipline the CUDA track uses):

	GOINFER_UPDATE_GOLDENS=1 go test -run TestMetalSnapshotGolden ./metal/

N-29: the re-bake this note demanded HAS happened — TestMetalSnapshotGolden reports
6 checkpoints byte-identical on this machine. A standing "EXPECTED TO FAIL" note turns
the suite's only absolute gate into noise: a red here is a REAL drift and must be read
as one. The G-02 history is kept below for provenance, in the past tense.

RE-BAKE DONE (audit G-02, fixed on Linux where this suite cannot run). The checkpoint call now
drives ForwardEmb with the production-scaled embedding row instead of Forward with a raw one, and
Forward/ForwardArgmax now apply the arch embed scale. For `gemma4-dense-scaled` (EmbedScale =
√hidden) that CHANGES the hashed stream — the stored entries pin the pre-fix, non-production
computation, so that CHANGED the hashed stream, and the golden was re-baked accordingly:

	GOINFER_UPDATE_GOLDENS=1 go test -run TestMetalSnapshotGolden ./metal/

`mixtral-tiny` has no embed scale and its entries must NOT move; if they do, something other than
G-02 changed and the re-bake should be refused pending investigation.

Regenerate too on a hardware change (different Mac). Runs on every `go test` for the two
COMMITTED models (mixtral-tiny, llama-attnfa-tiny — no heavy-model dependency, always
available). gemma4-dense-scaled (449 MB) is NOT committed — over GitHub's practical push
limit, unlike the tiny ones — so it's a local-only fixture regenerated deterministically via
`scripts/pin_gemma4_dense_scaled.py`; when it's absent this test skips it and still fully
checks the other two (keyed comparison, not positional — see below), rather than failing
opaquely or silently losing coverage for the always-available fixtures too (fixed 2026-09-21;
every OTHER consumer of that fixture already skipped gracefully, this test was the outlier).
Coverage: mixtral-tiny is full-causal (attention softmax denom over
>256 keys → the width coupling at multi-iteration depth) + rmsnorm_quant; gemma4-dense-scaled covers
rmsnorm_f32 + qk_norm. llama-attnfa-tiny (added 2026-09-21, R2 golden-coverage follow-up) covers
`attention_fa`, DEFAULT ON past depth 1024 (metal/model.go's attnFADepthFloor, 1536 until B-P03) — the other two
fixtures both fail canUseAttnFA's head_dim==128 guard (8 and 256 respectively) and can never
dispatch it regardless of depth, so this kernel had zero coverage from this suite until this
fixture existed. Its checkpoints straddle the floor exactly (1022 declines, 1023 engages) so an
off-by-one at the boundary is caught, not just steady-state behavior on either side. Union = every
pinned-width reduction kernel plus `attention_fa` this build DISPATCHES.

N-28: this used to claim attention_f32 as well. It is not covered and cannot be — model.go
hard-wires `r.kvF32 = false` and builds kv_store_f32/attention_f32 only inside `if r.kvF32`,
so both are dead code no path reaches. A coverage claim naming a kernel nothing dispatches is
the "a doc comment claiming coverage is not coverage" class: the reader matches the claim to a
plausible kernel name and stops. See §A2-Metal.
prodEmbedRow fills dst with token id's LAYER-0 INPUT exactly as production builds it:
decoder.embedResident dequantizes the embedding row and multiplies by the arch's embed scale
(Gemma's √hidden). Mirroring it here is what makes the golden a reference for the SHIPPED
computation rather than for an entry point production never calls (audit G-02).
```

## TestMetalSnapshotGolden.attnfa

Moved from `metal/snapshot_golden_test.go` (the comment above `TestMetalSnapshotGolden.attnfa`) on 2026-10-09.

```text
R2: attention_fa, DEFAULT ON past attnFADepthFloor=1024 (metal/model.go; 1536 until B-P03). Every other
fixture here has head_dim != 128 (mixtral-tiny: 8, gemma4-dense-scaled: 256), so
canUseAttnFA's hd==128 guard declines on both, regardless of depth — neither can ever
cover this kernel. llama-attnfa-tiny (scripts/pin_llama_attnfa_tiny.py) is a plain dense
GQA Llama shaped to clear every other guard too (no sandwich/postOnly/parallelBlock/
attnSink/kvI8/lora/MoE/DeltaNet/qGate/window — see canUseAttnFA). Checkpoints straddle
the floor exactly: curNKeys = pos+1, so pos=1022 (curNKeys=1023) is the last declining
position and pos=1023 (curNKeys=1024) is the first engaging one. 900 is a shipped-kernel-only
control below the floor; 1100 confirms the engaged kernel stays stable past the boundary, not
just at it. Re-baked for B-P03's floor move (G-04; it was 1400 / 1534 / 1535 / 1600 at 1536).
```

## TestMetalSnapshotGolden.skip

Moved from `metal/snapshot_golden_test.go` (the comment above `TestMetalSnapshotGolden.skip`) on 2026-10-09.

```text
gemma4-dense-scaled (449 MB) is NOT committed — GitHub's practical push limit is well
under that, so it can't be, unlike mixtral-tiny/llama-attnfa-tiny — and every OTHER
consumer of this fixture (metal/gemma4_dense_scaled_test.go, the cuda/gpu prefill tests,
decoder/gemma4_moe_forward_test.go) already skips gracefully when it's absent, pointing
at scripts/pin_gemma4_dense_scaled.py (deterministic, seed 0) to regenerate it. This test
was the one outlier that didn't, so a fresh machine got an opaque t.Fatalf instead of a
clean skip. Skipping one model here must not cost the OTHER (committed, always-available)
fixtures' coverage — see the schema-tolerant comparison below, keyed by (Model,Quant,Depth)
rather than positional/count equality, so mixtral-tiny and llama-attnfa-tiny still get
fully checked on any machine even when this one is locally absent.
```

## TestMetalSpecPrefillRegression

Moved from `metal/spec_prefill_regression_test.go` (the comment above `TestMetalSpecPrefillRegression`) on 2026-10-09.

```text
TestMetalSpecPrefillRegression is the Metal half of the 2026-08-31 speculative-prefill
regression that was measured on CUDA and asserted on Metal by interface only.

THE DEFECT: decoder/model.go's generateInto ingests a prompt through the batched Prefiller
seam (m.resident.(Prefiller).PrefillLast); decoder/spec_ngram.go's genNgramInto instead
loops target.resident.Forward(...) one token at a time. Speculative decode therefore pays a
per-prompt-token cost that plain generation does not, and the gap grows LINEARLY in prompt
length — on CUDA, 2.66 ms/prompt-token, R² 0.9977, which at 839 tokens made the speculative
path 4.1x SLOWER than not speculating at all.

THE GATE IS THE SLOPE, NOT THE RATIO. The spec-vs-off ratio moves with draft acceptance,
which moves with the corpus, so it flaps; the slope of (spec - off) against prompt length is
the defect's signature and is nearly acceptance-independent. Bar: 0.50 ms/prompt-token
(CUDA measured 2.66 with the bug, 0.12 without).

WHY THIS ASSERTS RATHER THAN LOGS: gpu/spec_ngram_resident_test.go measures the right
quantity and only t.Logf's it — its own header says "speedup is logged per workload" — so a
0.3x printed and failed nothing for six weeks. The slope check below is t.Fatalf.

WHY THE CORPUS IS READ FROM DISK: specWorkloads/ngramWorkloads are hand-written and
deliberately copy-heavy (4-7x the copy density of real code) and only 36-74 tokens long,
which is precisely why a regression that needs LENGTH to show went unseen. This reads real
repository source at run time instead.

METAL-SPECIFIC PRECONDITION — GOINFER_METAL_BATCHED_PREFILL=1 IS MANDATORY HERE, and a green
without it is meaningless. Historically (when this test was written) Metal's PrefillLast
(metal/backend.go) declined by default — Metal's batched prefill was not bit-identical to its
decode path (54% stream divergence, a figure once measured by TestMetalPrefillDivergenceRate,
docs/ollama-chase.md; that test no longer exists — superseded by TestPrefillGateVsReference's
pooled §3.2 criteria, G-07 audit-metal-2026-09-12.md). Batched prefill is now default-ON above
metalFastPrefillFloor (M-06/M-02, same audit), so this override is no longer strictly load-
bearing for a prompt past the floor — kept anyway so this test's precondition never depends on
the floor's current value or default state. Below the floor (or with the override removed),
PrefillLast still declines and generateInto falls through to the same per-token loop the
speculative path already uses, so there would be NO asymmetry to measure and the slope would
come back ~0 for a reason that has nothing to do with the fix. Metal now ALSO implements
ResidentPrefillKV (M-01, same audit: ForwardNoLogits skips the LM head on every prefill token
but the last) — but that lives inside residentPrefillSeed, which genNgramInto already shares
with generateInto (see that call site's own comment), so both arms of THIS test's comparison
benefit from it identically. No new asymmetry: Metal still has exactly one exposure to the
original bug (the batched-vs-per-token prefill seam this test targets), gated behind this
variable.
```

## TestSpecVerifyCurveMetal

Moved from `metal/spec_verify_curve_test.go` (the comment above `TestSpecVerifyCurveMetal`) on 2026-10-09.

```text
CRITICAL, and the reason this needs stating before any number below: `PrefillLast` — the batched
primitive this test times — is NOT bit-identical to decode's int8 path (its f16-MMA activation
path once measured 54% stream divergence, §A2-Metal, docs/ollama-chase.md — historical
record; the test that produced that figure, TestMetalPrefillDivergenceRate, no longer exists,
superseded by TestPrefillGateVsReference's pooled §3.2 criteria, G-07 audit-metal-2026-09-12.md).
PrefillLast is now default-ON above metalFastPrefillFloor for admitted architectures (it declined
by default only historically, when this comment was written); GOINFER_METAL_BATCHED_PREFILL=1
here forces it on regardless of the floor, for exactly the "measurement/TTFT-at-the-cost-of-
exactness" use this test is. So every number this test produces characterizes a kernel that is
NOT currently usable as P10's verify oracle on Metal — P10's own design requires the verify step
to reproduce sequential greedy exactly (00-core's lossless contract). A real Metal P10 leg needs
that bit-identity gap closed FIRST; this test answers "is the timing shape even worth it", not
"is this safe to ship".
```

## TestSwigluQuant_MassiveChannel

Moved from `metal/swiglu_massive_test.go` (the comment above `TestSwigluQuant_MassiveChannel`) on 2026-10-09.

```text
TestSwigluQuant_MassiveChannel isolates the BOS geglu bug (b04d799) to KERNEL vs DATA. The L0
BOS trace found swiglu_quant emitting ~0 for the dominant channel (gate=12.14, up=-18.49, so
geglu should be -224.4). This drives the SHIPPED swiglu_quant (allKernels) with exactly that
value at one channel and small values elsewhere. If the dequantized output there is ~-224 the
kernel is fine and the r.gu data was corrupted upstream; if it's ~0 the kernel itself drops the
massive geglu channel.
```

## skipIfMemoryDeclined

Moved from `metal/testshared_test.go` (the comment above `skipIfMemoryDeclined`) on 2026-10-09.

```text
skipIfMemoryDeclined skips a resident parity test whose resident build the memory guard declined. That is "this machine
does not have the free memory right now", and no forward ran, so it is not a parity result. It used to fail as
"admission says it should be admitted", which `go run ./cmd/gate gpu` reports as "a Metal forward moved":
TestMellumResidentParity did exactly that on 2026-09-30 (needs 4.30 GB; the guard's budget is 70% of live free memory,
about 3.4 GB with the owner's apps open). A skip is still named under the gate's "does NOT cover" list, so the gap
stays visible. Any other decline (a feature or admission mismatch) still fails at the caller.
```

## TestMaxThreadgroupStageBytes

Moved from `metal/tgbudget_test.go` (the comment above `TestMaxThreadgroupStageBytes`) on 2026-10-09.

```text
N-32: DeltaNet's out-proj staging (deltanet.go dispatches dp.valueDim*2) was missing
from this budget, so a model whose value dim exceeds every other staged width passed
the M-11 check and then exceeded the device limit at dispatch. Only these two rows
exercise the new term — without them the fix is ungated, which is where this test was
after the first attempt at adding them silently no-opped.
```

## TestMetalThetaAB

Moved from `metal/theta_ab_test.go` (the comment above `TestMetalThetaAB`) on 2026-10-09.

```text
TestMetalThetaAB — does wiring the MEASURED Theta actually make Metal faster?

Theta was reachable only as 0.5 on every backend, because AdaptiveDepth's
domain was [0,1) and Metal measures 1.006-1.048. Under 0.5 the controller
drafts; under the measured value it declines to draft, because a Metal verify
node costs a full target step (ForwardN is a loop of single-token Forwards, so
T(n) = n*T(1) — measured linear to n=16).

That predicts the speculative path under Theta=0.5 is SLOWER than not
speculating at all on Metal, and that the wired default recovers it by
declining. This measures that rather than asserting it. Three arms, one
prompt, interleaved:

	off        plain Generate, no speculation — the do-nothing arm, which is
	           the whole point: "beats every configuration" means nothing if
	           off wins, and here off is EXPECTED to win against Theta=0.5
	theta=0.5  the shipped-until-now behaviour, forced explicitly
	wired      Theta unset, so verifyTheta() supplies the measured 1.02

The assertion is deliberately weak in one direction and strong in the other:
`wired` must not be materially slower than `off` (it should be within noise of
it, since it declines to draft), and it must beat `theta=0.5`. Nothing here
claims speculation is bad in general — it claims this backend's verify is not
batched, which is exactly what item (2) would change.
```

## TestThetaProbe_Metal

Moved from `metal/theta_probe_test.go` (the comment above `TestThetaProbe_Metal`) on 2026-10-09.

```text
TestThetaProbe_Metal measures Theta — the marginal cost of one extra verify node, in units
of one single-token target step — on the Metal resident path. Method is identical to the
CUDA probe (cuda/theta_probe_test.go) and the CPU control (decoder/theta_probe_test.go) so
the three numbers are directly comparable: seed `depth` positions, then time ForwardN over a
ladder of widths, and take Theta = (least-squares slope of T(n)) / T(1).

decoder/spec_adaptive.go ships Theta = 0.5, calls it the batched-CPU value, and says
"measure it". CPU measured 0.456, CUDA 0.155-0.251. Metal was unmeasured.

WHAT TO EXPECT HERE, AND WHY IT IS NOT THE CUDA STORY. CUDA's low Theta comes from a verify
that streams the weights ONCE for the whole block, so the marginal node is far cheaper than a
step. Metal's ForwardN (metal/backend.go:metalResident.ForwardN) is NOT a batched kernel — it is a plain loop of
single-token Forward calls. If that is the whole story then T(n) = n*T(1) and Theta ~ 1.0,
which would mean the controller is running Metal on a constant that is too LOW and therefore
OVER-drafting — the opposite direction from CUDA, where 0.5 is too high and under-drafts.
That is a prediction from reading the dispatch, and the point of this test is to measure it
rather than assert it.

TruncateTo is a NO-OP on Metal (metal/backend.go:metalResident.TruncateTo), unlike CUDA where the probe leans on
it to hold context depth constant between timed calls. It is safe here for the reason its own
comment gives — KV positions are overwritten on write and attention reads only keys[0..pos],
so re-running ForwardN at the same startPos re-attends over the same span. It is called
anyway, so the two probes stay line-for-line comparable.
```

## TestTowerGemmKernel

Moved from `metal/tower_attn_test.go` (the comment above `TestTowerGemmKernel`) on 2026-10-09.

```text
TestTowerGemmKernel is S17 lever B's kernel gate on Metal (docs/tasks/task-multimodal-support-2026-10.md, "S17's
Metal lever B", registered before the code): tower_gemm_w32 and tower_gemm_w16 against a float64 A·Wᵀ + bias from the
f32 inputs, at the towers' projection shapes (rows trimmed so the host reference stays fast) and at edge shapes (M, N
and K off every tile multiple). Bars: every output within 1e-5 (w32: f32 accumulation order only) or 1e-3 (w16: the
weights rounded to f16) of its own Σ|a·w| + |bias|. (The first f16-activation kernel's bar was 2e-3; it passed here
and failed the tower bars, see the doc.) Weights at a tower's scale (σ 0.02), so a dropped bias is far over either bar. Each
planted defect must miss it, on both kernels.

tower_gemm_w8 is G-S18e (S18 on the Mac, registered before the code): int8 weights in groups of 32 with an f32 scale
each, against a float64 reference over the same dequantized weights (float(q)·scale), so the bar is 1e-5 (only the
accumulation differs); its fourth planted defect takes each group's scale from the next group.
```

## TestVLImageTurn_metalResident

Moved from `metal/vl_image_turn_test.go` (the comment above `TestVLImageTurn_metalResident`) on 2026-10-09.

```text
TestVLImageTurn_metalResident is S3's tiny-fixture check (2026-10-07, first run: Gemma 3 8/8 tokens identical,
Qwen2.5-VL 6/8) (docs/tasks/task-multimodal-support-2026-10.md): a Gemma 3 and a
Qwen2.5-VL image turn decode on a Metal resident, from the committed tiny checkpoints and their goldens' image
features, against the CPU decoder (both int4). It is the first image turn on a Mac's GPU for either family; the
registered S3 gate is the served turn on the real checkpoints.
```

## TestW8Native_F3_closerToF32

Moved from `metal/w8_native_gate_test.go` (the comment above `TestW8Native_F3_closerToF32`) on 2026-10-09.

```text
TestW8Native_F3_closerToF32 is gate F3: on one token sequence (the prompt, then the CPU f32 model's greedy
continuation, 24 positions), the mean per-position KL(f32 ‖ Metal int8int8) is at most 1.10 × KL(f32 ‖ CPU
int8int8). KL(f32 ‖ Metal int4) is reported beside it. Positions 0 and 1 are left out, as residentParity's cosine
leaves them out. The models load one at a time at a 1024-position context, so a 16 GB Mac holds the f32 reference
only while it runs; by day the fit guard still refused it there (needs ~2.8 GB against a 2.2 GB budget), so F3
runs on the night queue.
```

## TestGenerate_warmRepeatMatchesCold

Moved from `metal/warm_cold_identity_test.go` (the comment above `TestGenerate_warmRepeatMatchesCold`) on 2026-10-09.

```text
TestGenerate_warmRepeatMatchesCold: the same prompt generated twice on one model emits the same tokens, the second time
reusing every cached position but the last. Before decoder.PrefillTailExact, a prompt the cold run had prefilled with
the batched pass ran its last position through decode on the repeat and diverged some tens of tokens later (on the
2-slot fixture, token 42 of 160); TestMC3Chain_newcomerJoinsAndBothMatchAlone met it as an "MC3 identity defect"
because its alone runs were cold and its joint runs warm. Single- and 2-slot models, at 12 tokens (below the 16-token
floor: the exact lane both ways), 40 (the pass on one slot, the step on two) and 100 (the pass), greedy.
```
