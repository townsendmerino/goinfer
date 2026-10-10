# decoder: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `decoder`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## TestA3FastAttentionDivergence

Moved from `decoder/a3_divergence_test.go` (the comment above `TestA3FastAttentionDivergence`) on 2026-10-09.

```text
A3 (G24) ships a DOCUMENTED DIVERGENCE, so the divergence must be a measured
number, not an adjective. This reports what a user gives up by enabling
GOINFER_CPU_FAST_ATTENTION, at the prompt depths the flag is for.

WHAT THIS FILE ACTUALLY ASSERTS, stated to match the body rather than to flatter it.
It loads the DENSE bench checkpoint, so every assertion below is about dense:
  - default OFF: with the env unset, output is bit-identical to acc64;
  - explicit "0" is identical to unset;
  - the divergence clears the kernel comment's own >= 0.99 bar.

It used to claim, here, that it also pinned "MoE excluded: the flag cannot turn f32
attention on for a MoE arch at all". It never did — `MoE` appeared exactly once in
this file, in that sentence, and `arch.MoE` zero times. The exclusion it advertised
as pinned had never been measured on a MoE at all, and an auditor reading the promise
and matching it to a test name would have stopped there. That is the failure recorded
as its own rule in CLAUDE.md; this comment is the correction.

The exclusion itself was dropped on 2026-08-29 after it WAS measured: the mechanism is
real (14.5% of moeMLP calls flip their top-k at 28 layers, 70.1% of the divergence) but
the magnitude does not support a categorical refusal — 1-cosine 2.126e-3 for MoE against
2.400e-3 for the dense case this flag already ships, depth-matched, with a 48/48
IDENTICAL greedy continuation. Those assertions live where the models are:
a3_moe_exclusion_test.go, a3_moe_routeflip_test.go, a3_moe_tokenlevel_test.go.
```

## TestA3FastAttentionDivergence.compare

Moved from `decoder/a3_divergence_test.go` (the comment above `TestA3FastAttentionDivergence.compare`) on 2026-10-09.

```text
COMPARE off VS fast, not base vs fast. Until 2026-08-31 `base` (unset) WAS the acc64
path, so base-vs-fast measured the trade; with the default flipped, unset IS fast and
that pair is now fast-vs-fast — it reported cosine 1.000000000 and tripped this test's
own "flag had no effect" guard, which was right to fire and pointing at the test.
```

## TestA3FastAttentionDivergence.identity

Moved from `decoder/a3_divergence_test.go` (the comment above `TestA3FastAttentionDivergence.identity`) on 2026-10-09.

```text
BIT-IDENTITY IS maxAbs == 0, not `cos == 1.0 && maxAbs == 0`. cos is a float64 quotient
of sums; for two bit-identical vectors it lands NEAR 1.0 and need not equal it, so the
old conjunction could read "not identical" for vectors that were. It never fired before
2026-08-31 because the two arms always differed, so the weaker half was never load-
bearing — until the floor made them agree below 512 and the inverse assertion ran.
```

## TestA3FanoutEndToEnd

Moved from `decoder/a3_fanout_e2e_test.go` (the comment above `TestA3FanoutEndToEnd`) on 2026-10-09.

```text
A3 fan-out, END TO END on a real checkpoint.

The kernel measurement (TestA3FanoutUtilization) says head-level fan-out is
3.3x on attention alone. That is a KERNEL ratio, and this repo has already
paid once for projecting one of those to a whole model: the 3.11x four-layer
slice that became 1.52x at full depth, and the ~13% this very item was costed
at by feeding a serial-vs-serial kernel ratio into an Amdahl model built on a
parallel-path profile share. So the shipping claim comes from here, not from
there.

Method: paired and interleaved (CLAUDE.md measurement discipline, rule 7 —
difference matched observations, never pool them). Each pair runs the SAME
prompt through the SAME model twice, once with the fan-out and once with
GOINFER_PREFILL_ATTN_WORKERS=1, which forces prefillAttnWorkers to 1 slot and
so takes the serial head loop — exactly the pre-A3 shape, with MatmulBT still
column-parallel inside it. That env var already existed as an A/B handle; no
measurement-only knob was added to the production path for this.

It also asserts the two arms' logits are BIT-IDENTICAL at every depth. That
is not decoration: it is what makes the timing a like-for-like comparison
rather than a race between two different computations, and it exercises the
fan-out through its real caller (forwardLayersN) rather than through a
hand-supplied calling convention.
```

## TestA3FanoutEndToEnd.depths

Moved from `decoder/a3_fanout_e2e_test.go` (the comment above `TestA3FanoutEndToEnd.depths`) on 2026-10-09.

```text
8192 added 2026-09-23 (P23, docs/queue-performance.md): the doc's own text says
"K=8192 is NOT measured here and is deliberately not extrapolated to" — the trend across
1024/2048/4096 is monotone but a fourth point is what actually answers whether it holds,
keeps climbing, or plateaus/reverses at the depth the f32 flag's own headline uses.
```

## TestA3FanoutUtilization

Moved from `decoder/a3_fanout_test.go` (the comment above `TestA3FanoutUtilization`) on 2026-10-09.

```text
A3 fan-out — is the f32 prefill attention path actually single-threaded?

The premise this test exists to check, quoted from G24's own harness
(g24_attnkernel_test.go) and carried into the queue and the Atlas as a
costed next move:

	"the f32 branch in attendBatchedHeads is single-threaded by construction
	 (its per-kv-group gather is shared mutable state)"

That sentence is true about the HEAD LOOP and was then read as if it were
true about the WORK. It is not. The f32 arm's two matmuls are
linalg.MatmulBT, which fans out internally over its N output columns
(parallelCols) above parThreshold = 1<<24 MACs. At the K=8192 tile shape
each call is ~268M MACs — 16x over that line — so the f32 path is already
using every core, just at the COLUMN level instead of the HEAD level.

G24 measured its two arms with MatmulBT forced serial, deliberately and
correctly, to get an arithmetic-plus-gather ratio. The error was downstream
of it: that serial-vs-serial ratio (8.37x) was fed into an Amdahl estimate
against a profile share measured on the PARALLEL production path, and the
gap between them was booked as recoverable headroom worth ~13%. Two
different parallelism states either side of one division sign.

So this measures the real attendBatchedHeads, at real Mellum2 geometry, and
reports UTILIZATION (CPU time / wall time) alongside wall time. Utilization
is the discriminator: a genuinely single-threaded path cannot exceed ~1.0x
no matter what its wall clock says, and no amount of Amdahl arithmetic can
argue it up.

Arms, all on identical inputs:

	acc64            production default before 2026-08-31 (head-parallel)
	f32              production default now (column-parallel inside MatmulBT)
	f32-serial       f32 with linalg forced serial — the do-nothing arm, and
	                 the ONLY arm the "single-threaded by construction" claim
	                 actually describes

Pre-registered reading (written before the first run):

	If f32 utilization is ~1x, the premise holds and head fan-out is worth
	building. If it is >2x, the premise is false, the ~13% is not there to
	recover, and the queue item closes as already-parallel. The band between
	1x and 2x is ambiguous and parks pending a profile.
```

## TestA3MoEExclusionIsMeasured

Moved from `decoder/a3_moe_exclusion_test.go` (the comment above `TestA3MoEExclusionIsMeasured`) on 2026-10-09.

```text
What is the MoE exclusion in A3/G24 actually worth?

forwardn.go excludes MoE from --cpu-fast-attention unconditionally, on a stated
mechanism: "an f32 QK reassociation flips a top-k expert at a near-tie and
cascades ... so MoE is excluded here rather than trusted to the operator." The
mechanism is real in kind — routing is discontinuous where a dense MLP is not.
But it was never MEASURED on a MoE: no MoE appears in the A3 kernel-ratio
record, and both tests in a3_divergence_test.go load the DENSE bench model,
including TestA3FastAttentionDivergence, whose doc comment claims it pins "MoE
excluded" while asserting nothing about MoE at all.

That matters now because the term the exclusion protects is the one that
dominates. On a Mellum2 slice, attention is 83.2% of prefill work at K=1024 and
97.1% at K=8192 (ON A 4-LAYER SLICE — the full model is lower, and the measured
full-model win is 1.52x/1.59x, not the slice's 3.11x) — so the excluded lever is
aimed at most of the cost while
expert-major batching competes for what is left.

This measures BOTH halves of the trade at once:

	COST — output cosine and max abs delta, acc64 vs f32, the same statistic
	       a3_divergence_test.go reports for dense (which ships at 0.9976).
	GAIN — wall-clock speedup of the same prefill.

It deliberately does NOT decide anything. A cosine is not a routing-flip count,
and a flip that changes generated TOKENS is the thing that would justify the
guard; this bounds the perturbation, and says so.

	GOINFER_HEAVY_TESTS=1 GOINFER_MELLUM_CKPT=... GOINFER_MELLUM_K=2048 \
	go test -tags goinfer_testhooks ./decoder/ -run TestA3MoEExclusionIsMeasured -v
```

## TestA3MoEExclusionIsMeasured.quant

Moved from `decoder/a3_moe_exclusion_test.go` (the comment above `TestA3MoEExclusionIsMeasured.quant`) on 2026-10-09.

```text
Quant is an axis here, not a detail. The full 28-layer Mellum2 is ~12 GB at
int8int8 and does not fit the 16 GB Mac at all, but ~6 GB at int4 does — and
int4 is what docs/completed/mellum2-resident.md actually runs. The attention swap under test
is quant-INDEPENDENT (acc64 vs f32 accumulation), but its SHARE is not: int4's
faster weight matmul raises attention's fraction of prefill and so raises the
speedup. So an int4 number is the operator-facing one and an int8int8 number is
the one comparable to this file's earlier runs; label which you quote.
```

## TestA3MoERouteFlips

Moved from `decoder/a3_moe_routeflip_test.go` (the comment above `TestA3MoERouteFlips`) on 2026-10-09.

```text
DOES A ROUTER FLIP EXPLAIN THE A3 MoE DIVERGENCE? — the mechanism experiment for the
--cpu-fast-attention MoE exclusion.

The state of the argument before this test. forwardn.go excludes MoE from A3/G24
unconditionally, on a stated mechanism: "an f32 QK reassociation flips a top-k expert at a
near-tie and cascades". TestA3MoEExclusionIsMeasured now puts the SYMPTOM at cosine
0.999787-0.999788 — an order of magnitude tighter than the 0.9976 dense already ships behind
the same flag. But a cosine cannot tell a routing flip from ordinary numeric drift, and the
guard is a claim about routing specifically. This measures the mechanism.

THE DECOMPOSITION, which is what makes it decisive. Three arms on one prompt:

	A  acc64 attention, natural routing          — the baseline; its routing is recorded
	B  f32 attention, natural routing            — total divergence (what the flag would ship)
	C  f32 attention, A's routing REPLAYED       — divergence with the routing term REMOVED

Arm C uses the existing moeSelOverride seam (built for E2's higher-precision replay). So:

	cos(A,C) ~= cos(A,B)  =>  routing flips contribute ~nothing; the divergence is ordinary
	                          drift and the guard is aimed at something that is not happening.
	cos(A,C) >>  cos(A,B)  =>  routing flips ARE the dominant term and the guard has its case.

THE PREDICTION, stated before the run so it can fail: given a symptom this small, arm C should
land close to arm B. If instead C is far cleaner than B, the exclusion is vindicated on its own
mechanism and this test says so — which is a real finding, not a failed test.

WHAT A FLIP COSTS is reported too, because "a flip happened" is not "a flip mattered".
norm_topk_prob renormalizes over the kept k, so swapping the k-th expert perturbs the sum by at
most its weight; the SMALLEST top-k weight therefore BOUNDS one flip's contribution. That is a
bound, not a margin — the dropped expert's own score is not in the trace — and it is reported
as one.

DIAGNOSTIC, NOT A GATE. It asserts only what must hold under either story (the arms are
comparable, the seams actually fired). The numbers are for a recorded decision.

	GOINFER_HEAVY_TESTS=1 GOINFER_DIAG=1 GOINFER_MELLUM_CKPT=... GOINFER_MELLUM_K=2048 \
	go test -tags goinfer_testhooks ./decoder/ -run TestA3MoERouteFlips -v -timeout 60m
```

## TestA3MoETokenLevel

Moved from `decoder/a3_moe_tokenlevel_test.go` (the comment above `TestA3MoETokenLevel`) on 2026-10-09.

```text
DOES THE A3 MoE DIVERGENCE CHANGE THE TOKENS A USER SEES?

The last gap before extending --cpu-fast-attention to MoE. Everything measured so far is a
COSINE on hidden states: 0.997874 at full depth, against 0.9976 for the dense case the flag
already ships. That comparison is depth-matched (both models are 28 layers) and it says the
categorical refusal is unsupported. What it does NOT say is what a user experiences.

The two errors differ IN KIND, which is why the cosine alone is not enough. Dense's divergence
is smooth numeric drift — every logit nudged slightly. MoE's is 70% ROUTING FLIPS, which are
discontinuous: 14.5% of moeMLP calls select a different expert set. Two perturbations with the
same cosine can behave differently in generated text, and generated text is the product.

WHAT IS COMPARED. The flag perturbs PREFILL only (forwardLayersN); decode is the same code on
both sides. So the arms share everything except the KV cache and first logits the prompt
produced, and any divergence in the continuation is downstream of exactly the thing under test.

WHAT IS REPORTED, and why a bare token-mismatch count would mislead: greedy decode is a
sequence of argmaxes, so ONE flip at a near-tie makes every later token "differ" without any
quality claim being warranted. So this reports the FIRST divergence and the baseline's
top1-vs-top2 margin there, normalized by the logit range — the repo's existing 3%-of-range
near-tie rule (gpu/kv_f16_test.go, gpu/kv_i8_parity_test.go). A first divergence at a near-tie
is the benign class those tests already accept; one at a real margin is not.

	GOINFER_HEAVY_TESTS=1 GOINFER_DIAG=1 GOINFER_MELLUM_CKPT=~/models/mellum2-unq \
	GOINFER_MELLUM_K=2048 GOINFER_MELLUM_N=48 \
	go test -count=1 -tags goinfer_testhooks ./decoder/ -run TestA3MoETokenLevel -v -timeout 120m
```

## TestAikitPinsAgree

Moved from `decoder/aikit_pin_test.go` (the comment above `TestAikitPinsAgree`) on 2026-10-09.

```text
TestAikitPinsAgree pins every module's aikit require to the ROOT's.

WHY. goinfer is five Go modules, each with its own go.mod, and each pinning
github.com/townsendmerino/aikit independently. Nothing checked that they agree. Measured
2026-09-06: the root sat at v1.37.0 while cuda, gpu, metal and demo/agent were all still on
v1.35.0 — a drift discovered only by reading four files by hand, prompted by a months-old git
stash that had once tried to fix the same thing at v1.33.0.

It matters more here than "tidiness" suggests, for two reasons:

  - The RELEASE ASSETS ARE BUILT FROM THESE MODULES. The Mac and Linux `goinfer-serve` binaries
    come from metal/cmd/serve and cuda/cmd/serve, so a submodule pinning an older aikit ships an
    older aikit to users — the same failure shape as the release-asset finding recorded in
    docs/tasks/task-first-hour.md, one dependency over.
  - The parity manifest's aikit_version tracks the ROOT's pin only. A submodule on a different
    aikit is numerics the staleness gate cannot see, which is exactly the hole that let
    aikit_version sit at v1.19.0 for seventeen versions.

aikit's own `gpupins` gate (`gpu backend version pins` in its CI, with `gpupins --fix`) checks both
the root aikit pin and the aikit/gpu pin across aikit's eight gpu backends. This test covers the
same two pins on the goinfer side, as two subtests: "aikit" and "aikit/gpu". (Until 2026-09-24 this comment claimed that parity while the test matched only
`aikit v…` — the regex's space after "aikit" never matches `aikit/gpu v…` — so metal sat on
gpu v0.33.1 while cuda was on v0.33.3, unseen.) Neither test can check that a pin is the LATEST
tag — that needs the network — only that the modules agree. Mid-cycle drift is EXPECTED between
releases — RELEASING.md's two-step tag bumps the submodules after the root tag — but expected is
not the same as unchecked, and the two-step is precisely the ritual a person can forget.
```

## TestAikitPinsAgree.gpu

Moved from `decoder/aikit_pin_test.go` (the comment above `TestAikitPinsAgree.gpu`) on 2026-10-09.

```text
testAikitGPUPinsAgree pins every module that requires github.com/townsendmerino/aikit/gpu to the SAME
version. There is no root pin to compare against (the root and gpu/go.mod require only aikit), so the
rule is agreement: today cuda and metal, which build the Linux and Mac `goinfer-serve` release
binaries. A module on an older gpu ships older kernels to users, and the parity manifest tracks the
ROOT's aikit_version only, so a gpu-only difference is numerics its staleness gate cannot see.
RELEASING.md's B-07 states this rule; before this test nothing enforced it (metal sat on v0.33.1
while cuda was on v0.33.3, found 2026-09-24).
```

## TestAPITiers_hardListMatchesDoc.substring

Moved from `decoder/api_tiers_test.go` (the comment above `TestAPITiers_hardListMatchesDoc.substring`) on 2026-10-09.

```text
G-10: a BARE substring match cannot fail. "Load" is inside "LoadGGUFBytes", so
deleting `decoder.Load` from the Hard section left this green — and the same held for
Model, Session, Config, Segment, Template, Tool, Turn, Meta and Has, every one of them
a substring of some longer name the document also lists. A check that cannot go red
for the ten most fundamental entries is not gating the document.

Match the name as the document actually writes it: inside backticks, optionally
package-qualified, and terminated — so `decoder.Load` matches and LoadGGUFBytes does
not lend it its letters.
```

## namedInDoc.scope

Moved from `decoder/api_tiers_test.go` (the comment above `namedInDoc.scope`) on 2026-10-09.

```text
SCOPED TO THE PACKAGE'S OWN SUBSECTION. Searching the whole Hard section let
`tokenizer.Load` satisfy an entry for `decoder.Load` — measured: with the doc's
`decoder.Load` deleted, the check still passed on the tokenizer line. The document is
organised as "### `<pkg>`" subsections, so that is the unit to search.

Within a subsection the TRAILING boundary does the work: it separates `Load` from
`LoadGGUFBytes`. The leading one only rejects a name buried inside a longer identifier
(Reload), so it excludes word characters but NOT a dot — the document qualifies by
whatever reads best, `decoder.Load` in one place and `Model.LoadSession` in another, and
both are the name being named.
```

## namedInDoc.entries

Moved from `decoder/api_tiers_test.go` (the comment above `namedInDoc.entries`) on 2026-10-09.

```text
WHOLE BACKTICKED ENTRIES, not a regex over prose. A word-boundary match still let a TYPE
be satisfied by a METHOD that happens to be qualified by it — `Model.LoadSession` answered
for `decoder.Model`, so deleting the type from the document changed nothing. Measured: of
the ten names G-10 lists, a boundary match caught only Load.

An entry names `ident` when some backticked span IS it (bare or package-qualified), or
ends in "."+ident — which covers `Model.LoadSession` naming LoadSession and
`Options.Quant` naming Quant, while `Model.LoadSession` does NOT name Model.
```

## AssetRegistry.gateside

Moved from `decoder/asset_registry_test.go` (the comment at the top of the file) on 2026-10-09.

```text
THE GATE SIDE OF THE SHARED ASSET REGISTRY (testdata/assets.json).

Every heavy gate used to resolve its own asset: read an env var, fall back to a path it spelled
out itself, and decide presence with os.Stat. The sweep's preflight did the same thing again in
bash with `[ -e ]`. Two implementations of "is this asset present", free to disagree, and they did:

  * a DIRECTORY satisfies `-e`, so preflight reported .gguf assets RESOLVED while naming the
    directory above them. Four gates were costed by that.
  * GOINFER_QWEN35_GOLDEN's real requirement is a readable manifest.json INSIDE the directory,
    which `-e` on the directory cannot express -- preflight said present, the gate skipped.
  * GOINFER_PREQUANT_GGUF had three different fallbacks across four call sites and, at
    loadInt4Model, none at all -- so one box ran different gates against different files.

Now both sides read testdata/assets.json and apply the predicate it states. The two
implementations (this file and scripts/asset_registry.py) are checked against each other by
TestAssetRegistry_agreesWithPreflight rather than assumed to match.
```

## TestAttendBatchedHeads_ctxBitsUnchanged

Moved from `decoder/attn_ctx_direct_test.go` (the comment above `ctxCase`) on 2026-10-09.

```text
Audit R-17 (the remainder): the AV kernels (MatmulAVAcc64 / MatmulAVAcc64Group) overwrite a contiguous destination, so attendBatchedHeads's K=1 paths write
each head's context straight into ctx instead of into scratch and then copying it there (the per-head `ch`, the grouped `gCtx`, and Arm B's `fullCtx`).

This pins every one of those paths to the ctx bits the code produced BEFORE that change: a hash of ctx per case, recorded from the unmodified code. The attention
kernels accumulate in f64 and are bit-identical across architectures by design (the grouped NEON port and its Go fallback included), so one table serves every
arch; if a case ever differs by architecture, the table must be keyed by GOARCH, not loosened.

	GOINFER_CTX_RECORD=1 go test ./decoder/ -run TestAttendBatchedHeads_ctxBitsUnchanged -v   # prints the table: only from code whose ctx path is the reference
```

## BenchmarkAttnDistinctBytes

Moved from `decoder/attn_distinct_bytes_bench_test.go` (the comment above `BenchmarkAttnDistinctBytes`) on 2026-10-09.

```text
BenchmarkAttnDistinctBytes is R13 (docs/tasks/red-october.md) step 0(iii): "the CPU's version of
the Metal collapse probe and CUDA's ncu traffic ratio" — it asks whether the real GQA layout
(nKV distinct KV heads, each read by nH/nKV query heads) is faster than an otherwise-identical
MHA-EXPANDED layout (nH distinct KV heads, one per query head, same per-head QKᵀ/scores·V work)
PURELY because of the byte-count difference, or whether the CPU's cache already dedups the
group's repeated reads so the two run at the same speed. Neither arm computes anything
meaningful (synthetic random data, no softmax, no correctness claim) — this is a memory-access-
pattern probe, not a model or a golden.

Reading the result: GQA MARKEDLY faster than expanded ⇒ the hardware is not deduping, every
query head pays real DRAM/cache traffic for its group's reads, and an explicit K/V-staging
kernel (reading each KV head's row once, sharing it across its group in registers/threadgroup
memory) recovers real bandwidth, not just µops — "the top of the band is live" in the brief's own
words. GQA and expanded running the SAME speed ⇒ the cache already dedups the repeated reads in
hardware (they're hot from the immediately-prior query head's pass), and a grouped kernel's whole
gain is the µop sharing R13's own Build section already banks on — no additional bandwidth win to
expect.

Real head shapes (NumHeads/NumKVHeads/HeadDim) come from loadBenchModel()'s own config so the
group size G = NumHeads/NumKVHeads matches what the served kernels actually see; the K/V data
itself is synthetic (rand, seeded) since only the ACCESS PATTERN is under test.
```

## withGroupedKernels

Moved from `decoder/attn_grouped_test.go` (the comment above `withGroupedKernels`) on 2026-10-09.

```text
TestAttendGroupedHeads_matchesPerHead is R13 Gate (2) (bit-identical, no
golden change — this compares directly rather than against a stored
golden) and Gate (4) (the wiring proof: attnGroupedRuns must be nonzero
with grouping on, at a shape that is actually eligible for it).
withGroupedKernels forces the grouped path's platform gate on for one test: these tests exist
to exercise and wire-prove the grouped path (Go fallback included), whatever this
architecture's shipped default is (cpu_tuning_other.go turns it off on non-arm64 — R9's Linux
attribution measured the fallback slower than per-head).
```

## TestAutoMetalPrecision_keepsTheLoadedPrecision

Moved from `decoder/auto_backend_test.go` (the comment above `TestAutoMetalPrecision_keepsTheLoadedPrecision`) on 2026-10-09.

```text
TestAutoMetalPrecision_keepsTheLoadedPrecision (R17, then slice 5 of docs/tasks/task-metal-int8-2026-10.md): when
auto chose metal, a model Metal would run only re-quantized (int8, int4mix) or not at all (f32) stays on the CPU at the
precision it loaded at, and says so. int8int8 goes to BuildResident, since Metal runs dense int8int8 natively, and is
kept only when the resident reports that it ran at int8int8; one Metal re-quantized (MoE, DeltaNet) is dropped for
the CPU. int4, and any model on a metal the user named, go resident as before. Without the guard, the model-included
0.5B goinfer-chat (an int8int8 bundle) went resident on Metal at int4 and loaded in 1.7 s against 0.17 s on the CPU
(exploratory runs, docs/measurements/r17-auto-backend-2026-10-01/).
```

## TestBackendReport_namesTheEffectiveBackend

Moved from `decoder/backend_report_test.go` (the comment above `TestBackendReport_namesTheEffectiveBackend`) on 2026-10-09.

```text
R2 (docs/measurements/cold-user-2026-09-06.md, finding #3): on a Mac the runtime printed
"decoder: metal backend not built in ... using cpu" and then, on the very next line,
"loaded 28-layer model ... [backend=metal quant=int4]". The warning scrolls past; the status
line is what gets pasted into an issue, and it named a backend that was not executing. The
cost on that box was 37.9 vs 82.3 tok/s.

These two tests are the gate. The first pins the semantics of the report; the second pins that
every banner in the tree actually uses it, which is the half that would have gone red on
v0.16.0 (all three banners formatted the REQUEST).
```

## BailingHybrid.header

Moved from `decoder/bailing_hybrid_test.go` (the comment at the top of the const block) on 2026-10-09.

```text
G5 (docs/completed/task-families-2026-09.md, batch 2) Bailing Hybrid parity (inclusionAI, Ling 3.0,
model_type "bailing_hybrid"): DeepSeek-style Multi-head Latent Attention alternating with Kimi
Delta Attention (KDA) every layer_group_size-th layer being MLA, over a DeepSeekMoE FFN.

MLA and the MoE router are pure composition of goinfer's existing deepseekArchitecture
primitives (verified field-for-field against the real modeling_bailing_moe_v3.py, parameterized
for two real naming departures — both mixers are self.attention not self.self_attn, and MLA's
output projection is self.dense not o_proj — plus an optional Laguna-shaped sigmoid output
gate). KDA is the one genuinely new primitive: a delta-rule recurrence structurally identical to
Gated DeltaNet but with a PER-CHANNEL decay (batch 1 F4's rehearsal, decoder/kda_rehearsal.go,
already proved this against fla-org/flash-linear-attention's actual reference,
maxAbsDiff 2.98e-08).

Regenerate (hand-assembled tiny checkpoint + golden, both reproducible — see
scripts/pin_bailing_hybrid_tiny.py's own docstring for why the real BailingMoeV3ForCausalLM
can't be instantiated on this Mac: its modeling file imports fla.ops.kda at module top level,
which transitively imports Triton, unavailable on this platform):

	~/.venv-nemotron3/bin/python scripts/pin_bailing_hybrid_tiny.py
```

## TestBlockSpecGenerate_commitsResIDsOnFullCompletion

Moved from `decoder/blockspec_commit_test.go` (the comment above `TestBlockSpecGenerate_commitsResIDsOnFullCompletion`) on 2026-10-09.

```text
TestBlockSpecGenerate_commitsResIDsOnFullCompletion is P-05's blockspec.go half (audit-2026-09-10):
BlockSpec.generate claimed resBusy and forgot resIDs (R-00) but never committed them back on a
completed generation, so a --drafter turn always left the resident cache cold for whatever ran
next (a plain Generate turn, or another BlockSpec turn). Uses a real loaded model (embedResident
needs real weights) with a stubbed drafter host/trunk so the seed step runs for real and
opt.MaxTokens == 1 ends the generation immediately after it, at the ONLY exit that commits.
```

## TestBlockSpecGenerate_reusesDrafterContextOnExtension

Moved from `decoder/blockspec_reuse_test.go` (the comment above `TestBlockSpecGenerate_reusesDrafterContextOnExtension`) on 2026-10-09.

```text
TestBlockSpecGenerate_reusesDrafterContextOnExtension is P-05's deferred half (audit-2026-09-10):
a SECOND generate() call on the SAME *BlockSpec instance, whose prompt is a strict extension of
the first call's committed sequence, must reuse both the target's resident KV (reuseFrom > 0)
AND the drafter's own context (TruncateContext(reuseFrom), not TruncateContext(0)) — seeding
and fusing only the new suffix, not re-embedding the whole prompt.
```

## TestBlockSpecGenerate_declinesDrafterReuseAfterOtherWriter

Moved from `decoder/blockspec_reuse_test.go` (the comment above `TestBlockSpecGenerate_declinesDrafterReuseAfterOtherWriter`) on 2026-10-09.

```text
TestBlockSpecGenerate_declinesDrafterReuseAfterOtherWriter is the safety half of P-05's
deferred fix: resIDs matching alone is not enough to trust the drafter's own context — a
DIFFERENT writer (a plain Generate turn, or another BlockSpec instance) can commit a
token-identical resIDs without ever touching THIS BlockSpec's drafter, leaving rd's context
stale relative to what resIDs now claims. Simulates that by committing resIDs directly
(bypassing s.generate, exactly like a plain Generate turn would) between two calls into the
SAME BlockSpec instance with an otherwise-reusable extension — the second call must still
TruncateContext(0), not trust the mismatched drafter state.
```

## TestBlockSpec_stopSetAgreesWithPlainDecoding

Moved from `decoder/blockspec_stops_test.go` (the comment above `TestBlockSpec_stopSetAgreesWithPlainDecoding`) on 2026-10-09.

```text
C-11: THE BLOCK-SPEC LOOP'S STOP SET MUST EQUAL THE ONE PLAIN DECODING USES.

Every other speculative loop calls target.isStop(tok, sp). This one rebuilt the set from
Cfg.EOSIDs() alone — config.json's eos_token_id, missing generation_config.json's additions
(which resolveEOSIDs merges into m.eosIDs) and the caller's StopIDs (the chat template's stops on
a served request). On the shipped pairing that is {151645} against {151645, 151643}: an
<|endoftext|> came out as content, generation continued to <|im_end|> or max_tokens, and
streamTokens decoded the stop token into the response.

The invariant is AGREEMENT, so the test compares the two predicates directly rather than
re-listing what the set should contain — a list would be a second copy of the same belief.
```

## TestBlockSpec_roundWidthRespectsBothBudgets

Moved from `decoder/blockspec_stops_test.go` (the comment above `TestBlockSpec_roundWidthRespectsBothBudgets`) on 2026-10-09.

```text
M-13: a round commits up to `width` tokens, and the budget was checked once per ROUND.
```

## capabilityRow.Checkpoint

Moved from `decoder/capability_matrix_test.go` (the comment above `capabilityRow.Checkpoint`) on 2026-10-09.

```text
Checkpoint is the recommended checkpoint for this family, if there is one — the row
pull/registry.go reads to turn a short name into a repo + file + digest.

IT LIVES HERE, IN THE GENERATOR, because docs/capability-matrix.json is GENERATED and
freshness-gated by this very test. Hand-editing the JSON is silently undone by the next
`-update`, which is how the first version of this shipped and went red in CI. Data that is
meant to survive belongs beside the family definition, not in the rendered artifact.

Omitted entirely for a family with no recommendation, so the JSON stays free of null keys.
```

## recommendedCheckpoints.nemotronh.needs

Moved from `decoder/capability_matrix_test.go` (the comment above `recommendedCheckpoints.nemotronh.needs`) on 2026-10-09.

```text
R8 (docs/measurements/cold-user-2026-09-06-nobara-pc.md): this checkpoint used to load
with "tokenizer.ggml.pre=\"dbrx\" is not a known pre-tokenizer; falling back to
cl100k" on every pull — a registry-recommended entry cannot ship with a tokenizer
decline. Fixed (tokenizer/gguf.go's byteLevelKnobs gained a measured "dbrx" case); the
tokenizer tier is stated here rather than only in the runtime banner, since this is the
line a `pull`-before-you-load reader actually sees.
```

## recommendedCheckpoints.nemotronh.tools

Moved from `decoder/capability_matrix_test.go` (the comment above `recommendedCheckpoints.nemotronh.tools`) on 2026-10-09.

```text
Attempted 2026-09-07: this checkpoint's decode path is CPU-only staged (R9 — no GPU
dispatch exists for int4 in the staged path) and its prefill is sequential (no batched
CPU prefill for this arch), so a full `serve check` run did not finish in a reasonable
time on this box and was not force-completed. Not yet measured (R11 gate: never
guessed) rather than extrapolated from a faster model's result.
```

## recommendedCheckpoints.gemma4

Moved from `decoder/capability_matrix_test.go` (the comment above `recommendedCheckpoints.gemma4`) on 2026-10-09.

```text
R7 follow-up (docs/measurements/cold-user-2026-09-06-nobara-pc.md, and review feedback on
that finding): Gemma-4-26B-A4B — this project's own §B4/§B4.1 host↔VRAM C′-streaming
anchor, the model that section's whole design is measured against — DOES have a real,
official download: `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` (Google's own QAT q4_0 GGUF;
README's existing "bake any model" example at a smaller Gemma-4 tier already used this
exact repo's naming convention). The earlier pass concluded no download existed because
every reference IN THIS TREE'S OWN TESTS is a local, unpinned safetensors dir
(GOINFER_GEMMA4_26B=~/models/gemma-4-26b-a4b-it) — that conclusion was about this tree's
test fixtures, not about whether Google ever published GGUF weights, and it should have
searched HF directly before ruling the model out rather than only grepping this repo.
sha256/bytes: HF API blobs=true on the real repo, 2026-09-07 — then CROSS-VERIFIED against
an actual local copy already on this box (~/models/gemma4-26b-gguf/gemma-4-26B_q4_0-it.gguf,
byte count matches exactly) via TestRegistry_digestsMatchLocalFiles, same as gpt-oss-20b.
```

## recommendedCheckpoints.gemma4.tools

Moved from `decoder/capability_matrix_test.go` (the comment above `recommendedCheckpoints.gemma4.tools`) on 2026-10-09.

```text
Not yet measured against `serve check`'s tools rows (R11 gate: never guessed). This
checkpoint is multimodal (image-text-to-text); Gemma-4's tool template was a known
partial (M-20, docs/tasks/task-embed-and-harness-ux.md §3.1) but that was fixed
2026-09-02 (chat/gemma4_tools.go's gemmaValue/gemmaParseValue, byte-exact call_result) —
the remaining reason to run this deliberately rather than infer it is simply that it is
not yet measured, not a template gap.
```

## recommendedCheckpoints.gptoss

Moved from `decoder/capability_matrix_test.go` (the comment above `recommendedCheckpoints.gptoss`) on 2026-10-09.

```text
R7 (docs/measurements/cold-user-2026-09-06-nobara-pc.md): the README's own "bigger than
your GPU"/"bigger than your RAM" examples named a size class ("qwen3.5-35b-a3b",
"20-35B MoE") with no resolvable owner/repo anywhere — a cold user could not find any
checkpoint actually large enough to need -moe-cache-experts or -stream-weights. This
family's own description above already claims "resident on an 8 GB card via
-moe-cache-experts, validated on the real 20B" — gpt-oss-20b is that real 20B, so it is
the one this project can actually recommend rather than a size class it cannot deliver.
sha256/bytes are the repo's own git-lfs-recorded digest (HF API blobs=true on
ggml-org/gpt-oss-20b-GGUF, 2026-09-07), the same kind of source release-assets.yml's
pinned model URLs use — not re-hashed from a local download (12 GB; impractical for this
entry the way the release workflow's embedded-tier fetch step already does for its own,
much smaller, pins).
```

## recommendedCheckpoints.gptoss.tools

Moved from `decoder/capability_matrix_test.go` (the comment above `recommendedCheckpoints.gptoss.tools`) on 2026-10-09.

```text
Not yet measured against `serve check`'s tools rows specifically (R11 gate: never
guessed). docs/integrations/claude-code.md's own 2026-09-02 measurement is the closest
existing evidence for this size class: Qwen2.5-7B-Instruct (a different checkpoint, not
this registry) held a 25-tool-schema agent loop where a 1.5B "re-calls the same tool
forever" — worth running this row against once gpt-oss-20b's own harness-scale result
is recorded, rather than assuming a same-class result transfers across families.
```

## gpuResidentEligible

Moved from `decoder/capability_matrix_test.go` (the comment above `gpuResidentEligible`) on 2026-10-09.

```text
gpuResidentEligible is the FULL truth the "GPU-resident" column claims: at least one real
backend actually admits this arch (features.go's ResidentEligible — the same
decodeRunnerEligible-shape-AND-feature-coverage-AND-MoE-capacity gate hardware_matrix_test.go's
rows already use, so the two generated docs can never disagree about the same family) — NOT
decodeRunnerEligible() alone, which only answers "is the shape representable" and previously let
LFM2 read "GPU-resident: yes" for a family no backend could run at all (caught 2026-08-31, fixed
there with a special-case shape decline). olmo3/olmo_hybrid (and, it turns out on checking,
cohere/cohere2/mistral3/smollm3 too — five PRE-EXISTING rows, not new ones) hit the same failure
mode a different way, through undeclared features rather than a shape incompatibility; using the
existing, more complete ResidentEligible here instead of a second one-off special case fixes all
of them at once and can't drift from hardware-matrix.md's own answer again.
```

## TestCapabilityMatrix.pullcopy

Moved from `decoder/capability_matrix_test.go` (the comment above `pullJSPath`) on 2026-10-09.

```text
pull/ embeds a BYTE COPY of the json (pull/registry.go's go:embed cannot reach ../docs), which
TestRegistry_embeddedMatrixMatchesTheDoc keeps in lockstep. -update writes it too: a regeneration that left the
copy behind went red in CI twice (2026-09-29: 49f594d0's qwen3_next row, then the 26B text on top of it).
```

## TestLastWithMass_neverReturnsAMaskedToken

Moved from `decoder/chunked_decay_test.go` (the comment above `TestLastWithMass_neverReturnsAMaskedToken`) on 2026-10-09.

```text
N-02: a float-rounding miss must not return a MASKED token.

drawChunked and drawFull walked the cumulative distribution and, if rounding left `r` past the
final cumulative sum, fell through to the LAST INDEX. The vector is masked — top-k and top-p zero
the excluded tail — so the last index is very often a token the filter deliberately removed, and
returning it emits something the caller configured to be impossible. spec_sample.go's drawTree
already walked back to the last entry with mass; these two did not.

~1e-16 per draw, so a contract nick rather than a live bug — but it is the contract that top-k
and top-p exist to provide.
```

## TestValidateConfigBounds_hostileCountsAreRefusedBeforeAllocation

Moved from `decoder/config_bounds_test.go` (the comment above `TestValidateConfigBounds_hostileCountsAreRefusedBeforeAllocation`) on 2026-10-09.

```text
M-10: two unbounded per-layer allocations from UNTRUSTED metadata — the M16 fatal-OOM gap,
reopened on the paths that did not get ggufLayerCount.

The failure mode is worth naming precisely, because it is not a panic: a count like
68719476736 asks for an allocation that is under Go's maxAlloc, so the runtime does not
reject it — it tries, and the process dies with a FATAL "out of memory" that no recover()
can catch. The .giw loader HAS a recover() and its doc promises a typed error; neither
helps. So the bound has to come before the allocation, not around it.

Checked at resolveArchitecture, which is the single point every source of config reaches —
.giw, safetensors, GGUF — rather than at the two JSON call sites the audit names. Putting it
at the callers would be the "one predicate, N consumers" shape that produced a large share
of this audit's findings.
```

## TestCPUBatchS0c_widthAA

Moved from `decoder/cpu_batch_s0_test.go` (the comment above `TestCPUBatchS0c_widthAA`) on 2026-10-09.

```text
TestCPUBatchS0c_widthAA re-runs S0b's width sweep with an A/A control, because S0b's E2 was confounded: its
"default" and "w16" arms are the SAME configuration (the default width is GOMAXPROCS = 16 on nobara) and read 1.41x
apart, so position in the rotation (after a long serial arm the workers are parked) mattered more than width. Here
there is no serial arm, the default is measured twice (A and A'), every timed block is preceded by an untimed warm
block, and blocks are longer. A width result counts only if A/A' agree within a few percent.
```

## TestCPURoofline_w4a8Batch

Moved from `decoder/cpu_roofline_ab_test.go` (the comment above `TestCPURoofline_w4a8Batch`) on 2026-10-09.

```text
TestCPURoofline_w4a8Batch re-measures R-06's parked GOINFER_W4A8_BATCH (q/k/v as one fork/join,
gate/up as one) against the roofline accounting: the fork/join cost the per-component GB/s table
attributes to the small projections predicts a win of about 3-4 ms of a 54 ms 1.5B token.

	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run TestCPURoofline_w4a8Batch -v -count=1 -timeout 30m
```

## TestFastAttnPromptFloor

Moved from `decoder/cpufastattn_default_test.go` (the comment above `TestFastAttnPromptFloor`) on 2026-10-09.

```text
TestFastAttnPromptFloor pins the prompt-length floor, which is what keeps flipping the default
from changing the output of every SHORT request for no speed. Measured before it existed: an
8-token prompt diverged at the third generated token of 24 and never re-converged, while the
win at that length is nil (1.15x at K=512, and the flag's headline 2.28x is at K=8192).

NOTE what is deliberately NOT here: a MoE exclusion. 66d0a05 removed it after measuring it, and
re-adding one during this change would have quietly reversed a decision made with evidence.
```

## TestDecode_fusedScheduleCostsNothingPerToken

Moved from `decoder/decode_alloc_test.go` (the comment above `TestDecode_fusedScheduleCostsNothingPerToken`) on 2026-10-09.

```text
P-01 / M-03: A DECODE ALLOCATION GATE. P-01's own note says one would have caught M-03, so here
it is.

M-03: headWorkerPool allocated a fresh fusedScratch per pool slot, per layer, per decoded token,
on a path that never reads it — decode runs acc64=true and attendBatchedHeads computes
fusedOK = !useAcc64 && treeMask == nil. Since 84e0f13 made GOINFER_FUSED_ATTENTION default-on,
that was the shipped default. Measured before the fix, TotalAlloc per decoded token on
qwen2.5-coder-0.5b int8int8, interleaved:

	context 256    11.9 MB/token (fused on)  vs  74 KB/token (=0)   160x
	context 1024   42.6 MB/token (fused on)  vs  74 KB/token (=0)   573x

linear in context, zero-filled, GC-churned, and unread. After: identical with the schedule on and
off.

The gate asserts the INVARIANT rather than a byte count, which would be a benchmark pinned as a
test and would drift with every unrelated allocation: turning the default-on fused schedule on or
off must not change what a decode step allocates, because decode never uses it.
```

## BenchmarkDecode

Moved from `decoder/decode_bench_test.go` (the comment above `BenchmarkDecode`) on 2026-10-09.

```text
BenchmarkDecode times steady-state single-token decode (the chat case:
batch=1, greedy). It loads the 0.5B at int8int8, prefills a short fixed
prompt, then runs b.N forward+sample steps in the timed loop — the same
forward/runLayers + LM head + sampler path Generate drives. Reports tok/s.

DO NOT COMPARE TWO RUNS OF THIS TAKEN AT DIFFERENT TIMES. Interleave the arms
in ONE session, alternating, and discard the first sample of each process.

The number that settles it, measured on this box (Ryzen 7 3700X) on
2026-08-12 with the same model, same binary shape, same everything:

	morning session   ~0.93–0.97 tok/s
	afternoon session ~0.98–1.03 tok/s     — a ~5% SESSION-LEVEL SHIFT

Both effects under test that day were smaller than that drift: an aikit
regression at −2.96% and its fix at +0.43%. A sequential before/after would
have been dominated by whichever session each arm happened to land in, and
would have reported whatever the box's mood was. The first attempt at exactly
that comparison produced "−4%" and was worthless.

Interleaving is not rigour for its own sake here — it is the only reason
either result means anything. Pre-register the noise floor before running
(2.0% of the pooled mean, ≈2.4σ, from an 8-sample characterization), define
the warm-up discard in advance, and apply it identically to both arms.
Worked examples with raw samples: docs/measurements/aikit-v1.17.0-decode-ab.md
and -v1.17.1-decode-ab.md.

It is the perf campaign's regression guard; run with profiles:

	go test ./decoder -run '^$' -bench BenchmarkDecode -benchmem \
	  -cpuprofile cpu.out -memprofile mem.out -benchtime 5s

Skips cleanly without the model asset (set GOINFER_PREQUANT_GGUF or drop the
gguf in testdata), like the other model-dependent tests.
```

## BenchmarkDecodeAtDepth.setup

Moved from `decoder/decode_depth_bench_test.go` (the comment above `BenchmarkDecodeAtDepth.setup`) on 2026-10-09.

```text
Reach depth via the BATCHED prefill path (forwardLayersN, K=depth) — O(depth)
via one wide matmul sweep, not O(depth²) via depth sequential single-token
forwards (the naive setup timed out at depth 2048: ~248s, almost all setup).
This is also what production prefill actually does, so it is the
representative setup, not just the cheap one.
```

## TestDeltaNetSnapshotCost

Moved from `decoder/deltanet_snapshot_cost_test.go` (the comment at the top of the file) on 2026-10-09.

```text
MEASUREMENT ONLY — prices the narrow DeltaNet state snapshot for MTP speculation.
Builds nothing: no snapshot/restore is wired into any decode path, specRollbackSafe is
untouched, and nothing here is called from non-test code.

WHAT IS BEING PRICED, and it is not the thing docs/qwen3_5_moe.md deferred. That entry scoped
"state checkpoints" for cross-call PREFIX REUSE — restore to an arbitrary earlier position,
later, possibly across requests. Speculation needs something much weaker: snapshot immediately
before a verify, restore on rejection, discard. One buffer, one round deep, lifetime of
milliseconds. The two were bundled because they share a root cause (decoder/deltanet.go:deltaState: the
state is fixed-size and NOT position-truncatable), not because they are the same size of
problem.

	GOINFER_QWEN35_08B=~/models/qwen3.5-0.8b \
	  go test -tags realckpt ./decoder/ -run TestDeltaNetSnapshotCost -v -timeout 30m
```

## TestDequantHeads_bitIdenticalToAikit

Moved from `decoder/dequant_aikit_test.go` (the comment above `TestDequantHeads_bitIdenticalToAikit`) on 2026-10-09.

```text
TestDequantHeads_bitIdenticalToAikit is the gate that licensed deleting dequantHeads' arithmetic
in favour of linalg.DequantizeRowsInt8Into (aikit docs/task-goinfer-kernel-moves.md, M5).

It stays after the swap on purpose. dequantHeads now *is* the aikit call, so today this compares
aikit against a frozen copy of the body goinfer used to own — which is exactly what makes it a
regression gate rather than a tautology: if a future aikit bump changes DequantizeRowsInt8Into's
bits, this fails here, in goinfer, where the KV cache contract lives, instead of surfacing as a
moved golden somewhere downstream.

RAW BITS, never a tolerance: the difference this class of change produces is signed zero and
1-ULP, both of which a tolerance test cannot see (measured in M5 — fakequant's sym branch differs
from aikit's int4 round trip in exactly 7/256 elements, every one of them -0 vs +0).
```

## TestDFlashDraftScaling.newPerRound

Moved from `decoder/dflash_draftscale_test.go` (the comment above `TestDFlashDraftScaling.newPerRound`) on 2026-10-09.

```text
One round commits the anchor plus the accepted drafts. 5 is close to the measured mean
accepted at the widths that matter (3.97 at k=7, 4.24 at k=8), so it is the realistic
number of new context rows per round.
```

## TestDFlashGemma4_diag

Moved from `decoder/dflash_gemma4_diag_test.go` (the comment above `TestDFlashGemma4_diag`) on 2026-10-09.

```text
TestDFlashGemma4_diag localizes the Gemma-4 pairing's 0.00 acceptance.

477 rounds, 480 tokens, mean accepted EXACTLY 0.00. That number is not "the drafter
transfers badly" — a merely mismatched drafter still lands the occasional newline or
closing brace. Zero across 477 first-position proposals says the drafter is being fed
something it cannot use, and the pairing has no gate-1 reference dump to localize it,
which is why this prints intermediate quantities instead of asserting a threshold.

Already RULED OUT, so they are not re-checked here:
  - the prompt: goinfer's Gemma4 template renders id-identical to HF's canonical one
    (24/24 ids), and its trailing `<|channel>thought\n<channel|>` is the EMPTY thought
    block, i.e. Gemma 4's NON-thinking prompt — the analogue of Qwen3's suppressor.
  - the embedding scale: DrafterEmbedBlock goes through embedToken, which applies
    arch.EmbedScale, so the block embedding matches the target's own convention.
  - the LM head: DrafterHeadLogits handles tied heads, softcap and logit scale, and
    argmax is invariant to the monotonic parts regardless.

What is left is the hidden states, so that is what this measures. The tell is SCALE: the
drafter's fc maps taps*hidden -> hidden with weights trained on the reference's residual
magnitudes, and Gemma 4's residual stream is unusually large (embedding scale sqrt(2816)
~ 53). If our captured norms are far from the drafter's expectation, everything downstream
is saturated noise and 0.00 is the expected consequence rather than a mystery.

	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run DFlashGemma4_diag -v
```

## TestDFlash_pairingDialects

Moved from `decoder/dflash_pairing_test.go` (the comment above `TestDFlash_pairingDialects`) on 2026-10-09.

```text
The CONFIG-DIALECT gate, and it exists because one publisher ships more than one spelling.

z-lab's four DFlash drafters differ in every config-driven dimension — taps, trunk depth,
block width, hidden, vocab, mask id — and in TWO SPELLINGS the 4B-only loader could not read:

  - block_size NESTED in dflash_config (top-level on the other three)
  - RoPE as rope_parameters (flat rope_theta on the other three)

One publisher, two dialects, and the 4B-only loader reported the 35B as "block_size must be
>= 2, got 0" — a supported pairing looking broken. Third instance of this class in P10 after
granite's flat-only rope and nemotron's hybrid_override_pattern, which is why it is gated
rather than fixed quietly.

TABLE, NOT A ONE-OFF. This started as a single 35B test. It is a table now because the 35B
turned out to be the OUTLIER — gpt-oss and gemma-4 both spell block_size at top level — and a
single-case gate would have left that as an assumption. The point of the table is that a
future drafter is one row, and the row records the dialect it exercises.

Every case asserts the SHAPE the loader derived, not merely that loading succeeded: a loader
that silently defaulted a dimension would still "load".
```

## TestDFlashAcceptance.maxNew

Moved from `decoder/dflash_accept_test.go` (the comment above `TestDFlashAcceptance.maxNew`) on 2026-10-09.

```text
maxNew MUST be several blocks, and this is a hard error rather than a note because a
short run does not produce a noisy tok/verify — it produces a systematically INFLATED
one, from two independent mechanisms:

 1. END-OF-RUN OVERSHOOT. The loop runs while generated < maxNew, so the final round is
    counted in FULL even though it overshoots — up to block-1 extra tokens credited
    against one verify. The smaller maxNew/block is, the less that overshoot is
    amortized.
 2. AN UNREPRESENTATIVE SLICE OF THE ANSWER. A truncated run measures whatever part of
    the output it reaches, and that part is not the workload. The DIRECTION of this one
    is model-dependent, which an earlier version of this comment got wrong: it asserted
    a general "easy prefix" inflation, on the theory that a coding answer opens with
    near-deterministic boilerplate. The 4B is consistent with that (7.11 -> 6.75 -> 6.14
    as the run lengthens). The 35B does the OPPOSITE, and sharply — 4.77 at maxNew=16 to
    8.15 at 48 — because its easy region is the code block, which arrives AFTER a prose
    preamble that a 16-token run never gets past.

    This is why the guard is a hard error rather than a directional correction. The bias
    is not "short runs read high"; it is "short runs read UNPREDICTABLY", and at
    maxNew=16 the two pairings rank in the opposite order from maxNew=48.

MEASURED, not assumed. The Qwen3-4B code suite, same pairing, same quant, non-thinking:

	              Qwen3-4B      Qwen3.6-35B-A3B
	maxNew=16     7.11          4.77      <- 35B ranks WORSE
	maxNew=48     6.75          8.15      <- and BETTER, at the very next length
	maxNew=160    6.14 (recorded gate-2 number)

So the inflation is ~1.16x from 16 to 160, and ~1.10x still remains at 48. Real, and
enough to matter when two pairings are being compared, but MUCH smaller than a figure
this comment briefly carried: an earlier draft cited 2.45x by comparing 7.11 against
2.90. That was wrong — 2.90 is the DISCREDITED thinking-mode measurement this document
supersedes with 6.14, so the 2.45x conflated truncation with the thinking-mode error.
Worth leaving recorded: the correct-looking baseline to grab is the one printed nearby,
not the one that survived re-measurement.

The conclusion that finding supported is unaffected, because it rested on a MATCHED
control rather than on the recorded number: at maxNew=16 the 35B reads 4.77 against the
4B's 7.11, so the second pairing accepts LESS, not more. Any two tok/verify numbers
being compared must share this setting; prefer 160, which is what gate 2 is recorded at.
```

## TestDFlashAcceptance.progress

Moved from `decoder/dflash_accept_test.go` (the comment above `TestDFlashAcceptance.progress`) on 2026-10-09.

```text
PER-PROMPT PROGRESS. Without this the harness logs once per SUITE, at the
end, so a run that takes half an hour on a big target emits its pairing line
and then nothing — indistinguishable from a hang. That ambiguity already
cost one healthy run, killed on the belief it was stuck (the actual culprit
there was an unbuffered pipe, but a silent harness is what made the belief
plausible). One line per prompt makes progress observable and gives a
running estimate of the final figure.
WHAT THE TARGET ACTUALLY PRODUCED, not what the suite label says it was asked
for. The suite names describe the PROMPT; acceptance is a property of the
OUTPUT, and on a reasoning-by-default model the two diverge silently. gpt-oss
is the case in point: harmony has no non-thinking form, so its `code` run
measured the analysis channel — reasoning prose — and produced a number that
looked comparable to three code numbers and was not. That was caught only by
inspecting an anchor token in a separate diagnostic. Printing a preview makes
it visible in the run that produced the number.
```

## TestDFlashAcceptance.steadystate

Moved from `decoder/dflash_accept_test.go` (the comment above `TestDFlashAcceptance.steadystate`) on 2026-10-09.

```text
STEADY STATE, reported alongside the raw ratio because the raw one has a third
small upward bias on top of the two the maxNew guard covers.

`generated` is seeded at 1 per PROMPT — the anchor, which prefill produced and no
verify round paid for. Over R rounds with P prompts:
    generated = P + R + sum(accepted)
    tok/verify = 1 + P/R + mean(accepted)
so the raw ratio carries a +P/R term that has nothing to do with the drafter and
shrinks as the run lengthens: +0.23 on the 35B's 4.77 at 13 rounds, +0.33 on the
4B's 7.11 at 9 rounds — ~5% each. 1 + mean(accepted) is the prompt-count-independent
figure, and is what two pairings should be compared on.

The gate below still uses the raw ratio, deliberately: it is the definition every
recorded number in docs/spec/08 was measured under, and silently redefining a metric
to move a number past its own bar is the move this whole file exists to prevent.
```

## noThinkSuffix

Moved from `decoder/dflash_accept_test.go` (the comment above `noThinkSuffix`) on 2026-10-09.

```text
noThinkSuffix returns the ids for "<think>\n\n</think>\n\n" — what Qwen3's template emits
for enable_thinking=False.

THIS USED TO BE A LITERAL []int{151667, 271, 151668, 271}, pinned from Qwen3-4B, and that was
a bug waiting for the second pairing. Qwen3.6-35B-A3B has a 248320-token vocab in which
<think> is 248068 and 151667 is an unrelated token — so the literal would have quietly fed
the 35B four wrong tokens, depressing acceptance in a way that looks exactly like "the
drafter transfers badly to this target". Resolve it through the tokenizer that ships with
the target, and verify rather than trust: the encode must produce the <think>/</think> ids
the tokenizer itself reports.
```

## dflashRun.template

Moved from `decoder/dflash_accept_test.go` (the comment above `dflashRun.template`) on 2026-10-09.

```text
THE TARGET'S OWN TEMPLATE, detected — not ChatML assumed.

This was hardcoded to chat.ChatML(), which is right for both Qwen3 pairings and wrong
for the other two: Gemma-4 uses <|turn>/<|channel> markers and gpt-oss uses harmony.
Feeding a Gemma target ChatML would not error — it would just measure the drafter
against a prompt format the target never sees, and increment 2 already measured what
that costs (raw vs chat: 0/15 vs 10/15 accepted).

An unrecognized template is a HARD ERROR rather than the library's raw-completion
fallback. Falling back would produce a number rather than a failure, and a plausible
acceptance figure measured off a malformed prompt is the single most expensive failure
mode this harness has (it has now produced three retracted conclusions in P10).
```

## BenchmarkDFlashTrunk

Moved from `decoder/dflash_accept_test.go` (the comment above `BenchmarkDFlashTrunk`) on 2026-10-09.

```text
BenchmarkDFlashTrunk times ONE block draft — the cost that decides increment 4's
architecture. If the CPU trunk is cheap relative to a resident GPU target step, the
target can go resident while the drafter stays on CPU. If it is not, the drafter has to
be ported to the GPU too, and increment 4 is a much bigger build.

This is the question Lever 2 already answered once the hard way: the DRAFT was the wall,
not the verify — a CPU draft against a GPU target measured 0.11×. Measure before
building, not after.

	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run '^$' -bench DFlashTrunk -benchtime 10x
```

## TestDFlash_referenceParity.header

Moved from `decoder/dflash_test.go` (the comment at the top of the const block) on 2026-10-09.

```text
P10 kill-gate 1 (docs/spec/08): the Go DFlash trunk must match the upstream reference
on dumped fixtures BEFORE any acceptance measurement. This is the gate 05 cleared too
late — a structurally-correct-looking head sat at alpha~0.3 for weeks because nothing
compared it to the reference at the tensor level.

The fixture is the reference's own output, not a reimplementation of it:
scripts/pin_dflash_trace.py runs z-lab/Qwen3-4B-DFlash-b16's shipped dflash.py
(unmodified, MIT) and dumps the drafter's INPUTS (fused_context, block_in) alongside
every layer output. So this test feeds the reference's inputs to our forward and
compares layer by layer — a mismatch localizes to a layer instead of surfacing as
"the logits are wrong".

Fixtures: testdata/dflash_qwen3_4b_golden.json (stats + drafted ids, committed) and
testdata/dflash_qwen3_4b_ref.safetensors (full f32 tensors, committed by an explicit
.gitignore exception — it is the reference's output, not weights, and costs ~24 GB of
downloads to recreate).

TIER: T3-in-practice, NOT a CI gate. The drafter WEIGHTS (GOINFER_DFLASH_F32, 2.1 GB)
cannot be committed, so this skips without them — labelled per
docs/parity-coverage-policy.md's rule that a committed golden over an uncommitted
checkpoint is a T3 wearing a T1's clothes. Regenerate the weights with
scripts/convert_dflash_f32.py.

THE GATE IS FALSIFIABLE, checked rather than assumed (2026-08-15): causal-instead-of-
bidirectional block, norming the fused context like the block, roping q at the
block-local position, and dropping the per-head k_norm are each REJECTED by it. A
first-run cosine of exactly 1.0 is precisely when that check is worth running.
```

## TestDFlash_refusesV2

Moved from `decoder/dflash_test.go` (the comment above `TestDFlash_refusesV2`) on 2026-10-09.

```text
TestDFlash_refusesV2 pins the refusal added after the P15 step-(0)/(2) audit.

It is a CONFIG-ONLY fixture on purpose: the danger this guards is that a DFlash 2 checkpoint
is field-compatible with the v1 loader, so the check has to fire before any tensor is read.
Measured on the real incoai/Qwen3.8-27B-DFlash2 (2026-08-20): 76.2% of its tensors are
v1-shaped and every config key this loader reads is present, so it loaded WITHOUT ERROR and
silently discarded 914,309,120 bytes of dynamic-conv + candidate-selector weights.

Why that mattered enough to gate: DFlash verify is lossless, so the failure produces correct
tokens at a worse acceptance rate — slower than v1, with nothing in the logs. A wrong answer
gets noticed; a silent 20% throughput loss does not.
```

## declaredIdentityDispatch

Moved from `decoder/dispatch_census_test.go` (the comment above `declaredIdentityDispatch`) on 2026-10-09.

```text
declaredIdentityDispatch is every use of an identity predicate in a dispatch position, with why it
is legitimate. Definition sites and comments are excluded by the scanner.

KEYED ON CONTENT, NOT POSITION. The first version keyed on file:line and tripped the moment a
mechanical edit (G2's minmax rewrite) removed three lines above a declared site: (in mlp.go) line 356 became
:353, the site itself unchanged. A census that cries wolf on every reformat is a census someone
disables, and the thing it is supposed to notice — a member being named in a dispatch — is a
property of the CODE, not of where the code sits.
```

## TestDSpark_referenceParity.header

Moved from `decoder/dspark_test.go` (the comment at the top of the const block) on 2026-10-09.

```text
P10 kill-gate 1 for DSpark (docs/spec/08): the Go DSpark forward must match DeepSpec's own
implementation on dumped fixtures before any acceptance claim rests on OUR code. DSpark's
acceptance is already measured (5.76 / 5.73 / 3.04) but that was measured through DeepSpec's
loop — this is what lets goinfer claim it.

It doubles as the test of the SHARED-TRUNK claim. `decoder/dspark.go` reuses `blockTrunk`
rather than reimplementing the forward, on the finding that DeepSpec's `_forward_backbone`
and z-lab's `DFlashDraftModel.forward` compute the same thing. If that were wrong, the
per-layer comparison below would diverge at layer 0 — the same code passes the DFlash
fixture, so a DSpark failure here would localize the difference rather than hide it.

Fixtures: testdata/dspark_qwen3_4b_golden.json + dspark_qwen3_4b_ref.safetensors, from
scripts/pin_dspark_trace.py. Weights are an asset (GOINFER_DSPARK_F32) — T3-in-practice,
not CI, exactly as the DFlash gate.
```

## TestHiddenLast_refusesMoreTokensThanTheContextWindow

Moved from `decoder/embed_bound_test.go` (the comment above `TestHiddenLast_refusesMoreTokensThanTheContextWindow`) on 2026-10-09.

```text
C-07: HiddenLast HAD NO LENGTH BOUND, only a vocab one.

It preallocates KV for len(ids) positions and then runs one sequential forward per token with no
context to cancel it. So an over-long input is not slow, it is a ~114 GB allocation (28 layers,
kvDim 1024, 500k positions) plus attention over up to len(ids) keys per token, holding the
caller's mutex until the process is OOM-killed. The serving embedder's only bound was C-21's
1 MiB of BYTES — about 500k tokens of short words.

The bound is asserted HERE, where the cost is incurred, so a caller other than the serving
embedder cannot reintroduce it.
```

## embedInt4Fixtures

Moved from `decoder/embed_int4_loaders_test.go` (the comment above `embedInt4Fixtures`) on 2026-10-09.

```text
Options.EmbedInt4 must reach every safetensors loader. Until 2026-10-09 nine family loaders (gpt2, granite, nemotron_h, phi3, glm_ocr, spark2_5, llama4_text,
gpt_oss, internlm2) quantized their embedding and LM head with quant.embedding() and never saw the flag, so --embed-int4 (the default everywhere but Metal) did
nothing on them: the head stayed int8 whatever the flag said. Option D's registered HeadTable() assertion caught it on phi3 (docs/tasks/task-multimodal-support-2026-10.md).
```

## TestEmbedInt4_everyLoader.bar

Moved from `decoder/embed_int4_loaders_test.go` (the comment above the cosine bar) on 2026-10-09.

```text
The bar is 0.5 for a stated reason, not a guess: a mis-laid int4 table reads near ZERO (the planted defects of the G-S10 gates read 0.1 to -0.1), and
the working tables on the established generic path read 0.95 to 0.99 on these tiny fixtures (llama-tiny 0.982, gemma2-tiny 0.951); the eight here read
0.86 (glm-ocr-tiny: hidden 48, a ragged last group of 16; a ragged group reconstructs at the same ~9% relative RMS as 64 columns, measured) to 0.999.
```

## TestEnvVars_docAndCodeAgree

Moved from `decoder/envvars_registry_test.go` (the comment above `TestEnvVars_docAndCodeAgree`) on 2026-10-09.

```text
N-42: docs/env-vars.md is declared a Hard-tier contract by docs/api-tiers.md, and it had
drifted in BOTH directions — 40 variables read by production code were absent from it
(including the escape hatches for all four default-ON changes, each of which alters greedy
output), and one it documented was read by nothing at all.

This is the TestAssetRegistry_noDirectReads shape the audit's fix text asks for: the doc and
the code are compared to each other, so neither can move without the other. It deliberately
does NOT check what the doc SAYS about a variable — only that every variable exists in both
places. A wrong description is a different problem; a missing entry is this one.
```

## TestEnvVars_docAndCodeAgree.rows

Moved from `decoder/envvars_registry_test.go` (the comment above `rowSeen`) on 2026-10-09.

```text
One ROW per variable. The same var documented twice drifts: GOINFER_CUDA_FLASH_DECODE had an
"(unset = off)" row and an "ON (S=16)" row at once after its default flipped, and
GOINFER_CUDA_GRAPHS_SYNC sat in both the operator and the diagnostics sections.
```

## TestEnvVars_docAndCodeAgree.operator

Moved from `decoder/envvars_registry_test.go` (the comment above `operatorDoc`) on 2026-10-09.

```text
M-56 (audit-2026-09-10.md): everything ABOVE the first "not contract" heading is an
operator-facing promise (docs/api-tiers.md's Hard tier); a var documented only there but
referenced by nothing except _test.go files is exactly GOINFER_GEMMA4_RESIDENT's shape —
a bring-up gate that went silently unread while its doc row kept promising an effect. The
phantom check below already tolerates a test-only reference for the diagnostics section
(test-only IS what "not contract" means there); operatorDoc is scanned separately so that
same tolerance can't hide a dead operator knob again.
```

## TestConfigEOSIDs_nullIsAbsent

Moved from `decoder/eos_null_test.go` (the comment above `TestConfigEOSIDs_nullIsAbsent`) on 2026-10-09.

```text
A config.json whose eos_token_id is JSON null (transformers writes the key with None when a model leaves it to
generation_config.json: Qwen3-ASR's text_config does) must have NO config-side stop id. json.Unmarshal of null into an int
succeeds and leaves 0, so EOSIDs() once returned [0], and token id 0 ended generation: in Qwen's vocabulary id 0 is "!",
so every Qwen3-ASR transcription stopped at its first "!" (G-S14c4, 2026-10-08: 2 of the 2 LibriSpeech clips with one).
```

## TestLoad_exactPrefillOptionIsAModelProperty

Moved from `decoder/exactprefill_test.go` (the comment above `TestLoad_exactPrefillOptionIsAModelProperty`) on 2026-10-09.

```text
TestLoad_exactPrefillOptionIsAModelProperty is M-26's chokepoint gate (docs/audit-2026-09-10.md),
revised 2026-09-24. decoder.Options.ExactPrefill used to be applied by Load setting the three
fast-prefill env vars — process-global and never undone, so every model loaded later in the same
process inherited it. It is now recorded on the Model and consulted by each backend's switch next
to its env var (CPU: Model.cpuFastAttention; CUDA: at resident build; Metal: on its resident). This
proves Load reports it on the model, applies it to this model's CPU prefill, and writes nothing to
the environment. The two-model inheritance case is TestExactPrefill_isPerModel.
```

## TestF16Rules_agree

Moved from `decoder/f16_rules_test.go` (the comment above `TestF16Rules_agree`) on 2026-10-09.

```text
TestF16Rules_agree pins that aikit's int4 scale storage (linalg.F32ToF16) and goinfer's F16Bits — the rule
every resident GPU backend converts int4 scales with — are the same conversion, bit for bit, so a CPU
weight and a GPU upload of the same tensor carry identical scales. Before aikit stored int4 scales as
binary16 this was a third copy (the GOINFER_INT4_F16_SCALES diagnostic's f32ToF16bits); all three were
checked equal on this same input before that copy was deleted.
```

## archFeatureProfile.smollm3

Moved from `decoder/features_test.go` (the comment above `archFeatureProfile.smollm3`) on 2026-10-09.

```text
SmolLM3: llama-shaped plus per-layer NoPE (FeatNoPE). G5 (docs/tasks/task-gpu-paths-2026-09.md):
cuda+metal now declare FeatNoPE (RopeInvFreqLayer zeroes the NoPE layers' invFreq table,
no new kernel), so this is SmolLM3's ONLY required feature and it now reaches both —
see the admission golden below. cohere2 also needs FeatNoPE, plus FeatLayerNorm/
FeatLogitScale/FeatParallelBlock — all now declared on both backends too (G5's last row).
```

## archFeatureProfile.olmo3

Moved from `decoder/features_test.go` (the comment above `archFeatureProfile.olmo3`) on 2026-10-09.

```text
Olmo 3: NormPostOnly + QKNormWhole, plus the standard sliding-window/YaRN features every
backend already has. G5 (docs/tasks/task-gpu-paths-2026-09.md) declares both on cuda+metal, so
this now reaches both.
FeatPerLayerRoPE IS needed (restored 2026-09-12, docs/audit-2026-09-10.md G-03/G-04's
disposition): YaRN applies to full_attention layers only, sliding_attention gets plain
RoPE at the same theta — genuinely different local/global tables. A prior revision of this
comment ("0b0f5c9") claimed the opposite from reading modeling_olmo3.py's forward rather
than calling it; `Olmo3RotaryEmbedding.forward` in the pinned transformers==5.15.0 takes an
explicit layer_type argument and returns different (cos, sin) per call, which is the tell
that a single shared table cannot be what runs. See registry.go's olmo3Architecture comment
for the full account.
```

## archFeatureProfile.olmo_hybrid

Moved from `decoder/features_test.go` (the comment above `archFeatureProfile.olmo_hybrid`) on 2026-10-09.

```text
Olmo Hybrid: qwen3_5's own FeatDeltaNet (shared math), plus olmo3's FeatPostOnlyNorm/
FeatQKNormWhole (its full-attention layers only, but the arch-level check reads
NormPlacement/QKNormWhole model-wide, not per-layer) and FeatNoPE (layerNoPE set
unconditionally — the release has no RoPE at all). All four are now declared on cuda+metal
(FeatDeltaNet/FeatNoPE already were; G5 added the other two), so this now reaches both too —
though getting there also surfaced a separate bug (both backends assumed every
qwen35Params-carrying family's full-attention layer used qwen3.5's double-width q-gate
scheme; Olmo Hybrid's is plain), fixed via a new Architecture.qwen35.AttnGate field.
```

## archFeatureProfile.cohere

Moved from `decoder/features_test.go` (the comment above `archFeatureProfile.cohere`) on 2026-10-09.

```text
Cohere's rotation is GPT-J PAIRWISE (rope_gptj), not NeoX: FeatPairwiseRoPE, added 2026-10-01
after the CUDA resident (which ran it on NeoX kernels) measured wrong on real Command-R7B and
Aya at int4. Declared by CUDA only; metal and webgpu DECLINE these families until their rope
kernels have a pairwise variant.
```

## archFeatureProfile.spark2_5

Moved from `decoder/features_test.go` (the comment above `archFeatureProfile.spark2_5`) on 2026-10-09.

```text
spark2_5: same attention-output-gate CPU-only forcing function as Laguna (FeatAttnOutputGate
— sigmoid here, softplus there; no resident backend implements either). Also gets
FeatGatedGELU "for free" from residentFeatures' derivation (!NonGatedMLP && Act != ActSiLU),
which is worth flagging explicitly: this is the FIRST family combining a gated MLP with
ActGelu (exact erf) rather than ActGeluTanh — GatedActResident passes the raw ordinal to
CUDA's glu_quant kernel, which has never been asked for this activation on the gated path,
so whether it's even correct there is UNVERIFIED. It doesn't matter for now: FeatAttnOutputGate
alone already forces CPU-only, so this combination never reaches a GPU kernel — but the next
family with a gated exact-GELU MLP and NO attention gate would need that verified for real,
not assumed safe by this precedent.
```

## archFeatureProfile.gemma

Moved from `decoder/features_test.go` (the comment above `archFeatureProfile.gemma`) on 2026-10-09.

```text
Gemma — VERIFIED against the real checkpoints via RequiredResidentFeatures (an earlier
hand-written guess here was wrong on three counts: it missed per-layer-rope / qk-norm /
sliding-window, and claimed rms-add-one for gemma4, which has RMSAddOne=false).
gemma3/gemma3_text are resident on CUDA. gemma4 needs the FINAL-logit softcap (30) — one
host-side tanh, which CUDA now ships (FeatFinalLogitSoftcap, 9a-P2) — not the attention
softcap. So it is feature-compatible with CUDA; its OWN forward (per-layer head_dim / K=V)
is what the resident geometry bridge addresses, and dense admission is env-gated
(GOINFER_GEMMA4_RESIDENT) in decodeRunnerEligible — a separate gate from this feature set.
Gemma 1 / CodeGemma and Gemma 2 (2026-09-30, gemma12.go). Gemma 2 needs the attention-score softcap, a per-layer
kernel no resident backend ships, so it is admitted nowhere and runs on the CPU until one does.
```

## archFeatureProfile.gemma4_text

Moved from `decoder/features_test.go` (the comment above `archFeatureProfile.gemma4_text`) on 2026-10-09.

```text
gemma4_text is the 26B-A4B MoE variant: gemma4's feature set + FeatMoE. It is
FEATURE-compatible with CUDA, but Gemma 4's MoE is the parallel dense‖MoE shape the
generic FeatMoE kernel cannot express, so decodeRunnerEligible declines a.MoE != nil
until Split B lands the delta — the feature set is necessary, not sufficient.
```

## admissionGolden.cohere

Moved from `decoder/features_test.go` (the comment above the cohere golden row) on 2026-10-09.

```text
Cohere / Command-R + Cohere2 / Command-R7B: G5's last row (docs/tasks/task-gpu-paths-2026-09.md).
FeatParallelBlock reuses the pre-attn norm's already-quantized activation as the MLP's
input too (no new kernel, a sequencing change); FeatLogitScale is a host-side multiply
after readback (the same shape as FeatFinalLogitSoftcap). FeatLayerNorm was already
declared on metal (GPT-2); cuda's twin (layernorm_quant, cuda/glue.cu) is genuinely new —
this backend had no mean-centered norm before. cohere2 needs FeatNoPE/FeatSlidingWindow
too, both already declared on both backends from earlier G5 rows, so it reaches the same
two backends cohere does.

CORRECTION 2026-10-01: CUDA ONLY now. Their rotation is GPT-J pairwise and the metal rope
kernel is NeoX half-split; the CUDA resident had the same defect until cuda/rope_pairwise.cu
(real Command-R7B / Aya-expanse-8B at int4, resident vs CPU: worst per-position cosine -0.075 /
-0.041, exact at position 0 only). The old gates used flat 0.02-std fixtures where a wrong
rotation reads 0.9997. FeatPairwiseRoPE is declared by cuda alone; metal declines these families
to the CPU path until the Mac session ports the kernels.
2026-10-09: Metal too (docs/tasks/task-metal-pairwise-rope-2026-10.md: rope_pw and twins, metal.TestPairwiseRoPEResidentParityMetal
and the real-checkpoint metal.TestPairwiseRoPERealMetal). WebGPU still declines.
```

## admissionGolden.gemma

Moved from `decoder/features_test.go` (the comment above the gemma golden row) on 2026-10-09.

```text
G6 (docs/tasks/task-gpu-paths-2026-09.md): webgpu declares FeatEmbedScale/FeatFinalLogitSoftcap/
FeatSandwichNorm/FeatGatedGELU — gemma3 (uniform head_dim) now reaches it for real,
verified against a genuine non-seeded checkpoint (testdata/gemma3-vl-tiny's text tower,
gpu.TestGemma3ResidentParityWebGPU, minCosine 0.9998). gemma4/gemma4_text/
gemma4_unified_text are FEATURE-compatible the same way gemma4_text's own MoE note below
already says feature-compatible isn't sufficient — dense Gemma 4's local/global head_dim
split needs a per-layer geometry seam this simplified feature-list model cannot express
(decoder.Model.PerLayerGeomOK, a residentMoECapacityOK-shaped runtime check outside the
ResidentFeature taxonomy), which webgpu's runLayer.ghd/gnKV fields exist for but
gpu/residency.go's per-layer builder never populates for any family. The REAL runtime
(decoder.ResidentEligible, TestGemma4Admission_unconditional) correctly declines gemma4 on
webgpu despite this table showing it feature-admitted — this golden intentionally tracks
the simplified feature-only model, not full runtime truth, matching this file's own MoE-cap
precedent (deepseek_v2/kimi_k2 below, admitted by feature but capped elsewhere).
```

## admissionGolden.gpt_oss

Moved from `decoder/features_test.go` (the comment above the gpt_oss golden row) on 2026-10-09.

```text
gpt_oss reaches ALL THREE backends on real end-to-end evidence now, which is not how it
looked for most of G7's life. metal declared on the tiny fixture (TestGptOssResidentParity,
cosine 0.9989); cuda declared 2026-08-31 on the REAL 20B, resident on an 8 GB card through
--moe-cache-experts (7/8 argmax-exact, min cosine 0.996392) — the stronger evidence of the
two, an odd inversion worth noting since the tiny fixture stopped reproducing the last
defect it had to catch. webgpu declared 2026-09-08 (G6): the sink threaded through every
attention kernel plus three brand-new MoE kernels (gpu/moe.go), verified against the real
tiny fixture (gpu.TestGptOssResidentParityWebGPU, minCosine 0.9943 — the same 0.95 floor
Metal's own gate uses on this fixture).
```

## admissionGolden.glm_ocr

Moved from `decoder/features_test.go` (the comment above the glm_ocr golden row) on 2026-10-09.

```text
GLM-OCR: CUDA only (2026-10-01, cuda/rope_pairwise.cu; cuda.TestGlmOcrResidentParityCUDA).
Admitted with the NeoX kernels it read resident-vs-CPU cosine -0.34 on glm-ocr-tiny; Metal and
WebGPU still have only NeoX rope kernels and decline (FeatPairwiseRoPE + FeatPairwiseMRoPE).
2026-10-09: Metal too (the pairwise twins; metal.TestGlmOcrResidentParityMetal). WebGPU still declines.
```

## admissionGolden.mellum

Moved from `decoder/features_test.go` (the comment above the mellum golden row) on 2026-10-09.

```text
mellum reaches cuda and metal by the SAME coupling and with VERY DIFFERENT evidence, so
the two are not interchangeable rows:
  - cuda (added 2026-08-31, G7): FeatRopeMscale was declared for gpt-oss's YaRN, and
    mellumArchitecture needs exactly {MoE, PerLayerRoPE, QKNorm, RopeMscale, SlidingWindow}
    — the other four were already declared, so the fifth admitted Mellum for free. That
    coupling was PRE-REGISTERED as a trap and then discharged by measurement rather than
    waived: TestMellumResidentParityCUDA on a real 4-layer slice gives 7/11 argmax-exact,
    0 hard fails, min cosine 0.994600. Mellum on CUDA is a MEASURED admission.
  - metal: still the undischarged version of that same side effect (G10) — no Mellum
    checkpoint was reachable there, so it was resolved by an owner call. G11 tracks the
    real-weight Metal proof. It is here because it IS what the code does, not because it
    is trusted.
```

## admissionGolden.qwen3_5

Moved from `decoder/features_test.go` (the comment above the qwen3_5 golden row) on 2026-10-09.

```text
The Gated-DeltaNet family collapses to the backends that implement BOTH the recurrence AND
the fused attention output gate (FeatDeltaNet) — the whole point of adding it as one taxon
bundling both departures rather than two. CUDA and WebGPU landed it first (2026-08-19/20);
Metal landed the recurrence kernels + the qGate softmax-layer wiring + a whole-model gate
against qwen3_5-tiny (worst cosine 0.9886, drift 0.0081, replay-after-Reset self-cosine
1.0 — TestQwen35ResidentParityMetal) and now admits too. CUDA still lacks it.
```

## TestResidentBackendFeatures_noOverclaim.cuda

Moved from `decoder/features_test.go` (the comment above the cuda row) on 2026-10-09.

```text
cgo-free CUDA: dense + QK-norm + sliding window + the Gemma set + partial rotary + MoE
(routed via mixtral-tiny, ungated shared expert via glm-tiny) + the FINAL-logit softcap
(host-side tanh in step(), 9a-P2 — Gemma 4). FeatRopeMscale landed 2026-08-31 (G7): the
YaRN attention_factor is folded into cos/sin inside all three rope kernels (glue.cu,
gemv_fwd.cu, prefill_batched.cu) with per-layer wiring from RopeMscaleLayer, and it is
gated by TestRopeMscale plus a real-weight TestMellumResidentParityCUDA (4-layer slice:
7/11 argmax-exact, 0 hard fails, min cosine 0.994600) — evidence first, declaration
after. FeatAttnSink + FeatOutBias joined 2026-08-31 (G7), and ONLY after a real
gpt-oss-20b forward ran resident on an 8 GB card via --moe-cache-experts: 7/8
argmax-exact, min cosine 0.996392 (TestGptOssResidentParityCUDA). 2224441 declared
FeatAttnSink once on kernel-level evidence and was correctly reverted; the difference
this time is the end-to-end run, which found three silent wiring defects no kernel test
could see. Still no per-layer rotary WIDTH, no ATTENTION softcap (FeatAttnLogitSoftcap —
a per-layer kernel), no MLA/SSM, and no GATED shared expert (Qwen2-MoE)... which is now
expressed as FeatMoEGatedShared (which CUDA does NOT declare), so the Qwen2-MoE decline
is in the shared taxonomy, not a hand-coded check.
G5 (docs/tasks/task-gpu-paths-2026-09.md) added seven more, across all five rows: FeatNoPE
(SmolLM3), FeatAttnTemp (Ministral 3), FeatPostOnlyNorm + FeatQKNormWhole (Olmo 3/Olmo
Hybrid), FeatLayerNorm + FeatParallelBlock + FeatLogitScale (Cohere/Command-R +
Cohere2/Command-R7B, the last row — FeatLayerNorm is a genuinely new kernel here,
cuda/glue.cu's layernorm_quant; the other two are wiring, no new kernel).
```

## TestResidentBackendFeatures_noOverclaim.cudaPairwise

Moved from `decoder/features_test.go` (the comment above the pairwise row of the cuda set) on 2026-10-09.

```text
2026-10-01: GPT-J pairwise rotation (cuda/rope_pairwise.cu), declared with the peaked-attention
gates (cuda.TestPairwiseRoPEResidentParityCUDA, cuda.TestGlmOcrResidentParityCUDA) and the
real Aya/R7B gate. metal declares both since 2026-10-09 (its own twins and gates); webgpu does not.
2026-10-07: FeatGemma4EModel (S1 on CUDA, docs/tasks/task-multimodal-support-2026-10.md): the PLE branch from existing
kernels, KV-shared layers aliasing their source's cache, per-layer FFN widths; G1c/G2c on the tiny E-model, graphs bit-exact.
```

## TestResidentBackendFeatures_noOverclaim.webgpu

Moved from `decoder/features_test.go` (the comment above the webgpu row) on 2026-10-09.

```text
G6 (docs/tasks/task-gpu-paths-2026-09.md) added six more: FeatEmbedScale (free — decoder
already applies it host-side), FeatFinalLogitSoftcap/FeatOutBias (existing-kernel
wiring), FeatSandwichNorm (defeats the fused residual epilogue, no new kernel),
FeatGatedGELU (a genuinely new kernel pair — this backend had no GELU-tanh-gated
activation before), FeatAttnSink (gpt-oss — the sink threaded through every attention
kernel plus three brand-new MoE kernels, the largest single item in G6).
```

## TestResidentBackendFeatures_noOverclaim.metal

Moved from `decoder/features_test.go` (the comment above the metal row) on 2026-10-09.

```text
GPT-2 (2026-08-18): LayerNorm/non-gated-MLP/learned-pos/out-bias all landed as real
kernels + full BuildResident/encodeLayer/encodeAttention wiring, validated end-to-end
against the real checkpoint (TestGPT2ResidentParity, min cosine 0.999) — not a
declaration ahead of the evidence. gpt-oss's attention sink + clamped-SwiGLU MoE +
custom router + YaRN rope mscale also landed (TestGptOssResidentParity, min cosine
0.9989 on the tiny fixture) — FeatRopeMscale ALSO admits Mellum as a documented side
effect (see docs/queue-correctness.md G10/G11): declared anyway on explicit user call,
Mellum's own real-weight Metal gate is tracked as G11, not yet landed. Gated-DeltaNet
mixer + fused attn output gate (deltanet.go/deltanet_kernels.go) landed with its own
end-to-end whole-model gate (TestQwen35ResidentParityMetal, qwen3_5-tiny: worst cosine
0.9886, drift 0.0081, replay-after-Reset self-cosine 1.0) — not ahead of it.
G5 (docs/tasks/task-gpu-paths-2026-09.md) added six more, across four rows (FeatLayerNorm was
already here, from GPT-2): FeatNoPE (SmolLM3), FeatAttnTemp (Ministral 3),
FeatPostOnlyNorm + FeatQKNormWhole (Olmo 3/Olmo Hybrid), FeatParallelBlock +
FeatLogitScale (Cohere/Command-R + Cohere2/Command-R7B, the last row — reusing GPT-2's
layernorm_quant and Gemma's softcap-shaped host readback respectively, no new kernel).
S1 (docs/tasks/task-multimodal-support-2026-10.md) adds FeatGemma4EModel: encodePLE (ple_gelu_mul plus
existing kernels), KV-shared layers aliasing their source's cache, per-layer FFN widths.
```

## TestResidentFeatures_derivationMatchesProfile

Moved from `decoder/features_test.go` (the comment above `TestResidentFeatures_derivationMatchesProfile`) on 2026-10-09.

```text
TestResidentFeatures_derivationMatchesProfile ties the two sources of truth together (C6):
the hand-written archFeatureProfile (the classification forcing-function) and residentFeatures()
(the derivation that drives the generated hardware matrix AND the runtime RequiredResidentFeatures).
They must agree for every registered arch — otherwise they silently disagree, as they did on
Mellum (a layer-0-only yarn-mscale sample missed its full-layer YaRN) and qwen2_moe (the gated
shared expert). Hardware-free — reads declared sets, runs in CI with no GPU.
```

## TestResidentMoECapacity_routerCap

Moved from `decoder/features_test.go` (the comment above `TestResidentMoECapacity_routerCap`) on 2026-10-09.

```text
TestResidentMoECapacity_routerCap is the M22 taxonomy gate (kimi_k2 doc-drift). WebGPU/CUDA score
experts into a fixed-size array (256) and groups into 32; a model past that routes on only the
first N — plausible-looking wrong output — so ResidentEligible must decline it even though every
FEATURE it needs is implemented. This is what keeps the generated hardware matrix honest: before
residentBackendMoECap, the feature-only predicate showed Kimi K2 (384 experts) WebGPU-resident
while the runtime (gpu/backend.go, M22) declined it.
```

## TestResidentMoECapacity_routerCap.metal

Moved from `decoder/features_test.go` (the comment above `TestResidentMoECapacity_routerCap.metal`) on 2026-10-09.

```text
Metal is DECLARED in the cap map for the first time (its shader really is 256 and rejects
above it, metal/moe.go's router kernel) — the map previously claimed "absent = uncapped", which was false.
```

## TestA3FastAttentionDivergence.floor

Moved from `decoder/a3_divergence_test.go` (the comment above the `want, wantName` selection) on 2026-10-09.

```text
THE DEFAULT FLIPPED 2026-08-31, so what "unset" must equal now depends on K.
Below fastAttnMinPrompt the f32 path is floored off and unset is still exactly acc64;
at or above it, unset IS the f32 path. Asserting both halves pins the floor itself,
not just the default — a floor set to 0 or to MaxInt would fail here rather than
silently change what every short request returns.
```

## representativeConfig.olmo_hybrid

Moved from `decoder/capability_matrix_test.go` (the comment in `representativeConfig`, the olmo_hybrid case) on 2026-10-09.

```text
Olmo Hybrid: qwen3_5's Gated DeltaNet (3-of-4 layers) + olmo3's own full-attention
shape (1-of-4), with MIXED NormPlacement per layer kind (the reason G2 paused for a
decision) and NO RoPE at all (rope_theta: null, the real release's own value) — so
this config exercises NoPositionEncoding for real, not the trivial "never checked"
case. MHA (NumKVHeads == NumHeads), matching the one released size fetched.
```

## TestEmbedInt4Knob

Moved from `decoder/embedint4_test.go` (the comment above `TestEmbedInt4Knob`) on 2026-10-09.

```text
TestEmbedInt4Knob gates Options.EmbedInt4 (#3): with int4 quant, the default pins
the embed/head table to int8 (logit-critical), but the opt-in knob relaxes it to
int4 — halving the largest resident tensor on a big-vocab small model. Asserts the
embed table's precision actually changes (int8 → int4) and both models still
generate. The quality cost is recorded in docs (≈2.3 pts top-1); this test just
gates that the knob takes effect and is lossless-to-load.
```

## admissionGolden.smollm3

Moved from `decoder/features_test.go` (the comment above the smollm3 golden row) on 2026-10-09.

```text
G5 (docs/tasks/task-gpu-paths-2026-09.md): FeatNoPE declared on cuda+metal (RopeInvFreqLayer
zeroes the NoPE layers' invFreq table, no new kernel) — smollm3's ONLY required feature,
so it now reaches both.
```

## admissionGolden.olmo3

Moved from `decoder/features_test.go` (the comment above the olmo3 golden row) on 2026-10-09.

```text
Olmo 3 / Olmo Hybrid: G5 declares FeatPostOnlyNorm+FeatQKNormWhole on cuda+metal (the
pre-norm skip reuses the already-shipped quant_vec kernel unchanged; the whole-vector
qk-norm reuses the already-shipped per-head qk_norm kernel with a collapsed grid — no new
kernel or PTX either way). olmo3 needed only those two (FeatRopeMscale/FeatSlidingWindow
were already declared); olmo_hybrid's FeatDeltaNet+FeatNoPE were already declared too, so
this is its ONLY remaining requirement — but reaching residency at all surfaced a real,
separate bug (both backends assumed every qwen35Params-carrying family's full-attention
layer used qwen3.5's own double-width q-gate scheme; Olmo Hybrid's is plain), fixed
alongside this via a new Architecture.qwen35.AttnGate field.
```

## admissionGolden.mistral3

Moved from `decoder/features_test.go` (the comment above the mistral3 golden row) on 2026-10-09.

```text
Ministral 3: G5 (docs/tasks/task-gpu-paths-2026-09.md) declares FeatAttnTemp on cuda+metal —
rope_kv/rope2's new qTempScale param (Q-only, post-rotation) — its ONLY required feature,
so it now reaches both, same shape as smollm3/FeatNoPE above. Not webgpu: unimplemented
there.
```

## TestResidentBackendFeatures_noOverclaim.known

Moved from `decoder/features_test.go` (the comment above the `known` taxonomy map) on 2026-10-09.

```text
G5 (docs/tasks/task-gpu-paths-2026-09.md): FeatAttnTemp/FeatPostOnlyNorm/FeatQKNormWhole
landed (rows 2-3) without being added here — the exact N-12 omission repeated, caught
only when the last row's full-suite run exercised this test again.
```

## TestResidentMoECapacity_routerCap.cap

Moved from `decoder/features_test.go` (the comment above the 512 cap check) on 2026-10-09.

```text
The cap was raised 256 -> 512 (MOE_MAX_E / MAXE) so Kimi-K2's 384 is now ADMITTED on the two
backends whose router scratch was widened. This is the point of the change: K2 is the shipped
family the old cap declined.
```
