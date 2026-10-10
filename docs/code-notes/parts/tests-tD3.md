# decoder: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `decoder`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## TestP19FusedAttention

Moved from `decoder/p19_fused_attn_test.go` (the comment above `TestP19FusedAttention`) on 2026-10-09.

```text
P19 — does the FUSED (FlashAttention-style) schedule beat the materialized one?

Step 0 established that goinfer materializes: G20 tiles over QUERY ROWS only,
`scores` is tile x nKeys, and attendBatchedHeads forbids a key-dimension split
because it "would re-associate the softmax denominator and the AV fold".
So the N-wide score row makes three trips through memory per tile — written by
QK^T, read-and-rewritten by the softmax, read again by scores*V. At the
production tile budget (attnScoreTileBytes = 8 MiB) that block is far past L2
on either machine, so the traffic is real.

This measures whether removing it wins, WITHOUT touching production code. It is
a prototype for a decision, not an implementation.

THE COMPARISON IS AT FIXED PRECISION, and that is the item's own decision rule:
"a win that only appears with f32 enabled is A3's win, not this item's". Both
arms are f32 over identical inputs, so the delta is the SCHEDULE and nothing
else. Gathers sit outside the timed region because both arms pay exactly the
same ones; including them would only dilute the effect being measured.

PRE-REGISTERED BAR (written before the first run, and before the kernel):

	>= 1.30x at K=8192 shapes -> fusion clears its own bar, prototype in production
	<  1.10x                  -> close the item
	1.10-1.30x                -> AMBIGUOUS, parks

NOT bit-identical by construction: the running-max rescale re-associates. That
is the item's stated cost, the same category as --cpu-fast-attention. So
correctness is a tolerance, declared here rather than after: cosine >= 0.9999
against the materialized arm.

Masking: neither arm masks (full attention over nKeys). Identical in both, so
the ratio is unaffected; it means the absolute times are not production TTFT.
```

## TestP19FusedAttention.vblk

Moved from `decoder/p19_fused_attn_test.go` (the comment above `TestP19FusedAttention.vblk`) on 2026-10-09.

```text
V, pre-transposed PER BLOCK and laid out block-major, ONCE. MatmulBT
computes a·bᵀ so the AV block needs b as [hd, n], and a column range of
the [hd, nKeys] `vt` is not contiguous. The first draft of this built
that transpose inside the timed loop, which would have charged the fused
arm for a layout cost the schedule does not actually imply — and made
the arm being tested lose for the wrong reason.
```

## TestP19FusedAttention.serial

Moved from `decoder/p19_fused_attn_test.go` (the comment above `TestP19FusedAttention.serial`) on 2026-10-09.

```text
SERIAL CONTROL. MatmulBT fans out over its N output columns, and the two arms
present very different N: the materialized QK^T has N=nKeys=8192, the fused
one N=kb (128..1024). So a fused loss could be a PARALLELISM artifact --
fewer columns to shard, more fork/joins -- rather than a property of the
schedule. Running both arms serial as well separates those. This repo has
already published one ratio that was mostly core count (G24's first pass,
17.6x against a documented ~3.7x), so the control is not optional.
```

## TestP19FusedAttention.rowparallel

Moved from `decoder/p19_fused_attn_test.go` (the comment above `TestP19FusedAttention.rowparallel`) on 2026-10-09.

```text
---------------------------------------------------------------------
ROW-PARALLEL ARMS — the control for this file's own stated caveat.

The arms above showed fusion losing 0.690x in parallel and washing (1.031x)
serially, and attributed the gap to MatmulBT's fan-out being over N OUTPUT
COLUMNS: materialized presents N=8192, fused presents N=kb. That is a
property of composing fusion over a column-parallel matmul, NOT of the
schedule -- so it left open whether a version parallelised over QUERY ROWS
changes the answer.

This tests exactly that, and controls the obvious confound: both arms are
row-parallel across the same worker count, and both use MatmulBT as a
SERIAL inner primitive (per-worker Workspace, threshold pinned). So kernel
quality is identical and the parallelism model is identical; the schedule is
again the only variable. Writing a hand-rolled SIMD inner loop instead would
have measured my scalar Go against aikit's tuned kernel and told us nothing
about scheduling.
```

## TestP19FusedAttention.hoist

Moved from `decoder/p19_fused_attn_test.go` (the comment above `TestP19FusedAttention.hoist`) on 2026-10-09.

```text
PER-WORKER SCRATCH AND WORKSPACES HOISTED OUT OF THE TIMED REGION. The first
version allocated the materialized arm's [n, nKeys] score buffer (1.4 MB per
worker) INSIDE the timed call while the fused arm allocated far less — which
would have charged the losing arm for allocation and inflated exactly the
result being claimed. Caught before quoting the number, not after.
```

## TestP19FusedAttention.causal

Moved from `decoder/p19_fused_attn_test.go` (the comment above `TestP19FusedAttention.causal`) on 2026-10-09.

```text
---------------------------------------------------------------------
CAUSAL ARMS — the condition production actually runs under.

The arms above are UNMASKED, and that flatters neither side by accident: it
omits the one asymmetry that matters most. Production's materialized path
computes the FULL QK^T over every key --
  matmul(qh, ws.kh[:nKeys*hd], scores[:kt*nKeys], kt, hd, nKeys)
-- and only then masks inside the softmax, so causality buys it NOTHING. A
key-blocked fused loop can skip a block that is entirely masked, and in
causal prefill roughly half of them are.

So the unmasked 1.75x is a floor, not the number. This measures the real one.
Rows are placed at absolute positions startPos+i with startPos = nKeys-ktProd,
i.e. the last tile of an nKeys-token prompt, which is the shape at K=8192.
SWEEP EVERY TILE, not just the last one. A prefill of nKeys tokens runs
nKeys/ktProd tiles, from row 0 (attends 1 key) to row nKeys-1 (attends all).
The FIRST version of this arm measured only the last tile -- where causal
masking skips almost nothing, so fusion's block-skip is worth ~zero while
materialized still gets its softmax narrowed. That is the least favourable
tile for the schedule under test, and parking the item on it would have been
parking it on the instrument. Production's cost is the SUM over tiles.
```

## TestFusedAttention_logitDivergence

Moved from `decoder/p19_fused_attn_test.go` (the comment above `TestFusedAttention_logitDivergence`) on 2026-10-09.

```text
TestFusedAttention_logitDivergence answers the question the golden raises but
cannot: when fusion changes the generated tokens, is that a NEAR-TIE FLIP or a
BUG? Those look identical from a token diff.

It compares the PREFILL LOGITS directly on a real checkpoint and reports both
the cosine (is the forward computing the same thing?) and the top-2 margin at
the final position (was the argmax decidable at all?). A high cosine with a
margin near the perturbation size is a tie flip; a low cosine is a defect.

This is the same shape as a3_divergence_test.go, which exists because the f32
flag needed the identical question answered.
```

## TestFusedAttention_logitDivergence.baseline

Moved from `decoder/p19_fused_attn_test.go` (the comment above `TestFusedAttention_logitDivergence.baseline`) on 2026-10-09.

```text
BASELINE ON THE SAME MODEL: how far does the f32 flag ALREADY move the
hidden states away from acc64? Comparing fusion's divergence against a
number from a different model (a3_divergence_test.go's 0.9976 at dense 1.5B)
would be the cross-machine mistake in another costume. Both arms here, one
checkpoint, one depth.
```

## TestFusedAttention_logitDivergence.bar

Moved from `decoder/p19_fused_attn_test.go` (the comment above `TestFusedAttention_logitDivergence.bar`) on 2026-10-09.

```text
The bar compares fusion's ADDITIONAL divergence against the divergence the
f32 flag already accepts on the SAME model. A kernel-level 0.9999 bar is the
wrong instrument for a 24-layer forward -- the flag itself only reaches
~0.998 there -- and applying it was my error, corrected here rather than
relaxed silently.
```

## TestFusedAttention_endToEnd

Moved from `decoder/p19_fused_attn_test.go` (the comment above `TestFusedAttention_endToEnd`) on 2026-10-09.

```text
TestFusedAttention_endToEnd — what the 1.69-1.73x kernel win is worth through
a real forward, which is the only number that justifies accepting a
user-visible output change.

Everything else measured for P19 is kernel-level. This repo has retracted two
projections in one day for composing a kernel ratio with a profile share, so
the shipping claim comes from here.

DENSE model on purpose: attention's share of prefill is what the win scales
with, and it is ~55-70% on dense at depth against 17.4% on the MoE profiled
2026-09-01. Dense is the favourable case, so a weak result here closes the
question for both.

Paired and interleaved with alternating lead. Both arms run the f32 path
(fastAttn=true); the ONLY difference is GOINFER_FUSED_ATTENTION.
```

## loadInt4Model.skip

Moved from `decoder/parity_int4_test.go` (the comment above `loadInt4Model.skip`) on 2026-10-09.

```text
BEHAVIOUR CHANGE, deliberate. This site previously skipped whenever GOINFER_PREQUANT_GGUF was
UNSET -- alone among the four readers of that variable, the other three fell back to
../testdata. So the same box ran TestSerializeWeightsTo_matchesBuffer and skipped
TestDecodeParityInt4 with nothing in either output naming the difference. Both now resolve
through the registry's candidate list. Under the sweep nothing changes (the preflight exports
the variable either way); under a bare `go test ./decoder`, this gate now RUNS where it used
to skip.
```

## parityWantInt4ByArch

Moved from `decoder/parity_int4_test.go` (the comment above `parityWantInt4ByArch`) on 2026-10-09.

```text
parityWantInt4 pins the 0.5B's int4 (W4A8) greedy continuation of parityPrompt
— the int4 analog of parityWant (which pins int8int8). It guards the WHOLE int4
forward, including the prefill→W4A8 switch (decoder/weightmat.go), against
silent numerics drift: greedy + fixed prompt → fully deterministic. The first
5 ids match parityWant (the prompt's strongest continuation survives 4-bit);
they diverge after as int4's coarser weights bend the path. Regenerate ONLY
when an int4 numerics change is intentional and parity-reviewed (set this to
nil for a capture run — the test logs the new list and skips).

Re-captured 2026-06-12 after the dense-attention fix (causalAttention routes
decode through the f64 attendBatchedHeads, batched-identical to prefill — see
attention.go). The PRIOR golden was recorded under the old decode≠prefill bug
(scalar f32 attendQuery, aikit 1.3.0) and diverged from int8int8 already at id
4 (474 vs parityWant's 750) — i.e. it pinned the buggy output. The fixed int4
decode now MATCHES int8int8 through id 4 (both 750) and tracks it one token
further before int4 lossiness bends off — the more-correct continuation.

RE-CAPTURED AGAIN 2026-08-15, and the same way: the golden was pinning the WORSE
of two int4 paths. Attributed by bisect to `7deb368` (2026-06-14, aikit 1.7.3 →
1.8.1) — a bump whose message is entirely about the Qwen2.5-VL vision encoder and
which asserts "no regression in the … decoder paths". It carried two aikit linalg
commits that are not about vision at all: `36ce824` "fold W4A8 weight scales
in-register" and `52890f5` wiring that kernel on NEON. Folding the scales changes
the W4A8 accumulation, which moves a greedy continuation.

EVIDENCE FOR RE-CAPTURING RATHER THAN CALLING IT A REGRESSION — this is the
human decision `parity-coverage-policy.md` requires before a first-run value may
be banked, and it is a measurement, not a judgement call. Same prompt, same 0.5B,
24 greedy tokens, scored by how many leading ids match an f32 forward:

	int8int8 (unchanged, its own gate green)   19/24
	int4 THIS golden                           11/24
	int4 the 2026-06-12 golden below it         5/24

The kernel change made int4 twice as faithful to f32; it also tracks int8int8 for
11 ids where the old golden bent away at 5. Reverting to hold the gate green would
pin the less correct path — the identical mistake the 2026-06-12 note describes.

WHY IT WENT UNSEEN FOR TWO MONTHS, which is the part worth fixing elsewhere: this
gate was SKIPPING when the kernel landed (it alone required GOINFER_PREQUANT_GGUF
to be set, while its three sibling readers fell back to ../testdata — see
loadInt4Model above). It only began running when the asset registry gave it the
fallback, and it went red on its first execution. A dependency bump moved a
numerics path with the one gate that watches it dark.

RE-CAPTURED 2026-09-28 for L1's binary16 int4 group scales (aikit v1.50.0, and v1.50.1's
bit-identical arm64 fix; docs/tasks/task-cpu-decode-peer-gap-2026-09.md). That change moves int4
numerics by design, and was owner-approved on 2026-09-27. This golden was not re-baselined with it:
only TestInt4_forwardParity was, and this one went red on arm64 at the merge (3cd62e6d passes,
5c85f7c0 fails, drift at id 13). It was found the next day by a whole-package run. The evidence:
  - The new list is exactly what the PRE-L1 build (3cd62e6d) produces with GOINFER_INT4_F16_SCALES=1,
    all 24 ids identical, so the drift is the intended f16 rounding and nothing else.
  - Scored against an f32 forward, as above: the old golden 11/24, THIS golden 11/24, int8int8 19/24.
    Both int4 paths leave f32 at the same id and differ from each other only after it, so this one is
    no less faithful.

PER ARCHITECTURE since 2026-09-30. The 2026-09-28 re-capture was made on arm64, and the binary16 scales move
arm64's path only: the v0.20.0 parity sweep on nobara-pc (amd64, 0ca36756) produced the PRE-L1 list exactly, all 24
ids, and so failed against the arm64 one at id 13. The CPU int4 path is bit-identical within an architecture but
not across them (arm64 fuses multiply-adds that amd64 rounds separately; docs/parity-coverage-policy.md, "arch-
scoped"), so one list cannot hold both. Each is its own architecture's golden, and per the evidence above the two are
equally faithful to f32 (11/24 each). An architecture with no entry is a capture run.
```

## parityWantInt4ByArch.vnni

Moved from `decoder/parity_int4_test.go` (the comment above `parityWantInt4ByArch.vnni`) on 2026-10-09.

```text
CAPTURED UNDER INTEL SDE 10.13.1, 2026-10-04 (sde64 -icx and -spr gave the same 24 ids), because no VNNI machine is on hand. The VNNI kernels are
integer arithmetic, which SDE executes exactly, so the list is what a real VNNI CPU computes, not an emulation artefact; a real VNNI host
(GitHub's EPYC 9V74 pool, Ice Lake and newer) is the check on that claim, and the first one to run this test with the asset is owed a look.
It parts from the AVX2 list at id 22 (333, not 8960) where AVX2's top two are 0.078 apart. Evidence and the f32 comparison:
docs/measurements/sde-2026-10-04/README.md.
```

## int4GoldenKey

Moved from `decoder/parity_int4_test.go` (the comment above `int4GoldenKey`) on 2026-10-09.

```text
int4GoldenKey is the key into parityWantInt4ByArch for THIS host: the architecture, plus "-vnni" on an amd64 host whose dispatcher is
using the AVX-512 VNNI W4A8 kernels. Those compute a different quantised result from the AVX2 path (not summation order: relative L2 0.072 on
the logits of the step where the two lists part, docs/measurements/sde-2026-10-04/README.md), so the one amd64 list failed on every VNNI CPU
and nothing saw it (CI skips this test: the asset is absent on the runners). It reads ACTIVE kernels, not detected ones, so a host whose
self-test stepped VNNI down to AVX2 compares against the AVX2 list, which is the path it is then running. Call it AFTER the first Load (the CPU self-test runs there).
```

## forcedInt4Closeness

Moved from `decoder/parity_int4_test.go` (the comment above `forcedInt4Closeness`) on 2026-10-09.

```text
forcedInt4Closeness replaces TestDecodeParityInt4's token-exact comparison under a forced CPU fallback (H1.3). The golden pins the default path's greedy continuation, whose
4th token is a 0.15 near-tie that a different rounding draw flips (measured: the pure-Go path reads 474 where the AVX2 golden reads 750), so the forced run asks the question that
does not depend on the draw: is the int4 forward still as close to the int8int8 forward, for the same ids, as it should be? The prompt and the golden's own 4-token prefix are checked
(measured 1 - cosine: 1.64e-2 and 7.98e-3 pure Go, 1.75e-2 and 8.09e-3 AVX2).

HOW SENSITIVE THIS IS, measured: it is a COARSE sanity bound. The int4-to-int8int8 distance is dominated by int4's own quantization noise, so the int4GroupSize 32 -> 64 mutation
moves it only from 1.64e-2 to 1.92e-2 and this check stays GREEN; 32 -> 128 reads 4.1e-2 and fails it. That mutation is caught by TestInt4_forwardParity's forced centered-cosine
floor (cosines 0.13 to 0.99 against 0.99). Do not read this check passing as evidence the int4 grouping is right.
```

## TestMoEExpertBatching_M1vsMN

Moved from `decoder/moe_expert_batch_test.go` (the comment above `TestMoEExpertBatching_M1vsMN`) on 2026-10-09.

```text
Is batching the MoE expert matmul over rows worth anything? — the cheap check
before anyone funds the restructuring.

WHY THIS IS BEING ASKED AGAIN. The 2026-09-01 full-model profile put
`swiGLUExpert` at 93.1% of moeMLP and moeMLP at 42.1% of prefill — so the
expert weight matmuls are ~39% of prefill, and the largest single bucket now
that A3 collapsed attention to 17.4%. At K=8192 the batched-prefill loop
(forwardn.go) calls moeMLP once PER ROW, and swiGLUExpert issues its three
matmuls at M=1, so an expert's weights are re-read for every token that
routes to it.

AND WHY THE EXISTING VERDICT MAY NOT COVER IT. docs/completed/task-moe-streaming.md
Lever 4 is PARKED on "expert-major MoE prefill batching is NOT a compute
lever", measured 2026-08-28 as `uniform` (every row picks the same experts)
against `varied` (real routing), with uniform called the CEILING. But BOTH
arms of that experiment call moeMLP per row at M=1. What uniform changes is
which weights get touched, so it captures the BANDWIDTH/locality half of
batching — the weights stay cache-resident across rows — and not the
M=1 -> M=N half, which is a different axis: a GEMV is latency/ILP-bound in a
way a GEMM over hundreds of rows is not. The two coincide only if the
workload is purely bandwidth-bound.

So that result answers "does routing diversity cost much?" (no — and today's
profile agrees, routeExperts is 1.7% of moeMLP) without bounding "would
batching rows into GEMMs help?". This measures the second question directly,
which is cheaper than arguing about the first.

METHOD. Real Mellum2 expert shapes, real int4 W4A8 weights through the same
`matmul(be, *WeightMat, ...)` entry point production uses. Same total rows in
both arms:

	M=1 arm    N separate calls, exactly what moeMLP does today
	M=N arm    one call over N rows

The comparison is TIME PER ROW. If M=N is not materially cheaper per row, the
parked verdict covers this too and the lever really is closed.
```

## TestMoEExpertBatching_M1vsMN.shape

Moved from `decoder/moe_expert_batch_test.go` (the comment above `TestMoEExpertBatching_M1vsMN.shape`) on 2026-10-09.

```text
Mellum2 config.json: hidden_size 2304, **moe_intermediate_size 896**,
64 experts, top-k 8.

896 IS THE EXPERT WIDTH AND 7168 IS THE DENSE ONE. The first version of
this benchmark used 7168 -- `intermediate_size`, which the DENSE FFN uses
-- and so measured expert matmuls 8x wider than the ones swiGLUExpert
actually issues (it is called with moe.IntermediateDim). Wider matmuls
amortise per-call overhead better, so that error understated the batching
win rather than inventing one, but it was still the wrong shape.
```

## TestMoEExpertMajor_bitIdentical.depth

Moved from `decoder/moe_expert_batch_test.go` (the comment above `TestMoEExpertMajor_bitIdentical.depth`) on 2026-10-09.

```text
Overridable because the e2e measurement returned ~4x at K=4096 -- far more
than batching the matmuls can explain -- and a speedup that comes partly
from doing LESS work would look exactly like that. The identity must be
checked at the depth the claim is made at, not only at the cheap one.
```

## loadMoEBitIdentModel

Moved from `decoder/moe_expert_batch_test.go` (the comment above `loadMoEBitIdentModel`) on 2026-10-09.

```text
loadMoEBitIdentModel resolves a real MoE checkpoint for the gate above, through the shared
asset registry (testdata/assets.json's GOINFER_MELLUM_CKPT entry) rather than a hand-rolled
os.Getenv("HOME")+"/models/mellum2-unq" fallback — the old form was invisible to `gate census`
and ignored GOINFER_MODELS pointing the models root elsewhere (audit-2026-09-02.md N-41).
assetPath itself skips the test when the asset is absent, so a nil error here means Load ran.
```

## TestMoEExpertMajor_endToEnd

Moved from `decoder/moe_expert_batch_test.go` (the comment above `TestMoEExpertMajor_endToEnd`) on 2026-10-09.

```text
TestMoEExpertMajor_endToEnd is P18's decision measurement.

The microbenchmark says the expert matmul is 1.55x-2.13x cheaper per row when
batched. It says nothing about the GATHER/SCATTER cost of collecting an
expert's scattered rows, which is the whole open question — and multiplying
the microbenchmark by moeMLP's profile share is precisely the projection this
repo retracted twice on 2026-09-01.

So this times the real forward with the flag on and off, paired and
interleaved with alternating lead, at the depth the pre-registered rule names.

PRE-REGISTERED DECISION RULE (docs/queue-performance.md P18, committed before
this ran): fund if the net is >=15% end-to-end at K>=4096; park if <8%;
8-15% is AMBIGUOUS and parks pending a second mechanism.
```

## laguna_real_test.go

Moved from `decoder/laguna_real_test.go` (the comment above `laguna_real_test.go`) on 2026-10-09.

```text
Real-checkpoint gate for Laguna (poolside) — the loader + forward on the actual
released Laguna-XS.2 (33B-A3B, bf16 safetensors, 14 shards).

WHY THIS EXISTS SEPARATELY FROM T1. The tiny goldens prove the math against HF at
4 layers and 8 experts with random weights. They cannot prove that the loader
reads a REAL Laguna checkpoint, and that is where this family has hidden every
one of its surprises so far — all three of these were found by reading the real
checkpoint, and none of them are visible in a tiny fixture built from a config:

  - q_norm/k_norm exist on every layer, are UNCONDITIONAL in the module, and are
    mentioned nowhere in config.json.
  - g_proj is [64, 2048] — per-HEAD — even though config says `gating: true`,
    which the sibling generations' module resolves to per-element.
  - experts ship PER-EXPERT (mlp.experts.N.*), not as the module's fused 3D
    parameters, and the shared expert is `shared_expert` (singular) while the
    module calls it `shared_experts`.

A wrong stride or a misread schema here produces correct shapes, finite values,
and confident nonsense — so the coherence bar is a distinct-TRIGRAM ratio over a
CHAT-TEMPLATED prompt, not a distinct-token floor on a raw completion. (A raw
completion prompt on an instruction-tuned checkpoint measures "did the forward
avoid total collapse"; on gemma-4-26b that manufactured a false "int4 is broken"
signal that survived a week.)

M.1 has no gate of this kind and will not get one on this box: it is ~220B
(89 shards, ~400GB bf16) against 62GB of RAM. Its code path is identical to
XS.2's apart from config, and this gates that path — the same call made for
Kimi K2, recorded in docs/task-laguna.md.

	GOINFER_HEAVY_TESTS=1 GOINFER_LAGUNA_XS2=~/models/laguna-xs2 \
	  go test -tags realckpt ./decoder/ -run TestLagunaReal -v -timeout 180m
```

## TestLagunaReal_oracle

Moved from `decoder/laguna_real_test.go` (the comment above `TestLagunaReal_oracle`) on 2026-10-09.

```text
TestLagunaReal_oracle is the T3 numeric row: the released bf16 weights matched against an
HF bf16 forward of the SAME weights, pinned offline via scripts/pin_sequential_oracle.py
(accelerate disk offload — the reference and goinfer are never resident at the same instant,
the same technique scripts/pin_qwen3next_real.py already proved on an 80B model). Until this
gate, TestLagunaReal_gate above was coherence-only BY DESIGN: it proves the loader reads the
real checkpoint correctly (three real surprises found that way, see the file doc comment) but
never compared a single logit against an independent reference.

int8, NOT int4 (changed 2026-09-18) — see docs/measurements/int4-neartie-laguna-qwen38-2026-09-18.md
for the full account. This comment used to say int8 "does not fit alongside f32 activations in
62GB of RAM" and cited qwen3next's own real gate as the same capacity-forced choice — THAT WAS
WRONG, and it was never actually measured before this comment asserted it. Measured 2026-09-18:
int8 peaks at ~43GB RSS on this box (62GB total, comfortable headroom), takes ~8 minutes, and —
the reason this matters beyond a comment fix — the int4 gate this comment used to defend had a
real, reproducible divergence at continuation[3] (int4: cosine 0.984776, wrong token; confirmed
via a floating-point-rounding-noise control that it was a genuine near-tie in int4's coarser
grid, not a wiring bug). At int8 that divergence is GONE: cosine 0.998845, all 8 continuation
tokens exact. Caught by independent review (Gemini) challenging the capacity claim directly
rather than accepting "accept the int4 divergence as permanent" on the strength of a premise
nobody had actually run.

	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestLagunaReal_oracle -v -timeout 60m
```

## TestMellum2PrefillProfile

Moved from `decoder/mellum2_prefill_profile_test.go` (the comment above `TestMellum2PrefillProfile`) on 2026-10-09.

```text
Mellum2 long-prefill profiler — the ATTENTION-vs-MoE-FFN split of a batched
MoE prefill, at a chosen prompt length.

WHY THIS EXISTS, and why the dense profile does not answer it. The K=8192
profile quoted in queue-performance.md (`MatmulAVAcc64` 51.1%, `MatmulQKAcc64`
18.7%, ~70% attention) is a DENSE 1.5B. Two levers are live for long-context
prefill and they attack different terms:

	f32 attention  (G24, shipped as --cpu-fast-attention)  -> the O(K²) attention term
	expert-major batching (docs/completed/task-moe-streaming.md Lever 4)  -> the O(K) per-row MoE FFN

Their shares move in OPPOSITE directions as K grows, so "which is binding" has
no single answer — it has a crossover, and the crossover is what this measures.
Two structural facts make the dense number a bad proxy for the MoE case:

 1. `--cpu-fast-attention` REFUSES MoE (G24), so on this model the attention
    lever is not merely unmeasured, it is unavailable as shipped.
 2. Mellum2 is 24/28 SLIDING-window layers at window 1024 (config.json
    `layer_types`). Only the 4 full_attention layers grow as O(K²); the other 24
    cap their nKeys at 1024. A dense model's attention share therefore
    OVERSTATES this one's at every K past the window.

Mellum2 is the only locally-available MoE that takes the batched path at all —
canBatchN excludes deepseek_v2 (MLA), qwen3_5_moe and gemma4, which is every
other MoE checkpoint in ~/models. It cannot go through loadBenchModel: that
resolves GOINFER_PREQUANT_GGUF, whose asset kind is "file", and this checkpoint
is a safetensors DIRECTORY. So it loads the directory directly, the way
mellum2_parity_test.go does.

Run (one quant per process):

	GOINFER_HEAVY_TESTS=1 GOINFER_MELLUM_K=2048 \
	GOINFER_MELLUM_PROF=/tmp/mellum-2048.prof \
	go test ./decoder/ -run TestMellum2PrefillProfile -timeout 3600s -v

Then: go tool pprof -top -nodecount=30 <prof>

SWAP IS RECORDED, NOT ASSUMED AWAY. This is a ~12 GB int8int8 model on a 16 GB
box, and a run that pages produces a plausible, wrong profile — the exact
failure class that made the withdrawn G15 cliff expensive. `vm.swapusage` is
read before and after and printed with the result; a non-zero delta invalidates
the run. It is deliberately NOT an RSS check: darwin's UBC reclaims under
pressure, so RSS reports what survived rather than what was asked for (it read
LOWER at the failure point than at the baseline in the Metal slot sweep).
```

## TestMellum2PrefillProfile.slice

Moved from `decoder/mellum2_prefill_profile_test.go` (the comment above `TestMellum2PrefillProfile.slice`) on 2026-10-09.

```text
Default to the FULL checkpoint; GOINFER_MELLUM_CKPT points at a layer slice
where the full one will not fit. A 4-layer slice is representative here and
that is a property of THIS model, not a general licence: Mellum2 interleaves
layer_types on a period of 4 (s,s,s,f), so 21 sliding / 7 full — exactly 25%
full_attention — is reproduced exactly by layers [0,4), and every layer is
`sparse`, so the MoE geometry is unchanged. What a slice does NOT preserve is
the embedding's share of the total (4 layers amortize it over 7x less work);
the LM head is excluded by construction, since forwardLayersN stops at the
layer stack.
```

## TestExpertPaging_bitExact.AcceptSlowMoE

Moved from `decoder/moepaging_test.go` (the comment above `TestExpertPaging_bitExact.AcceptSlowMoE`) on 2026-10-09.

```text
AcceptSlowMoE: the 512 MB budget is chosen to FORCE eviction, which the load-time working-set guard (S4, 2026-09-23) predicts is
slower than its 2.0 tok/s floor on a big MoE; throughput is not what this gate checks. Without it this gate returned a load error
from 2026-09-23 on, and being env-gated (GOINFER_MOE_GIW) nothing ran it until 2026-10-06 (docs/queue-engineering.md B13).
```

## TestExpertPaging_touchesEveryFamilyThatBuildsAPager

Moved from `decoder/moepaging_test.go` (the comment above `TestExpertPaging_touchesEveryFamilyThatBuildsAPager`) on 2026-10-09.

```text
TestExpertPaging_touchesEveryFamilyThatBuildsAPager is M-34's gate (audit-2026-09-10): the
pager and its budget banner are built generically for ANY mmap-backed MoE (newExpertPager
registers every w.Layers[li].Experts entry regardless of architecture), but gpt-oss, Llama 4
and Nemotron 3 Nano's own forward loops indexed lw.Experts directly with no pager.touch call
at all — so -stream-weights enforced no RAM bound for these three families even though the
banner claimed one. This drives each family's own real MoE forward function directly
(hand-built minimal Architecture/LayerWeights, same discipline as
TestNemotron3NanoMoE_forward) with m.pager wired to a real mmap-backed pager keyed by
exactly the addresses (&lw.Experts[e]) the fixed code now touches, and asserts
pager.stats() shows nonzero hits+misses afterward — which it did NOT before this fix
(misses would stay 0 forever, since the mapped experts were never faulted through the
cache at all).
```

## TestExpertBufferPool_capNeverExceedsBudget

Moved from `decoder/moepaging_test.go` (the comment above `TestExpertBufferPool_capNeverExceedsBudget`) on 2026-10-09.

```text
task-never-swap-2026-09.md S5: one gap in decoder/moepool_test.go's already-thorough pool
coverage (byte-exact-through-compute refill, top-K self-eviction, LRU order, cross-stream lock
correctness — read before adding here, most of what a first pass wrote turned out to duplicate
it and was removed). What was missing: every existing slot-count test is minSlots-driven
(TestExpertBufferPool_topKNeverSelfEvicts deliberately gives a budget of ONE expert so minSlots
has to override it); none exercises the OTHER direction — a budget generous enough that IT,
not minSlots, decides the slot count. TestExpertBufferPool_capNeverExceedsBudget is that case.
```

## TestMC3_concurrentGenerationsFillSteps

Moved from `decoder/mc3_batch_test.go` (the comment above `TestMC3_concurrentGenerationsFillSteps`) on 2026-10-09.

```text
TestMC3_concurrentGenerationsFillSteps pins the straggler window's timing. With a step that takes far longer than the
window (a GPU's), 4 generations decoding together must run as full 4-wide steps, token after token. The window is
timed from when the resident came free: timed from each token's submission, a token submitted during a run has
"waited" the whole run when it ends, runs at once without the others, and the generations phase-lock into split
runs — measured on Metal as 3 batched + 1 solo on every token, 255 of 256 runs started by the timeout.
```

## TestMC3_topPConcurrentMatchesAlone

Moved from `decoder/mc3_batch_test.go` (the comment above `TestMC3_topPConcurrentMatchesAlone`) on 2026-10-09.

```text
TestMC3_topPConcurrentMatchesAlone: a top-p generation keeps the device top-K path under MC3 (it was simply off there
until 2026-09-30, which cost CUDA top-p 26%: docs/measurements/topp-regression-2026-09-30.md), and concurrent
generations still each get exactly the tokens they get alone. Two filters: one the K candidates usually serve, and one
(top_p near 1) whose rows fall back to the full row. The fake's Full fails if any other write reached the shared
logits buffer after its row, which is the hazard: the fallback must be read while the generation holds the resident.
```

## longPromptFastWant

Moved from `decoder/longprompt_golden_test.go` (the comment above `longPromptFastWant`) on 2026-10-09.

```text
longPromptFastWant is the greedy continuation of longPromptIDs(768) on the bench model through
the DEFAULT (f32) prefill path. Regenerate ONLY when a numerics change to that path is
intentional and reviewed — the same contract parityWant carries for the exact path.

KEYED BY GOARCH, and that is a FINDING, not boilerplate. Measured 2026-09-01 on the same
checkpoint and commit: arm64 and amd64 diverge at the FIRST generated token
(11 714 279 ... vs 13 715 522 ...). The exact kernel does not do this — parityWant is one
list for both — so cross-arch token reproducibility is something the f32 default gives up,
beyond the "decode != prefill" the flag documents. gc fuses x*y+z into FMA on arm64 and not
on amd64 (docs/parity-coverage-policy.md, "CPU reference is arch-scoped"), and f32 attention
has no f64 accumulator to absorb the difference.

A single shared list here would have been a permanently red gate on one of the two CI runners.
REGENERATED 2026-09-01 for the P19 fused schedule becoming the default. The
pre-fusion lists were:

	arm64 {11, 714, 279, 3491, 374, 429, 279, 2038, 374, 537, 3238, 438, 3601, 13, 576, 1465}
	amd64 {13, 715, 522, 2599, 397, 522, 1551, 397, 522, 2599, 397, 522, 1551, 397, 522, 2599}

AN ODDITY WORTH RECORDING RATHER THAN GLOSSING: arm64-fused reproduces the
pre-fusion AMD64 list exactly, all 16 tokens. That is not mystical -- that
sequence is a repeating attractor (522, 2599, 397, 1551 cycling) and two
different numerical paths can fall into the same loop -- but it does say these
continuations sit on knife-edges, which is the tie-flip reading rather than the
defect one. Independently supported: fusion's model-level divergence from acc64
(cosine 0.998262) is within 2e-5 of what the f32 flag already produces on its
own (0.998283), measured on one checkpoint at one depth.

The prompt is synthetic nonsense (700 + (i*7919)%9000), so a degenerate
continuation is garbage-in-garbage-out on either path and is not read here as a
quality signal. What the golden pins is REPRODUCIBILITY, not quality.
```

## TestLongPromptFast_forwardParity

Moved from `decoder/longprompt_golden_test.go` (the comment above `TestLongPromptFast_forwardParity`) on 2026-10-09.

```text
TestLongPromptFast_forwardParity closes the coverage hole that flipping --cpu-fast-attention's
default exposed: every other forward golden uses a prompt SHORTER than fastAttnMinPrompt, so
they all take the exact kernel and the f32 path — the shipped default — had no golden at all.

WHY THAT MATTERED CONCRETELY, not hypothetically. scripts/refresh_parity_hashes.sh is the
sanctioned release valve for a hashed-core edit, and it reported the default flip as a
"non-numeric core refresh" — true of everything the goldens touched, false of the change. A
green that covers only the unchanged side of an edit is the exact failure Q1 documents.

THE NAME ENDS IN _forwardParity DELIBERATELY: that is what the refresh script's selector
matches ((_forwardParity|_logitParity|_textParity)$|^TestGGUF_.*_parity$). A gate that the
release valve does not run is a gate that does not protect the release valve.
```

## TestLongPromptFast_forwardParity.env

Moved from `decoder/longprompt_golden_test.go` (the comment above `TestLongPromptFast_forwardParity.env`) on 2026-10-09.

```text
Default path (no env override): this is what a user gets. Pinned via t.Setenv (restores
after the test, unlike the raw os.Unsetenv this replaced) to the DEFAULT-on value for both
knobs that can perturb this golden's exact float sequence — not just
GOINFER_CPU_FAST_ATTENTION. A developer with GOINFER_FUSED_ATTENTION=0 legitimately
exported (fusedattn.go's documented opt-out) used to see this golden fail with
"f32 prefill continuation drifted", a false positive: the golden was recorded fused, and an
ambient =0 silently switched this run to the materialized (non-fused) arithmetic path
instead (N-41 (09-02)).
```

## forcedLongPromptCloseness

Moved from `decoder/longprompt_golden_test.go` (the comment above `forcedLongPromptCloseness`) on 2026-10-09.

```text
forcedLongPromptCloseness replaces TestLongPromptFast_forwardParity's token-exact comparison under a forced CPU fallback (H1.3). The golden pins one realization of the f32 fast-attention path
(a documented, accepted divergence from the exact path), and the top four exact logits sit within 0.21, so a different realization reorders the continuation (the pure-Go path reads token 304
first where the AVX2 golden reads 11). The question that does not depend on the draw: how far is the fast path from the EXACT path on this build (measured 1 - cosine at 768 tokens: 8.3e-3
pure Go, 1.6e-2 AVX2, and the exact path itself is bit-identical across the two builds)?
```

## TestGenerateNgramSpeculative_residentCommitsAcceptedSequence

Moved from `decoder/model_cancel_commit_test.go` (the comment above `TestGenerateNgramSpeculative_residentCommitsAcceptedSequence`) on 2026-10-09.

```text
TestGenerateNgramSpeculative_residentCommitsAcceptedSequence is audit R-03's gate for
spec_ngram.go's resident branch (the OTHER writer named there, alongside spec_eagle.go — see
that file's own note on why its forget calls are out of scope): on completion, resIDs must be
(a prefix of) prompt+everything actually emitted, not stay forgotten from the pre-prefill
residentForgetIDs call. Recurrent families never reach this branch at all
(validateNgramSpec rejects them before the goroutine starts), so there is nothing to gate for
the forget-otherwise half of R-01/R-03's general shape here.

Exactly prompt+emitted is NOT always achievable: a round's own trailing token is streamed
before it is forwarded (forwarding happens at the START of the next round, as that round's
targetVerify seq[0] — see the "one asymmetry" note in spec_ngram.go), so a return right after
that specific emit is one token behind the stream. Safe (the cache is never claimed to hold
more than it truly does) but the test has to allow for it rather than require exact equality.
```

## TestGenerateSpeculative_forgetsDraftResIDs

Moved from `decoder/model_cancel_commit_test.go` (the comment above `TestGenerateSpeculative_forgetsDraftResIDs`) on 2026-10-09.

```text
TestGenerateSpeculative_forgetsDraftResIDs is R-00's shape on the SECOND Model (V-09,
docs/review-2026-09-04.md): GenerateSpeculative's target claim forgets resIDs before any
resident write (R-00), but the draft's own claim just below it skipped the forget entirely.
GenerateSpeculative never COMMITS either model's resIDs (only R-01 phase 1/2 would let it), so
the correct postcondition for BOTH is nil after any call — the same invariant target's own
claim already held, now held by draft too.

A stale resIDs matters here specifically because it is reachable through the public API on
its own: a caller holding a draft *Model can call target.GenerateSpeculative once (which used
to leave draft.resIDs however it was BEFORE this call, silently) and then call draft.Generate
directly — a completely unrelated turn — which would trust a resIDs value describing content
the draft's resident KV no longer holds (this call already overwrote it).
```

## TestLagunaGGUF_gate.fitguard

Moved from `decoder/laguna_gguf_test.go` (the comment above `TestLagunaGGUF_gate.fitguard`) on 2026-10-09.

```text
The fit guard is bypassed for this model only, as in TestQwen35GGUF_gate. This gate checks the LOADER's parity,
not memory planning, and the guard prices the 18.9 GB mapped checkpoint as resident for the whole load and the KV
cache at the model's full context (56.5 GB needed against a 35.9 GB budget on nobara-pc's 62 GB, 2026-10-01),
which refused a load this box completed in the v0.19.0 sweep. requireHeavyModel keeps it to the box that holds it.
```

## mxfp4_test.go

Moved from `decoder/mxfp4_test.go` (the comment above `mxfp4_test.go`) on 2026-10-09.

```text
M2 of aikit's goinfer-kernel-moves task: the MXFP4 arithmetic that used to live in
decoder/mxfp4.go now lives in aikit (embed.MXFP4Scale / DequantMXFP4Blocks /
DequantMXFP4Split), gated there raw-bit against frozen copies of these bodies.

THIS FILE STAYS, pointed at aikit, because it holds something aikit's gate cannot: a fixture
extracted from a REAL gpt-oss:20b tensor and dequantized by the reference `gguf` Python library
(TestMXFP4_bitExactGolden). aikit has no Python, so its own vectors are Go-generated and
self-referential by construction; this one is an independent oracle. Deleting it with the
implementation would have removed the only check that the packing matches what the reference
library actually produces for bytes off a real checkpoint.
```

## TestMXFP4_splitIsSequentialNotGGML

Moved from `decoder/mxfp4_test.go` (the comment above `TestMXFP4_splitIsSequentialNotGGML`) on 2026-10-09.

```text
TestMXFP4_splitIsSequentialNotGGML pins the intra-block nibble order of the SAFETENSORS
layout, which is NOT GGML's.

This test exists because the opposite was assumed and written down. Phase 0 recorded that
safetensors MXFP4 differed from GGUF "only in addressing — no new numerics", and the first
implementation reused the GGML core. Dequantizing a real gpt-oss expert both ways and
diffing against the same weight through the already-validated GGUF reader settled it:
cosine 0.081 for GGML order, 1.000000 for sequential. The bug it would have shipped is the
worst kind — finite values, correct shapes, plausible magnitudes, entirely wrong weights.

So this asserts the two orders DISAGREE in exactly the documented way, rather than
asserting they agree (which the earlier version of this test did, and passed, because the
implementation shared their shared mistake).
```

## TestLFM2_arch

Moved from `decoder/lfm2_test.go` (the comment above `TestLFM2_arch`) on 2026-10-09.

```text
TestLFM2_arch pins the descriptor facts that the LFM2 forward depends on and that a
config-key mistake would silently zero. The two explicit >0 assertions are not padding:
both were real bugs, found 2026-08-31 by differencing against HF on the released
LFM2.5-2.6B, and both produced a running, fluent, WRONG model rather than an error.

  - NormEps came back 0 because LFM2 spells the key "norm_eps" and the adapter read
    cfg.RMSNormEps. Uniform 1.0185x scale on the first norm; logits cosine 0.897.
  - AttnScale came back 0 because the Architecture literal simply omitted it. Every q·k
    score is then 0, so softmax returns a UNIFORM average over the context. Invisible at
    one token (softmax of one element is 1.0 at any scale) — it needs >= 2 tokens to show.

Both had a MATCHING argmax against HF while broken, so neither a smoke test nor a greedy
decode would have caught them.
```

## TestLFM2_textParity.tier

Moved from `decoder/lfm2_test.go` (the comment above `TestLFM2_textParity.tier`) on 2026-10-09.

```text
tiny-golden is SUB-T3, and the merge derives status from the method, so this lands as
"experimental" rather than "validated" — which is the honest tier for a family whose
only numeric gate is a seeded 4-layer fixture. The released LFM2.5-2.6B was differenced
against HF by hand during bring-up (bit-exact after the two fixes), but that ran off a
5 GB local checkpoint with no committed gate, so it is NOT claimed here.
```

## emitParityRow.method

Moved from `decoder/parity_emit_test.go` (the comment above `emitParityRow.method`) on 2026-10-09.

```text
THE METHOD IS VOCABULARY, NOT FREE TEXT (B15). mellum's gate emitted "real-oracle" —
one word short of the T3 name "real-model-oracle" — and the merge wrote it into the
manifest verbatim, where it read as a method no tier rule recognises. A string typed
at ~20 call sites and never compared to a list is a defect waiting for a rename, so
it is checked HERE, at the source, rather than only by the tier gate downstream.
```

## emitParityRow.nan

Moved from `decoder/parity_emit_test.go` (the comment above `emitParityRow.nan`) on 2026-10-09.

```text
A NaN metric is not a numerics failure — cosineToFull (forward_test.go) documents
returning NaN when its full-logit-dump reference is simply absent on this box (a
gitignored, locally-regenerated fixture), and %.5f prints that as a literal NaN
token, which is not valid JSON and breaks TestParityManifest_merge for every OTHER
row in the same run, not just this family's. Found on an arm64 sweep 2026-09-06:
TestMixtral_forwardParity's tiny-golden checks (argmax/sample/top-k) all passed
against testdata/mixtral_forward_golden.json, but testdata/mixtral_forward_full.json
was missing, so cos came back NaN and this call still tried to record it as if it
were a measured value. Same shape as a "SKIP (no row expected)" real-model gate —
this gate has weaker evidence than usual, so it emits nothing rather than garbage.
```

## TestLoadAdapter_rejectsEveryOwnForwardFamily

Moved from `decoder/lora_test.go` (the comment above `TestLoadAdapter_rejectsEveryOwnForwardFamily`) on 2026-10-09.

```text
TestLoadAdapter_rejectsEveryOwnForwardFamily pins V-12 (docs/review-2026-09-04.md):
LoadAdapter used to hand-list arch.gemma4/arch.qwen35 as the own-forward families to
reject, instead of deriving the check from arch.ownForward() — the single table
(decoder/arch.go's ownForwards) that runLayers itself dispatches on. LFM2 fell out of
that hand-list (it wasn't in it to begin with), so LoadAdapter validated an LFM2 adapter
cleanly and registered it — but runLayersLFM2 takes no lora parameter at all, so the
registered adapter silently did nothing at generate time. This is the same "one
predicate, seven consumers" bug class canBatchN (decoder/forwardn.go) was fixed for
after an identical LFM2 omission crashed decode (audit-2026-09-02 C-01/C-02).

The rejection happens before any file I/O (loadLoRA is only called after the switch),
so a bogus dir is enough to isolate the check.
```

## TestMoEPagingSpike.lever2

Moved from `decoder/moepaging_spike_test.go` (the comment above `TestMoEPagingSpike.lever2`) on 2026-10-09.

```text
Lever-2 verdict: plain LFU is WORSE than LRU (classic establishment pathology — a re-faulted hot
expert restarts at freq=1 and is re-evicted before it accumulates count). LFU-aging fixes that
and beats LRU, but ONLY at impractically-tight budgets (3–4 GB); at the realistic ≥8 GB range all
three converge because LRU already keeps the hot set warm on this stationary-skewed signal
(frequency ⇒ recency). So a frequency-aware policy is NOT worth building — keep EvictLeastRecent.
```

## lfuSim

Moved from `decoder/moepaging_spike_test.go` (the comment above `lfuSim`) on 2026-10-09.

```text
lfuSim runs classic LFU of capacity C over the access stream: on a miss over budget it evicts the
resident page with the LOWEST access count SINCE it became resident (ties broken by LRU — the
oldest among the coldest). The MoE router's demand is a skewed-FREQUENCY signal (the spike above:
the hottest 10% of experts absorb ~72% of accesses, stable across tokens), which is LFU's sweet
spot — once the hot set is warm it is never the victim. This is Lever 2's candidate vs the pager's
current EvictLeastRecent (LRU). Eviction scans the resident set (O(C)); the trace is small.
```

## TestLagunaDFlash.mask

Moved from `decoder/laguna_dflash_test.go` (the comment above `TestLagunaDFlash.mask`) on 2026-10-09.

```text
mask_token_id is TOP-LEVEL in this dialect. Getting it wrong is silent and
expensive: the drafter still runs and still produces lossless output, just
badly — P10 measured a known-good pairing fall from 1.60x to 0.66x on exactly
this mistake.
```

## TestLagunaDFlash_acceptance

Moved from `decoder/laguna_dflash_test.go` (the comment above `TestLagunaDFlash_acceptance`) on 2026-10-09.

```text
TestLagunaDFlash_acceptance measures what the pairing is actually worth: how many
tokens per round the target ACCEPTS from the drafter.

WHY ACCEPTANCE AND NOT SPEEDUP. Block drafting is lossless by construction — every
emitted token is one the target's own argmax produced — so no correctness test can
tell a good drafter from a bad one. Acceptance is the only signal, and P10 learned
that the hard way twice (a wrong mask token turned 1.60x into 0.66x while output
stayed perfectly valid; a drafter fed the wrong embeddings would do the same).

This deliberately does NOT report a speedup. Laguna is CPU-only in goinfer today
(FeatAttnOutputGate makes every resident backend decline it), and P10's own
kill-gate found the DRAFT, not the verify, was the wall — a CPU draft against a
CPU target is a different regime from the GPU-resident numbers gate 3 reported.
Measuring wall-clock here would produce a number that says more about this box
than about the pairing.

	GOINFER_HEAVY_TESTS=1 GOINFER_LAGUNA_XS2=~/models/laguna-xs2 \
	  GOINFER_LAGUNA_DFLASH=~/models/laguna-xs2-dflash \
	  go test -tags realckpt ./decoder/ -run TestLagunaDFlash_acceptance -v -timeout 180m
```

## TestLFM2_multiTokenPrefillMatchesSequential

Moved from `decoder/ownforward_test.go` (the comment above `TestLFM2_multiTokenPrefillMatchesSequential`) on 2026-10-09.

```text
C-01, END TO END, ON THE COMMITTED FIXTURE. Every prompt of >=2 tokens reached the generic
batched stack, whose first act on an LFM2 conv layer is rmsNorm against a QNorm that was never
loaded — an index-out-of-range in the Generate goroutine, where net/http's handler recover does
not reach. TestLFM2_textParity stayed green throughout because it drives m.forward one token at
a time and never calls prefillLogits.

The assertion is not "it does not panic" but "the two paths agree": prefillLogits documents
itself as bit-identical to the sequential prefill, and a fix that merely stopped the crash while
taking a different path would satisfy the weaker claim.
```

## TestOwnForward_lifecycleSeamsRefuseEveryOwnForwardFamily

Moved from `decoder/ownforward_test.go` (the comment above `TestOwnForward_lifecycleSeamsRefuseEveryOwnForwardFamily`) on 2026-10-09.

```text
THE OTHER FOUR CONSUMERS OF THE SAME FACT (audit §0 theme 1). Each hand-listed the own-forward
families; between them they had drifted by two — lfm2 was missing from all four and gpt-oss from
HiddenLast — so an LFM2 model reached seams documented to refuse it and got nil rows or a pooled
vector from a path nobody had checked. Driven off the table so a family added there is covered
here without anyone remembering to.
```

## TestOwnForward_recurrentBitMatchesTheCacheKinds

Moved from `decoder/ownforward_test.go` (the comment above `TestOwnForward_recurrentBitMatchesTheCacheKinds`) on 2026-10-09.

```text
The two views of "recurrent" must not disagree: the table's Recurrent bit decides before a cache
exists (speculative rollback), KVCache.hasRecurrentState decides once one does (truncate,
snapshot, session reconcile). A family recurrent in one sense and not the other is the state that
produced C-02, only with the halves swapped.
TestOwnForward_recurrentBitMatchesTheCacheKinds is DERIVED, not listed (audit 2026-09-10 C-03,
G-05). It used to carry a hand-written map of family -> which cache field to set, and
bailing_hybrid was on NEITHER side of it: table Recurrent=false and absent from the map, so it
passed as "consistently non-recurrent" while its KDA state leaked across sessions. Both
predicates it compared omitted KDA, so their agreement proved nothing — what was missing was an
INDEPENDENT view of what the cache actually holds.

So the cache is asked directly: each own-forward family's REAL cache is built by NewCache from
its representativeConfig, and its recurrent kinds are found by reflection (recurrentKinds). The
registry bit, both hasRecurrentState views and speculative rollback must all agree with that.
```

## mellum_slice_test.go

Moved from `decoder/mellum_slice_test.go` (the comment above `mellum_slice_test.go`) on 2026-10-09.

```text
REAL-WEIGHT layer-slice oracle for Mellum2 (JetBrains Mellum2-12B-A2.5B-Instruct) — the
CPU half of G11.

WHY IT EXISTS, AND WHY IT IS NOT A SYNTHETIC FIXTURE. G10 declared FeatRopeMscale for
Metal to unblock gpt-oss's YaRN, which as a documented side effect also admits Mellum
onto the Metal resident path — a second GPU path for this family with ZERO end-to-end
validation. G10's attempt to close that with a hand-rolled random-weight fixture failed
for a reason worth not repeating: at realistic dims a plain dense qwen2 control with NO
QK-norm at all also misses the 0.95 cosine bar against fully-random weights (0.898).
Random weights lack the outlier structure real checkpoints have, so int4/int8 noise
swamps whatever feature is under test. A synthetic fixture CANNOT discriminate a real
bug from that noise floor.

A real slice can. It keeps what matters — trained weights, with their real routing
distributions, real QK-norm scales and real YaRN interaction — and drops only depth.
Layers [0,4) is the coverage floor, not a size choice: Mellum2's real layer_types is a
3:1 sliding/full interleave, so layer 3 (0-indexed) is the FIRST full_attention layer
and therefore the first one carrying YaRN. A 3-layer slice would gate the sliding path
and silently skip the mscale that G10 actually changed. Verified against the released
config before slicing rather than assumed.

This is the DECODER-side gate: it proves goinfer's own CPU forward matches the HF f32
reference on real weights. The Metal half (metal/mellum_real_test.go) runs residentParity
against the SAME slice; see docs/queue-correctness.md G11 for the handoff, including how
to regenerate this slice bit-identically (it is 4 GB, so the checkpoint is gitignored and
only the golden is tracked).

	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestMellumSlice -v
```

## TestLoad_resolvesAutoWeightCacheBudgetFromLiveProbe

Moved from `decoder/layerpaging_test.go` (the comment above `TestLoad_resolvesAutoWeightCacheBudgetFromLiveProbe`) on 2026-10-09.

```text
TestLoad_resolvesAutoWeightCacheBudgetFromLiveProbe is S4 item 2's WIRING gate
(task-never-swap-2026-09.md): TestResolveWeightCacheBudget (fitguard_test.go) proves the
arithmetic in isolation; this proves decoder.Load actually calls it before newLayerPager sees
WeightCacheBytes, through a REAL .giw load. Injecting hostRAMAvailable to two different live
figures must move newLayerPager's own window arithmetic in the direction each figure implies: a
modest available produces a budget smaller than this fixture's total resident weight bytes
(window pinned well under its layer count, so a REAL pager is built) while a generous available
makes the whole model "fit" (window >= n, so newLayerPager returns nil — no streaming needed).
Reverting the model.go wiring (the `opts.WeightCacheBytes = resolveWeightCacheBudget(...)` line)
makes BOTH cases build off aikit's own AutoBudget() instead, indistinguishable from each other
since the live probe would never be consulted.

Needs the 1.5B fixture, not the registry's default 0.5B one: at 0.5B (0.59 GB resident), every
available figure large enough to clear guardGIWFit's own ~1 GB KV+scratch floor (S4 item 1)
already halves to a budget bigger than the whole model — there is no available figure where
BOTH guards are exercised on that fixture. 1.5B (1.66 GB resident) leaves a real gap between the
two floors (empirically probed: 2.5 GB available streams at window=27 of 28 layers; 3 GB+ fits
outright).
```

## TestLlama4_chunkedAttentionStart

Moved from `decoder/llama4_chunk_test.go` (the comment above `TestLlama4_chunkedAttentionStart`) on 2026-10-09.

```text
M-05: attention_chunk_size was read from config and never applied. The RoPE layers attended
[0, pos] on every position, so from position C on they saw keys HF's block-diagonal chunked
mask removes. Nothing caught it because every parity gate uses a sequence shorter than C,
where chunked and full-causal are the same function.

WHAT IS PINNED HERE is the mask this build implements: a query at p attends [(p/C)*C, p] on a
RoPE layer, and [0, p] on a NoPE layer. The audit rates the HF semantics medium-confidence
(recalled, not read), so this states the rule explicitly rather than burying it — if HF turns
out to chunk differently, this test names exactly what to change.
```

## TestLaguna_textParity.refs

Moved from `decoder/laguna_test.go` (the comment above `TestLaguna_textParity.refs`) on 2026-10-09.

```text
Each fixture was generated against ITS OWN generation's modeling_laguna.py
(see scripts/pin_laguna_tiny.py), and the references were verified to actually
apply the per-layer-type RoPE — xs21's HF model carries rotary_emb with
inv_freq len 4 + YaRN mscale 1.3466 and swa_rotary_emb with len 8 + scaling
1.0, exactly the widths and scalings the adapter derives. Without that check a
silently-degenerate reference would have made this gate meaningless.
```

## TestTruncateTo_resetsConvWindowAlone

Moved from `decoder/kvcache_recurrent_test.go` (the comment above `TestTruncateTo_resetsConvWindowAlone`) on 2026-10-09.

```text
A THIRD KIND OF RECURRENT STATE, ALONE. TestTruncateTo_resetsRecurrent above sets mamba AND
delta on the same cache, so a guard checking either one stays green — and a third kind is
invisible to it. That is exactly what happened: LFM2's short-conv window is mutated in place per
token like the other two, resetRecurrent() already cleared it, and the guard that CALLS
resetRecurrent named only mamba and delta. On an LFM2-only cache the reset was unreachable, so
TruncateTo(0) left the window intact and conversation B's first K-1 tokens convolved over
conversation A's last Bx vectors at every conv layer (audit-2026-09-02 C-02).

This cache carries ONLY conv, which is the whole point of it being a separate test.
```

## TestLlamaTiny_textParity

Moved from `decoder/llama_tiny_test.go` (the comment above `TestLlamaTiny_textParity`) on 2026-10-09.

```text

THE BASELINE ARCHITECTURE HAD NO TINY FIXTURE. `llama` is a required parity gate and the most
common architecture in the ecosystem, and until 2026-09-02 the only llama checkpoints anywhere
here were the Linux box's gitignored llama3.2-1b (2.4 GB), tinyllama-awq (731 MB) and
tinyllama-gptq (733 MB). So no other machine could exercise the arch at all, and the .giw census
round-tripped 21 families without ever touching it. Found by the census's completeness gate
reporting the box's untracked fixtures (audit-2026-09-02 C-03 follow-on).

Deliberately plain — GQA 2:1, SwiGLU, RMSNorm, UNTIED head, rope_theta at Llama-3's 500000.0.
Every other family's descriptor is a deviation from this one, so a break here is a break in the
thing they all deviate FROM. The untied head matters on its own: it is a separate LMHead tensor
the serializer carries, and the tied families cannot exercise that path.
```

## TestLlama4Real_gate.fitguard

Moved from `decoder/llama4_real_test.go` (the comment above `TestLlama4Real_gate.fitguard`) on 2026-10-09.

```text
THIS MODEL DOES NOT FIT, AND THE GATE OPTS OUT ON PURPOSE.

Scout is 107.8B elements; at int4 (0.625 bytes/element including group scales) that is
~62.7 GB resident, against nobara-pc's 62.7 GB of RAM — 100% of the machine, with nothing
left for the KV cache, activations or the OS. The load-time fit guard (decoder/fitguard.go)
therefore refuses it, and the refusal is CORRECT: measured 2026-09-06 during the parity
sweep, and the estimate was checked rather than trusted (no vision tensors in this GGUF,
element count matches Scout's published 109B).

What the guard revealed is that this gate has been completing by PAGING, which is the exact
failure mode the guard was written for after a cold-user run watched a 16 GB Mac go +7.8 GB
into swap in five seconds. The gate is kept as-is rather than shrunk, because it is the only
real-checkpoint coverage llama4_text has and it inspects architecture rather than measuring
speed — so paging costs time here, not correctness. The opt-out is explicit so that nobody
reads a passing llama4 gate as evidence that this model fits.
```

## runMellum2Golden.emit

Moved from `decoder/mellum2_parity_test.go` (the comment above `TestMellum2_forwardParity.emit`) on 2026-10-09.

```text
Record the validated metrics (no-op unless GOINFER_MANIFEST_EMIT; skipped if any check
above failed). int8int8 (serve default) vs HF bf16 → real-model-oracle; argmax exact when green.
The method string is the T3 vocabulary name, checked by emitParityRow: "real-oracle" was
written here for months and reached the manifest verbatim (B15).
Emit only from the forward gate (not the window gate) to avoid a double row.
```

## TestMellum2_logitParity

Moved from `decoder/mellum2_parity_test.go` (the comment above `TestMellum2_logitParity`) on 2026-10-09.

```text
TestMellum2_logitParity gates the Mellum2 forward (MoE 64/top-8, 3:1
sliding/full interleave, YaRN-on-full RoPE, QK-norm) against the HF bf16
oracle on a chat-templated prompt — argmax is the first answer token (50195
"Paris"), so this also gates coherence. int8int8 (the serve default) vs bf16:
measured sample-256 cosine 0.99955 (floor 0.98, the Gemma 4 12B reference).
```

## TestMellum2_windowParity

Moved from `decoder/mellum2_parity_test.go` (the comment above `TestMellum2_windowParity`) on 2026-10-09.

```text
TestMellum2_windowParity pins the sliding-window EVICTION path on the real
checkpoint: a 1441-token prompt (> the 1024 window), so the local layers attend
only within the window while the YaRN'd full layers see everything. The
next-token logits still match the HF bf16 oracle (measured cosine 0.99636).
(Inc3 real-model proof; the synthetic unit-level proof is
TestMellum2_slidingWindowEviction.)
```

## TestLoadAdapter_dimMismatchRejects

Moved from `decoder/lora_compute_test.go` (the comment above `TestLoadAdapter_dimMismatchRejects`) on 2026-10-09.

```text
TestLoadAdapter_dimMismatchRejects gates C-03 (audit-metal-2026-09-12.md): LoadAdapter (the
compute-time path, #7) validated only that a delta's tensor NAME matched a known projection —
never that its [Out,In] shape matched the ACTUAL base projection. A same-family adapter trained
against a different-size base (e.g. one more attention head) passed straight through to every
resident backend's SetAdapter, which trusts In/Out from the checkpoint (Metal only range-checks
rank); a mismatched Out overruns the kernel's own output bound and writes past the Q slot into
K/V. The merge-at-load path (weights.go's loadProj -> loraAdapter.merge) already made exactly
this check; validateComputeTimeDims is its twin for the compute-time path.
```

## TestGuardGIWFit_floorRefusalNamesTheFloorsNeed

Moved from `decoder/metal_fit_test.go` (the comment above `TestGuardGIWFit_floorRefusalNamesTheFloorsNeed`) on 2026-10-09.

```text
TestGuardGIWFit_floorRefusalNamesTheFloorsNeed: the refusal says what the floor needs. It used to print the whole
window's need under "even at the 2048-token floor" (about 32 GB for a 128k-window 7B, against a floor need under 1 GB).
```

## TestResolveMoEPagerPool

Moved from `decoder/moe_pager_mode_test.go` (the comment above `TestResolveMoEPagerPool`) on 2026-10-09.

```text
The CPU expert pager's mode resolves in one place, in this order: Options.MoEPager, then
GOINFER_MOE_PREAD_CPU, then MoEPagerDefault. A library caller used to get mmap on darwin while
serve got pool, because serve applied its default by setting the env var.
```

## TestResidentMoECap_pinsKernelConstant

Moved from `decoder/moecap_kernel_pin_test.go` (the comment above `TestResidentMoECap_pinsKernelConstant`) on 2026-10-09.

```text
TestResidentMoECap_pinsKernelConstant gates M-17: the feature-matrix MoE caps in
residentBackendMoECap MUST equal the router kernel's actual fixed-array bounds, or the
hardware-matrix generator declines archs BuildResident admits (or vice versa) — the
one-source-of-truth invariant the map's own comment promises. This reads the kernel sources
and compares. The cuda groups cap was 32 while cuda/moe.cu's MOE_MAX_G is 64, so an arch with
n_group in 33..64 was published not-CUDA-resident while the runtime admitted it. The test
extracts each constant from its source and pins the cap to it; a future kernel bump that
forgets the map (or vice versa) fails here.
```

## TestResidentMoECap_runtimeDeclineReadsTheDeclaration

Moved from `decoder/moecap_kernel_pin_test.go` (the comment above `TestResidentMoECap_runtimeDeclineReadsTheDeclaration`) on 2026-10-09.

```text
M-31: the pin above reads the KERNEL sources and features_test.go asserts ResidentEligible.
Both were green while gpu/residency.go carried its own hardcoded 256/32 — so a 384-expert
Kimi-K2 was published "✅ resident" in both generated matrices, admitted by ResidentEligible,
and then declined to CPU by a line naming a number nothing else agreed with. Neither existing
test reads the file that actually makes the decision.

So this one does. It asserts the RUNTIME decline site does not restate a literal cap — the
numbers must come from ResidentBackendMoECap — because a literal there is precisely the drift
the map exists to prevent, and it is invisible to a test that only greps the kernel.
```

## nemotron35lightning_real_test.go

Moved from `decoder/nemotron35lightning_real_test.go` (the comment above `nemotron35lightning_real_test.go`) on 2026-10-09.

```text
Real-checkpoint gate for NVIDIA-Nemotron-3.5-Lightning-30B-A3B (nemotron_h MoE) — F2 of
docs/completed/task-families-2026-09.md. NOT a new family: Phase 0 found this checkpoint's config.json
identical to the already-T3'd Nemotron 3 Nano's (docs/completed/queue-correctness.md G4) in
every architecturally meaningful field, including the exact 52-block layer pattern (23 mamba /
23 moe / 6 attention, same order). This gate exists to confirm the ACTUALLY TRAINED weights
behave the way that identical architecture predicts — a tiny fixture cannot catch a wrong
tensor name, a transposed expert stack, or a router bias read from the wrong key.

This goes through the same realLogitOracleQuant helper every other real-checkpoint gate uses,
so it DOES call emitParityRow like the others — but it is deliberately left OUT of
cmd/gate/parity.go's emitGates list (the manifest's "nemotron_h" row is keyed by registry
model_type, not by checkpoint, and is already `validated` from Nano's T3, cosine 0.997668;
TestNemotron3NanoReal_oracle itself isn't in emitGates either — the row was populated once by a
direct run + manual merge, not by the routine sweep). A `go run ./cmd/gate parity` sweep will
still run this gate (it's in parityRealckptGates) and PASS/FAIL/skip on it, but won't touch the
manifest; running it directly with GOINFER_MANIFEST_EMIT=1 would still emit a PARITY_ROW line,
and merging that WOULD overwrite Nano's specific numbers with Lightning's — a deliberate choice
for whoever runs it by hand, not something this gate silently does on a routine sweep. This
run's result is recorded in docs/completed/task-families-2026-09.md's F2 section as confirmatory
evidence for the same family.

	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestNemotron35LightningReal -v -timeout 90m
```

## TestNemotron3NanoReal_oracle.int8

Moved from `decoder/nemotron3nano_real_test.go` (the comment above `TestNemotron3NanoReal_oracle.int8`) on 2026-10-09.

```text
int8 WEIGHTS, f32 ACTIVATIONS — not the int8int8 every other family uses, and the
difference is measured rather than assumed:

	int8int8   cosine 0.978086   (int8 activations)
	int8       cosine 0.997668   (f32 activations)

The forward is correct; the sensitivity is real. This model routes 6 of 128 experts
(4.7%), far sparser than the comparable MoE families here (deepseek_v3 0.99951,
qwen3_5_moe 0.99333, granitemoehybrid 0.99566, all at int8int8), and its own DENSE
parent scores 0.99574 at int8int8. Quantizing activations perturbs the router enough to
flip which experts run, which is a discrete change no amount of averaging smooths —
the expert-flip cliff already recorded for granite's MoE stack.

So this is a deployment fact worth carrying, not a threshold dodge: DO NOT run this
family's MoE variant with int8 activations.
```

## nemotron_moe_real_test.go

Moved from `decoder/nemotron_moe_real_test.go` (the comment above `nemotron_moe_real_test.go`) on 2026-10-09.

```text
Real-GGUF gate for Nemotron 3 Nano (nemotron_h MoE, 30B-A3B) — the loader + MoE forward on
actual released quantized weights.

WHY THIS EXISTS SEPARATELY FROM T3. T3 (nemotron3nano_real_test.go) proves the forward against
an HF oracle, but it loads SAFETENSORS. The GGUF path is a different loader with a different
expert layout — safetensors ships one tensor per expert
(experts.I.{up,down}_proj), GGUF fuses ALL experts into one 3-D tensor per projection
(blk.N.ffn_{up,down}_exps.weight) — and it had never completed a forward pass. It was verified
by reading a real file's header and dequantizing one layer's tensors by hand; correct dims and
sane values, which is not the same as a model that runs.

That gap matters here more than usual: a fused expert stack read with the wrong stride
produces correct shapes, finite values, and confident nonsense. This session has now seen that
failure mode five times in a different subsystem.

THE PROMPT GOES THROUGH THE CHAT TEMPLATE, and the coherence bar is a distinct-TRIGRAM ratio,
not a distinct-token floor. The dense gate (TestNemotronReal_gate) carries an audit note
saying exactly this: its raw completion prompt on an instruction-tuned checkpoint measures
"did the forward avoid TOTAL collapse", not coherence, and on gemma-4-26b that manufactured a
false "int4 is broken" signal that survived a week — which a distinct<3 floor would not have
caught, because repetition has more than 3 distinct tokens. New gate, so it starts with the
pattern that note recommends rather than inheriting the one it warns about.

	GOINFER_HEAVY_TESTS=1 GOINFER_NEMOTRON3NANO_GGUF=~/models/nemotron3nano-gguf/... \
	  go test -tags realckpt ./decoder/ -run TestNemotron3NanoMoEReal -v -timeout 60m
```

## TestNemotronReal_gate.audit

Moved from `decoder/nemotron_real_test.go` (the comment above `TestNemotronReal_gate.audit`) on 2026-10-09.

```text
AUDIT NOTE (da5a6ec): raw completion prompt on an instruction-tuned checkpoint,
gated only by the distinct<3 floor below — which measures "did the forward avoid
TOTAL collapse", not coherence. On gemma-4-26b-a4b-it a raw prompt manufactured a
false "int4 is broken" signal that survived a week, and distinct<3 would not have
caught it (repetition has >3 distinct tokens). This gate currently passes, so the
completion is in-distribution ENOUGH for this checkpoint — but when it is next
revalidated, adopt TestGemma4_26B_gate's pattern (render the family chat template +
distinctTrigramRatio floor) instead of trusting distinct<3.
```

## TestP13_mmapAliasRisk_classifiesDTypes

Moved from `decoder/p13_mmap_release_test.go` (the comment above `TestP13_mmapAliasRisk_classifiesDTypes`) on 2026-10-09.

```text
P13: the safetensors loader used to hold the SOURCE mapping for the model's whole life. On a
55.6 GB bf16 checkpoint that is 46.8 GB RSS against GGUF's 24.5 GB for an IDENTICAL 17.9 GB Go
heap, and 1.69x slower decode — dead bf16 pages competing with the hot quantized weights for
page cache. loadWeights now closes the source at end of load when nothing can alias it.

THE GATE IS mmapAliasRisk, AND THIS TESTS THE GATE RATHER THAN THE OUTCOME. Getting it wrong in
the safe direction costs the old behaviour; getting it wrong in the unsafe direction is a
use-after-free on weights the decode is reading, which would surface as garbage output or a
SIGBUS far from here.
```

## TestP13_rssAfterLoad

Moved from `decoder/p13_mmap_release_test.go` (the comment above `TestP13_rssAfterLoad`) on 2026-10-09.

```text
TestP13_rssAfterLoad reports resident set size after a load, so the release can be measured
rather than asserted. DIAGNOSTIC — P13's own entry says "do not assume the 1.69x transfers",
because that figure came from a box holding 46.8 GB of 62 GB and is a page-pressure artifact as
much as a mapping one. This prints; it does not gate.
```

## parity_emit_b15_test.go

Moved from `decoder/parity_emit_b15_test.go` (the comment above `parity_emit_b15_test.go`) on 2026-10-09.

```text
B15 regression gates: the manifest WRITER must not produce claims the manifest READER has to
reject. One sweep with EMIT_MANIFEST=1 promoted four families from experimental to *validated*
while their method still said tiny-golden, and wrote mellum's method as "real-oracle" — a
string no tier rule recognises. TestParityManifest_methodTier caught both, which is that gate
working; these two catch them at the source, and — the part that matters — they run in plain
CI, where the emitter itself never does.
```

## TestGenerate_residentPathAllocatesNoHostKV

Moved from `decoder/model_lazy_kv_test.go` (the comment above `TestGenerate_residentPathAllocatesNoHostKV`) on 2026-10-09.

```text
TestGenerate_residentPathAllocatesNoHostKV is P-01(a) (audit-2026-09-10): Model.Generate used
to allocate the full host KV cache (m.NewCache) BEFORE the resBusy CAS even ran, so a
resident-and-won call paid for capacity it never touched. NewCache is the sole place
prefillEnters advances (R13's own discipline: "a test can OBSERVE that a check placed one line
too late produces the identical error text" — same idea, applied to allocation instead of a
refusal), so a zero delta here is a direct, non-inferred proof no host KV was allocated, not
just that generation still produced the right tokens.
```

## olmo_hybrid_test.go.verified

Moved from `decoder/olmo_hybrid_test.go` (the comment above `olmo_hybrid_test.go.verified`) on 2026-10-09.

```text
VERIFIED AGAINST A REAL Olmo-Hybrid-7B CHECKPOINT (HTTP Range on its safetensors header), not
just modeling_olmo_hybrid.py's source — the source alone, and even a local save_pretrained
round-trip through this transformers version's own conversion_mapping.py, both produced tensor
names/splits that do NOT match the real release; see scripts/pin_olmo_hybrid_tiny.py's own
docstring for the full account.
```

## loadMistralWindowGolden

Moved from `decoder/mistral_tiny_window_test.go` (the comment above `loadMistralWindowGolden`) on 2026-10-09.

```text
THIS TEST EXISTS BECAUSE THE FIXTURE'S WEIGHTS WERE NEVER COMMITTED. Only its config files were (the *.safetensors ignore rule), so the CUDA and WebGPU
window tests and `gate identity`'s mistral asset had nothing to load on a fresh checkout, and no CPU test consumed the golden at all: the CPU
sliding-window path had no committed parity gate against HF. A missing fixture or golden FAILS here instead of skipping, for the reason D11 found in
D2's gate: a gate whose fixture can vanish into a skip is not a gate.
```
