# decoder: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `decoder`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## parityManifest

Moved from `decoder/parity_manifest_test.go` (the comment above `parityManifest`) on 2026-10-09.

```text
parityManifest mirrors testdata/parity_manifest.json. familyParity uses
json.RawMessage for fields the test must preserve verbatim on -update
(metrics/status/dates/etc.) while still letting us read+rewrite deps_hash and
inspect uses/own/validated_at. Field order matches the on-disk schema so that
re-marshaling produces a stable, zero-diff layout.

No AikitVersion field (G-01, audit-2026-09-10): the manifest used to carry a hand-typed
"aikit_version" string mixed into every family's deps_hash, which drifted from the real pin —
stale seventeen versions once (CHANGELOG v0.17.0's "re-arm"), then drifted again two releases
later when that fix re-TYPED the value instead of deriving it. freshDepsHash now reads the
ROOT go.mod's aikit require directly (rootAikitVersion, below) at hash time, so there is no
stored value left to drift from reality — the file this field used to duplicate.
```

## writeManifest

Moved from `decoder/parity_manifest_test.go` (the comment above `writeManifest`) on 2026-10-09.

```text
writeManifest serialises the manifest back to disk FAITHFULLY.

Two round-trip defects made scripts/refresh_parity_hashes.sh unusable — it correctly ABORTED on
"the update changed more than deps_hash" every time, which is the guard working, but it meant the
sanctioned goldens-gated refresh could not actually be used:

  - json.MarshalIndent HTML-escapes, turning a ">" inside a `reference` string into "\u003e".
  - `Method` was a plain string, so a JSON `null` came back as `""`.

Neither changed any meaning, and both showed up as a diff the abort guard could not distinguish
from a real edit. Fixed here rather than by loosening the guard: the guard was right.
```

## TestParityManifest_merge.machine

Moved from `decoder/parity_manifest_test.go` (the comment in `TestParityManifest_merge`, above the machine default) on 2026-10-09.

```text
A hardcoded specific box's name here is a mislabel waiting to happen, not a
convenience: it WAS "linux-62gb" unconditionally, so a merge run from any OTHER
machine silently stamped every row as having been validated on nobara's 62 GB
box. Caught 2026-09-06 running the arm64/Metal sweep from a 16 GB Mac — five
rows it re-validated (granite-dense, ministral3, olmo3, qwen3_moe, smollm3) got
overwritten from the correct "mac" to the wrong "linux-62gb", because nobody
had ever needed to run this from a second machine before. GOOS-GOARCH can never
be wrong the way a remembered hostname can; it is less pretty than "mac" or
"linux-62gb", but pass GOINFER_MANIFEST_MACHINE explicitly for that.
```

## rootAikitVersion

Moved from `decoder/parity_manifest_test.go` (the comment above `rootAikitVersion`) on 2026-10-09.

```text
rootAikitVersion reads the ROOT go.mod's aikit require directly (G-01, audit-2026-09-10) —
freshDepsHash's replacement for the manifest's old hand-typed "aikit_version" field. Deriving
it here instead of storing it makes drift structurally impossible: the hash always mixes in
whatever go.mod actually pins, at the moment it is computed, not whatever someone last typed
into the JSON.
```

## TestRootAikitVersion_readsFromGoMod

Moved from `decoder/parity_manifest_test.go` (the comment above `TestRootAikitVersion_readsFromGoMod`) on 2026-10-09.

```text
TestRootAikitVersion_readsFromGoMod is G-01's direct unit gate (audit-2026-09-10): asserts
rootAikitVersion actually reads go.mod rather than returning a stale or hardcoded value.
Deliberately does NOT assert a specific version string — that would just be a second place to
remember to bump on every aikit release, the exact hand-typed-value failure mode this fix
removes. Instead: independently re-parse go.mod with a SEPARATE regexp match (not calling
rootAikitVersion's own machinery) and require exact agreement, so a bug in the function itself
(wrong path, wrong pattern) cannot pass by coincidence.
```

## TestParityManifest_fresh.emptySets

Moved from `decoder/parity_manifest_test.go` (the comment in `TestParityManifest_fresh`, above the empty-set check) on 2026-10-09.

```text
A validated family whose uses/own sets name no files hashes nothing but the aikit version,
so no edit to its forward can ever restale it (audit-2026-09-10 G-02: olmo_hybrid sat
validated with "uses": [], "own": []). Its green would cover nothing. Keyed on status, not
validated_at: an experimental family (bailing_hybrid) carries a validated_at but claims no
validation, and gets its sets when it is promoted.
```

## TestParityManifest_methodTier

Moved from `decoder/parity_manifest_test.go` (the comment above `TestParityManifest_methodTier`) on 2026-10-09.

```text
TestParityManifest_methodTier is the claim-discipline gate: it makes "validated" MEAN T3.

WHY IT EXISTS. `parity-coverage-policy.md` has always defined which methods clear T3, but nothing
enforced it — `Method` was parsed as a `json.RawMessage` and never compared to the list. That let
five rows sit at `status: validated` with `method: tiny-golden` / `tiny-golden+coherent`: a T1
artifact (cosine vs the family's own seeded tiny golden — no released checkpoint involved)
recorded in a T3 slot, and therefore counted as "supported". The staleness gate could not catch it
because it keys on `deps_hash` freshness, which says nothing about how the row was validated.

THE HONEST ALTERNATIVE, so a weak row does not have to lie. `status: "experimental"` records a
family whose gate is real but sub-T3. Such a row keeps its method and metrics, renders distinctly
in the capability matrix, and is EXCLUDED from the supported count. Downgrading is not a
regression — it is the matrix finally saying what the row always was.
```

## applyParityRows

Moved from `decoder/parity_manifest_test.go` (the comment above `applyParityRows`) on 2026-10-09.

```text
applyParityRows folds PARITY_ROW lines into m and returns the families it touched. It is a
FUNCTION rather than inline loop body so the B15 regression test can drive the real merge
instead of a re-implementation of it — the defect it guards against (status promoted without
the method to support it) was exactly the kind a parallel test-only copy would have missed.
```

## applyParityRows.status

Moved from `decoder/parity_manifest_test.go` (the comment in `applyParityRows`, above the status assignment) on 2026-10-09.

```text
STATUS IS DERIVED FROM THE METHOD, NOT ASSERTED (B15). This used to read
`f.Status = "validated"` unconditionally, so one sweep with EMIT_MANIFEST=1 promoted
glm4_moe, mixtral, qwen2_5_vl and qwen2_moe to *supported* on tiny-golden evidence —
the published capability matrix's "supported" count, upgraded by a tool, from rows
that said tiny-golden right next to the promotion. TestParityManifest_methodTier
caught it, which is the gate working; but a writer that produces claims a reader has
to reject is the wrong shape. The rule the tier gate enforces is now the rule the
writer applies: T3 method ⇒ validated, anything else ⇒ experimental. A row can
therefore DEMOTE a family whose evidence was downgraded, instead of leaving a stale
"validated" standing over a weaker method.
```

## parthreshold_sweep_bench_test.header

Moved from `decoder/parthreshold_sweep_bench_test.go` (the comment at the top of the file) on 2026-10-09.

```text
The decode-matmul parallelism threshold sweep. int4ParThreshold (weightmat.go, 1<<20)
and DefaultDecodeParallelThreshold (tune.go, 300K) are the fan-out crossovers, in MACs,
below which a decode matmul runs SERIAL. Both were tuned on a Ryzen 7 3700X (8 desktop
cores); the M1 Pro (6 P + 2 E, very different memory latency) is Phase 5's rig and was
never swept — a wrong threshold understates any benchmark we eventually publish. This
isolates the crossover from the model: it drives the SAME kernels (MatmulBTW4A8Into /
MatmulBTW8A8Into) matmul() dispatches, at Gemma-4-26B-A4B's real decode shapes (M=1),
across a bracket of thresholds, with a per-call Workspace exactly as the forward does.

Run + read (per-shape ns/op; lower is better; the winning threshold is the largest one
whose ns/op is still ~serial-beating for every shape it must fan out):

	go test ./decoder -run '^$' -bench 'ParThresholdSweep' -benchtime 200ms

Interpreting it: for each shape the ns/op is flat until the threshold rises ABOVE that
shape's MAC count, at which point it jumps to the serial cost. The optimum is the
threshold that keeps every shape you want parallel below the jump while leaving truly
tiny ops (which over-parallelize — thread spawn > work) serial. Compare the M1 Pro
optimum against the constants; if it differs materially, the constant wants to be
per-platform (GOOS/arch or GOMAXPROCS-derived), not a universal default.
```

## TestPrefillAbsoluteTable

Moved from `decoder/prefill_absolute_test.go` (the comment above `TestPrefillAbsoluteTable`) on 2026-10-09.

```text
Re-measure benchmarks.md §A's ABSOLUTE CPU prefill table.

The recorded cells are dense 1.5B `int8int8`, prefill + 1 token, on an M1 Pro:

	170 tok    3.3 s   (51.5 tok/s)
	620 tok   19.7 s
	1520 tok  93.2 s
	3020 tok 334.9 s   (9.0 tok/s)

They predate FOUR changes that landed 2026-09-01 — the f32 prefill default,
A3's head fan-out (1.92× @K=4096), P18's expert-major MoE prefill (4.36×, MoE
only, so inert for this dense model), and P19's fused schedule (+8%). The page
was marked stale rather than guessed at; this produces the replacement.

CONFIGURATION IS MATCHED TO THE ORIGINAL ON PURPOSE: same model class, same
quant, same prompt lengths, same "prefill + 1 token" quantity. A re-measurement
that quietly changes the cell definition is not a re-measurement.

Best-of-3 rather than the original's single shot — an improvement to the
method, stated so the two are not read as identically obtained.
```

## TestAdmitPrefillMemory_refusesAnOversizedRequest

Moved from `decoder/prefill_budget_test.go` (the comment above `TestAdmitPrefillMemory_refusesAnOversizedRequest`) on 2026-10-09.

```text
R13 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): the load-time guard prices the
worst case a request COULD reach; it cannot see the request that actually arrives. This is the
request-time counterpart — driven with numbers shaped like the actual failure (a 7B-class
model whose weights alone fit, but whose KV+scratch for a real agent-sized prompt does not).

R13-follow-on: total RAM here is AMPLE (so Load succeeds and the load-time guard has nothing to
say) but currently-AVAILABLE RAM is tight — the shape of the live re-run's actual failure: the
load-time guard correctly auto-pinned a smaller context against total RAM, and the request
still swapped, because other processes on the real machine had already claimed most of what
the guard assumed was free.
```

## TestAdmitPrefillMemory_residentPathSkipsHostKV

Moved from `decoder/prefill_budget_test.go` (the comment above `TestAdmitPrefillMemory_residentPathSkipsHostKV`) on 2026-10-09.

```text
TestAdmitPrefillMemory_residentPathSkipsHostKV is P-01(b) (audit-2026-09-10): a request that will
actually run the stateless GPU-resident path never allocates the host KV AdmitPrefillMemory
prices, so pricing it anyway could 413 a request an 8 GB CUDA box would have served entirely in
VRAM. availBytes is chosen so the window is narrow: prefill scratch alone must fit (residentPath
admits), but scratch+KV together must not (residentPath=false — the CPU/staged path, which does
need the term — still refuses). If admission stopped depending on residentPath at all, the first
assertion would go green vacuously; if the KV term were never skipped, the second would fail.
```

## TestCachedHostRAMAvailable_rateLimits

Moved from `decoder/prefill_budget_test.go` (the comment above `TestCachedHostRAMAvailable_rateLimits`) on 2026-10-09.

```text
TestCachedHostRAMAvailable_rateLimits is P-13 (audit-2026-09-10): AdmitPrefillMemory runs on
every request that reaches prefill, and the raw probe forks+execs (vm_stat on darwin). This
asserts the rate-limiting directly — calls within the TTL window must not reach the underlying
probe — rather than only asserting AdmitPrefillMemory still returns the right answer, which
would pass whether or not caching ever happened.
```

## prefill_decline_warn_test.header

Moved from `decoder/prefill_decline_warn_test.go` (the comment at the top of the file) on 2026-10-09.

```text
B20 (docs/queue-engineering.md) gate: residentPrefillSeed's decline must actually reach an
operator-visible log naming the reason, not merely be read internally.

WHY THIS EXISTS. V-05 (docs/review-2026-09-04.md) fixed a SILENT batched-prefill decline by
making the launch refuse loudly (checkPrefillShmem et al.) instead of crashing, but B20 found
the caller one level up, residentPrefillSeed, still threw that refusal away after checking it
only for cancellation — the reason was reachable and correct but never reached an operator.
5cc4854 (2026-09-04, later the same day as the review) wired warnPrefillDeclined into the
fallback, but nothing asserted on the log line's CONTENT — a doc comment claiming coverage is
not coverage (CLAUDE.md's own rule) applies just as much to an unasserted log line as to an
unasserted test body.
```

## TestResidentPrefillSeed_DeclineIsLoggedWithReason.capture

Moved from `decoder/prefill_decline_warn_test.go` (the comment in `TestResidentPrefillSeed_DeclineIsLoggedWithReason`, above the Generate call) on 2026-10-09.

```text
Generate MUST be started INSIDE the capture, not before it. captureStderr installs its pipe
by assigning the os.Stderr global, and warnPrefillDeclined reads that global from the
generation goroutine — so starting the goroutine first is a genuine data race on os.Stderr
(caught by -race in CI on both linux and darwin, 2026-09-12, run 34711378811; the detector
fired during this test and so blamed it, while the write was this line's own).

It was also a correctness bug independent of the detector: the decline is logged during
PREFILL, which is the first thing the goroutine does, so any decline emitted between
Generate returning and the swap landing went to the REAL stderr and was missed. The test
would then assert on output it never captured — a flake that looks like a missing log line.
Starting inside the closure makes the capture cover the whole generation, which is what the
assertion below already assumed.
```

## TestWarnPrefillDeclined_FiresOncePerReason

Moved from `decoder/prefill_decline_warn_test.go` (the comment above `TestWarnPrefillDeclined_FiresOncePerReason`) on 2026-10-09.

```text
TestWarnPrefillDeclined_FiresOncePerReason gates N-35 (audit-metal-2026-09-12.md): the dedup key
is now the decline's normalized reason, not a single process-lifetime gate. B20's original fix
sketch wanted per-reason dedupe; the FIRST shipped version was coarser (a bare sync.Once) —
TestWarnPrefillDeclined_FiresOncePerProcess used to pin that coarser behaviour deliberately, on
the grounds that a future change to it should be a decision, not a silent drift. This is that
decision: on Metal, nearly every prompt under the fast-prefill floor declines with a message
that differs only in its promptLen, so a bare sync.Once let the very first short prompt
permanently silence a later, genuinely different decline (a resident-cap refusal, an OOM) an
operator would want to see. Numbers are normalized out of the key so same-shape declines with
different byte/token counts still collapse to one line, while a differently-worded reason gets
its own.
```

## TestCheckpointSHA256_directory

Moved from `decoder/prefill_ref_dir_key_test.go` (the comment above `TestCheckpointSHA256_directory`) on 2026-10-09.

```text
A safetensors checkpoint directory keys a reference by its files (D-B01's first directory cell failed with "is a
directory"): the key is stable, ignores goinfer's own sidecars and hidden files, and changes when a weight byte does.
```

## TestPrefillGateReference

Moved from `decoder/prefill_ref_gen_test.go` (the comment above `TestPrefillGateReference`) on 2026-10-09.

```text
TestPrefillGateReference is Phase A of docs/completed/task-prefill-gap.md §3.1's L1 re-run: build the CPU
f32-activation reference logits that BOTH Metal arms (metal/prefill_gate_ref_test.go, Phase B)
are scored against. §3's first form scored Metal's fast (f16-activation) path against Metal's
own exact (int8-per-row-activation) path and called the exact path truth — but the exact path is
itself a quantisation, guaranteed to disagree with the fast path for a reason that has nothing
to do with a defect (see the doc's §3.1 correction). This reference is what actually has no
activation-precision loss to attribute anything to.

Runs in its OWN process, deliberately separate from the Metal gate: a 7B CPU reference and a 7B
Metal resident sharing 16GB at once on this machine is exactly the failure mode that produced a
real kernel panic here before a model was ever intentionally run oversized. Two models:
  - S (1.5B): Options{Backend:"cpu", Quant:""} — f32 weights, f32 activations, ~6GB.
  - D7 (7B):  Options{Backend:"cpu", Quant:"int8"} — weight-only per-row int8, f32 activations,
    ~7.6GB. The f32 weights for D7 would be ~28GB and do not fit; per the brief, if the q4_k_m
    GGUF cannot load under "int8" this subtest logs why and skips — D7 is the confirmation
    model, S is the one the decision rests on.

The reference is the CPU backend's own prefill (PrefillLogitsForTest, the batched prompt path —
weights streamed once and reused across all K positions, ~1.7-2x faster than a naive per-token
loop and bit-identical to it) with GOINFER_CPU_FAST_ATTENTION forced to "0", i.e. the exact
f64-accumulating attention kernel, never the f32-fast one that is default ON elsewhere in this
tree. The 64-token greedy continuation past the prompt still runs one token at a time
(ForwardForTest) — inherent to greedy decoding (each step needs the previous step's own choice),
and cheap next to K up to 3900.

PARALLEL ACROSS PROMPTS: the 10 prompts per (model, K) run concurrently, one goroutine per CPU
(capped), each with its own *KVCache. This is safe because a CPU forward's mutable scratch state
lives on the KVCache it's given (decoder/model.go's cache.scr), not on the shared *Model — the
only mutex on *Model guards LoRA adapter swaps, a different concern. Verified empirically, not
just argued: before the real run, a short preflight sends the SAME prompt through N concurrent
workers on independent caches and requires bit-identical seed logits; a real corruption would
show up there in seconds rather than being discovered hours into results that already cost the
wall-clock this parallelism exists to avoid.

Output is NOT written into the repo — these are large, per-machine, per-session binaries
(10 prompts x 65 x vocab float32s per cell) meant only for the Phase B run on this box, not a
committed artifact. Written to ~/goinfer-logs/prefill-ref/<model>-K<k>-p<i>.bin via
decoder.WritePrefillReferenceForTest.

	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run TestPrefillGateReference -v -timeout 4h
```

## TestPrefillGateReference.k512

Moved from `decoder/prefill_ref_gen_test.go` (the comment in `TestPrefillGateReference`, above the models table) on 2026-10-09.

```text
K=512 joins the decision set here (docs/completed/task-prefill-gap.md §4 L1, 2026-09-09): CUDA's floor was
set from a MEASURED K=512 cell, never interpolated between 256 and 1024 (§3, "a floor placed
between two measured cells would be interpolating a fidelity result nobody took") — Metal's
floor needs the same discipline if this run ships.
```

## TestPrefillGateReference.residentContext

Moved from `decoder/prefill_ref_gen_test.go` (the comment in `TestPrefillGateReference`, above the Load call) on 2026-10-09.

```text
ResidentContext is a GPU-resident-KV concept (decoder/model.go's Options doc: "Ignored
off the residency path") — it does not change what the CPU backend actually allocates,
only what fitCheckFor prices the load's KV term at (decoder/fitguard.go's effCtx: pinned
ResidentContext when set, else the model's own MaxPositions — 32768 here, priced whether
or not this run ever reaches it). Pinning it to what this run actually touches
(maxK+continuationN) turns an 8.4 GB / 12.0 GB-available guard threshold, priced against
a context length nothing below ever requests, into an honest ~6.8 GB / 9.8 GB one.
```

## refWorkers

Moved from `decoder/prefill_ref_gen_test.go` (the comment above `refWorkers`) on 2026-10-09.

```text
refWorkers lets a run cap how many prompts are prefilled CONCURRENTLY, via
GOINFER_CPU_REF_WORKERS. The default (8) is unchanged and right for the shallow standing cells.

AT DEPTH, CONCURRENCY IS A LOSS, NOT A WIN — measured, and expensively. The exact f64-accumulating
attention materialises K x K score rows per head, which at K=8000 sit far outside the 3700X's
32 MB of L3; eight prompts doing that at once contend for memory bandwidth until each is slower
than running them in series:

  - S (1.5B, f32) at K=8000: eight concurrent prompts took 88 min EACH; the last two, running as a
    pair, took 10.6 min each. Throughput 0.091 -> 0.189 prompts/min on FEWER workers.
  - D7 (7B, int8) at K=8000 with 8 workers wrote no file in 8h50m; its KV fill (RSS growth, which
    matched S's own timeline to 2 min) put it ~8 of 28 layers in, projecting 24-28 h against a
    16 h test timeout. One prompt alone at f32 fits K=1024/2048/4096 at 2.11/3.32/6.83 min and
    projects ~17.5 min at K=8000: the 8-worker run was paying a 10x+ contention penalty.
  - two workers at K=2048 (f32) bought +8.6% throughput over one — so the break-even is somewhere
    between 2048 and 8000, and one worker is the safe choice at the deep cells.

Record: docs/measurements/vsum-split-fidelity-2026-09-13.md, deviation D2.
```

## TestPrefillAttnPoolInvariance

Moved from `decoder/prefillattnpool_test.go` (the comment above `TestPrefillAttnPoolInvariance`) on 2026-10-09.

```text
G16 gate — prefill attention's head-parallel fan-out must be BIT-IDENTICAL to
the serial path it replaces.

A1's constraint is the whole reason this is allowed at all: "Parallelism may
only split independent outputs across workers/registers — heads, ...". Splitting
heads across workers therefore cannot change any value, only who computes it.
That is a claim about the code, and this is the test that makes it a fact.

A1 asserted its own bit-identity through the parity goldens and left no
pool-invariance test, so this is new: same prompt, same cache, pool len 1 vs
the budgeted count, compared float-for-float. Exact equality, not a tolerance —
a tolerance here would silently accept the reassociation A1 exists to prevent.
```

## TestPrefillAttnWorkerBudget

Moved from `decoder/prefillattnpool_test.go` (the comment above `TestPrefillAttnWorkerBudget`) on 2026-10-09.

```text
The worker count must stay inside its caps and honor its override.

NOTE — this test was rewritten when G20 landed, and the reason matters more
than the assertions. As written for G16 it asserted that K=32768 "must fall
back to serial, a 4 GB slot", because an untiled slot's scores buffer was
K*nKeys floats. G20 tiles the query rows, so no such slot exists any more: a
slot is one row tile wide (attnScoreTileBytes), and six workers at 32k cost
~150 MB, not ~25 GB. The old assertion was not wrong when written; its premise
was removed. That is why it is replaced rather than relaxed — the property it
protected (unbounded per-slot growth must not happen) is now asserted directly,
against the real tiled size, by TestAttnRowTileBoundsScratch.
```

## TestPrefillAttnWorkerBudget_countsFusedScratch

Moved from `decoder/prefillattnpool_test.go` (the comment above `TestPrefillAttnWorkerBudget_countsFusedScratch`) on 2026-10-09.

```text
TestPrefillAttnWorkerBudget_countsFusedScratch is P-05's gate: prefillAttnWorkers' budget must
bound what newHeadWorkerPool ACTUALLY allocates. newHeadWorkerPool allocates a fusedScratch
(sBlk+tmp+acc+mRun+lRun+vBlk) ALONGSIDE the materialized shape (scores/kh/vt/qh/ch) whenever
GOINFER_FUSED_ATTENTION is enabled — "both exist while fusion is a flag" — so the real per-slot
footprint is the materialized shape PLUS fusedScratch, not the materialized shape alone. Before
the fix, the budget counted only the materialized shape and oversubscribed
prefillAttnScratchBudget by ~25% at K=nKeys=8192.
```

## TestNewHeadWorkerPool_skipsMaterializedWhenFused

Moved from `decoder/prefillattnpool_test.go` (the comment above `TestNewHeadWorkerPool_skipsMaterializedWhenFused`) on 2026-10-09.

```text
TestNewHeadWorkerPool_skipsMaterializedWhenFused is P-05's completion (audit-2026-09-02): the
budget-accounting fix above closed the measurable oversubscription, but left this doc's own
disposition text recording that "vt is unused when fusion is ACTIVE ... and scores ... is unused
whenever fusion is active REGARDLESS of useAcc64" as a genuine, un-eliminated allocation — real
memory allocated and never touched. This asserts the elimination directly, not just that the
paths it feeds still compute the right answer (TestFusedAttention_matchesMaterialized and
TestAttendF32Fanout_bitIdentical already gate that): a caller that can promise fusedOK stays true
for the whole pool's lifetime (wantFused=true) gets NIL vt/scores when fusion is actually enabled,
and the ordinary fully-allocated pool otherwise — proving both the win and that no caller silently
loses a buffer it needs (which is the nil-slice-access risk the original disposition declined to
risk without this exact three-way condition pinned down).
```

## TestDecodeScratch_headWorkerPool_skipsKhVtUnderAcc64

Moved from `decoder/prefillattnpool_test.go` (the comment above `TestDecodeScratch_headWorkerPool_skipsKhVtUnderAcc64`) on 2026-10-09.

```text
TestDecodeScratch_headWorkerPool_skipsKhVtUnderAcc64 is P-03 (audit-2026-09-10), the decode-path
sibling of P-05 (09-02, newHeadWorkerPool): the acc64 kernels (MatmulQKAcc64/MatmulAVAcc64) read
keys/vals directly with strided addressing, so kh/vt are unused whenever useAcc64 is true — and
attention.go's decode path hardcodes acc64 := true unconditionally, making this the ONLY case
headWorkerPool's one caller ever reaches. Both states are checked directly (kh/vt nil under
acc64, still allocated under !acc64) so this doesn't just prove "the acc64 case works" while
silently also proving a caller that genuinely needs kh/vt would be left with nothing.
```

## prefillcancel_test.header

Moved from `decoder/prefillcancel_test.go` (the comment at the top of the file) on 2026-10-09.

```text
Gates for G18 — prefill honoring cancellation.

The before-state these pin: the serve layer passed r.Context() into drive
correctly, but the context stopped at generateInto — prefillLogits,
forwardLayersN and runLayersFromEmbedN took no context at all. An abandoned
client therefore left a core prefilling to completion, measured at 47:38 of
CPU with nothing attached, and a retrying harness stacked one such generation
per retry.

These use an ALREADY-cancelled context and a mid-flight cancel rather than a
long prompt, so they are fast and deterministic: what is being gated is that
the loop looks at all, and that it stops promptly once it does.
```

## prefillpath_test.header

Moved from `decoder/prefillpath_test.go` (the comment at the top of the file) on 2026-10-09.

```text
The prefill half of the seam gate (resident_seam_test.go).

WHY THIS EXISTS. The resident DECODE path being silently CPU-only was one bug class; the batched
PREFILL declining silently is the same class one layer down, and it shipped. `--backend cuda
--quant int8int8` builds a full resident decode path (ResidentActive is true, decode runs at 0.7×
int4 — everything looks healthy), but the batched prefill GEMV is int4-only, so every prompt takes
the sequential per-token loop instead of one weight-stationary pass: measured 1.73 s vs 0.19 s on a
300-token prompt (9×), 4.56 vs 0.22 CPU-seconds (20×). generateInto's fallback discards the decline
error by design, so nothing — no log, no field, no error — said so.

These tests need no GPU: they fake residents that decline the way the CUDA backend declines.
```

## defaultProgressInterval

Moved from `decoder/progress_test.go` (the comment above `defaultProgressInterval`) on 2026-10-09.

```text
A long test must be legible WHILE it runs: at any moment the operator should be able to tell
whether it is stuck and roughly how much is left. t.Logf cannot do that — it is buffered until
the test returns, and dropped entirely for a passing test without -v. That is not a style
preference; the tests wired to this helper each ran for 2–15 minutes emitting nothing at all,
which is indistinguishable from a hang until the whole suite ends.

So: os.Stderr, unbuffered, independent of -v, on a TIME ticker rather than an iteration count
(an iteration count picked for one machine goes silent on a slower one, which is exactly when
the heartbeat matters most). TEST_PROGRESS_INTERVAL overrides the cadence; "0" silences it.

RUN THESE UNDER -v. `go test` buffers a package's output and DISCARDS it entirely when the
package passes, so without -v these lines never appear — measured, not assumed: a passing run of
TestA3FastAttentionDivergence emitted a 55-byte log holding only the "ok" line. os.Stderr does
not dodge that; what it buys over t.Logf is that under -v the lines stream AS THEY HAPPEN
(verified: 5s apart by wall clock mid-test) instead of arriving in one dump when the test ends.
```

## progress.emit.eta

Moved from `decoder/progress_test.go` (the comment in `emit`, above the ETA) on 2026-10-09.

```text
From the RECENT rate, not from total elapsed over total done. A test whose counted
work follows a long uncounted phase — an 80B checkpoint load, say — otherwise
divides 14 minutes by one finished item and reports eta=2h20m for work that took
six. Measured on exactly that run. Uneven() still turns it off entirely where the
items differ in cost and no window makes the projection honest.
```

## progress.emit.rate

Moved from `decoder/progress_test.go` (the comment in `emit`, above the rate) on 2026-10-09.

```text
Rate over the LAST interval, not the whole run. A cumulative average is still digesting
cold-cache page-ins minutes in -- measured here: mellum2's cumulative eta read 31m, 25m,
22m, 20m on successive ticks while the machine had not actually changed speed that much. A
recent rate tracks what it is doing NOW, and stays honest on uneven work where an ETA cannot.
```

## progress.emit.io

Moved from `decoder/progress_test.go` (the comment in `emit`, above the io field) on 2026-10-09.

```text
Bytes read by this process. A phase with nothing to count -- loading a 162GB checkpoint, say
-- otherwise produces a heartbeat that proves only that the process is ALIVE, not that it is
getting anywhere, and those are the two states the reader needs to tell apart. This is the
exact signal that had to be dug out of /proc/PID/io by hand while the qwen3next oracle sat
silent for ten minutes. Linux-only; absent elsewhere, and simply omitted there.
```

## progress.ioProgressFrom

Moved from `decoder/progress_test.go` (the comment above `progress.ioProgressFrom`) on 2026-10-09.

```text
ioProgressFrom is the rate arithmetic, split from the /proc read so it can be driven with known
values on any platform. The bug it now covers only showed on a real 162GB load.
```

## progress.ioProgressFrom.window

Moved from `decoder/progress_test.go` (the comment in `ioProgressFrom`, above the lock) on 2026-10-09.

```text
Its own timestamp, NOT lastAt. emit() calls rate() first, which sets lastAt = now, so
reusing it here made dt zero every time and the io rate never printed once — visible only
on a real 162GB load, where the field showed a total and never a rate.
```

## TestIORateSurvivesTheItemRateWindow

Moved from `decoder/progressmeta_test.go` (the comment above `TestIORateSurvivesTheItemRateWindow`) on 2026-10-09.

```text
emit() calls rate() before ioProgress(), and rate() stamps its own window. When both read the
same timestamp field, ioProgress always saw dt == 0 and silently never printed a rate — the
total appeared, the rate never did, and it took a 162GB load to notice. The windows are separate
now, and this drives the arithmetic with known values so the regression cannot return unseen.
```

## TestRateTracksRecentWindowNotAllHistory

Moved from `decoder/progressmeta_test.go` (the comment above `TestRateTracksRecentWindowNotAllHistory`) on 2026-10-09.

```text
The ETA is built from the recent rate. That matters when counted work follows a long UNCOUNTED
phase: the qwen3next oracle spends ~13 minutes loading an 80B checkpoint before its first
countable token, and a cumulative estimate divided that whole span by one finished item and
announced eta=2h20m for work that took six minutes. The recent window has to dominate.
```

## hiddenAllGolden

Moved from `decoder/prompt_hidden_all_test.go` (the comment above `hiddenAllGolden`) on 2026-10-09.

```text
D11's gate (docs/tasks/task-constrained-confidence.md, Route C): the final-norm hidden state at EVERY prompt position must match HF's
last_hidden_state[0, :] per position, cosine >= 0.9999 and relative L2 <= 1e-5, in f32, on the tiny Qwen3.5 checkpoints (dense, MoE, and the
derived qwen3_5-tiny-normw whose random final-norm weight is what lets a missing, doubled or mis-weighted final norm fail: cosine alone cannot see
a uniform scale, and the other tiny checkpoints' final-norm weights are all 0, a scale of exactly 1 under the add-one RMSNorm).

UNLIKE TestPromptHidden_matchesHF, A MISSING CHECKPOINT OR GOLDEN FAILS: that test skips when qwen3_5-tiny-normw has no model.safetensors, and D2
committed its config but not that file, so on a fresh checkout the one fixture that sees the final norm was skipped (found 2026-10-02; the file is
committed now). A gate whose key subtest can vanish into a skip is not a gate.

Both the public path and the sequential path are compared with HF, so a mutation of either final-norm site fails. Regenerate the golden with
scripts/pin_prompt_hidden_all.py.
```

## TestPromptHiddenAll_batchedMatchesSequential.bar

Moved from `decoder/prompt_hidden_all_test.go` (the comment in `TestPromptHiddenAll_batchedMatchesSequential`, above the bar) on 2026-10-09.

```text
f32: every position within 1e-6. int4: the activations are quantized to int8, so a few-ulp difference between the batched and
per-token kernels can tip ONE activation code and move one position by about 1e-3 (measured on the Mac's arm64 CI runner, 2026-10-02:
1 of 64 positions at 1.75e-3, position 47, the 16 after it back at <=1e-7, so it is a flip and not carried state; the same fixture
unquantized has none above 1.2e-7). A real batched-path bug elevates many positions, so the int4 bar is a flip ALLOWANCE (at most 1
position or 2% above 1e-6, each under 1e-2, the median still within 1e-6), not a looser per-position number.
```

## TestGiwInt4LabelNotMix

Moved from `decoder/quant_label_test.go` (the comment above `TestGiwInt4LabelNotMix`) on 2026-10-09.

```text
TestGiwInt4LabelNotMix is the T1-6 regression. A .giw baked with `-quant int4` has int4
projections but an int8-pinned embedding / LM head (the logit-critical default; the
EmbedInt4 knob relaxes it). quantLabel used to scan those tables, see int4 coexisting with
int8, and report "int4mix" — while the batched-prefill gate, which inspects only the seven
int4 projections, correctly batched. So /health showed `decode_path: …(int4mix)` beside
`prefill_batched: true`, and the label named a quant the bundle is not.

The .giw path is what triggers the inference: a direct Load records the requested quant
string and returns it verbatim, never inferring. So the round-trip through SerializeWeights
is load-bearing here, not incidental.
```

## TestQuantNoiseFloor_gemma4MoE

Moved from `decoder/quant_noise_floor_test.go` (the comment above `TestQuantNoiseFloor_gemma4MoE`) on 2026-10-09.

```text
TestQuantNoiseFloor_gemma4MoE is the Split-B pre-flight: it measures, on CPU, the two things that
actually predict whether a resident (GPU) MoE kernel can hit parity on this fixture — measured
BEFORE any kernel is written, at the cost of a few CPU forwards and no GPU.

It began as a pure int4-vs-f32 "noise floor" (CPU-at-quant vs CPU-f32) gated at 0.97, on the theory
that a resident backend can only agree with CPU-int4 as well as int4 agrees with f32. The 0.97 bar
turned out UNCALIBRATED (not the floor irrelevant): the Split-A dense two-geometry control PASSES
the resident gate (cuda-int4 vs cpu-int4) at cosine 0.979 while its own int4-vs-f32 floor is only
0.880 (NOISE_FLOOR_CKPT=../testdata/gemma4-dense-twogeom-tiny). The floor is a CONDITIONING PROXY,
correlated with — not independent of — resident parity (both moved together: hidden=64 floor bad +
gate 0.82; hidden=256 floor 0.88 + gate 0.979). CUDA-vs-CPU-int4 is only PARTLY common-mode: same
quantized weights, but each side quantizes activations with its own rounding/grouping, and how much
that difference amplifies is exactly the conditioning the floor measures. One control point fixes
0.88-was-fine for that fixture, not a general threshold — so keep the floor REPORTED as a warning
signal, demoted from a hard gate.

What a resident MoE kernel can get wrong that a dense one can't is a ROUTING FLIP: quant noise near
a router tie picks a DIFFERENT expert — a different computation, not a small numeric error. So the
gate is (1) routing agreement 100% and (2) a min routing MARGIN wide enough that the tighter
cpu-int4-vs-gpu-int4 gap can't flip it either. (History: the Split-A dense fixture at hidden=64
manufactured a phantom "bug" — cuda resident cosine drifted to 0.82, pure int8-activation
sensitivity, fixed by hidden≥256. Same instinct built this fixture at hidden=64; measure first.)
```

## TestQuantNoiseFloor_gemma4MoE.gate

Moved from `decoder/quant_noise_floor_test.go` (the comment in `TestQuantNoiseFloor_gemma4MoE`, above marginFloor) on 2026-10-09.

```text
GATE. What a resident MoE kernel can get wrong that dense can't is a ROUTING FLIP — the one
discrete failure mode, unrecoverable by any kernel. So the pre-flight gates on:
  (1) routing agreement 100% (int4 must not flip the top-k vs f32 — a flip is a fixture defect), and
  (2) a min int4 routing margin comfortably above the perturbation, so the residual CUDA-vs-CPU-int4
      gap can't flip it either. 0.02 ≈ 2 pts of router prob; mm4=0.12 gives ~6× headroom, and the
      reported f32→int4 erosion shows the actual margin loss for context.

The int4-vs-f32 logit floor is a WARNING SIGNAL, not a gate — a conditioning proxy, CORRELATED with
(not independent of) resident parity: at hidden=64 the floor was bad AND the resident gate was 0.82;
at hidden=256 the floor was 0.88 AND the gate was 0.979 — both moved together. The one control point
(dense two-geom: f32-floor 0.880 → resident 0.979, NOISE_FLOOR_CKPT=…/gemma4-dense-twogeom-tiny)
establishes 0.88 was fine FOR THAT FIXTURE, NOT that any lower value is fine in general. So the 0.97
bar was uncalibrated, not wrong to measure — keep it REPORTED and demoted. If the resident MoE gate
comes back marginal, this low floor (0.79) is the first suspect, and the number is already on record.
```

## qwen25vl_layout_split_real_test.header

Moved from `decoder/qwen25vl_layout_split_real_test.go` (the comment at the top of the file) on 2026-10-09.

```text
G-S3c's root cause (docs/tasks/task-multimodal-support-2026-10.md, S3): on the Mac, the Qwen2.5-VL image turn's
first token differs between `--backend metal` ("Table") and `--backend cpu` ("Quarter"), and the first token comes from
the CPU prefill in both arms (Metal has no resident m-RoPE prefill). The suspect is the CPU's int4 layout, which Load
keys on Options.Backend: canonical int4 under "metal" (wantsRow4Fallback false), the arm64 row4 repack under "cpu".
```

## qwen25vl_real_test.header

Moved from `decoder/qwen25vl_real_test.go` (the comment at the top of the file) on 2026-10-09.

```text
Real-model gate for Qwen2.5-VL (Qwen/Qwen2.5-VL-3B-Instruct, model_type "qwen2_5_vl") — the
T3 promotion of the qwen2_5_vl family from tiny-golden to a released checkpoint.

THE ORACLE SHAPE DIFFERS FROM EVERY OTHER FAMILY IN THIS BATCH. The existing tiny golden
(TestQwen25VL_e2eChain) is already e2e encoder→decoder, but on SYNTHETIC pixel_values with
no real processor in the loop. This gate instead runs the real AutoImageProcessor on a real
image (testdata/qwen25vl_preprocess_image.png — pre-sized so smart_resize is a no-op,
isolating decoder-on-real-weights from resize/bicubic parity, which
pin_qwen25vl_preprocess.py already pins separately) through the real vision encoder and the
real text decoder. Fixture: scripts/pin_qwen25vl_real.py.

SCOPED TO THE PREFILL FORWARD ONLY — NOT GREEDY CONTINUATION. Every other real-checkpoint
gate in this session's finishing pass checks a multi-step greedy continuation using
m.forward(id, cache) for plain text tokens; decoding PAST an image block (m-RoPE position
continuation from the image grid's max position) is a genuinely different code path that no
existing Go test exercises yet. Building that here would risk conflating a new test
harness's own correctness with the checkpoint's — this gate proves the real vision
encoder + real decoder produce correct logits on the real weights, which is the claim that
matters for T3; continuation-after-image is a separate, unbuilt capability, not silently
assumed.
```

## TestQwen2Moe_forwardParity.stat

Moved from `decoder/qwen2moe_test.go` (the comment in the forward-parity test, above the fixture stat loop) on 2026-10-09.

```text
Stat EVERY file Load needs, not just the dir and not just one of them. The failure mode runs
both ways: the tokenizer/config JSONs can end up present (a stray `git add`) while the large
model.safetensors stays uncommitted, and — as seen on a box where the HF download was
interrupted — model.safetensors can be present while config.json is absent. Either way a
partial fixture slips past a one-file guard and Fatalfs in Load, which reads as a numeric
parity regression: it is what made scripts/refresh_parity_hashes.sh refuse a provably
non-numeric refresh. An incomplete fixture must SKIP, like the siblings.
```

## TestQwen35GGUF_gate.fitguard

Moved from `decoder/qwen35_gguf_gate_test.go` (the comment in `TestQwen35GGUF_gate`, above the Load call) on 2026-10-09.

```text
The fit guard is bypassed for this model only. This gate checks the LOADER's parity, not memory planning, and
the guard now prices the 37 GB mapped checkpoint as resident for the whole load (79.5 GB needed against a 35.9 GB
budget on nobara-pc's 62 GB, 2026-09-30), which refused a load this box completed in the v0.19.0 sweep: the
mapping is page cache, not anonymous memory. requireHeavyModel keeps it to the box that holds it.
```

## TestQwen35GGUF_gate.coherence

Moved from `decoder/qwen35_gguf_gate_test.go` (the comment in `TestQwen35GGUF_gate`, above the argmax bar) on 2026-10-09.

```text
Coherence-vs-bf16 bar. The GGUF path legitimately carries MORE quant error
than Gate 2 (Q8_0→int8 double-quant vs a single bf16→int8 step), so it sits
just under Gate-2's 74/80 + 0.99466: one box measured argmax 68/80, cosine min
0.99445, the two misses being rank-2 near-ties (gap ~0.003 of logit range).

The min-cosine value is BOX-SENSITIVE: the Q8_0 dequant→forward runs through
SIMD/FMA whose reduction order differs across CPUs, so the worst-position cosine
varies by ~0.0015 between machines with no code change. The v0.10.0 release box
(Linux RTX-2070) measures argmax 69/80, cosine min 0.99298 — and a bisect confirmed
v0.9.2 produces the IDENTICAL 0.99298 there (i.e. NOT a regression; the 0.99445
above was a different machine). So the min-cosine floor is set below the lower
observed machine with margin; the REAL loader-defect detectors are the guards
ABOVE — minCos<0.98 (a bug craters well past this), the near-tie requirement
(maxDivFrac>0.03), and argmax<66 — none of which are box-sensitive.
```

## TestQwen35GGUF_gate.rebaseline

Moved from `decoder/qwen35_gguf_gate_test.go` (the comment in `TestQwen35GGUF_gate`, above the min-cosine floor) on 2026-10-09.

```text
RE-BASELINED 2026-08-22 for v0.15.0. The 0.992 min floor was set at 2583a2b and
never revisited; `6d4fc79` ("qwen35 family: quantize the projections that were f32
at every quant") crossed it. That commit is a DELIBERATE bandwidth trade — 1.60x
decode, 7.4x TTFT — and it re-baked its SIBLING gate (TestQwen35Real_gate2FullModel,
int8-vs-bf16, floor ≥0.98: 0.99333 -> 0.99069) while this one was missed. Bisected
on this box at `5bcaf53`, adjacent pair, same toolchain both sides; the Go 1.27 bump
was the obvious suspect and is REFUTED (bit-identical 0.98740 at go 1.26.6 and 1.27.0):

	33879dd (parent)  argmax 69/80  min 0.99298  mean 0.99846   PASS
	6d4fc79           argmax 68/80  min 0.98740  mean 0.99608   FAIL vs 0.992

The MEAN now carries the systematic-drift duty and the min only catches catastrophic
single steps — the same split TestQwen35GGUF_vsSafetensors was resolved to, and for
the same reason: min is box-sensitive here (~0.0015 across CPUs, per the note above)
so it cannot also police drift. Both bars sit below the measured value with margin
rather than at it; nudging a floor to just-clear the observation is how a real
regression gets blessed, which is the standard 6d4fc79's own message set.
```

## TestQwen35GGUF_locateDivergence.question

Moved from `decoder/qwen35_gguf_locate_test.go` (the comment at the top of the file) on 2026-10-09.

```text
WHERE does the GGUF path diverge from the safetensors path? — the localizer for B13's last
standing red (docs/queue-release.md; the v1.0 gate §1).

THE QUESTION THIS ANSWERS, and why neither existing gate answers it.
TestQwen35GGUF_vsSafetensors reports one number at the TOP of the stack (min cosine 0.987835
over 80 teacher-forced steps, mean 0.998114) and calls it a loader bug. TestQwen35GGUF_weightDiff
reports that every transform-bearing tensor in layers 0-3 agrees to cos >= 0.99998, with a
UNIFORM relL2 ~0.0057 — the Q8_0-vs-bf16 dequant floor — and no tensor standing out. Those two
results are consistent with two very different stories:
```

## TestQwen35GGUF_locateDivergence.diag

Moved from `decoder/qwen35_gguf_locate_test.go` (the comment in `TestQwen35GGUF_locateDivergence`, above the env gate) on 2026-10-09.

```text
DIAGNOSTIC, NOT A GATE — and it must not sit in the release sweep's path (2026-08-19).
the parity sweep's realckpt cell runs `-run 'Qwen35|Real_gate'` under a 120m timeout, and this
test's name matches. On the v0.14.0 prep sweep the two B13 diagnostics together burned ~41
minutes of that budget and the run TIMED OUT inside this one — which pushed
TestQwen35GGUF_weightDiff, a required 40-second gate, off the end of the run entirely. A gate
that DID NOT RUN is a blocker by the sweep's own rule, so an instrument that asserts almost
nothing took a real gate down with it.

Gated by env rather than renamed on purpose: Go's -run matches substrings, so excluding it by
name would mean stripping "Qwen35" from a qwen3.5 test — worse discoverability to work around
a scheduling problem. This way the name stays, and the sweep reports an honest SKIP.
```

## TestQwen35GGUF_vsSafetensors.floors

Moved from `decoder/qwen35_gguf_oracle_test.go` (the comment in `TestQwen35GGUF_vsSafetensors`, above the floors) on 2026-10-09.

```text
THE FLOORS, AND WHY THEY ARE WHAT THEY ARE (reclassified 2026-08-18, decider Francis;
evidence in docs/queue-release.md under B13).

This gate previously asserted `min >= 0.998` and called any miss "loader bug (not Q8_0
quant)". It failed at min 0.987835 (step 63) with mean 0.998114 — i.e. it required EVERY
step to be at least as good as the average, which no spread can satisfy. Three measurements
say the residual is quant noise, not a defect:

  1. TestQwen35GGUF_weightDiff: every transform-bearing tensor bit-exact or at a UNIFORM
     relL2 ~0.0057 (the Q8_0-vs-bf16 floor), worst 0.999980 — and the ROUTER is BIT-IDENTICAL
     (maxAbs=0), so routing differences are not a mis-read router.
  2. TestQwen35GGUF_locateDivergence: divergence is present at layer 0 and decays smoothly
     and NON-MONOTONICALLY to the top with no step. A localized defect cannot recover; that
     curve recovers repeatedly.
  3. TestQwen35GGUF_routeFlipAtOutlier: the two containers pick different top-8 sets in 779
     of 3200 (step,layer) pairs. With a bit-identical router that is the ROUTER'S INPUT
     differing by quant noise at a decision boundary — top-8 of 128 near-tied scores flips
     easily — and each flip is a legitimate alternative, not a wrong choice. It is also why
     flip COUNT does not predict cosine (the flipped expert's WEIGHT is what matters) and
     why a min-over-80 statistic is the wrong thing to floor.

So the gate now floors the statistic that is stable (the mean) and keeps a min floor set
from measurement with headroom, for catastrophic single steps. Both are derived from two
independent reproductions of the same numbers (2026-08-12 and 2026-08-18), not chosen to
make red green: the mean bar sits ~0.001 under a measured 0.998114 and the min bar ~0.008
under a measured 0.987835. A real transform bug does not land in that gap — it craters
cosine, which is what the three probes above independently confirm is not happening.

MEAN RE-BASELINED 2026-09-03 (mechanism confirmed by commit bisection, not inferred). 6d4fc79
("quantize the projections that were f32 at every quant") stopped keeping the DeltaNet
in/out-proj and gated-softmax q/k/v/o projections f32-always: both loaders now independently
quantize them to int8 via quantizeWM. Previously those tensors stayed in continuous f32 on
both sides of this comparison, so the pre-existing Q8_0-vs-bf16 delta had nothing to round
against; now each side re-quantizes its own slightly-different f32 view, and a delta that is
small in f32 can land the two sides in different int8 buckets — new, systematic divergence on
exactly the tensors this gate compares. Bisected on nobara (same box, same untouched
checkpoints/golden since 2026-06-08): 33879dd (6d4fc79's parent) reproduces the ORIGINAL
0.998114/0.987835 to six decimals; 6d4fc79 itself reproduces 0.995803/0.985140, also to six
decimals, across four independent runs (three different commits plus the original overnight
sweep). The commit already re-baselined a sibling gate hit by the same mechanism
(TestQwen35Real_gate2FullModel, mean 0.99837->0.99644) but missed this one, which lives
outside the gate-ledger manifest as an unlisted blocker — it went red silently for two weeks
until an overnight parity sweep caught it. Min keeps its 2026-08-18 floor: 0.985140 still
clears 0.980 with real headroom, so only the statistic that actually broke moves.
```

## TestQwen35GGUF_routeFlipAtOutlier.question

Moved from `decoder/qwen35_gguf_routeflip_test.go` (the comment above `TestQwen35GGUF_routeFlipAtOutlier.question`) on 2026-10-09.

```text
DOES A ROUTER FLIP EXPLAIN THE DIP? — the mechanism experiment for B13's last standing red.

The state of the argument before this test. TestQwen35GGUF_vsSafetensors reports min cosine
0.987835 at step 63 against a mean of 0.998114 and calls it a loader bug. Two probes contradict
that label: weightDiff finds every transform-bearing tensor bit-exact or at a uniform
relL2 ~0.0057 Q8_0 floor, and locateDivergence finds a smooth, NON-MONOTONIC decay across all 40
layers with no step (a localized defect cannot recover, and that curve recovers repeatedly).
From those two, "a ~0.5% weight delta flips borderline top-k router choices at a few positions"
is an INFERENCE about the outlier steps. This test measures it instead.

THE PREDICTION, stated before the run so it can fail. Both containers are teacher-forced through
the same 80 steps; every moeMLP call's top-k selection is recorded on each side (moeSelTrace, the
existing seam). If routing flips are the mechanism:

 1. the steps with the LOWEST logit cosine carry the MOST flipped layers, and
 2. step 63 — the 0.987835 outlier — is at or near the top of the flip ranking, and
 3. the great majority of steps have ZERO flips (which is why the mean sits at 0.998).

If instead flips are spread evenly across steps, or step 63 has none, the near-tie story is
WRONG and the dip needs another explanation — which is a real finding, not a failed test. So
this asserts only what any story must satisfy (the two runs are comparable) and prints the
correlation for the recorded decision.
```

## TestQwen35GGUF_routeFlipAtOutlier.diag

Moved from `decoder/qwen35_gguf_routeflip_test.go` (the comment in `TestQwen35GGUF_routeFlipAtOutlier`, above the env gate) on 2026-10-09.

```text
DIAGNOSTIC, NOT A GATE — and it must not sit in the release sweep's path (2026-08-19).
the parity sweep's realckpt cell runs `-run 'Qwen35|Real_gate'` under a 120m timeout, and this
test's name matches. On the v0.14.0 prep sweep the two B13 diagnostics together burned ~41
minutes of that budget and the run TIMED OUT inside this one — which pushed
TestQwen35GGUF_weightDiff, a required 40-second gate, off the end of the run entirely. A gate
that DID NOT RUN is a blocker by the sweep's own rule, so an instrument that asserts almost
nothing took a real gate down with it.

Gated by env rather than renamed on purpose: Go's -run matches substrings, so excluding it by
name would mean stripping "Qwen35" from a qwen3.5 test — worse discoverability to work around
a scheduling problem. This way the name stays, and the sweep reports an honest SKIP.
```

## TestQwen35GGUF_weightDiff.header

Moved from `decoder/qwen35_gguf_weightdiff_test.go` (the comment at the top of the file) on 2026-10-09.

```text
Cheap loader-vs-quant disambiguator (no full-model forward, no 60-min oracle).
The bf16-golden gate (qwen35_gguf_gate_test.go) failed by a hair (argmax 68/80,
cosine 0.99445 vs the 74 / 0.99466 bars), with the divergences being rank-2
near-ties. That's the signature of quant noise at the threshold, NOT a cratered
loader — but the bf16 gate can't prove it. This does: it loads only the first 4
layers of BOTH containers at f32 (the existing loadQwen35Slice for safetensors,
a sibling for the GGUF) and diffs every TRANSFORM-bearing tensor directly —
the V-head un-tile, the q‖gate fused q_proj, the −exp(A_log) bake, the norm(+1)
un-bake. The safetensors loader is Gate-1 bit-exact vs HF, so it is the
reference. A correct GGUF loader ⇒ each tensor matches to the Q8_0-vs-bf16
dequant delta (cosine ≳ 0.9995); a transform bug ⇒ that one tensor's cosine
craters, pinpointing it. Slice 0–3 spans both layer kinds (DeltaNet 0,1,2 +
softmax 3).
```

## TestQwen35GGUF_weightDiff.router

Moved from `decoder/qwen35_gguf_weightdiff_test.go` (the comment in `TestQwen35GGUF_weightDiff`, above the router comparison) on 2026-10-09.

```text
THE ROUTER, added 2026-08-18 and the reason is worth keeping. This probe was written to
disambiguate loader-vs-quant for the oracle's cosine gap and it checked every
transform-bearing tensor EXCEPT the one that decides which experts run. Then the
mechanism experiment (qwen35_gguf_routeflip_test.go) measured routing directly and found
the two containers choosing DIFFERENT top-8 sets in 779 of 3200 (step,layer) pairs — 79
of 80 steps affected. Pervasive routing divergence with an undiffed router is exactly
where a real loader defect could still hide, so the blind spot gets closed here.

Both loaders keep the router at f32 on purpose ("the router is logit-critical", no
quant), so this is a straight comparison: agreement at the same ~0.0057 Q8_0 floor as
every other tensor means the flips are quant noise on near-tied experts; anything worse
means the router itself is mis-read and the flips are a defect.
```

## wmDense

Moved from `decoder/qwen35_gguf_weightdiff_test.go` (the comment above `wmDense`) on 2026-10-09.

```text
wmF32 exposes a WeightMat's values as f32 for comparison. The router is loaded unquantized by
both paths, so this is a read rather than a dequant; it fails loudly if that ever changes,
because comparing a quantized router against an f32 one would manufacture a difference this
probe would report as a loader bug.
wmDense reconstructs w as dense f32 whatever precision it is stored in.

It used to be an f32-ONLY accessor whose failure message hardcoded "router". That held
until `6d4fc79` made the qwen35 projections honour Options.Quant, after which this gate
died on a PRECONDITION instead of a measurement — and blamed the wrong tensor while doing
it, because t.Helper() reports the caller's line but the message named a tensor the helper
knew nothing about. The tensor name is now a parameter for that reason.

Dequantizing is sound for this probe: both sides run through the SAME quantizer from the
same source values, so a real transform bug (wrong un-tile, q‖gate, un-bake) still craters
cosine. Only the noise floor moves, and it moves for both columns equally.
```

## TestQwen38GGUF_load.fitguard

Moved from `decoder/qwen3_5_gguf_test.go` (the comment in the Qwen3.8 GGUF gate, above the Load call) on 2026-10-09.

```text
The fit guard is bypassed for this model only, as in TestQwen35GGUF_gate and TestLagunaGGUF_gate. This gate
checks the LOADER's parity, not memory planning, and the guard prices the 15.3 GB mapped checkpoint as resident
and the KV cache at the model's full context (66.6 GB needed against a 29.5 GB budget on nobara-pc, 2026-10-01).
Whether that refuses depends on what the gates before it left in the process: the same box passed this gate in
sweep run 2 and was refused in the scoped re-run that afternoon. requireHeavyModel keeps it to the box that holds it.
```

## TestQwen38GGUF_weightDiff.floor

Moved from `decoder/qwen3_5_gguf_weightdiff_test.go` (the comment at the top of the file) on 2026-10-09.

```text
THE FLOOR IS PER TENSOR, AND THAT IS THE WHOLE POINT — it was not, and the gate was wrong
for it. A single 0.999 bar was inherited from the MoE sibling, whose asset is a uniform
Q8_0 file. This one's asset is unsloth's UD-Q4_K_M, a DYNAMIC quant carrying nine ggml
types chosen per tensor by sensitivity, and under one whole-file bar the gate stopped being
a statement about the loader: it became a statement about whichever tensor the quantizer
spent the fewest bits on. It duly failed on its first-ever execution (2026-09-06) at
k_proj 0.997047 / in_proj_z 0.996974, printing "loader transform bug" — and those are
exactly, and only, the Q4_K tensors (blk.3.attn_k, blk.{1,2}.attn_gate), sitting where the
first-principles Q4_K dequant estimate of ~0.9967 says they should, while their Q5_K and
Q6_K siblings in the same layer cleared the bar. Nothing was wrong with the loader. See
ggufQuantCosFloor (gguf_tensorquant_test.go) for the budgets and the arithmetic.
```

## TestQwen38Real_oracle

Moved from `decoder/qwen3_5_realckpt_test.go` (the comment above `TestQwen38Real_oracle`) on 2026-10-09.

```text
TestQwen38Real_oracle is the T3 numeric row: the released bf16 weights matched against an
HF bf16 forward of the SAME weights, pinned offline via scripts/pin_sequential_oracle.py
(accelerate disk offload, the same technique pin_qwen3next_real.py proved on an 80B model).
Until this gate, TestQwen38Real_gate above was coherence-only — this doc's own manifest text
said plainly that no bf16 reference forward had ever been run. The released checkpoint is a
vision-language wrapper (Qwen3_5ForConditionalGeneration) even though only the text path is
used; the pin script loads it via AutoModelForImageTextToText with pixel_values=None, the
same shape mistral3's own real-checkpoint gate needed.

int8, NOT int4 (changed 2026-09-18) — see docs/measurements/int4-neartie-laguna-qwen38-2026-09-18.md.
This comment used to say int8 "does not fit alongside f32 activations in 62 GB of RAM" — WRONG,
and never actually measured. Measured 2026-09-18: int8 peaks at ~27GB RSS here, ~90s. The int4
gate this defended had a real, reproducible divergence at continuation[2] (int4: cosine
0.993235, "Paris" repeating instead of a newline; confirmed via a floating-point-rounding-noise
control as a genuine near-tie in int4's coarser grid, sharing the same signature Laguna showed
despite the two families sharing no mixer — both resolved by the same fix). At int8: cosine
0.999886, all 8 continuation tokens exact.
```

## TestQwen3MoeReal_oracle.quant

Moved from `decoder/qwen3moe_real_test.go` (the comment in `TestQwen3MoeReal_oracle`, above the oracle call) on 2026-10-09.

```text
int8 WEIGHTS, f32 ACTIVATIONS as the starting quant, not int8int8: this family routes 8 of
128 experts (6.25%), the same order of sparsity as nemotron_h's MoE variant, which measured
a real router-flip cliff at int8 activations (cosine 0.978086 int8int8 vs 0.997668 int8) —
docs/completed/queue-correctness.md G4. Starting from the safer quant and measuring int8int8
separately (if this passes) avoids re-deriving that same finding the hard way.
```

## qwen3next_oracle_test.header

Moved from `decoder/qwen3next_oracle_test.go` (the comment at the top of the file) on 2026-10-09.

```text
Real-checkpoint T3 for Qwen3-Next (qwen3_next, 80B-A3B hybrid) — the row the family has been
missing since 2026-08-17, when it was blocked as "no full reference forward of a 163GB bf16
model fits 62GB".

THAT BLOCKER WAS ABOUT CO-RESIDENCY, AND CO-RESIDENCY IS NOT REQUIRED. The reference and
goinfer never have to be alive at the same instant: scripts/pin_qwen3next_real.py writes the
bf16 logits to a JSON golden through accelerate's disk offload, and this test reads that file
back in a separate process. Pinning offline is how every tiny golden in this repo is already
made; the earlier slice route generalised the SLICE when what needed generalising was the
PINNING.

WHY THIS IS NOT COVERED BY qwen3_5_moe's T3, though the two share forward_qwen35.go. The
shared-path proxy in docs/parity-coverage-policy.md needs the same forward file(s) AND the same
deps_hash. qwen3_next has decoder/qwen3next.go of its own and a distinct hash, and that file is
exactly where its three real deltas live — computed layer_types from full_attention_interval,
flat partial_rotary_factor, and the fused in_proj_qkvz/in_proj_ba split. A proxy row would have
asserted that qwen3_5_moe's oracle covered code qwen3_5_moe never executes.

QUANT IS int4 BY CAPACITY, NOT BY CHOICE, and the number this produces has to be read knowing
that. 80B at int8 is ~80 GB against 62 GB of RAM; int4 is ~40 GB and fits. There is no int8 run
to fall back to on this box, so a low cosine here cannot be re-run at higher precision to
separate quant noise from a defect — the per-layer SHAPE has to do that instead (smooth and
non-monotonic is noise; a cliff is a defect, and a defect cannot recover). The decision rule
was pre-registered in docs/queue-correctness.md G5 before this ran.

The sparsity is the reason to expect trouble: 10 of 512 experts is 1.95% active, sparsest in the
table by a factor of two. nemotron3nano at 6/128 (4.7%) measured 0.978 with int8 ACTIVATIONS
against 0.9977 with f32 ones, and its recorded cause is the expert-flip cliff.
```

## TestR9_w4a8BatchAt7B

Moved from `decoder/r9_decode_attribution_test.go` (the comment above `TestR9_w4a8BatchAt7B`) on 2026-10-09.

```text
TestR9_w4a8BatchAt7B is a follow-up to R-06 (docs/tasks/task-recompute-audit.md), which measured
GOINFER_W4A8_BATCH's fused q/k/v and gate/up matmuls on the 1.5B only (1.071x arm64/Metal,
1.066x amd64/CPU -- both ambiguous, parked). R9 step 1's own finding (MLP's share of the token
grows with model size, largest at 7B) raises the natural follow-up: does R-06 behave differently
at 7B? w4a8BatchEnabled (decoder/weightmat.go) is a package-level var read once from
GOINFER_W4A8_BATCH at process start, so this test cannot toggle it mid-process -- it reports
this run's own mean/stdev over repeated decode windows (fresh KV cache each, same loaded model,
no reload between repeats) and is meant to be run TWICE, once per env setting, the results
compared by hand (or by a small script) rather than paired within one process. That is a real
limitation next to R-06's own interleaved design -- disclosed, not hidden -- see the record.
```

## r9_s05_fold_ab_test.header

Moved from `decoder/r9_s05_fold_ab_test.go` (the comment at the top of the file) on 2026-10-09.

```text
arm64 only: the S-05 fold is an arm64 SDOT kernel, and aikit v1.47.0 declares its A/B toggle
(linalg.SetW4A8RowFold / W4A8RowFold) in an arm64-tagged file, so this file does not compile
on linux/amd64 without the tag (CI run 35763888743 found that the hard way).
```

## TestR9_s05FoldAB

Moved from `decoder/r9_s05_fold_ab_test.go` (the comment above `TestR9_s05FoldAB`) on 2026-10-09.

```text
TestR9_s05FoldAB is the end-to-end arm of aikit S-05 (docs/task-simd-audit.md: the
-8 centering folded into the SDOT accumulator of the M=1 W4A8 decode kernel,
bit-identical). The kernel's single-core win is measured in aikit's own harness; this
is the separate token-level number the decision rule asks for — one loaded 1.5B, depth
128, 24 greedy tokens, the fold flipped IN-PROCESS via linalg.SetW4A8RowFold, ABBA
pairs, paired ratio with a win count (the same shape as TestR9_cpuTuningAB). The
per-component split names where any change lands: the matmul terms (q/k/v, o,
gate+up, down, LM head) are the only ones the kernel touches.

Pre-registered band (docs/measurements/s05-centering-fold-2026-09-22.md): expected
~0 — step 0's split puts the 1.5B's MLP matmuls at 54% of the read ceiling and ~40% of
the kernel's hot rate per worker, i.e. fan-out-bound (S-02), where a faster kernel is
mostly hidden. ≥3% paired on 3/3 pairs = reaches the token; 1.5–3% = ambiguous, parked;
<1.5% = does not reach the token, as predicted. The ship decision is the single-core
harness either way; this number is reported beside it, not gated on.
```

## oracleCosFloor

Moved from `decoder/real_oracle_test.go` (the comment above `oracleCosFloor`) on 2026-10-09.

```text
oracleCosFloor is the last-logit cosine bar for a T3 real-model oracle, BY PRECISION.

There was one bar for a long time, 0.99, and its comment said what it was: "int8 W8A8 vs bf16 —
same bar as the deepseek real gates". That was fine while every real oracle was int8. It stopped
being fine when qwen3_next arrived at int4 — not by choice but by capacity, since 80B at int8 is
~80 GB against 62 GB of RAM — and was measured against a bar calibrated on a population it is
not in. int4 is a coarser grid than int8; holding both to one number is the same category of
error as holding gpt2's int4 goldens to the tiny fixtures' absolute gate.

An UNREGISTERED precision is a hard failure rather than a default. Silently inheriting a bar
belonging to some other precision is exactly how the int4 gate ended up judged by an int8 number,
and a default would let the next precision repeat it without anyone deciding anything.
```

## recurrent_census_test.header

Moved from `decoder/recurrent_census_test.go` (the comment at the top of the file) on 2026-10-09.

```text
The recurrent-state census (audit 2026-09-10 C-03 / G-05): helpers that ASK THE STRUCT which
recurrent kinds a KVCache holds, and generic fill / check-zero over them, so no test here carries
a list of kinds or families to forget. hasRecurrentState's own comment predicted the failure —
"the fourth kind will be added here, once, or it will be missed at four sites again" — and the
fourth kind (KDA) was missed. With these, a fifth is covered the day its field is added.
```

## TestRequantBar_DoubleQuantCost

Moved from `decoder/requant_test.go` (the comment above `TestRequantBar_DoubleQuantCost`) on 2026-10-09.

```text
TestRequantBar_DoubleQuantCost measures what Metal's DOUBLE quantization costs, per model.

Metal's BuildResident cannot consume the decoder's int4 — it requires an int8 load and
re-quantizes to its own W4A8, so a weight travels f32 → int8 (per-row, scale=max|row|/127)
→ int4 (group-32, scale=maxabs/7). CUDA and WebGPU upload the decoder's int4 byte-identically
and pay the int4 step ONCE. That asymmetry has always been known; what was never measured is
whether it MATTERS, and the assumption was that it is a rounding detail.

It should not be a detail for a model with high per-row dynamic range, and the mechanism is
specific: the int8 step's scale is set by the row's LARGEST element. A group whose own maxabs
is far below the row max therefore lands on only a handful of int8 levels BEFORE int4 ever
sees it — the group scale then spreads 15 int4 levels over a signal already crushed to 3 or 4.
Where a row is flat (row max ≈ group max) the int8 grid is finer than the int4 one everywhere
and the first step is nearly free. So the cost is a property of the WEIGHTS, not the kernel,
and it predicts exactly the split measured on Metal: gemma3 loses 0.104 of logit cosine
against its own CPU-int4 twin while the qwen control loses nothing.
```

## fakeResidentAdapter

Moved from `decoder/resident_adapter_seam_test.go` (the comment above `fakeResidentAdapter`) on 2026-10-09.

```text
It also implements Prefiller (PrefillLast) — audit C-01/G-06 (09-10): the real CUDA/Metal
batched-prefill launch reads no adapter state at all, so it is not "the adapter's math done a
different way" but a SEPARATE, adapter-blind path. Giving the fake the same shape (increments
prefillCalls, otherwise identical to a per-token Forward loop) makes TestSeam_AdapterSession
BindsAndClearsResidentAdapter's ≥8-token prompt able to actually reach it if
residentPrefillSeed's hasAdapter guard were ever dropped — before this, the fake had no
Prefiller method at all, so the 3-token prompt below the batching floor and the missing
interface made the gate blind to C-01 by construction (G-06's own finding).
```

## TestWithResidency_releasesHostMemoryOnlyAfterASuccessfulBuild

Moved from `decoder/resident_release_test.go` (the comment above `TestWithResidency_releasesHostMemoryOnlyAfterASuccessfulBuild`) on 2026-10-09.

```text
TestWithResidency_releasesHostMemoryOnlyAfterASuccessfulBuild pins the load-time release added after a measured
leak-in-effect: with a real gpt-oss-20b on CUDA the process sat at ~39 GB RSS for ~5 minutes after the load
finished (Go's scavenger returning the host-side packing lazily) and dropped to ~22 GB within ~10 s once the
heap was released explicitly. The release must happen after a resident build succeeds — and must NOT happen for a
decline, which built nothing and would only pay for a pointless collection.
```

## resident_seam_test.header

Moved from `decoder/resident_seam_test.go` (the comment at the top of the file) on 2026-10-09.

```text
The seam gate.

WHY THIS EXISTS. `serve --backend cuda|metal|webgpu` was silently CPU-only on EVERY backend
for five weeks (0eefd77 -> 7557723). TWO independent bugs lived in the same gap, either of
which alone was fatal:

  - Options.Validate rejected the backend NAME, so `serve --backend cuda` failed at flag
    validation even where the module was built in (727f198).
  - Model.Generate gates the GPU on `useGPU := resident != nil && prefillFrom == 0 &&
    commit == nil`, and Session.Generate ALWAYS sets commit — so routing a request through
    the session cache silently disabled residency (7557723).

Neither was caught, because nothing ever asserted "is this actually running on the GPU?".
Every test called decoder.Forward directly; all 12 cmd/serve tests pin Backend: "cpu"; and
ResidentActive() was asserted only inside gpu/. The flagship feature did not work at all and
the suite was green. It was found by noticing a tok/s number, which is not a gate.

The tests below need NO GPU and NO downloaded model — they fake the resident backend and use
the committed tiny fixture — so this seam is gated in CI, on every push, rather than by a
human remembering to look at throughput.
```

## loadAccountingFixture

Moved from `decoder/residentneed_test.go` (the comment above `loadAccountingFixture`) on 2026-10-09.

```text
docs/tasks/task-memory-accounting-2026-09.md item 2: `fit`'s verdict (Plan) and Metal's resident guard
(ResidentNeedBytes, via metal/backend.go's residentNeedBytes) must be the same number by construction. Before,
Plan("metal") had no host-copy term and priced Metal's KV at f32, so on a directly loaded model it asked for
about half of what the guard would then demand (measured: 2.34 vs 4.33 GB on qwen2.5-coder-1.5b).
```

## TestRopeParameters_singleBaseArchs

Moved from `decoder/ropeparams_test.go` (the comment above `TestRopeParameters_singleBaseArchs`) on 2026-10-09.

```text
That is not a niche path — it is any Llama/Mistral/Qwen safetensors saved by a current
transformers. It was found by generating a tiny Mistral fixture with transformers 5.12 and
watching it fail to load, while the committed phi3-tiny (same transformers, same config
shape) loaded fine. Each arch has its own architecture func, which is exactly how one got
the fix and the others did not; the shared backfillFlatRope helper is the answer to that,
and this table is what stops the next single-base arch from re-opening it.
```

## BenchmarkDecodeSerialVsParallel

Moved from `decoder/s09_serial_bench_test.go` (the comment above `BenchmarkDecodeSerialVsParallel`) on 2026-10-09.

```text
BenchmarkDecodeSerialVsParallel — S-09.1, the corrected A/B.

THE 2026-08-11 MEASUREMENT COMPARED TWO PARALLEL ARMS. BenchmarkDecode sets only the PROCESS
GLOBAL (linalg.SetParallelThreshold), but since 2026-08-01 decode runs on a PER-WORKSPACE
threshold that newDecodeScratch installs on every scratch it builds (scratch.go: `ws :=
&linalg.Workspace{}; ws.SetThreshold(DefaultDecodeParallelThreshold)`). The global is
therefore overridden before the first token, so GOINFER_PAR_THRESHOLD=<huge> did not make the
"serial" arm serial — and "serial 54.77 vs parallel 54.34 tok/s" was two parallel runs
differing by 0.8%, which is inside this box's noise.

S-02's whole premise ("serial ties parallel, so fork/join is net-neutral") rests on that
number, so it has to be re-taken against a genuinely serial arm.

SERIAL IS PROVEN, NOT ASSUMED. Setting the threshold is the same kind of act that failed last
time, so the arm also counts goroutine spawns: parallelSpawnCols starts one goroutine per
shard per matmul, so a real serial arm shows a flat goroutine count while a parallel one
spikes. The counter is sampled rather than instrumented because parallelSpawnCols is
unexported in aikit — but a flat maximum across thousands of matmuls is unambiguous.
```

## sampler_chunked_test.header

Moved from `decoder/sampler_chunked_test.go` (the comment at the top of the file) on 2026-10-09.

```text
SAMPLER MICROBENCHMARKS IN THIS PACKAGE MAY NOT PRODUCE QUOTABLE FIGURES.

They are a tool for deciding WHERE TO LOOK. No number they emit belongs in a doc, a queue item, a
commit message, or a comparison between two commits. This is a standing prohibition, not advice,
and it was earned twice in a single investigation (G26, 2026-08-27):

  1. BenchmarkExpChunked's ~96 us was used to argue that "sampling is a low-single-digit
     percentage of per-token time, so no sampler change can move end-to-end by 5.9%." The real
     in-situ sampled tail is 703-950 us — 7-10x larger. The bound retired the correct hypothesis
     for two rounds.
  2. A whole-path Sampler.Sample benchmark reported HEAD 23% SLOWER at 152k vocab. Measured
     end-to-end on a 151936-vocab model, HEAD is 31% FASTER. Not a mis-scaled magnitude — an
     INVERTED SIGN, and it had already been filed as a finding before the end-to-end run.
```

## benchExpChunked

Moved from `decoder/sampler_chunked_test.go` (the comment above `benchExpChunked`) on 2026-10-09.

```text
G26. The temp-only draw (no truncation) goes through sampleChunked, NOT topFilterLogits — so the
P10 benchmarks in sampler_selection_test.go measure the top_p path and say nothing about this one.
phi3-mini regressed 5.9% at temperature 1.0 while gaining 2.7% at temp+top_p, which is the split
those two paths would produce. P10 is the only functional change to this file since the anchor.
```

## registered-before-code.clauses

Moved from the header comments of `decoder/pixtral_real_test.go`, `decoder/quant_consistency_real_test.go`, `decoder/qwen3asr_real_test.go`, `decoder/qwen3asr_test.go`, `decoder/qwen3asr_wer_real_test.go` and `decoder/qwen3vl_image_real_test.go` (the clause dropped from each) on 2026-10-09.

```text
pixtral_real_test.go:          (S10, Ministral 3 (Pixtral), registered 2026-10-09 before any code)
quant_consistency_real_test.go: G-31a2 ... "Gemma 4 31B on nobara ... G-31a", registered before this code
qwen3asr_real_test.go:         G-S14c1 ... and G-S14c3 ... (registered before this code)
qwen3asr_test.go:              G-S14c1 and G-S14c2 ... (registered before this code)
qwen3asr_wer_real_test.go:     G-S14c4's goinfer arms (..., registered before this code)
qwen3vl_image_real_test.go:    G-S10c ... (S10, registered before this ran)
```

## TestPrefillCoverageAudit.oldclaims

Moved from `decoder/prefill_coverage_test.go` (the doc comment above `TestPrefillCoverageAudit` and the comment in its body, above the reason check) on 2026-10-09. The doc comment named seven PrefillLast guards; the body asserts and models only not-resident and MoE, and the comment was reworded to say so. Recorded here because it is a coverage claim the body never made.

```text
TestPrefillCoverageAudit enumerates every validated family against the cuda PrefillLast guards
(decline on: not-resident, MoE, gemma4-moe, sandwich norms, qk-norm, K=V, non-uniform geometry).
It reports, per family, whether it GETS batched prefill or FALLS BACK, and which guard fires — so
"extend the guard" work can be scoped by which guard blocks the most families. Guards are read from
the resolved Architecture (the same flags cuda/backend.go sets the resident from); int4-weight and
over-cap are checkpoint/prompt-specific, not family-inherent, so they are noted, not tabulated.

qk-norm and sandwich norms are NO LONGER guards: batched prefill applies them via
qk_norm_batched / rmsnorm_f32_batched (bit-identical per token; validated on real Qwen3-1.7B
and Gemma-3-4B by cuda.TestPrefillLast_qwen3 / _gemma3). So qwen3 and gemma3 now BATCH; MoE
(glm4_moe/qwen2_moe/mixtral) and the not-resident classes still decline. (gemma3 batches only
when the gemma resident path is enabled — GOINFER_GEMMA4_RESIDENT; else it stays staged.)
(K=V is a Gemma-4-only property, nested in gemma4Params; Gemma-4 trips sandwich/gemma4-moe
first, so K=V never surfaces as the binding guard — omitted.)
```

## qwen3next_slice_test.header.dropped

Moved from `decoder/qwen3next_slice_test.go` (the file header, two clauses) on 2026-10-09.

```text
REAL-WEIGHT layer-slice oracle for Qwen3-Next — the T3 the macbook's Phase 0/1 left
open and tagged `linux`.
...
checkpoint-layout delta the macbook flagged — the FUSED DeltaNet input projections
```

## qwen3vl_real_test.header.dropped

Moved from `decoder/qwen3vl_real_test.go` (the file header) on 2026-10-09.

```text
(testdata/assets.json) — not present on this box as of 2026-09-08, so this gate SKIPS cleanly
rather than being silently absent from the registry.
```

## qwen3moe_test.tinygolden.dropped

Moved from `decoder/qwen3moe_test.go` and `decoder/qwen2moe_test.go` (the tiny-golden comment above `emitParityRow`) on 2026-10-09; both claims were stale once the real-checkpoint gates (`TestQwen3MoeReal_oracle`, `TestQwen2MoeReal_oracle`) existed.

```text
The real Qwen3-30B-A3B (bf16 ~61GB) is a Linux-box T3, not yet run here.
(real Qwen1.5-MoE-A2.7B not on this box)
```

## qwen2_real_test.header.dropped

Moved from `decoder/qwen2_real_test.go` (the file header) on 2026-10-09; the model size was wrong for this file (the gate loads Qwen2.5-0.5B).

```text
1.7B fits an f32 forward in RAM
```

## perrow_phase0b.dropped

Moved from `decoder/perrow_phase0b_test.go` (the comment above `TestPerRowScalePhase0b`) on 2026-10-09.

```text
(weight-space, 1.24× rel-error for per-row symmse)
```
