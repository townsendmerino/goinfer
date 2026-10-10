# decoder: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `decoder`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## TestComputeLogprobs_reusesProvidedScratch

Moved from `decoder/sampler_extra_test.go` (the comment above the test) on 2026-10-10.

```text
TestComputeLogprobs_reusesProvidedScratch is P-07 (audit-2026-09-10): computeLogprobs called
softmaxStable directly, allocating a fresh full-vocab []float64 on every logprobs:true
request; SampleWithInfo now passes its own distBufN scratch through. Correctness (matches a
nil-scratch call exactly) and the reuse itself (the internal softmax write lands in the
caller's own buffer) are both asserted — a fix that reused the wrong buffer or silently
stopped reusing would each pass a test that checked only one side.
```

## TestSampler_minPAboveOne

Moved from `decoder/sampler_minp_test.go` (the comment above the test) on 2026-10-10.

```text
M-08: MinP > 1 with no TopK panicked inside the sampler.

The min-p threshold is maxL + T·ln(minP); for minP > 1 that sits ABOVE maxL, so the
candidate set comes back EMPTY and the "always keep the top token" clamps then slice ips[:1]
on an empty slice or read ips[len(ips)-1] at index −1. Measured before the fix:
`index out of range [-1]`, panicking in the Generate goroutine. min_p is not on the HTTP
surface, so this reached users through the library and `goinfer-chat --min-p`.

TWO INDEPENDENT DEFENCES were added, and they are tested independently — together they mask
each other, so a single end-to-end "does not panic" test would go green with either one
removed and prove neither.
```

## TestTopFilterLogits_MatchesReference

Moved from `decoder/sampler_selection_test.go` (the comment in the seed-sweep body) on 2026-10-10.

```text
The seed sweep is run in PARALLEL SHARDS. Every case still runs — the shards partition the
same seed range and assert the same things — but wall time divides by the core count.

Why it matters: CI runs `go test -race`, and this sweep is pure computation with no
goroutines and no shared state, so the race detector finds nothing here while costing ~10×.
At 400 seeds × 15 cfgs × 4 temps that pushed the whole decoder package past the 600 s default
timeout on CI's slower runner (locally 63 s un-raced) — main went red on a TIMEOUT, not a
failure. Sharding is the coverage-neutral fix; reducing the sweep would have traded away the
exactness gate that justifies the optimization, which is the wrong thing to trade.
Seed selection: every seed normally, an evenly-STRIDED subset under -race (the detector has
nothing to find in this pure-compute sweep — see sampler_sweep_race_test.go). Striding rather
than truncating keeps the selection spread across the whole range, so both logit shapes
(tie-heavy / tie-free, which alternate on seed parity) stay represented. All 15 configs and
all 4 temperatures run for every selected seed either way.

C8 (docs/completed/task-ci-speed-2026-09.md): the stride below WAS this loop's whole point
and it went missing in 324f63c9 ("go fix: modernize"), which rewrote
`for s := 0; s < seeds; s += sweepSeedStride` into `for s := range seeds` — every seed ran
under -race for months while the mode string said "strided subset", and this test was 217 s
of every CI run. The check right after the loop is the guard that shape needs: a build that
declares a stride must be seen to apply it.
```

## TestSamplingThroughputGate

Moved from `decoder/sampler_selection_test.go` (the comment above the test) on 2026-10-10.

```text
TestSamplingThroughputGate asserts the top-p/top-k cliff is gone: sampling at
temperature+top_p must run within a bounded factor of the TEMPERATURE-ONLY baseline
(amendment 4 — gated against temp-only, not greedy). A full-vocab sort regression shows
up as ~7× (the reported 100→15 tok/s); the gate factor sits below that and above the
real post-fix ratio.
```

## TestSamplingThroughputGate.race

Moved from `decoder/sampler_selection_test.go` (the comment in the test body, at the raceEnabled skip) on 2026-10-10.

```text
A WALL-CLOCK RATIO measured under the race detector measures the DETECTOR, not the sampler.
Observed on CI: this gate passed on linux/-race and failed on darwin/-race at 3.86× and 3.99×
against a 3.0× bound, on two commits that touched no sampler code — the instrumentation does
not scale the two arms equally, and a shared runner adds noise on top. Left on, it is a
permanently red gate that says "full-vocab selection has regressed" when nothing has.

It is not skipped into oblivion: ci.yml runs this test WITHOUT -race on every push, which is
where a real regression would show. Timing gates belong in an un-instrumented run.
```

## TestSamplingThroughputGate.bound

Moved from `decoder/sampler_selection_test.go` (the comment in the test body, above the factor bound) on 2026-10-10.

```text
RE-BOUNDED after P2b (2026-08-09). This gate compares temp+top_p against temp-only, and P2b made
the DENOMINATOR ~4.7× faster (parallel chunked normalization), so the ratio rose from 0.86× to
3.88× at 262k WITHOUT the filtered path regressing — top_p itself went 6.98 ms → 6.72 ms over the
same change. A gate whose baseline moves is measuring two things at once; the bound is raised to
keep it catching real filtered-path regressions rather than firing on an improvement elsewhere.

Note the synthetic logits here overstate top_p's cost: randLogits is near-Gaussian, so a 0.95
nucleus over 262k entries retains an enormous candidate set, where REAL peaked decode logits
retain a handful. The e2e A/B on real models shows the filtered path ~2× FASTER after P2b
(gemma3-1b 56.3 → 117.0 tok/s), the opposite of what this ratio suggests in isolation.

RE-ANCHORED after R7b (2026-09-20), the bar NOT moved. Production temperature-only sampling became
Gumbel-max, ~1.3-1.8x cheaper on the CPU (1.49 -> 1.16 ms at 262k), which moved this ratio from 4.14x to
5.26x while top_p itself was unchanged (6.17 -> 6.09 ms) — the same "baseline moved" hazard as above. The
denominator is now the LEGACY chunked inverse-CDF draw, kept unchanged in sampler_chunked_ref_test.go, so
the ratio means exactly what it meant when the 5.0x bar was set.
```

## benchSample

Moved from `decoder/sampler_selection_test.go` (the comment above the function) on 2026-10-10.

```text
benchSample times one sampling configuration, taking the BEST of three runs rather than one.

WHY BEST-OF-N AND NOT A SINGLE MEAN. testing.Benchmark's NsPerOp is a mean over b.N, and a mean
tracks scheduling jitter upward — it has a floor but no ceiling. Measured 2026-08-31 on x86,
three consecutive isolated runs of the temp-only arm at V=262144: 1.356 / 1.694 / 1.994 ms, a
47% spread, while the temp+top_p arm over the same runs moved only 6% (6.54 / 6.72 / 6.93 ms).
The RATIO those produce swings 3.48-4.83 — and none of that motion is the sampler.

The minimum is the right estimator here because the quantity is floored: the fastest observed
run is the one least contaminated by whatever else the machine was doing. It cuts the x86
spread from 1.35x to 0.39x (3.48-4.83 becomes 3.85-4.24) and leaves arm64 where it was
(4.81-4.85).

IT DOES NOT MAKE THE RATIO MACHINE-INDEPENDENT, and an earlier draft of this comment claimed
it would — predicted, not measured, and the measurement refuted it. x86 settles near 4.0 and
arm64 near 4.83: a real 0.7x gap between machines, on the same code. So the bound below is
effectively set by whichever machine runs hottest on this ratio, which is arm64, at ~3% margin.
That is a property of the RATIO design (see the note on the bound), not something a better
estimator fixes.
```

## BenchmarkFilterRef32k

Moved from `decoder/sampler_selection_test.go` (the comment above the 32k benchmarks) on 2026-10-10.

```text
32k is phi3-mini's vocab, and it was NOT in this benchmark's population when P10 (4da116d)
reused the full-vocab scratch buffer: only 152k and 262k were measured. The §B5 re-anchor then
found phi3-mini DOWN 5.8% at temperature 1.0 while the 152k and 262k models gained 8-9% on the
same configuration (G26). A change validated on two large vocabs and shipped for all of them
needs the small one measured too.
```

## TestSweepCoverage_fullSweepRunsSomewhere

Moved from `decoder/sampler_selection_test.go` (the comment above the test) on 2026-10-10.

```text
TestSweepCoverage_fullSweepRunsSomewhere is the gate on the gate.

The exactness sweep is strided under -race, and BOTH root CI jobs run -race — so the full
24,018-case sweep runs only because ci.yml carries an explicit non-race step for it. That is a
coupling between a build tag and a YAML file, invisible from either side: delete the step and the
gate silently shrinks to a subset in every job, with nothing red. (An earlier version of this
change asserted the full sweep "still runs in the non-race job" when no such job existed.)

This reads ci.yml and fails if the step is gone. It is deliberately a string check rather than a
YAML parse: what matters is that SOME step runs this test without -race, and the cheapest honest
way to assert that is to look for it.
```

## sweepSeedStride.race

Moved from `decoder/sampler_sweep_race_test.go` (the comment above the const block) on 2026-10-10.

```text
Under -race the exactness sweep runs a STRIDED SUBSET of seeds rather than all of them.

WHY. TestTopFilterLogits_MatchesReference is pure computation: no goroutines, no shared state,
no channels. The race detector therefore has nothing to find in it, while costing ~34× (measured
on this box: the sweep is 12.3 s sharded without -race; the decoder package went from a 600 s
timeout panic to 414 s once sharded, and the sweep is the dominant term). On a 4-core CI runner —
where the parallel sharding recovers much less than on a many-core box — the full sweep is
plausibly 8–9 minutes of EVERY CI run, for a second execution of cases the non-race job already
proved. Paying that indefinitely is the waste; the tax is not buying coverage.

WHAT IS PRESERVED. This strides the SEED axis only. Every one of the 15 parameter configs and all
4 temperatures still run, for every seed selected — so the parameter space is spanned, not
truncated to a prefix. Striding (rather than taking the first N) keeps the selected seeds spread
across the whole range, so the tie-heavy and tie-free logit shapes (which alternate on seed
parity) both stay represented.

COVERAGE NOTE (corrected). An earlier version of this comment claimed the full sweep "still runs
in the non-race job" — it did not. BOTH root CI jobs run `go test -race`, so gating on race mode
alone would have meant the full 24,018-case sweep ran in NO CI job at all: a real weakening,
asserted to be none. ci.yml now carries an explicit non-race step that runs the full sweep (and
the throughput gate) on every push. If that step is ever removed, this gate silently shrinks
again — TestSweepCoverage_fullSweepRunsSomewhere fails if the two drift apart.
The stride is ODD on purpose. The sweep picks its logit shape by seed parity (`s%2 == 0` →
tie-heavy, else tie-free), so an EVEN stride would select only even seeds and silently drop the
tie-free shape entirely — a subset that no longer spans the space it claims to. 7 alternates
parity on every step, keeping both shapes represented.
```

## TestSampleFromTopK_matchesFullPath

Moved from `decoder/sampler_topk_test.go` (the comment in the draw-count branch) on 2026-10-10.

```text
C8 (docs/completed/task-ci-speed-2026-09.md): under -race the DRAW axis is strided by
the same constant the exactness sweep strides its seeds with — every (vocab, shape,
config) cell still runs, with 1/7 of its draws — because this is pure computation the
detector finds nothing in, and it was 369 s of every CI run (44 s un-raced). The full
draw count runs in ci.yml's non-race sampler-gates step, which
TestSweepCoverage_fullSweepRunsSomewhere requires to name this test.
```

## TestEnsureCPUSelfTest_recordsAPass

Moved from `decoder/selftest_test.go` (the comment at the Elapsed check) on 2026-10-10.

```text
Windows' monotonic clock ticks every ~0.5-15 ms and the CPU self-test takes about 1 ms (less with the DotProd kernels
aikit v1.56.0 enabled there), so a pass there can honestly record 0: root-windows-arm64 read "elapsed 0s" at ee07ced7.
```

## serialize_census_test.header

Moved from `decoder/serialize_census_test.go` (the comment above the package clause's imports) on 2026-10-10.

```text
THE SILENT-DROP CENSUS: does a .giw round-trip preserve every per-layer field the loader
populated? Generic over the struct, not over a remembered list.

WHY THIS EXISTS. `canSerialize` is a hand-maintained blocklist of families the writer cannot
express, and on 2026-08-19 it was found to have DRIFTED for two of them:

  gpt_oss  accepted -> AttnSinks 8 -> 0, and the bundle LOADED CLEAN (silent wrong answers)
  laguna   accepted -> reader rejected it at load ("layer 1 QProj: 128 rows, arch expects 64")

Neither was exotic: gpt-oss needed ONE []float32 field written. The defect was not difficulty, it
was that adding a field to LayerWeights and forgetting to add it to serialize.go produces no
error anywhere — and the gate meant to catch that (TestCanSerialize_refusesUnrepresentable) asks
only "is this family on the list?", which is the same memory that failed in the first place.

So this test asks the struct instead: populate a model, serialize, load, and compare EVERY field
of LayerWeights by reflection. A field that was non-empty and comes back empty is a silent drop,
whatever family introduced it and whoever forgot it. Refused families are skipped — refusing is a
correct answer; silently dropping is not.

It runs on committed tiny fixtures only: no assets, no GPU, no network.
```

## censusList.lfm2

Moved from `decoder/serialize_census_test.go` (the comment above the lfm2-tiny line) on 2026-10-10.

```text
Added 2026-09-02 by TestSerializeCensus_everyFixtureIsListedOrExcluded, which found
eleven committed fixtures this list had never mentioned. lfm2-tiny is the one that
mattered: serialize.go dropped its entire conv mixer (audit-2026-09-02 C-03).
```

## censusList.llama

Moved from `decoder/serialize_census_test.go` (the comment above llama-tiny) on 2026-10-10.

```text
llama-tiny (2026-09-02): the plain `llama` arch had NO fixture anywhere in the tree, only
goldens — so the census round-tripped 21 families without ever touching the most common
architecture in the ecosystem, and a required parity gate's family at that. The only
llama checkpoints here are the box's gitignored llama3.2-1b / tinyllama-awq /
tinyllama-gptq, none of them tiny. scripts/pin_llama_tiny.py, 720 KB, tracked.
```

## censusList.glmocr

Moved from `decoder/serialize_census_test.go` (the comment above glm-ocr-tiny) on 2026-10-10.

```text
glm-ocr-tiny (2026-10-01; tracked since glm_ocr O1, ff79b92a, but never listed — found red by the targeted
decoder run for the pairwise-rope change): the pairwise rotation flag (ropeInterleave) and the m-RoPE
sections are the per-Architecture fields a GIW round trip could drop.
```

## censusList.mistralwindow

Moved from `decoder/serialize_census_test.go` (the comment above mistral-tiny-window) on 2026-10-10.

```text
mistral-tiny-window (2026-10-02): the exclusion said "config-only, Load cannot open it", which was a symptom, not a fact: the weights existed
locally and were gitignored (*.safetensors), so on a fresh checkout Load had nothing to open. They are committed now. SlidingWindow is the
per-Architecture field a GIW round trip could drop.
```

## censusExcluded

Moved from `decoder/serialize_census_test.go` (the comment above the map) on 2026-10-10.

```text
censusExcluded names a committed model fixture the census deliberately does NOT round-trip, with
the reason. Every fixture must be in censusList or here — TestSerializeCensus_everyFixtureIsListedOrExcluded.

A LIST THAT NOBODY CHECKS IS THE DEFECT THIS TEST WAS BUILT TO REPLACE, AND IT CAME BACK. The
census's own preamble says the guard before it failed because it "asks only 'is this family on
the list?'" — and the replacement asked a list too, 21 hand-maintained paths. ../testdata/lfm2-tiny
has been committed since ed112b0 and was on none of them, so `grep shortConv decoder/serialize.go`
returned zero matches for an entire shipped family while this census reported green over 21
others (audit-2026-09-02 C-03). Discovery plus an explicit exclusion is what makes the list a
decision instead of a memory.

The reasons here are COST, measured, not judgement: the census loads, serializes and greedy-decodes
every fixture, and 29 of them together take 0.27s.
```

## censusExcluded.localdrops

Moved from `decoder/serialize_census_test.go` (the comment inside the map, above the gptoss120-slice entry) on 2026-10-10.

```text
The Linux box's nine local drops, 549 MB to 17 GB and ~44 GB together. Recorded as decisions
rather than left to be re-reported every run. TWO reasons apply to all of them and both are
measured, not judgement:

 1. COST. This census loads, serializes and greedy-decodes every fixture, and its whole design
    point is that 30 of them together take 0.19 s — cheap enough to run in every CI. The
    smallest of these is 549 MB; mellum-mellum2-slice at 4.0 GB was still running after 90 s.
 2. NO NEW FIELDS. Eight of the nine are an architecture the census already round-trips, and
    the field set this test compares is a function of the ARCHITECTURE. A second gemma4 or a
    third laguna adds bytes, not coverage.

The ninth is different and is recorded in docs/QUEUE.md rather than hidden here: llama3.2-1b,
tinyllama-awq and tinyllama-gptq are the ONLY fixtures anywhere for the plain `llama` arch,
and there is no tiny one. See llama-tiny below, which is the answer to that.
```

## TestSerializeCensus_everyFixtureIsListedOrExcluded

Moved from `decoder/serialize_census_test.go` (the comment above the test) on 2026-10-10.

```text
EVERY TRACKED FIXTURE IS LISTED OR EXPLICITLY EXCLUDED.

The census above is only as complete as its fixture list, and that list is hand-maintained — the
exact shape whose failure it was written to replace. This walks both testdata roots, treats an
entry with a config.json (or a .gguf/.giw file) as a model fixture, and requires each to be
censused or excluded with a written reason. It is a STAT plus one `git ls-files`, not a load, so
it costs nothing and cannot be the reason someone trims the list.

TRACKED, NOT "PRESENT", AND CERTAINLY NOT "COMMITTED" — two wrong scopes, each caught by a
machine that had a different set of files:

  - "committed" was the first, and it was never checked. Most fixture checkpoints are GITIGNORED
    local drops; 12 of the 35 this machine holds are tracked, and CI's checkout has 14 in total.
    Four exclusions naming large untracked fixtures read as STALE there and turned CI red.
  - "present in this checkout" was the second. The Linux box carries local drops nobody else has
    — `gptoss120-slice`, `laguna-xs2-slice`, and three SCRATCH dot-directories — so a rule that
    demands every present fixture be listed fails on any machine with extras, which makes the
    gate an override habit rather than a gate.

Tracked is the scope that is the same everywhere, and it is a real scope, not a retreat: a new
family's tiny fixture gets committed, which is exactly how ../testdata/lfm2-tiny (the fixture
whose absence from the list let serialize.go drop an entire family's conv mixer) got there. It
is tracked, so the defect this gate exists for is inside the blocking set.

An untracked fixture that is present and unlisted is REPORTED, never asserted on: it may be
worth adding, and it may be somebody's scratch copy, and this test cannot tell.
```

## TestLFM2_serializeRoundTripsTheConvMixer

Moved from `decoder/serialize_lfm2_test.go` (the comment above the test) on 2026-10-10.

```text
C-03: A CRC-VALID BUNDLE THAT LOADS CLEAN AND NIL-DEREFS AT THE FIRST FORWARD.

`grep shortConv decoder/serialize.go` returned ZERO matches. cmd/prequant loaded an LFM2
checkpoint, wrote every field except the conv mixer, appended a valid CRC, and selfCheck passed
because selfCheck only Loads. Serving the bundle, the first token reached conv layer 0 with
lw.shortConv == nil and panicked in the decode goroutine. That is the R3 shape: the artifact is
well-formed by every check that runs on it, and wrong.

Two halves, and BOTH are needed. The v8 tail makes a correctly-written bundle round-trip. The
validateShapes presence check makes an INCORRECTLY-written one — a pre-v8 bundle, or a future
writer that forgets again — fail at load instead of at the first forward, where the panic is in
a goroutine no handler recovers.
```

## TestSerializeWeights_roundTrip

Moved from `decoder/serialize_test.go` (the comment at the aliasing check) on 2026-10-10.

```text
Aliasing: the deserialized q8 and its scales both point INTO blob (zero-copy). Scales were a copy until
.giw weights v12 (cdae727d, 2026-09-24) aligned them so the reader aliases them; this heavy-model test still
expected the copy and failed the first time it ran after (the v0.20.0 parity sweep).
```

## TestSerialize_unpopulatedLayersOmitsLabel

Moved from `decoder/serialize_test.go` (the comment above the test) on 2026-10-10.

```text
TestSerialize_unpopulatedLayersOmitsLabel is the dangerous half of B11, gated directly and
without a heavy asset. writeHeadGlobals used to decide whether to write the v5 quant-label
field by asking `wr.sink == nil` — "are we the buffered writer" — as a proxy for "do we have
the real weight data yet". Those disagree for the true GGUF streaming transcode, which writes
the header on a freshly make()'d, all-zero Layers slice BEFORE any layer has streamed in: at
that moment quantLabel()'s own "nothing matched" case returns "native" — a REAL quant mode, not
an empty string — so calling it unconditionally would bake a FALSE "native" label into every
genuinely-streamed bundle. hasPopulatedLayers() is the correct guard (data availability, not
writer identity); this asserts it actually withholds the label when the data is not there.

Walks the header by hand (no heavy checkpoint, no architecture resolution needed on write: only
json.Marshal(w.Cfg) must succeed, and a zero Config does) rather than round-tripping through
LoadSerializedWeights, which would additionally require a resolvable arch.
```

## TestSessionFastAttnDivergence

Moved from `decoder/session_fastattn_test.go` (the comment above the test) on 2026-10-10.

```text
TestSessionFastAttnDivergence — THE TEST decoder/forwardn.go:cpuFastAttention HAS BEEN CITING ALL ALONG.

It did not exist anywhere in the tree. The doc comment above cpuFastAttention says
"TestSessionFastAttnDivergence pins the new behaviour", which is the exact "a doc comment
claiming coverage is not coverage" class CLAUDE.md describes — one file away from where that
rule was written. Whoever audited the claim would find a plausible name, match it to the
sentence, and stop.

What it now pins is the honest version of the contract: with the fast kernel on (the
default), a prompt at or above fastAttnMinPrompt gives DIFFERENT logits from the exact
kernel — the split-invariance loss the comment describes and docs/server.md used to deny —
and below the floor the two are identical, because the floor turns the fast path off.
```

## TestLoadSession_allocationCeilingNotBodyRatio

Moved from `decoder/session_hostile_test.go` (the comment above the test) on 2026-10-10.

```text
TestLoadSession_allocationCeilingNotBodyRatio pins the guard that replaced the body-ratio bound
on 2026-09-05, from both sides — the valid blob the old bound REJECTED, and the hostile one the
new bound must still refuse.

The old check bounded pos by `len(body)/(numLayers·kvDim)`, asserting that "each of the pos
positions stores at least one byte across the numLayers·kvDim KV". kvsnapshot.go's own writer
contradicts that: a never-written ring serialises count/nLive/stride and stops, and a KV-shared
layer stores nothing, so a well-formed body can carry zero KV bytes at pos > 0. That made the
guard reject valid snapshots — including, in production, any session longer than ~8·W on an
all-sliding-window model, since a ring layer only ever stores min(count, W) rows.

CI CANNOT SEE THIS FILE'S FAILURES WITHOUT A FIXTURE. loadTestModel skips when no model is
present, so on a fixture-less runner these tests skip and the package reports ok — which is how
the original defect sat on main unnoticed. If this test starts skipping in an environment that
is supposed to have a model, that is the finding, not a pass.
```

## TestSession_cancelledSweepOnAWrappedRingGoesCold

Moved from `decoder/session_rewind_test.go` (the comment above the test) on 2026-10-10.

```text
C-05: A CANCELLED BATCHED PREFILL ON A WRAPPED RING LEAVES STALE K/V THAT THE NEXT WARM TURN
READS AS HISTORY.

G18 made the batched sweep abortable per LAYER. Layers below the abort point have already
commitBatch'd — their ring count is startPos+K — while c.pos is still startPos, because advanceTo
runs only at the end of a completed sweep. reconcile then truncates to c.pos, which rewinds those
rings by K on a wrapped window: ring.truncate cannot restore the rows the commit evicted and
returns false. reconcile discarded that bool, so the session stayed WARM over K/V RoPE'd at
positions the next turn will read as much earlier history. The request that was cancelled reports
a clean end, so nothing else in the stack notices.

This drives reconcile directly. A full serve round-trip would need a client that disconnects
mid-prefill; the cache state it produces is what is built here, and it is the state that matters.
```

## TestRopeInvFreqLayer_NoPEIsZero

Moved from `decoder/smollm3_test.go` (the comment above the test) on 2026-10-10.

```text
TestRopeInvFreqLayer_NoPEIsZero is G5's real gate (docs/tasks/task-gpu-paths-2026-09.md, FeatNoPE):
a resident backend's rope kernel gets an all-zero invFreq table for a NoPE layer instead of a
new kernel path (RopeInvFreqLayer, decoder/residency.go), which is exact identity rotation by
construction (cos(pos·0)=1, sin(pos·0)=0 for every position — verified against the shipped
kernel math directly, cuda/gemv_fwd.cu and metal/kernels.go's rope2, both
`c=cos(th)*scale,s=sin(th)*scale` with th=pos·invf[dd]).

This is deliberately a PURE, backend-agnostic unit test of RopeInvFreqLayer itself, not a
resident-vs-CPU cosine comparison on testdata/smollm3-tiny. Measured directly (not assumed):
on that fixture (hidden=64, seeded/synthetic weights, `TestSmolLM3ResidentParityMetal`),
zeroing ONLY the NoPE layer, zeroing NO layer, and zeroing EVERY layer's rope all land within
~0.0006 cosine of each other against the CPU reference (worst cosine 0.9617/0.9619/0.9624) —
the SAME "cannot discriminate a real bug from int8-on-random-weights noise" finding this
backend's own Mellum-on-Metal note (features.go) already recorded for a different family. A
resident-vs-CPU cosine floor on this fixture would therefore pass or fail independent of
whether NoPE is implemented correctly, which is worse than no gate — this test is the real one.
```

## spark25_real_test.header

Moved from `decoder/spark25_real_test.go` (the comment above the package clause) on 2026-10-10.

```text
Real-model gates for Spark-X2.5-1.7B (model_type "spark2_5", Spark2_5ForCausalLM) — Gate 2
(docs/tasks/task-spark-x2-5.md): the T3 promotion of the spark2_5 family from Gate 1's
tiny-random synthetic fixture to a released checkpoint. 1.7B is the smaller of the two
released sizes (task doc's own "start with the 1.7B... then the 4B"). Fixture:
scripts/pin_spark2_5_real.py.

TestSpark25Real_gate and TestSpark25Real_fitGuardLongContext need OPPOSITE fit-guard states and
must be run as SEPARATE invocations, not together: this machine's real headroom (~8 GB) is
narrower than the checkpoint's ~6.4 GB f32-resident need at the guard's 70% conservative margin
(a real, monitored bypass was needed and approved to pass the first gate at all — see
docs/measurements/ once Gate 2/3 are written up), so TestSpark25Real_gate needs
GOINFER_NO_FIT_GUARD=1 to load; TestSpark25Real_fitGuardLongContext specifically checks that an
oversized pin gets REFUSED, so it needs the guard ACTIVE (unset) to mean anything — run with
GOINFER_NO_FIT_GUARD=1 set, it silently no-ops (Load succeeds, the test's own "expected a clean
refusal" assertion fails on an unrelated cause).
```

## TestSpark25Real_fitGuardLongContext

Moved from `decoder/spark25_real_test.go` (the comment above the test) on 2026-10-10.

```text
TestSpark25Real_fitGuardLongContext is Gate 3 (docs/tasks/task-spark-x2-5.md): "the fit guard
must price KV correctly for a sliding-window model at long context — which is exactly the case
where a naive calculation is most wrong. Verify the guard's behaviour on spark2_5 at 131072
explicitly." The task doc's own hazard is real: an HF user hit a kernel OOM running
Spark-X2.5-4B at -c 131072 with llama.cpp's --swa-full (which explicitly DISABLES that engine's
sliding-window memory saving, so every layer priced at full context). goinfer's own M-28 fix
(decoder/arch.go's kvPositionsAt/kvBytesForCtx) is the generic mechanism that's supposed to
prevent the equivalent mistake here — this checks it against the REAL checkpoint's geometry,
not just a synthetic fixture (fitkv_test.go's TestFitGuard_slidingWindowFlattensPastTheWindow
already covers the generic mechanism on olmo3-tiny; this is the family-specific confirmation
the task doc asks for, on real numbers).
```

## TestSpecAdaptiveSwitch_offByDefaultUnaffected

Moved from `decoder/spec_adaptive_switch_test.go` (the comment above the test) on 2026-10-10.

```text
TestSpecAdaptiveSwitch_offByDefaultUnaffected: with -spec-adaptive off, a spec generation's own exclusive claim
now goes through claimExclusive (holders == 0), not a bare CAS on resBusy — the hardening this file's other test
forced (a bare CAS could succeed at the same moment an MC3 holder, admitted via bt.claim, was mid-step: bt.claim
only READS resBusy, it never sets it). Production never creates this combination (-spec without -spec-adaptive
forces concurrency to 1 in openai.go's setConcurrency, so a spec generation and an MC3 holder never coexist),
but the test constructs it directly to prove the guard does not depend on that. Two outcomes are both correct:
the spec generation wins the claim (resident, mc3Fake's path) or loses to an already-admitted holder and falls
back to the staged CPU path (the tiny fixture's real weights, a genuinely different computation, not a
degraded one) — this is the pre-existing M9 guard's own contract ("both still complete correctly, only the
loser loses resident speed"), unrelated to this session's changes. NOT compared against an always-resident
reference: unlike matchesPlainDecode, a mid-run fallback to a different code path is not expected to be
id-identical to the path it did not take — that was this test's own first-draft mistake, not a real
divergence (caught by running enough repeats: it failed ~1 run in 5, every time on a real, different, but
individually valid continuation, e.g. `got=[171 197 164...]` vs `alone=[102 186 172...]` — two distinct valid
greedy paths, not a wrong one).
```

## TestVerifyTheta_asksWhetherForwardNIsBatched

Moved from `decoder/spec_adaptive_test.go` (the comment above the test) on 2026-10-10.

```text
M-14: verifyTheta keyed on the BACKEND NAME, so every CUDA model got 0.251 — a value measured
on dense 0.5B/1.5B, where the batched prefill pass actually runs. But cuda's ForwardN falls
back to one `step` per row for every MoE / K=V / non-uniform / non-int4-or-int8 model, and a
loop of single-token forwards has Theta ≈ 1 by construction.

So on a resident MoE the controller was told a verify node costs a quarter of a decode step
and drafted 8, when each node costs a FULL step: nine sequential steps per round for ~6.7
committed tokens at high acceptance, worse below it. Metal hit the same shape, MEASURED it
(1.006–1.048, linear to n=16) and ships 1.02 to disable speculation. This makes CUDA reach the
same conclusion the same way — by asking the resident instead of inferring from its name.
```

## TestThetaFor_cudaConstantUnchanged

Moved from `decoder/spec_adaptive_test.go` (the comment above the test) on 2026-10-10.

```text
The CUDA constant itself is unchanged and still measured — M-14 is about WHEN it applies, not
what it is. Pinned so a change to one is not mistaken for the other.

Metal ships thetaFor("metal") = 0.96 (re-measured 2026-09-17 across {0.5B,1.5B} qwen2.5-coder and
{0.6B,1.7B} Qwen3, depth 128-2048; down from 1.02+ when ForwardN was an unbatched loop). N-49
tripwire: batched verify reports via VerifyPathReporter, unblocking speculative decoding on
Metal for all non-paged models.
```

## TestSpecHitRate

Moved from `decoder/spec_hitrate_probe_test.go` (the comment above the test) on 2026-10-10.

```text
TestSpecHitRate is Step-6 Step-0b: measure the SPECULATIVE hit rate for Metal MoE expert paging
directly, on the CPU MoE path — the business case for building speculation at all. It does NOT
rest on CUDA's 38-slot VRAM hit rate (81.6%, different hardware/ratio); it measures the strict
quantity speculation needs: given the previous token's expert set plus a bounded per-layer LRU
pool, is every expert in THIS token's top-8 already resident (no stall)?

Model: gemma4-26b-A4B (the real target: 30 MoE layers, nE=128, top-8). Greedy autoregressive
generation from a diverse seed (deterministic + repo-reproducible; a repetition guard flags the
case where greedy loops, which would inflate locality). Router top-8 per MoE layer per token is
captured via SetRouterCaptureForTest. Reports, per layer and aggregated:
 1. exact-set match  P(top8_t == top8_{t-1})   — strictest "speculate from previous token".
 2. coverage(N)      P(top8_t ⊆ per-layer LRU-N pool, which includes last token's set for N>=8).
 3. stalls/token(N) = Σ_layers (1 - coverage_L(N)); implied ms/token at 0.230 ms/stall.

GATE: compare (3) at an affordable N against 6.9 ms/token (full synchronous paging = a stall at
every one of the 30 MoE layers). Recovers most → build speculation. Recovers < half → synchronous
paging ships and speculation is dropped.
```

## TestGenNgramInto_residentReusesWarmPrefix

Moved from `decoder/spec_ngram_resident_reuse_test.go` (the comment above the test) on 2026-10-10.

```text
TestGenNgramInto_residentReusesWarmPrefix is P-05 (audit-2026-09-10): R-03 already made this
function COMMIT the accepted sequence correctly on exit (residentCommitIDs), but nothing on
ENTRY ever consulted what that commit left behind — every round cold-prefilled the whole
prompt from position 0 regardless, so a --spec/--drafter agent loop got no prefix reuse despite
committing one every round. This drives two rounds back to back: round 1 cold (nothing
committed yet), round 2 with a prompt that STRICTLY EXTENDS round 1's own committed sequence
(the real agent-turn shape residentReuseLen exists for) — and asserts round 2's
Generation.PrefillReused is nonzero, the same observable proof generate_vl.go's own reuse
tests use, not just that generation still produces output.
```

## TestOptFwdEligible

Moved from `decoder/spec_optfwd_test.go` (the comment in the temperature-cap loop) on 2026-10-10.

```text
Every NEGATIVE case below uses a temperature INSIDE the cap on purpose: at 0.7 they would be
refused for being too hot, and would pass without exercising the property they name.
The threshold itself. Above optFwdMaxTemp the overlap is a MEASURED loss (2.8-6.8% on
phi3-mini across T=0.4-1.0), so the common chat range must be excluded -- this is the whole
behaviour change, and 0.7 being ineligible is the point rather than a regression.
```

## realisticInputs.turn2

Moved from `decoder/spec_suffix_probe_test.go` (the comment in realisticInputs, above agentTurn2) on 2026-10-10.

```text
Turn 2 of an agent loop. The system prompt, the first user turn, its search
results and the first answer are all RE-SENT verbatim — that resend is the
structural repetition an agent loop really has, and it is why cross-request
scope buys less on a stateless surface than the paper's setting implies.

The SECOND search returns DIFFERENT source, which is the whole correction
here: a first attempt reused the same snippet for both turns, the model
copied it back, and the pre-registered loop guard excluded the trace
(distinct-trigram 0.297). That was a defect in the INPUT, not in the guard —
a real second query retrieves different code — so the input was fixed and
the guard left alone. Recorded because it is exactly the artifact
arXiv 2604.26469 warns about, caught by the rule written in advance.
```

## policyMostRecentDeep

Moved from `decoder/spec_suffix_replay_test.go` (the comment above the function) on 2026-10-10.

```text
policyMostRecentDeep is the shipped policy with the MaxMatch probe cap raised
16 -> 64. Added because the step-1 histogram showed 14 of 20 hits sitting at
EXACTLY 16 on code-continue-2 — matches pinned at the cap are matches being
truncated, and match length is the alpha-hat signal (spec_ngram.go's
ngramAlphaAnchors tops out at 16 => 0.97). This asks whether the cap is
costing accepted tokens or only mis-reporting confidence.
```

## tailCycle

Moved from `decoder/spec_suffix_replay_test.go` (the comment above the function) on 2026-10-10.

```text
tailCycle is the SECOND, INDEPENDENT loop detector, added when the first one
turned out to be ambiguous on code. distinct-trigram measures REPETITIVENESS,
and real Go source is legitimately repetitive (`float64`, `\n\t`, receiver
names) — the shipped step-1 code trace scored 0.715 against a 0.70 bar written
for PROSE coherence. Conflating "repetitive" with "looping" would either
discard the exact traffic this task is about, or pass a genuinely degenerate
trace. So this checks the actual failure mode instead: is the TAIL periodic?

It returns the smallest period p (<= 32) such that the last `repeats*p` tokens
are exactly periodic with period p, and how many repeats that runs for. A real
infinite loop shows a small p repeating many times; repetitive-but-progressing
code does not.

Deliberately NOT a replacement for the pre-registered trigram rule: that rule
stays binding, this one is reported beside it, and a DISAGREEMENT between them
is itself the finding.
```

## spec_theta_test.header

Moved from `decoder/spec_theta_test.go` (the comment above the first test) on 2026-10-10.

```text
Theta's domain and the per-backend wiring.

The bug these pin: AdaptiveDepth documented and enforced Theta in [0,1) and
reset anything outside it to 0.5. Metal measures 1.006-1.048 (two models, two
depths, 2026-09-01), so EVERY value Metal actually has was rejected and
replaced by 0.5 — the most over-drafting setting on the dial, chosen
automatically at the one moment the measurement said "do not draft at all".
```

## TestTreeUpside

Moved from `decoder/spec_tree_upside_test.go` (the comment above the test) on 2026-10-10.

```text
It replays the drafters over a real constrained greedy generation (agent-loop
case, where the router already wins 4.25 tok/round) and counts tree-recoverable
positions. ~0 ⇒ trees add nothing for these greedy sources ⇒ defer.
```

## TestSSMPrecisionLocalize

Moved from `decoder/ssm_precision_localize_test.go` (the comment before the D1 pass) on 2026-10-10.

```text
E1 refuted "f32 SSM compute is the lever" (100%). The one thing R2 keeps f32 but
the resident quantizes is the int8 MAMBA projections (W8A8 weights + int8 act).
D1 isolates that on the otherwise-perfect CPU path; D2 then forces R1's (f64)
routing onto it — the REAL router-island test, on the path that reproduces the gap.
```

## TestSSMPrecisionLocalize.header

Moved from `decoder/ssm_precision_localize_test.go` (the comment that sat above `ssmLocHeldOut`, away from the test it describes) on 2026-10-10. The new header above the test lists the D1 and D2 passes, which the old text did not, and drops the stale count of "three" passes.

```text
Precision-localization for the granite-resident quality gap (measurement only,
docs/ssm-int8-quality.md). All runs use the cpu backend (f32 weights throughout),
so weight precision is excluded BY CONSTRUCTION — the only variables are the SSM
compute precision (f64 vs f32, ssmForceF32) and a forced high-precision routing
override. One model load, three teacher-forced passes vs the f64 reference (R1):

	E1: cpu f32 SSM                — does downcasting the SSM compute to f32 reproduce
	                                 the ~66% resident drop? (confirms SSM compute is the
	                                 lever, weights excluded)
	E2: cpu f32 SSM + f64 routing  — force R1's (f64) expert selection+weights onto the
	                                 f32-SSM forward. Recovers → router-selection island;
	                                 doesn't → the whole SSM recurrence needs ~f64.
```

## TestDeclinedToCPUReason

Moved from `decoder/staged_device_note_test.go` (the comment above the test) on 2026-10-10.

```text
R9 (docs/measurements/cold-user-2026-09-06-nobara-pc.md, corrected on review): "cuda-staged
(int4)" on an 8 GB card sat at the idle VRAM baseline (464 MiB, unchanged) for a full request
sampled at 1 Hz. Cuda's and metal's own Backend.MatmulBT implementations are bare CPU calls
and neither implements QuantBackend at all, so their "staged" path never reaches the device at
ANY quant — DecodePath() folds that into the same requested-vs-effective shape
BackendSummary() already uses (see decoder/backend_report_test.go), rather than naming a
"cuda-staged"/"metal-staged" path that was never real. declinedToCPUReason is the pure,
Model-free piece of that string; DecodePath()'s own wiring is covered by R9's real-hardware
verification (docs/tasks/task-first-hour.md), not a synthetic Model here.
```

## TestMain

Moved from `decoder/suite_main_test.go` (the comment above TestMain) on 2026-10-10.

```text
The decoder test suite's process-wide setup: a soft memory limit and a periodic reclaimer so the
suite's RSS tracks the working set (see boundSuiteMemory), and mem_probe_test.go's instruments.
(This lived in eagle_fixture_test.go beside the EAGLE fixtures until EAGLE's code was removed.)
```

## boundSuiteMemory

Moved from `decoder/suite_main_test.go` (the comment above the function) on 2026-10-10.

```text
boundSuiteMemory caps the suite's peak RSS with a soft GOMEMLIMIT. The heap-vs-RSS probe
(mem_probe_test.go) delivered the verdict: `go test ./decoder/` balloons to ~31 GB RSS pre-GC but
collapses to ~4 MB HeapInuse / ~8 GB RSS after a forced GC — the "in-use" 31 GB is transient
allocation churn across the many tiny logic tests, freed but neither collected (pacing-based GC
only triggers on heap doubling) nor returned (lazy MADV_FREE). NOT a leak (HeapInuse→4 MB) and NOT
mmap (nonGo RSS-Sys negative). A soft memory limit paces GC to keep that garbage from piling up, so
RSS tracks the working set (a few MB live) instead of the high-water — no thrash on a memory-tight
box, no swap. It cannot starve a real test: nothing holds more than a few MB live. Respects a
user-set GOMEMLIMIT (the runtime already applied it); tune or disable with GOINFER_TEST_MEMLIMIT
(GiB, or "off").
```

## startMemReclaimer

Moved from `decoder/suite_main_test.go` (the comment above the function) on 2026-10-10.

```text
startMemReclaimer returns freed heap pages to the OS on a short interval, so the suite's RSS
tracks the CURRENT working set instead of accumulating a high-water. The probe verdict: the peak
(~24-31 GB) is transient garbage from a cluster of legitimately heavy tests (serialize
round-trips, GGUF loaders — each briefly a few GB live) whose freed pages the runtime holds under
lazy MADV_FREE and never reclaims across the following tests. GODEBUG=madvdontneed=1 would return
them eagerly but can only be set at process start; debug.FreeOSMemory() does the same on demand,
and is CHEAP here because it does a GC over a live heap that is only ~4 MB — the cost is the
madvise of freed spans, not a scan. Every ~3s keeps the peak near a single test's working set.
stop() ends it (before the post-suite probe, so its GCs don't muddy the FreeOSMemory delta).
Disable with GOINFER_TEST_MEMLIMIT=off (same knob as the soft limit).
```

## TestSwapWatch_keyingOnRSSWouldHaveMissedR11c

Moved from `decoder/swapwatch_test.go` (the comment above the test) on 2026-10-10.

```text
TestSwapWatch_keyingOnRSSWouldHaveMissedR11c is the inverting-guard mutation check
docs/tasks/task-never-swap-2026-09.md S3 asks for, encoded as a test so the mistake it guards
against cannot come back silently. CLAUDE.md: "A guard that INVERTS under the condition it
exists for is worse than no guard — it actively reassures." The real R11(c) run
(docs/measurements/metal-moe-autopager-m26-2026-09-20.md) measured, in the SAME build, both
figures below: swap-used genuinely grew from 2.3 to 12 GB and NEVER came back down, while the
test's own RSS reading went "7 MB -> 892 MB -> falling" — because darwin reclaims MTLBuffer
pages under pressure as fast as they are written, so RSS reports what survived, not what was
asked for (S0's own reading of this run). Run BOTH trajectories through the exact same watch:
a swap-keyed watch never resumes (correct — the machine never stopped losing memory); an
RSS-keyed watch resumes mid-spiral, because its own falling tail reads as recovery. That
resume-while-still-losing is the inversion, shown directly rather than argued.
```

## theta_probe_test.header

Moved from `decoder/theta_probe_test.go` (the comment above thetaFit) on 2026-10-10.

```text
---------------------------------------------------------------------------
Theta — the marginal cost of one extra verify node, measured.

spec_adaptive.go's own words: "Theta is the relative cost of one extra verify
node on *this backend* — measure it." The shipped default is 0.5, described as
the batched-CPU ForwardN value. It has never been measured on any backend; the
GPU-resident path uses the CPU constant today (docs/spec/02).

DEFINITION, so the CPU control and the CUDA probe compute the same number:

	T(n)  = wall time of one verify pass over n tokens at a fixed context depth
	Theta = dT/dn / T(1)

i.e. the slope of T(n) expressed in units of one single-token target step. The
AdaptiveDepth rule D = floor(ln Theta / ln alpha) is exactly "extend while the
chance of reaching depth d beats the marginal node cost", so this is the
quantity the controller wants.

THIS FILE IS THE CONTROL. It measures Theta on the staged CPU path, where the
design says it should come out ~0.5. If the instrument cannot reproduce the one
value already believed, its answer on CUDA is not trustworthy either — that is
the whole reason the control exists and runs first.
---------------------------------------------------------------------------
```

## TestMatmulInt4_MConsistent

Moved from `decoder/w4a8_prefill_test.go` (the comment above the test) on 2026-10-10.

```text
TestMatmulInt4_MConsistent verifies the int4 weight matmul gives bit-identical
per-row results at M=1 (decode) and M=K (prefill). This holds only because both
now run the W4A8 integer kernel (weightmat.go): before prefill was routed onto
MatmulBTW4A8 it used the f32-activation MatmulBTQ4, so the two paths differed.
The W4A8 dot product for a given (row, output) is independent of M, so the
match is exact — this is the self-contained gate for the prefill→W4A8 switch
and the prefill↔decode numerics-seam removal.
```

## TestRow4_coldTouchLatency

Moved from `decoder/w4a8_row4_coldtouch_probe_test.go` (the comment above the test) on 2026-10-10.

```text
TestRow4_coldTouchLatency is the experiment the first two ruled-out hypotheses
(mmap-vs-heap: +1.7% noise; row4-vs-canonical on gemma4 shapes: row4 is +57-67%
FASTER, matching the original 1.6-1.75x claim) point to directly: both of those
tests reused the SAME small set of already-resident experts thousands of times,
so they measured warm, steady-state kernel speed -- never the real production path
(expertPager.touch() -> WILLNEED -> matmul) on a genuinely cold, first-ever touch.
Real decode touches ~240 DISTINCT experts per token under a real cache budget; this
test replicates that shape directly: many distinct experts, each touched exactly
once, through the real pager, comparing kind-3 (canonical dispatch) against kind-4
(row4 dispatch) per-touch latency.
```

## TestW4A8Row4GiwKind_loadTimeAndMemoryDelta

Moved from `decoder/w4a8_row4_giwkind_loadcost_test.go` (the comment above the test) on 2026-10-10.

```text
TestW4A8Row4GiwKind_loadTimeAndMemoryDelta measures the comparison the .giw
kind-4 decision actually turns on: not kind-3-.giw vs kind-4-.giw load time
(neither does an in-RAM repack — repackW4A8Row4IfEligible is wired only into
the GGUF/safetensors streaming loaders, never into LoadSerializedWeights,
docs/completed/task-w4a8-neon-bandwidth.md), but a GGUF load that pays the in-RAM
repack cost (TestW4A8Row4_loadTimeAndMemoryDelta's own +100ms/+223.6MB
numbers, on the same fixture) vs. a kind-4 .giw load that gets Int4Row4()
populated for FREE — zero-copy mmap alias, no repack computation, per
WrapInt4Row4's own contract.

Same rigor as the sibling test: order-alternated best-of-3, real 0.5B GGUF
fixture, informational (not a pass/fail gate).
```

## TestW4A8Row4_skippedForMetalBackend

Moved from `decoder/w4a8_row4_metalskip_test.go` (the comment above the test) on 2026-10-10.

```text
TestW4A8Row4_skippedForMetalBackend is the end-to-end wiring proof for M-07 (audit-
metal-2026-09-12.md): a Backend:"metal" load must have NO row4 layout on its Q/K/V/gate/up
projections (isBatchedProjTensor's five standard names — quantizeBatchedProjWM's own scope),
while a Backend:""  (unspecified) load on the SAME checkpoint keeps row4 on those same tensors
exactly as before this change. o_proj/down_proj are ALSO now row4-free on Metal (M-07's second
half, 2026-09-13: quantizeWMSkipRow4 threaded through quantizeWM's ~40 family-specific call
sites) — see TestW4A8Row4_skippedForMetalBackend_MoE for router/expert coverage, which this
dense-only fixture doesn't exercise.
```

## TestRow4_mmapVsHeapResident

Moved from `decoder/w4a8_row4_mmapvheap_probe_test.go` (the comment above the test) on 2026-10-10.

```text
TestRow4_mmapVsHeapResident is the discriminating experiment: same kernel, same
bytes, one variable (memory source), on a HANDFUL of experts -- not the whole
model. The first version of this test heapified all ~7680 expert tensors at once
(~14 GB of fresh heap on this 16 GB box) and drove the machine into severe swap
thrashing (15 of 16 GB swap used) before being killed; this version times raw
MatmulBTW4A8Into calls directly on a small, fixed set of experts instead of running
full model decode, keeping the extra heap footprint under 100 MB.

The resident-path 1.6-1.75x figure (docs/completed/task-w4a8-neon-bandwidth.md) was measured
against heap-resident repacked bytes (RepackInt4Row4, the GGUF/safetensors streaming
loaders). The kind-4 .giw path's row4 bytes are mmap-aliased even when NOT paged (no
StreamWeights) -- Load() always mmaps a .giw file read-only regardless of streaming
mode. If mmap-resident calls are slower than heap-resident calls on the identical
bytes, the quiet-machine gemma4 gap (docs/completed/task-zeno-compare.md's "Quiet-machine
re-measure") is memory-source mechanics (TLB/page-fault residue on mapped pages),
not paging-machinery overhead.
```

## TestRow4_vsCanonical_gemma4Shapes

Moved from `decoder/w4a8_row4_vs_canonical_probe_test.go` (the comment above the test) on 2026-10-10.

```text
TestRow4_vsCanonical_gemma4Shapes is the follow-up the mmap-vs-heap result directly
suggests: memory source is ruled out (TestRow4_mmapVsHeapResident, +1.7% noise), so
if gemma4's paged decode is still ~47-49% slower with kind-4 than kind-3
(docs/completed/task-zeno-compare.md's "Quiet-machine re-measure"), the remaining candidate is
that the row4 kernel itself is not faster than canonical on THESE SPECIFIC expert
shapes -- the original 1.6-1.75x figure (docs/completed/task-w4a8-neon-bandwidth.md) may have
been measured on different tensor dimensions. Calls both free-function kernels
directly on the same real gemma4 expert bytes, same activation, same dst -- isolates
kernel choice as the only variable.
```

## TestRow4GiwKind_qwen35_pagedEviction

Moved from `decoder/w4a8_row4_realmodel_test.go` (the comment above the budget const) on 2026-10-10.

```text
Post-fix, a kind-4 bundle registers only ONE span per expert (row4 when usable,
canonical otherwise) -- the pageable total is back to matching kind-3's ~15.6 GB,
not the pre-fix doubled ~31.2 GB. 3 GB (~19% residency) forces real eviction over
a 40-token run without landing in the near-zero-residency, all-miss regime.
```

## TestResidentWeightBytesPaged_capsAtSlots

Moved from `decoder/weightbytes_test.go` (the comment above the test) on 2026-10-10.

```text
TestResidentWeightBytesPaged_capsAtSlots is M-02's break-it-first gate for the accounting half
of the fix: before it, the guard always used the unpaged sum, so a model that would fit under
GOINFER_METAL_MOE_SLOTS paging (a few GB) was declined on the number it would need fully
resident (tens of GB, per the audit's Qwen3.5-35B-A3B example). Verifies three properties, then
cross-checks the exact paged byte count against an independently-written per-layer formula
rather than trusting the production code's own arithmetic.
```

## TestLoadWeights_goldenChecksums

Moved from `decoder/weights_test.go` (the comment above the P13 arms loop) on 2026-10-10.

```text
Run BOTH sides of P13, because they answer different questions and only
one of them existed before P13 did.

  p13_off     — the source mapping is retained, so every assertion below
                runs, INCLUDING the stored-dtype check that proves the
                BF16 widen path ran. That check reads w.st, which P13
                closes by default, so without this arm it would quietly
                stop running rather than fail.
  p13_default — the shipped path, where the mapping is released at end of
                load. The checksums here are what says that releasing it
                did not corrupt the weights already loaded out of it,
                which is the one thing P13 could plausibly break and the
                exact risk mmapAliasRisk is guarding.

P13 closing w.st is also why this test panicked rather than failed when it
was first run against a checkpoint: the Mac skips it for want of the asset,
so the nil deref only ever appeared on the box.
```

## TestSpecHitRate.input

Moved from `decoder/spec_hitrate_probe_test.go` (the comment above the input-stream setup) on 2026-10-10.

```text
TRUSTWORTHY input needs a REAL, diverse token stream. This box has no in-package tokenizer, and
self-generated text from the 26B BASE model (greedy OR temp=1.0 top-k=40 sampling) degenerates
into a low-entropy loop (4–12% distinct tokens) whose expert-locality is NOT representative of
real generation. So the primary path is GOINFER_SPEC_TOKENS=<file of whitespace-separated real
token ids> (e.g. a corpus tokenized elsewhere / on the box), teacher-forced. Absent that, it
falls back to seeded sampling and LOUDLY flags the result as unrepresentative.
```

## TestSpecHitRate.giwpath

Moved from `decoder/spec_hitrate_probe_test.go` (the comment above the GOINFER_SPEC_PROBE_GIW read) on 2026-10-10.

```text
N-41: was a hardcoded /Users/<me>/ path — the last surviving dev-home path after G-06,
so this probe could only ever run on one machine and silently did nothing anywhere else.
```

## benchFilterFreshScratch

Moved from `decoder/sampler_selection_test.go` (the comment above the function) on 2026-10-10.

```text
benchFilterFreshScratch is the PRE-P10 shape: a fresh full-vocab buffer per call, which is what
sampleChunked/chunkedZ did before 4da116d. Paired against benchFilter's reused scratch this
isolates P10 itself, rather than the optimized path as a whole — the existing Ref/New pair
compares two different algorithms and cannot answer the G26 question.
```
