# Task: Zeno head-to-head — feasibility and matched-depth comparison

> Scoping/measurement doc. Opened 2026-08-24 from `docs/prompts/zeno-compare-phase0.md`. Context:
> Icosa's Zeno (r/ollama post, 2026-08) ships 4-bit Qwen3.5-35B-A3B on 16 GB Macs via disk
> offloading, posting 10k prompt — 214 tok/s prefill / 8.7 tok/s decode; 541 prompt — 66/13.2 tok/s;
> llama.cpp best-config at 8.7-12.6 prefill / 3.5 decode. Vendor-posted, one table image, treat as
> directional. goinfer's nearest measured point is gemma4 26B-A4B paged at 16.98 tok/s
> (`docs/completed/gemma4-resident-scope.md`).

## Phase 0, Part A — can goinfer run the actual model?

**Rig:** Apple M1 Pro, 16 GB unified memory, macOS. Model: `unsloth/Qwen3.5-35B-A3B-GGUF`,
`Qwen3.5-35B-A3B-Q4_K_M.gguf`, 22.0 GB (22,016,023,168 bytes), single file, downloaded and verified
2026-08-24 (`GGUF` magic bytes confirmed; matches the size independently reported by both
`unsloth` and `bartowski`'s GGUF repos for this quant — weight-identity with Zeno's own shipped
copy cannot be verified, per the brief's own caveat, since Zeno is closed-source).

**Step 1 (disk):** done before downloading, as required. 47 GB free at the start; freed ~8 GB more
via `go clean -cache` + `brew cleanup -s` (both safe, regenerable caches) before starting, landing
at 54-57 GB free — enough margin for the 22 GB GGUF, confirmed after the fact.

**Step 2 (prequant → paged load): BLOCKED — real, precise, fixable finding, not yet fixed.**

```
go run ./cmd/prequant -o ~/models/qwen3.5-35b-a3b-int4.giw -quant int4 \
  ~/models/Qwen3.5-35B-A3B-Q4_K_M.gguf
```

killed by the kernel after growing to **40,544 MB (40.5 GB) resident** — confirmed via
`log show --predicate 'eventMessage CONTAINS "prequant"'`: `memorystatus: killing largest
compressed process prequant [81483] 40544 MB`. 16 GB total RAM (`sysctl hw.memsize`), swap nearly
exhausted (`vm.swapusage`: 5.35/6.14 GB used) before the kill. Output `.giw` was a 17-byte stub —
the process died before writing any real tensor data.

**Mechanism, characterized (not fixed) per this doc's own house discipline:**
`decoder.StreamTranscodeGGUF` (`decoder/gguf.go:1236-1242`) has an explicit carve-out:

> `// Dedicated-loader families (qwen35) can't stream; they fit resident.`
> `if arch.qwen35 != nil { w, berr := buildWeightsFromGGUF(cfg, arch, g, q, embedInt4, nil, ""); ... }`

and the streaming branch itself refuses the family outright (`decoder/gguf.go:1508-1510`):

> `if arch.qwen35 != nil || arch.gemma4 != nil { return nil, fmt.Errorf("...streaming transcode
> unsupported for %s (load resident + prequant instead)", arch.Name) }`

`cmd/prequant`'s own doc comment names the assumption directly (`internal/prequant/prequant.go:56-60`):
"transcode the GGUF straight into the bundle, ONE LAYER at a time... peak RAM is ~one layer rather
than the whole resident model... The dedicated qwen35/gemma4 loaders fall back to a resident build
inside StreamTranscodeGGUF (**those models fit**)." **That assumption is what broke**: it held for
every qwen35-family model tried before now (all smaller), and for gemma4 26B-A4B on a bigger box —
a 35B-A3B on a 16 GB Mac is the first case where it doesn't. The generic loader path already proves
per-layer stream-and-free works on this codebase (`streamQuantized`, `decoder/weightmat.go:123` —
"the whole model never sits resident"); qwen35's dedicated loader was written before that pattern
existed and was never migrated onto it. This is a genuine, fixable bring-up gap, not an inherent
MoE or quantization-format limit. The separately-known P12 finding (qwen35 family running f32
attention/DeltaNet scratch regardless of `Options.Quant`, `docs/queue-performance.md`) is NOT this
issue — P12 was fixed 2026-08-19 and is a *decode-time bandwidth* cost, not this *load-time
resident-peak* one; both live in the same family's bring-up debt but are independent findings.

**Phase 0's own framing says this is a complete, valuable outcome on its own**: "the real
checkpoint won't load/page/decode... file it as the bring-up gap it reveals." Recorded here as
that gap. Cleanup done: the 17-byte stub `.giw` removed, disk back to ~50 GB free.

**Steps 3 (coherent decode) and 4 (size the f32-scratch handicap) — not reached.** Both depend on
the model loading at all; deferred until the streaming fix (below) lands and this step is re-run.

**Step 3 (coherent decode), re-run 2026-08-24 after the streaming fix landed:**

```
go run ./cmd/serve -addr 127.0.0.1:8099 \
  -model "zeno-test=~/models/qwen3.5-35b-a3b-int4.giw,stream,weight-cache=6"
```

Loaded successfully — no OOM. `decoder: expert paging: 10240 experts, 15.6 GB total, 6.0 GB
budget`; `loaded "zeno-test": 40-layer model (vocab 248320) in 1m17.066s`; `decode path: cpu
(int4mix)`; `prefill path: sequential` (this arch has no batched CPU prefill). Resident RSS
settled to **~2.4-2.7 GB** after load, well inside the 6 GB expert-cache budget and far under the
16 GB ceiling.

A real chat completion (`POST /v1/chat/completions`, 19-token prompt, `max_tokens: 40`,
`temperature: 0`) returned coherent, on-topic prose (a legible reasoning trace opening
`<think>\nThinking Process:\n\n1. **Analyze the Request:**...`) in **46.57 s wall time** —
reported at the time as ~0.86 tok/s (completion tokens only). **Corrected below** (see "Diagnostic:
the ~1160 ms/token gap"): this architecture's prefill is unbatched and costs the same per-token as
decode, so dividing only by completion tokens smears prefill's one-time cost into the denominator.
The correct steady-state figure, dividing by every forward pass (prompt + completion), is
**~1.2-1.4 tok/s** across five real runs. No crash, no OOM, no incoherent output — Part A's
original blocking question ("can goinfer run the actual model at all") is now answered **yes**;
the rate itself is characterized fully in the diagnostic section below, not re-derived here.

Against the brief's own reference points: Zeno's own posted 8.7 tok/s decode (10k-prompt row) and
llama.cpp's best-config 3.5 tok/s decode are both above this corrected ~1.2-1.4 tok/s. goinfer's
expert-demand-paging path here is synchronous, per-token, per-miss disk faults against a 6 GB
cache over 15.6 GB of experts — unsurprising that it's the slowest of the three; this is a first
real number, not a tuned one (no cache-budget sweep, no speculative/prefetch paging attempted).

**Step 4 (size the f32-scratch handicap): not run this pass.** The P12 fix already closed the
"f32 attention/DeltaNet scratch regardless of Options.Quant" finding (2026-08-19) for the general
case; whether any f32-scratch cost still measurably taxes THIS specific streamed/paged qwen35 path
is a distinct, sizing-only question the brief flagged as owed, not answered here — this pass's
budget went to proving the model runs at all (step 3) rather than to isolating that cost.

## Diagnostic: the ~1160 ms/token gap (2026-08-24, `docs/prompts/qwen35b-paged-decode-diagnostic.md`)

**Fixes nothing — diagnosis only, per that brief.** Closes Phase 0's last open item (the
f32-scratch handicap) and answers where the real per-token cost goes. Method: the fourth outing of
this repo's env-gated component-stub/timing approach (piggybacked on the existing
`GOINFER_DECODE_TIMING` flag), plus a static weight-kind inspection and an `iostat` cross-check.
All instrumentation was added, used to record the numbers below, then reverted — the working tree
carries only the permanent, env-gated `decoder/qwen35_paged_diag_probe_test.go` (a static inspector,
no hot-path cost, same disposition as the existing `*_probe_test.go` files).

**Headline number corrected first.** Part A's "~0.86 tok/s" divided wall-clock by completion tokens
alone. This architecture's prefill is NOT batched — "prefill path: sequential... this arch has its
own per-token forward" (its own startup log) — so a prompt token costs the same as a completion
token, and dividing only by completion tokens smears the one-time prefill cost into that smaller
denominator, understating the steady-state rate. Dividing by ALL forward passes (19 prompt + N
completion) across 5 real runs gives **703-817 ms/token, 1.22-1.42 tok/s** (avg ~1.30 tok/s) — the
correct steady-state figure, ~50% faster than the original framing. Still a ~13x gap to gemma4's
16.98 tok/s, not the ~20x the uncorrected number implied, but not a small one either.

**f32-scratch handicap: SIZED, and CLOSED for this checkpoint.** Direct inspection of the real
loaded `.giw` (`TestQwen35PagedDiag_weightKinds`, 245s to load+walk 40 layers) found every dominant
projection genuinely `int4`: DeltaNet's `inProjQKV`/`inProjZ`/`outProj` (506.8 MB across all 30
DeltaNet layers), the 10 softmax layers' q/k/v/oProj (137.2 MB), and all 10240 experts (16.36 GB).
The ONLY non-quantized body weight is the router (f32, 83.9 MB across 40 layers — this is also
why the load-time label reads "int4mix": `quantLabel()`'s `hasOther` case, not a mixed body) plus
small per-layer gate/norm vectors (~20 MB). Together under 0.6% of 18.19 GB total resident weight
bytes. P12's 2026-08-19 fix holds for this real streamed prequant checkpoint — **this is not the
gap's explanation**, closing the Phase 0 brief's last open item with a clean negative.

**LM head: verified fast, by tracing the actual load call, not just reading the fix.** `embMat`
(`decoder/gguf.go:1402`), the SAME shared closure used by every family including qwen35 for
`w.Embed`/`w.LMHead`, calls `quant.embeddingWith(embedInt4)` — the exact function the 2026-08-24
W8A8 fix (`a11c56b`) changed to tag int4-mode embed/head `quantInt8I8` instead of weight-only
`quantInt8`. Confirmed by tracing the call site this model's load actually goes through (the
dispatch-inertness lesson: a fix existing in the code is not the same as this model's load path
calling it) — not independently re-measured by field inspection this pass, which would need one
more 4-minute reload; the code-path trace is strong enough evidence to not spend it.

**The stub split** (three real decode runs, `GOINFER_DECODE_TIMING=1`, component sums cross-checked
against the framework's own independent "forward" timer — reconciled to within 2.3%, matching the
A0 splits' bar):

| bucket | ms/token | % of forward | method |
|---|---|---|---|
| MoE (paged routing + expert matmul + I/O) | ~540 | **70.5%** | wrap `moeMLP` call |
| DeltaNet recurrence (30 of 40 layers) | ~146 | 19.0% | wrap `gatedDeltaNetStep` call |
| LM head | ~53 | 7.0% | wrap `logitsFromHidden`'s matmul |
| Softmax attention (10 of 40 layers) | ~27 | 3.5% | wrap `qwen35Attention` call |
| sample / logitProc / embed | <2 | <0.3% | existing `decodeTiming` buckets, confirmed negligible |

Sum ≈ 766 ms/token vs the framework's own independently-measured "forward" average of 764 ms/token
(0.26% off) — the four qwen35 buckets plus existing overhead buckets account for essentially all
of it, no missing component.

**MoE's 70.5% is dominated by I/O, not compute — cross-checked, not stubbed** (stubbing the actual
page-fault away would mean touching aikit's `mmap.SpanCache`, explicitly out of scope). `iostat`
during a real decode run measured **~400-630 MB/s sustained, ~24.8 GB total read for one 79-token
request** — **1.52x** the entire 16.36 GB expert pool, against a 6 GB pager budget. Naive warm/cold
reruns (same prompt, back-to-back) showed no speed difference (~1017 vs ~1044 ms/token before this
correction) — NOT evidence of a compute-bound path, as first assumed.

**CORRECTED (2026-08-24, before the paging campaign started):** the original explanation here —
"the pager evicts via `MADV_DONTNEED`, which drops physical pages from the OS page cache too" — is
**wrong on darwin**, this machine's platform. `aikit/mmap/madvise_darwin.go` (confirmed current at
v1.26.0) documents `MADV_DONTNEED` as a deliberate no-op on this OS: there is no syscall that forces
a resident drop on this read-only mapping, so the pager's budget is bookkeeping, not an enforced
cap, and the real replacement decision belongs to macOS's Unified Buffer Cache reclaiming under
genuine system memory pressure — independent of the pager's own LRU signal. This was already
documented in `docs/task-moe-streaming.md`, missed here for want of a prior-art check before
writing the explanation (see `docs/prompts/qwen35b-paging-campaign.md`'s own prior-art correction).
The observation itself (no warm/cold difference despite heavy real I/O) stands; only the mechanism
was wrong. The observed 24.8 GB also exceeds a naive miss-count estimate (logical miss rate 79-84%,
~1.6 MB/expert ⇒ ~8-10 GB) by a real margin — either read amplification (fetching more than the
touched bytes per fault) or faults invisible to the pager's own hit/miss counter (a logical "hit"
not guaranteeing the physical page survived pressure-driven reclaim). Worth sizing further before
any fix, not concluded here.

**Pager stats vs gemma4:** hit rate 79.0% → 82.4% → 83.6% across three back-to-back identical-prompt
runs (popular-expert reuse climbing slightly) — genuinely close to gemma4's reported 81.6%. Similar
hit rate, ~13x slower steady-state tok/s: the gap is NOT primarily "worse cache behavior" the way a
naive first guess would have it. Likely candidates left open for the next campaign: per-fault byte
cost, total working-set size relative to cache (qwen35's 16.36 GB pool vs gemma4's, not compared
here), and DeltaNet's ~19% tax that gemma4 (no recurrent layers) simply doesn't carry.

**Closing recommendation, sizing only — not attempted here:** the paged-MoE I/O path is the clear
top lever (70.5% of forward, and the read-amplification finding suggests real headroom beyond a
naive "more cache" fix); DeltaNet's scalar-Go recurrence is the clear second (19%, "never been on
any perf campaign" per the brief — the A1 attention-threading wins do not reach it). LM head and
softmax attention are both already fast and not worth further chasing here.

## Paging campaign, Step 0: the read-rate probe and ceiling math (2026-08-24)

`docs/prompts/qwen35b-paging-campaign.md` (rescoped after a prior-art correction — see
`[[qwen35-paging-campaign-rescoped]]` in memory; two of its four levers were already answered by
`docs/task-moe-streaming.md`) asked for a cheap three-way read-rate probe before committing to any
owned-buffer engineering. Real spans from the real 35B-A3B `.giw` (900 experts/set, ~472 MB/set,
disjoint), same file, same box:

| method | rate | vs. today |
|---|---|---|
| A) fault-driven mmap touch (today's production path) | **0.322 GB/s** | 1.0x |
| B) single-threaded `pread` | 1.211 GB/s | 3.8x |
| C) async `pread`, queue depth 8 | **1.785 GB/s** | 5.5x |
| D) async `pread`, queue depth 16 | 1.414 GB/s | 4.4x (worse than C — diminishing returns past QD8 on this SSD) |
| SEQ) 1 GB single contiguous read | 2.679 GB/s | 8.3x (the real ceiling; random small reads cost more than sequential, expected) |

Confirms the hypothesis cleanly: page-fault-driven reads are effectively queue-depth-1 I/O, and
QD8 `pread` (not QD16) is this box's sweet spot for ~1.6 MB random reads.

**Ceiling math, honestly bounded — the campaign's own required step before building.** The MoE
bucket (~540 ms/token, 70.5% of forward) is not cleanly separable into I/O-wait vs. compute time
without instrumenting inside `aikit/mmap.SpanCache` (out of scope), so the projected win is a band
over plausible I/O fractions (50/75/90%) and two baseline rates (this probe's clean 0.32 GB/s vs.
the live diagnostic's 0.4-0.6 GB/s average), applying the QD8 rate (1.785 GB/s) only to the assumed
I/O portion:

| I/O fraction of MoE bucket | projected tok/s | vs. today's ~1.29 |
|---|---|---|
| 50% | 1.67-1.80 | 1.3-1.4x |
| 75% | 1.97-2.25 | 1.5-1.8x |
| 90% | 2.20-2.65 | 1.7-2.1x |

**This falls short of the campaign's own ≥3x/≥4 tok/s working target on the read-path lever alone**
— exactly the "if that ceiling comes out under ~4 tok/s, say so before spending the week" case the
brief itself anticipated. The router-time prefetch/overlap lever (WILLNEED the next layer's routed
experts while the current layer computes) composes with this and was described in the brief as
"multiplicative" — reaching 3x+ likely needs both, not the read-path fix alone, and that
composition is unmeasured. Building the owned-buffer `pread` path itself (new pager internals,
threading through `moeMLP`, bit-identical gates, `-race`, the gemma4 regression gate) is a real,
multi-file engineering effort that this ceiling math does not unambiguously justify on its own —
recorded here as the checkpoint the campaign's own Step 0 called for, before committing to it.

## Streaming-transcode fix — scoped as its own task, not folded into Phase 0

Deliberately **not attempted as part of Phase 0** — "feasibility only, no benchmarking yet" is this
doc's own charter, and implementing per-layer streaming for a dedicated loader is real engineering,
not a quick characterization. Scoped here so the record exists independent of when (or whether) the
fix lands, and because it has value beyond this comparison: the "dedicated-loader families fit
resident" assumption was already going to be tested by Qwen3.6-35B-A3B (`docs/qwen3_5_moe.md`
scoping), a roadmap target regardless of any Zeno comparison.

**Shape:** mirror the generic path's per-layer stream-and-free inside qwen35's dedicated loader —
both call sites that currently hit the `sink=nil`/resident-fallback branch (`StreamTranscodeGGUF`'s
own qwen35 carve-out, which both the general load path and `cmd/prequant` route through, per
`internal/prequant/prequant.go:56-60`'s own comment — there is exactly one code path to fix, not
two). gemma4 shares the same carve-out (`decoder/gguf.go:1509`) but is NOT in scope here — it fits
resident on the boxes it's been run on; touch only the qwen35 branch unless a gemma4-specific
gap surfaces independently.

**Correctness gate, free from the situation itself:** build the same `.giw` from the qwen3_5_moe-tiny
checkpoint (the existing T3 fixture) through both the old resident path and the new streaming path
— outputs must be byte-identical, tensor for tensor, the same standard `TestSerializeWeights_roundTrip`
already holds other families to. Combined with the existing family parity goldens (`qwen3_5_moe-tiny`
forward parity, the DeltaNet golden) staying green, this makes the change provably inert: a loader
refactor, not a numerics event. T3 re-run required regardless (core file).

**Acceptance:** the qwen3_5_moe-tiny .giw round-trip is byte-identical old-path vs new-path; all
qwen35-family parity gates stay green; the real 35B-A3B prequant conversion completes without OOM
on this 16 GB Mac, producing a real (non-stub) `.giw`. Peak RSS during the real conversion is worth
logging even though there's no fixed target — the generic path's own "~one layer" framing sets the
expectation, not a specific number.

**DONE, 2026-08-24.** No tiny qwen35 GGUF fixture exists (`qwen3_5_moe-tiny` is safetensors), so
the round-trip gate ran against the real 22 GB GGUF with `NumLayers` clamped to 2
(`decoder/qwen35_streaming_test.go`, `TestQwen35StreamingTranscode_matchesResident`) — a genuine
subset of the real model's real tensors, small enough to safely also run the old resident path for
comparison. It found one real, designed (not buggy) difference: the old resident-then-serialize
path bakes a resolved `quantLabel` ("int4mix") into the header because `w` is fully materialized
before `writeHeadGlobals` runs; the new streaming path calls `writeHeadGlobals` before any layer
streams in, so `hasPopulatedLayers()` correctly defers the label to "" (pre-existing B11 logic,
`decoder/serialize.go:170-185`) — exactly how every other already-streaming family already behaves.
Every other byte agrees, and both bundles resolve to the same live-inferred label once loaded. All
qwen35-family goldens (`TestQwen35_forwardParity`, `TestQwen3_5_textParity`,
`TestSerializeQwen35_roundTrip`) stayed green, and the change is provably non-numeric (proven by the
above), so `scripts/refresh_parity_hashes.sh` was used for T3 rather than the full 12-minute gate —
27 goldens green, 0 failed (one heavy Mellum2 checkpoint was skipped for a missing local shard,
unrelated to this change and to any family this fix touches).

Real conversion: `go run ./cmd/prequant -o ~/models/qwen3.5-35b-a3b-int4.giw -quant int4
~/models/Qwen3.5-35B-A3B-Q4_K_M.gguf` completed in a few minutes, peak observed RSS **~7.9 GB**
(vs. the 40.5 GB OOM kill before this fix), producing a real 22,146,331,551-byte `.giw`
(`wrote ...: 21120 MB (int4)`). Verified loadable: read back through the `internal/giw` bundle
reader + `decoder.LoadSerializedWeights` — 40 layers, `NumLayers=40`, `VocabSize=248320`, no error.

## Go/no-go for Phase 1

**Part A: CLEARS, as of 2026-08-24** — the streaming fix landed, the real 35B-A3B checkpoint now
loads (mmap + expert demand-paging, ~2.4-2.7 GB resident, 6 GB budget) and decodes coherent prose
at a corrected **~1.2-1.4 tok/s steady-state** (see "Diagnostic: the ~1160 ms/token gap" — the
original ~0.86 tok/s conflated prefill's cost into the completion-token divisor), below both of the
brief's reference points (Zeno 8.7, llama.cpp 3.5) but a real, working, now fully-diagnosed number
rather than a blocked one. Step 4 (sizing the f32-scratch handicap) is DONE — sized and closed
(not the gap's explanation); the diagnostic also ranked where the remaining time goes (paged MoE
I/O ~70%, DeltaNet recurrence ~19%).

**Overall: still NO-GO for Phase 1, gated on Part B alone.** Part A no longer blocks; Part B (Zeno
install feasibility) has not been pursued and gates on Francis's explicit checkpoint before
installing anything, independent of Part A's outcome — that checkpoint has not happened. Nothing in
this pass changes that gate. Should Phase 1 proceed later, it inherits a known, ranked cost profile
rather than an unexamined one — a fix campaign for the paged-MoE I/O path and/or the DeltaNet
recurrence would be the natural precursor to a headline benchmark, not a requirement to run one.

## Not in scope (this doc, both phases)

- Any Phase 1 benchmarking, before Part A clears.
- Fixing anything the diagnostic found (paged-MoE I/O, DeltaNet recurrence) — sizing only, per both
  the Phase 0 brief and the diagnostic's own brief.
- Batched MoE prefill work, Qwen3.6 (wrong match for this comparison), anything W4A8 (that campaign
  owns its own files and box time — this is a side quest and shares no files with it).
- gemma4's identical carve-out — noted, not touched, unless it independently blocks something.
