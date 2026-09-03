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
`decoder.StreamTranscodeGGUF` (`decoder/gguf.go:1561-1567`) has an explicit carve-out:

> `// Dedicated-loader families (qwen35) can't stream; they fit resident.`
> `if arch.qwen35 != nil { w, berr := buildWeightsFromGGUF(cfg, arch, g, q, embedInt4, nil, ""); ... }`

and the streaming branch itself refuses the family outright (`decoder/gguf.go:1551-1553`):

> `if arch.qwen35 != nil || arch.gemma4 != nil { return nil, fmt.Errorf("...streaming transcode
> unsupported for %s (load resident + prequant instead)", arch.Name) }`

`cmd/prequant`'s own doc comment names the assumption directly (`internal/prequant/prequant.go:65-69`):
"transcode the GGUF straight into the bundle, ONE LAYER at a time... peak RAM is ~one layer rather
than the whole resident model... The dedicated qwen35/gemma4 loaders fall back to a resident build
inside StreamTranscodeGGUF (**those models fit**)." **That assumption is what broke**: it held for
every qwen35-family model tried before now (all smaller), and for gemma4 26B-A4B on a bigger box —
a 35B-A3B on a 16 GB Mac is the first case where it doesn't. The generic loader path already proves
per-layer stream-and-free works on this codebase (`streamQuantized`, `decoder/weightmat.go:125` —
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
(`decoder/gguf.go:1448`), the SAME shared closure used by every family including qwen35 for
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

**Closing recommendation — SUPERSEDED, see below.** This originally read "the paged-MoE I/O path
is the clear top lever," reasoning from the `iostat`-measured ~25 GB/request headline number alone.
The admit-time instrumentation below (2026-08-24) found that conclusion wrong: MoE's bucket is
compute-dominated (~86%), not I/O-dominated. DeltaNet's scalar-Go recurrence (19%) and LM head/
attention being already fast still stand.

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
brief itself anticipated. Building the owned-buffer `pread` path itself (new pager internals,
threading through `moeMLP`, bit-identical gates, `-race`, the gemma4 regression gate) is a real,
multi-file engineering effort that this band-based ceiling math did not unambiguously justify on
its own — which is exactly why the next step (below) measured the split directly instead of
banding it.

## Paging campaign: the admit-time I/O-vs-compute split, measured directly (2026-08-24)

The band above assumed an I/O fraction rather than measuring one, because separating I/O from
compute inside the MoE bucket looked like it needed instrumenting inside `aikit/mmap.SpanCache`
(out of scope). It doesn't: the admit point is visible at goinfer's own pager layer
(`decoder/moepaging.go`'s `touch`), which is where `moeMLP` calls in before the expert matmuls run.

**The trap this had to avoid:** with mmap, I/O time doesn't happen where you look for it. Pages
fault lazily, inside the expert matmuls that read them — so naively timing just the admit call
(`cache.Touch`, a `WILLNEED` hint, not a real read) would undercount I/O to near zero, with the
real fault-wait hiding inside what looks like "compute." The instrument forced materialization at
admit instead: a timed byte-per-page touch loop over the just-admitted span, converting the hidden
fault-wait into an explicit number and leaving the subsequent matmul timing as genuine compute.
This changes execution shape (I/O serializes at admit instead of interleaving with compute) —
acceptable for a diagnostic, not for production, and the absolute per-token numbers under this
instrumentation are NOT comparable to the uninstrumented baseline for that reason; only the I/O-
vs-compute ratio within a run is trustworthy. Added, used, reverted (same disposition as
`GOINFER_STUB_MATMUL`) — `decoder/moepaging.go`, `decoder/forward_qwen35.go`, `decoder/model.go`,
none of it shipped.

**Result, two consecutive real requests, same box:**

| run | total(moeMLP) | io (admit, forced) | compute (residual) |
|---|---|---|---|
| 1 | 860.0 ms/tok | 143.3 ms (16.7%) | 716.7 ms (83.3%) |
| 2 (isolated delta) | 843.7 ms/tok | 94.1 ms (11.2%) | 749.6 ms (88.8%) |
| average | — | **13.9%** | **86.1%** |

**The MoE bucket is compute-dominated, not I/O-dominated — the opposite of the `iostat`-based
conclusion above, which is corrected, not merely superseded.** Applying this split to the
diagnostic's own clean 540 ms MoE bucket: ~75 ms I/O, ~465 ms compute. Re-running the read-path
projection with the REAL split instead of a band: even the QD8 `pread` win (5.5x, from the probe
above) applied to just the 75 ms I/O slice yields a new MoE bucket of ~479 ms — a **1.09x**
end-to-end improvement, not the 1.3-2.1x the band suggested and nowhere near the campaign's ≥3x
target.

**Decision, per the branch this instrumentation exists to resolve — SUPERSEDED, see the section
below.** This originally routed away from the owned-buffer `pread` build (I/O is only ~14% of the
bucket) toward the parked int32-per-group GEMV as "the next campaign's opening fact." A finer,
fifth-level split (below) found the actual "compute" attribution, and it changes which lever
funds: the paged path is provably running the WRONG kernel, not a slow one — a much cheaper,
lower-risk fix than a numerics-affecting kernel rewrite, and the pread work's justification shifts
from I/O speed (weak) to being the enabling plumbing for this fix (strong).

## Paging campaign, one level finer: compute was a location, not an attribution (2026-08-24)

Francis's own catch: ~86% of the 540 ms MoE bucket is ~465 ms/token of "compute" over the touched
expert bytes — an effective ~3 GB/s, an order of magnitude below the canonical W4A8 kernel's own
measured ~40 GB/s at 6 workers. That gap meant "compute" was a location, not yet an attribution —
the same position the A0 split was in before it found the non-matmul floor was all attention.

**Fifth outing of the stub method, one level inside the MoE bucket.** Timed router / gather /
GEMV-by-projection-shape (Gate+Up vs. Down) / shared-expert, on the real 35B-A3B, same disjoint-
instrumentation discipline as every split before it (added, measured, reverted —
`decoder/mlp.go`, `decoder/moepaging.go`, `decoder/model.go`, `decoder/forward_qwen35.go`):

| bucket | ms/token | % of compute |
|---|---|---|
| Gate+Up (2 matmuls/expert call) | 513.6 | 62.7% |
| Down (1 matmul/expert call) | 237.3 | 28.9% |
| shared-expert block | 37.2 | 4.5% |
| router (matmul + top-k select) | 28.3 | 3.5% |
| SiLU activation + gather/accumulate | 2.9 | 0.35% |
| **sum** | **819.3** | (vs. compute 819.8 — 99.94% accounted) |

91.6% of "compute" is genuinely GEMV time — no missing component, no surprise concentration in
routing or gather glue. Per call: **~609 µs for a Gate/Up matmul, ~563 µs for Down** — both real
int4 W4A8 GEMVs at similar MAC counts, no shape-specific asymmetry.

**The threshold-bug-class hypothesis (this campaign's leading suspect) was tested directly and
REFUTED.** `int4ParThreshold` (the fan-out cutover for int4 matmuls) was temporarily forced to
`1<<30` — guaranteed above any single expert projection's MAC count, so every call would run
serial instead of parallel — and re-measured on the same real request. Result: **slower, not
faster** (Gate+Up 513.6→761.1 ms, Down 237.3→363.5 ms, +48%/+53%). Parallelization is already
helping here, not hurting; this is not a fourth instance of the threshold bug class, and forcing
serial made the case decisively rather than ambiguously. Threshold reverted immediately after
measuring.

**The actual mechanism, found by reading the code the numbers pointed at.** `decoder/weightmat.go`
documents it directly: the arm64 split-half + 4-row-interleaved W4A8 kernel
(`dotW4A8SplitHalf4Row`, shipped and measured 1.6-1.75x over the canonical kernel,
`docs/task-w4a8-neon-bandwidth.md`) requires a load-time repack (`RepackInt4Row4`) that is wired
into the GGUF/safetensors streaming loaders but **deliberately NOT into the `.giw` loader** — a
paged MoE expert's heap-resident repacked copy would sit alongside its pageable mmap alias and pin
that memory permanently, defeating paging for exactly the tensors it exists to bound. aikit's own
`MatmulBTW4A8Into` dispatch confirms the consequence in its own doc comment: *"a WeightMat that was
never repacked (paged tensors, by construction, since a read-only mmap span has no load-time
repack step) simply always takes the fallback branch here."* This model loads via `.giw` +
`-stream-weights` — exactly that path. **Every one of the ~25,280 per-token expert GEMV calls
measured above is running the older, canonical kernel — never the shipped, proven, 1.6-1.75x
faster one.** This is a known, deliberate, already-documented gap, not a new bug — but it had never
been sized against a real paged 35B-A3B before this split.

Whether the remaining gap to the bulk 40 GB/s figure fully closes even with the row4 kernel is NOT
established here — some of it is plausibly inherent to M=1 single-vector GEMVs (Workspace setup,
per-call activation quantization) that a bulk multi-row benchmark amortizes away and no kernel
choice eliminates. What IS established: a real, sizeable, already-shipped, bit-identical lever is
sitting unused specifically on the paged path.

**Projected win, applied to the diagnostic's clean 540 ms MoE bucket** (91.6% of the 465 ms compute
share is GEMV-attributable → ~426 ms; the measured 1.6-1.75x split-half range applied to only that
slice, io and non-GEMV compute unaffected):

| split-half speedup | new MoE bucket | new forward | tok/s | vs. today's ~1.29 |
|---|---|---|---|---|
| 1.6x (measured low) | 380 ms | 607 ms | 1.62 | 1.26x |
| 1.75x (measured high) | 358 ms | 584 ms | 1.68 | 1.31x |

**Weighed against the parked int32-per-group GEMV, per Francis's own framing — let the split pick:**
the int32-per-group GEMV is numerics-affecting (changes decode bits for every int4 model, full T3
re-validation) with an unmeasured payoff. Wiring the split-half kernel to the paged path reuses an
**already-shipped, already-proven-bit-identical** kernel (`TestDotW4A8SplitHalf4Row_bitIdenticalToCanonical`
already gates it), a **known ~1.3x** projected here — bigger than the read-path/pread lever's own
measured 1.09x — with **zero golden churn**, and its two gating numbers (load-time and resident-
memory cost of the repack) already measured in the W4A8 item-3+4 harness phase. The split picks
this lever, not the kernel rewrite. Design note for whoever builds it: the repack-vs-paging conflict
(`decoder/weightmat.go`'s own comment) is exactly what the owned-buffer `pread` architecture
resolves as a side effect — a pread'd fill is already an owned copy, not an mmap alias, so
repacking it in place adds no new pin-vs-page conflict. The pread work's justification shifts from
I/O speed (weak, 1.09x) to being the enabling plumbing for this kernel fix (strong, ~1.3x) — same
engineering effort, different and stronger reason to build it. Not built in this pass; sized and
handed to the next one.

## The .giw kind-4 lever — SHIPPED 2026-08-24 (`docs/task-w4a8-neon-bandwidth.md`'s "Format follow-on")

Built the same day the split picked it: a new on-disk weightMat kind carrying the split-half +
4-row-interleaved layout alongside the canonical bytes, both zero-copy mmap-aliased at load —
full details, numbers, and code pointers in `docs/task-w4a8-neon-bandwidth.md`'s own "`.giw` kind
4 — SHIPPED" section (aikit v1.27.0's `WrapInt4Row4`/`MappedSpanRow4`, goinfer `giwVersion` 7,
`cmd/prequant -row4`, the `expertPager`/`layerPager` span-registration fix).

**Simpler than this doc anticipated: the owned-buffer `pread` architecture turned out NOT to be
required.** The paragraph above predicted the repack-vs-paging conflict would need pread's
owned-copy semantics to resolve. It doesn't — writing the row4 bytes to DISK at prequant time and
mmap-aliasing them back (exactly like the canonical bytes already are) sidesteps the conflict
entirely: there is no "repack in place" step to worry about, because the correctly-shaped bytes
are already sitting in the file before the pager ever touches them. The pread lever's
justification reverts to what it was before this finding — I/O speed alone, measured at 1.09x —
since the kernel-upgrade payoff (the stronger reason floated above) is now available without it.

**Verified at small scale first** (0.5B fixture, 2026-08-24): bit-identical dispatch proven; load-time
and RAM-delta numbers measured (kind-4 load is 0.111s and adds ~0 resident memory vs. GGUF's
in-RAM-repack 2.01s/+223.6MB). The first 35B re-prequant attempt hit a real, honestly-recorded
disk-space near-miss (a full-model kind-4 bundle grew past 43GB, larger than the ~38GB estimate,
before the attempt was stopped to avoid filling the disk) — the at-scale run was deferred at that
point rather than forced.

## At-scale acceptance run — both real checkpoints, same day (2026-08-24)

Disk headroom cleared (rebuildable build output, caches, and unrelated model checkpoints freed;
see the session record for the full list) and both real bundles were re-prequanted to kind 4:
**gemma4-26b-int4-row4.giw** (31,422,003,214 bytes, +94.9% vs. kind-3's 16,119,795,238 — built via
a giw-to-giw re-serialize since the 26B's original safetensors source is not present on this box,
reusing the exact same `SerializeWeightsToRow4` path `cmd/prequant` uses) and
**qwen3.5-35b-a3b-int4-row4.giw** (43,157,420,911 bytes / 41,158 MB, +94.9% vs. kind-3's
22,146,331,551 — direct one-pass `cmd/prequant -row4` from the GGUF, peak RSS 5.66 GB, 555s). The
near-identical +94.9% growth on two structurally different MoE models (3,840 vs. 10,240 experts)
says the ratio is a property of the format (every eligible int4 tensor roughly doubles), not of a
particular model's shape.

**Correctness: fully green, both models, both real files.**
- `TestRow4GiwKind_gemma4_identicalToCanonical`: kind-3 vs. kind-4, resident, byte-identical over 24
  tokens (383s).
- `TestRow4GiwKind_gemma4_pagedEviction`: full-resident vs. paged (1 GB budget against the kind-4
  bundle's 22.6 GB pageable total — canonical + row4 spans both count), byte-identical, `hits=0
  misses=7680 evictions=7498` (573s). Zero hits is a real, if extreme, non-vacuous eviction proof —
  worth a second look at a realistic budget before trusting it as representative (see below).
- `TestRow4GiwKind_qwen35_pagedEviction`: full-resident vs. paged (6 GB budget against 31.2 GB
  pageable total), byte-identical over 40 tokens, `hits=12939 misses=2421 evictions=307` — real
  eviction at a realistic operating point (1735s; an earlier attempt at 8 tokens/6 GB budget found
  the working set fit entirely inside the budget with zero evictions — too short a run to be a
  useful proof, not a failure, and an even-smaller 2-GB-budget attempt on 16 tokens ran over an
  hour without finishing before being abandoned in favor of the realistic-budget rerun).
- Kind-3 gemma4 loads and decodes correctly throughout (it's the baseline every kind-3-vs-kind-4
  comparison above uses) — version-aware read confirmed, both kinds live side by side on disk.
- T3: green (`TestParityManifest_fresh`, 27/27 families enforced) — expected, since kind-4 dispatch
  is unreachable for any pre-existing caller and touches no canonical bytes.

**Throughput: a real regression, not the projected ~1.3x — the headline finding of this run.**
Measured via `cmd/serve` + `/v1/chat/completions`, steady state (`total_tokens / wall_time`,
prompt+completion together, matching this doc's own corrected method above), 3 runs each, greedy
(`temperature: 0`), fixed prompt:

| model | kind | budget | pageable total | residency | tok/s (3 runs) | avg |
|---|---|--:|--:|--:|---|--:|
| gemma4-26b | 3 | 4 GB | 11.3 GB | 35.4% | 1.162 / 1.119 / 1.104 | **1.13** |
| gemma4-26b | 4 | 4 GB | 22.6 GB | 17.7% | 0.714 / 0.784 / 0.742 | **0.75** (−34%) |
| gemma4-26b | 4 | 8 GB | 22.6 GB | 35.4% (matched) | 0.854 / 0.837 / 0.797 | **0.83** (−27%) |
| qwen3.5-35b-a3b | 3 | 6 GB | 15.6 GB | 38.5% | *(documented above: 1.22-1.42, avg ~1.30)* | **1.30** |
| qwen3.5-35b-a3b | 4 | 12 GB | 31.2 GB | 38.5% (matched) | 0.872 / 1.044 / 0.890 | **0.94** (−28%) |

**The doubled-footprint-at-fixed-budget explanation is ruled out as the sole cause, not just a
confound.** Kind 4 registers BOTH the canonical and row4 spans as pageable
(`decoder/moepaging.go`'s `addExpert` collects `MappedSpan` and `MappedSpanRow4` as separate
entries), so at a fixed absolute budget its residency fraction is roughly half of kind-3's — that
alone would predict a slowdown from more misses. But re-running gemma4 at a budget matched to
kind-3's *residency fraction* (8 GB, same 35.4%) only closed part of the gap (−27% instead of
−34%) — it did not come close to parity, and the 35B run at its own matched fraction (38.5%,
mirroring the documented kind-3 config exactly) shows the same ~28% regression. Two structurally
different models, two different budgets, matched-fraction methodology, same ~25-30% result: this
is a real, repeatable effect, not a paging-budget artifact and not run-to-run noise.

**What this is NOT**, ruled out by the correctness gates above: it is not the dispatch-inertness
bug this whole campaign has been alert for — `MatmulBTW4A8Row4Into` is confirmed reached (the
correctness gates prove kind-4's bytes are read and produce identical output; the aikit dispatch
code reads only `q4Row4`/`q4Row4Scales` when present, never touching canonical bytes in that
branch, so this is not a hidden double-read at the KERNEL level). The kernel is being exercised; it
is exercised *slower*, end-to-end, than not using it, on the paged path specifically.

**CONFIRMED, not a hypothesis — the file-size check that comes first found the real mechanism
directly (Francis's own redirect: check expected-vs-actual bytes before locality).** The size
arithmetic itself came back exact — every eligible expert (7680/7680 on the gemma4 fixture checked)
duplicates at precisely the designed 1:1 ratio (nibbles 1.0000x, f32 scales 1.0000x, no unexpected
bloat; the f16-scale compaction in `docs/task-giw-f16-scales.md` really was never folded in, by
design) — so the on-disk growth is not itself a bug. But re-reading `decoder/moepaging.go`'s
`addExpert` with that confirmed in hand exposed the REAL mechanism one layer up, at the PAGER, not
the kernel: `addExpert` registers BOTH a kind-4 expert's canonical span AND its row4 span under the
SAME cache key (`spans = append(spans, wm.MappedSpan(...), wm.MappedSpanRow4(...))`), and
`aikit/mmap.SpanCache.Touch` iterates **every** span under a key and issues `Advise(s, true)`
(`MADV_WILLNEED`) on **all of them**, unconditionally — it has no concept of "this span won't
actually be read this dispatch." `aikit/mmap/madvise_darwin.go` confirms `MADV_WILLNEED` is a real,
working prefetch hint on this platform (unlike `MADV_DONTNEED`, a documented no-op here) — so
**every cold touch of a kind-4 expert schedules a real disk read-ahead of BOTH copies**, even
though the row4-preferring kernel (`M == 1 && w.q4Row4 != nil`) only ever reads the row4 half. Every
miss pays for reading ~2x the bytes it needs — a per-touch I/O tax that survives matched-residency-
fraction budgets exactly because it isn't a hit-rate effect at all; it's a fixed multiplier on every
miss, consistent with the ~25-30% measured across both models and both budget configurations, and
consistent with this repo's own prior finding that the paged MoE bucket is I/O-heavy, not
compute-heavy, on this box.

**The fix, not implemented here** (real pager code, its own dedicated pass — out of scope for this
proof-running task per its own brief, same as the original acceptance brief's "no new pager work"):
`addExpert`/the pager's registration should register (and therefore `Touch`/WILLNEED/DONTNEED) only
the span the dispatch will actually read for that expert — row4 when `Int4Row4()` reports `ok`
(the M==1 case this decode path always is; both prefill and decode are unbatched single-token
forwards here, confirmed by "prefill path: sequential" in every load banner above), canonical
otherwise — not both indiscriminately. That is a real, well-understood, one-mechanism fix, not a
speculative rewrite: the confirmed cause is a stale span-registration list, not a kernel or layout
problem.

**Superseded by Pass 1 below** (same day) — the fix identified here landed and was measured.

## Pass 1: the pager fix, measured (2026-08-24)

**The fix** (`decoder/moepaging.go`'s `addExpert`, `decoder/layerpaging.go`'s per-layer equivalent):
register only the span the M==1 decode kernel will actually read — row4 when `MappedSpanRow4`
returns non-empty, canonical otherwise — never both. Same existing `.giw` files (no re-prequant
needed; the bug was in the pager's runtime registration, not the on-disk format).

**Correctness re-confirmed, all three gates green:**
- `TestRow4GiwKind_gemma4_identicalToCanonical`: byte-identical, 385s (unchanged from before the fix
  — this gate doesn't touch the pager at all, resident-only).
- `TestRow4GiwKind_gemma4_pagedEviction`: byte-identical, 1GB budget against the now-correctly-halved
  **11.3 GB** pageable total (was 22.6 GB pre-fix) — `hits=4256 misses=3424 evictions=3059`, a real
  55.4% hit rate (was `hits=0` pre-fix at the same budget — the fix didn't just speed things up, it
  measurably changed cache behavior for the better, since the budget now buys real coverage instead
  of being split across two copies of everything).
- `TestRow4GiwKind_qwen35_pagedEviction`: byte-identical over 40 tokens. Pageable total correctly
  halved to **15.6 GB** (was 31.2 GB) — the original 6GB budget (sized for the old doubled total, a
  38.5% fraction) no longer forced eviction post-fix (`hits=12949 misses=2411`, zero evictions) and
  the test correctly FAILED rather than pass vacuously; re-run at 3GB (the same 19.2% fraction
  against the corrected total) restored real eviction, `hits=12939 misses=2421 evictions=307` —
  identical hit/miss counts to the pre-fix run at matched fraction, as expected (same logical access
  pattern; only the bytes moved per miss changed). Wall time (2269s) was longer than the pre-fix
  run at this config, an anomaly attributed to a genuinely busy machine during this specific run
  (another concurrent process was doing heavy work on the same box) rather than to the fix itself —
  flagged honestly rather than smoothed over, and superseded by the throughput cells below, which
  are the actual test of the fix and were measured with iostat cross-checked for the same
  contamination risk.

**Throughput: the regression is gone. The projection is not yet reached.**

| model | budget | kind-3 | kind-4 pre-fix | kind-4 post-fix | post-fix vs kind-3 | post-fix vs pre-fix |
|---|--:|--:|--:|--:|--:|--:|
| gemma4-26b | 4 GB | 1.128 | 0.747 (−34%) | **1.008** | −11% | **+35%** |
| gemma4-26b | 8 GB | 1.128 | 0.829 (−27%) | **1.121** | −0.6% (parity) | **+35%** |
| qwen3.5-35b-a3b | 12 GB | ~1.30 | 0.935 (−28%) | **1.088** | −16% | **+16%** |

All tok/s figures are 3-run averages, same method as the original acceptance run (steady-state,
`total_tokens / wall_time`). The 8GB gemma4 cell lands at kind-3 parity; the 4GB gemma4 and 12GB
qwen3.5-35b cells land noticeably closer but still measurably behind kind-3 — not the projected
1.3x GAIN over kind-3, just most of the way back from a 27-34% loss to roughly even. **This
machine was under real, acknowledged concurrent load from an unrelated process during this
session** (confirmed mid-run), which plausibly adds noise to both the pre- and post-fix numbers —
the direction and rough magnitude of the improvement (fixing a real, confirmed 2x-I/O-per-miss bug)
is not in doubt, but the exact residual gap to kind-3 (and whether it fully closes on a quiet
machine) is not yet a clean, noise-free number.

**Physical-bytes-read-per-token check: attempted, contaminated, not trustworthy this pass.**
`iostat -Id disk0`-bracketed byte counts around each benchmark cell were collected as planned, but
the same concurrent unrelated process contaminates disk0's cumulative counters — the deltas include
bytes neither this decode nor this box's other test activity can account for cleanly. Recorded
here as an honest negative rather than reported as a number: **re-run this specific check on a
quiet machine** before trusting a bytes/token figure, or (better, per the original ask) add a
proper counter to `aikit/mmap.SpanCache` itself (bytes actually WILLNEEDed, summed inside `Touch`)
so the check doesn't depend on external tool contamination at all — that would also make it a
permanent, durable assertion inside the heavy-gated tests rather than a one-off shell script,
closing the original "so I/O waste can't hide behind byte-identical gates again" ask properly.
**Done in the quiet re-measure below** (`aikit` v1.28.0's `SpanCache.AdvisedBytes`).

## Quiet-machine re-measure (2026-08-24): I/O is exonerated, a real compute-side gap remains

Built the durable byte-counter (`aikit` v1.28.0, `SpanCache.AdvisedBytes()` — cumulative bytes
passed to `MADV_WILLNEED` over every miss, wired into `expertPager.advisedBytes()` and a new
`assertAdvisedBytesSane` check in the heavy-gated tests), then re-ran everything with other
processes closed.

**How much noise there actually was, measured directly:** re-running kind-3 gemma4 at the exact
same 4GB config on a quiet machine gave **2.917 tok/s**, against the earlier reading of 1.128 —
**2.6x**, from machine load alone. Every absolute number in the sections above this one was taken
on a busy box (confirmed mid-session: an unrelated process was doing heavy concurrent work) and
should be read as directionally informative, not as calibrated absolutes.

**Correctness, re-confirmed with the new byte check:** both models' correctness gates re-ran clean
— gemma4 kind-3 vs. kind-4 still byte-identical (290.96s); gemma4 paged eviction byte-identical,
`hits=4256 misses=3424 evictions=3059`, **`AdvisedBytes` = 2.9 MB/miss against a 2.9 MB average
expert size — an exact 1.0000x ratio**; 35B paged eviction byte-identical over 40 tokens,
`hits=12939 misses=2421 evictions=307` (identical to every prior run at this config — fully
deterministic), **`AdvisedBytes` = 1.5 MB/miss against a 1.5 MB average expert — again exact**.
**The pager fix is complete and provably non-wasteful on both real models** — this is no longer a
hypothesis or an approximation, it is a direct assertion inside the test suite now.

**Throughput, quiet machine, both models fresh (kind-3 also re-measured fresh — its own
old numbers were equally noise-contaminated, including the previously-documented ~1.2-1.4 range
for the 35B):**

| model | kind | budget | tok/s (3-run avg) | vs kind-3 |
|---|---|---|--:|--:|
| gemma4-26b | 3 | 4 GB | ~~2.917~~ | — |
| gemma4-26b | 4 | 4 GB | ~~1.546~~ | ~~−47.0%~~ |
| gemma4-26b | 4 | 8 GB | ~~1.487~~ | ~~−49.0%~~ |
| qwen3.5-35b-a3b | 3 | 6 GB | **1.605** | — |
| qwen3.5-35b-a3b | 4 | 12 GB | **1.408** | **−12.3%** |

**STRUCK 2026-08-25 — withdrawn, not deleted. See "Supersession (2026-08-25)" below**, which
re-ran exactly these two gemma4 cells on a different day and found the gap gone (kind-4 FASTER at
both budgets) — including the kind-3 reference itself drifting 2.917→2.059 tok/s at the identical
4GB config with zero code change in between, the clean proof that this table's absolute numbers
were never as stable as a single quiet run made them look. The qwen3.5-35b-a3b row is untouched —
not re-measured this pass, no reason yet to doubt it.

**This is the opposite of what the earlier noisy readings suggested for gemma4.** On the busy
machine, gemma4's kind-4 gap looked smaller (−11% to −27%) and improving with budget. Quiet, it is
LARGER (−47% to −49%) and **budget-invariant** — doubling the cache from 4GB to 8GB (35.4% → 70.8%
residency) made no measurable difference. 35B's gap, by contrast, shrank on the quiet machine
(−12.3%, vs. −16% noisy) and stayed roughly consistent with its own fresh kind-3 baseline.

**What this rules out, definitively:**
- **I/O waste** — `AdvisedBytes` proves exactly the expected bytes are requested per miss, on both
  models, at the ratio a correctly-fixed pager should produce (1.0000x). The double-WILLNEED bug is
  fully closed; nothing hides behind the byte-identical gates on this axis anymore.
- **Cache/hit-rate effects** — gemma4's regression is identical within noise at both 35.4% and
  70.8% residency. If the gap were about miss rate, more cache would have closed it partway; it
  didn't move at all.

**What remains, unconfirmed:** a genuine compute-side cost specific to the row4 kernel running
against **mmap-paged** bytes rather than the **heap-resident** bytes the original 1.6-1.75x
figure was measured against (`docs/task-w4a8-neon-bandwidth.md`'s plumbing phase, GGUF/safetensors
streaming loaders — heap-backed, never paged). Candidate mechanism, not verified: cold TLB entries
for freshly-mapped pages vs. a long-lived heap allocation whose translation stays warm across a
whole decode session, even once the underlying bytes are page-cache-resident (a real distinction
this repo hasn't instrumented — perf counters, not spans/bytes, would be needed). **35B's much
smaller gap (−12.3% vs. gemma4's −47-49%) argues this isn't purely mechanical either** — the same
kernel, the same fix, two different real checkpoints, two very different outcomes; model shape
(gemma4's fused gate‖up experts vs. 35B's separate gate/up/down, different hidden dims, different
expert counts) likely matters and isn't understood yet.

~~**Superseded by the cold-touch investigation below**, which found the actual mechanism.~~
**STRUCK 2026-08-25 — the cold-touch investigation's own mechanism claim is itself withdrawn.
See "Supersession (2026-08-25)."**

## ~~The cold-touch investigation: found it (2026-08-24)~~ — WITHDRAWN 2026-08-25, see below

Three cheap, discriminating experiments, in the order the prior finding suggested them.

**1. Memory source (mmap vs. heap), warm data — ruled out.** Loaded gemma4 kind-4 resident, timed
24 sample experts' row4 kernel calls (mmap-backed), then heap-copied the same bytes in place
(`linalg.WrapInt4Row4` with fresh `[]byte`/`[]float32` allocations — same values, different backing
store) and re-timed. A warm-up control (repeating the mmap measurement before touching anything)
separated real memory-source effects from ordinary second-run speedup. Result: **+1.7%,
warm-up-adjusted — noise.** Identical bytes decode at the same speed whether mmap-paged or
heap-resident, once warm. (First attempt at this test heapified ALL ~7680 expert tensors at once
— ~14 GB of fresh heap on a 16 GB box — and drove the machine to 15/16 GB swap before being killed;
redesigned to heapify 24 samples and time direct kernel calls instead of full decode, keeping the
extra footprint under 100 MB.)

**2. Row4 vs. canonical kernel speed, gemma4's actual shapes, warm data — also ruled out (row4 is
faster, as originally claimed).** Called both free-function kernels (`linalg.MatmulBTW4A8Into`,
`linalg.MatmulBTW4A8Row4Into`) directly on the same real bytes, same activation, same shapes
(`gateUp`: 1408×2816 fused; `down`: 2816×704), order-alternated to cancel warm-up bias. Row4 is
**+57.6%** faster on `gateUp` and **+67.4%** faster on `down` — landing right in the claimed
1.6-1.75x range. The kernel is not slow on gemma4's specific dimensions; if anything it's exactly
as fast as promised, once warm.

**3. Cold-touch latency through the real production path — this is it.** Both prior experiments
reused the same small set of already-resident samples thousands of times, measuring warm
steady-state speed. Real decode touches ~240 DISTINCT experts per token under a real cache budget,
most of them NOT already resident. This experiment replicates that: loaded both kind-3 and kind-4
gemma4 paged (512 MB budget, forcing near-total misses), and for 187 distinct, never-before-touched
experts each, timed `expertPager.touch(key)` (the real pager call: mutex, LRU, `MADV_WILLNEED`)
immediately followed by the real `MatmulBTW4A8Into` call — the exact sequence real decode runs,
once per expert, no repeats.

| component | kind-3 (total, 187 touches) | kind-4 (total, 187 touches) | delta |
|---|--:|--:|--:|
| `touch()` (pager: mutex, LRU, WILLNEED) | 363.4ms | 385.4ms | +6.1% (noise) |
| `matmul` (the kernel itself) | 137.1ms | ~~231.7ms~~ | ~~**+69.0%**~~ |
| total per-touch | 2.68ms/touch | 3.30ms/touch | +23.3% |

**STRUCK 2026-08-25 — see "Supersession" below.** Not reproduced: 3/3 re-runs (2026-08-25, a
methodologically-corrected two-file real-pager sweep) found row4's kernel FASTER cold, not 69%
slower, on this exact shape.

~~**The pager is exonerated a second way** — `touch()` costs the same regardless of kind, confirming
(independently of `AdvisedBytes`) that the fix from the prior section is complete and the pager
itself isn't the source of anything. **The kernel is where it lives, and the direction inverted
from experiment 2**: row4 was +57-67% FASTER than canonical on warm, repeatedly-touched data: it is
**69% SLOWER on cold, first-touched data**. Two different, physically coherent regimes, not a
contradiction — but the mechanism is NOT the one first written here.~~

**Correction: "interleaving defeats the hardware prefetcher" does not survive reading the actual
assembly, and is retracted.** `dotW4A8SplitHalf4Row` (`aikit/linalg/dot_w4a8_arm64.s`) reads
`packed4` via four sequential `VLD1.P 16(R1)` post-increment loads per group — row0's chunk, then
row1's, then row2's, then row3's, each the NEXT 16 bytes in one contiguous buffer (that is what
"4-row-interleaved storage" means physically: the four rows' data for one group sit adjacent on
disk). The canonical kernel is also a linear scan, just through one row at a time instead of four.
Both are straight sequential address streams; a next-line hardware prefetcher has no obvious reason
to handle one better than the other on pattern-detection grounds. **The real mechanism is not yet
pinned down at the microarchitecture level** — the more likely candidate, unverified, is that row4
keeps 4 independent accumulator chains in flight per group specifically to hide WARM fold latency
(the kernel's own doc comment: "4 independent FMLA chains... come from 4 genuine distinct outputs"),
and that design may interact differently with a cold DRAM/page-cache fetch's much higher latency —
e.g. outstanding-load capacity, or how long a hardware prefetcher takes to ramp up on a fresh
stream — than a single accumulator chain does. This is a hypothesis, not a measured fact; confirming
it needs real microarchitectural profiling (cache-miss/TLB perf counters), which this pass did not
attempt. **Recorded honestly as an open mechanism, not chased further this pass** — a fix (explicit
software prefetch, e.g. `PRFM`, was floated as a candidate) would be premature against a mechanism
this uncertain; measure the actual bottleneck before writing an assembly change aimed at it.
~~**Superseded below — the real profiling this section deferred was run 2026-08-25 and confirmed the
4-accumulator-chain candidate.**~~ **STRUCK — that profiling section is itself withdrawn; see
"Supersession" below.**

~~**This also explains 35B's much smaller gap, tentatively.** 35B's separate gate/up/down experts and
different shapes may simply be less sensitive to whatever this cold-access cost actually is than
gemma4's fused gate‖up layout — consistent with (though not separately confirmed against) the two
models' very different regression sizes (gemma4 −47 to −49% vs. 35B −12.3%), but resting on the
same not-yet-pinned-down mechanism above.~~

~~**Where this leaves Pass 2 (giw v8, single representation):** the cold-vs-warm split is now
characterized precisely (the WHAT — 69% slower cold, budget-invariant), even though the WHY at the
microarchitecture level is still open. **v8's reconstruct-canonical-at-load design does not address
this either way** — the row4 bytes are still read cold, in their same on-disk layout, during paged
decode regardless of whether canonical also lives on disk; nothing about storing one representation
instead of two changes the row4 access pattern itself. Fixing the actual regression needs the
mechanism pinned down first (real profiling, not spans-and-bytes), then either a layout change (if
a row4 variant exists with better cold-read behavior without losing the warm-data win) or accepting
that row4 belongs only on the resident path (never dispatch it under `-stream-weights`) until one
does. v8's surviving justifications (disk halving, structurally eliminating the double-fetch bug
class, the no-bigger-files ruling) are independent of this finding and still stand on their own —
but its paged-decode story, if it has one, needs the mechanism understood first, not v8 itself.~~
**See "Supersession (2026-08-25)" for where this actually leaves Pass 2 and v8.**

## ~~Real hardware-counter profiling: mechanism confirmed (2026-08-25)~~ — INTERPRETATION WITHDRAWN, see below

The prior section's open question — front-end (prefetch/fetch-decode) or back-end (execution/memory
stall) — answered with real PMU counters, not inference from wall-clock alone.

**Method:** `xcrun xctrace record --template 'CPU Counters'` (macOS's Instruments CLI; no root/SIP
changes needed, confirmed working via a trivial `--launch -- /bin/echo hello` smoke test first). A
temporary standalone tool (`cmd/_coldprofile`, deleted after this pass, never committed) loads the
real `qwen3.5-35b-a3b-int4-row4.giw` resident and calls ONLY the canonical or ONLY the row4 free
kernel directly against a disjoint, never-before-touched slice of MoE expert tensors per run — gemma4
was the original model but its source checkpoint isn't available locally to rebuild a fresh row4
`.giw` after disk pressure forced its deletion twice this pass (recorded, not incidental — see below);
qwen3.5 was substituted since the mechanism under test is the kernel/format, not the specific model.

**A real methodological trap, found and fixed before trusting any number:** `xctrace --launch`
records the ENTIRE process lifetime, including this model's ~2-2.5 minute mmap/load phase, which
dominates wall-clock and therefore raw trace size regardless of how many experts get touched — the
first attempt (N=400 canonical) produced a **23 GB raw `.ktrace`** (found under `$TMPDIR`, not the
final `.trace` bundle, which stayed a few MB) and drove the machine to `ENOSPC` for the second time
this session; a same-day repeat at N=20 produced **19.2 GB**, proving trace size tracks wall-clock
duration, not touch count, and that shrinking N alone doesn't fix it. **Fix:** attach xctrace AFTER
load completes instead of launching fresh — the tool signals readiness via a marker file once loaded,
then polls for a second file before starting the touch loop (with a 4s hold on each side so the
attach/detach handshake isn't racing a touch loop that finishes in under a second). This cut every
subsequent trace to **~60-70 MB**, letting two independent replicates run safely.

**Result, N=20 experts × 2 replicates each, cold (never-before-touched), real production kernels:**

| metric | canonical (r1 / r2) | row4 (r1 / r2) |
|---|---|---|
| active on-core scheduling bursts | 255 / 261 | 560 / 487 (~2x more) |
| front-end delivery-bound (Instruction Delivery Bottleneck) | 20.1% / 21.0% | 15.3% / 14.5% |
| back-end processing-bound (Instruction Processing Bottleneck) | 35.1% / 31.6% | 43.0% / 40.5% |
| discarded (speculation waste) | 3.75% / 3.64% | 2.59% / 2.11% |
| useful | 41.1% / 43.8% | 39.1% / 43.0% |

(Apple's "CPU Bottlenecks" top-down categorization — `cycle`, `delivery`, `discarded`, `processing`,
`useful` — extracted from the `MetricTable` schema via `xcrun xctrace export --xpath`; this is a
derived front-end/back-end split, not a raw L1D-miss count directly, a real limit of this pass worth
stating plainly.)

**WITHDRAWN 2026-08-25 (interpretation only — the table above stands as data, unchanged):** the
paragraph originally here read the table as "row4 stalls harder in the back end, confirmed
mechanism." That reading is retracted. The burst-count and top-down-split NUMBERS above are real
and replicated (r1/r2 agree) — nothing about the measurement itself is in question. What's
withdrawn is the INTERPRETIVE LEAP from "this is what a cold row4 touch's scheduling looks like" to
"this is why row4 is slower cold." **A scheduling signature characterizes behavior, not cost** —
it was read through the assumption that row4 WAS measurably slower cold, an assumption the
"Supersession" section below found does not reproduce. With that assumption gone, the honest
reading of this same table is just: row4's cold access pattern produces more, shorter on-core
bursts and a different front-end/back-end split than canonical's — a real, measured DIFFERENCE in
scheduling shape, with no established link to which one is faster. Confirming that link (if one
exists) needs the counter data re-taken alongside a same-day throughput number that actually shows
a gap to explain — this data alone never should have been read as settling why by itself.

## Supersession (2026-08-25): the cold penalty does not reproduce, in either direction

A follow-on harness pass (direct instruction, not a `docs/prompts/` brief this time) set out to
build two mechanism-aimed remedies for the 69%-cold-penalty finding above — software prefetch and
chain/line de-sharing — and validate them against it. Both were built (aikit:
`dotW4A8SplitHalf4RowPrefetch`, `dotW4A8SplitHalf4RowDeshared`, correctness-proven bit-identical to
production row4 across 5 shapes, warm-intact per `TestW4A8Row4ColdFix_warmIntact`: row4 baseline
1.826x vs canonical, every PRFM distance exactly 1.000x, de-sharing 0.993x). But measuring them
against the cold penalty required re-touching the real gemma4 shape cold — and that re-measurement
is what withdraws the premise, not just tests the remedies.

**Held to the same standard as what it reverses.** This section does NOT declare "row4 is faster
cold, actually" as a new headline — that claim has exactly the evidentiary standing the 69%/47-49%
findings above had before today: a result from one measurement session on one machine. The
different-day-reproduction rule this incident itself argues for (below) applies to today's own
numbers first. **Status: reversed under re-measurement, pending different-day confirmation** — not
confirmed, not the new mechanism. The `cmd/prequant -row4` flag's cold-paging warning stays in the
help text unchanged until that confirmation actually runs; nothing here is a green light to
dispatch row4 under `-stream-weights`.

**Kernel-level, 3 independent runs, corrected methodology.** A single-file test design (canonical
bytes read via `wm.Int4()` off the SAME kind-4 file used for row4) was caught mid-pass as a real
bug: the pager's double-WILLNEED fix registers ONLY the row4 span for a kind-4 tensor, so
`touch()` never prefetches that tensor's canonical bytes at all — the real disk read silently
leaks into the "matmul" timing window, making canonical look ~9x slower than it is. Fixed by
loading canonical from its own dedicated kind-3 file (`gemma4-26b-int4.giw`) and row4 from the
kind-4 file (`gemma4-26b-int4-row4.giw`) as two separate model instances — exactly
`TestRow4_coldTouchLatency`'s original two-file design, which this bug never actually violated;
the violation was introduced fresh in this pass's own first draft.

| run | canonical kernel-only | row4 kernel-only | row4 vs canonical |
|---|--:|--:|--:|
| 1 (old kind-3 file, fresh row4 file) | 4.938 ms/call | 2.349 ms/call | **2.1x faster** |
| 2 (same pair, genuine repeat, `-count=1`) | 4.376 ms/call | 0.930 ms/call | **4.7x faster** |
| 3 (fresh-vs-fresh: both files built today, ruling out file-age/fragmentation as a confound) | 3.128 ms/call | 0.884 ms/call | **3.5x faster** |

Row4 is faster cold in all 3 runs, by a wide and noisy margin (2.1x-4.7x) — the opposite direction
from the 69%-slower finding, and the magnitude swings enough between runs 1 and 2 (same file pair,
same day) that "faster" is the only stable claim; the exact ratio is not.

**End-to-end, the number the whole investigation was actually chasing.** The kernel microbenchmark
was never the real question — it was the paged-decode throughput gap in the "Quiet-machine
re-measure" table above (struck). Re-ran the identical two cells today, same method (steady-state,
prompt+completion total tokens / wall time, 3-run average, `Load` + `Model.Generate` in-process —
adapted from the original `cmd/serve`+HTTP method to cut an HTTP-overhead variable the relative
comparison doesn't need):

| model | kind | budget | tok/s today | vs kind-3 today | tok/s 2026-08-24 | vs kind-3 then |
|---|---|--:|--:|--:|--:|--:|
| gemma4-26b | 3 | 4 GB | 2.059 | — | ~~2.917~~ | — |
| gemma4-26b | 4 | 4 GB | 3.062 | **+48.8%** | ~~1.546~~ | ~~−47.0%~~ |
| gemma4-26b | 4 | 8 GB | 2.618 | **+27.2%** | ~~1.487~~ | ~~−49.0%~~ |

Both budgets, kind-4 is faster end to end today, not 47-49% slower. Three independent layers — the
aikit kernel microbenchmark, the corrected real-pager kernel sweep, and this direct end-to-end
decode — now all point the same direction on the same day.

**The cleanest single piece of evidence is the kind-3 row against itself.** Same model, same
budget, same dispatch code, zero commits touching `dotW4A8`/`MatmulBTW4A8Into`'s canonical path in
between: **2.917 tok/s → 2.059 tok/s, a 29.4% drift with no code change.** Original measurement at
commit `4abd64679f5aa714fa28b978f6cf4a22f957fe18` (2026-08-24 20:24:02 -0700); today's re-measurement
at commit `32f854b52984ca58600280ba6f2a1b081a7f4b65` (2026-08-25 15:23:49 -0700). If the untouched
reference number moves 29% between two sessions on the same box, a 47-69% delta between two
DIFFERENT representations measured on DIFFERENT days is not distinguishable from that same noise
floor. This one row is the whole instability case, in miniature.

**Leading environmental candidate — listed, not asserted; checkable from this doc's own dates.**
This session spent 2026-08-24 in a genuine disk-space crisis (directly observed: free space down to
118 MB at one low point), and 2026-08-25's re-measurements all ran with 45-114 GB free, post-cleanup.
Near-full APFS volumes are known to read cold (non-cached) data measurably slower than the same
volume with headroom — SLC-cache exhaustion on the physical NAND and heavier allocation-metadata
overhead both bite hardest exactly when free space is scarce. Every "row4 is slower cold" reading in
this doc was taken during the near-full era; every "row4 is faster (or not slower) cold" reading
postdates the cleanup. That correlation is exact for the two dated measurements above and is offered
as the leading candidate explanation for the reversal — NOT confirmed, since isolating disk-fill
level as a controlled variable (fill the volume back to near-100% and re-run the SAME two files)
wasn't attempted this pass. It would also parsimoniously explain why two OTHER micro-findings in
this same window flip-flopped (next paragraph) without needing three separate unrelated causes.

**The meta-finding, and a proposed rule.** This is the THIRD single-machine micro-benchmark result
from this box in the current window to fail to reproduce: the W4A8 issue-width probe (marginal-FMA
ratio 1.11 "issue-limited" → re-measured 0.99-1.03, "does not reproduce," `dot_w4a8_arm64.s`'s own
comment on `dotW4A8FoldSDOT4Acc`), the sampler scratch-reuse buffer change (5-6% slower → ~5%
faster, same machine, opposite sessions), and this cold-penalty finding now. Three for three is a
pattern, not bad luck. **Proposed rule for this repo's own house discipline: any single-machine
micro-benchmark result that will drive more than a day of downstream implementation work must
reproduce on a different day — ideally a different thermal/uptime/disk-fill state — before any
remedy gets built against it.** This investigation followed every existing rule the campaign had
(quiet box, `AdvisedBytes` counters, real production paths, real checkpoints) and still spent real
engineering time building two working NEON kernels against a number that was never stable. That is
not a process failure under the old rules — it is a gap the old rules didn't cover, and this
incident is what writes the new one in.

**A third instance of "the instrument was the finding."** This campaign has now hit this shape three
times: the "interleaving defeats the hardware prefetcher" retraction above (the explanation, read
from an assumption, didn't survive reading the actual assembly — the instrument was the flawed
hypothesis, not a measurement); the double-WILLNEED pager bug (found only because building
`AdvisedBytes` as a durable counter revealed it — the instrument built to verify a fix is what
surfaced the real one); and now this pass's own single-file test design, which manufactured a fake
9x canonical slowdown before the fresh-vs-fresh control caught it. Each time, fixing the
measurement apparatus turned out to be more valuable than whatever the flawed apparatus originally
reported. Recorded at full value, not folded quietly into a "methodology note" — this is a repeating
shape worth naming.

**The remedies: built, validated, premise-void.** `dotW4A8SplitHalf4RowPrefetch` and
`dotW4A8SplitHalf4RowDeshared` (`aikit/linalg/dot_w4a8_arm64.s`), their Go wrappers
(`MatmulBTW4A8Row4PrefetchInto`, `MatmulBTW4A8Row4DesharedInto`, `RepackW4A8Row4Deshared*`,
`aikit/linalg/matmul_w4a8_row4_variants_arm64.go`), and their correctness tests
(`matmul_w4a8_row4_variants_arm64_test.go`) are committed to aikit, parked per this repo's
dead-end convention — not released, not wired into `WeightMat.MatmulBTW4A8Into`'s dispatch, no
production path reaches them. `TestW4A8Row4ColdFix_warmIntact` is kept as a standalone regression
guard (real value independent of this saga: it proves the remedies don't cost anything warm,
whatever happens to the cold question). The numbers this pass measured (bit-identical, warm-intact,
kernel-cold-faster-not-slower on this box today) are recorded here specifically so nobody rebuilds
either kernel blind against the withdrawn 69% number. If the different-day confirmation below
somehow finds a real cold penalty after all, both kernels are shovel-ready, not rebuilt from zero.

**What's next, gated on one thing.** A different-day (different session, ideally different
thermal/uptime/disk-fill state) re-run of exactly the two end-to-end cells above — `gemma4-26b-int4.giw`
(kind-3) and `gemma4-26b-int4-row4.giw` (kind-4) at 4GB and 8GB via
`TestGemma4EndToEndThroughput` (`decoder/gemma4_endtoend_throughput_test.go`, kept as a committed
regression tool for exactly this) — decides which way the ledger closes: gap confirmed gone (this
whole saga closes as measurement instability, full stop) or gap confirmed real (the mechanism hunt
reopens at the pager/span-interaction or kind-4 on-disk-geometry layer, with the row4 KERNEL itself
now formally exonerated by 3 corrected same-direction readings). **Both `gemma4-26b-int4.giw` and
`gemma4-26b-int4-row4.giw` are kept on disk specifically for that re-run** — the standing
end-of-session cleanup rule for campaign-generated `.giw` files is suspended for these two files
only, until the confirmation runs.

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
`internal/prequant/prequant.go:65-69`'s own comment — there is exactly one code path to fix, not
two). gemma4 shares the same carve-out (`decoder/gguf.go:1561`) but is NOT in scope here — it fits
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
`decoder/serialize.go:177-192`) — exactly how every other already-streaming family already behaves.
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
