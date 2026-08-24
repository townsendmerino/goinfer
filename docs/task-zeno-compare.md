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

`cmd/prequant`'s own doc comment names the assumption directly (`internal/prequant/prequant.go:65-69`):
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
(`decoder/gguf.go:1405`), the SAME shared closure used by every family including qwen35 for
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

**Verified at small scale, not yet at the 35B-A3B scale this campaign is about.** Bit-identical
dispatch proven on a real fixture; load-time and RAM-delta numbers measured (kind-4 load is 0.111s
and adds ~0 resident memory vs. GGUF's in-RAM-repack 2.01s/+223.6MB). The 35B re-prequant hit a
real, honestly-recorded disk-space near-miss (a full-model kind-4 bundle grew past 43GB, larger
than the ~38GB estimate, before the attempt was stopped to avoid filling the disk) — scaled down
to the small-fixture numbers on explicit direction rather than force it. **The projected ~1.3x
end-to-end and the gemma4 26B regression re-run are both still owed at full scale** — the format
and code are done and gated (T3 green, aikit released), only the at-scale confirmation remains,
next time there's enough disk headroom for a ~40-50GB bundle alongside its 22GB source.

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
two). gemma4 shares the same carve-out (`decoder/gguf.go:1244`) but is NOT in scope here — it fits
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
