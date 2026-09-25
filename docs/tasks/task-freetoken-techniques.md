# Task: FreeToken techniques — five leads to chase

> Scoping doc. Opened 2026-08-27 from a Cowork comparison of goinfer against FlashML's
> FreeToken (arXiv:2608.16157) — published at
> claude.ai/code/artifact/cd014aea-6e94-4f6d-b71a-196ba99b6bfa. Context: FreeToken is a
> Python, NVIDIA-only MoE serving engine (MIT-CSAIL/UC Berkeley team, 8.3k★) claiming
> 39.3 tok/s for a 35B-A3B model on an 8 GB laptop GPU and working decode up to 753B on
> a workstation GPU — well past goinfer's own tested MoE-offload ceiling (§C of the
> comparison). This doc checks five of its named techniques against goinfer's actual
> tree, not just its docs, and against goinfer's own prior art before proposing anything
> as new (CLAUDE.md's measurement-discipline rule).
>
> **Status: scoping only — nothing here is committed work.** Two leads turn out to
> already be identified next-levers blocked on a dependency; one turns out to be a
> named-and-deferred track goinfer already has a term for; one is a deliberate,
> reasoned decision worth revisiting only for a specific regime, not a bug; two are
> genuinely new. Read alongside `docs/completed/task-moe-streaming.md` and `docs/completed/qwen3_5_moe.md`,
> not instead of them.
>
> **Doc-review correction, 2026-09-13 (doc otherwise LIVE, corrections applied in place —
> see inline notes at each affected section, not restated here).** Two things this doc
> called settled moved since 2026-08-27: (1) the GPU-resident session-skip "not a gap" note
> was reversed six days after this doc opened — `3358e6b` shipped general resident prefix
> reuse (attention-only families; hybrid/recurrent families still excluded, which folds into
> Lead 1); (2) Lead 5 ("genuinely new... track, don't start") has since started —
> `docs/task-l01-hybrid-moe-cpu-gpu.md` is the live design/build record, with real code
> landed 2026-09-10 (`cuda/l01_cpu_offload.go`, correctness-verified, not yet wired in).
> Leads 1, 3, and 4 were checked against the current tree and stand as written.

> **Doc-review correction, 2026-09-22 (doc otherwise LIVE, corrections applied in place —
> see inline notes at each affected section).** `docs/completed/task-moe-streaming.md`, this
> doc's own citation target for Leads 2 and 4, has since been archived to
> `docs/completed/task-moe-streaming.md` — its remaining levers landed. Two of the five
> leads resolved since the 09-13 pass: (1) **Lead 2 (async-H2D overlap) SHIPPED
> 2026-09-22** as the C′ compute/DMA overlap (`GOINFER_MOE_DMA_OVERLAP`, aikit
> `gpu.Event`/`Queue.UploadAsyncAt`), 1.27× on the real 26B, plus its own "refinement"
> (prefill preloading a layer's full expert set) shipped 2026-09-21 as Lever 4 /
> R11(b)-P20 (expert-major prefill batching), 2.66×–2.26× on the real 26B — see below.
> (2) **Lead 5 (bandwidth-adaptive CPU/GPU co-execution) ran its own pre-registered
> funding measurement 2026-09-21 and was KILLED**, not merely "not yet a funding
> decision": CPU-offloaded experts permanently poison the C′ cache, an ~11× regression
> (`docs/measurements/l01-funding-cell-2026-09-21.md`) — `task-l01-hybrid-moe-cpu-gpu.md`
> itself already records this; this doc's own summary did not. Leads 1, 3, and 4 were
> re-checked against the current tree (`cuda/resident.go`'s `mapBytes`, `MoECacheSlots`)
> and stand exactly as written in 2026-09-13 — genuinely unbuilt, unmeasured, unowned by
> any live doc.

| Lead | Status | Priority |
|---|---|---|
| 1. State-checkpoint KV reuse for hybrid/recurrent models | **checked 2026-09-22: still unblocked, no real agent-loop traffic exists to spike against** — orphaned | high |
| 2. Async-H2D overlap for the CUDA expert cache | **SHIPPED 2026-09-22** (`GOINFER_MOE_DMA_OVERLAP`, 1.27× real 26B) — see below | done |
| 3. Pin the CUDA expert-stack buffer after filling, not before | **measured 2026-09-22, PARKED by the rule, overridden to DEFAULT ON**: real decision measurement on the real 26B, 1.105× (10.5%), below the 15% ship bar but 5/5 winning trials, zero numerics risk — see below | done (default) |
| 4. Pool the CUDA expert cache globally instead of per-layer | **measured, PARKED 2026-09-22**: unevenness real (2 of 30 layers), bounded upside +0.4 pp hit rate, too small to build for — see below | medium → parked |
| 5. Bandwidth-adaptive CPU/GPU co-execution | **KILLED 2026-09-21**, real hardware, ~11× regression (cache-poisoning) — see below | closed |
| — GPU-resident session-skip | **corrected 2026-09-13: fixed in general, not just "not a gap"** — see note below | Lead 1 is now the live remainder |

---

## Lead 1 — state-checkpoint KV reuse for hybrid/recurrent models

**Status: goinfer already named this.** `docs/completed/qwen3_5_moe.md`'s "Hybrid cache (decision:
correctness-first)" section states plainly, for `qwen3_5_moe`'s 30 DeltaNet layers + 10
full-attention layers: because the recurrent state is not position-truncatable, cross-call
prefix KV reuse (`Session`, v0.3.0) **falls back to full recompute** on cold or non-extending
prefixes (`docs/completed/qwen3_5_moe.md:132` — the resident path's exact-strict-extension case
was fixed 2026-09-03, R-01 phase 0; the general fallback for anything else stands), and
"optimizing those for hybrid models (state checkpoints) is a later track." That's the same shape
as FreeToken's "semantic anchor checkpoints": a compressed recurrent state can't be sliced like
an attention KV cache, so it needs its own full-state checkpoint instead of a truncate-and-reuse.

What FreeToken adds beyond the one-line deferral: checkpoints aren't taken at arbitrary
positions — they're anchored to special-token boundaries (thinking segments, tool calls
and their outputs, turn boundaries), indexed in a prefix tree, and on a context edit the
system restores from the deepest checkpoint whose position still survives. Attention
layers reuse KV via the same tree; recurrent layers restore the nearest surviving
full-state checkpoint and recompute only the suffix.

**Proposed shape for goinfer.** Extend `Session`/`sessionLRU` (`decoder/session.go`)
with a parallel checkpoint store for the `DeltaState{S, convState}` half of a hybrid
sequence's cache, keyed the same way the existing prefix match is keyed, but only at
token positions that line up with a message/tool-call/turn boundary — goinfer already
renders and parses tool calls per-family (`chat/tools.go`), so boundary positions are
knowable at generation time, not something to invent. Full recompute stays the fallback
when no checkpoint survives, exactly as today.

**Sizing: unscoped.** First step is probably a spike measuring how often a real agent
loop (the `demo/chat` coding assistant, or a dsh-style tool-using session — see
`docs/scoping-dsh-goinfer.md`) actually re-sends a boundary-aligned prefix versus one
that's genuinely edited mid-context. If edits are rare, the win is large and cheap; if
agent frameworks routinely truncate or reorder history, the checkpoint hit rate might
disappoint the way `scoping-dsh-goinfer.md` already flags "context stuffing" as an open
friction point.

**Priority: high.** Biggest of the five, already on goinfer's own deferred list, and it
lands on the qwen3.5/3.6-class models that are also the ones losing hardest to
FreeToken on raw throughput (§C of the comparison) — a cheaper agentic re-turn is worth
more there than anywhere else in the tree.

**Checked, 2026-09-22: still unblocked, and still nothing to measure it against.** The
"first step" this lead names — a spike measuring how often a real agent loop re-sends a
boundary-aligned prefix versus edits history mid-context — needs real agent-loop
traffic. None exists in this repo to analyze: `demo/chat` is a client binary with no
session logs, and `docs/scoping-dsh-goinfer.md`'s "context stuffing" mention is a single
passing reference, not a dataset. Building a synthetic traffic generator to produce a
number would be inventing the answer rather than measuring it — the same trap this
repo's own measurement discipline warns against, just on the input side instead of the
output side. Genuinely open, genuinely unblocked, and needs either real traffic capture
from an actual agent session or someone to say what proxy would count as evidence
before a spike can start.

---

## Lead 2 — async-H2D overlap for the CUDA expert cache (already your own next lever)

**Status: already scoped, not started.** `docs/completed/task-moe-streaming.md`'s "§C′ — VRAM
expert cache" section (line 356 as of 2026-09-13; the doc has grown since this lead was
written) ships step 2 (the real LRU cache, `GOINFER_MOE_CACHE_SLOTS`) on **synchronous**
H2D DMA, and says outright that the next lever is a gocudrv v0.3.0 bump for async-H2D
overlap of the miss DMAs with compute, to collapse the remaining per-token bytes toward
the ~50 MB estimate. FreeToken's "double-buffered prefill streaming" — compute layer *l*
from one buffer while a transfer stream loads layer *l*+1's experts into the other — is a
second, independent data point that this class of overlap is worth the dependency bump,
from a team with real throughput numbers to show for it.

**Sizing caveat, 2026-09-13.** The same `docs/completed/task-moe-streaming.md` §C′ also carries a later,
more precise "Production-config decomposition" (dated 2026-08-03, so already on record
when this lead was written, just not folded in here): at the current 38-slot config
(89.1% LRU hit), miss-DMA is measured at 4.28 ms/tok against a ~29.3 ms/tok forward total
— so async-H2D overlap has at most ≤4.3 ms/tok to hide, not the open-ended "collapse
toward ~50 MB" framing above might suggest. Still worth building (a free ~15% at this
config, more as slot count or model size grows the miss rate), just a bounded win against
the current cache, not a second multiplier on top of it.

**One refinement worth folding in when this gets picked up.** FreeToken's description
implies the *prefill* case specifically preloads the complete expert set for the next
layer, not just the routed ones — plausible, since a prefill chunk of 128+ tokens
touches most experts anyway, so guessing "all of them" for the next layer beats waiting
on this layer's routing to know which ones. goinfer's `layerPager`
(`decoder/layerpaging.go`) already does exactly this kind of windowed, deterministic,
routing-independent lookahead for dense weights — it's the pattern to copy for the
expert case, not something to invent. Decode can stay reactive (one token, routing is
cheap and known immediately); the win is specifically in the prefill-chunk case, which
is also where Lever 4 (`docs/completed/task-moe-streaming.md:226`, expert-major prefill batching)
already lives — these two probably want to be scoped together rather than as separate
patches.

**Depends on:** the gocudrv v0.3.0 bump `docs/completed/task-moe-streaming.md` already flags as a
prerequisite; probably sequences after Lever 4.

**Priority: high** — smallest conceptual gap between what goinfer already built and
what this needs, and it's the direct lever under the 11.33 → 16.98 tok/s numbers in
the comparison.

**Correction, 2026-09-22: SHIPPED, both halves.** The dependency named above turned out
unnecessary: aikit was already past v0.3.0 (`gpu/v0.33.2` by the time this landed), and the
overlap didn't need a version bump so much as new primitives on top of what was there
(`Event`, `Queue.UploadAsyncAt`, `Queue.ZeroAsync`). `GOINFER_MOE_DMA_OVERLAP` (default on)
records the router's completion on an event instead of draining the stream, DMAs cache
misses on a second queue, and lets independent GPU work (the dense branch on hybrid
layers, hit-expert ranks) run underneath — bit-identical by construction, **1.27× on the
real 26B (30.4 → 38.7 tok/s)**, paired ABBA, 8/8 pairs 1.269–1.288×
(`docs/measurements/moe-streaming-decode-overlap-ceiling-2026-09-22.md`). The "refinement"
above — preload a prefill chunk's full expert set rather than wait on per-row routing —
shipped the day before as Lever 4 / R11(b)-P20: route every row first, admit each distinct
expert once per chunk, **2.66×/2.50×/2.39×/2.26× at M=512/2048/4096/8012 on the real 26B**
(`docs/measurements/p20-expert-major-m26-2026-09-21.md`), default on
(`GOINFER_CUDA_MOE_EXPERT_MAJOR`). Both leads inside this one are closed.

---

## Lead 3 — pin the CUDA expert-stack buffer after filling, not before

**Status: unverified until this pass, now fairly precise.** `cuda/resident.go`'s
`mapBytes` — the function that stages the full ~11.4 GB expert stack into pinned host
memory as the C′ DMA source — allocates the pinned buffer first
(`r.dev.NewMappedHostBuffer(len(src))`) and copies into it second
(`copy(mb.Bytes(), src)`). FreeToken's FTW loader does the reverse: populate ordinary
(pageable) host memory with parallel direct I/O first, and pin only afterward —
pinning empty pages forces the OS to zero-fill and fault them in; pinning
already-populated pages skips that.

This isn't a guess about where the cost lives: `docs/completed/task-moe-streaming.md`'s §C′ already
attributes the real 26B model's **4m49s load time** to "the 11.4 GB pinned alloc +
copy," in those words. Whether reordering the two operations actually moves that
number depends on details this doc can't see from here — whether `NewMappedHostBuffer`'s
underlying `cuMemAllocHost` zero-fills, and whether aikit's `gpu` package exposes a
populate-then-pin path at all — but it's a cheap, narrow experiment against an
already-known, already-measured cost, not a redesign.

**Priority: medium** — small blast radius, but needs an aikit-side primitive that may
not exist yet; check `gpu.NewMappedHostBuffer`'s actual implementation before assuming
this is a goinfer-side change.

**Correction, 2026-09-22: measured twice — a microbenchmark, then the real decision
measurement — and PARKED.** The primitive already existed one level below aikit's own
wrapper (gocudrv's `RegisterHost`); a standalone microbenchmark at the real 11.4 GB
scale found populate-then-register winning every trial, 1.33×–4.46×, with a named
mechanism (pinning already-resident pages needs no page-cache reclaim; allocating fresh
pinned memory does). That was flagged as not production-representative (repeated
in-process trials, no cache reset between them), so the follow-up built it properly —
`aikit Device.RegisterMappedHostBuffer` + `Queue.UploadAsyncAtFrom` (additive; needed
because the C′ DMA overlap's `.Host()` call returns nil for this new origin), wired
behind `GOINFER_MOE_PIN_REGISTER` in `cuda/resident.go`'s `mapBytes`, every existing C′
correctness gate re-run with it on — and ran the real decision measurement: **one 26B
load per sample, fresh process each time, ABBA, 4 vs 5 trials: 1.105× — 10.5% faster.**
Against the rule pre-registered before either measurement (ship ≥15%, park 5–15%, kill
<5%): **PARK.** New never lost a trial, but the effect is diluted inside a ~46s total
load dominated by disk read and tensor decode, not the ~11.4 GB pin step alone — a
different regime from the isolated microbenchmark. **Owner override, same day: default
flipped ON anyway** (same shape as R2/R8 elsewhere in this campaign) — 5/5 winning
trials and zero numerics risk (the DMA source's bytes are identical either way) made
this a real win worth taking despite missing its own bar. `GOINFER_MOE_PIN_REGISTER=0`
restores the measured do-nothing arm. `docs/measurements/lead3-pin-order-2026-09-22.md`.

---

## Lead 4 — pool the CUDA expert cache globally instead of per layer

**Status: half-true already.** The CPU-side mmap pager (`decoder/moepaging.go`'s
`expertPager`) already pools every layer's experts into one shared `SpanCache` with
frequency-aware (`EvictLeastRecent`) eviction — confirmed by reading `newExpertPager`:
the member-collection loop walks every layer in `w.Layers`, and every layer's experts
land in the same cache. That already matches what FreeToken calls a "shared LRU
residency space."

The CUDA path doesn't: `GOINFER_MOE_CACHE_SLOTS` is a **per-layer** slot count
(`decoder/model.go:302`, `internal/serveapp/main.go:243`), auto-capped to free VRAM at
load. `docs/completed/task-moe-streaming.md`'s §C′ never discusses pooling it across layers — every
mention of slot budgeting there is per-layer. If expert "hotness" is uneven across
layers (plausible, and apparently never measured either way — the doc's own hit-rate
findings are all reported per-run, not per-layer), a fixed depth either wastes slots on
a cold layer or starves a hot one, and the 11.33 / 16.98 tok/s numbers in the
comparison might move for free: no new VRAM, just a different split of the same budget.

**First step:** instrument per-layer hit rate at a fixed total budget (sum of the
current per-layer slots) on the real 26B run, the same way the 77.5%-hit-rate finding
in `docs/completed/task-moe-streaming.md` was produced, and see whether the hit-rate distribution
across layers is actually uneven before building anything.

**Priority: medium** — cheap to measure, uncertain payoff until measured. Exactly the
kind of claim CLAUDE.md's measurement discipline says to check before building:
"measure don't assume."

**Correction, 2026-09-22: measured, PARKED — real unevenness, bounded upside too small to
build for.** `PerLayerCacheStatsForTest` (new, `cuda/testhooks.go`) + `TestMoEPerLayerHitRate`
ran the real 26B at today's shipped per-layer budget (29 slots, auto-capped). All 30 layers
are MoE; two of them (layers 10, 11) sit ~1.9 pp below the 96.91% mean, the rest cluster within
~1 pp of each other — real, but concentrated in two outliers, not a smooth gradient. Computed
upside if those two were pulled to the other 28's mean via slots taken from elsewhere in the
same total budget: **~5.7% fewer misses, hit rate 96.91% → ~97.31% (+0.4 pp)** — the ceiling
of what a global pool buys here, before the cost of building and validating a cross-layer LRU.
That ceiling is smaller than it would have been a day ago: `GOINFER_MOE_DMA_OVERLAP` (Lead 2,
shipped the same day) already hides most of a miss's cost under concurrent GPU work, so a
0.4 pp hit-rate gain now buys less wall-clock than it would have pre-overlap. One run, not
paired — the effect is small enough that it is worth confirming before trusting the exact
number, but not worth building around at this size either way.
(`docs/measurements/lead4-perlayer-hitrate-2026-09-22.md`.) **Parked, not killed**: revisit
if a different model/config shows larger unevenness, or if per-layer LRU maintenance itself
becomes a measured cost worth removing on its own terms.

---

## Lead 5 — bandwidth-adaptive CPU/GPU co-execution

**Status: genuinely new, no equivalent in goinfer today.** FreeToken's split —
q⋆ ≈ m·(B_P / B_H), running some routed experts on the CPU while streaming others to
the GPU concurrently, sized to measured PCIe bandwidth versus CPU MoE-kernel
throughput — has no counterpart here. Every current path is GPU-resident-only (§C′) or
CPU-only (the mmap pager) per architecture; nothing blends the two live.

This is the biggest architectural lift of the five, and the one most likely to collide
with the "ARCHITECTURAL COST" note already on record in `docs/completed/task-moe-streaming.md`'s §C′:
the on-device router exists specifically to avoid a host readback that would stall the
pipeline, and a live CPU/GPU split would need to reintroduce some form of host-visible
routing decision — the exact thing that section says the current design exists to
avoid. Worth a scoping pass of its own before any code, not a quick add-on to leads 1–4.

**Priority: low for now** — track it, don't start it, until leads 1–2 land and there's
a clearer read on how much headroom is actually left on the table.

**Correction, 2026-09-13: this has since started, ahead of the "track, don't start" call
above.** `docs/audit-2026-09-02.md` L-01 borrows this lead's mechanism directly (and notes
that "the architectural objection recorded in `docs/task-freetoken-techniques.md` Lead 5 is
smaller than when written," since C′ already pays a host-visible routing readback every
layer). `docs/task-l01-hybrid-moe-cpu-gpu.md` is the live design/build record: an isolated
microbenchmark on the real CUDA box's host CPU found sending every missed expert to CPU, run
in parallel, beats today's GPU-compute-plus-DMA cost at every measured miss count (1.32× at
m=1 to 3.40× at m=8); the aggregate host-CPU occupancy and multi-tenant-contention questions
are both resolved in favour of the simpler "send everything" mechanism; and the partial-sum
merge path is sketched against the real kernel signatures. **Real code landed 2026-09-10**
(`cuda/l01_cpu_offload.go`): the pinned-host expert extraction, correctness-verified against
an independent ground truth (cosine 0.99964–0.9999999 across nine expert/input combinations)
— not yet wired into `loadRoutedExperts`, no async overlap or merge kernel yet, and not a
funding decision until the real concurrent prototype and the pre-registered paired measurement
on Qwen3.6-35B-A3B (fund ≥1.3×, park <1.15×) run. `docs/task-l01-hybrid-moe-cpu-gpu.md`, not
this doc, is the one to read for Lead 5's current state.

### Pre-registered risk: Lead 5 and the speculation program may be antagonistic

**Nobody had written down that these two collide, and they do.** Lead 5 and the spec
program (`docs/spec/`) are planned independently; this note exists so a hybrid design
starts with the constraint rather than discovering it.

Measured 2026-09-02 (`docs/measurements/spec-x-pager-2026-09-02.md`, pre-registered;
prompted by a field report of a hybrid-split 176B MoE whose throughput collapsed when
drafting was enabled — that report is the origin of the question and no number from it is
used). The finding on the paths that exist today:

- **The reported mechanism does not occur here.** A width-K verify does not present K
  positions' routing to the pager at once, so it cannot overflow a slot budget tuned on
  decode traffic. The pager's demand instrument reads exactly topK distinct experts per
  staging event in every configuration, and the behaviour is identical at 16/32/64
  slots/layer.
- **It cannot occur, for a more basic reason: on CUDA nothing that needs the pager is
  allowed to speculate at all.** Four independent gates (MoE batched-verify decline,
  recurrent-rollback refusal, windowed-rollback refusal, and MLA having no CUDA resident
  path) between them refuse every model large enough to need expert streaming. The
  intersection is empty.

**What that means for Lead 5 specifically, and it is not "no risk".** The absence of the
effect today is an artifact of speculation being unavailable, not of the two mechanisms
being compatible. Lead 5 would change exactly the terms that produce the absence:

1. A CPU/GPU split needs host-visible routing, which is the "ARCHITECTURAL COST" already on
   record in `docs/completed/task-moe-streaming.md` §C′ — and a host-visible router is also the thing that
   would let a batched MoE verify exist. Building one removes the first gate, and the
   pager-thrash question then becomes live for the first time, **untested**, because it has
   never been possible to test it.
2. `thetaFor` (`decoder/spec_adaptive.go`) keys the adaptive controller's cost constant on
   BACKEND NAME, not on whether the verify is actually batched. A MoE on CUDA already gets
   0.251 — measured on dense targets — while its verify is a per-token loop. That is
   currently harmless only because the gates fire first. Any Lead 5 that enables speculation
   on a split-execution MoE inherits a cost model calibrated on the wrong arch class, and
   the error direction is over-drafting.

**So: if Lead 5 proceeds, it must pre-register both questions before the first number** —
whether speculation has to be disabled under split execution, and whether the slot budget
must be renegotiated per verify width. `cuda/spec_pager_interaction_test.go` and the demand
instrument (`PagerStageStatsForTest`) are in place to answer them; they currently report a
refusal, which is the right answer for today and the wrong one to carry forward.

**Correction, 2026-09-22: Lead 5's funding measurement ran and it was KILLED, not a still-
open funding decision.** `task-l01-hybrid-moe-cpu-gpu.md`'s §9 remainder — the goroutine-
per-expert parallel CPU compute, the merge kernel, async overlap alongside the GPU's own
hit-path launches — was built (`cuda/l01_cpu_offload.go`, `GOINFER_CUDA_L01_CPU_OFFLOAD`,
bit-identical per `TestL01_e2eDecode_matchesBaseline`) and run against the pre-registered
bar this doc itself names (fund ≥1.3×, park <1.15×) on the real target: Qwen3.6-35B-A3B,
CUDA, RTX 2070 SUPER. **Result: 0.083–0.094× — an ~11× regression, not a win at any
margin.** Mechanism: CPU-offloaded experts permanently poison the C′ cache — the offloaded
miss never lands in a VRAM slot, so every later routing to that expert misses again,
collapsing the hit rate to 0% (`docs/measurements/l01-funding-cell-2026-09-21.md`). The
"speculation-antagonism" risk registered above is now moot for this specific mechanism (it
was killed before reaching a design where that collision could occur), but the general
point — a CPU/GPU split needs host-visible routing, which is also what a batched MoE
verify needs — stands for any *future*, differently-shaped attempt at this lead.

---

## Revisit, don't "fix": GPU-resident models skip session/prefix reuse

**Not a lead — a note, and a correction to how the original comparison framed it.**
The first draft of the goinfer-vs-FreeToken comparison called this a gap. It isn't
one, as of when this doc was opened (2026-08-27): `README.md:828-831` at the time stated
"the resident decode path is fast enough that the per-request session optimization isn't
worth it. The OpenAI API is stateless [clients resend the whole conversation], so this is
a throughput trade, not a correctness change." That was a considered decision, not an
oversight, and `docs/legacy-benchmarks.md:361` and `docs/scoping-dsh-goinfer.md:32` both
restated it as a known, documented trade-off rather than a bug.

**Correction, 2026-09-13: the decision was reversed six days after this doc opened, and the
quoted README passage no longer exists at that path.** `3358e6b` (2026-09-02) shipped
resident prefix reuse on the resident KV path itself (`decoder/resident_reuse.go`) — the
"deliberate trade" above was traded back once it was cheap enough to build. Warm-turn TTFT
on the resident path is now 7–8 ms against ~1700–1900 ms cold (`docs/benchmarks.md` §B10,
2026-09-02) — roughly 250× — and `docs/roadmap.md`'s own "Superseded — June claims" table
lists this exact claim as retired: `the resident path is "stateless (no prefix-reuse)"` →
`resident prefix reuse shipped 2026-09-02 (3358e6b); agent turn 3 on the 1.5B 9.13 s → 0.42 s`.
The quoted README passage moved into `docs/completed/gpu-residency-coverage-2026-06.md:280`
as an archived record of the trade as it stood before the fix; README.md itself is 377 lines
today and doesn't contain it. `docs/scoping-dsh-goinfer.md:32` was already corrected in place
(it now says "GPU-resident models re-prefill each turn (no prefix-KV reuse on the resident
path)" only as one of "the two known trade-offs" the staged/CPU path avoids — check it in
context, that framing is about `cmd/serve`'s separate session cache, not the resident path's
own KV reuse).

**What's actually still true, and it's the same regime this note already named.** The fix is
gated on `hasRecurrentState()`: hybrid/recurrent families (Gated DeltaNet, Mamba-2 — the
qwen3.5/3.6-MoE class, also `docs/legacy-benchmarks.md:369`'s "serve caveat" box, itself
marked "no longer true" for the general case) don't get the same reuse — they got the
narrower exact-strict-extension rule instead (Lead 1, above; `docs/completed/qwen3_5_moe.md`'s
2026-09-12 correction). So the regime this note flagged as worth reopening — "a large, slow
MoE model... in a long, growing agentic conversation, where re-prefilling the entire history
every turn costs real wall-clock" — is exactly where the fix does NOT reach yet, and it's
also exactly Lead 1's open work, not a separate question. This note's job is now folded into
Lead 1; nothing further to revisit here on its own.

---

## Sources

- The comparison this opened from: claude.ai/code/artifact/cd014aea-6e94-4f6d-b71a-196ba99b6bfa
- `docs/completed/qwen3_5_moe.md` — "Hybrid cache (decision: correctness-first)"; `:132` for the
  fallback-to-full-recompute line
- `docs/completed/task-moe-streaming.md` — §C′ (`:346`), Lever 1 (`:107`), Lever 3 (`:185`),
  Lever 4 (`:226`)
- `README.md` — the Gemma-4 26B slot table (the GPU-resident session-skip quote this doc
  originally cited at `:828-831` has since moved — see the 2026-09-13 correction inline)
- `docs/legacy-benchmarks.md:361,366,369`, `docs/scoping-dsh-goinfer.md:32`
- `decoder/moepaging.go`, `decoder/layerpaging.go`, `decoder/session.go`,
  `cuda/resident.go` (`mapBytes`)
- FreeToken: arXiv:2608.16157; github.com/FlashML-org/FreeToken
- Added 2026-09-13 doc-review correction: `docs/task-l01-hybrid-moe-cpu-gpu.md` (Lead 5's
  live state), `docs/audit-2026-09-02.md` L-01, `docs/roadmap.md` ("Superseded — June
  claims" table), `docs/benchmarks.md` §B10 (warm/cold TTFT), `docs/completed/
  gpu-residency-coverage-2026-06.md:280` (where the old README quote moved),
  `decoder/resident_reuse.go`, `docs/completed/qwen3_5_moe.md` (2026-09-12 correction)

## Next step

Nothing here has been opened into a `docs/prompts/` brief yet, per the usual
Cowork-drafts / vscode-claude-executes split — this is the scoping pass, not the work
order. Say which lead(s) to open first and a brief can follow the same shape as
`docs/completed/zeno-compare-phase0.md` (archived 2026-09-13, its own scope complete).

**As of 2026-09-22: Leads 3 and 4 both ran their full measurement to a decision and
landed on PARK — real, small, direction-consistent effects that don't clear their own
pre-registered bars.** Only Lead 1 remains genuinely open, and it is blocked on data
that doesn't exist in this repo, not on anyone's time. Leads 2 and 5 are closed
(shipped and killed respectively). Nothing here is unblocked-and-unmeasured any more.

<!-- doc-reviewed: 2026-09-22 -->
