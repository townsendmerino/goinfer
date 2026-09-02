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
> genuinely new. Read alongside `docs/task-moe-streaming.md` and `docs/qwen3_5_moe.md`,
> not instead of them.

| Lead | Status | Priority |
|---|---|---|
| 1. State-checkpoint KV reuse for hybrid/recurrent models | named, deferred (`docs/qwen3_5_moe.md`) | high |
| 2. Async-H2D overlap for the CUDA expert cache | already scoped (`task-moe-streaming.md` §C′) | high |
| 3. Pin the CUDA expert-stack buffer after filling, not before | unverified, now precise | medium |
| 4. Pool the CUDA expert cache globally instead of per-layer | half-true already (CPU path has it) | medium |
| 5. Bandwidth-adaptive CPU/GPU co-execution | genuinely new | low — track, don't start |
| — GPU-resident session-skip | **not a gap** — deliberate decision, see note below | revisit only for slow MoE |

---

## Lead 1 — state-checkpoint KV reuse for hybrid/recurrent models

**Status: goinfer already named this.** `docs/qwen3_5_moe.md`'s "Hybrid cache (decision:
correctness-first)" section states plainly, for `qwen3_5_moe`'s 30 DeltaNet layers + 10
full-attention layers: because the recurrent state is not position-truncatable,
cross-call prefix KV reuse (`Session`, v0.3.0) **falls back to full recompute**
(`docs/qwen3_5_moe.md:115`), and "optimizing those for hybrid models (state
checkpoints) is a later track." That's the same shape as FreeToken's "semantic anchor
checkpoints": a compressed recurrent state can't be sliced like an attention KV cache,
so it needs its own full-state checkpoint instead of a truncate-and-reuse.

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

---

## Lead 2 — async-H2D overlap for the CUDA expert cache (already your own next lever)

**Status: already scoped, not started.** `docs/task-moe-streaming.md`'s "§C′ — VRAM
expert cache" section (line 346) ships step 2 (the real LRU cache,
`GOINFER_MOE_CACHE_SLOTS`) on **synchronous** H2D DMA, and says outright that the next
lever is a gocudrv v0.3.0 bump for async-H2D overlap of the miss DMAs with compute,
to collapse the remaining per-token bytes toward the ~50 MB estimate. FreeToken's
"double-buffered prefill streaming" — compute layer *l* from one buffer while a
transfer stream loads layer *l*+1's experts into the other — is a second, independent
data point that this class of overlap is worth the dependency bump, from a team with
real throughput numbers to show for it.

**One refinement worth folding in when this gets picked up.** FreeToken's description
implies the *prefill* case specifically preloads the complete expert set for the next
layer, not just the routed ones — plausible, since a prefill chunk of 128+ tokens
touches most experts anyway, so guessing "all of them" for the next layer beats waiting
on this layer's routing to know which ones. goinfer's `layerPager`
(`decoder/layerpaging.go`) already does exactly this kind of windowed, deterministic,
routing-independent lookahead for dense weights — it's the pattern to copy for the
expert case, not something to invent. Decode can stay reactive (one token, routing is
cheap and known immediately); the win is specifically in the prefill-chunk case, which
is also where Lever 4 (`task-moe-streaming.md:226`, expert-major prefill batching)
already lives — these two probably want to be scoped together rather than as separate
patches.

**Depends on:** the gocudrv v0.3.0 bump `task-moe-streaming.md` already flags as a
prerequisite; probably sequences after Lever 4.

**Priority: high** — smallest conceptual gap between what goinfer already built and
what this needs, and it's the direct lever under the 11.33 → 16.98 tok/s numbers in
the comparison.

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

This isn't a guess about where the cost lives: `task-moe-streaming.md`'s §C′ already
attributes the real 26B model's **4m49s load time** to "the 11.4 GB pinned alloc +
copy," in those words. Whether reordering the two operations actually moves that
number depends on details this doc can't see from here — whether `NewMappedHostBuffer`'s
underlying `cuMemAllocHost` zero-fills, and whether aikit's `gpu` package exposes a
populate-then-pin path at all — but it's a cheap, narrow experiment against an
already-known, already-measured cost, not a redesign.

**Priority: medium** — small blast radius, but needs an aikit-side primitive that may
not exist yet; check `gpu.NewMappedHostBuffer`'s actual implementation before assuming
this is a goinfer-side change.

---

## Lead 4 — pool the CUDA expert cache globally instead of per layer

**Status: half-true already.** The CPU-side mmap pager (`decoder/moepaging.go`'s
`expertPager`) already pools every layer's experts into one shared `SpanCache` with
frequency-aware (`EvictLeastRecent`) eviction — confirmed by reading `newExpertPager`:
the member-collection loop walks every layer in `w.Layers`, and every layer's experts
land in the same cache. That already matches what FreeToken calls a "shared LRU
residency space."

The CUDA path doesn't: `GOINFER_MOE_CACHE_SLOTS` is a **per-layer** slot count
(`decoder/model.go:167`, `internal/serveapp/main.go:278`), auto-capped to free VRAM at
load. `task-moe-streaming.md`'s §C′ never discusses pooling it across layers — every
mention of slot budgeting there is per-layer. If expert "hotness" is uneven across
layers (plausible, and apparently never measured either way — the doc's own hit-rate
findings are all reported per-run, not per-layer), a fixed depth either wastes slots on
a cold layer or starves a hot one, and the 11.33 / 16.98 tok/s numbers in the
comparison might move for free: no new VRAM, just a different split of the same budget.

**First step:** instrument per-layer hit rate at a fixed total budget (sum of the
current per-layer slots) on the real 26B run, the same way the 77.5%-hit-rate finding
in `task-moe-streaming.md` was produced, and see whether the hit-rate distribution
across layers is actually uneven before building anything.

**Priority: medium** — cheap to measure, uncertain payoff until measured. Exactly the
kind of claim CLAUDE.md's measurement discipline says to check before building:
"measure don't assume."

---

## Lead 5 — bandwidth-adaptive CPU/GPU co-execution

**Status: genuinely new, no equivalent in goinfer today.** FreeToken's split —
q⋆ ≈ m·(B_P / B_H), running some routed experts on the CPU while streaming others to
the GPU concurrently, sized to measured PCIe bandwidth versus CPU MoE-kernel
throughput — has no counterpart here. Every current path is GPU-resident-only (§C′) or
CPU-only (the mmap pager) per architecture; nothing blends the two live.

This is the biggest architectural lift of the five, and the one most likely to collide
with the "ARCHITECTURAL COST" note already on record in `task-moe-streaming.md`'s §C′:
the on-device router exists specifically to avoid a host readback that would stall the
pipeline, and a live CPU/GPU split would need to reintroduce some form of host-visible
routing decision — the exact thing that section says the current design exists to
avoid. Worth a scoping pass of its own before any code, not a quick add-on to leads 1–4.

**Priority: low for now** — track it, don't start it, until leads 1–2 land and there's
a clearer read on how much headroom is actually left on the table.

---

## Revisit, don't "fix": GPU-resident models skip session/prefix reuse

**Not a lead — a note, and a correction to how the original comparison framed it.**
The first draft of the goinfer-vs-FreeToken comparison called this a gap. It isn't
one. `README.md:828-831` states the reasoning directly: "the resident decode path is
fast enough that the per-request session optimization isn't worth it. The OpenAI API
is stateless [clients resend the whole conversation], so this is a throughput trade,
not a correctness change." That's a considered decision, not an oversight, and
`docs/benchmarks.md:799` and `docs/scoping-dsh-goinfer.md:32` both restate it as a
known, documented trade-off rather than a bug.

Worth reopening only for the specific regime where the reasoning is least likely to
hold: a large, slow MoE model (Gemma-4 26B at 11–17 tok/s, or slower) in a long,
growing agentic conversation, where re-prefilling the entire history every turn costs
real wall-clock — unlike whatever faster, smaller model the "fast enough" call was
presumably made against. If Lead 1's checkpoint work happens, it's a natural moment to
re-measure this specific case rather than treat the original decision as settled
forever for every model size.

---

## Sources

- The comparison this opened from: claude.ai/code/artifact/cd014aea-6e94-4f6d-b71a-196ba99b6bfa
- `docs/qwen3_5_moe.md` — "Hybrid cache (decision: correctness-first)"; `:115` for the
  fallback-to-full-recompute line
- `docs/task-moe-streaming.md` — §C′ (`:346`), Lever 1 (`:107`), Lever 3 (`:185`),
  Lever 4 (`:226`)
- `README.md` — the GPU-resident session-skip note (`:828-831`); the Gemma-4 26B slot
  table
- `docs/benchmarks.md:799`, `docs/scoping-dsh-goinfer.md:32`
- `decoder/moepaging.go`, `decoder/layerpaging.go`, `decoder/session.go`,
  `cuda/resident.go` (`mapBytes`)
- FreeToken: arXiv:2608.16157; github.com/FlashML-org/FreeToken

## Next step

Nothing here has been opened into a `docs/prompts/` brief yet, per the usual
Cowork-drafts / vscode-claude-executes split — this is the scoping pass, not the work
order. Say which lead(s) to open first and a brief can follow the same shape as
`docs/prompts/zeno-compare-phase0.md`.
