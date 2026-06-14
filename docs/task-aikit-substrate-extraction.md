# Plan (goinfer): promote the weight-paging substrate to aikit

> **Status: DEFERRED — written now, executed later.** This is the goinfer half of
> a two-repo extraction (the aikit half:
> `aikit/docs/task-mmap-residency-leaf.md`). The substrate (mmap + madvise +
> span-residency) **just shipped** with ideas #2/#4 (`ideas-weight-memory.md`) and
> is still settling. Refactoring it onto an aikit dependency *before* it stops
> moving would couple goinfer to a public API that's still finding its shape.
> **Fire only when the trigger below is met.**

## Trigger (when to execute)

Execute when **all** hold:

1. **Phase 1 is stable** — `expertPager` / `layerPager` have gone one goinfer minor
   without an API-shape change.
2. **GGUF Phase 2 (#1) is resolved** — zero-copy GGUF has landed or been shelved,
   so we know whether the span-cache must serve heap-backed *and* mapping-backed
   spans (it changes the aikit `SpanCache` contract).
3. **aikit has shipped the leaf** — `aikit/mmap` + `linalg.WeightMat.MappedSpan`
   are tagged in an aikit release (the aikit plan ships them as Experimental in the
   next minor after v1.7.3). **Never refactor against an untagged aikit.**

At trigger time, re-read this code before refactoring — it will have moved.

## Why move it down

The substrate's mechanism is generic; only the *demand signal* is goinfer's. The
read-only mmap primitive is already duplicated 3× across the ecosystem (`aikit/ann`,
`aikit/embed`, and now `goinfer/decoder/mmap_unix.go`), and `madvise` + the
span-residency LRU are generic OS mechanism aikit lacks entirely. Worse, aikit's
own flagship `ann.LoadFlatI8Mmap` — int8 codes aliased from a read-only mapping,
the *identical substrate* as our expert weights — has **no residency control**, so
the exact capability we built (page a large int8 mapping under a RAM budget) is one
aikit wants too. The generic half belongs one layer down; goinfer keeps the
model-specific half.

## What goinfer drops / refactors

**Delete (moves to `aikit/mmap`):**
- `decoder/mmap_unix.go` + `decoder/mmap_other.go` (`mmapReadOnly`/`munmap`) →
  `aikit/mmap.MapReadOnly` / `Unmap`.
- `decoder/madvise_unix.go` + `decoder/madvise_other.go` (`madviseBytes`) →
  `aikit/mmap.Advise`.
- `availableRAMBytes` + `autoWeightBudget` (the /proc/meminfo budget helper) →
  the leaf's system helper.
- `alignedMappedSpan` (operates on `linalg.WeightMat`) →
  `linalg.WeightMat.MappedSpan(base, end)` in aikit.

**Refactor onto `aikit/mmap.SpanCache` (keep the file, gut the generic core):**
- `expertPager` becomes a thin wrapper: build a `SpanCache` over the expert spans,
  and `touch(ex)` → `cache.Touch(spans[ex])`. The **MoE-router demand signal stays
  here** (the `moeMLP` `topK` hook, `mlp.go`).
- `layerPager` likewise: a `SpanCache` (or a sequential-window variant) plus the
  **layer-order prefetch (L+1) signal**, which stays here. If `SpanCache`'s LRU
  doesn't cleanly express "prefetch ahead + release behind a window," keep
  `layerPager`'s windowing local and only borrow `Advise` + the alignment helper —
  decide at trigger time against the real API.

## What stays in goinfer (do NOT move)

- The two pagers' **demand logic** — router top-k selection, layer-order
  prefetch, knowledge of `LayerWeights` / `expertWeights`. That's the whole point
  of the split: aikit owns *which bytes are a span and how to page them*; goinfer
  owns *which span to fault and when*.
- The `.giw` format and `serialize.go` (serializes decoder `Weights`) — goinfer's.
  (The envelope-discipline de-dup is a separate soft watch in the aikit plan; not
  part of this lift.)
- Tiered KV (`cmd/serve/sessions.go`, idea #8) — serve/KV-specific.

## Design-for-extraction north star (apply NOW, while it's still goinfer-local)

So the eventual lift is a clean move, not a rewrite, keep these true in the interim
as #2/#4 get tuned:

- Keep `madviseBytes` / `mmapReadOnly` **free functions with no decoder types in
  their signatures** (already true) — they must lift verbatim.
- Keep the **page-alignment math** (`alignedMappedSpan`'s rounding) separable from
  the **`WeightMat` extraction** (the `Int8()/Int4()` lookup) — the former goes to
  the leaf, the latter to `linalg`. Don't fuse new logic across that seam.
- Keep the `expertPager` LRU **demand-signal-agnostic** (it already only knows
  spans + bytes + budget) — resist adding MoE-specific fields to it; put any new
  router logic in `moeMLP`, not the cache.

## Dependency + gate

- **Bump:** `go.mod` `require github.com/townsendmerino/aikit` from `v1.7.3` to the
  leaf-bearing release, in the same commit as the refactor.
- **This is a pure refactor — behavior must not change.** The existing bit-exact
  gates are the proof and must stay green **unchanged**:
  `TestMadvise_dontneedRefaultsIntact` (model-free; it migrates down to aikit with
  the code — re-point goinfer's at the aikit symbol or drop it as now-aikit's),
  `TestExpertPager_lruEviction`, `TestExpertPaging_bitExact` (asset-gated),
  `TestLayerPaging_bitExact` (model-free). If any needs editing beyond the import
  path, the extraction changed behavior — stop and reconcile.
- No new user-facing surface: `--stream-weights` / `--weight-cache` flags and their
  semantics are unchanged; only the implementation moves.

## Re-evaluation checklist (the "look at the code again" step)

Before executing, confirm against the then-current code:

1. Did GGUF Phase 2 change what a "span" is (heap-backed K-quant blocks vs
   mapping-backed int4/int8)? If so, the `SpanCache` contract must account for it.
2. Did `layerPager`'s window/prefetch diverge enough from an LRU that it shouldn't
   share `SpanCache`? (Acceptable outcome: only `Advise` + alignment lift; the
   windowing stays goinfer-local.)
3. Is there now a *second* aikit consumer (FlatI8 paging shipped)? That's the
   signal the API generalized correctly — if aikit's use and goinfer's use don't fit
   one `SpanCache` shape, the abstraction isn't ready; carry the duplication another
   release rather than force it.
