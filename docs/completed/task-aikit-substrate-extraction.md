# Plan (goinfer): promote the weight-paging substrate to aikit

> **Status: DONE (2026-06-17).** aikit v1.9.0 shipped the leaf (`aikit/mmap`:
> `MapReadOnly`/`Unmap`/`Advise`/`SpanCache`/`PageAlignedInterior`/`AutoBudget`/
> `AvailableRAM`, plus `linalg.WeightMat.MappedSpan`); goinfer refactored onto it
> in the same commit as the `v1.8.1→v1.9.0` bump. Outcome vs the plan below:
> - **Deleted** all 5 primitive files (`mmap_{unix,other}.go`, `madvise_{unix,darwin,
>   other}.go`) — they lifted verbatim to `aikit/mmap`. `alignedMappedSpan` →
>   `WeightMat.MappedSpan`; `autoWeightBudget`/`availableRAMBytes` → `mmap.AutoBudget`/
>   `AvailableRAM`.
> - **expertPager** became a thin `mmap.SpanCache[*expertWeights]` wrapper (the LRU
>   moved down); the MoE-router demand signal (`touch`, called from `moeMLP`) stayed.
> - **layerPager** kept its windowed prefetch local (it is NOT an LRU, so it does not
>   use `SpanCache` — re-eval checklist #2's acceptable outcome); it borrows only
>   `Advise` + `MappedSpan` + `AutoBudget`.
> - **Tests:** the LRU unit test + the refault test are now aikit's
>   (`TestSpanCache_*`, `TestMadvise_dontneedRefaultsIntact`) and were dropped from
>   goinfer; the goinfer-unique `TestMadvise_dontneedFreesRSS` was re-pointed at
>   `mmap.MapReadOnly`/`Unmap`/`Advise` (passes — freed 262/262 MB). The end-to-end
>   bit-exact gates (`TestExpertPaging_bitExact`, `TestLayerPaging_bitExact`) are
>   unchanged (asset-gated; skip without the fixtures on this box).
> - **go.mod:** minimal aikit bump in root + `gpu/` + `demo/agent/`, kept gpu-free
>   (the maintainer's cross-module `-tags gpu` builds use an uncommitted go.work, so a
>   bare `go mod tidy` wrongly injects `goinfer/gpu` — avoided).
>
> Original plan retained below for the record.

## Trigger (FIRED — evidence)

Was: execute when all three hold. Status as of 2026-06-17:

1. ✅ **Phase 1 is stable.** `expertPager` / `layerPager` have gone *many* minors
   without an API-shape change — GLM, Granite, Nemotron, DeepSeek-V2/V3, Kimi, and
   Qwen2.5-VL all landed since without touching the pager surface. (It only *grew*
   in the OS-mechanism direction: a new `madvise_darwin.go` — generic paging
   mechanism, exactly what belongs in the leaf.)
2. ✅ **GGUF Phase 2 (#1) is resolved.** The native-block path is decided D-first
   (transparent `.giw` cache) with the native-block kind gated on the Q6_K spike —
   so a "span" stays **mapping-backed** for the foreseeable path; the `SpanCache`
   contract does not need to serve heap-backed K-quant blocks. Contract settled.
3. ⏳ **aikit ships the leaf** — the one remaining gate. The aikit-side prompt
   (`prompts/aikit-mmap-residency-leaf.md`) does this; goinfer refactors **only
   after** `aikit/mmap` + `linalg.WeightMat.MappedSpan` are tagged. **Never refactor
   against an untagged aikit.**

So: hand aikit its prompt now; this goinfer refactor unblocks the moment aikit tags
the leaf. **Re-read this code before refactoring — it has moved** (e.g. the darwin
madvise path is new since this plan was written).

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

- **Bump:** `go.mod` `require github.com/townsendmerino/aikit` from the current
  `v1.8.1` to the leaf-bearing release, in the same commit as the refactor.
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
