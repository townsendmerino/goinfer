# Model × hardware matrix — generate it from the taxonomy, never hand-maintain

> The "can I run model X on my hardware?" table is the #1 user question — worth having.
> But a **hand-maintained** one rots: the current README coverage table **already has
> drifted cells** — Gemma-3-Metal and MoE-CUDA both landed resident this cycle, yet the
> committed table still shows them as CPU-fallback. That's the whole case. goinfer already
> computes the truth in the feature taxonomy; the fix is to **generate** the table from it,
> freshness-tested like the capability matrix, so it can't drift and CI catches it.

## Data model

- **Rows:** every registered family (from the registry — same source `capability-matrix.md`
  already walks).
- **Columns:** CPU · WebGPU · CUDA · Metal.
- **Cell:** `resident` / `CPU fallback` (+ a curated caveat marker where memory-bound).
- **CPU is always ✓** — the reference path every family runs on, and the fallback floor.

## The one rule that keeps it honest: call the runtime's admission predicate, not a copy

Resident-vs-fallback is **two** gates today: the taxonomy subset check
(`len(MissingResidentFeatures(arch, ResidentBackendFeatures[backend])) == 0`) **and**
`decodeRunnerEligible(arch)` (the own-forward exclusions — llama4 / gemma4 / qwen35 / …). A
generator that checks only the taxonomy would print `llama4` as CUDA-resident (its
`missing == []` after the MoE flip) when the runtime actually declines it. So the table
would *lie*.

**Factor admission into one exported predicate and have both the runtime and the generator
call it:**

```go
// decoder/features.go — single source of truth for "does this backend run this arch resident?"
func ResidentEligible(a *Architecture, backend string) bool {
    return decodeRunnerEligible(a) &&
        len(a.MissingResidentFeatures(ResidentBackendFeatures[backend])) == 0
}
```

The runtime's residency decision routes through `ResidentEligible`; the generator calls it
per (family × backend). The table can never disagree with behavior — and this **retires the
"load-bearing hand-list" risk** we flagged after the `FeatMoE` flip: the own-forward
exclusion becomes part of one derived predicate, tested once.

## Memory caveats — curated, not generated

Whether a *specific checkpoint* fits (Metal Mistral-7B > 16 GB; MoE unified-memory) is a
per-size, per-hardware fact the taxonomy can't know. So: **generate the resident/fallback
cell body; curate the fit caveats as footnotes** keyed to cells. There are only a handful,
and they change rarely.

## Output + freshness gate

- **Generator** (`go test ./decoder -run HardwareMatrix -update`): walk the registry, build
  each family's descriptor (arch flags only — no weights, same as the capability matrix),
  call `ResidentEligible(arch, backend)` for each backend, emit the markdown table to
  `docs/hardware-matrix.md` (or a section of `capability-matrix.md`), sorted by family.
- **`TestHardwareMatrix_fresh`**: regenerate in-memory, diff vs the committed doc, fail on
  drift; `-update` rewrites. **Model-free → runs in CI every push** (like
  `TestParityManifest_fresh` / `TestCapabilityMatrix`). This kills the stale-cell class dead
  — including the two cells that are stale *right now*.
- **Break-it-first** (per the gate-audit discipline): in a test, flip one entry in
  `ResidentBackendFeatures`; confirm the generated cell flips **and** the freshness test goes
  RED; revert. Proves the gate is non-vacuous.

## The README slice (compact, curated — ideally generated too)

The README keeps a **small popular-models** table that links to the generated full one.
Best: tag registry entries with a `popular` flag so the generator emits this slice as well
(then it's freshness-tested too). Minimum: a test asserts the README slice's cells equal the
generated full table's for those rows — so the README can't drift from truth.

**Draft — cells are ILLUSTRATIVE; the generator fills them (I hand-derived these and two are
exactly the cells that keep drifting, which is the point):**

| Family | CPU | WebGPU | CUDA | Metal |
|---|---|---|---|---|
| Qwen 2.5 / 3 · Llama | ✅ | ✅ | ✅ | ✅ |
| Mistral · Phi-3 | ✅ | ✅ | ✅ | ✅ ¹ |
| Gemma 3 | ✅ | ✅ | ✅ | ✅ *(gen)* |
| MoE — Mixtral · Qwen-MoE · GLM | ✅ | ✅ | ✅ *(gen)* | ✅ ² |
| DeepSeek-V2/V3 · Kimi (MLA) | ✅ | ✅ | CPU | CPU |
| Granite-4.0-H · Nemotron-H (SSM) | ✅ | ✅ | CPU | CPU |
| Gemma 4 | ✅ | CPU | CPU | CPU |

> Every family runs on the pure-Go **CPU** path; the GPU columns show resident acceleration,
> with automatic CPU fallback (never wrong output) everywhere else — guaranteed by the
> feature taxonomy. Full generated matrix: [capability-matrix.md](capability-matrix.md).
> ¹ Metal Mistral-7B needs > 16 GB unified memory. ² MoE fit is unified-memory-bound.

## Why this is the right shape (one line)

Generate the truth from the taxonomy (the only source that stays correct), freshness-gate it
in CI (so a backend change updates the table or fails the build), curate only the handful of
memory caveats by hand, and let the README show a verified slice — so "on what hardware"
never rots the way it already has.
