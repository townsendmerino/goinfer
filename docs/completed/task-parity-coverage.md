# Plan: close the parity-coverage gap (make green CI meaningful, bind the claims)

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.

## Status (archived 2026-09-13, doc-review)

**COMPLETE.** The doc's own purpose — bind the README/matrix "supported" claims to a current,
enforced validation record — shipped, and shipped bigger than scoped here.

- **Item 1 (manifest + staleness detector): shipped, as scoped.** `testdata/parity_manifest.json`
  + `decoder/parity_manifest_test.go`'s `TestParityManifest_fresh` implement exactly the
  per-family `uses`/`own` dependency-set hashing §1c specifies (the two-bucket-core draft this
  doc rejected never shipped), including the later `serialize`-set carve-out. Verified green at
  archival: `GOWORK=off go test ./decoder -run TestParityManifest_fresh -v` → `36/36 families
  enforced`.
- **Item 3 (per-family CI tiny golden + backfill): shipped.** `TestParityManifest_fresh`'s
  COVERAGE check requires a manifest row for every family the capability matrix claims; every
  family in this doc's backfill checklist is present in the current 36-family manifest.
  `docs/parity-coverage-policy.md` — the durable contract this doc always deferred to — grew the
  T1/T2/T3 tier names and absorbed the per-family checklist into its own "Definition of done for
  a new family."
- **Item 2 (small-real-model sweep): moved, not fully built.** The concept lives on as T2 in
  `docs/parity-coverage-policy.md`'s Definition of Done ("the family's smallest real model added
  to the sweep list"), and the (closed 2026-08-15) `docs/just-before-1.0-parity-backfill.md`
  campaign folded "record continuations" into its own Bucket A. But no dedicated
  `TestSmallModelSweep` or `pin_smallmodel_continuations.py` (§2b) was ever committed — the
  automated harness this item scoped is an orphan: no live doc owns building it.
- **Item 4 (extract the shared `weightDiff` helper): not done.** Still two independent
  `Test*_weightDiff` functions (`decoder/qwen35_gguf_weightdiff_test.go`,
  `decoder/qwen3_5_gguf_weightdiff_test.go`), sharing some same-package helpers
  (`loadQwen35GGUFSlice`, `tensorAgreement`) organically rather than through the single extracted
  helper this item scoped. No family's manifest row uses `method: "weightDiff"` as its primary T3
  method today. Small, low-priority, unowned — not worth a task doc on its own.

Every instruction below (the numbered items, effort estimates, "Trigger status: FIRED — do this
now") is a record of what was asked and planned at the time, not a live task.

> **Audience:** implementation plan for the contract in
> `parity-coverage-policy.md`. Four work items, ordered by leverage; each builds
> on scaffolding that already exists (the parity sweep, the `pin_*.py` oracle
> scripts, the `weightDiff` pattern, the `gemma-3-270m` small-model precedent).
> None requires running a multi-GB checkpoint in CI.

## The gap (one paragraph)

Dozens of real-checkpoint parity tests `t.Skip` when assets are absent, so green
CI proves only the model-free + tiny-synthetic tests — not any family's actual
numerics. And nothing ties the README support list to a current passing record:
a loader refactor can break gemma4 with CI still green. The work below makes the
gap **visible and tracked**, adds a **cheap broad real-model net**, and makes the
per-family **CI golden** universal.

---

## Item 1 — validation manifest + staleness detector (highest leverage)

**Why first:** it's the only thing that fixes false-green on the big models we
*can't* put in CI, and it's the mechanism behind the policy's claim discipline.

**1a. `testdata/parity_manifest.json`** — the source of truth. Named **shared
sets** + a per-family **dependency set** (the `core_hash`/two-bucket model is
dropped — see 1c for why it leaked):

```json
{
  "aikit_version": "v1.7.3",
  "shared_sets": {
    "core":      ["decoder/model.go","decoder/forwardn.go","decoder/attention.go","decoder/mlp.go","decoder/kvcache.go","decoder/rope.go","decoder/ropescale.go","decoder/rmsnorm.go","decoder/registry.go","decoder/arch.go","decoder/config.go"],
    "loaders":   ["decoder/weights.go","decoder/gguf.go"],
    "serialize": ["decoder/serialize.go"],
    "quant":     ["decoder/weightmat.go","decoder/gptq.go","decoder/awq.go"],
    "moe":       ["decoder/mlp.go"],
    "deltanet":  ["decoder/deltanet.go","decoder/deltanet_chunked.go"],
    "mamba2":    ["decoder/mamba2.go","decoder/mamba2_chunked.go"]
  },
  "families": {
    "gemma4": {
      "validated_at": "e3eb033",
      "date": "2026-06-13",
      "reference": "HF bf16 (google/gemma-3-4b)",
      "machine": "linux-62gb",
      "metrics": { "argmax_pct": 92.5, "cosine_min": 0.99466, "cosine_mean": 0.99859 },
      "method": "full-forward-oracle",       // or "weightDiff" when the oracle OOMs
      "uses": ["core","loaders","quant"],    // the shared sets this family's numerics depend on
      "own":  ["decoder/forward_gemma4.go","decoder/gemma4.go"],
      "deps_hash": "sha256:…"                 // over (∪ uses' files) ∪ own ∪ aikit_version
    }
  }
}
```

**1b. The staleness test (`TestParityManifest_fresh`, T1 — model-free, in CI).**
For each family: resolve its `uses` sets to file lists, union with `own`, hash
that set's contents plus the recorded `aikit_version`; if it differs from
`deps_hash`, **fail** with `"parity stale for <family>: numerics changed since
<validated_at> — re-run T3 (go run ./cmd/gate parity) and update
parity_manifest.json"`. Pure hashing + JSON read; needs no assets, runs every push.

**1c. The hashing rule — a per-family dependency set, NOT a two-bucket core (this
fixes a real false-green hole).** The first draft modeled numerics as a 6-file
"shared core" + per-family files. That **leaks**, two ways, and leaking is exactly
what Item 1 exists to prevent:

- **The core was too narrow.** Cross-family numerics and loader logic also live in
  `registry.go`, `gguf.go`, `weights.go`, `config.go`, `arch.go`, and the
  quant/dequant path (`weightmat.go`/`gptq.go`/`awq.go`). A bug in `gguf.go`
  dequant moves many families' numerics but, under the narrow core, **trips no
  hash → silent green.**
- **Shared *primitives* belong to a subset, which neither bucket can express.**
  `mamba2.go` is shared by **{Granite, Nemotron}**; `deltanet.go` by
  **{qwen3_5_moe}**. A `mamba2.go` change must mark *those* families stale — not
  everyone (over-trip), not no-one (under-trip).

**Fix:** drop the global-core special case. Each family declares the **full set of
files its numerics depend on** via `uses` (named shared sets, including the *wide*
core, the loaders, the quant path, and any shared primitive) + `own`. The shared
sets express the subset mappings directly: `mamba2.go` lives only in the `mamba2`
set, so editing it re-hashes only families whose `uses` includes `mamba2`. A
`gguf.go` edit re-hashes every family that `uses` `loaders` — i.e. all of them —
closing the dequant hole.

**`serialize` is its own set, and the forward-oracle rows do NOT `use` it.** `serialize.go`
is the `.giw` write+read round-trip; the forward / real-model / tiny oracles load from
safetensors or GGUF and run the forward — they never execute `serialize.go`, so it is not a
dependency of their numerics, and a serialize-only edit (a new `.giw` quant layout, the
Mellum2 dir-input path) must not re-stale them. The one place serialize *does* affect
numerics — the int4 deserialize→resident seam — has its own gate (`gpu.TestGIWInt4_resident`
+ the `internal/giw` round-trip tests), so that risk is covered without coupling it to every
forward row. A future family validated *via* a `.giw` oracle would add `serialize` to its
`uses`; none does today.

Two concrete rules: **(i) full paths always** — `attention.go` exists in *both*
`gpu/` and `decoder/`, so the manifest uses `decoder/attention.go` (the `gpu/`
backend is gated separately, policy's backend axis). **(ii) coarse is fine in the
*over*-trip direction** — `registry.go` holds every family's adapter, so editing
one adapter re-hashes all families that `use` `core`; that's safe (re-validate
more than needed). Only narrow (AST-of-the-adapter-func) if the over-trip is
actually noisy. The forbidden direction is under-trip, which the dependency set
eliminates.

**1d. the parity sweep emits the manifest rows** it validates (commit, metrics,
method, hashes), so a T3 run *updates the truth* rather than printing to a log
that drifts. The roadmap's hand-recorded numbers (the qwen3.6 92.5%/0.99466 etc.)
become the seed data.

**Effort:** low–medium. Mostly the manifest schema + one hashing test + a sweep
flag. **Gate:** the staleness test itself is the gate; seed the manifest from the
families validated to date.

---

## Item 2 — small-real-model-per-family sweep (broadest coverage / effort)

**Why:** the cheap net between T1 (structure) and T3 (full numerics) — it catches
"loads but emits garbage" across the *whole* claimed matrix on hardware that
actually fits.

**2a. Model list** — the smallest published checkpoint per claimed family,
downloadable by `huggingface-cli` (the `gemma-3-270m` pattern already in
`decode_test.go`): Gemma (270m), Qwen2.5 (0.5B), Qwen3 (small), a small Llama /
Mistral, a tiny Mixtral, Mellum, GPT-2. One per family is enough — this tier is
"does the pipeline run," not per-quant numerics.

**2b. Harness** — extend the parity sweep / a `TestSmallModelSweep` (asset-gated
on a sweep dir): for each model, load → generate N greedy tokens → assert against
a **recorded continuation** (pinned by a new `scripts/pin_smallmodel_continuations.py`,
committed). Runs on CPU, fits any box.

**2c. Schedule** — nightly/weekly on the box (or a `make sweep` target), not on
PR CI. A failure files against the family. Feeds T2 of the policy.

**Effort:** medium (mostly per-family checkpoint wrangling + pinned continuations).
Highest coverage gain relative to today.

---

## Item 3 — per-family CI tiny golden (make green mean something everywhere)

**Why:** several families already have this (gemma forward golden, qwen35
deltanet); make it universal so green CI covers *every* claimed family with no
asset.

**3a. Deterministic tiny checkpoint per family** — a seeded random-weights model
at tiny dims (few layers, small hidden/vocab) exercising the family's unique
structure (MoE routing + shared expert, QK-norm, partial RoPE, sliding window,
the hybrid scan). Generated in-test (no committed multi-MB blob).

**3b. Committed golden** — pin the tiny model's reference logits once offline via
the HF reference (extend the `pin_*.py` family), commit the few-KB golden;
in-test compare argmax + cosine. Runs in CI, no external asset.

**3c. Backfill** — audit the claimed families; any without a T1 golden gets one or
is downgraded to experimental per the policy. Make 3a/3b a required item in the
new-family checklist (so GLM and Granite land with it).

**Effort:** low per family (the `pin_*.py` + tiny-checkpoint pattern exists);
the cost is breadth (backfill).

---

## Item 4 — extract the shared `weightDiff` helper (two call sites already exist)

**Why:** `weightDiff` proves a loader without an oracle — diff the GGUF load
against the bit-exact safetensors loader to Q8_0 tolerance. It's further along
than "one precedent": **two** call sites already use the pattern —
`decoder/qwen35_gguf_weightdiff_test.go` (`TestQwen35GGUF_weightDiff`) **and**
`decoder/glm4moe_gguf_test.go`. So Item 4 is **extract the shared helper from the
two existing sites**, then make it the *standard* T3 substitute for any family
whose full forward oracle OOMs (the big MoE / dense families), recorded in the
manifest with `"method": "weightDiff"`.

**Effort:** very low — it's a refactor of two committed tests into one helper plus
a documented requirement, not new mechanism.

---

## Sequencing, effort, triggers

1. **Item 1 (manifest + staleness)** — do first; it's the policy's enforcement
   mechanism and seeds from data we already have. Small.
2. **Item 3 (CI tiny goldens + backfill)** — in parallel; makes green meaningful
   and is the cheapest per family.
3. **Item 2 (small-real sweep)** — next; the broad net, a bit more wrangling.
4. **Item 4 (weightDiff standardization)** — folds in as big families are
   (re)validated.

**Trigger status: FIRED (2026-06-15) — do this now.** The trigger was "GLM /
Granite landing." They've effectively landed: `glm4moe{,_air,_gguf}_test.go`,
`granite{,_real}_test.go`, `mamba2.go` + `mamba2_chunked.go`, and
`pin_glm_*`/`pin_granite_*` are all committed. So the harness is being
retrofitted, not pre-built — which makes Item 1 (stop the silent-skip bleeding)
more urgent, not less. Order remains 1 → 3 → 2 → 4; start immediately.

## Backfill checklist (the currently-claimed families)

For each of {Gemma 3, Gemma 4 (incl. E/PLE, 12B), Qwen2/2.5, Qwen3, Qwen3.5/3.6-MoE,
**GLM (4.5-Air / 4.6)**, **Granite 4.0 Hybrid**, Llama 2/3, Mistral, Mixtral, GPT-2,
Mellum, Mellum2, Gemma 3 VL}: confirm a T1 golden (Item 3), a manifest row (Item 1)
by full-oracle or `weightDiff` (Item 4), and a T2 sweep entry (Item 2). Anything
missing all three is relabeled experimental in the README until filled — this is
the concrete output that makes the support claims honest. **GLM and Granite are
newly landed, so they're the highest-priority rows to validate first** (their
`pin_*`/`weightDiff` tests already exist — wire them into the manifest).
