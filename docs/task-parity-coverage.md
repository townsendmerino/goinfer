# Plan: close the parity-coverage gap (make green CI meaningful, bind the claims)

> **Audience:** implementation plan for the contract in
> `parity-coverage-policy.md`. Four work items, ordered by leverage; each builds
> on scaffolding that already exists (`parity_sweep.sh`, the `pin_*.py` oracle
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

**1a. `testdata/parity_manifest.json`** — the source of truth. One entry per
family:

```json
{
  "core_hash": "sha256:…",                 // shared numerics files (see 1c)
  "families": {
    "gemma4": {
      "validated_at": "e3eb033",
      "date": "2026-06-13",
      "reference": "HF bf16 (google/gemma-3-4b)",
      "machine": "linux-62gb",
      "metrics": { "argmax_pct": 92.5, "cosine_min": 0.99466, "cosine_mean": 0.99859 },
      "method": "full-forward-oracle",       // or "weightDiff" when the oracle OOMs
      "files": ["registry.go:gemma4Architecture", "forward_gemma4.go", "gemma4_*.go"],
      "files_hash": "sha256:…"
    }
  }
}
```

**1b. The staleness test (`TestParityManifest_fresh`, T1 — model-free, in CI).**
For each family: recompute the hash over its declared `files` set and the shared
`core_hash`; if either differs from the recorded value, **fail** with
`"parity stale for <family>: numerics changed since <validated_at> — re-run T3
(scripts/parity_sweep.sh) and update parity_manifest.json"`. Pure hashing + JSON
read; needs no assets, so it runs every push.

**1c. The hashing rule (coarse-first, refine if noisy).** Two inputs:
- **Shared core** — `attention.go`, `mlp.go`, `model.go`, `kvcache.go`, `rope.go`,
  `rmsnorm.go`, plus the active `linalg` version. A change here marks **all**
  families stale (correct: it can move everyone's numerics).
- **Per-family files** — the family's arch-adapter, tensor schema, and dedicated
  forward/loader files. A change marks just that family stale.

Start coarse (whole-file hashes). If the core hash trips too often on no-op edits
(comments/formatting), narrow to AST-of-exported-numerics or accept the re-run
cost — don't over-engineer before it's actually noisy.

**1d. `parity_sweep.sh` emits the manifest rows** it validates (commit, metrics,
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

**2b. Harness** — extend `parity_sweep.sh` / a `TestSmallModelSweep` (asset-gated
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

## Item 4 — systematize `weightDiff` as the "oracle won't fit" tier

**Why:** the qwen3.6 precedent (`TestQwen35GGUF_weightDiff`) proved a loader
without an oracle by diffing GGUF against the bit-exact safetensors loader to
Q8_0 tolerance. Make it the *standard* T3 substitute for any family whose full
forward oracle OOMs (the big MoE / dense families), recorded in the manifest with
`"method": "weightDiff"`. No new mechanism — a documented requirement + reuse.

**Effort:** low (pattern exists; apply per big family).

---

## Sequencing, effort, triggers

1. **Item 1 (manifest + staleness)** — do first; it's the policy's enforcement
   mechanism and seeds from data we already have. Small.
2. **Item 3 (CI tiny goldens + backfill)** — in parallel; makes green meaningful
   and is the cheapest per family.
3. **Item 2 (small-real sweep)** — next; the broad net, a bit more wrangling.
4. **Item 4 (weightDiff standardization)** — folds in as big families are
   (re)validated.

**Trigger to prioritize all four:** the GLM / Granite families landing
(`task-model-families-glm-granite.md`) — they're the first families to go through
the completed "definition of done" checklist, so the harness should exist as they
land rather than be retrofitted. Until then, Item 1 alone already stops the
silent-skip bleeding and is worth doing immediately.

## Backfill checklist (the currently-claimed families)

For each of {Gemma 3, Gemma 4 (incl. E/PLE, 12B), Qwen2/2.5, Qwen3, Qwen3.5/3.6-MoE,
Llama 2/3, Mistral, Mixtral, GPT-2, Mellum, Mellum2, Gemma 3 VL}: confirm a T1
golden (Item 3), a manifest row (Item 1) by full-oracle or `weightDiff` (Item 4),
and a T2 sweep entry (Item 2). Anything missing all three is relabeled
experimental in the README until filled — this is the concrete output that makes
the support claims honest.
