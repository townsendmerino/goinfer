# Plan: a generated, community-readable capability matrix (from the registry, not instead of it)

> **Audience:** internal planning, `roadmap.md` Track-style. Goal: give the
> community a readable map of *what model families goinfer supports and how each
> is configured* — **generated from the Go registry**, so the code stays the
> single source of truth and the artifact can't drift. Explicitly **not** a
> rewrite of the registry into JSON.

## The decision (and why it's generate-not-author)

The registry is Go: `registry.go` maps each `model_type` → an `archAdapter`
function that returns an `Architecture` descriptor + tensor schema. That function
is **two things**: a flat, declarative descriptor (the struct it returns —
`QKNorm`, `RotaryDim`, `SlidingWindow`, `MoEConfig`, `Norm`/`Act` enums …) *and*
real imperative logic (HF class-default backfill, `rope_parameters` format
handling, computed fields like `AttnScale = QueryPreAttnScalar^-0.5`,
`ValidateAssumptions`). The descriptor is data; the adapter is code.

So:

- **Authoring the registry in JSON is rejected.** The declarative descriptor is
  the easy ~20% of a family; the config-quirk normalization + validation + parity
  golden is the 80%, and none of that reduces to static JSON without inventing a
  config-transform DSL that's *less* readable than Go and loses the type checker
  and `go test`. (transformers/vLLM keep architectures as code for exactly this
  reason — goinfer is in good company.)
- **Generating a readable artifact *from* the registry is the win.** Discoverability
  is a real gap for a 15-family runtime, and a generated matrix serves it without
  inverting the source of truth.

## Deliverables

1. **A capabilities view derived from the registry.** A `go generate` step (a small
   program under `cmd/` or a `-update` test) that, for every `model_type` in
   `registry`, resolves its adapter against a **representative config** and reads
   the *family-constant* fields off the resulting `Architecture` (+ tensor schema +
   loader support). Emits:
   - **`docs/capability-matrix.json`** — the machine-readable view (one object per
     family).
   - **`docs/capability-matrix.md`** — a rendered table, **organized by the
     coverage axes** (softmax-GQA / gated-linear / state-space / latent-KV, MoE
     vs dense, hybrid), i.e. the matrix the coverage-axis positioning
     (`task-model-families-next.md` Step 0) already calls for.
2. **A freshness gate (CI, model-free).** A test that regenerates and `git diff`s —
   dirty tree ⇒ fail "regenerate the capability matrix." Plus: every `registry`
   key must be representable (have a resolvable config), so adding a family without
   a matrix row fails CI. This is what keeps the artifact honest.
3. **README link.** Point the positioning section at `capability-matrix.md`.

## What goes in a row (family-constant traits only)

Capture what's invariant per family, **not** per-checkpoint dims (hidden size,
expert counts, layer counts vary by checkpoint — exclude or mark "varies"):

- `model_type` alias(es) + display name + one-line description
- **Sequence mixer / coverage axis** — softmax-GQA · gated-linear (DeltaNet) ·
  state-space (Mamba-2) · hybrid(ratio) · (latent-KV when MLA lands)
- **MoE** — dense | sparse; shared always-on expert y/n (counts: "varies")
- **Sliding window** — none | all-layer | interleave-pattern
- **QK-norm** y/n · **RoPE style** — full · partial · dual-base · m-RoPE · learned/none
- **Norm** (RMS/LayerNorm + placement) · **activation** (SwiGLU/GeGLU/GELU/ReLU²)
- **Tied LM head** — yes/no/checkpoint-dependent
- **Loaders** — safetensors · GGUF (+ transform-reverser?) · GPTQ · AWQ
- **Modality** — text-only | vision tower (which)
- **GPU residency-eligible** — from `DecodeRunnerEligible` (yes/no)
- **Parity status** — joined from `testdata/parity_manifest.json`: `validated_at`,
  method (full-oracle | weightDiff | layer-slice), metrics. So the matrix reads as
  *what's supported + how it's configured + how well it's validated.*

## Source-of-truth: derive, with a tiny annotation only where needed

Primary approach: **read the fields off the adapter's output** (resolve against a
representative config) — drift-proof, because it *is* the code's behavior. Most
families already have a tiny/parity config to resolve against; add a minimal canned
config per family that lacks one (cheap, and useful for the parity-coverage tiny
goldens too). A handful of presentation-only fields with no `Architecture`
equivalent (display name, one-line description, coverage-axis label) live in a
small `familyDoc` map next to the registry — annotation, not a second descriptor —
and the coverage test asserts every registry key has one.

## Tie-ins (this isn't a standalone artifact)

- **Coverage-axis positioning** (`task-model-families-next.md` Step 0): this
  generated, axis-organized matrix *is* the rendered table that step asks for —
  building it discharges part of that step.
- **Parity manifest** (`task-parity-coverage.md`): the matrix joins each family to
  its manifest row, so "supported" and "validated" appear together — directly
  serving the claim-discipline rule (no support claim without a gate).
- **New-family definition of done**: add "matrix row generated + parity row" to the
  `parity-coverage-policy.md` checklist so GLM/Granite/Nemotron/MLA each land in the
  matrix automatically.

## Non-goals

- Inverting the source of truth — the registry stays Go; the matrix is a *view*.
- A JSON authoring format / config-transform DSL for adapters.
- Per-checkpoint dims or expert counts in the matrix (they vary; mark "varies").
- Hand-editing the generated files (mark them generated; the freshness gate
  enforces it).

## Effort / risk

**Low–medium.** The generator is a small registry-iterating program + a markdown
template; the freshness gate is regenerate-and-diff. The only real work is ensuring
every family resolves against a representative config (overlaps with the
parity-coverage tiny-golden backfill) and choosing the family-constant column set.
No change to the hot path, the registry, or any loader.
