# Task: Cloudflare's Clairic — System One extraction on goinfer (E0–E5) — 2026-10

> **Status: SCOPED 2026-10-19. Waiting on weights.** Clairic is Cloudflare's System One *extraction* model, in preview
> on Workers AI; the announcement says "We intend to release the model open-weights soon." This doc sits **beside** the
> parked decisions work (`docs/tasks/parked/task-constrained-confidence.md`, parked 2026-10-05, owner: "I have more
> important work") and does not reopen it: Clairic's release is not one of that doc's two reopen triggers. **This doc's
> own trigger is Clairic's weights published under a licence that permits redistribution.** E0 is desk work that can run
> before the trigger if the owner wants the API on record; nothing from E1 on starts before it.

## 0. What Clairic is (Cloudflare's post and model reference, read 2026-10-19)

- **What it does.** Typed values from unstructured input: extraction, computation, estimation and forecasting. It
  "processes input tokens with a single forward pass" and generates no output tokens. Workers AI id
  `@cf/cloudflare/clairic`; context 65,536 tokens; fields that do not fit one 8,192-token pass are split across parallel
  passes. Images: PNG, JPEG or WebP, up to 4, 4 MiB each and 8 MiB total; no remote URLs. `state` must be non-empty
  unless images are sent; at least one field, no upper limit.
- **Request:** `model`, `state`, `fields`, and optionally `images`, `reference_date` and `options`. A field has a `type`
  and optional `instructions`; type-specific keys read so far: `number` takes `decimals` and `unit`, `list` takes
  `items`, `estimate` takes `min`, `max` and `step`. The post also names `time`, `datetime`, `duration` and `forecast`,
  which the reference page's summary did not list. E0 records the complete set.
- **Response:** `model`, `answers` keyed by field name, `truncated`, `usage` (`state_tokens`, `image_tokens`,
  `schema_tokens`, `input_tokens`, `output_tokens`, `passes`); the schema also lists `state_text`. Each answer carries
  `type` and `value`, plus: `text` → `p_null`, `span`, `guess`, `confidence`; `date` and `number` → `p_null`, `guess`,
  `confidence`; `list` → `spans`, `confidence`; `estimate` → `mean`, `median`, `p10`, `p90`, `p_below`, `p_above`.
- **Not stated anywhere yet:** base model, parameter count, a licence for the weights (the model reference has hosted
  terms only), a Hugging Face repo.
- **Its relation to Clef.** The post calls both "System One models because they answer in a single pass", but presents
  Clairic as a separate class and does not say they share a backbone or an API. The APIs do differ: Clef takes
  TypeSafe's `questions` (`noul`, `choice`, `score`), Clairic takes `fields`.

## 1. What goinfer already has, and what it does not

- **`POST /v1/systemone`** in TypeSafe's shape, answered three ways: label scoring on the served model
  (`internal/decide`), a trained decision head, or — route C — Cloudflare's Clef (`isClefDir` in
  `internal/serveapp/systemone_clef.go`; the `internal/clef` package: `Encode`, `LoadHead`, `Head`, `Model`).
- **Route C is the template for this doc.** D10–D14 (2026-10-01 to 2026-10-04,
  `docs/completed/task-clef-route-c-2026-10.md`) read Clef-flash's encoder and head contract in full, found by a merge
  check that it is Qwen3.5-9B with its adapters merged — so it **loads as a plain `qwen3_5`** and needed no new decoder
  work — plus a 121.8M-parameter joint head, built a reference fixture from Cloudflare's own module (f32 and bf16, 150
  records), and gated goinfer against it (`TestEncode_matchesReference`, `TestHead_matchesReference`,
  `TestLoadHead_refusesWhatItWasNotVerifiedAgainst`). Record: `docs/measurements/decisions-d10-clef-2026-10-02.md`.
- **`qwen3_5` carries a vision tower** in the capability matrix, which matters if Clairic's backbone is the same family,
  since Clairic accepts images.
- **Not built:** any extraction surface — fields, text spans with character offsets, number and date values, quantile
  intervals, `guess` beside `p_null`, multi-pass splitting for wide schemas, `reference_date`.
- **A correction to carry.** D0 §6 (`docs/measurements/decisions-d0-prior-art-2026-09-27.md`, "Peer engines: none ships
  a decisions endpoint") was true on 2026-09-27. Ollama's server now routes `/v1/systemone` (read from its
  `server/routes.go` on 2026-10-12 for `task-integration-harness-2026-10.md`'s research; confirm). The endpoint is no
  longer goinfer's alone. What still is, as far as checked: running Cloudflare's System One models natively, gated
  against Cloudflare's own reference.

## 2. Items

### E0 — the API on record (desk; no weights; minutes)

From the Clairic model reference and the Workers AI REST reference, record the request and response JSON verbatim where
the pages show it, in the style of D0 §3 ("TypeSafe's `POST /v1/systemone`"), into
`docs/measurements/clairic-e0-api-2026-10.md`: every field type with its parameters, every answer shape, `options`,
`reference_date`'s semantics, the limits, the error shapes, and the multi-pass rule. Where a type is named but not shown
(the post's `time`, `datetime`, `duration`, `forecast`), record it as named-not-shown rather than inferring its shape.
**E0 decides E3's route shape.** Date the record: the model is in preview and its API may move before the weights land.

### E1 — the weights, read (trigger: the release)

D10's shape: pin the repository, revision and sha256 of every file, including Cloudflare's custom module (pinned by
sha256 as D10 pinned `joint_schema_model.py`, since `trust_remote_code` runs it). Then, in this order, each a stop
point:

1. **Licence.** Forbids redistribution → record it and stop. goinfer's releases put models inside binaries.
2. **Backbone merge check** against the families goinfer loads, as D10 found Clef-flash to be plain `qwen3_5`. A family
   goinfer already loads → continue. A new family → stop: that is a family-support item, scoped on its own, and it is
   the real cost of Clairic.
3. **The module read in full:** prompt assembly and per-fragment tokenization (route C's encoder tokenizes each piece
   separately, with no merges across boundaries), span conventions (token indices or characters, and how character
   offsets are recovered), the head config and tensor list, every readout (span pointers, values, `p_null`, quantiles),
   image preprocessing, and the multi-pass split.
4. **An op goinfer lacks** in the head → name it, and scope it here before E2.

### E2 — the reference fixture (Linux box, local disk)

As D10: run Cloudflare's module over a fixed record set — at least 150 records, covering every field type, with
images, with records that should answer null, and with a schema wide enough to need more than one pass — and keep the
token ids, spans and every head output at f32 and bf16. **Disk first.** D10 kept Clef-flash on the archive because
`/home` on nobara had 9 GB free, and CLAUDE.md forbids timed runs from the archive. Clairic needs local space before any
timed run.

### E3 — encoder and head in Go, gated against the fixture

Most likely an `internal/clairic` package beside `internal/clef`, sharing what is identical (the Python-compatible JSON
rendering, the tokenizer plumbing). Gates, mirroring route C:
- the encoder is token-exact on every fixture record;
- head outputs fall within the fixture's own f32-to-bf16 spread;
- answers equal the reference's answer function on every record: value, `p_null`, `confidence`, `guess` and span
  offsets exactly; quantiles within the spread the reference itself shows between f32 and bf16.

**Route.** E0 decides it. Lean: a route that takes Clairic's body unchanged, so Cloudflare's own client code works
against goinfer with only a base-URL change — the same reason `/v1/systemone` takes TypeSafe's shape unchanged.

### E4 — quantized arms and the default

As D6b: f32, bf16, `int8int8` and `int4` against the fixture. **Registered before running, per field type:**
agreement with the f32 reference of at least 98% on span and value exact match (D6b's top-1 bar); interval endpoints
within one `step` or 1% relative; `p_null` calibration reported with its error. D6b's precedent stands: when a
quantized arm misses the bar, the default is the owner's call, recorded with what it does not claim. Also expect
route C's hardware wall: Clef on CUDA needed 9.5 GB at `int8int8` against the 8 GB card, with `int4` refused. Name it
early if Clairic is the same size.

### E5 — the cascade, measured (optional)

Cloudflare's post runs Clairic → Clef → GLM-5.3 at 19.3 s against 61.7 s for an all-LLM pipeline. goinfer can hold
Clairic, Clef and a local LLM in one process with multi-model serving. One labelled measurement, on one box, of that
pipeline shape against an all-LLM arm on the same box; the all-LLM arm is the do-nothing arm. Only if E3 passes and the
owner wants the claim.

## 3. Not in scope

- The parked decisions doc's open items (jevx end to end, the Clef-flash registry tag), which keep their own triggers.
- Training or fine-tuning anything.
- Any hosted-only behaviour that the weights do not carry, if E0 finds some (`options` is the candidate to check).

## 4. Conventions

Cite code by declaration name and docs by path and heading (CLAUDE.md's citation rule). Checkpoints from `~/models` on
the box doing the run, never the archive. Pin every `trust_remote_code` module by sha256. Estimate before any run;
runs go by day unless they genuinely need the night (owner, 2026-10-19). Stage explicit paths, and leave changes
uncommitted for review.

## Sources

- [Cloudflare: Clairic](https://blog.cloudflare.com/clairic-extraction-model/) (2026-10)
- [Workers AI: Clairic model reference](https://developers.cloudflare.com/workers-ai/models/clairic/)
- [Workers AI: extraction models](https://developers.cloudflare.com/workers-ai/concepts/extraction-models/)
- [Workers AI REST: run a model](https://developers.cloudflare.com/api/resources/ai/methods/run/)

<!-- doc-reviewed: 2026-10-19 -->
