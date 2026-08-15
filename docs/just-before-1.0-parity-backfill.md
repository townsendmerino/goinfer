# Just before v1.0: the parity-backfill campaign (fill every "pending" matrix row)

> **Audience:** internal release planning. The capability matrix
> (`docs/capability-matrix.md`) shows **14 of 20 families as `pending`** in the
> Parity column. Per `parity-coverage-policy.md`, a family is only **supported**
> (vs experimental) once it has a current T3 manifest row — so v1.0 should not ship
> claiming 20 families while 14 are unbacked. This is the campaign to clear them.
>
> **Key framing: this is validation + recording, NOT engineering.** Every pending
> family already has its committed T1 tiny golden *and* a parity test harness (e.g.
> `granite_real_test.go`, `glm4moe_gguf_test.go`, `mistral_test.go`, and the
> `*_forward_golden.json` / `*_tiny_text_golden.json` fixtures). "Pending" means the
> **T3 real-checkpoint row in `testdata/parity_manifest.json` is empty**, not that
> code or goldens are missing. Per family: get the checkpoint → run the existing
> gate → transcribe metrics into the manifest. The matrix re-joins automatically.

## CAMPAIGN CLOSED — 2026-08-15 (`1cf8ab2` + the manifest follow-up)

> **Zero `pending` rows remain; the staleness tripwire enforces 23/23 families.** The last four
> (`gpt2`, `granitemoehybrid`, `kimi_k2`, `nemotron_h`) cleared at T3 — `full-forward-oracle`,
> `real-model-oracle` ×2, and `shared-path (via deepseek_v3)`. Nothing was demoted. Measurements and
> the per-family narrative are in `queue-release.md` §E2; what belongs *here* is the correction to
> this doc's central claim.
>
> **"Validation + recording, NOT engineering" was wrong for half the remainder.** Neither hybrid
> could LOAD its released checkpoint: granite rejected IBM's 4.56-era flat `rope_theta` (it demanded
> `rope_parameters`) and then roped a model whose config says `position_embedding_type: "nope"`;
> nemotron_h read `layers_block_type` where NVIDIA ships `hybrid_override_pattern`, and the wrong
> embedding tensor name. Granite's roped forward measured f32 cosine 0.9936 with a diverging
> continuation, against 0.9995 and exact once fixed — a real correctness bug in a family the matrix
> already listed as supported.
>
> **Why the doc could believe otherwise, and the rule that generalizes.** Both families' T1 fixtures
> are *generated* by instantiating a config on the installed transformers, so each encodes that
> version's spelling of the schema and **cannot disagree with the loader it is gating**. The doc's
> "every pending family already has its committed T1 tiny golden *and* a parity test harness" was
> true and yet did not imply what it was taken to imply. **A generated fixture proves the forward
> against itself on the schema; only a downloaded checkpoint proves the loader against the world.**
> That distinction is worth adding to `parity-coverage-policy.md` the next time it is edited — it is
> the same self-consistent-gate blind spot the policy already names, arriving through the fixture
> rather than through the oracle.
>
> **What is genuinely left is the five `experimental` rows** (`glm4_moe`, `mixtral`, `qwen2_5_vl`,
> `qwen2_moe`, `llama4_text` — all `tiny-golden`), which the §"category this doc missed" section
> below predicted and which `TestParityManifest_methodTier` now enforces rather than merely
> describes. They are labelled honestly and excluded from the supported count; upgrading them is a
> separate piece of work, not this campaign.

## Current state — RECOMPUTED FROM THE MANIFEST (2026-08-09) — superseded by the close-out above

> **The June snapshot below the fold is superseded.** The manifest is the source of truth, not this
> doc; recomputed at HEAD from `testdata/parity_manifest.json` + the staleness gate. **Drift in both
> directions, as expected — and one category this doc never contemplated.**
>
> **23 families, not 20.** Landed since June: `cohere`, `cohere2`, `gpt-oss` (and the doc's list
> omitted `gemma4`/`llama4_text` from its pending arithmetic).
>
> **Genuinely `pending`: 4, not 14.** The relay era cleared most of the campaign in passing —
> 14 rows carry Aug 4–5 `validated_at` commits.

| category | count | families |
|---|---|---|
| **T3-backed** (valid method, fresh hash) | **14** | cohere, cohere2, deepseek_v2, deepseek_v3, gemma3, gemma4, gpt-oss, llama, mellum, mistral, phi3, qwen2, qwen3, qwen3_5_moe |
| **`pending`** (empty row) | **4** | `gpt2`, `granitemoehybrid`, `kimi_k2`, `nemotron_h` |
| ⚠ **`validated` at a NON-T3 method** | **5** | `glm4_moe`, `mixtral`, `qwen2_5_vl`, `qwen2_moe` (`tiny-golden`); `llama4_text` (`tiny-golden+coherent`) |

### ⚠ The category this doc missed: `status: validated` at a method T3 does not accept

`parity-coverage-policy.md` defines **T3 valid `method` values** as exactly
`full-forward-oracle`, `real-model-oracle`, `weightDiff` (+ layer-slice), and
`shared-path (via <family>)`. **`tiny-golden` is not among them** — it is a T1 artifact recorded in
a T3 slot. Five rows sit in that gap today, and **nothing catches it**:
`decoder/parity_manifest_test.go` reads `Method` as `json.RawMessage` and **never validates it
against the allowed list**, so the staleness gate is silent (it keys on `deps_hash` freshness only).

So the honest count of rows that cannot back a "supported" claim is **9, not 4** — and four of those
five were *already* assigned stronger methods by this doc's own buckets (`glm4_moe` → weightDiff +
layer-slice, `mixtral`/`qwen2_moe` → Bucket B oracle, `qwen2_5_vl` → Bucket A full oracle). They were
recorded at the weaker method instead. **`llama4_text` is therefore not "the one weak row" the Polish
section describes — it is one of five.**

**Recommended (flagged for the owner, not decided here):** add method validation to the manifest gate
so this cannot recur silently. That is a core-adjacent test change and per Rule 1 must land *before*
the freeze, not after.

### The coupling caveat has grown: 10 → **14** families on one `deps_hash`

`f9e6caf6` is now shared by **14** families: cohere, cohere2, gemma3, glm4_moe, gpt2, llama, mellum,
mistral, mixtral, phi3, qwen2, qwen2_5_vl, qwen2_moe, qwen3. Of those, **9 are currently our
strongest rows**, 4 are the sub-T3 rows, and 1 (`gpt2`) is pending. **One core edit re-stales all
14 simultaneously** — Rules 1 and 2 are more load-bearing than when written, not less.

Other groups: `81476749` ×3 (deepseek_v2, deepseek_v3, **kimi_k2** — note the alias already shares the
parent's hash, which is exactly what makes its shared-path row sound); the remaining 6 families each
have their own hash and do not couple.

### `validated_at` vs `deps_hash` after the goldens-gated refreshes — coherent, by exception

`ed81e13` (P1) and `ca29d6c` (cap-raise) both carry `Deps-Hash-Refresh` trailers. The refresh script
updates **`deps_hash` only, preserving `validated_at`** by design. Consequence: rows say
`validated_at: b6dfbb3` (2026-08-04, an ancestor of HEAD — verified) while `deps_hash` reflects
HEAD-era source. That technically violates this doc's own Definition of Done ("`validated_at` … must
match the family's current `deps_hash`"), and it is the *sanctioned* exception: the goldens ran green
at each refresh, which is what licenses preserving the validation date. **The staleness gate is green
at HEAD — no row is stale.** Recording it so nobody "fixes" the mismatch by re-dating rows that were
never re-validated.

<details><summary>Superseded June 2026 snapshot (kept for provenance)</summary>

## Current state (2026-06-16 manifest)

**Validated (6):** `qwen3_5_moe`, `deepseek_v2`, `deepseek_v3`, `gemma4`, `phi3`
(full/real-model oracle); `llama4_text` (`coherent-generation` only — weakest tier,
see §"Polish").

**Pending (14):** `gemma3`, `gpt2`, `llama`, `mistral`, `qwen2`, `qwen3`,
`qwen2_5_vl`, `qwen2_moe`, `mellum`, `mixtral`, `granitemoehybrid`, `nemotron_h`,
`glm4_moe`, `kimi_k2`.

</details>

## The buckets (by oracle feasibility — determines method + which box)

### Bucket A — small, full bf16 oracle, fits any box (7: run-and-record)

`gemma3`, `gpt2`, `llama`, `mistral`, `qwen2`, `qwen3`, `qwen2_5_vl` (text decoder).
Sub-1B–7B checkpoints, generic forward, each with a committed `*_forward_golden.json`
— several are effectively pre-proven. **Method:** `full-forward-oracle`. **Work:**
download the smallest published checkpoint + run the existing forward test +
transcribe argmax/cosine. ~an afternoon for the whole bucket, mostly download time.

### Bucket B — mid-size MoE/hybrid, fits linux-62gb (5: one focused session)

`qwen2_moe` (~14B), `mellum` (12B), `granitemoehybrid` (H-Tiny ~7B), `nemotron_h`
(Nano-2 ~9–12B), `mixtral` (8×7B). Small ones get a full oracle; the ones whose bf16
reference won't co-reside (Mixtral) use the **int8-resident-vs-bf16 oracle already
used for DeepSeek-V2/V3** (`method: real-model-oracle`). Granite/Nemotron have
`*_real_test.go` waiting. **Work:** one big-box session, run + record.

### Bucket C — oracle infeasible → weightDiff + layer-slice (1)

- **`glm4_moe`** (106B/355B) — closer than it looks: `glm4moe_gguf_test.go` *is* the
  weightDiff test. Run it + add a bf16 layer-slice (`qwen35_realckpt` Gate-1 pattern).
  **Method:** `weightDiff` (+ slice). Run-and-record.

### Bucket D — alias families, validated by proxy (1: `kimi_k2`)

**Correction (2026-06-16): Kimi is DONE as a family** — registered (`"kimi_k2":
deepseekArchitecture`), the routing fix is in (`sigmoid := … || cfg.ModelType ==
"kimi_k2"`), and `TestKimi_descriptor` / `TestKimi_routingDefault` /
`TestKimi_textParity` (tiny golden) all pass. It is *not* "behind" and is *not*
blocked on anything.

It shows `pending` only because its dedicated manifest row is empty — but its
**numerics ride `deepseek_v3`'s forward path** (same `forward_deepseek.go`, same
`deps_hash` `3bec44…`), and that path is **already real-oracle-validated**
(Moonlight-16B-A3B). Kimi adds only two config scalars (64 heads / 384 experts) +
the routing default, all covered by the descriptor/routing/tiny-golden tests. So a
1T real checkpoint is **not** needed.

**Fill its row via shared-path provenance**, not a new oracle run:
`method: shared-path (via deepseek_v3)`, `validated_at` = the commit
`forward_deepseek.go` is at (Kimi's `deps_hash` equals deepseek's, so they validate
together), reference noting "Kimi-specific scalars covered by `TestKimi_*` + tiny
golden." This is the **cheapest** row in the campaign, not the hardest. (General
rule: any family that *aliases* another's adapter inherits that family's real-oracle
validation for the shared surface; only its config-delta needs separate coverage —
which the descriptor + tiny-golden tests already give.)

## The coupling caveat (plan around this — it's the non-obvious part)

**Ten pending families share one `deps_hash` (`…fabb…`)** — every generic-forward
family (`gemma3, gpt2, llama, mistral, qwen2, qwen3, qwen2_5_vl, qwen2_moe, mellum,
mixtral`) has no family-specific forward file, so its numerics surface is just
`core + loaders + quant`. Consequence: **one edit to a core file (`attention.go`,
`mlp.go`, `model.go`, a loader…) re-stales all ten at once.** Two rules follow:

1. **Validate the generic-forward batch together, at one commit** — don't dribble
   them in across many commits that each touch core.
2. **Do the batch as late as possible before the freeze.** If you validate early and
   then land any core change, the whole batch flips back to `pending` and the work is
   wasted. The family-specific ones (`granitemoehybrid`, `nemotron_h`, `glm4_moe`,
   plus the already-done Gemma-4/DeepSeek) have their own hashes and don't couple.

## Sequencing

1. **Freeze the core numerics surface first.** Land any last `attention.go` / `mlp.go`
   / loader change *before* starting Bucket A/B — because of the shared hash, doing it
   after voids the batch.
2. **Bucket A** (afternoon, any box) — clears 7 rows and doubles as the **T2
   small-model sweep** entries (`parity-coverage-policy.md` T2), so do both in one
   pass: record continuations *and* the manifest row.
3. **Bucket B** (one linux-62gb session) — 5 rows, int8-resident-vs-bf16 where needed.
4. **GLM** (`glm4_moe`) — run the existing weightDiff + a layer-slice; record.
5. **Kimi** (`kimi_k2`) — **cheapest, not trailing.** Record the shared-path row
   (`method: shared-path (via deepseek_v3)`) at the deepseek commit; no checkpoint,
   no oracle run. Do it whenever — it's a one-line manifest edit.

After 1–5: **all 14 cleared** → v1.0 can claim every shipped family with a current
gate, exactly what the claim-discipline rule requires.

## Definition of done (per family)

- Real checkpoint (and bf16 reference, or GGUF+safetensors for weightDiff) acquired.
- Existing gate run on the right box; method chosen per bucket.
- `parity_manifest.json` row filled: `status: validated`, `validated_at` (the commit
  the surface is at — must match the family's current `deps_hash`), `date`,
  `reference`, `machine`, `method`, `metrics`.
- `capability-matrix.{json,md}` regenerated (`go test ./decoder -run CapabilityMatrix
  -update`) — `pending` flips to the recorded method/metrics.
- T2 sweep entry added (folds in free during Bucket A).

## Polish (optional, not "pending")

`llama4_text` is validated but only at `coherent-generation` (no metrics — the
weakest method). For a uniformly strong v1.0 matrix, upgrade it to a real oracle (a
small Llama-4 checkpoint, int8-resident-vs-bf16). Not required to clear "pending," but
it's the one *recorded* row worth hardening before the release.

## What this is not

- Not new model work — adapters, loaders, and tiny goldens all exist (Kimi
  included — it's done; only its manifest row is unfilled).
- Not a CI change — T3 stays asset-gated/off-CI; this fills the manifest the
  model-free staleness test reads.
- Not a 1T-checkpoint hunt for Kimi — it rides the already-validated `deepseek_v3`
  path; its row is recorded by shared-path provenance, not a new oracle.
