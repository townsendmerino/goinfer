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

## Current state (2026-06-16 manifest)

**Validated (6):** `qwen3_5_moe`, `deepseek_v2`, `deepseek_v3`, `gemma4`, `phi3`
(full/real-model oracle); `llama4_text` (`coherent-generation` only — weakest tier,
see §"Polish").

**Pending (14):** `gemma3`, `gpt2`, `llama`, `mistral`, `qwen2`, `qwen3`,
`qwen2_5_vl`, `qwen2_moe`, `mellum`, `mixtral`, `granitemoehybrid`, `nemotron_h`,
`glm4_moe`, `kimi_k2`.

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
