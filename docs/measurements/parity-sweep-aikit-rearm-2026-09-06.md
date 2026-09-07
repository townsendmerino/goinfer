# Parity sweep — re-arming the aikit staleness gate (2026-09-06)

**Machine:** `nobara-pc`, amd64, 62.7 GB RAM. **Tree:** goinfer `eaa8bb23`, aikit **v1.36.0**.
**Command:** `EMIT_MANIFEST=1 go run ./cmd/gate parity -logdir ~/gate-logs/parity-2026-09-06`.
**Wall clock: 2h51m.** Log archived outside the worktree (a gate log inside it turns PASS into
INCONCLUSIVE) and outside `/tmp`.

## Why it was run

`testdata/parity_manifest.json` mixes a top-level `aikit_version` string into every family's
`deps_hash`. It read **`v1.19.0`** while `go.mod` read **`v1.36.0`** — hand-typed, seventeen
versions stale — so the staleness gate could not fire on any aikit change and had been vouching
for nothing. M2 (MXFP4 moved into aikit) made that acute: it relocated a family's numerics from a
file the gate watches to a dependency it was blind to.

Bumping the field to `v1.36.0` restaled **all 35 validated families**, which is itself the proof
the field is load-bearing. Clearing that with `scripts/refresh_parity_hashes.sh` would have been
the forbidden move — that script exists for provably NON-numeric edits, and an aikit bump is
exactly the case where numerics may legitimately have moved. So the gate was re-run for real.

## Result: seventeen versions of aikit drift cost nothing measurable

| cell | pass | skip | fail | wall |
|---|---|---|---|---|
| `./decoder/ ./tokenizer/` | 493 | 50 | 1 | 1h12m |
| realckpt real-model gates | 33 | 4 | 2 | 1h38m |

**15 rows emitted across 14 families.** Every one at **argmax 100%** except the single known case:

| family | method | cosine (min/mean) |
|---|---|---|
| cohere, cohere2, llama, mistral, phi3, qwen2, qwen3 | full-forward-oracle | **1.000000** |
| gemma3 | full-forward-oracle | 0.99972 |
| deepseek_v3 / deepseek_v2 | real-model-oracle | 0.99951 / 0.99924 |
| nemotron_h (×2) | real-model-oracle | 0.99767 / 0.99574 |
| granitemoehybrid | real-model-oracle | 0.99566 |
| qwen3_next | real-model-oracle | 0.98988 |
| **qwen3_5_moe** | full-forward-oracle | 0.99069 / 0.99644 — **argmax 77.5%** |

`qwen3_5_moe` is the known deliberate bandwidth trade bisected to `6d4fc79`
(`docs/queue-engineering.md`), not a new regression.

## The three failures, each dispositioned

**1. `TestParityManifest_fresh` — expected, self-inflicted.** Bumping `aikit_version` restales
every family by construction; it resolves when this sweep's own rows merge. Not a numerics result.

**2. `TestLlama4Real_gate` — the new fit guard refused it, and the guard was RIGHT.** Scout is
107.8 B elements; at int4 (0.625 bytes/element) that is ~62.7 GB resident against this box's 62.7
GB of RAM. The estimate was checked rather than trusted: no vision tensors in the GGUF, element
count matches Scout's published 109 B.

> **The gate has been passing by paging, and nothing said so.** Re-run with
> `GOINFER_NO_FIT_GUARD=1`: it PASSES in **827.8 s** and generates correctly (*"Paris. Paris is
> famous for its iconic landmarks like the Eiffel Tower, Notre"*) — while driving swap from a
> ~4 GB baseline to **35.4 GB**, falling back to 3.1 GB on exit. That is a 31 GB swap excursion to
> complete one architecture gate. The opt-out is now explicit in the test with this arithmetic
> recorded, so a green llama4 gate cannot be misread as "this model fits".
>
> Process note: the guard shipped after `go vet -tags realckpt` passed, but the realckpt gates
> were never RUN against it. This sweep is the first time they executed. Vetting a tagged build is
> not running it.

**3. `TestQwen38GGUF_weightDiff` — pre-existing, and its FIRST EVER execution.**
`docs/task-families-2026-09.md:533` records it as *"added, not yet RUN"*. It fails at `k_proj`
cosine **0.997047** against a 0.999 bar (worst tensor 0.996974, `in_proj_z`), and the test's own
message attributes it to a loader transform rather than Q8_0/Q4_K quantization.

Confirmed **not** caused by today's work: re-run on the pre-M2 tree at aikit v1.35.0 it reproduces
**bit-identically** — 0.997047 and 0.996974 to six digits. The `qwen3_5` family is
`status: experimental` / `method: tiny-golden+coherent`, so no validation claim is broken. Note it
is a DIFFERENT test from `TestQwen35GGUF_weightDiff`, whose failure is separately bisected to
`6d4fc79`; conflating the two would misattribute both.

## What this sweep did NOT re-validate — read this before trusting the green

Three families kept `status: validated` while their `deps_hash` absorbed the aikit bump. Two have
real evidence from today that simply was not stamped; one has none.

| family | ran today? | disposition |
|---|---|---|
| **gemma4** | `TestGemma4_26B_gate` (int4 + int8int8) **passed** | evidence exists; the gate does not call `emitParityRow`, so `validated_at` still reads `d64afe4` / 2026-06-16 |
| **gpt-oss** | `TestGptOssReal_gate` + `_logitParity` **passed**, plus M2's paired cell (argmax 244/244, cosine 0.999058 both sides) | same — `validated_at` still reads `25a4711` / 2026-07-27 |
| **kimi_k2** | no gate of its own | **covered, and this row originally said otherwise — see the correction below** |

**The gap this exposes is in the emitter, not the runner.** A gate that passes but never calls
`emitParityRow` leaves the manifest unable to distinguish "re-validated today" from "never
re-run" — both look identical after a merge refreshes the hash. gemma4 and gpt-oss are exactly
that case: real evidence from today, unstamped.

### Correction — kimi_k2 is covered, and this document first claimed it was not

This row originally read *"its staleness was cleared with zero evidence"* and named kimi_k2 as the
one family whose green meant only "the hash matches". **That was wrong**, and checking it rather
than repeating it is the reason it is corrected here rather than quietly edited away.

kimi_k2's `own` and `uses` sets are byte-identical to deepseek_v3's, so its `deps_hash` is
**literally the same string** — `sha256:ec563d89…` for both. Its `method` is the recognised
vocabulary entry **`shared-path (via deepseek_v3)`**, and its `reference` field records the
argument: same `forward_deepseek.go`, with Kimi's config-delta (64 heads, 384 experts top-8,
sigmoid `noaux_tc` routing) covered by `TestKimi_descriptor` / `TestKimi_routingDefault` /
`TestKimi_textParity` at tiny-golden cosine 1.000000.

And **deepseek_v3 was re-validated in this very sweep** — real-model-oracle, argmax 100%, cosine
0.99951. Refreshing a hash that is the same hash as a family re-validated the same day is a
tautology, not a gap. There is no direct gate because the checkpoint is 1T and the download is
infeasible on any box here; that is a recorded constraint, not an omission.

**What the mistake was.** "No `PARITY_ROW` emitted" was read as "no evidence", when the manifest
already models exactly this case with a shared-path method and a stated reference. The emitter
follow-up below is still worth doing — but its purpose is gemma4 and gpt-oss, whose evidence
genuinely is unstamped, **not** kimi_k2, which is modelled correctly.

Also not covered: **4 of 39 assets are unresolved on this box** (`GOINFER_QWEN3_FP8`,
`GOINFER_QWEN3_BF16`, and two others). Their gates skipped and are counted as blockers by the
runner's own convention — that count is about the assets, not the tree.

## Follow-ups this run earned

1. Make `emitParityRow` mandatory for any gate a family's `validated` status depends on. The
   tempting stronger version — have the merge REFUSE to refresh a hash for a family that emitted
   nothing — is wrong as stated: it would red kimi_k2, which is modelled correctly as a
   shared-path family and has no gate of its own by design. The rule has to exempt a family whose
   `deps_hash` equals that of a family re-validated in the same run, which is checkable.
2. `TestQwen38GGUF_weightDiff` needs a disposition — it is a first-run red on an experimental
   family, and its 0.999 bar has never been calibrated against what that loader actually achieves.
3. The Metal half of the sweep has never run here; `C3` remains owed on `macbook-arm64`.
