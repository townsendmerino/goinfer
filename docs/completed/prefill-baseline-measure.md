# Task (goinfer, MacBook): CPU prefill vs Ollama — measure the missing number

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.

> **Status, doc-reviewed 2026-09-13 — SUPERSEDED, archived.** The premise this doc opened with
> ("there is no current CPU prefill-vs-Ollama number anywhere in the record") stopped being true
> five days after it was written. `6f54fbf3` (2026-09-01, "the first peer CPU-prefill comparison,
> weight-matched") produced the goinfer-vs-Ollama CPU prefill table it asked for — same
> methodology (weight-matched GGUF, `num_gpu:0`, unique prompt per cell so Ollama's aggressive CPU
> cache can't flatter it, `LINEAR_FIT_INVALID` handled honestly) — recorded at
> `docs/measurements/cpu-peer-prefill-2026-09-01.md`: 2.98x behind at K=512, narrowing to 1.80x at
> K=3900. `de0cc654` (2026-09-05) superseded that row after aikit's S-01 register-blocked int4
> tile: `docs/measurements/cpu-peer-prefill-2026-09-05.md` has goinfer at 1.54x behind at K=512
> and **0.91x — AHEAD — at K=3900**, whole-curve marginal ratio 0.86x. Both numbers are the
> canonical CPU-prefill-vs-Ollama figures in `docs/benchmarks.md` §A ("Apple Silicon CPU
> prefill") today.
>
> **What was not executed to the letter, and is not owned by a live doc:** both measured rows are
> **1.5B only** — 0.5B CPU prefill vs Ollama is still unmeasured (the 2026-09-01 doc says so
> under "Not claimed"). Neither row used this doc's exact prescription: n≥6 paired prompts per
> length (the actual runs used 4), prompt lengths 541/2048/8192 (the actual runs used
> 512/1024/2048/3900), a separately-labeled cold first-request cell, or the specific
> today-vs-Aug-22-baseline attribution cell this doc asked for (the 2026-09-05 doc instead
> attributes to aikit v1.31.0-vs-v1.34.0 across a four-day window of unrelated goinfer changes
> too, and says plainly it "should not be quoted as" an isolated A/B). None of these gaps are
> tracked in a live queue item as of this review — flagged here rather than filed, since the
> qualitative finding this doc was chasing (CPU prefill is no longer clearly behind Ollama; it is
> ahead at depth on the flagship 1.5B cell) is already established and cited.

> **For:** Claude Code, in `~/tmcode/goinfer`, on the M1 Pro. Written 2026-08-26. Measurement
> only — this is the Step 0 of any future prefill campaign, and no optimization happens here.
> **Prior art (mandatory):** `docs/ollama-chase.md` §3b — CUDA prefill is 4.7x behind with a
> known 61% GEMV / 39% attention split and a format-imposed component; CUDA is NOT this task.
> `docs/benchmarks.md`'s serve-vs-decode note (prefill amortization on the sequential
> full-logits path). The abandoned depth-2048 cell in `docs/completed/task-w4a8-neon-bandwidth.md`
> (CPU long-prompt prefill pain, measured by accident). `--metal-fast-prefill` (gated,
> non-bit-identical, not this task). The qwen35 MoE family has NO batched prefill — dense
> models only here. CLAUDE.md's measurement section and aikit rules 3+7 govern method.

## Why now

Every campaign since Aug 22 was decode-only, but three of its levers are shared with the
batched forward — A1's attention kernels (minus threading, which batched deliberately keeps
serial), the W4A8 split-half kernel at M>1 on resident arm64, and the W8A8 LM head — so CPU
prefill has probably improved for free and nobody has measured it. There is no current CPU
prefill-vs-Ollama number anywhere in the record. This task produces it, plus the attribution.

## Cells

Prefill rate = client-timed TTFT ÷ prompt tokens, both engines on identical GGUF weights,
`bench_peer`-style discipline (quiet box, server restart between cells, bench-local files).
Define the standard cell as the SECOND request after server start — weights warm, KV empty —
and record the first request separately as a labeled cold cell, never averaged in.

- Models: 0.5B and 1.5B int4 (the ledger's flagship cells).
- Prompt lengths: 541 and 2048; add 8192 only if goinfer-side wall-clock is tolerable — if it
  isn't, that fact is itself the result for that cell, recorded as such.
- Paired per prompt (rule 7), n ≥ 6 prompts per length, win counts alongside deltas.
- Ollama same cells, same prompts, same timing method.

## The attribution cell — answers "did the decode campaigns move prefill?"

Check out the pre-campaign baseline commit (the Aug 22 state, e.g. the commit the original
diagnosis ran against), build it, and run the same goinfer cells same-day on the same box.
Today-vs-baseline, paired, isolates what A1 + W4A8 + the head fix gave prefill for free —
turning "almost certainly improved, unmeasured" into a number. Label the baseline build
clearly; do not let its binary or artifacts survive the session.

## Deliverable

A measurement doc in `docs/measurements/` (provenance per `benchmarks.md`: box, quant,
commit, method) with: the goinfer-vs-Ollama prefill table, the today-vs-baseline attribution
table, and the cold-cell numbers. No component split unless a result is surprising enough to
demand one — and then only as a recorded recommendation for the next campaign, not work done
here. Relay the final table for the ledger, which gains a prefill row from this.

## Not in scope

Any optimization or code change; CUDA and Metal; the MoE/35B family; `--metal-fast-prefill`
changes; the O(L²) attention (its cost will show in the 2048/8192 cells — measuring it is the
point, fixing it is a campaign).
