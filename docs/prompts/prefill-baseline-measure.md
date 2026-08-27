# Task (goinfer, MacBook): CPU prefill vs Ollama — measure the missing number

> **For:** Claude Code, in `~/tmcode/goinfer`, on the M1 Pro. Written 2026-08-26. Measurement
> only — this is the Step 0 of any future prefill campaign, and no optimization happens here.
> **Prior art (mandatory):** `docs/ollama-chase.md` §3b — CUDA prefill is 4.7x behind with a
> known 61% GEMV / 39% attention split and a format-imposed component; CUDA is NOT this task.
> `docs/benchmarks.md`'s serve-vs-decode note (prefill amortization on the sequential
> full-logits path). The abandoned depth-2048 cell in `docs/task-w4a8-neon-bandwidth.md`
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
