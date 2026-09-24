# T0 — how often does an `auto` tool call fail to parse? (2026-09-23)

Measurement item T0 of [`task-tool-grammar-union-2026-09.md`](../tasks/task-tool-grammar-union-2026-09.md). It gates T1
(the union grammar). Harness: `scripts/bench_tool_failure.py`.

## Pre-registration (written and committed BEFORE any matrix run)

**Question.** With the 12-tool schema `scripts/w4_transcript_base.json` (the schema real agents send) and
`tool_choice: auto`, what fraction of the calls a model *attempts* reach the harness unusable?

**Design.** One `serve` per (model, quant). The 10-turn transcript is replayed with a NAMED, constrained, greedy
tool_choice to build a per-model history (so history is well-formed and on-distribution). At each turn the history
so far is then sent with `auto`: once greedy, and 30 seeds at T=0.7 — 10 greedy + 300 sampled requests per cell.
Teacher-forced, so one bad sample cannot compound. Smoke-tested on the 0.5B at 2 samples/turn to prove the harness
runs and classifies (40 requests, output discarded, not counted); that smoke run is where the `unwrapped` class was
found — it was **not** in the doc's definition and is added as a separate column, not folded into the headline.

**Cells.** int4: Qwen2.5-Coder 0.5B, Qwen2.5-Coder 1.5B, Qwen2.5 7B, Llama-3.2 1B (the no-opener llama3 form,
which T2 flagged as a design question); int8: 0.5B and 1.5B (7B int8 does not fit the 8 GB card — not run,
disclosed). MoE (Qwen3.6-35B-A3B q4_k_m via C′) after the dense cells, if its template resolves to a
constrainable family.

**Headline statistic.** `failures / attempted`, where attempted = parsed + unknown_name + unparsed + truncated
and failures = unknown_name + unparsed + truncated. Point estimate with a 95% Wilson interval, per cell, greedy and
T=0.7 separately. Doc definition kept: `unparsed` = the family opener is present and the body did not parse.

**Decision rule (the doc's, made operational).**
- A cell is **GREEN** if its point estimate is < 2% and the CI upper bound is < 4%; **RED** if its point estimate is
  > 10%; otherwise **AMBIGUOUS**. A cell with < 30 attempted calls is **UNRESOLVED** and cannot count green.
- **Documentation fix only** (write the number into `docs/server.md`, stop) iff every small-model cell
  (0.5B, 1.5B, llama3-1B, both quants, both temperatures) is GREEN.
- **T1 ships without further argument** iff any cell a harness would realistically use is RED. Realistic =
  1.5B and above; the 0.5B is reported but **cannot alone** trigger this (a harness user does not pick a 0.5B
  for agent work — a judgement made here, before the numbers, and disclosed).
- Anything else → the judgement band; the number goes to Francis, no default.

**Second, independent pre-registration.** `unwrapped` (bare call JSON, no wrapper) is reported as its own rate,
and `any_unusable = failures + unwrapped` over (attempted + unwrapped). If any cell has headline < 2% but
`any_unusable` ≥ 10%, the finding is **"T1 as scoped is insufficient there"**: a union grammar armed on the
wrapper never fires when the model omits the wrapper, so T2's arming design must be revisited before T1 is
built. This can disagree with the headline on purpose.

**Not measured, stated up front.** A second, malformed call in a parallel-call turn (the server drops raw text once
one call parsed) — the rates are a floor. Greedy has n=10 per cell, so a greedy rate below ~30% is not resolvable
per cell and greedy is read as a sanity row, not evidence. One transcript (coding agent, one tool schema) — no other
harness's prompt shape, no chat-template variation, no system prompt. Both arms run the shipped defaults, so the
R6 flash-decode lane is active on the deeper turns (it is not bit-identical to exact attention).

## Results

_(filled in after the runs)_
