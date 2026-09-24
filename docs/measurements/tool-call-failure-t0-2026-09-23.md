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

Machine `nobara-pc` (RTX 2070 SUPER, driver 595.91.07). Every cell used ONE `serve` binary, built 2026-09-23 21:15 from the tree
carrying the R6 flash-decode default (so the lane is active on the deeper turns) — the per-cell `header.commit` differs
(`cfb76654` / `7e642e22`) only because the harness was launched at different commits; the binary did not change. Raw cells,
per-sample classification and the run summaries: [`tool-call-failure-t0-2026-09-23/`](tool-call-failure-t0-2026-09-23/).
Serving quant is int4 (the GGUF's q4_k_m weights through the W4A8 path) unless the row says int8. 10 greedy + 300 sampled
(T=0.7, seeds `1000*turn+i`) requests per cell; the MoE row is smaller (below).

| cell | attempted | parsed | unknown name | unparsed | truncated | **unwrapped** | prose | failures / attempted (95% Wilson) | verdict (pre-reg) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder 0.5B int4 | 0 | 0 | 0 | 0 | 0 | **246** | 54 | 0 / 0 | UNRESOLVED |
| Qwen2.5-Coder 0.5B int8 | 0 | 0 | 0 | 0 | 0 | **254** | 46 | 0 / 0 | UNRESOLVED |
| Qwen2.5-Coder 1.5B int4 | 0 | 0 | 0 | 0 | 0 | **223** | 77 | 0 / 0 | UNRESOLVED |
| Qwen2.5-Coder 1.5B int8 | 0 | 0 | 0 | 0 | 0 | **237** | 63 | 0 / 0 | UNRESOLVED |
| Llama-3.2 1B int4 | 157 | 148 | 7 | 2 | 0 | 0 | 143 | 9 / 157 = 5.7% (3.0–10.5%) | AMBIGUOUS |
| Qwen2.5 7B int4 | 111 | 97 | 10 | 3 | 1 | 0 | 189 | **14 / 111 = 12.6% (7.7–20.1%)** | **RED** |

Greedy (n=10 per cell) is a sanity row only, as pre-registered: 0.5B/1.5B 8–9 unwrapped; llama3-1B 6 parsed / 4 prose; 7B 2 parsed /
8 prose. Nothing is resolvable at n=10.

### What the pre-registered rules say

1. **"Documentation fix only" — NOT met.** It needed every small-model cell GREEN; none is (four UNRESOLVED, one AMBIGUOUS).
2. **"T1 ships without further argument" — MET, by the 7B.** Point estimate 12.6% > 10% on a size a harness would realistically use.
   Read it with the two caveats below before leaning on the number.
3. **The second, independent rule — applies by its intent, not its letter.** It was written for "headline < 2% but
   `any_unusable` ≥ 10%". The four Qwen2.5-Coder cells have a headline that is *undefined* (0 attempted calls), not < 2%, so the literal
   condition does not hold; but every intended call in them was `unwrapped` (any_unusable = 100% of intended calls). I apply the rule as
   written for: **T1 as scoped is insufficient for those models** — a union grammar armed on `<tool_call>` never arms, because the
   model never emits it. Owner can dispute the application; the counts are above.

### What the failures are (this changes what T1 would buy)

- **The Qwen2.5-Coder 0.5B and 1.5B never wrap their calls.** 0 wrapped calls in 1,200 sampled requests across four cells; they emit
  bare `{"name": "read_file", "arguments": {...}}` and goinfer's chatml parser (`chat/tools.go`, `parseChatMLTools`) only recognises the
  `<tool_call>` wrapper, so the harness receives prose. It is not the server: the same server with the 7B wraps and parses, and it is
  not the prompt length or history — **control (inline one-off, 1.5B int4, T=0.7, seeds 0–19, no history): 2 tools + "Read notes.txt"
  → 0 wrapped / 13 bare / 7 prose; 2 tools + the fixture's turn-1 task → 0 / 12 / 8; 12 tools + "Read notes.txt" → 0 / 11 / 9; 12 tools +
  turn-1 task → 0 / 8 / 12** (0 of 80). The chatml renderer matches Qwen's published tool prompt and renders history calls with the
  wrapper. Different fix from T1: a parser that also accepts bare call JSON naming a supplied tool, or forcing the wrapper.
- **The 7B's failures are mostly invented tool names, not malformed JSON.** 10 of 14 are `unknown_name` (8× `git_commit`, plus
  `git_add`/`git_checkout`/`git_merge`); 3 unparsed + 1 truncated. That is exactly what T1's name trie prevents by construction. A
  further 10 of its 97 parsed calls carry arguments that miss a required key or have a wrong top-level type (informational column) —
  T1's per-tool schema would prevent those too: 24 of 111 attempted calls (21.6%) are not clean calls to a real tool with valid args.
- **llama3-1B's failures are garbage tool names** (`init 5.1.2`, `let internal/ratelimit/limiter.go`, `TestLimiter_Burst`) and 2 broken
  bodies, spread across turns; 18 of 157 (11.5%) are not clean calls once the 9 bad-args parses are counted.
- **Prose is common and legitimate** (7B 63%, llama3 48%, at steps where the tool result already answers the task). These are the
  baseline for the non-forcing gate in T6.

### Caveats that bear on the decision

- **The 7B's 12.6% is one prompt.** 9 of its 14 failures are on turn 9 (6 unknown-name + 3 unparsed), 3 on turn 10, 1 on turn 7, 1 on
  turn 5. The 30 samples of a turn share a prompt, so they are not independent evidence about "turns in general", and the Wilson
  interval above (which treats them as independent) is narrower than the real uncertainty. What is established: there exist ordinary agent
  turns on which a 7B invents a tool roughly one time in three. What is not: a population failure rate.
- **No system prompt was sent.** Real harnesses send their own; Qwen's published template injects a default one when none is given and
  goinfer's renderer does not. Not tested whether that changes the Coder models' wrapper behaviour.
- **First-order failures only** (blind spot in the harness docstring). One transcript, one tool schema, one backend (CUDA).
- **1.5B int8's first attempt was discarded, not re-rolled selectively:** it died with a 503 from the server's swap guard when I ran a
  `-race` build alongside it, produced no data, and was re-run from scratch with the same settings.

### MoE (Qwen3.6-35B-A3B)

_In progress at the time of the dense write-up; see the next section when filled._
