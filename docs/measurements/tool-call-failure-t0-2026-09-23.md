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

### MoE (Qwen3.6-35B-A3B) — NOT RUN

The cell was attempted and could not be completed on this box without defeating a safety guard, so **there is no MoE number** and the
T0 matrix has no MoE row. What was learned trying:

- The q4_k_m GGUF is refused by the load fit guard (51.7 GB peak on a 62 GB box).
- The canonical int4 `.giw` loads, but WITHOUT `GOINFER_MOE_CACHE_EXPERTS=1` (C′, opt-in) the resident upload OOMs the 8 GB card and
  serve silently falls back to CPU: one 754-token request took **235 s**, i.e. 20+ hours for the pre-registered 320.
- With C′ on it runs on the GPU (6.8 GB VRAM; the flash-decode lane correctly stays off for C′, no message) but with no session reuse
  each request re-prefills: ~26 s at 754 prompt tokens, ~71 s at 1.8K, and the transcript reaches 3–4K. A reduced cell (10 samples per
  turn, ~2 h) was launched detached.
- It died at the start: the server's **swap guard tripped** (baseline 18.8 GB swap in use, +0.66 GB over its +512 MB limit once the
  ~22 GB model was resident) and refused requests, a 503. That guard is a deliberate protection, not a fault, and the box already
  carries 15–19 GB of swap from other work; I did not raise the threshold or evict anything to get around it.
- Anecdote, NOT data: three hand-sent turn-1 requests (n=3) returned a `<think>` block followed by a parsed `read_file` call, so this
  family does wrap its calls under `auto`. Nothing here supports a rate.

To finish it: free the box's swap (or run on a machine where the 35B fits), then
`GOINFER_SERVE_CUDA=… python3 scripts/bench_tool_failure.py out.json --model MoE-35B-A3B --quant auto --samples 10 --temp 0.7`.
The reduced 10-samples-per-turn size would be a disclosed deviation from the pre-registered 30.

## Peers: how llama.cpp and Ollama handle the same failure (2026-09-24)

**Read from source.** llama.cpp `427291b` (2026-09-05, `~/mycode/peers/llama.cpp`); Ollama v0.32.5 `tools/tools.go` +
`tools/template.go` (fetched at the tag).
- **llama.cpp** — for a Qwen2.5-style template: lazy grammar, trigger word `<tool_call>`, body = choice of supplied tool-name
  literals + that tool's parameter schema; `required` applies it from token 1. Output before the trigger is content. So it already
  ships what T1+T2 propose, and it does NOT recover a bare, unwrapped call on this template. For Qwen3-Coder (a different XML call
  form) it makes the first `<tool_call>` optional because "Qwen3-Coder models may occasionally omit" it, arming on the complete
  `<function=NAME>` of a supplied tool so the model cannot invent a name there.
- **Ollama** — no constraint in the tool parser. The tag is the template text before `{` in its ToolCalls block (`<tool_call>` for
  qwen2.5-coder); after the tag it scans for the longest supplied tool name and then an arguments object, so a slightly malformed body
  with a real name still parses and an unknown name yields no call. With no tag in the template, the tag is `{` and a call is only
  parsed if the output's first non-space byte is `{`/`[`. A bare call on a tagged template is returned as content.

**Measured (control, Ollama 0.32.5, its own library templates, T=0.7, seeds 0–19, 12 tools, no history, `auto`):**

| model (Ollama tag) | prompt | parsed | bare JSON | prose |
|---|---|---|---|---|
| qwen2.5-coder:1.5b-instruct-q4_K_M | "Read notes.txt" | 0 | 3 | 17 |
| qwen2.5-coder:1.5b-instruct-q4_K_M | fixture turn 1 | 0 | 6 | 14 |
| qwen2.5-coder:7b-instruct-q8_0 | "Read notes.txt" | 0 | 19 | 1 |
| qwen2.5-coder:7b-instruct-q8_0 | fixture turn 1 | 0 | 19 | 1 |

Ollama's template already strengthens the instruction ("… within <tool_call></tool_call> with NO other text") and the Coder models
still omit the wrapper; so do the Coder **7B** at q8. The Coder family drops the wrapper at every size tested, on both engines — the T0
7B row is Qwen2.5-7B-*Instruct*, which wraps; that is a different model, not a size effect. 80 requests; a control, not a matrix row.

## Follow-up A — lenient bare-call parser (chatml/mellum2). Pre-registration, written BEFORE the code

Owner decision 2026-09-24: option 3 (parser first, then T1). This section is the plan and its gates; results are appended below it.

**Change.** For the two wrapper families that stream (chatml, mellum2), an output that yields no `<tool_call>` calls is re-read: if its
first non-space byte starts a JSON object with a string `name` that is EXACTLY one of the supplied tool names and an object-valued
`arguments` (or `parameters`), it is one call, with an empty lead. Design choices, made now:
- **First object only.** The 0.5B was seen emitting LISTS of speculative calls (`read_file`, `write_file`, `edit_file`… one after
  another). Executing all of them would be worse than the prose it replaces; the wrapper form still accepts every call, unchanged.
- **Anchored at the start of the output.** Prose followed by a JSON object is NOT re-read. Conservative on purpose; it may leave some
  calls unparsed, which the live re-measure will show.
- **The streamer holds any output whose first non-space byte is `{`** (G21's invariant: a streamed byte cannot be unsent, and a bare
  call's lead is empty, so nothing of it may be emitted as prose). Such an output that is NOT a call is delivered at the end instead, same
  bytes.
- Only `ParseToolCallsFor(out, tools)` gains the leniency; `ParseToolCalls(out)` is unchanged. mistral, llama3, gemma4: untouched.

**Gates (pass/fail; fixed now).**
- **G1 non-forcing / parity.** For every output whose first non-space byte is not `{`, `ParseToolCallsFor` equals `ParseToolCalls`
  (differential test over generated strings and a fuzz target). Red-before-green: a deliberately over-eager variant must fail it.
- **G2 no invented calls.** Bare JSON with an unknown name, a non-string name, missing/non-object arguments, an array, or trailing junk
  that is not a complete object stays prose.
- **G3 streaming.** The byte-by-byte prefix invariant of `TestProseStreamerMatchesParser` holds for bare-call and bare-JSON-prose outputs
  including leading whitespace.
- **G4 live re-measure, same harness, same seeds, same serve build.** (a) **7B int4 and llama3-1B must reproduce their archived class
  counts EXACTLY** (97/10/3/1/189 and 148/7/2/0/143) — they never emit bare JSON, so any change is a regression, not an effect.
  (b) **0.5B and 1.5B int4:** `prose` must equal the archived 54 / 77 (every change must come out of `unwrapped`); `parsed` becomes >0; I
  report parsed, unknown-name, args-invalid and on-intended. If generations do not reproduce bit-for-bit under the same seed, that is
  stated and (a)/(b) are read with that tolerance, not silently loosened.

**Not claimed.** That the calls it recovers are GOOD. The 0.5B/1.5B's bare calls were seen with placeholder paths (`path/to/file`) and
irrelevant paths; recovering the form does not recover the judgement. `on_intended` and `args_invalid` are reported so this is visible.

### Follow-up A — results (2026-09-24)

Change: `0b5da39e` (`chat.ParseToolCallsFor`, wired into all four tool surfaces). New build = that commit, CUDA `serve`;
old build = the T0 binary. Same harness, same seeds, same box. Raw data: [`tool-call-failure-t0-2026-09-23/followup-a/`](tool-call-failure-t0-2026-09-23/followup-a/).

**G1–G3 PASS** (`chat/bare_tool_call_test.go`), each shown able to fail: an over-eager variant (`{` anywhere) reddens G1, dropping
the tool-name check reddens G2 and the fuzz target, a streamer that does not hold reddens G3. `FuzzParseToolCallsFor`: 60 s,
978K execs, clean; added to `fuzz-weekly.yml`.

**G4(b) PASS** — the two models the change targets, T=0.7, 300 samples each:

| cell | parsed | unwrapped | prose | unknown / unparsed / truncated | args invalid | on intended tool |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder 0.5B int4, T0 → now | 0 → **236** | 246 → 10 | 54 → **54** | 0 / 0 / 0 | 1 | 25 of 236 |
| Qwen2.5-Coder 1.5B int4, T0 → now | 0 → **219** | 223 → 4 | 77 → **77** | 0 / 0 / 0 | 0 | 21 of 219 |

Prose is unchanged to the sample, so every recovered call came out of `unwrapped`: the change converts calls and does not touch
answers. The residual `unwrapped` rows are bare JSON the parser correctly refuses — names that are not supplied tools
(`rate_limit`, `rate_limiter`, `sleep`, `allow`) or a body that does not decode. **The recovered calls are mostly the wrong tool**
(on-intended 9–11%, against 27% for the 7B's parsed calls): the form is fixed, the judgement is not, as pre-registered.

**G4(a) PASS, with one disclosed change to the comparison.**
- 7B: the new build reproduces the archived T0 counts EXACTLY (97 / 10 / 3 / 1 / 189 at T=0.7; 2 / 8 greedy). Pass as written.
- llama3-1B: the new build gave 173 / 9 / 2 / 1 / 115 against the archive's 148 / 7 / 2 / 0 / 143, so it fails the literal bar. But
  the archive does not reproduce **from its own binary**. The T0 binary run today, same harness and seeds: greedy is byte-identical
  run to run and old vs new (ids stripped); turn-1 sampled 30/30 byte-identical old/old and old/new; and the full old-binary llama3
  cell today matches the new build **310/310 per sample**. Only the archived run differs (17/30 at turn 1, where there is no history).
  So the pre-registered comparison against the archive is invalid for this cell, and the valid comparison, same-day old binary vs
  new build, passes with no difference. Forcing the R6 lane off did not bring the archive back (it is not the cause).
- What made the archived llama3-1B run differ is **not established**. The Qwen cells and the 7B did reproduce their archives
  (prose counts to the sample; 7B in full), so it is specific to that run. Both readings of llama3-1B (5.7% and 6.5%, CIs
  3.0–10.5% and 3.8–11.0%) are AMBIGUOUS under the T0 rule, so the T0 verdict does not change.
