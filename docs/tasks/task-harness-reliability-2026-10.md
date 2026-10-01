# Task: harness breadth and tool-call reliability, both dialects, whole sessions (2026-10)

> **Status: SCOPED 2026-10-01, not started.** Merges two proposals made the same day: harness breadth (how many wire
> dialects the coding-agent harnesses speak, and whether `serve check` predicts both) and tool-call reliability
> widened past T0 (more traces, whole sessions, more families and backends). Gates run cheapest first and stop at
> the first negative one (§2). To start after v0.20.0 is tagged.

## 1. What already exists, so it is not re-measured

- **T0 and the union grammar** (`docs/tasks/task-tool-grammar-union-2026-09.md`, T0–T7 done 2026-09-24). T0
  ([`tool-call-failure-t0-2026-09-23.md`](../measurements/tool-call-failure-t0-2026-09-23.md), harness
  `scripts/bench_tool_failure.py`) replayed a 12-tool agent schema (`scripts/w4_transcript_base.json`) and a 10-turn
  transcript with `tool_choice: auto` on Qwen2.5-Coder 0.5B and 1.5B, Qwen2.5 7B and Llama-3.2 1B, 10 greedy plus 300
  samples at T=0.7 per cell: Qwen2.5-7B had 14 of 111 attempted calls unusable. The union grammar took that to 0, with
  prose turns byte-identical (630/630; [`tool-union-2026-09-24.md`](../measurements/tool-union-2026-09-24.md)).
- **What T0 did not establish**, in its own record: no MoE, one transcript, CUDA only. Every turn was replayed from a
  fixed, correct history (teacher-forced), so a bad call could not derail what followed. Still open from the union
  task: llama3 calls that follow prose, and the union on a speculative-decoding server.
- **Measured harness successes, one per dialect.** `docs/integrations/opencode.md:117`: Qwen2.5-7B-Instruct q4_k_m on
  nobara-pc (CUDA, 8 GB) "ok — real tool calls, real completion" (OpenAI dialect). `docs/integrations/claude-code.md`:
  an agent loop verified end to end on 2026-09-02 (Anthropic dialect). Re-read both at the start; correct this section
  if either has changed.
- **`serve check`'s tool rows** (`internal/servecheck/check.go`): `tools, OpenAI` (:460) and `tools, harness-scale`
  (:547, using `harnessScaleTools()` at :572), both on `/v1/chat/completions`. There is no `/v1/messages` row, so the
  dialect Claude Code uses cannot be predicted per checkpoint; the 2026-09-02 result was a hand measurement.

## 2. Gates, cheapest first

### Gate 0 — desk work: how many wire dialects do the harnesses speak?

For each of Pi, OpenCode, Hermes, OpenClaw, Codex, Claude Code, Oh My Pi and Cline, from its own source or docs (cite
the file or URL and the version or commit read): the endpoint it posts to; whether tool schemas use OpenAI `tools[]`
or Anthropic `tools`; whether it requires streaming; and any header, auth scheme or request field it needs that
`serve` does not provide. Report the count of genuinely distinct dialects and name any harness needing something
missing.

### Gate 1 — an Anthropic-dialect harness-scale row in `serve check`

A `/v1/messages` twin of `tools, harness-scale`, reusing `harnessScaleTools()` translated to the Messages tool shape,
with the existing row's three outcomes: ok, skip when the model declines to call (not a failure), fail. Registered in
the same change as its tests. **It is not a check until it reproduces the two known results:** ok against the 7B on
the box that produced the 2026-09-02 numbers, and skip against a 1.5B.

### Gate 2 — the model tables, both dialects

Run both harness-scale rows against `phi3-mini-4k`, `granite-4.0-h-tiny`, `gpt-oss-20b` and `gemma-4-26b-a4b`
(opencode.md:118), and the corresponding rows of claude-code.md's table, from `~/models` on the box doing the run.
Record per checkpoint: the outcome per dialect, the hardware and backend, the resident footprint, and whether the
outcome was memory fit or capability. opencode.md's own finding holds: memory fit and tool-calling capability are
separate axes, so a "not measured" with no box named says nothing. A checkpoint that does not fit the box is
recorded as never reached, not as a capability result.

### Gate 3 — T0 widened: several traces, whole sessions

- **Traces:** at least three real harness traces (opencode, Claude Code and one more from Gate 0's list), each a
  recorded multi-turn session with 5 or more tools and `tool_choice: auto`, kept under `docs/measurements/`.
- **Two modes per trace.** *Teacher-forced*, as T0: each turn from the recorded history, counting parsed, unparsed,
  unknown-name and truncated calls. *Free-running*: the session continues from the model's own previous turns, with
  tool results simulated from the trace, until it finishes or fails. The free-running mode is what T0 could not
  measure.
- **Cells:** the checkpoints of Gate 2 that fit, plus one MoE (gpt-oss-20b or Gemma 4 26B-A4B), and the newer template
  families (Harmony, Gemma 4, Qwen3.5); CUDA on nobara-pc, and Metal on a quiet Mac. Both dialects where the harness
  speaks both.

## 3. Pre-registration

**Gate 0:**
- **Two dialects, nothing missing:** the breadth gap is a documentation gap. Write it up and close Gates 0–2 with no
  code; Gate 3 still runs.
- **A harness needs a field or endpoint `serve` lacks:** that harness becomes its own scoped item with the field named,
  not started inside this task.

**Gate 2:**
- **Skips under harness-scale in both dialects:** a size result, not a server defect. Record the size, as the 0.5B and
  1.5B rows already do.
- **Passes one dialect and fails the other:** the interesting case, and the only one that justifies further work here.
  Report it with the request and response bodies.

**Gate 3:**
- **Session-killing failure**, defined now: in free-running mode, the session ends without completing its task
  because of a tool call that could not be parsed, named an unsupplied tool, had invalid arguments, or because the
  model looped (the same call 3 times in a row) or ran past 2× the trace's turn count.
- **Headline per cell:** unusable calls ÷ attempted (teacher-forced, as T0, with a 95% Wilson interval) and
  session-killing failures ÷ sessions (free-running).
- **Decision:** for a cell a harness would realistically use (Gate 2 ok), if session-killing failures are **0** and
  teacher-forced unusable calls are under 2%, record the cell as reliable and move on. Any session-killing failure
  there is a finding, reported with its transcript, and becomes its own scoped item if it is a server or grammar
  defect rather than the model's choice.

**Not claimed:** none of this measures what gates adoption. Speed and reliability are separate questions, and a clean
result here shows only that tool calls are not what breaks.

## 4. Conventions and cost

Checkpoints from `~/models` on the box doing the run; counts are reported with the regime (backend, quant, greedy or
T=0.7, seeds), and spread where there is any. These are counts, not timings, but at T0's size (310 requests per cell),
times several traces, models and two dialects, Gates 2 and 3 are night-queue work on nobara-pc; Mac cells need an
idle machine. Leave changes uncommitted for review where the brief asks for that.

**Doc hygiene, only if quick:**
- opencode.md's "Retiring this page" (:126) says the page stays until a registry checkpoint has a measured
  harness-scale ok. If Gate 2 satisfies that, say so; do not retire the page inside this task.
- Both integration pages use "honestly", "honest picture" or "honest state" (opencode.md :105, :109, :122). Strike the
  words and say the same thing directly ("the narrower picture", "what is measured").
