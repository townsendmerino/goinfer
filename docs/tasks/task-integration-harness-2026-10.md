# Task: integration and harness work — context first, and what no doc owns yet (2026-10)

<!-- citations-at: 6e50c60941e7 -->

> **Status: STARTED 2026-10-12 (§0a).** Steps 1 and 3 are built and a gap nobody had listed is fixed: Claude Code
> 2.1.268 could not complete a turn against goinfer (H-e). Its real first request is 19,692 tokens, not 6–10K, which
> moves the thesis's numbers (§0a). **Was: SCOPED 2026-10-12.** A map across the docs that already own integration work, an order
> for them, six items none of them owns (I1–I6; I7 added 2026-10-12 on the owner's word), and amendments the 2026-10-12 research supports for four existing
> docs (§5). It re-scopes nothing owned elsewhere. Facts about third-party harnesses are a desk read of each project's
> own docs or source on 2026-10-12 (§Sources). Harnesses ship weekly, so each such fact is a lead until a trace
> confirms it.

## 0. Where integration work stands (read at `6e50c60941e7`)

| item | owner | state |
|---|---|---|
| Wire-dialect census of eight harnesses | `task-harness-reliability-2026-10.md` Gate 0 | **done** 2026-10-09 (`docs/measurements/harness-dialects-2026-10-09.md`): three dialects, gaps H-a to H-d |
| H-a Responses tool shape (stopped Codex outright) | same | **fixed**, merged 2026-10-11; unrunnable tool types are dropped and named in `X-Goinfer-Tools-Dropped` |
| H-b `/v1/models` context fields | same | **fixed**: `context_window`, `context_length`, `max_model_len` (`internal/serveapp/openai.go:533`, `:537`, `:538`) |
| H-c Anthropic overflow wording | same | **open, as a decision**; I1 proposes the rule |
| H-d Anthropic keep-alive form | same | open pending a trace; §5 has a lead |
| `/v1/messages` harness-scale row | same, Gate 1 | **next**; `serve check`'s rows still end at `tools, harness-scale` (`internal/servecheck/check.go:551`) |
| Model tables in both dialects; T0 widened to whole sessions | same, Gates 2–3 | open |
| Union tool-call grammar | `task-tool-grammar-union-2026-09.md` | **shipped** 2026-09-24, default-on (Qwen2.5-7B unusable calls 14/111 → 0) |
| Mellum2.1 harness-scale | `task-mellum21-2026-10.md` Gate 2 | **ok** on the 8 GB card through expert streaming (45 of 64 slots), 7.4 GB VRAM, 71 tok/s: the second measured harness-scale success, and the first MoE |
| In-process facade (`llm`) | `task-embed-and-harness-ux.md` §2 | unstarted, unblocked |
| Fill-in-the-middle and editors | `task-editor-fim-2026-09.md` | unstarted; F0 gates it |
| Desktop sidecar prerequisites | `parked/task-bindings.md` A0 | parked |
| Default context raised to 16384 on CUDA | `task-first-hour.md` R19 | shipped 2026-10-01; Metal followed on 2026-10-13 (M2, `task-minimax-code-compat-2026-09.md`): up to 16384 in 2 slots when memory holds it |
| Per-request log | `task-first-hour.md` R25 | shipped: `-log-requests` (`internal/serveapp/reqlog.go:15`) |

## 0a. Results, 2026-10-12 (the first day)

Record: `docs/measurements/integration-harness-2026-10/claude-code-census-2026-10-12.md`. Branch
`integration-harness-2026-10`.

- **H-e, new, fixed: Claude Code sends a `system` message inside `messages` on every turn, and goinfer answered 400.**
  Found by the first trace of I4. Claude Code 2.1.268 could not complete one turn. `/v1/messages` now keeps such a
  message in place as user-side text (`internal/serveapp/anthropic.go:anthropicTurns`); a `system` message in first
  place and every other unknown role are still a 400. The rendering is a choice: hoisting it into the system prompt
  would change the prompt's prefix each time one is added, and no chat template has a system turn after the first.
- **Step 1, the recipe: done**, from the trace and Claude Code's documentation (`docs/integrations/claude-code.md`).
- **I4, Claude Code's cell: measured.** The full client's first request is **19,692 tokens** (24 tools, 66 KB of
  schema); `--bare` sends 1,063 (3 tools). A session-title request of 761 tokens comes first.
  - **It does not fit 16,384.** By I4's own rule Claude Code is *not viable* at the window R19 and M2 chose as the
    harness floor; it needs 24,615 with the 25% margin, so `-ctx 32768`. Whether an 8 GB card holds a 7B there is
    unmeasured.
- **I4, opencode and Codex: measured** (the owner said to install them; nobara-pc, pinned versions;
  `docs/measurements/integration-harness-2026-10/harness-census-2026-10-12.md`). opencode 1.19.0: **7,165 tokens**,
  10 tools. Codex 0.162.1: **5,850 tokens**, 9 tools. Claude Code 2.1.296: 20,199. Both of the first two are viable at
  16,384 by I4's rule.
- **H-f, new, fixed: goinfer dropped Codex's agent instructions.** Codex sends them as the first of two `developer`
  messages and no `instructions` field; only the last system message was used, so 3,553 tokens of instructions never
  reached the model. Every system message is now kept (`internal/serveapp/openai.go:messagesToTurns`): leading ones
  joined in order, a later one in place as on `/v1/messages`.
- **I6's premise does not hold on Codex 0.162.1:** its request for an unknown model carries no `apply_patch` and no
  `custom` tool; edits would go through `exec_command`. I3's Codex cell still decides.
- **Step 3, harness-reliability Gate 1: built.** `serve check` has a `tools, harness, Messages` row
  (`ToolsHarnessMessages` in `internal/servecheck`, newer than this doc's pin). Against the 1.5B on Metal it reads skip, as the
  OpenAI-dialect row does. The 7B reading on nobara-pc is in that task's record.
- **Corrections to this doc's desk read**, from the trace and from Claude Code's documentation as read on 2026-10-12:
  - *"Without `ANTHROPIC_DEFAULT_HAIKU_MODEL` its background requests name a model goinfer does not serve"* (§2 I2):
    **not so on 2.1.268.** The title request names `ANTHROPIC_MODEL`; the variable redirects it. Traced.
  - *"its auto-compact window cannot be set below 100K"* (§1, §2 I2): the 100,000 floor is
    `CLAUDE_CODE_AUTO_COMPACT_WINDOW`'s. `CLAUDE_CODE_MAX_CONTEXT_TOKENS` sets the window Claude Code assumes, and its
    documentation gives no minimum. Whether it then compacts at a 32K window is a trace nobody has run.
  - *Overflow wording* (§1, I1): the documentation lists `Prompt is too long` and, since 2.1.228, a gateway token
    `capability_rejected: prompt_too_long`; it names no HTTP status. It also says a request whose input plus
    `max_tokens` exceeds the limit is retried with a smaller `max_tokens`. Claude Code sends `max_tokens: 32000` on
    every request, so I1's reserved-output term must not take the request's cap at face value: against a 32K window
    it would never fit.
  - *The idle timeout* (§5, H-d): 180 s through a custom base URL once feature flags are fetched, 300 s before.
    goinfer's 10 s heartbeat is inside both.
  - *Claude Code's own position:* its documentation says routing it to non-Claude models through a gateway is not
    supported. The recipe stands on traces, and a release can break it.
- **I5, the first piece (owner 2026-10-12: the transport before the facade):** `internal/inproc` holds the
  RoundTripper that serves a request with a handler in the same process. A reply streams frame by frame, closing the
  body or cancelling the context cancels the handler, and a handler's panic is an error and not a crash; its tests
  run under the race detector. **Not built yet:** the constructor that gives it serve's handler (the seam in serve's
  entry point), the public Experimental package, and the SDK module of the gate.
  - **A correction for that gate:** `testdata/tiny-qwen2-moe` is gitignored, so it is not "a committed tiny
    checkpoint". The tracked fixtures that carry a tokenizer are the tiny GGUFs (`testdata/glm-tiny.gguf`,
    `testdata/gemma1-tiny.gguf`); the in-process tests need one of those, or CI skips them.
- **A small item the trace adds:** the title request asks for JSON through `output_config.format` (a `json_schema`).
  `/v1/messages` ignores the field. goinfer can constrain to a schema, so honouring it is cheap; unowned.

## 1. The thesis: dialects are mostly done; context is the integration problem now

Gate 0 found three dialects and four gaps, and two of the four were fixed within two days. What remains in every
harness is one problem: each assumes a context window goinfer does not serve on the hardware it targets.

| harness | window assumed when not told | what it does at the limit |
|---|---|---|
| Claude Code | 200K for an unrecognised model id, unless `CLAUDE_CODE_MAX_CONTEXT_TOKENS` says otherwise (§0a) | compacts and retries on an error it recognises as "prompt is too long" |
| Codex | 272K for an unknown model slug | sees overflow only as an SSE `response.failed` with `code: "context_length_exceeded"`; a 400 body is not mapped |
| opencode, Kilo | none: in opencode's own source, "an unknown window never triggers auto compaction" | regex-matched overflow → compact |
| Pi | 128,000 | compacts at window − 16,384 — **at a 16K window that is every turn**; regex overflow → one compact-and-retry |
| Oh My Pi | 128K, or what `/v1/models` publishes | as Pi for configuration; compaction not read |
| Hermes | reads `/v1/models` (works since H-b), else 128K; **refuses any window under 64K** | text-matched overflow → compress |
| Cline | configured | `code: "context_length_exceeded"`, or a 400/413/422 naming context length or window → compact |
| Goose | `GOOSE_CONTEXT_LIMIT`, default 128K; compacts at 80% | 413, or a 400 naming context length → fallback |

On an 8 GB card goinfer serves 16,384 to a 1.5B at its default, and about 5.5K to a 7B holding its four KV slots
(R19's table; a 7B agent passes `-ctx 16384` and gets fewer slots). Metal's default was 4096 when this was written
and is up to 16384 since M2. opencode's first request
alone was 11,137 tokens (R19). So unless each harness is told the real window, **every one of them overruns
goinfer's**, and what happens next depends entirely on how the overflow error is worded. Today:

- **Chat Completions:** a 400 with `type: invalid_request_error` and **no `code` field**. `writeErr`
  (`internal/serveapp/helpers.go:390`) writes only `message` and `type`; the message carries
  "context_length_exceeded" in parentheses (`internal/serveapp/openai.go:1299`), which some matchers find in the text.
- **Messages:** `invalid_request_error` with the same text through `writeAnthropicErr`
  (`internal/serveapp/anthropic.go:154`), which Claude Code's "prompt is too long" matcher does not recognise.
- **Responses:** no `response.failed` event is ever emitted. The route's events are `response.created`,
  `response.output_text.delta`, `response.completed` and `response.cancelled`, so Codex cannot recover from overflow.

That is why context comes first here. Whether a model can call tools now has a measured answer for two checkpoints on
the 8 GB card. Whether a session survives its first long hour on a 16K window has no answer at all.

**What goinfer has that integrations should lean on.** Not dialect breadth: Ollama serves Chat Completions, Responses
(since v0.13.3) and Anthropic Messages (since v0.14.0), so three dialects is parity rather than a difference. What no
peer has:

1. it is a Go package with no native dependency, so it can run **inside** a Go program (I5) as easily as beside one;
2. its constrained decoder makes tool calls well-formed by construction (shipped), and could honour grammars that
   harnesses already send (I6);
3. its measurements: which local model holds which harness on which hardware, with provenance, is a table nobody
   publishes (I3).

## 2. New items (no existing doc owns these)

### I1 — Overflow that helps: one rule for all three dialects (resolves H-c)

H-c was left open on purpose. The server-cap message was worded so a client does *not* compact against a server limit
(`contextLengthErrorFor`, `internal/serveapp/openai.go:1308`), because opencode once compacted 34 times in a row
(R19). That loop happened because the request's **fixed part**, system prompt plus tool schemas at about 11K tokens,
was larger than the 8,192 window. No compaction can shrink the fixed part, so retrying could never fit. When it is the
**history** that overflows, compaction is exactly the right response, server limit or not: on an 8 GB card the server
cap usually *is* the hard limit, because the card cannot hold more.

So the discriminator should be whether compaction can help, not whose limit it is:

- **fixed part + reserved output ≥ window:** compaction cannot help. Keep an operator-facing message naming `-ctx` and
  its ceiling, and word it so that no matcher in §Sources fires. Several match "context length" or "context window"
  anywhere in a 400 body (Cline among them), so today's wording would set off exactly the loop R19 recorded. State the
  sizes instead: the fixed part needs N tokens, the server's limit is M.
- **otherwise:** the wording each dialect's harnesses recognise.
  - Chat Completions: 400, `error.code: "context_length_exceeded"`, message containing "maximum context length is M
    tokens" (the research's reading of the Pi, opencode, Cline, Goose and Hermes matchers; confirm each).
  - Messages: 400 `invalid_request_error`, message beginning "prompt is too long: N tokens > M maximum" (Claude Code).
  - Responses, streaming: SSE `response.failed` with `code: "context_length_exceeded"` (Codex); non-streaming, the
    Chat Completions body.

The fixed part is computable on every route. Messages carries `system` and `tools` as separate fields; Chat
Completions has system- and developer-role messages and `tools`; Responses has `instructions`, developer-role items
and `tools`. Reserved output is the request's own output cap, with a floor (start at 1,024) so a request with no cap
does not count as fitting with zero room.

**Amended 2026-10-13, before the code, from the matchers' sources and the census.**
- **Reserved output is capped at a quarter of the window** (and floored at 1,024, or a quarter of a window smaller
  than 4,096). Claude Code sends `max_tokens: 32000` on every request (§0a), so the request's own cap would put every
  Claude Code overflow in the first branch at any window under about 52K. The server already clamps the output to
  the room left, so what compaction needs is room for a reply, not the cap the client asked for.
- **Pi's and opencode's matchers were read** (`overflow.ts` and `provider-error.ts`, fetched 2026-10-13). Both hold a
  generic `context[_ ]length[_ ]exceeded` pattern over the error text, which today's message carries: so on those two
  a fixed part that cannot fit compacts today, R19's loop. Both also match "maximum context length is N tokens" and
  "prompt is too long". The first branch's message must match none of their patterns; a test holds both lists.
- **A route that does not know its fixed part** (`/v1/completions`, jobs, batches, image turns) answers with the
  second branch's wording, as it effectively does today.

**The trade, stated.** With the second branch a harness compacts against a `-ctx` the operator could have raised, and
the operator may not notice. `-log-requests` shows every overflow, and the message can still name `-ctx` when the fit
planner had room for more. The owner decides. This doc recommends the rule because on a small card it is the
difference between a long session that degrades and one that dies.

**Gate.** Unit tests per dialect through the real handlers, both branches, with a mutant per branch. Then live, inside
I3: each of Claude Code, opencode and Codex driven past the window. Pre-registered: **pass** if each overflow is
followed by at most one compaction and the session continues; **fail** on three or more consecutive compactions, or
the session ending on the overflow. A fail with the fixed part fitting means the wording is wrong for that harness.

**I1 built 2026-10-13 (the unit half of its gate; the live half is I3's).**
- **Code:** one typed overflow error with the two readings, the fixed part measured on each chat route by rendering
  the system text and tool definitions with one empty user turn, and a writer per dialect.
- **Read, through the real handlers on the committed tiny chat model (a 128-token window):** on `/v1/chat/completions`
  (with and without tools), `/v1/messages` and `/v1/responses` (streaming and not), a conversation that is too long
  gets its dialect's curable overflow, and a system prompt or tool definition that leaves no room gets the operator
  message with no overflow code. Both wordings are held to Pi's and opencode's pattern lists.
- **Mutants, one per branch, both red and reverted:** the incurable reading never taken (every fixed-part case
  failed), and taken whenever the fixed part is known (every long-conversation case failed).
- **Not covered:** Cline's, Goose's and Hermes' matchers were not read; the tests hold the phrases the doc names for
  them. No real agent has been driven past its window. `/v1/completions`, jobs, batches and image turns do not
  measure a fixed part and answer with the curable wording.

#### I1 live gate, second design (I1-v2)

**Registered 2026-10-14 at `main` `5c609661`** (the predictions commit; the design was approved by the owner the same day, with the
amendments folded in below). From this commit on, the rules, windows, arms, void classes and predictions below do not move.
Nothing has run.

**Why a second design.** The first design (3c in I3's run plan) read the night of 2026-10-13 as Codex VOID, opencode VOID and Claude Code
FAIL (I3's read, below). It failed in two ways. Qwen3-4B fetched every `bulk` file in one step, which no compaction can cure. And in
`edit` and `fix`, as in Claude Code's `bulk` run, the agent's **own** compaction fired before goinfer's limit was reached, so no
overflow could be attributed to goinfer.

**Model and windows.**
- **Model:** Qwen3-4B int4, f16 KV, for all three agents. It is the one model measured to drive all three on the 8 GB card (KV0), and its
  first requests are measured in I3: Codex 5,669, opencode 7,200, Claude Code 17,711 tokens. These are the fixed parts, `F`.
- **Served window `W`:** Codex 15,616, opencode 17,664, Claude Code 28,672, the most Qwen3-4B holds.
  - goinfer refuses a prompt of `W` tokens or more (`newContextOverflow`), so the history room before the first overflow is `W - F`:
    9,947, 10,464 and 10,961.
  - The reserved output only decides which wording is used. Every cell is on the curable side: `F` plus the reserve is under `W`
    (Codex 9,573, opencode 11,616, Claude Code 21,807).
- **Claude Code's output cap is 4,096** (`CLAUDE_CODE_MAX_OUTPUT_TOKENS`), written by `connect` from branch
  `connect-cc-window-2026-10`, so goinfer's reserve for it is 4,096. Codex and opencode keep `connect`'s quarter of the window.

**The told window, `2 x W`: a MEASUREMENT CONFIGURATION, not a recommended user setting.** After `connect -write`, the runner raises
the window the agent is told to `2 x W`:
- Codex: `model_context_window`, and `model_auto_compact_token_limit` = `2 x W` less the output limit;
- opencode: `limit.context`;
- Claude Code: `CLAUDE_CODE_MAX_CONTEXT_TOKENS`.

The runner then reads the file back and refuses the run if any value is not what was set (Claude Code's output cap included). The
purpose is to put the agent's own compaction threshold above goinfer's. The first context event is then goinfer's overflow answer, and
the agent's response to that answer is what is measured. A user should run `connect`'s own values, which tell the agent the real window.

**Arms per agent.**
- **graded:** three runs.
- **control:** one run with the true window (`connect`'s values unchanged), graded by the same rules and reported only. It shows the
  agent's own behaviour.
- **planted:** one run, told `2 x W`, against a measurement binary built at the same commit with one change: `helpless()` returns true.
  Every overflow then gets the operator-facing wording with no overflow code, which no agent should compact on, so the expected grade
  is FAIL. This proves the gate can go red. It is a measurement build, not a production change.

**Task `chain`.**
- 24 files in `data/`, each about 1,000 tokens of record lines (48 lines). Every line starts with the file's own name, so a file's
  content identifies it.
- Each file ends with `next: data/<name>.txt`, under a name drawn per run. The 24th ends with `END` and a code word.
- Beside them, 40 decoy files of the same shape, chained only to one another.
- The prompt: start at `data/start.txt`, follow `next:` one file per tool call until a file says `END`, and answer with the code word.
- A model cannot fetch the chain in one step, because each name is known only from the file before it. One that reads the directory's
  contents reads decoys, and is caught.
- The chain is about 24,000 tokens, 2.2 to 2.4 times the room, so a run that follows it overflows about twice, each time curably.
- Success is mechanical: the code word in the agent's own reply, and exit 0.

**Recorded per run.** The fixed recorder, with prompt sizes on every route (a 400's size from its text, `response.failed`'s from its
usage). Each agent's own events in their order: chain-file reads by index, and compactions (Claude Code's `compact_boundary`, Codex's
`compacted` session record, opencode's compaction parts). Also, for each answered request, whether its reply was clamped by the window.

**Per run, the void checks come first.** A run is VOID, and says nothing about the wording, on the first of these that holds:
- *box:* a `halted` answer or a 5xx;
- *never reached:* no overflow answer;
- *agent first:* the wire shows the history shrinking by 500 tokens or more before goinfer's first overflow answer, or the agent reports
  a compaction and the wire shows no overflow. This is the told window not taking;
- *incurable step:* one step adds more than `W - F - 1,000`;
- *off-chain:* a decoy's lines appear in the agent's output, or one step grows the prompt by more than 2.5 times the run's median step;
- *summary too large:* an overflow answer comes straight after a shrink, and no request is answered after it;
- *clamped end:* the session's last answered reply was clamped by the window (it ended on a cut reply, not on an overflow answer);
- *model stopped:* exit 0, no error, the last request answered normally, and `END` never read.

**Per run, otherwise:**
- **PASS:** at most one compaction between consecutive new chain reads (each overflow followed by at most one compaction), compactions
  no more than the overflow answers, `END` read, the code word in the reply, and exit 0.
- **FAIL:** the session ends on an overflow answer or an error (exit not 0, an error result, or the last request an overflow); or three
  compactions with no new chain read between them (I1's own word, *consecutive*, which 3c had turned into a count per run); or two
  compactions between consecutive new chain reads; or more compactions than overflow answers.
- Overflow answers in a row stay reported and ungraded, because Codex trims and re-sends. The count of clamped replies is reported.

**Per agent:**
- **PASS:** at least two gradable graded runs, all PASS, and the planted run FAIL. If the planted run is VOID, it is run once more. A
  planted run that is VOID again parks a PASS. A planted run that PASSes voids the gate for that agent.
- **FAIL:** two or more gradable graded runs FAIL. A FAIL with the fixed part fitting means the wording is wrong for that agent, as I1
  says.
- **Ambiguous, parked:** otherwise. Three more graded runs the next night (a registered repeat), then the same rule over all gradable runs.
  If fewer than two are still gradable, the agent is VOID.

**Predictions, written before registration.** Each line gives the expected verdict and the void classes expected to fire.
- **Codex, graded: PASS (0.55).** In I3 it compacted and continued after runs of overflow answers. Expected voids: *off-chain* (0.25:
  Qwen3-4B batched reads under Codex in I3), *clamped end* (0.1). FAIL 0.25, parked 0.2.
- **Codex, control: VOID *agent first* (0.8),** as in I3's `edit` runs (12 to 20 of its own compactions, no overflow).
- **Codex, planted: FAIL (0.75).** VOID *never reached* or *model stopped* (0.2).
- **opencode, graded: FAIL (0.45).** In the 3c smoke and I3's `bulk` runs it ended on `ContextOverflowError` after one compaction. PASS
  0.3, parked 0.25. Expected voids: *summary too large* (0.2), *model stopped* (0.15).
- **opencode, control: VOID *agent first* (0.75).**
- **opencode, planted: FAIL (0.85).**
- **Claude Code, graded: PASS (0.45).** In I3 it answered goinfer's one overflow by re-sending its compaction smaller and going on. FAIL
  0.25, parked 0.3. Expected voids: *model stopped* (0.35: Claude Code x Qwen3-4B passed 1 of 5 tasks in I3), *summary too large* (0.1).
- **Claude Code, control: VOID *agent first* (0.85),** as in I3's `bulk` run.
- **Claude Code, planted: FAIL (0.7).**

**Pilots, one per agent: one graded-configuration `chain` run, not counted.** A pilot passes when all of these hold:
- **the told window took effect:**
  - the values read back from the configuration file are `2 x W` (and 4,096 for Claude Code's output cap);
  - no agent compaction and no wire shrink comes before goinfer's first overflow answer;
  - the largest prompt answered before that overflow is at least `W` less two files' worth (the agent grew its history to goinfer's
    limit, not to a threshold of its own);
  - for Codex, the session log's `model_context_window` is 95% of `2 x W`, the share it took of the window `connect` wrote in I2;
- the chain was followed (no *off-chain*);
- the median step is between 700 and 1,300 tokens.

A pilot that passes writes `pilot-ok-<agent>` beside its record. The night script refuses an agent without it.

**The pin, named at pilot time.** One serve binary, and its planted twin, built at one `main` commit that contains the merge of
`kv3-cuda-2026-10` and the tips of `connect-cc-window-2026-10` and `connect-cc-model-slots-2026-10`. The build script refuses a commit
that does not contain all three (`git merge-base --is-ancestor`). The commit is named in the pilot's record and used unchanged by the
night jobs.

**What runs when, and what each waits on.**
- **Every pilot and job waits on three merges to `main`:**
  - `kv3-cuda-2026-10`, which waits on tonight's suite-a and suite-b reading green;
  - `connect-cc-window-2026-10`;
  - `connect-cc-model-slots-2026-10`.
- **Claude Code pilot:** by day, when nobara's timing lock is free; about 30 minutes.
- **Codex and opencode pilots:** a night job, about 30 minutes for the two.
- **Night job, Codex and opencode:** 3 graded + 1 control + 1 planted each, about 12 minutes a run, about 120 minutes. It is queued after
  their pilots, in the same night or a later one.
- **Night job, Claude Code:** 3 graded + 1 control + 1 planted. Re-estimated from its pilot before queueing; with KV3 CUDA's prefill the
  17.7K first request should no longer take 4.5 minutes.
- **Not run or queued yet:** all of the above.

**The scripts, prepared 2026-10-14 after registration.** They are in `docs/measurements/integration-harness-2026-10/matrix/`, and none
of them changes a rule above.
- `build-i1v2-pin.sh <main commit>` builds the serve binary and its planted twin. It refuses a commit without the three merges, and
  copies the matrix scripts at that commit into `~/goinfer-bench/i1v2-2026-10`.
- `plan-i1v2.json` holds the configurations: three pilots, then a graded, a control and a planted configuration per agent.
- `run-i1v2.sh` runs them. `PILOT=1` takes pilot configurations only and writes `pilot-ok-<agent>`. Otherwise the script refuses any
  agent without that file. Either way it holds the timing lock.
- `gate_i1v2.py` grades the runs, or the pilots with `--pilot`.
- `tasks.py` gains `chain`. `run_cell.py` sets and reads back the told window and checks the output cap. `agents.py` records each
  agent's own events in order.

**How the scripts were checked, without a model or the GPU:**
- `connect` from the I3 pin wrote each agent's file against a fake `/v1/models`, and the told window was set and read back. Codex
  keeps its 3,904-token output room. Claude Code's file says 7,168 there, so until the pin carries `connect-cc-window-2026-10`'s cap
  the runner would refuse its runs, which is intended.
- Following `next:` through a generated chain walks the key in order.
- The planted patch applies to `overflow.go` exactly once.
- The grader returns the registered class on synthetic PASS, never-reached, agent-first, incurable, off-chain and both FAIL cases.
- Event parsing on I3's recorded runs reproduces their compaction counts and reads Codex's window as 8,937 of 9,408 (95%).
- `run-i1v2.sh` against an empty directory, and against a fake pin with no pilot, refuses before starting anything.
- `build-i1v2-pin.sh main` refuses, because `main` does not yet contain `kv3-cuda-2026-10`.

**The Claude Code pilot by day,** once the pin exists and the lock is free:
`setsid nohup env PILOT=1 ONLY=i1v2-pilot-cc bash ~/goinfer-bench/i1v2-2026-10/matrix-pinned/run-i1v2.sh </dev/null >/dev/null 2>&1 &`.

**Three things the owner should know before the pilots; none of them moves a rule:**
- The registered order of the void checks puts *never reached* before *agent first*. A control run that compacts by itself and never
  meets an overflow therefore reads *never reached*, not the *agent first* its prediction names. The second clause of *agent first* cannot
  fire.
- In I3's opencode `edit` r2 the wire shows a shrink, yet opencode's JSON output had no compaction part. If opencode's compactions do
  not show in its output, the grader cannot count them. The pilot prints that comparison as a diagnostic, and a mismatch is for the owner
  to rule on before opencode's night job.
- `connect-cc-window-2026-10`'s tip, as read on 2026-10-14, adds a warning only. The 4,096 cap the registration names has to be in the
  pinned build, or every Claude Code run refuses.

### I2 — `connect`: write each harness's configuration with the window goinfer actually serves

Magnitude advertises one-click connection to eight harnesses. Read from its connectors, that is a config writer: it
patches each harness's own config file in place, keeps unrelated settings, saves a restore record, and never launches
the harness. That is the right shape. goinfer can do it with more to go on, because it knows the window it serves,
the checkpoint's `serve check` results, and (after I4) each harness's first-request size.

`goinfer-serve connect <harness> [--model name]`, against a running server:

- **prints** the configuration by default; `--write` patches the file in place, keeping unrelated settings, with a
  backup and a restore record; `--undo` restores it byte for byte. Editing someone's dotfiles is invasive, which is
  why printing is the default.
- fills the base URL, the model id, **the served window** (from the server's own `/v1/models` `context_window`) and an
  output limit;
- **warns or refuses** when the harness's measured first request (I4) does not fit the window, when the checkpoint's
  `tools, harness-scale` row skipped, or when the harness has a floor the window misses (Hermes, 64K; Pi's 16,384
  compaction reserve);
- never launches the harness and installs nothing into it. Magnitude also drops a skill file into each harness; that
  is not an engine's job.

Where each harness keeps its configuration, from the desk read (pin versions in the implementation):

| harness | where | what `connect` sets |
|---|---|---|
| Claude Code | `~/.claude/settings.json`, `env` block | `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL`, **`ANTHROPIC_DEFAULT_HAIKU_MODEL`** (background work; replaces the deprecated `ANTHROPIC_SMALL_FAST_MODEL`), `CLAUDE_CODE_MAX_CONTEXT_TOKENS` |
| Codex | `~/.codex/config.toml` | `model_provider`, `model`; `[model_providers.goinfer]` with `base_url`, `env_key`, `wire_api = "responses"` (`"chat"` is now a config error); `model_context_window`, `model_auto_compact_token_limit` |
| opencode, Kilo | `opencode.json`, `~/.config/kilo/kilo.json` | `provider.goinfer` with `npm: "@ai-sdk/openai-compatible"`, `options.baseURL`, `models.<id>.limit.context` and `.output` (unset, opencode never compacts) |
| Pi, Oh My Pi | `~/.pi/agent/models.json`, `~/.omp/agent/models.yml` | a provider with `api: "openai-completions"`, the base URL, per-model `contextWindow` |
| Hermes | `~/.hermes/config.yaml` | `providers.<n>` and `model.provider: custom:<n>`; the window is read from `/v1/models`; refuse below 65,536 |
| OpenClaw | `~/.openclaw/openclaw.json` | `models.providers.<id>` with `baseUrl` and `contextWindow` |
| Cline | owns its file | print `cline auth -p openai-compatible -b URL -k KEY -m M` and the `contextWindow` setting |
| Goose | environment | `GOOSE_PROVIDER=openai`, `OPENAI_HOST`, `OPENAI_BASE_PATH`, `GOOSE_MODEL`, `GOOSE_CONTEXT_LIMIT` |
| Aider | environment and `.aider.model.metadata.json` | `OPENAI_API_BASE`, `OPENAI_API_KEY`, model `openai/<id>`, `max_input_tokens` (Aider does not compact; overflow ends the turn) |
| Copilot CLI | environment | `COPILOT_PROVIDER_TYPE=openai`, `COPILOT_PROVIDER_BASE_URL`, `COPILOT_MODEL`, `COPILOT_PROVIDER_MAX_PROMPT_TOKENS` |

Two Claude Code details belong in `connect`, and are in its recipe since 2026-10-12 (§0a). Its background requests go
to the main model unless `ANTHROPIC_DEFAULT_HAIKU_MODEL` names another. And `CLAUDE_CODE_MAX_CONTEXT_TOKENS` tells it
the served window; whether its own compaction then triggers at a small window is untraced, so I1's error is still the
path to rely on. Where the box holds two
models, multi-model serving can give the background slot a small one (`-model main=… -model fast=…`).

Start with one harness per dialect (Claude Code, opencode, Codex), then the rest of Magnitude's list.

**Gate.** Per harness: `connect --write`, then one headless turn (I3's smoke cell) succeeds, and `--undo` leaves the
file byte-identical to before. Tests parse each harness's published config schema where one exists.

**I2 built 2026-10-13 for Claude Code, opencode and Codex (the unit half of its gate).**
- **`goinfer-serve connect <agent> [url]`**, package `internal/connect`: reads `/v1/models`, prints the configuration,
  `-write` patches the agent's file with a backup and a restore record beside it, `-undo` restores it.
- **Read by its tests:** each JSON file keeps its other settings and their order; Codex's TOML keeps every line it
  has no business with and has its provider table replaced, not doubled; `-write` twice then `-undo` is byte for byte
  the original for all three, and removes a file `connect` created; `-undo` refuses a file edited since; an agent
  whose first request does not fit is refused without `-force`; the window is read from the real model list.
  A mutant that restored one byte too many turned the byte-identity test red.
- **Departures from the sketch.** The output limit is a quarter of the window, between 1,024 and 8,192. Codex's
  auto-compaction limit is set to the window less that output limit, which is a reading of what the key means, not a
  measurement. Codex's settings are written at the top level, as the census ran it, not as a profile.
- **Not done:** the other eight agents in the table; parsing each agent's published schema; the warning from a skipped
  `tools, harness-scale` row.

**I2's live half, read 2026-10-13** (`docs/measurements/integration-harness-2026-10/connect_live.py`; the final binary, the 1.5B at `-ctx 32768`,
a throwaway `HOME` and work directory per run, `-write` always given an explicit `-file` inside them, so no real config file was named; Claude
Code 2.1.296, opencode 1.19.0, Codex 0.162.1). Each agent twice: **fresh** (no file before, so `-undo` must remove it) and **existing** (a file with
unrelated settings, so `-undo` must return its bytes).

| agent | case | `-write` | headless turn | reply read from the agent's own output | `-undo` |
|---|---|---|---|---|---|
| Claude Code | fresh | ok | exit 0, 91 s | `ok` | byte-identical, file removed |
| Claude Code | existing | ok | exit 0, 91 s | `await<byte[]>` (the 1.5B at default sampling; the turn completed) | byte-identical |
| opencode | fresh | ok | exit 0, 5 s | `ok` | byte-identical, file removed |
| opencode | existing | ok | exit 0, 5 s | `ok` | byte-identical |
| Codex | fresh | ok | exit 0, 2 s | `ok` | byte-identical, file removed |
| Codex | existing | ok | exit 0, 1 s | `ok` | byte-identical |

- **Verdict against its registered rule** (a headless turn succeeds, `-undo` leaves the file byte-identical or gone): **PASS, 6 of 6 on the
  mechanics.** One reply is not `ok` (Claude Code, existing case): the turn ran to its end and the same configuration answered `ok` in
  the fresh case, so that is the 1.5B's sampling and not the file.
- **It passed only after three defects were fixed** (defects 1 to 3 above), one of them invisible to the first version of the script (a false pass).
  The script now reads the reply from each agent's own output; run on the build before the stream fix it reports Codex as having no reply.
- **Codex's two keys.** It accepts top-level `model_context_window` and `model_auto_compact_token_limit` without a warning and honours the
  first (its log: `model_context_window: 31129`, 95% of 32,768). Whether it compacts by itself before the window is reached: **yes, by the
  evidence of the 3c smoke** (it compacted and finished two runs at a 12,288 window), **but not when one step jumps past its headroom**, as
  in the 32,768 probe above. The registered reading is 3c's.

### I3 — The real-harness matrix

`task-harness-reliability-2026-10.md` Gate 3 replays recorded traces, teacher-forced and free-running, with tool
results simulated from the trace. That isolates the model, which is its point, and so it cannot see what ends sessions
in practice: compaction, overflow recovery (I1), idle watchdogs (H-d), background requests, the harness's own retries,
and dropped tool types (Codex). This item drives the actual harness binaries headless against goinfer, with real tool
execution, on a fixture repository.

**Headless entry points** (desk read): Claude Code `claude --bare -p "…" --output-format stream-json` (needs
`ANTHROPIC_API_KEY` or an `apiKeyHelper`); Codex `codex exec --json -c model_provider=goinfer -m <id> "…"` (never
`--oss`, which calls Ollama's `/api/*`); opencode `opencode run -m goinfer/<id> --format json "…"`; Pi `pi -p`; Oh My Pi
`omp -p`; Hermes `hermes -z`; OpenClaw `openclaw agent exec --json`; Cline `cline -P openai-compatible --json`; Goose
`goose run --no-session -q --output-format stream-json -t`; Aider `aider --message … --yes-always`; Copilot CLI
`copilot -p … -s --allow-all-tools`.

**Fixture:** a small Go module, copied fresh for each run, with five tasks whose success is checked mechanically.

1. *read*: "What does `Add(2, 3)` in calc.go return? Answer with the number." Pass: the answer contains 5.
2. *search*: "Which file defines `ErrNotFound`?" Pass: the answer names it.
3. *edit*: "Rename `sum` to `total` everywhere." Pass: `go build ./...` succeeds and no `sum(` remains.
4. *write*: "Add a test for `Reverse` and make it pass." Pass: `go test ./...` passes and a new test names `Reverse`.
5. *fix*: "`TestParse` fails; fix the code, not the test." Pass: `go test ./...` passes and the test file's hash is
   unchanged.

**Sandbox, non-negotiable.** These harnesses run shell commands chosen by a small model, with permission prompts
bypassed for unattended runs. Each run is in a container, or under an unprivileged user, where only the fixture copy is
writable and the only reachable address is goinfer's.

**Recorded per run:** task passed; turns; tool calls; calls the parser could not use; `X-Goinfer-Tools-Dropped` seen;
429s; overflow errors and compactions; wall time; per-turn prompt tokens and time to first token from
`-log-requests`; peak VRAM; harness version; goinfer commit; model, quant, backend, box and `-ctx`. Harness system
prompts embed the date, directory and git state, so two runs are not token-identical even when greedy; record
repetitions as repetitions.

**Screening scope:** one harness per dialect (Claude Code, opencode, Codex) × the two checkpoints with measured
harness-scale successes on nobara-pc (Qwen2.5-7B-Instruct q4_k_m at `-ctx 16384`, and Mellum2.1 int4 with expert
streaming) × five tasks × one repetition. That is 30 runs at about 3 minutes each, roughly 1.5 h plus setup: one night
job, well inside the 3 h cap. Estimate it properly before `night.py add`.

**Pre-registered reading, per (harness, model) cell:** **works** if at least 4 of 5 tasks pass with no session ending
on an error; **fragile** at 2–3; **no** at 0–1. A fragile cell gets three repetitions the next night before anything
is published. Output: `docs/integrations/harness-matrix.md`, regime-labelled and dated, one row per cell. This is the
publishable table.

**Sequence:** after I1 (otherwise the matrix re-measures a known overflow failure), I2 (otherwise every cell is
configured by hand) and Gate 1.

#### What the live gates found first, 2026-10-13 (day, nobara-pc; this is what the plan below is built on)

I2's live gate (`connect -write`, one headless turn, `-undo`, byte identity; script `docs/measurements/integration-harness-2026-10/connect_live.py`)
and the work to run I3 turned up four defects and four facts. Each was found by running the real binary.

- **A false pass, caught.** The first live run read "the reply contains ok" off the whole output, which also matches `"tokens"`.
  It showed Codex passing. Codex had printed `OutputTextDelta without active item` and returned **no message at all**.
  The script now reads each agent's own reply (Claude Code's `result`, opencode's text part, Codex's `agent_message`).
- **Defect 1, fixed: the Responses stream carried no item lifecycle.** A `/v1/responses` stream held `response.created`,
  `response.output_text.delta` and `response.completed` only. Codex assembles its turn from `response.output_item.done`, so through
  goinfer it saw an empty turn, text and tool calls alike. The stream now announces and closes every item
  (`output_item.added`, `content_part.added`, `output_text.done`, `content_part.done`, `output_item.done`,
  `function_call_arguments.delta` and `.done`) and numbers every event with `sequence_number`. `sdktest` holds it for a message and a
  forced function call, and reads it through the official OpenAI SDK; three planted defects (an item never closed, a frozen sequence
  number, a function call announced already complete) each turned the gate red. After it, a headless Codex turn returns `ok`, and a
  7B steered by Codex ran three real `exec_command` calls. **This had to be fixed before the Codex cell could mean anything.** It is
  outside the brief's list; it is the item §5 had listed as "Responses emits neither `response.output_item.*` nor `response.failed`".
- **Defect 2, fixed: `connect` ignored flags after the URL.** `connect codex URL -write` printed instead of writing, and
  `connect codex URL -undo` restored nothing (package `flag` stops at the first operand). Every test put the flags first. Flags
  may now follow the URL, and a second URL is an error.
- **Defect 3, fixed: Claude Code at a 32,768 window compacted at once and failed.** It asks for `max_tokens: 32000` on every
  request and counts that against the window `CLAUDE_CODE_MAX_CONTEXT_TOKENS` tells it, so it sees no room. Pilot, 7B at `-ctx 32768`,
  the "read" task, goinfer never rejected a request: Claude Code ended after 705 s on `Prompt is too long · automatic compaction failed:
  summarization produced empty response`. With `CLAUDE_CODE_MAX_OUTPUT_TOKENS` set to the output limit (8,192) the same task ran to
  its end (284 s, no compaction). One run each; exploratory. `connect` now writes that variable. (This answers §0a's open "whether it then
  compacts at a 32K window": left alone, it does, and it fails.)
- **Fact: Codex honours the window `connect` writes.** Its own session log records `model_context_window: 31129`, which is 95% of the
  32,768 written. **It does not recover from an overflow on a sampling request.** Probe (7B, `-ctx 32768`): the model made three parallel
  `seq` calls whose outputs (about 6.4K tokens each, Codex's 10K-token cap) took the history from 6.0K to 25.1K in one step; the next
  request was answered with `response.failed` (I1's curable branch: 25.1K plus the reserved quarter-window reply is over 32,768) and
  Codex ended the turn with "ran out of room in the model's context window". Codex's own auto-compaction reads the usage the server
  reports after a response; it had no chance to act. Whether it compacts when history grows in steps smaller than its headroom is what
  the Codex overflow run (3c) reads. A candidate for `connect`, **not changed**: its auto-compact limit (window less the output limit)
  leaves 8,192 tokens of headroom and one Codex tool output can be 10,000.
- **Fact: Codex and opencode run in seconds, Claude Code in minutes, because of CUDA's batched prefill.** On the 7B a 7K-token opencode
  request prefills in 6.8 s. Past about 12,200 attended keys the batched prefill declines (its attention keeps one score per key in
  shared memory: 49,664 B at 12,288 keys, over the device's 49,152 B, and it has no split-KV path) and the request takes the
  sequential path, **about 15 ms a token on the 7B**: Claude Code's 17.7K-token first request took **289 to 303 s** to its first
  token, and a 10K-token tool output through Codex took 158 s. This is not in the plan of any doc I could find;
  `task-kv-architecture-2026-10.md` KV3 (prefill in blocks everywhere) is the nearest item and is about bookkeeping, not this kernel
  limit. I did not touch it. It sets the cost of every cell with a context past 12K, and it is why the overflow runs for opencode
  and Codex use a 12,288 window.
- **Fact: the swap guard halts a long run on this box.** `GOINFER_SWAP_GUARD` defaults to 512 MB over the swap in use when the server
  came up, and swap here grows by that much in minutes with 48 GB free (it was 6.3 GB at the first start of the day and 9.4 GB by
  the fourth). Mellum2.1 tripped it on its first request, twice; the 7B once. A tripped guard answers every request with a 503
  `halted` until swap falls back, which it does not. The matrix sets it to 4,096 MB (still armed; recorded in the job's provenance),
  voids any run that saw a `halted` answer, restarts the server and redoes the cell, up to two more times.
- **Fact: opencode needs a `ripgrep` it downloads on first use.** In the first sandboxed pilot its `grep` and `glob` tools all failed
  with "ripgrep execution failed", which would have scored opencode's cells as model failures. The sandbox provides the `rg` Codex
  ships, read-only, in the place opencode looks. With it the same tasks use the tools.

#### I3 run plan, registered 2026-10-13, before the run

**3a. The sandbox.** Built and proved (`docs/measurements/integration-harness-2026-10/matrix/sandbox.py`). **It needs nothing from the
owner:** unprivileged bubblewrap 0.11 and `socat` are already installed, and user namespaces are on.
- *Design.* Every namespace is unshared (user, pid, ipc, uts, cgroup, **net**); the environment is cleared, so no host variable
  reaches the agent; the root is a tmpfs remounted read-only, with `/usr`, the Go toolchain, a few files of `/etc` and the
  harness install bound read-only; **the host's home directory, `.ssh`, `/run`, `/var` and its other mounts do not exist inside**.
  Writable: the fixture copy (`/work`), a per-run `HOME`, and an in-memory `/tmp`; all are made for the run. The one way out of the
  network namespace is a unix socket bind-mounted from the host: a `socat` inside listens on `127.0.0.1:<goinfer's port>` and
  relays to it, a `socat` on the host relays the socket to a logging proxy (`wire_proxy.py`) in front of goinfer. A loopback with one
  port, and nothing behind it but goinfer. `--die-with-parent --new-session`, `ulimit -u 4096 -f 2 GiB`. The mechanical checks of
  each task (`go build`, `go test`) also run inside it: the model wrote that code.
- *Proof, the self-test, run at the start of every job and fatal if it fails:* from inside, 32 attempts, 28 of which must fail and four of which must work: writes to `/usr`, `/etc`, `/etc/ssl`, `/usr/local/go`, `/bin`, `/`, `/proc/sys`, a path under the fixture that climbs out of it,
  and through two symlinks planted in the fixture (one to a host directory, one to `/usr`); reads of the host's home, `.ssh` and
  `/etc/shadow`; TCP to 1.1.1.1:443, 8.8.8.8:53, 169.254.169.254:80, `127.0.0.1:22`, the box's LAN address, **a canary listener on
  this host's loopback and one on all addresses**, and the port beside goinfer's; DNS; the docker and session-bus sockets; a signal to
  a host pid. The four that work: write to `/work`, `HOME` and `/tmp`, and `GET /v1/models` through the relay. After the probe
  the host is checked for any file it may have written. All pass.
- *The self-test can go red:* three planted weaknesses (a writable host directory where a read-only one belongs, the shared host
  network, a writable root) are each caught, with the escape named (`write /etc/ssl/x`, `tcp 1.1.1.1:443`, `write /etc/x`). A job
  runs these too, and runs nothing if one is missed. One planted weakness I first tried (a writable bind of `/usr`) was *not* caught,
  because the host's own file permissions already refuse the write; it was replaced by one that does reach the host.
- *Found by use:* the first matrix start ran the self-test before goinfer was up and refused to run, which is the intended failure.
- *What it does not contain:* CPU, memory or disk exhaustion beyond the two `ulimit`s (a runaway is bounded by its per-run timeout);
  the GPU, which the agent cannot address; goinfer itself, which an agent can send any request (the API has no admin route without
  `-allow-admin`). Nothing runs unattended outside it.

**3b. What Claude Code's cell is: the real client, `-ctx 32768`, not `--bare`.**
- *The 7B can hold it now.* With CUDA f16 KV on main the 7B at `-ctx 32768` loads with **one** resident KV slot (1.8 GB of KV, 6.75 GB
  of VRAM in use of 8.19), against about 5.5K positions in four slots in R19's table. 32,768 is over I4's 25,249 for Claude Code's
  20,199-token first request, so by I4's rule it is *viable*. Mellum2.1 holds 32,768 too (its sliding-window layers keep 1,536 positions, so 32,768 costs 1.0 GB of KV; resident, 7.3 GB of VRAM), so Claude Code gets a Mellum cell.
- *Why not `--bare`.* `--bare` is a different client: no hooks, memory, `CLAUDE.md` discovery or plugins, 3 tools instead of 21 to 24, a
  1,063-token prompt. A row labelled "Claude Code" in a published table has to be what a person runs; the bare figure would be the
  flattering one. **`--bare` is not a cell here.** It stays a measured lower bound (I4's 1,063).
- *What it costs, so the owner can overrule the choice:* the one KV slot is shared by the main request and Claude Code's side requests
  (title, compaction), and the batched prefill declines past 12.2K keys, so each run pays one cold prefill of about 18K tokens, **about
  5 minutes**, and a compaction attempt pays another. Pilot runs: 284 s (a read task, no compaction). A Claude Code run is therefore
  5 to 10 minutes and the cell 25 to 50 minutes; its per-run timeout is 30 minutes.
- *Settings:* those `connect` writes (including `CLAUDE_CODE_MAX_OUTPUT_TOKENS` after defect 3), unmodified.

**3c. I1's live gate, operationalised before it runs.** Registered in I1: pass if each overflow is followed by at most one compaction and
the session continues; fail on three or more consecutive compactions, or the session ending on the overflow. Made measurable, after two
exploratory smoke runs showed what the first design got wrong:
- *Why the first design was dropped.* It read eight 9K-token files "one at a time" at a 12,288 window. The 7B asked for all eight at once,
  so one step added 46K tokens, which no compaction can cure: the turn's own new content exceeded the window. That says nothing about
  the wording. The same smoke showed Codex answering four overflows in a row (its compaction request, trimmed and asked again) and then
  finishing, so "overflow answers in a row" is not "consecutive compactions" either.
- *Windows at 1.6 x each harness's first request* (I4's census), so that ordinary task growth, a few turns of tool output, crosses them
  step by step: Codex **9,408** (first request 5,850), opencode **11,520** (7,165), Claude Code **28,672** (20,199; 1.42 x, the 7B and Mellum
  hold no more than 32,768). All three are viable by I4's rule. Tasks: `bulk` (eight files of about 2,500 tokens, read one after another),
  and the matrix's own `edit` and `fix`, whose histories grow every turn. Three repetitions for opencode and Codex, one for Claude Code.
- *Read from the wire log and the harness's own output, per run* (`matrix/gate_i1.py`): overflows (a 400 naming the prompt or the context, or
  `response.failed`); compactions (the harness's own count, else what the wire shows: a tool-less request of 1,000 tokens or more after the first
  request with tools, or a prompt under 60% of the largest so far); whether the session continued (exit 0 with the harness's final answer, the
  last request not an overflow). Overflow answers in a row are reported, not graded.
  - **PASS:** at least one overflow, compactions not above overflows and under three, session continued.
  - **FAIL:** the session ended on the overflow or an error, or three or more compactions in the run, or more compactions than overflows.
  - **VOID:** no overflow and no compaction (the window was never reached), or an **incurable jump**: one step added more tokens than fit beside the
    fixed part once everything older is compacted (new content = the rejected prompt less the last one answered; room = window less the
    session's first request less 1,000 for the summary). Either way the run says nothing about the wording.
  - **NOTE:** no overflow, but the harness compacted by itself first: a pass for the session, no evidence about the wording.
  - A FAIL with the fixed part fitting means the wording is wrong for that harness, per I1.
- *Exploratory smoke of the gate itself* (7B, 12,288, two runs each, discarded; it exercised the grader and found the design flaw above):
  opencode ended both runs on `ContextOverflowError` after 4 and 1 compactions (FAIL by this rule); Codex compacted and finished both
  (PASS), after four consecutive overflow answers each time.
- *Prediction, written first (these are the first reading of the smoke, so they are weak):* Codex PASSes (0.6): it compacts and
  continues. opencode FAILs (0.55): it compacts and then meets the overflow again. Claude Code is the unknown (0.4 PASS): its pre-flight
  check fired in the pilot before goinfer was asked, and it has now been given an output cap.

**The matrix, one job in two parts.** The reading rules of I3 stand (works at 4 of 5 with no session ending on an error; fragile at 2 to 3;
no at 0 to 1). The five tasks, the fixture and the recorded fields are I3's; the fixture is `docs/measurements/integration-harness-2026-10/matrix/fixture`
(stored as `.tmpl`), checked mechanically inside the sandbox. **Task 1 can pass without a tool call** (`Add(2, 3)` is 5 for any
`Add`); the registered pass rule is kept and `tool_calls` is in the record. Plan: `matrix/plan-i3.json`; runner `matrix/matrix.py`;
night script `matrix/run-i3-night.sh`.

| config | model, window | harness | tasks x reps | est. wall |
|---|---|---|---|---|
| `q3-4b-16k` (was `q7b-16k`) | Qwen3-4B int4, 16,384, f16 KV (was Qwen2.5-7B-Instruct q4_k_m) | opencode, Codex | 5 x 3 | 15 min |
| `q3-4b-28k-cc` (was `q7b-32k-cc`) | the same at 28,672 (was the 7B at 32,768), one KV slot | Claude Code | 5 x 1 | 45 min |
| `mellum-16k` | Mellum2.1 int4mix, C' expert streaming (45 of 64), 16,384 | opencode, Codex | 5 x 1 | 40 min |
| `mellum-32k-cc` | Mellum2.1 at 32,768, one KV slot | Claude Code | 5 x 1 | 45 min |
| `q3-4b-9k-codex-ovf` (was `q7b-9k-codex-ovf`) | Qwen3-4B at 9,408 (I1 live gate) | Codex | bulk, edit, fix x 3 | 10 min |
| `q3-4b-11k-opencode-ovf` (was `q7b-11k-opencode-ovf`) | Qwen3-4B at 11,520 (I1 live gate) | opencode | bulk, edit, fix x 3 | 10 min |
| `q3-4b-28k-cc-ovf` (was `q7b-28k-cc-ovf`) | Qwen3-4B at 28,672 (I1 live gate) | Claude Code | bulk, edit x 1 | 35 min |

**Model swap, 2026-10-13 (owner decision, after KV0).** KV0's pilots (`docs/measurements/kv0-traffic-2026-10-13.md` §1a, §1b) found that
Qwen2.5-7B-Instruct cannot drive opencode, Codex or Claude Code: 20 sessions made 1 to 6 requests each and none passed, because it
answers in prose or writes its tool call as text, and Qwen2.5-Coder-7B made 1 or 2. Qwen3-4B made 9-request chain sessions in opencode
and Codex and 4 to 22 requests on the repository tasks. The five cells that ran the 7B now run **Qwen3-4B int4**
(`/home/francis/models/qwen3-4b`, `--quant int4`, f16 KV): the configurations are renamed `q3-4b-*` (the old names are in the table), and
Claude Code's window is **28,672** (1.42 x its 17.7k first request under Qwen3's template with connect's settings), because 32,768
positions of Qwen3-4B's KV are 4.8 GB. Mellum's cells are unchanged. The I1 live-gate windows (9,408, 11,520, 28,672) keep their sizes.
The plan, the script header and the queue entries (A and C re-queued under the same names with the new `ONLY` lists) carry the change; the serve
binary stays the one pinned at queue time, which declines the batched prefill past 12,160 keys, so a Claude Code first request in
these cells takes the per-token path (about 4.5 minutes, KV0) until the fix of `docs/measurements/kv3-cuda-2026-10-13.md` is in a
pinned binary. The estimates are the old ones.

The 30 runs of the registered screening scope are all present (3 harnesses x 2 checkpoints x 5 tasks); the 7B's opencode and Codex cells
carry two more repetitions each, and the three `-ovf` configurations are the I1 gate. The registered "one repetition" is kept as the
reading rule: the first repetition decides the cell. The two extra repetitions of the cheap 7B cells are the registered follow-up for
fragile cells, run in the same night because they cost minutes; they are reported beside it, not instead of it.
- **Three queue entries**, each under "3 h or less": job A (`q3-4b-16k`, `q3-4b-28k-cc`) estimated 80 min; job B (`mellum-16k`,
  `mellum-32k-cc`) 130 min; job C (the three `-ovf` configurations, now Qwen3-4B) 70 min; 255 min in all, on the one box and so one after another. The wall figures above come from pilots (below), not from the
  1.5 h the screening scope guessed before the census.
- **Per-run timeouts** (a hang fails loudly): 15 min for opencode and Codex, 25 min for Mellum, 30 to 40 min for Claude Code. The job has a
  deadline and starts no run past it; a run that sees a `halted` answer is voided and redone.

*Pilots behind the estimates (exploratory, one run each unless said, never quoted as results):*
- opencode and Codex on the 7B at 16,384, ten tasks in 2 minutes (3 to 21 s each; the 7B passed 2 of 10, mostly by giving up or by
  narrating a tool call). In the pilot the Codex "edit" run printed a patch as prose where it had no `apply_patch`; one run decides
  nothing about I6 (below).
- Claude Code on the 7B at 32,768: 705 s ending on the compaction failure (defect 3); 284 s after the output cap.
- Mellum2.1 at 16,384 with `--moe-cache-experts`, Codex and opencode, read/search/edit: 2.3 to 4 minutes a run, a 6.1K-token first request
  takes **about 67 s** to its first token (about 90 tokens a second of prefill) and the next turn of the same session does not reuse it
  (68 s again). Five of six passed. Two leads, not fixed: Codex's "read" answer on Mellum began `5</think> maybe? Or just "5"?`, the model's
  reasoning in the reply text on the Responses route; and Mellum fills 7.38 GB of the card.
- Mellum2.1 at 32,768: loads resident, 7.33 GB, KV 1.0 GB (not run beyond loading).

**Viability, I4's rule, for every cell in the plan:** Codex 5,850 x 1.25 = 7,313 and opencode 7,165 x 1.25 = 8,956 are under 16,384 (and
under their overflow windows, 9,408 and 11,520); Claude Code 20,199 x 1.25 = 25,249 is under 32,768 and 28,672. No cell is refused by `connect`.

**What did not run, and why.** I3's matrix itself (it is the night job), the live Responses-compaction read for gradual growth (3c is it),
the second-session time-to-first-token comparison of I4 (needs a model), Kilo (its first request, 15,144 tokens, is not viable at 16,384) and
Hermes (it refuses a window under 64,000) as I3 cells (the screening scope is the three dialect leaders).

**Amended 2026-10-13, before the run: the serve binary was re-pinned** at `81791568` (owner: "fix and re-pin"). Codex sends no output cap
and was served 512 tokens a turn, thinking included; it now gets the limit `connect` wrote. Only the Codex cells change. The rules and the
predictions stand. Record: `docs/measurements/integration-harness-2026-10/i3-repin-2026-10-13.md`.

#### I3 read, 2026-10-14 (the night of 2026-10-13), graded against the plan above

Record and raw data: `docs/measurements/integration-harness-2026-10/i3-night-2026-10-13.md`. All three jobs are valid: the sandbox self-test
passed and caught its planted weaknesses in each, the planned model and window were served, the record counts are the plan's (35, 15,
20), and no run saw a `halted` answer, a 429 or a 5xx.

| cell | r1 (decides) | **verdict** | beside it |
|---|---|---|---|
| opencode x Qwen3-4B, 16,384 | 0/5 (1/5 as recorded; `read` r1 was a false pass) | **no** | r2 3/5, r3 3/5 |
| Codex x Qwen3-4B, 16,384 | 3/5 | **fragile** | r2 2/5, r3 3/5 (the same-night follow-up); `write`, `fix` 0 of 6 |
| Claude Code x Qwen3-4B, 28,672 | 1/5 (two 30-minute timeouts) | **no** | both timeouts as passes would give fragile |
| opencode x Mellum2.1, 16,384 | 4/5 | **works** | |
| Codex x Mellum2.1, 16,384 | 2/5 | **fragile** | three repetitions queued; publication waits for them |
| Claude Code x Mellum2.1, 32,768 | 1/5 (two timeouts) | **no** | |

**I1's live gate:** Codex **VOID** (every run an incurable jump, a run that never reached the window, or a harness that compacted
first), opencode **VOID** (the same), Claude Code **FAIL** (one run: 1 overflow, 3 compactions, the session ended on Claude Code's
"Autocompact is thrashing"). The FAIL is curable by hand count, so it is not void (a 5,154-token step against 9,961 of room), and the
fixed part fits. **The FAIL stands as registered (owner, 2026-10-14).** The cause observed, which does not change the verdict: Claude
Code's own auto-compaction (`trigger: auto`), twice before goinfer had answered any overflow and then a third time, ending on its own
"Autocompact is thrashing". The window, 28,672, was barely above its 17,711-token first request plus the 7,168 reserved for output. The
predictions (Codex PASS, opencode FAIL, Claude Code the unknown) were not tested for the two VOID harnesses. With Qwen3-4B the gate as
built cannot read the wording on Codex or opencode, because the model reads all eight `bulk` files in one step. A fresh registration
is registered in I1 (§2, "I1 live gate, second design (I1-v2)").
The record also notes an outside read on the box during job B, at 21:19, and shows that it moved no graded cell.

**Recorder defects found by the grade, fixed in `matrix/`:** an empty reply was graded against the raw output, which produced the false
pass; `/v1/messages` prompt sizes were never recorded, so the I1 grader's incurable-jump test skipped Claude Code without a warning.
**Leads, not fixed:** Claude Code subagents ask for `claude-sonnet-5-5` and get a 404 (`connect` does not set that slot); Mellum2.1 on
`/v1/responses` can spend a whole turn in reasoning and send an empty message; Mellum2.1 on `/v1/messages` writes tool calls as text with
broken JSON.


### I4 — First-request census (by day, minutes per harness)

Everything above needs one number per harness that nobody has measured except for opencode: the size of its first
request. With `-log-requests` it is nearly free. Serve a small model on nobara-pc at a window nothing overflows
(`-ctx 32768`), point each harness at it, send one trivial prompt headless, read `prompt_tokens` from the log line,
and record the tool count. Then start a second session from a different directory and compare its first turn's time to
first token with the first session's. A large drop means the shared system-prompt prefix was reused, which tells a
later decision what reusing a harness's prompt across sessions would buy. No code.

**Census of the other eight, 2026-10-13** (`docs/measurements/integration-harness-2026-10/harness-census-2026-10-13.md`; nobara-pc, each pinned under
`~/harnesses`, first request recorded headless, Qwen2.5 tokenizer, a CPU goinfer for the counts; nothing needed a login or a paid account, so none
was skipped; no request body is committed):

| harness, version | first request | tools | viable at 16,384 (I4's rule) |
|---|---|---|---|
| Pi 1.2.0 | 1,330 | 4 | yes |
| Oh My Pi 18.10.3 | 6,970 | 11 | yes |
| Hermes 0.21.7 | 11,276 | 24 | the request fits; **Hermes exits under a 64,000 window** before sending any chat request |
| OpenClaw 2026.10.3 | 7,166 | 11 | yes |
| Cline 3.0.71 | 4,606 | 25 | yes |
| Goose 1.54.0 | 4,001 (+ a 413-token title request) | 17 | yes |
| Aider 0.86.2 | 593 empty directory (a real repo adds up to a 4,096-token map) | none | yes |
| Kilo 7.8.8 | 15,144 | 23 | **no** (18,930 needed) |

Things the census found that bear on later work: Cline asks for `max_tokens` equal to the window less its own input estimate and ignores its
`maxTokens` setting; Pi's default compaction reserve of 16,384 equals a whole 16K window; Goose and Aider exit 0 on an error; Hermes probes
`/api/v1/models`, `/api/tags`, `/v1/props`, `/props` and `/version` before it decides (what goinfer answers to those was not read); Cline leaves a
detached hub daemon behind each run. For an unattended sandbox: Cline, Kilo, Oh My Pi, Aider and Goose look up other hosts (telemetry,
catalogs, package indexes), which a network namespace with one address refuses; none needed a network to start. Install quirks (OpenClaw wants
Node 24, Oh My Pi wants Bun, Aider wants Python under 3.13) are in the record. The agent that ran it also created `~/.hermes`, `~/.pi`, `~/.bun`,
`~/.config/kilo` and `~/.cache/kilo` in the real home directory with early `--help` probes, and left them.

Known so far: opencode 11,137 tokens (measured, R19); **Claude Code 19,692 (measured 2026-10-12, 2.1.268; 1,063 with
`--bare`; the earlier 6–10K was an estimate and wrong)**; **opencode 1.19.0 7,165 and Codex 0.162.1 5,850
(measured 2026-10-12)**; Pi under 1K (an estimate). **Viability, pre-registered:** a harness is *viable* on a (box, model, `-ctx`) when its first
request plus 25% fits the window; *marginal* when the request fits but the margin does not; *not viable* otherwise, and
`connect` refuses it.

### I5 — An in-process transport: the HTTP API as the library API

Every Go SDK and agent framework checked accepts a caller-supplied HTTP client:

| library | hook |
|---|---|
| `openai-go`, `anthropic-sdk-go` (official) | `option.WithHTTPClient` (anything with `Do(*http.Request)`), `option.WithMiddleware` |
| Genkit Go | `compat_oai.OpenAICompatible{…, Opts}` with `option.WithHTTPClient`; the Anthropic plugin likewise; the Ollama plugin has no hook |
| Eino | `HTTPClient` on `openai.ChatModelConfig` and `claude.Config` |
| LangChainGo | `openai.WithHTTPClient`, `anthropic.WithHTTPClient` |
| ADK-Go | `openaimodel.ClientConfig{HTTPClient, API}`; `API` defaults to the Responses API |

So an `http.RoundTripper` that calls goinfer's handler directly (no listener, no port, no daemon, no auth) gives
in-process local inference to all of them with no framework-specific code. It brings everything the server already
does: three dialects, schema-constrained output, union-grammar tool calls, and reasoning separation
(`-reasoning-format`, `internal/serveapp/think.go:16`). It is "no sidecar" for language models, and a llama.cpp
wrapper cannot offer it without cgo.

**Against the facade** (`task-embed-and-harness-ux.md` §2): complementary. The facade is a goinfer-native API for
someone starting from nothing (`Open`, `Chat`, `Into[T]`); the transport is for someone who already writes against an
SDK or a framework, which is most Go developers building agents. The transport needs no API design, since the HTTP API
is the contract and is already tested, and it is smaller. Its cost is JSON encode and decode and SSE framing per
request and per token: negligible beside inference for chat, and worth one by-day measurement for embeddings in a
loop. **Lean: the transport first.** The owner decides.

**The real work is one seam.** The mux is built inside the serve entry point (`internal/serveapp/main.go:668`), after
flag parsing and model loading, and `internal/serveapp` cannot be imported. Extract a constructor,
`NewHandler(Config) (http.Handler, io.Closer, error)`, into a public Experimental sub-package (the reasoning of
`task-embed-and-harness-ux.md` §7 F.1 for the facade's place applies), and build the transport on it:

- streaming through an `io.Pipe`, so SSE reaches the SDK incrementally rather than buffered;
- the request's context cancels the generation (the halt path stops within one token, `task-halt-2026-09.md`);
- operator choices stay flags or `Config` fields, with no new `GOINFER_*` reads (`testdata/env_reads.txt`).

**Keep the root `go.mod` at two requires.** It lists only aikit and `golang.org/x/text` today, and the README leans on
that. SDK-driven tests go in their own module, stitched in CI with `go work init` as `demo/agent` is.

**Gate.** In that module, against a committed tiny checkpoint (`testdata/tiny-qwen2-moe` carries a tokenizer, R25):
the official `openai-go` and `anthropic-sdk-go` examples for chat, streaming, tools and structured output pass
in-process; one smoke test each for Genkit, Eino, LangChainGo and ADK-Go (ADK-Go's default exercises the Responses
route, and so H-a's fix); a mid-stream cancel stops generation and leaves no goroutine behind; the root `go.mod` is
unchanged.

**I5, the seam, built 2026-10-13 (its SDK gate is not).**
- **In `internal/serveapp`:** the route table and the listener-independent checks moved out of the entry point
  unchanged, and a constructor builds a handler from the binary's own flag set, so a setting cannot mean one thing on
  the command line and another in process. The flags that configure the listener or the process are refused by name.
- **Public, Experimental:** package `server`, two functions (`NewHandler`, `Transport`); `docs/api-tiers.md` lists it.
- **Read on the committed tiny chat model with `net/http`:** a chat completion answers; a stream arrives as separate
  events and ends with `[DONE]`; `-api-key` is enforced; a stream cancelled partway leaves no goroutine; after
  `Close` the model is gone (404). The root `go.mod` is unchanged.
- **The SDK module, built the same day:** `sdktest/`, a module of its own (`openai-go/v3` v3.77.0,
  `anthropic-sdk-go` v1.80.0), stitched into CI with `go work init` and listed in `gate quick`. On the committed tiny
  model with its window set to 2,048 (a tool definition's prompt does not fit the fixture's 128):
  - OpenAI SDK: a chat completion, a stream, a forced tool call whose arguments parse to the schema, a
    schema-constrained reply, the Responses API, and a conversation past the window read back as `*openai.Error` with
    code `context_length_exceeded` (I1, end to end);
  - Anthropic SDK: a message, a stream accumulated to a stop reason, a forced `tool_use`, and the overflow as a 400
    saying "prompt is too long:".
  - The root `go.mod` is unchanged: two requirements.
- **Still owed for I5's gate:** one smoke test each for Genkit, Eino, LangChainGo and ADK-Go; and the mid-stream cancel
  through an SDK (it is read with `net/http` only). The typed `Config` the item sketched was not built: the
  constructor takes flags.

### I6 — Codex's `apply_patch`, honoured rather than dropped (candidate; desk check first)

H-a's fix drops tool types the server cannot run and names them in `X-Goinfer-Tools-Dropped`. For Codex that includes
its `custom` tool with `syntax: "lark"`, which is how Codex sends `apply_patch`, its edit tool. Whether Codex can still
finish an edit task without it, through shell commands, is unverified; I3's Codex cell answers that.

If it cannot, this is the item that makes Codex usable, and few local engines could do it: Codex sends the grammar
with the tool. Desk check first: read the grammar Codex sends for `apply_patch`, and whether `constrain`, which
compiles JSON-schema and struct grammars today, can express it through a translator of modest size. If it can, honour
`custom` tools with their grammar and emit `response.custom_tool_call_input.delta`. If not, record why and stop.
**Decided by I3's Codex cell:** if edits complete without it, park this; if they fail, do the desk check.

### I7 — Raise the default context toward 32768 when there is room (owner, 2026-10-12)

The census (§0a) put Claude Code's first request at about 20,000 tokens: over the 16,384 that CUDA's and M2's
fit-by-default aim for. Asked what to do about it, the owner said: "it would be nice to auto-raise it if there's room".

- **The change:** the fit-by-default candidate goes from 16,384 to 32,768 on CUDA (`fitDefaultCtx`) and on Metal
  (`MetalFitDefaultCtx`, once M2 is merged), with `goinfer-chat fit`'s default and the banner's harness line following.
  Nothing else in the rule moves: the planner still shrinks to what fits beside the weights, the margin and every KV
  slot, never under 4,096, and the opt-outs (`--fit=off`, paged MoE, an unknown memory figure, an explicit `-ctx`)
  stay.
- **What it costs, to say in the CHANGELOG:** KV is allocated up front in every slot, so a model with room holds up
  to twice the KV it does at 16,384. A model without room lands where it does today.
- **Not decided yet, for the registration that precedes the code:** whether the context should still give way to
  slots all the way (today's rule) when the candidate is this large, or stop at a harness-sized floor first. On an
  8 GB card with four slots that choice decides what a 1.5B gets.
- **To build after M2 merges** (both touch the same constants and tests), with the CUDA twins on nobara-pc.

**G-I7-speed, the Mac half, registered 2026-10-12 before it runs (night).** A larger KV must not slow short turns.
- **Instrument:** G-M2-speed's, unchanged (`task-minimax-code-compat-2026-09.md`, "G-M2-speed"): `scripts/bench_peer.py`
  phase A at depth 128, `BENCH_BACKENDS=metal`, `BENCH_MODELS=1.5B,7B`, `BENCH_ENGINES=goinfer,goinfer_old`,
  `BENCH_ABBA=3`, `BENCH_REQUIRE_BACKEND=1`; grader `grade_m2_speed.py`. **New arm:** a trial build, M2's branch with
  the Metal candidate set to 32,768 (branch `m2b-candidate-32768`, `af0d89ab`; a one-line change made to be timed,
  not the change that will merge). **Old arm:** main at `c75a283a`, which holds 4,096. Neither is given `-ctx`.
  Script: `docs/measurements/integration-harness-2026-10/run-g-i7-night.sh`.
- **Statistic and rule:** as G-M2-speed. Per model, the median over six blocks of new / old; SHIP at 0.97 or more on
  both models, PARKED in [0.95, 0.97) on either, FAIL below. VOID if the new arm did not resolve above 4,096 in every
  cell, the old arm's is not 4,096, a cell errored or a block lacks an arm.
- **What a verdict licenses:** SHIP says a 32,768 default does not slow a depth-128 turn on Metal on these two
  models. It says nothing about CUDA (its own gate, with the code) or about memory.
- **Also read, not graded:** one request of 20,199 tokens (Claude Code's first request) to each model on the trial
  build, with swap before the load and after the answer.
- **Stopping rule:** fixed, 24 cells. **Cost basis:** G-M2-speed's estimate plus two long prefills: a 45 min job.
- **Predictions, before the run:** the median is within 1% of 1 on both models, 0.7 (G-M2-speed's reasoning: a step
  at depth 128 reads the first 129 positions whatever the buffer's size). The 7B resolves 32,768 with 2 slots, 0.4:
  it needs about 8.7 GB of live budget (4.5 GB of weights, 3.8 GB of KV, the margin). If it resolves less, the
  comparison is of what the new default gives on this Mac, and the record says which context that was.

**G-I7-speed, the Mac half, read on the night of 2026-10-12: SHIP on speed, and a refusal that changes I7's design.**
Records: `docs/measurements/integration-harness-2026-10/i7-night-2026-10-12/`.

| model | context the trial build resolved | new (tok/s) | old (tok/s) | new / old by block | median |
|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | 32,768 in every cell | 93.60 | 93.45 | 1.0045 1.0019 1.0014 1.0016 1.0017 1.0013 | **1.0017** |
| Qwen2.5-7B | 25,736 or 32,768, by the memory free at each load | 30.70 | 30.71 | 0.9999 1.0013 1.0001 0.9980 1.0020 1.0000 | **1.0000** |

- **A KV of twice M2's size does not slow a depth-128 turn on Metal** on these two models.
- **The 20,199-token request, not graded:**
  - the 1.5B at 32,768 answered a 20,954-token prompt in 34.2 s with no swap growth;
  - **the 7B refused it with a 413.** It had resolved 31,328 positions in 2 slots (3.3 GB of KV), which left 2.5 GB
    available, and the per-request admission priced the prompt's prefill at 3.21 GB
    (`decoder/prefill_budget.go:Model.AdmitPrefillMemory`). So the larger default took the memory the request it was
    raised for needs. The night job shows FAILED for this refusal.
- **What that means for the code, before it is written:**
  - **The fit must leave room for the prefill of the request the context is for**, or the bigger number is
    decoration. On a 16 GB Mac with a 7B that probably means one slot at a harness-sized context, or two at 16,384:
    the slots-against-context question above, now with a measurement behind it.
  - **The admission's estimate needs checking against the resident path.**
    `decoder/prefill_budget.go:prefillScratchBytes` prices the feed-forward scratch of one sweep over the whole
    prompt (8 bytes x the feed-forward width x the prompt's tokens), and calls itself an upper bound. Metal's
    prefill runs in chunks, so the real figure may be far smaller. Measure it before trusting either the refusal or
    the room.
    - **Checked 2026-10-13, in the source: the refusal was right, and "runs in chunks" was wrong.** Metal's batched
      pass allocates its half-precision scratch for the whole prompt in one call (two residual-width buffers, the
      q/k/v and attention outputs, and three feed-forward-width ones): about 144 KB a prompt token on Qwen2.5-7B
      (3.0 GB at 20,954 tokens) and 67 KB on the 1.5B. The chunked prefill exists but runs only while other
      generations are decoding. The admission's own formula is the CPU's and lands near Metal's figure by
      coincidence. I8 below is the fix.
- **Predictions scored.** "The median is within 1% of 1 on both models (0.7)": yes. "The 7B resolves 32,768 with 2
  slots (0.4)": in some cells and not others; 31,328 on the served load.

### I8 — On Metal, prefill a long prompt in blocks and admit it by the block (owner, 2026-10-13: "do steps 1 and 2 now")

**Why.** I7's night read above: a 7B with 31,328 positions refused a 20,954-token request, correctly, because the
batched pass's scratch grows with the whole prompt. The same term is what a context past 32,768 would hit first
(9.5 GB for a 64K prompt on a 7B). Bounding it is the precondition for I7's code and for any longer window.

**The change, two parts.**
1. **Blocks.** The function both resident generation paths share for the prompt takes a suffix longer than the
   resident's block in passes of that many rows, each continuing the last at its position, and builds each block's
   embeddings as it goes. A resident opts in by reporting a block size; only Metal does. Metal's block is the number
   of rows whose scratch fits **512 MiB** (a multiple of 64, never under 256): about 3,700 rows on the 7B and 7,900
   on the 1.5B, so a prompt under that length is one pass, as today. The last pass keeps at least 8 rows. A block the
   pass declines sends the rest down the token-by-token path, as a declined whole pass does today.
2. **Admission.** A request on a resident that reports its own scratch is priced at that figure for one block (or
   for the prompt, when it is shorter), plus the block's host embeddings, in place of the CPU formula over the whole
   prompt. CPU requests, and residents that report nothing (CUDA, WebGPU), are priced as before.

**Not in this item:** image turns (their pass is one call and already stops at 4,096 keys); CUDA and WebGPU, whose
scratch is device memory priced today by the CPU's formula against host memory (worth its own look); the 32,768
ceiling; I7's default.

**Gates, registered before the code.** All on the M1 Pro; the tests name the gate.

| gate | when | instrument | rule |
|---|---|---|---|
| G-I8-1 identity, tiny | day | Metal tiny fixtures (dense, generic MoE, Gated-DeltaNet), through a generation and not the pass alone: the same prompt with the block forced to 64, 100 and 256 rows against one pass | seed logits, generated tokens and every K/V row equal bit for bit |
| G-I8-2 identity, real | day | Qwen2.5-Coder-1.5B int4 on Metal, a 6,500-token prompt, block forced to 2,048 rows (four passes) against one pass | last-row logits equal bit for bit |
| G-I8-3 the price covers the pass | day | a counter of the bytes the pass really allocates, on each tiny fixture and the 1.5B | reported figure >= allocated, and within 2% + 64 KB of it |
| G-I8-4 admission | day | the admission through its caller's arguments, memory probe set by the test hook | a prompt whose whole-pass price exceeds the room and whose block price fits is admitted on a blocked resident, and refused when the resident reports no block |
| G-I8-5 speed | day, in process | the 1.5B, a 12,288-token prompt, block forced to 2,048 rows (six passes, the 7B's shape at 21k) against one pass; five alternating pairs after one warm-up pair; statistic: median over pairs of blocked / whole prefill time | **PASS at 1.03 or under**; PARKED in (1.03, 1.05], owner decides; FAIL above 1.05, blocks do not become the default |
| G-I8-6 the refused request | night | serve the 7B with `-ctx 24576`, send last night's 20,954-token request; swap read before the load and after the answer | answered, swap growth 64 MB or under |

- **Planted defects, each must turn its gate red:** a block continued one position late (G-I8-1); the price without
  its feed-forward term (G-I8-3); admission pricing the whole prompt on a blocked resident (G-I8-4).
- **What a verdict licenses.** G-I8-1 and -2 say blocks do not change a reply on the families Metal's pass covers.
  G-I8-5 is about the time to prefill one long prompt on one model; it is not a served TTFT and is quoted as
  in-process. G-I8-6 says the request I7 was raised for is answered on this Mac, at night, with nothing else running.
- **Stopping rule:** fixed as written; G-I8-5's ten timed prefills run once. **Cost basis:** each 12,288-token
  prefill on the 1.5B is about 20 s by last night's rate, so G-I8-5 is about 5 minutes with the load; the rest are
  seconds; G-I8-6 is a 10-minute night job.
- **Predictions, before any run:** G-I8-2 holds at real size, 0.85 (the pass routes on the whole prompt's length and
  the tiny chunk-invariance tests pass; the risk is a tile-shape effect only a real width shows). G-I8-5 reads 1.03 or
  under, 0.7 (the attention work is the same; each extra pass costs one LM-head row and its buffer setup). G-I8-6 is
  answered, 0.8 (KV 2.8 GB in two slots and about 0.55 GB a block; the risk is the 7B not going resident at 24,576).

**I8 results, 2026-10-13 (day; the night gate was dropped unread). Merged to main the same day** as the "KV-pre" step
of `docs/tasks/task-kv-architecture-2026-10.md`, whose KV3 extends blocks and block admission to every resident. Code `d7a100ba` on branch
`integration-harness-2026-10`. Records: `docs/measurements/integration-harness-2026-10/i8-day-2026-10-13/`.

| gate | read | verdict |
|---|---|---|
| G-I8-1 identity, tiny | dense fixture, 1,000 tokens, blocks of 64, 100 and 256; four tiny MoE fixtures (one a Gated-DeltaNet hybrid), block of 16: tokens, log-probabilities and every K/V row (and the DeltaNet window and state) equal one pass | **PASS** |
| G-I8-2 identity, real | Qwen2.5-Coder-1.5B, 6,500 tokens, passes of 2,048 / 2,048 / 2,048 / 356 rows against one: 0 of the logits differ | **PASS** |
| G-I8-3 the price covers the pass | every run above allocated exactly the reported figure (equal to the byte, so inside the 2% band); the 1.5B is 65.5 KB a row | **PASS** |
| G-I8-4 admission | a 20,000-token request with room for a block and not for the whole-prompt price: admitted on a blocked resident; refused with no block, with a block that does not fit, and on the CPU path | **PASS** |
| G-I8-5 speed, in process | 1.5B, 12,288 tokens, blocked / whole by pair: 0.9972 0.9971 1.0292 1.0021 1.0238; **median 1.0021** (one pass 17.3 to 17.6 s) | **PASS** (bar 1.03) |
| G-I8-6 the refused request | **not read:** queued as `i8-7b-refused-request`, then dropped the same day on the owner's word (a re-architecture of the KV is planned, which makes it moot) | none |

- **All three planted defects turned their gates red** and were reverted: a block continued one position late
  (the decoder's fake-resident tests and the Metal dense and MoE identity tests); the price without its feed-forward
  term (the Metal scratch assertions); admission pricing the whole prompt on a blocked resident (the admission test).
- **What a block is on these models.** The 1.5B: 8,000 rows, so a prompt under 8,008 tokens is still one pass. The 7B,
  computed from its dimensions and not loaded today: 144 KB a row, a block of 3,712 rows (0.54 GB), so last night's
  20,954-token request would be six passes and priced at about 0.59 GB with its embeddings, against the 1.77 GB of
  room it was refused for 3.21 GB under. That is a calculation: G-I8-6 would have measured it and was not run.
- **Amendments to the registration, 2026-10-13.**
  - *G-I8-1's fixtures (written after the first run showed it).* The tiny MoE fixtures hold 64 to 100 positions, so
    their block is 16 rows on a prompt of 56 to 92 tokens, not 64 / 100 / 256. **No Gated-DeltaNet model is split at
    real width:** Qwen3.5 0.8B has a head dimension of 256, runs the exact attention kernel, and so reports no block
    (asserted: a forced block leaves it one pass). DeltaNet's block coverage is the tiny hybrid only.
  - *G-I8-6 gained a do-nothing arm (before any run):* main at `e2ffb953` with the same `-ctx 24576`, expected to
    refuse. If it answers too, the verdict says the night's memory and not the change may explain the answer.
- **Also read, not registered and not graded:**
  - the 1.5B's own block on the same 12,288 tokens (passes of 8,000 and 4,288): median 1.0087 of one pass;
  - G-I8-6's script once by day: **VOID on both arms**, the 7B did not go resident (5.75 GB needed, 4.97 GB of live
    budget). So there is no daytime answer for the 7B.
- **Predictions scored.** G-I8-2 holds at real size (0.85): yes. G-I8-5 at 1.03 or under (0.7): yes, 1.0021. G-I8-6
  (0.8): not scored, the gate was not run.
- **What this leaves.**
  - A model on the exact attention kernel (a head dimension over 128, or one that is not a multiple of 8), an
    E-model, a layer-major route, `--kv i8` and `--exact-prefill` are not split: their scratch still grows with the
    prompt where they take the batched pass at all, and they are priced by the old formula.
  - The speed reading is one model and one length. A 7B-width block on a deeper prompt has not been read at all.
  - CUDA and WebGPU requests are still priced by the CPU's formula against host memory.

## 3. Considered and not recommended (recorded so they are not re-proposed)

- **Ollama's native API (`/api/*`).** It looked like the reach play; the evidence says otherwise. Of the apps checked,
  only Enchanted and Home Assistant core speak Ollama's API exclusively. Zed, Continue, Open WebUI, AnythingLLM, n8n
  and Page Assist all offer an OpenAI-compatible option, and Cline's Ollama provider has an "OpenAI Compatible"
  sibling (Gate 0). Building it means a fourth wire to keep conformant (NDJSON streaming, tool arguments as objects
  rather than strings, `/api/show` capabilities) for two apps. The one thing lost: Zed's Ollama provider reads the
  context length and capabilities from `/api/show`, where its `openai_compatible` provider makes the user type them,
  and I2 covers that. Revisit only for a named integration that matters and cannot be reached otherwise.
- **Advertising model ids that contain "claude" or "anthropic"**, so Claude Code's opt-in `/model` discovery lists
  them. Magnitude uses an `anthropic-local/` prefix because it proxies; I2 sets `ANTHROPIC_MODEL` directly.
- **An MCP server for local embeddings or extraction.** Useful (a cloud agent handing private-data work to a local
  model), but it belongs with ken's code-search surface, not the engine. A Go MCP host that wants local sampling
  (`CreateMessageHandler`) gets it through I5.
- **Roo Code** (shut down 2026-05-15) and **Continue** (read-only and unmaintained since July 2026) as targets.
- **Diffusion and image-generation integrations:** a different engine (a scheduler and a VAE), outside this lane.

## 4. Order, and what runs when

| step | what | when |
|---|---|---|
| 1 | Claude Code recipe: add `ANTHROPIC_DEFAULT_HAIKU_MODEL` and the window variable (§5) | **done 2026-10-12** |
| 2 | I4 census | Claude Code, opencode, Codex **done 2026-10-12**; the other eight **done 2026-10-13** |
| 3 | harness-reliability Gate 1 (the `/v1/messages` row) | **built 2026-10-12** |
| 4 | I1 overflow rule, with per-dialect tests | by day; the live confirmation joins I3 |
| 5 | I2 `connect` for Claude Code, opencode, Codex | by day |
| 6 | I3 screening | night queue, nobara-pc, after 3–5: **run 2026-10-13, read 2026-10-14** (above); Codex x Mellum2.1's three repetitions queued 2026-10-14; `docs/integrations/harness-matrix.md` after they read |
| 7 | I5 transport | by day, in parallel with 2–6; depends on none of them |
| 8 | I6 | decided by I3's Codex cell: **parked** until that cell reads; the pilot's one Codex edit printed a patch as prose, which is one run |
| 9 | the FIM and sidecar amendments (§5) | when those docs are next picked up |

**Not claimed:** none of this makes goinfer faster, and on an 8 GB card the window stays small. These items make the
limit visible before a session starts (I2, I4), survivable when it is reached (I1), and they publish where it actually
falls (I3).

## 5. Amendments to existing docs (for whoever next opens them; not applied here)

**`task-harness-reliability-2026-10.md`**
- **H-c:** replace "open as a decision" with I1's rule, or record why the owner prefers the current wording.
- **H-d:** the desk read found that Claude Code aborts a stream after 300 s with no *bytes* received, adjustable with
  `CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS`. If the limit is byte-level, the 10 s SSE-comment heartbeat
  (`internal/serveapp/helpers.go:274`) already satisfies it with a 30× margin. The trace still decides; this says what
  to expect. It also answers `task-embed-and-harness-ux.md` §7 F.4, which records the timeout value as unknown.
- **Gate 2:** the "harness-scale: skip — too small" verdicts for the small Qwen coders were measured in early
  September, before the union grammar shipped and before a bare call naming a supplied tool was accepted as a call
  (both by 2026-09-24). A model that wrote a bare call then would have read as "answered without calling". Re-run them;
  small models, by day.
- **Gate 2:** add Mellum2.1's result (ok, 7.4 GB, expert streaming; `task-mellum21-2026-10.md`), and Hermes's 64K
  window floor as a hardware gate: on an 8 GB card Hermes cannot be used unless the served window is at least 65,536.
- **Codex:** after H-a, `apply_patch` is dropped (I6), and the Responses route emits neither `response.output_item.*`
  nor `response.failed`, both of which Codex reads. I3's Codex cell is the first trace.

**`task-editor-fim-2026-09.md`, F2's targets**
- Continue is read-only and unmaintained since July 2026: do not write `docs/integrations/continue.md`.
- llama.vscode and llama.vim post to llama.cpp's `/infill` (`input_prefix`, `input_suffix`, `input_extra`, plus
  `n_predict: 0` warm-up calls that pre-cache context). That is a route to add in F1, and the warm-ups map directly onto
  the doc's §3 property that typing forward is nearly free.
- Tabby's `openai/completion` kind sends `prompt` and `suffix` to `/completions`, which F1's `suffix` covers. Twinny
  builds FIM prompts client-side for `/v1/completions` and needs nothing new. JetBrains AI Assistant's inline
  completion takes an OpenAI-compatible endpoint (wire format unverified), and Mellum2, JetBrains' own FIM model, is
  already F1's first family.

**`parked/task-bindings.md`: unpark A0 alone**
- The convention to copy is Jan's llama.cpp worker: listen on port 0, print exactly one JSON line `{port, pid, models}`
  on stdout once listening, exit when stdin closes (the one parent-death signal that survives a force-quit in both
  Tauri and Electron), and answer `/health` with 503 until the model is loaded.
- Port 0 is unusable today without that line: serve calls `ListenAndServe` (`internal/serveapp/main.go:922`), which
  never reports the port it bound. The admin socket already shows the unix-listener pattern A0's first item wants
  (`internal/serveapp/admin_socket.go:45`).

**`docs/integrations/claude-code.md`** (applied 2026-10-12, with §0a's corrections)
- Add `ANTHROPIC_DEFAULT_HAIKU_MODEL` (background work; `ANTHROPIC_SMALL_FAST_MODEL` is deprecated) and
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS` set to the served window, and say that an unrecognised model id is otherwise assumed
  to have 200K.

## 6. Conventions

Checkpoints come from `~/models` on the box doing the run, never the archive. Anything over about ten minutes goes on
the night queue with an estimate (`night.py add --est`), one timed run per box. The decision rules above are registered
before their runs. New operator choices are flags or `Config` fields, not environment reads. Harness versions are
pinned in every record, because a harness that changes its prompt or its overflow matcher changes the result. Stage
explicit paths; the SDK-test module keeps the root `go.mod` unchanged.

## Sources

Desk read 2026-10-12, from each project's own docs or source:

- Claude Code: [env vars](https://code.claude.com/docs/en/env-vars), [model config](https://code.claude.com/docs/en/model-config), [LLM gateway protocol](https://code.claude.com/docs/en/llm-gateway-protocol), [gateway connect](https://code.claude.com/docs/en/llm-gateway-connect), [headless](https://code.claude.com/docs/en/headless)
- Codex: [model-provider-info lib.rs](https://github.com/openai/codex/blob/main/codex-rs/model-provider-info/src/lib.rs), [client.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/client.rs), [responses_error.rs](https://github.com/openai/codex/blob/main/codex-rs/codex-api/src/sse/responses_error.rs), [model_info.rs](https://github.com/openai/codex/blob/main/codex-rs/models-manager/src/model_info.rs)
- opencode: [providers](https://github.com/anomalyco/opencode/blob/dev/packages/web/src/content/docs/providers.mdx), [compaction.ts](https://github.com/anomalyco/opencode/blob/dev/packages/core/src/session/compaction.ts), [provider-error.ts](https://github.com/anomalyco/opencode/blob/dev/packages/ai/src/provider-error.ts); Kilo: [README](https://github.com/Kilo-Org/kilocode/blob/main/README.md)
- Pi: [models.md](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/models.md), [compaction.md](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/compaction.md), [overflow.ts](https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/utils/overflow.ts); Oh My Pi: [models.md](https://github.com/can1357/oh-my-pi/blob/main/docs/models.md)
- Hermes: [providers](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/integrations/providers.md), [model_metadata.py](https://github.com/NousResearch/hermes-agent/blob/main/agent/model_metadata.py), [error_classifier.py](https://github.com/NousResearch/hermes-agent/blob/main/agent/error_classifier.py)
- OpenClaw: [custom providers](https://github.com/openclaw/openclaw/blob/main/docs/gateway/config-tools/custom-providers.md); Cline: [CLI reference](https://github.com/cline/cline/blob/main/docs/cli/cli-reference.mdx), [error-classification.ts](https://github.com/cline/cline/blob/main/sdk/packages/llms/src/providers/error-classification.ts)
- Goose: [environment variables](https://github.com/block/goose/blob/main/documentation/docs/guides/environment-variables.md), [http_status.rs](https://github.com/block/goose/blob/main/crates/goose-providers/src/http_status.rs); Aider: [OpenAI-compatible](https://github.com/Aider-AI/aider/blob/main/aider/website/docs/llms/openai-compat.md); Copilot: [VS Code language models](https://code.visualstudio.com/docs/copilot/customization/language-models), [CLI BYOK](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/use-byok-models)
- Roo Code: [README](https://github.com/RooCodeInc/Roo-Code/blob/main/README.md); Continue: [README](https://github.com/continuedev/continue/blob/main/README.md)
- Magnitude: [README](https://github.com/magnitudedev/magnitude/blob/main/README.md), [integrations overview](https://github.com/magnitudedev/magnitude/blob/main/docs/integrations/overview.mdx), [connectors](https://github.com/magnitudedev/magnitude/tree/main/packages/harness-connections/src/connectors)
- Ollama: [routes.go](https://github.com/ollama/ollama/blob/main/server/routes.go), [api.md](https://github.com/ollama/ollama/blob/main/docs/api.md), [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility), [Anthropic compatibility](https://docs.ollama.com/api/anthropic-compatibility). Apps: [Zed local models](https://github.com/zed-industries/zed/blob/main/docs/src/ai/use-a-local-model.md), [Home Assistant OpenAI config flow](https://github.com/home-assistant/core/blob/dev/homeassistant/components/openai_conversation/config_flow.py), [Enchanted](https://github.com/gluonfield/enchanted)
- Fill-in-the-middle: [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [llama.vscode](https://github.com/ggml-org/llama.vscode/blob/master/src/llama-server.ts), [Tabby completion](https://github.com/TabbyML/tabby/blob/main/crates/http-api-bindings/src/completion/mod.rs), [Twinny](https://github.com/twinnydotdev/twinny/blob/main/src/common/provider-validation.ts), [JetBrains custom models](https://www.jetbrains.com/help/ai-assistant/use-custom-models.html)
- Go hooks: [openai-go requestoption.go](https://github.com/openai/openai-go/blob/main/option/requestoption.go), [anthropic-sdk-go requestoption.go](https://github.com/anthropics/anthropic-sdk-go/blob/main/option/requestoption.go), [Genkit compat_oai](https://github.com/firebase/genkit/blob/main/go/plugins/compat_oai/compat_oai.go), [Eino model components](https://github.com/cloudwego/eino-ext/tree/main/components/model), [LangChainGo llms](https://github.com/tmc/langchaingo/tree/main/llms), [ADK-Go openaimodel](https://github.com/google/adk-go/blob/main/model/openaimodel/openaimodel.go)
- Sidecars: [Tauri sidecar](https://v2.tauri.app/develop/sidecar/), [Jan's llama.cpp worker](https://github.com/janhq/jan/blob/main/src-tauri/plugins/tauri-plugin-llamacpp/src/engine/worker.rs)
