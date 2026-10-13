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
- **Not done:** the live half of the gate (a headless turn on each written file); the other eight agents in the
  table; parsing each agent's published schema; the warning from a skipped `tools, harness-scale` row.

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

### I4 — First-request census (by day, minutes per harness)

Everything above needs one number per harness that nobody has measured except for opencode: the size of its first
request. With `-log-requests` it is nearly free. Serve a small model on nobara-pc at a window nothing overflows
(`-ctx 32768`), point each harness at it, send one trivial prompt headless, read `prompt_tokens` from the log line,
and record the tool count. Then start a second session from a different directory and compare its first turn's time to
first token with the first session's. A large drop means the shared system-prompt prefix was reused, which tells a
later decision what reusing a harness's prompt across sessions would buy. No code.

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
| 2 | I4 census | Claude Code, opencode, Codex **done 2026-10-12**; the rest when installed |
| 3 | harness-reliability Gate 1 (the `/v1/messages` row) | **built 2026-10-12** |
| 4 | I1 overflow rule, with per-dialect tests | by day; the live confirmation joins I3 |
| 5 | I2 `connect` for Claude Code, opencode, Codex | by day |
| 6 | I3 screening | night queue, nobara-pc, after 3–5 |
| 7 | I5 transport | by day, in parallel with 2–6; depends on none of them |
| 8 | I6 | decided by I3's Codex cell |
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
