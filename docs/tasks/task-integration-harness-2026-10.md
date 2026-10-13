# Task: integration and harness work — context first, and what no doc owns yet (2026-10)

<!-- citations-at: 6e50c60941e7 -->

> **Status: STARTED 2026-10-12 (§0a).** Steps 1 and 3 are built and a gap nobody had listed is fixed: Claude Code
> 2.1.268 could not complete a turn against goinfer (H-e). Its real first request is 19,692 tokens, not 6–10K, which
> moves the thesis's numbers (§0a). **Was: SCOPED 2026-10-12.** A map across the docs that already own integration work, an order
> for them, six items none of them owns (I1–I6), and amendments the 2026-10-12 research supports for four existing
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
| Default context raised to 16384 on CUDA | `task-first-hour.md` R19 | shipped 2026-10-01; Metal unchanged at 4096 |
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
(R19's table; a 7B agent passes `-ctx 16384` and gets fewer slots). Metal's default is 4096. opencode's first request
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

**The trade, stated.** With the second branch a harness compacts against a `-ctx` the operator could have raised, and
the operator may not notice. `-log-requests` shows every overflow, and the message can still name `-ctx` when the fit
planner had room for more. The owner decides. This doc recommends the rule because on a small card it is the
difference between a long session that degrades and one that dies.

**Gate.** Unit tests per dialect through the real handlers, both branches, with a mutant per branch. Then live, inside
I3: each of Claude Code, opencode and Codex driven past the window. Pre-registered: **pass** if each overflow is
followed by at most one compaction and the session continues; **fail** on three or more consecutive compactions, or
the session ending on the overflow. A fail with the fixed part fitting means the wording is wrong for that harness.

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

### I6 — Codex's `apply_patch`, honoured rather than dropped (candidate; desk check first)

H-a's fix drops tool types the server cannot run and names them in `X-Goinfer-Tools-Dropped`. For Codex that includes
its `custom` tool with `syntax: "lark"`, which is how Codex sends `apply_patch`, its edit tool. Whether Codex can still
finish an edit task without it, through shell commands, is unverified; I3's Codex cell answers that.

If it cannot, this is the item that makes Codex usable, and few local engines could do it: Codex sends the grammar
with the tool. Desk check first: read the grammar Codex sends for `apply_patch`, and whether `constrain`, which
compiles JSON-schema and struct grammars today, can express it through a translator of modest size. If it can, honour
`custom` tools with their grammar and emit `response.custom_tool_call_input.delta`. If not, record why and stop.
**Decided by I3's Codex cell:** if edits complete without it, park this; if they fail, do the desk check.

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
