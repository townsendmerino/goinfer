# Harness wire dialects: Gate 0 of the harness-reliability task (2026-10-09)

Desk work for `docs/tasks/task-harness-reliability-2026-10.md` §2 Gate 0. Two research agents read each harness's
published package or source on 2026-10-09; every row names what was read. Each claimed gap was then checked
against `internal/serveapp` at `origin/main` 5802692a (the "serve today" column). Nothing here was run against a
live server: these are desk results, and a desk claim about a third-party client is a lead until a trace confirms it.

## Sources read

| harness | project and version read |
|---|---|
| Pi | npm `@earendil-works/pi-ai` and `@earendil-works/pi-coding-agent` 1.1.0 (2026-10-07); the repo now publishes as `earendil-works/pi` (the old `badlogic/pi-mono` move was not confirmed) |
| OpenCode | npm `opencode-ai` 1.18.35; repo `anomalyco/opencode` `dev` read 2026-10-09; custom providers via `@ai-sdk/openai-compatible` 3.0.67 |
| Hermes | PyPI `hermes-agent` 0.19.0 (Nous Research) |
| OpenClaw | npm `openclaw` 2026.9.9 (2026-10-08) |
| Codex | `openai/codex` `4aa94dc` = rust-v0.162.1 |
| Claude Code | v2.1.296 (npm latest 2026-10-09); code.claude.com `llm-gateway-protocol` and `llm-gateway-connect` |
| Oh My Pi | `can1357/oh-my-pi` `703a261` = 18.8.7 (npm `@oh-my-pi/pi-coding-agent`; "Fork of Pi") |
| Cline | `cline/cline` `d20b337` = 4.1.23 |

## Per harness

| harness | wire for a custom base URL | tools | streaming | fields beyond the basics |
|---|---|---|---|---|
| Pi | `api:` one of `openai-completions` ("most compatible"), `openai-responses`, `anthropic-messages` in `~/.pi/agent/models.json`; per-model `contextWindow` (default 128000) | OpenAI function tools; 4 defaults (read, bash, edit, write); `strict` only with `compat.supportsStrictMode` | always; `stream_options.include_usage`; reads `reasoning_content`/`reasoning` | `store:false`, `max_completion_tokens`, `reasoning_effort`, `developer` role; Responses: `include:["reasoning.encrypted_content"]`; Anthropic: `anthropic-beta: fine-grained-tool-streaming-2025-05-14`, `cache_control` and `eager_input_streaming` on tools, refuses to run with no API key |
| OpenCode | `@ai-sdk/openai-compatible` (Chat Completions, the documented default) or `@ai-sdk/openai` for Responses; `limit.context` by hand | OpenAI function tools, ~11–13 built-ins; `tool_choice` passed through | always; `include_usage` forced on; 300 s header and chunk timeouts | `max_tokens`, optional `reasoning_effort`; never calls `/v1/models` |
| Hermes | `chat_completions` for `provider: custom`; `anthropic_messages` only if the URL ends `/anthropic`; Responses needs a named `custom_providers` entry | OpenAI function tools; core list of 52 (availability-gated) | prefers streaming, falls back; `include_usage` always | context from `/v1/models` `max_model_len` / `context_length` / `max_tokens`, else 128K; **refuses a window under 64K** |
| OpenClaw | `openai-completions` default when `baseUrl` is set; also `openai-responses`, `anthropic-messages`, `ollama` | OpenAI function tools; `tool_choice:"auto"` | always; usage chunk when `compat.supportsUsageInStreaming` | adapter derived from pi-ai; sends no `store` or cache hints to custom URLs |
| Codex | **Responses only**: `wire_api = "chat"` now fails at config parse (`codex-rs/model-provider-info/src/lib.rs:100-133`) | Responses tools of type `function`, `namespace`, `tool_search`, `web_search`, **`custom`** (freeform, `syntax:"lark"`, used for apply_patch); `tool_choice:"auto"`, `parallel_tool_calls` | required; reads `response.output_item.added`/`.done`, `response.output_text.delta`, `response.custom_tool_call_input.delta`, `response.completed` (usage); a stream closing before `response.completed` is an error | no `instructions`: the system prompt is a `role:"developer"` input item with an `id`; `store:false`, `include`, `prompt_cache_key`, `client_metadata`, `reasoning.summary` |
| Claude Code | Anthropic Messages, `POST /v1/messages?beta=true`; `count_tokens` optional; best-effort `HEAD /api/hello` | Anthropic `tools:[{name,description,input_schema}]` with `strict`/`defer_loading` | required; `ping` events feed its idle watchdog; must end `message_delta` + `message_stop` | `anthropic-version`, open-ended `anthropic-beta`; `thinking:{type:"adaptive"}`, `output_config`, `context_management`, `cache_control`, block-array `system`; a 400 is retried without `thinking`/effort but **not** without `context_management` or tool fields; unknown model ids assumed 200K context; `/v1/models` discovery opt-in and keeps only ids containing "claude"/"anthropic" |
| Oh My Pi | `api:` required per provider (`openai-completions`, `openai-responses`, `anthropic-messages`); implicit llama.cpp/Ollama providers default to `openai-responses`; `auth: none` | OpenAI function tools, 31 built-ins, `strict:true` when allowed and retried without on a 400 | always; `include_usage` | `/v1/models` discovery reads `max_model_len`/`context_length`, else 128K context / 32K output |
| Cline | "OpenAI Compatible" = `@ai-sdk/openai-compatible` (Chat Completions); Anthropic provider via `@ai-sdk/anthropic` (base URL must include `/v1`); its Ollama provider uses native `/api/chat` | AI-SDK function tools, no `toolChoice` | `streamText`; needs a final usage chunk | none found; model id typed in |

## Dialect count

**Three distinct dialects**, not two: OpenAI Chat Completions (Pi, OpenCode, Hermes, OpenClaw, Oh My Pi, Cline),
OpenAI Responses (Codex only, and optional in Pi, OpenCode, OpenClaw, Oh My Pi), and Anthropic Messages (Claude
Code; optional in Pi, OpenClaw, Oh My Pi, Cline, Hermes). Ollama native `/api/chat` (Cline's Ollama provider) is a
fourth wire that `serve` does not speak, but Cline offers "OpenAI Compatible" for the same job, so it is a doc
pointer, not a gap.

## Each claimed gap checked against serve

`serve` decodes request bodies with plain `json.Unmarshal` (no `DisallowUnknownFields` anywhere in
`internal/serveapp`), so an unknown request field is ignored rather than refused. That closes most "must accept"
items at once.

| claimed need | serve today (5802692a) | verdict |
|---|---|---|
| `stream_options.include_usage` usage chunk on Chat Completions | `internal/serveapp/helpers.go:357-375` emits it on the plain, tool and vision streams (M-26) | have |
| `max_completion_tokens` | `internal/serveapp/openai.go:594`, preferred over `max_tokens` | have |
| `developer` role | `internal/serveapp/openai.go:1399` aliases it to `system` (G12) | have |
| `store`, `reasoning_effort`, `prompt_cache_key`, `client_metadata`, `include`, `strict:false` | ignored as unknown fields | have (ignored) |
| Anthropic `thinking:{type:"adaptive"}` | `internal/serveapp/anthropic.go:34-59` handles `enabled`/`adaptive`/`disabled` | have |
| Anthropic `cache_control`, unknown block types | `internal/serveapp/anthropic.go:94-95`, `internal/serveapp/responses.go:328`: accepted and ignored | have |
| Anthropic `output_config`, `context_management`, tool `strict`/`defer_loading`, `?beta=true` | unknown fields ignored; the mux matches the path only | have (ignored) |
| Anthropic `ping` during a slow prefill | one `event: ping` at stream start (`internal/serveapp/anthropic_stream.go:65`); the 10 s heartbeat thereafter is an SSE **comment** (`: ping`, `internal/serveapp/helpers.go:328`), not an `event: ping` | **unverified**: whether Claude Code's idle watchdog counts SSE comments is not in its docs. Gate 1 or a trace settles it |
| Context-overflow wording Claude Code compacts on (`prompt is too long`) | the overflow text is "prompt is too large for the model's context window of N tokens (context_length_exceeded)" (`internal/serveapp/openai.go:214`) | **gap, small**: auto-compaction may not trigger on our wording |
| `/v1/models` `context_length` / `max_model_len` (Hermes, Oh My Pi) | publishes `context_window` only (`internal/serveapp/openai.go:564`) | **gap, small**: both fall back to 128K and will overrun a smaller served window; Hermes's 64K floor cannot see ours |
| **Responses `tools` in the Responses shape** `{type:"function", name, description, parameters}` (Codex; Pi/OpenCode/OpenClaw/Oh My Pi in Responses mode) | `/v1/responses` decodes `tools` into the Chat Completions shape `{type, function:{name,…}}` (`internal/serveapp/responses.go:35`, `toolSpec` at `internal/serveapp/openai.go:689`; `respondTools` reads `t.Function.Name` at `internal/serveapp/responses.go:358`). The only tests (`internal/serveapp/responses_test.go:137`) send the Chat shape | **gap, real**: a flat Responses-shaped function tool arrives with an empty name and no schema |
| Responses `custom` (lark freeform), `namespace`, `tool_search`, `web_search` tool types (Codex) | not recognised | **gap**: at minimum strip or refuse by name; `custom` is how Codex sends apply_patch |
| Responses `role:"developer"` input items carrying an `id` | items decode `type`/`role`/`content`; `id` ignored; the role passes to `messagesToTurns` | likely have; confirm in the Responses fix's tests |
| Responses SSE `response.output_item.done`, `response.completed` with usage, `response.custom_tool_call_input.delta` | `response.completed` with usage at `internal/serveapp/responses.go:236`, `internal/serveapp/responses.go:328`, `internal/serveapp/responses.go:437`, `internal/serveapp/responses.go:481` | `completed` have; `custom_tool_call_input.delta` absent (follows from the `custom` gap) |
| Ollama native `/api/chat` (Cline's Ollama provider) | not served | outside the surface; point Cline at "OpenAI Compatible" |

## Outcome under the pre-registration

The Gate 0 rule's first branch ("two dialects, nothing missing") does not hold: there are three dialects and
harnesses need things `serve` lacks. Per the second branch, each becomes its own scoped item with the field named:

- **H-a. Responses tool shape** (Codex, and every harness's Responses mode): accept the flat Responses function
  tool; strip or refuse non-function tool types by name; Codex's `custom` apply_patch tool is the one that matters.
  This is the only gap that stops a harness outright.
- **H-b. `/v1/models` context fields**: publish `context_length` and `max_model_len` beside `context_window`.
- **H-c. Anthropic overflow wording**: phrase the Messages-route overflow error so Claude Code's auto-compaction
  recognises it (`prompt is too long`).
- **H-d. Anthropic keep-alive form** (unverified): whether a comment heartbeat keeps Claude Code's idle watchdog
  alive during a long prefill. Settle with a trace before changing anything.

Gate 1 (the `/v1/messages` harness-scale row) is unaffected by these and proceeds.

**Status (same day, branch `libsurface-harness-2026-10`):** H-a fixed (both tool shapes and a flat named
`tool_choice` decode; unrunnable tool types are dropped and named in `X-Goinfer-Tools-Dropped`; `TestServe_responses`
step 4b end to end). H-b fixed (`context_length`, `max_model_len`). H-c open as a decision, not a defect: the
resident-cap overflow message was worded on purpose so a client does not compact against a server limit
(`contextLengthErrorFor`, opencode compacted 34 times), and Claude Code compacting on "prompt is too long" is that
same trade on the Messages route. H-d open pending a trace.
