# Claude Code against goinfer, traced 2026-10-12 (I4's first cell, and a gap it found)

`docs/tasks/task-integration-harness-2026-10.md` I4, for Claude Code. M1 Pro (16 GB), Claude Code **2.1.268**, run
headless from an empty directory with one prompt ("Reply with the single word ok.", `--max-turns 1`). Two
instruments, both in this directory:

- `capture.py`: a local recorder that stores each request and answers 400, so the request's shape is read without a
  model (`capture_summary.py` prints the shapes). The bodies are not committed: they carry Claude Code's own prompt.
- `cc_census.py`: `goinfer-serve` (this branch, `--backend metal`, Qwen2.5-Coder-1.5B at int4, `-ctx 32768`,
  `-log-requests`) and the same headless prompt; token counts are the server's own `prompt_tokens`.

Token counts are under the Qwen2.5 tokenizer and move by a few percent on another family. Nothing here is a timing.

## What Claude Code sends

| | bare (`claude --bare -p`) | full (`claude -p`, a fresh empty `HOME`) |
|---|---|---|
| requests for one prompt | 2: a session-title request, then the turn | the same 2 |
| title request | no tools, a 3.3 KB system prompt, **761 tokens**; asks for JSON through `output_config.format` (a `json_schema`) | the same |
| the turn: tools | 3 (`Bash`, `Edit`, `Read`), 3.2 KB of schema | **24 with an API key, 22 with a bearer token**, 66 KB and 57 KB of schema |
| the turn: prompt | **1,063 tokens** | **19,692 tokens** (24 tools) |
| `max_tokens` | 32,000 | 32,000 |
| other fields | `thinking: {type: adaptive, display: omitted}`, `output_config: {effort}`, `metadata` | those and `context_management` |
| message roles | `user`, then **`system`** | `user`, then **`system`** |

- **Every turn carries a `system` message inside `messages`** (the date, as text blocks), under the beta header
  `mid-conversation-system-2026-04-07`. `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` does not remove it.
- **Which model the title request names:** the main model (`ANTHROPIC_MODEL`) with either credential variable; with
  `ANTHROPIC_DEFAULT_HAIKU_MODEL` set, that model. So background work does not name an unserved model by default, and
  the variable is how to send it to a second, smaller served model.
- `--bare` reads `ANTHROPIC_API_KEY` (or an `apiKeyHelper`); the full client took `ANTHROPIC_AUTH_TOKEN` or
  `ANTHROPIC_API_KEY`.

## What goinfer did with it

- **Before this branch: a 400 on every turn.** `message 1: role must be "user" or "assistant", got "system"`. Claude
  Code 2.1.268 could not complete one turn against goinfer. The check was deliberate (a wrong-API role must not be
  folded into the conversation) and predates the mid-conversation system message.
- **With the fix** (`internal/serveapp/anthropic.go:anthropicTurns`: a `system` message after the first is kept in
  place as user-side text): both requests answer 200. Bare: the reply is `ok`. Full: the 1.5B wrote a tool call as
  text in a code fence, which is the known small-model result and not a route defect.
- **The full first request does not fit 16,384 positions.** 19,692 tokens is over the window the recipes name, over
  CUDA's fit-by-default candidate and over Metal's. It needs `-ctx 32768` (24,615 with I4's 25% margin), or `--bare`.
- **Not honoured:** `output_config.format` on `/v1/messages`, so the title comes back as free text and not the JSON
  asked for. `context_management` is ignored. Neither stops a turn.

## What was not read

No session longer than one turn, so nothing about compaction, overflow recovery or the idle watchdog. No second
session for the prefix-reuse comparison I4 asks for. One Claude Code version, one model family, the Mac only.
