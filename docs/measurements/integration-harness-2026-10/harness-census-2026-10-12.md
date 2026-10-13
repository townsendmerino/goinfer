# First-request census: Claude Code, opencode, Codex (I4), 2026-10-12

`docs/tasks/task-integration-harness-2026-10.md` I4, one harness per dialect. nobara-pc; the harnesses installed from
npm at pinned versions into one directory (not globally). Two phases, scripts in this directory:

1. **Capture** (`harness_capture.py`, `capture.py`): each harness runs once, headless, from an empty directory with a
   fresh `HOME`, against a recorder that stores the request and answers 400. No model answers, so no harness runs a
   tool. `capture_summary.py` prints the shapes; the bodies are not committed (they hold each harness's own prompt).
2. **Replay** (`harness_replay.py`): each recorded body is posted to goinfer (serve at `1a6021cf`, CUDA,
   Qwen2.5-Coder-1.5B at int4, `-ctx 32768`) with streaming off and the output capped at one token. The size is
   goinfer's own `prompt_tokens`. Log: `replay-nobara-2026-10-12.log`.

Sizes are under the Qwen2.5 tokenizer and move by a few percent on another family. Nothing here is a timing.

| harness, version | route | first request | tools sent | fits 16,384 with I4's 25% margin? |
|---|---|---|---|---|
| Claude Code 2.1.296 | `/v1/messages` | **20,199 tokens** | 23 (67 KB of schema) | **no** (25,249 needed) |
| Claude Code 2.1.268 (the Mac, `claude-code-census-2026-10-12.md`) | `/v1/messages` | 19,692 | 24 | no |
| Claude Code, `--bare` | `/v1/messages` | 1,063 | 3 | yes |
| opencode 1.19.0 | `/v1/chat/completions` | **7,165 tokens** | 10 (21 KB) | yes (8,956) |
| Codex 0.162.1 | `/v1/responses` | **5,850 tokens** | 9, of which goinfer drops 2 | yes (7,313) |

- **opencode also sends a title request first** (540 tokens, no tools). Its main request has the system prompt as a
  `system` message and `max_tokens: 4096`, the output limit this run's config gave it.
  - R19 measured 11,137 tokens on 2026-10-01 with the opencode of that day. This is a different version; the two
    are not one measurement.
- **Codex sends no `instructions` field.** Its input is two `developer` messages (17.0 KB of agent instructions,
  then 2.3 KB about skills) and two `user` messages (an environment block, then the prompt).
  - **goinfer kept only the last `developer` message**, so the instructions were dropped and the request counted
    **2,297 tokens**. Fixed on this branch (`internal/serveapp/openai.go:messagesToTurns`); 5,850 is with the fix.
  - Its tools: `exec_command`, `write_stdin`, `request_user_input`, `view_image`, `get_goal`, `create_goal`,
    `update_goal` (functions), a `namespace` tool `multi_agent_v1` and `web_search`. goinfer drops the last two and
    names them in `X-Goinfer-Tools-Dropped`. **There is no `apply_patch` and no `custom` tool** in this version's
    request for an unknown model, so I6's premise does not hold today.
  - It sends no output cap, `reasoning: {summary: auto}` and `include: [reasoning.encrypted_content]`.
- **Claude Code** is as the Mac trace found: a `system` message inside `messages`, `max_tokens: 32000`.

## What was not read

One request a harness and no reply, so nothing about a second turn, compaction, overflow, or what each harness does
with goinfer's stream. That is I3. The other harnesses in the task doc's table are not installed.
