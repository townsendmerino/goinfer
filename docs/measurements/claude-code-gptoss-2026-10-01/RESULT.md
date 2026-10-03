# A real Claude Code run against gpt-oss-20b on the Anthropic route (2026-10-01)

**Exploratory functional check, not a measurement.** One run, sampled (Claude Code's own settings), no repeat. It answers one question that was open — "the Anthropic
route is unit-tested but not run live against gpt-oss" — and says nothing about speed.

| | |
|---|---|
| client | Claude Code 2.1.286 (the VS Code extension's native binary), `-p`, `--bare`, `--tools Read`, `--max-turns 6`, isolated `HOME`, dummy `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL=http://127.0.0.1:8100` (loopback only), the model name and the three `ANTHROPIC_DEFAULT_*_MODEL` / `ANTHROPIC_SMALL_FAST_MODEL` set to `gptoss`; `run.sh` |
| server | goinfer `37e72a71` (tree clean), `cuda/cmd/serve` built `-tags cuda`; `-model ~/models/gpt-oss-20b-MXFP4.gguf -backend cuda -moe-cache-experts -ctx 32768 -served-model-name gptoss -log-requests`; decode path `cuda-resident (int4mix)` |
| machine | nobara-pc, RTX 2070 SUPER 8 GB, NVIDIA driver 595.91.07; the model read from `~/models` on NVMe |
| task | `notes.txt` holds "The codeword is: heliotrope-42"; prompt: "Use the Read tool to read notes.txt, then tell me the codeword it contains." |

**Outcome: it worked.** `result: success`, `is_error: false`, 3 turns, 32.2 s, final answer "The `notes.txt` file contains the string **`heliotrope-42`**." The server log shows three
`POST /v1/messages ... status=200` lines (318, 600 and 706 prompt tokens; 246, 65 and 22 completion tokens; no panic or error). The loop was a real Anthropic-dialect tool loop:
`thinking` block → `tool_use` → `tool_result` → `thinking` → `tool_use` → `tool_result` → `text`, with the thinking blocks (empty signatures, as serve emits them) accepted back by Claude Code.

**Found: the first call named the tool `read`, not `Read`.** Claude Code answered `Error: No such tool available: read. Tool names are case-sensitive: call Read instead.`; the model read that,
reasoned "functions.Read", and called `Read`. goinfer did not change the case: nothing on the Harmony path lowercases a name (`chat/harmony_tools.go` takes the recipient after `functions.` as it is); the
model wrote `to=functions.read` itself, against a declaration `type Read = ...`. A harness that does not return a helpful error for an unknown tool would have stopped there. The call's form is parsed,
not constrained (`docs/server.md`, gpt-oss tool calls), so this is the expected failure class; a unique case-insensitive match to a declared tool could be corrected by the parser. **Not done** — it changes
what the model's output means, so it is a decision, not a fix.

**Not covered.** More than one tool (the run offered only `Read`); Claude Code's default tool set and its ~15–20k-token first request (`--bare` made this one 318 tokens; at the 29 tok/s prefill below
that default request is minutes, unmeasured); `-moe-cache-experts`' prefill rate is **29 tok/s** (a 3,068-token prompt took 105 s through `/v1/messages`, calibration before the run), decode about 15 tok/s —
the CPU path was tried first and abandoned (a 3k-token prompt had not finished prefilling after 10 minutes); streaming was not independently inspected (the event stream Claude Code printed is its own
rendering); a second run to see how often the first call is mis-cased.

Raw: `claude-events.jsonl` (the event stream with the 256 per-token `thinking_tokens` progress events removed), `serve.log`, `run.sh`.
