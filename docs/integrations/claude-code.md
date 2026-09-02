# Claude Code → goinfer

Claude Code speaks the Anthropic Messages API, so it points at `/v1/messages` with three env
vars. Verified end to end **2026-09-02** on the numbers below.

```bash
goinfer-serve -model coder=~/models/qwen2.5-7b-instruct-q4_k_m.gguf \
  -quant int4 -backend cuda -ctx 16384

ANTHROPIC_BASE_URL=http://127.0.0.1:8080 \
ANTHROPIC_AUTH_TOKEN=goinfer \
ANTHROPIC_MODEL=coder claude
```

All three env vars are required. Non-loopback binds refuse to start without `-api-key`.

## What passed, and the numbers

**Model class:** Qwen2.5-**7B**-Instruct q4 → int4, CUDA-resident. A model that can hold an
agent loop is the requirement, not just a context window: it must emit a tool call *and then
answer from the result*. A 1.5B re-calls the same tool forever, which looks like a server bug
and is not one.

| | measured |
|---|---|
| tool loop (`glob` → `read` → answer) | **3 turns, 2.83 s**, correct answer, ends `stop_reason: end_turn` |
| streamed tool call event order | `message_start → ping → content_block_start → content_block_delta → content_block_stop → message_delta → message_stop` (no `[DONE]`) |
| `usage` on a **streamed tool call** | present (this was M-26; it used to appear on the plain text stream only) |
| **TTFT, realistic agent turn** | **8.85 s** for a 2,293-token turn (25 tool schemas + a 3.5 KB system prompt) ≈ **259 tok/s prefill** |
| TTFT, small turn (2 schemas) | 0.89 s |

Provenance: RTX 2070 SUPER, NVIDIA driver 595.91.07, `-quant int4 -backend cuda`, greedy,
warm, `/v1/messages` non-streaming for the loop and streaming for the event/usage rows;
turn size from this server's own `/v1/messages/count_tokens`.

## What to expect, and what will bite

**Every turn re-prefills its whole prompt.** A CUDA/WebGPU-resident model decodes statelessly —
the resident KV lives on the GPU while a session's prefix cache is CPU-side, and they cannot
both be the source of truth — so prefix reuse is off, whatever `--kv-sessions` says. The
startup banner states this. At ~259 tok/s prefill that is the 8.85 s above **on every turn**,
and it grows with your conversation. It is the dominant cost of an agent loop here.

**Known-open, so do not debug them as your setup:**
- `tool_choice: "any"` (or OpenAI `"required"`) with **two or more** tools does not force a
  call — the model may answer in prose. A *named* tool (`{"type":"tool","name":…}`) and the
  single-tool case both work, because those are unambiguous and take the constrained path.
  Audit N-18.
- Gemma-4's tool rendering disagrees with its own template after the first turn (M-20). Use a
  ChatML-template model (Qwen2.5 above) for tool work.

`thinking`, `cache_control` and `metadata` are accepted and ignored.

## Retiring this page

Per `docs/task-embed-and-harness-ux.md` §3.5, a recipe is retired when `serve check` covers
what it says. `goinfer-serve check <url>` already covers the model list, streamed chat with
usage, structured output, stop sequences and `count_tokens`; the tool-loop and agent-turn-TTFT
rows above are what it does not cover yet.
