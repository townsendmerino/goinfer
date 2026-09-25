# Task: goinfer as a minimax-code backend (M0–M7) — 2026-09

> **Status: SCOPED 2026-09-25, unstarted.** M0 is a measurement and gates the rest: every band below
> is a prediction from reading both codebases, and M0 is where it becomes a number. M1 is docs and can
> land with M0. M2–M4 are the fixes the reading says matter; M5–M7 are verify-then-decide.
>
> **What minimax-code is.** MiniMax's terminal coding agent (`github.com/MiniMax-AI/minimax-code`,
> MIT, read at `4198174`, 2026-09-25). Besides its own MiniMax account mode it has a bring-your-own-model
> mode (`mcode provider add --base-url … --api-format …`) speaking `openai-completions`,
> `openai-responses` or `anthropic-messages`, all three of which goinfer serves. Its LLM client is a
> vendored `pi-ai` (`third_party/pi-mono/packages/ai`). It is the same user as mode 3 in
> [`task-embed-and-harness-ux.md`](task-embed-and-harness-ux.md) — "point my tools at it" — on a harness
> this repo has not tested.
>
> **Siblings.** [`../integrations/opencode.md`](../integrations/opencode.md) and
> [`../integrations/claude-code.md`](../integrations/claude-code.md) (the two harness pages this one
> copies the shape of) · [`task-tool-grammar-union-2026-09.md`](task-tool-grammar-union-2026-09.md) (the
> many-tools grammar M0 relies on) · [`task-fit-to-hardware.md`](task-fit-to-hardware.md) (owns the
> context-default policy M2 amends) · [`task-memory-accounting-2026-09.md`](task-memory-accounting-2026-09.md)
> (the pricing M2 uses) · `docs/server.md` · `docs/api-tiers.md`.

---

## 0. What the reading established (2026-09-25, goinfer `a5d66ffd`)

**Works as-is, by the code on both sides** (not yet run together — M0):

| minimax-code sends (openai-completions, custom URL) | goinfer |
|---|---|
| system prompt as `role: "developer"` when its thinking toggle is on | treated as `system` (`internal/serveapp/openai.go:1129`) |
| `max_completion_tokens` (its default for an unrecognised URL) | honoured, preferred over `max_tokens`; clamped to the context, not refused; ceiling 131072 (`internal/serveapp/openai.go:40`) |
| `stream: true` + `stream_options.include_usage` | supported; final usage chunk |
| `store: false`, `prompt_cache_key`, tools' `strict: false`, `reasoning_effort` | unknown fields ignored (plain `json` decode) |
| tool calls read from `delta.tool_calls[i]` with `index` | emitted in that shape (`internal/serveapp/tools.go`) |
| `Authorization: Bearer <key>` (its `--api-key-env` is mandatory) | ignored when serve has no `--api-key` |
| overflow detection by error text (`utils/overflow.ts`, generic `/context[_ ]length[_ ]exceeded/i`) | the 400 reads "… context window of N tokens (context_length_exceeded)" (`internal/serveapp/openai.go:167`) — matches, so its compaction should fire |

**Will bite, in order of likelihood:**

1. **Context size.** minimax-code assumes a 200,000-token window for a custom model unless given
   `--context-limit`, and sends its system prompt (`packages/local-runtime-v2/assets/agents/_default/prompt-base-all.md`
   alone is 7.7 KB) plus the tool schemas on every turn — roughly 5–8k tokens before the first user
   message, by estimate from the files (M0 measures it). goinfer's resident defaults are 4096 on Metal
   (`metal/model.go:23`) and 4096→8192 fit on CUDA (`cuda/resident.go:48`). The first turn may not fit;
   the second almost certainly will not. It does not read goinfer's `/v1/models` `context_window`.
2. **Thinking models answer the connection test slowly.** `--use` sends a **non-streaming** `ping` with
   `max_tokens` = the configured output limit (default 16,384) and a **10 s** timeout
   (`packages/local-runtime-v2/src/service/model-system/connectivity/test-connection.ts`). goinfer has no
   way to turn a model's thinking off — nothing reads `enable_thinking`, `chat_template_kwargs` or
   `reasoning_effort` — so a Qwen3-class model thinks about "ping" until the timeout. The same absence
   costs every real turn.
3. **Thinking arrives as answer text.** goinfer never emits `reasoning_content` (the llama.cpp
   convention pi-ai reads, `providers/openai-completions.ts`, `reasoningFields`), so `<think>` text is
   shown as the reply and is resent as assistant history, spending context.
4. **Images with tools are refused.** minimax-code always sends tools; a pasted screenshot gets goinfer's
   deliberate 400 "tools are not supported together with image inputs" (`internal/serveapp/openai.go:656`,
   N-16 / R-08).
5. **The other two formats are unverified.** `openai-responses` sends item types the route declares out
   of scope ("reasoning items", `internal/serveapp/responses.go` header); `anthropic-messages` sends
   pi-ai's Anthropic shape (cache_control, thinking config, beta headers) — not traced.

---

## The items

| # | item | size | status |
|---|---|---|---|
| M0 | Run it for real — both boxes, a scripted task set, a pre-registered pass rule | S | gates M2–M7 |
| M1 | `docs/integrations/minimax-code.md` + a `serve check` row shaped like its first turn | S | with M0 |
| M2 | A context default an agent harness fits in — Metal fit-by-default, a harness warning | M | after M0 |
| M3 | A thinking switch: `enable_thinking`, `chat_template_kwargs`, `reasoning_effort`, `--thinking` | M | after M0 |
| M4 | Emit reasoning separately: `reasoning_content` deltas and Anthropic `thinking` blocks | M | after M3 |
| M5 | Images together with tools on the vision path | L | verify-then-decide |
| M6 | `/v1/responses` tolerates what pi-ai sends | S–M | verify-then-decide |
| M7 | `/v1/messages` tolerates what pi-ai's Anthropic client sends | S | verify-then-decide |

---

### M0 · Run it for real

**Goal.** Replace §0's reading with a measured compatibility record, and size M2–M4 from it.

**Setup.** Mac (Metal) and nobara (CUDA), each: `goinfer-serve --model coder=<qwen2.5-coder-7b-instruct
q4_k_m> --ctx 32768` (CUDA: the largest `--ctx` the card admits for 7B — the `--ctx` help records 20000
fitting on the 2070 SUPER), `mcode provider add --name goinfer --base-url http://127.0.0.1:8080/v1
--api-format openai-completions --model coder --api-key-env MCODE_PROVIDER_API_KEY --context-limit <ctx>
--output-limit 4096 --use`. A second model with thinking (a Qwen3-class checkpoint goinfer serves) for
the thinking cells. Pin the mcode version in the record (`mcode --version`).

**Task set.** Five scripted tasks in a scratch repo with a failing test each (read → edit → run tests),
the shape of minimax-code's own quick-start example, run with `mcode "<task>"` non-interactively; the
same five on both boxes.

**Record, per turn** (serve's request log plus `usage`): prompt tokens, `usage.prefill_reused_tokens`,
TTFT, decode tok/s, whether the reply was a tool call, parse failures, any 4xx/5xx with its body, and
every compaction minimax-code performs. Per task: pass/fail, turns, wall-clock.

**Pre-registered rule** (commit `docs/measurements/minimax-code-compat-PREREGISTERED.md` before the first
run): **compatible** = `--use` passes on the non-thinking model on both boxes, ≥4/5 tasks complete on at
least one box, zero goinfer 5xx, zero malformed tool calls. **Partially compatible** = connects and
completes ≥1 task. Anything else is a finding to fix before M1 ships. Also registered, so they can be
wrong in public: the first-turn prompt is 5–8k tokens; `--use` fails within 10 s on the thinking model;
at least one `context_length_exceeded` → compaction cycle occurs by task 5 at ctx 32768.

**Record.** `docs/measurements/minimax-code-compat-2026-MM-DD.md` + raw logs under `~/goinfer-logs/`;
`benchmarks.md` untouched (this is a compatibility record, not a speed row).

### M1 · The integration page and a `serve check` row

**Build.** `docs/integrations/minimax-code.md` in the opencode page's shape: the exact
`mcode provider add` line, why `--context-limit` must equal goinfer's `--ctx` (and what happens if it
does not), `--output-limit`, which API format to use (openai-completions until M6/M7 say otherwise),
the thinking caveat until M3 lands, the images-with-tools refusal, and M0's measured result — including
failures, at full value. Link it from the README's harness list and `docs/server.md`.

`internal/servecheck/check.go` already has a harness-scale tools row built from opencode's agent; add a
"minimax-code first turn" row: a system prompt and tool set of the size M0 measured, sent against the
served model's context, reporting `ok` / `won't fit at ctx N` before the user configures anything.

**Gate.** The page's commands are copied from a run, not written from memory; the new check row
reproduces M0's fits/doesn't-fit answer at two `--ctx` values.

### M2 · A context default an agent harness fits in

**Goal.** A user who starts `goinfer-serve` without `--ctx` and points an agent at it gets a context the
agent's first turns fit in, or a message saying so at startup — never a mid-session wall.

**Read first.** `task-fit-to-hardware.md` Phase 2 (CUDA's fit-by-default context: up to 8192 when the
card affords it, and the 2026-09-15/16 regression it caused on `-moe-cache-experts` loads — a larger
default ate the expert cache's VRAM; the fix keeps 4096 there), `metal/model.go` `resolveMetalCtxCap`,
the S4 live memory ceiling (`task-never-swap-2026-09.md`), `task-memory-accounting-2026-09.md` (the one
KV formula to price with), `docs/tasks/task-embed-and-harness-ux.md` mode 3.

**Build.**
1. **Metal fit-by-default context**, the CUDA Phase-2 shape: an unpinned Metal load grows its context
   from 4096 toward `metalCtxCapMax` (32768) while `Plan("metal")`'s need stays under the live ceiling
   with the documented margin; never on a paged-MoE load (the same companion-allocation lesson). The
   banner names the resolved number and why.
2. **A harness floor warning** at startup when the resolved context is below 16384 tokens (the M0
   number decides the exact floor): one line naming the harnesses that will not fit and the `--ctx` that
   would, priced by the same function.
3. **The 400 names both numbers** — "prompt is N tokens; context window is M" — so a harness's log says
   what to set. Keep `context_length_exceeded` in the text (it is what pi-ai matches).

**Decision rule.** Ships when M0's task set, re-run with no `--ctx`, reaches the same completion count as
M0's explicit `--ctx` run on the Mac, and the dense decode rate at depth 128 is within 3% of today's
(`TestProdThroughput`-class; a larger KV must not slow short turns).

### M3 · A thinking switch

**Goal.** A client can turn a family's thinking off, and a server operator can set the default.

**Read first.** `chat/templates.go` (how each family's template renders; Qwen3's template takes an
`enable_thinking` variable, gpt-oss's harmony format takes a reasoning level), the tool-call union's
template census (`task-tool-grammar-union-2026-09.md` T4 — which families render what), and what pi-ai
sends per `thinkingFormat` (`providers/openai-completions.ts`: `enable_thinking` for `qwen`,
`chat_template_kwargs.enable_thinking` for `qwen-chat-template`, `reasoning_effort` for `openai`).

**Build.** Accept, in priority order, `chat_template_kwargs.enable_thinking`, top-level `enable_thinking`,
then `reasoning_effort` (`none`/`minimal` → off; `low`/`medium`/`high` → on, and mapped to gpt-oss's
reasoning level). A server flag `--thinking auto|on|off` sets the default when the request says nothing
(`auto` = the family's own default, today's behaviour). Families with no switch ignore it and say so in
`/v1/models` (`"thinking_control": false`). Same fields on `/v1/responses` (`reasoning.effort`) and
`/v1/messages` (`thinking: {type: "disabled"}`).

**Gate.** Per family with a switch: the rendered prompt with thinking off is byte-identical to the HF
template rendered with `enable_thinking=False` (the template-parity tests' method); a greedy reply to
"ping" on a Qwen3-class model contains no think block. M0's thinking-model `--use` cell passes with
`--thinking off`.

### M4 · Emit reasoning separately

**Goal.** Thinking text leaves the answer channel: minimax-code (and every pi-ai / llama.cpp-convention
client) shows it as reasoning and does not resend it as history.

**Build.** Parse the family's think delimiters at the stream boundary (the same holdback machinery that
already keeps stop strings and partial UTF-8 out of chunks) and emit `delta.reasoning_content` on
`/v1/chat/completions`, `thinking` content blocks on `/v1/messages`, reasoning items on `/v1/responses`;
non-streaming responses carry `message.reasoning_content`. On input, drop `reasoning_content` from resent
assistant messages before templating (the templates that re-render thinking decide that themselves).
`--reasoning-format none` keeps today's inline behaviour.

**Gate.** Stream reassembly tests: the concatenation of reasoning + content deltas equals today's single
content stream byte for byte, across every chunk boundary inside a delimiter; a tool call emitted after
a think block still streams as `delta.tool_calls`. `usage.completion_tokens` unchanged (reasoning counts,
as OpenAI counts it).

### M5 · Images together with tools

**Status: verify-then-decide.** The refusal is deliberate: the vision path renders no tools, so a
tools+image request would silently drop the tools. Lifting it needs the VL templates (Gemma 3, Qwen2.5-VL,
Qwen3-VL) to render tools and the vision drive path to run the tool turn. Decide after M0 whether
minimax-code users actually paste images; if the answer is rarely, keep the refusal and document it in M1.

### M6 · `/v1/responses` tolerates what pi-ai sends

**Build (verify first).** Capture one real mcode turn with `--api-format openai-responses` against
goinfer (M0's harness, request logging on); list every input item type and top-level field; make the
route accept them — `reasoning` items skipped on input, `include`/`store`/`prompt_cache_key` ignored,
`function_call`/`function_call_output` round-trip (already implemented) — and record which were already
fine. **Decision:** if the captured shape needs more than skip-and-ignore, say so in M1 and recommend
openai-completions instead.

### M7 · `/v1/messages` tolerates what pi-ai's Anthropic client sends

**Build (verify first).** Same capture with `--api-format anthropic-messages` (note minimax-code rewrites
the base URL for this format — `normalizeMessagesBaseUrlForPi`; confirm the path it actually posts to).
Expected: `cache_control` markers, `anthropic-version`/beta headers, `thinking` config, tool definitions
with `input_schema`. Ignore what goinfer cannot use; M3 maps `thinking`. Record the result in M1.

---

## Sequencing

M0 and M1 together (a day; the page ships with the measurement in it). Then M3 (it fixes both the
connection test and every thinking-model turn), M2, M4. M6 and M7 are an hour each once M0's capture
harness exists; M5 waits on M0's evidence.

## What this doc does not claim

- That goinfer has been run with minimax-code. §0 is both codebases read side by side; M0 is the run.
- Anything about agent quality. A 7B coder model's decisions inside a harness with eight-plus tools
  are the model's, not the server's; M0 records completion rates so the page can say what to expect.
- That minimax-code should change. Reading `/v1/models`' `context_window` would remove item 1 of §0 on
  the client side; that is theirs to decide, and M2 does not depend on it.

<!-- doc-reviewed: 2026-09-25 -->
