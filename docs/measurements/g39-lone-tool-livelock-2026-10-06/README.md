# G39: why `serve check` failed `tools, OpenAI` on Qwen2.5-Coder-7B (2026-10-06): goinfer, not the model

The cold-user run (`docs/measurements/cold-user-2026-10-05-nobara-pc.md`, B) saw `serve check` fail 2 of 9 rows on Qwen2.5-Coder-7B (`tools, OpenAI` turn two and `stop sequences`) while the opencode recipe said
Qwen2.5-7B-Instruct passed everything. G39 asked which part was the model, the template or goinfer. Exploratory, nobara-pc, RTX 2070 SUPER, `serve` built from the working tree (`cuda/cmd/serve`, `-tags cuda`),
`-backend cuda -ctx 16384`, int4, temperature 0, both q4_k_m GGUFs. The check outputs before and after are in this directory.

## What was ruled out, in order

1. **Not the model's template.** goinfer renders a tool conversation with its own chatml renderer, and the dump for the `serve check` turn-two conversation is byte-identical for the two models. (The two GGUFs' own templates
   differ in one place: Coder's tool instruction has doubled braces, `{{"name": …}}`. It does not matter here, because goinfer does not use the model's template for tools.) The official-template renderings differ from goinfer's
   in small ways (a default `You are Qwen…` system line, JSON key order, no blank line before `<tool_call>`), the same for both models.
2. **Not the prompt.** Sending turn two to the Coder model as a RAW completion, with each of six prompts (goinfer's rendering, Coder's own template, that template with the braces fixed, Instruct's template, goinfer's plus the default
   system line, goinfer's without the blank line), gave the same correct answer every time: "The current weather in Paris is 14 degrees Celsius with rain." The same conversation through `/v1/chat/completions` called the tool again.
3. **It is the chat route.** With tools in the request and the default `tool_choice`, turn two called `get_weather("Paris")` again. With `tool_choice: "none"`, or with no tools in the request, it answered. **With a SECOND tool in
   the request it answered too**, on the Instruct model as well as the Coder.
4. **The Instruct model fails the same row today.** `serve check` against Qwen2.5-7B-Instruct on this tree: `tools, OpenAI` FAIL, the identical message (`check-instruct-before.txt`). The recipe's "all 8 checks passed" dates from
   2026-09-08, and that pass never tested this: the row only began requiring a real answer to the tool result on 2026-09-11 (e458c715, G-12), while the forcing it exposes is older than 2026-08-07 (418362db moved it). So this is an old
   bug that the check newly reveals, not a recent regression.

## Cause

`forcedTool` (`internal/serveapp/tools.go`) returned the request's only tool as FORCED whenever exactly one tool was supplied and `tool_choice` was `auto` or absent, so the call was constrained to that tool's grammar from the first token of
EVERY turn, including the one after the tool's own result. A client with a single tool could never get an answer: the check's name for it, the agent-livelock shape, is exactly this. The comment on the function said the model was "still
free to answer in prose"; it was not. (The Anthropic route was already right: `anthropicForcedTool` never forces under `auto`.)

## Fix

The lone-tool convenience now ends at the tool result: under `auto` (or no `tool_choice`) it is lifted when the conversation already ends in a tool result (`endsWithToolResult`). The first call of a lone-tool request is still
constrained, an explicit `required` or a named function is still forced on every turn, and `none` is never forced. The cases were committed before the code (c69a6f87) and go red when the new condition is removed.

## After

| | Qwen2.5-7B-Instruct | Qwen2.5-Coder-7B |
|---|---|---|
| `tools, OpenAI` | FAIL → **ok** (`call get_weather → result → answer in 2 turns`) | FAIL → **ok** |
| `tools, harness-scale` | ok | skip (it writes the call as a fenced block; `-lenient-tool-calls` turns that into ok) |
| `stop sequences` | ok | **FAIL, unchanged** |
| `serve check` | 9 of 9 (exit 0) | 1 of 9 failed |

The one remaining Coder failure is the model's: asked `Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.` with stop `5`, it replies `The count is now 10.` and never reaches a `4`, which the row reports as "proves nothing about it". The Instruct
model counts. Nothing in goinfer is involved.

## What this does not settle

- It is one server start per model and one sampling setting (temperature 0). The before/after on `tools, OpenAI` is a deterministic row, not a rate.
- Whether the fix helps agents with a single tool in practice was not measured; the opencode runs of the same day used its dozen tools, where this path never applied.
