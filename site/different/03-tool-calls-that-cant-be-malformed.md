---
title: "Tool calls that can't come out malformed"
area: "Agents"
order: 3
summary: "When a model starts a tool call, goinfer holds it to valid JSON, a tool you supplied, and that tool's argument schema. It does not pick the tool for the model."
stand: "goinfer builds one grammar over every tool a request supplies and applies it while the model writes a call. The call's form is guaranteed, and the choice of tool is still the model's."
measured: 2026-09-24
reviewed: 2026-09-29
facts:
  - {label: "guaranteed", value: "valid JSON, a supplied tool name, arguments that fit that tool's schema"}
  - {label: "Qwen2.5-7B, 12 tools, auto", value: "14/111 unusable calls became 0"}
  - {label: "a prose answer", value: "decodes as it would with no constraint"}
  - {label: "not covered", value: "Gemma 4 and other families: parsed only"}
doesnt:
  - title: "It doesn't choose the right tool."
    text: "The grammar fixes the form of a call, not the model's judgement. Qwen2.5-Coder 0.5B and 1.5B write their calls as bare JSON, which goinfer now accepts when the name is a supplied tool. Those recovered calls were on the tool the transcript intended 9–11% of the time, against 27% for the 7B's parsed calls. A well-formed call to the wrong tool is still possible."
  - title: "It doesn't cover every model family."
    text: "goinfer reads the tool-call format from the checkpoint's own chat template. Calls are constrained for the chatml and mellum2 formats (Qwen, Nemotron-3-Nano, Mellum2, Granite 4.2), for mistral v0.3, and for llama3. Gemma 4's own call syntax is parsed only: goinfer reads the call out of the reply but does not constrain it. Gemma 3, gpt-oss and ministral have no tool support here, and several other families have no recognised template. No mistral v0.3 checkpoint was on the test machine, so mistral v0.3 support is by design and has not been confirmed by a run."
  - title: "A tool schema it can't compile turns the constraint off."
    text: "The grammar supports a subset of JSON Schema: type, enum, const, properties, required, additionalProperties false, items, minItems, maxItems and minimum 0. A keyword outside it (pattern, oneOf, $ref, maxLength), a freeform object with no declared properties, or an object with more than 64 properties makes the whole tool list uncompilable. The request then decodes with no constraint and no error, except that a named tool_choice gets a 400."
  - title: "Under auto it only starts at the call's opener."
    text: "The model can still answer in prose. Qwen2.5-Coder models practically never write the `<tool_call>` opener, so the constraint never switches on for them. They rely instead on a parser that accepts a bare call naming a supplied tool. A server started with speculative decoding leaves auto unconstrained. A llama3 call that follows prose in the same reply is only parsed, not constrained."
  - title: "The measurement is narrow."
    text: "One agent transcript, one tool schema, CUDA only (one RTX 2070 SUPER), no mixture-of-experts model. A grammar cannot prevent a length stop, so a call cut off by max_tokens is still truncated."
figures:
  - {text: "14/111", source: "docs/measurements/tool-union-2026-09-24.md"}
  - {text: "12.6%", source: "docs/measurements/tool-call-failure-t0-2026-09-23.md"}
  - {text: "24 of 111", source: "docs/measurements/tool-call-failure-t0-2026-09-23.md"}
  - {text: "595.91.07", source: "docs/measurements/tool-union-2026-09-24.md"}
  - {text: "12/185", source: "docs/measurements/tool-union-2026-09-24.md"}
  - {text: "4/185", source: "docs/measurements/tool-union-2026-09-24.md"}
  - {text: "188", source: "docs/measurements/tool-union-2026-09-24.md"}
  - {text: "310/310", source: "docs/measurements/tool-union-2026-09-24.md"}
  - {text: "630/630", source: "CHANGELOG.md"}
  - {text: "0.999", source: "docs/measurements/tool-union-2026-09-24.md"}
  - {text: "1.001", source: "docs/measurements/tool-union-2026-09-24.md"}
  - {text: "0.807", source: "docs/measurements/tool-union-2026-09-24.md"}
  - {text: "236", source: "docs/measurements/tool-call-failure-t0-2026-09-23.md"}
  - {text: "219", source: "docs/measurements/tool-call-failure-t0-2026-09-23.md"}
  - {text: "1,200", source: "docs/measurements/tool-call-failure-t0-2026-09-23.md"}
  - {text: "9–11%", source: "docs/measurements/tool-call-failure-t0-2026-09-23.md"}
  - {text: "27%", source: "docs/measurements/tool-call-failure-t0-2026-09-23.md"}
sources:
  - "docs/server.md"
  - "docs/tool-call-coverage.md"
  - "docs/measurements/tool-union-2026-09-24.md"
  - "docs/measurements/tool-call-failure-t0-2026-09-23.md"
  - "docs/tasks/task-tool-grammar-union-2026-09.md"
  - "docs/integrations/claude-code.md"
  - "docs/integrations/opencode.md"
  - "docs/env-vars.md"
  - "CHANGELOG.md"
  - "constrain/tools_union.go"
  - "constrain/lazy.go"
  - "constrain/schema.go"
  - "internal/serveapp/tools.go"
---

## The problem

An agent asks a model to call a tool, and then a program has to act on the reply. The program expects JSON with a tool name it knows and arguments of the right shape. A language model writes text one token (a word or a piece of one) at a time, and it can get any of that wrong: a broken bracket, a tool that does not exist, a required argument left out.

The failure was measured before it was fixed. On 2026-09-23, on a Linux PC (Ryzen 7 3700X, RTX 2070 SUPER 8 GB, NVIDIA driver 595.91.07), Qwen2.5-7B-Instruct was given a 12-tool schema like the ones real agents send, with `tool_choice: auto`, which lets the model decide whether to call a tool. Of the calls it attempted, 14/111 were unusable (12.6% in the record). Most named a tool that does not exist, mainly `git_commit`. Another 10 parsed but had a missing or wrong-typed argument: 24 of 111 in all.

## What goinfer does

When a request carries tools, goinfer builds a grammar from all of them: a set of rules for which text is allowed, applied while the model writes rather than checked afterwards. While the model writes a call, it can only produce a call that is valid JSON, names one of the supplied tools, and has arguments that fit that tool's schema. The same approach holds a model to a Go struct's shape in [A Go struct the model can't break](/different/04-a-go-struct-the-model-cant-break/).

Start a server (here the 7B from the measurement, on an NVIDIA GPU), then send an ordinary OpenAI-style request with tools:

```sh
goinfer-serve -model coder=~/models/qwen2.5-7b-instruct-q4_k_m.gguf -quant int4 -backend cuda -ctx 16384

curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "coder", "tool_choice": "required",
  "messages": [{"role": "user", "content": "What is in notes.txt?"}],
  "tools": [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file",
      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "list_files", "description": "List a directory",
      "parameters": {"type": "object", "properties": {"dir": {"type": "string"}}, "required": ["dir"]}}}
  ]}'
```

The reply carries the call in the usual shape. These values show the shape; they are not from a recorded run: `message.content` is null, `message.tool_calls` holds `{"id": "call_…", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"notes.txt\"}"}}`, and `finish_reason` is `tool_calls`. The name will be `read_file` or `list_files`, and never `git_commit`. The `/v1/messages` (Anthropic) and `/v1/responses` routes use the same constraint.

## How it works

At each step the model gives a score to every token in its vocabulary, and the next token is picked from those scores. goinfer first checks every token against the grammar and sets the score of any token that would break it to minus infinity, so it cannot be picked.

The grammar for several tools combines the one-tool grammars. They all run side by side, and only the matching one continues once the tool name is written. Tests check that the combined grammar accepts exactly what the one-tool grammars accept between them: for example, that `read_fil` is not a name, and that one tool's arguments are rejected under another tool's name.

What starts the constraint depends on `tool_choice`. A named tool, or `required` (Anthropic `any`), is constrained from the first token. Under `auto`, nothing is restricted until the model writes its own call opener: the marker its chat format puts before a call, `<tool_call>` for Qwen or `[TOOL_CALLS]` for mistral. After that opener the call is constrained, and each further call in the same turn is constrained on its own. llama3 has no opener, so its constraint switches on when the reply begins `{"name": "`.

A prose answer under `auto` is left alone. On the same transcript, every output of Qwen2.5-Coder 1.5B and of Llama-3.2 1B was byte-identical with and without the constraint: 310/310 for the 1.5B and 310/310 for llama3-1B, and 630/630 in the changelog's total, which adds the 7B's ten greedy turns. Decode speed (how fast the reply's tokens come out) on a prose turn was 0.999 to 1.001 of the unconstrained speed. A first build turned off the GPU's fast token-picking for the whole turn, prose included. It ran at 0.807 of unconstrained speed on the 1.5B at temperature 0.7, and was not shipped default-on.

## What was measured

All rows: the Linux PC (RTX 2070 SUPER, driver 595.91.07), CUDA, checkpoints in the q4_k_m format (weights quantized to about 4 bits), and the same 12-tool, ten-turn coding-agent transcript. At each turn the conversation so far was sent once greedy (always taking the most likely token) and 30 times sampled at temperature 0.7 with fixed seeds: 10 greedy plus 300 sampled requests per model. The history each request carries is fixed, so one bad reply cannot spoil the later turns.

| Date | Model | Result | Record |
|---|---|---|---|
| 2026-09-23 to 2026-09-24 | Qwen2.5 7B | unusable calls 14/111 without the constraint; 0 with it (0 unknown names, 0 unparsed, 0 invalid arguments; one call cut off by `max_tokens` is reported apart) | [tool-union-2026-09-24](https://github.com/townsendmerino/goinfer/blob/main/docs/measurements/tool-union-2026-09-24.md) |
| 2026-09-24 | Llama-3.2 1B | under `auto`, unusable calls 12/185 to 4/185; among 188 calls where the constraint switched on, 0 invented names, broken bodies or invalid arguments | [tool-union-2026-09-24](https://github.com/townsendmerino/goinfer/blob/main/docs/measurements/tool-union-2026-09-24.md) |
| 2026-09-24 | Qwen2.5-Coder 0.5B and 1.5B | the constraint never switches on (they never write the opener); a parser fix takes parsed calls from 0 to 236 and 219 of 300 | [tool-call-failure-t0-2026-09-23](https://github.com/townsendmerino/goinfer/blob/main/docs/measurements/tool-call-failure-t0-2026-09-23.md) |

The tests that assert well-formedness are in the `constrain` package. `TestToolGrammar_property` ([tool_grammar_test.go](https://github.com/townsendmerino/goinfer/blob/main/constrain/tool_grammar_test.go)) generates 100 seeded calls per family and checks wrapper, name and arguments. `TestToolCallsGrammar_acceptsExactlyTheUnionOfSingleToolGrammars` ([tools_union_test.go](https://github.com/townsendmerino/goinfer/blob/main/constrain/tools_union_test.go)) checks the multi-tool grammar case by case.

The Qwen2.5-Coder row is a different fix. Those models write the call as bare JSON, without the `<tool_call>` wrapper, and goinfer used to return it as prose, so the agent got no call at all: 0 wrapped calls in 1,200 sampled requests (the 0.5B and the 1.5B, each at int4 and int8). goinfer now accepts a bare JSON object at the start of the reply whose name is one of the supplied tools.

## Use it

- The constraint is on by default. Setting `GOINFER_TOOL_UNION=0` in the server's environment turns off the multi-tool constraint and restores the earlier behaviour (a single forced tool is still constrained).
- `tool_choice` selects the mode. A named function is constrained to that tool, and naming one that is not in `tools` is a 400. `required` (Anthropic `any`) forces a call to one of the tools. `auto` constrains a call once the model starts one.
- A server started with speculative decoding, which guesses several tokens ahead and checks them in one step ([Faster, with the same words](/different/14-faster-same-words/)), leaves `auto` unconstrained. `required` and named tools are still constrained there.
- `goinfer-serve check http://127.0.0.1:8080` sends a dozen-tool request to a running server and reports whether the model called the tool, before you point an agent at it. [Find out before your agent does](/different/19-find-out-before-your-agent-does/) covers it.
- [Tool-call coverage](https://github.com/townsendmerino/goinfer/blob/main/docs/tool-call-coverage.md) lists which families are constrained, parsed only, or have no tool support, and on what evidence.
