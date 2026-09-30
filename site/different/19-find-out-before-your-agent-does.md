---
title: "Find out before your agent does"
area: "Agents"
order: 19
summary: "serve check sends a dozen-tool schema to a loaded model before you configure opencode, and fit sizes a checkpoint for your machine first. Both are measured."
stand: "Two commands answer the expensive questions before an agent asks them: will this checkpoint fit here, and will it still call tools when the schema looks like a real agent's."
measured: 2026-09-18
reviewed: 2026-09-29
facts:
  - {label: "will it call tools", value: "`goinfer-serve check`, row `tools, harness-scale`: `ok` or `skip`"}
  - {label: "will it fit", value: "`goinfer-chat fit <model>`"}
  - {label: "`skip` means", value: "pick another checkpoint"}
  - {label: "worked, measured", value: "Qwen2.5-7B-Instruct q4_k_m, CUDA, RTX 2070 SUPER 8 GB"}
doesnt:
  - title: "It doesn't guarantee a long agent session."
    text: "The check sends one request at temperature 0 and a short prompt, and the harness-scale row stops after the first turn (the call). opencode's real turns on the working setup were 7,165 and 7,399 input tokens. An `ok` means the checkpoint can call a tool among 12; it does not say it will hold up over many turns."
  - title: "A skip is an answer about the checkpoint, not a fault to fix."
    text: "The record's reason is \"too small to tool-call under a real schema\". There is no flag that turns a skip into an ok. The advice in the opencode page is to stop and pick a different checkpoint."
  - title: "The Tools row is thin, and it is dated."
    text: "Only one of the five registry checkpoints has a measured harness-scale result (qwen2.5-coder-0.5b, skip, 2026-09-07); the other four say \"not yet measured\". That result predates the 2026-09-24 change that grammar-constrains tool calls on Qwen models, and it has not been re-measured since."
  - title: "The evidence is a handful of runs."
    text: "Two checkpoints (a 1.5B and a 7B) were run through both the check and opencode, on one Linux box. The smaller ones were checked but not always then driven by opencode, as the table says."
  - title: "fit is not a promise that the load will be gentle."
    text: "fit loads the checkpoint itself to count bytes (32.4s for gpt-oss-20b), and on 2026-09-18 it reported that model comfortably fitting just before a real load drove swap growth on a machine with 37+ GB free. The load-time guard was changed afterwards; whether fit's own report now prices the same transient peak is not stated in the records."
figures:
  - {text: "12", source: "docs/integrations/opencode.md"}
  - {text: "7,165", source: "docs/integrations/opencode.md"}
  - {text: "7,399", source: "docs/integrations/opencode.md"}
  - {text: "6,824 / 8,192 MiB", source: "docs/integrations/opencode.md"}
  - {text: "26.2 s", source: "docs/integrations/opencode.md"}
  - {text: "9,917", source: "docs/measurements/cold-user-2026-09-18-nobara-pc.md"}
  - {text: "32.4s", source: "docs/measurements/cold-user-2026-09-18-nobara-pc.md"}
  - {text: "54.60 GB", source: "docs/measurements/cold-user-2026-09-18-nobara-pc.md"}
  - {text: "6.88 GB", source: "docs/measurements/cold-user-2026-09-18-nobara-pc.md"}
sources:
  - "docs/integrations/opencode.md"
  - "docs/integrations/claude-code.md"
  - "docs/measurements/cold-user-2026-09-06-nobara-pc.md"
  - "docs/measurements/cold-user-2026-09-07-macbook-arm64.md"
  - "docs/measurements/cold-user-2026-09-18-nobara-pc.md"
  - "docs/tasks/task-first-hour.md"
  - "docs/capability-matrix.json"
  - "internal/servecheck/check.go"
  - "internal/fitcmd/fit.go"
---

## The problem

An agent like opencode or Claude Code does not send a chat message. It sends the whole tool list and a long system prompt with every turn, and it expects the model to answer with a real tool call. Two things go wrong, and both are slow to find out from the agent itself.

The first is size. A checkpoint that calls a tool when offered one function can stop doing it when offered a dozen. On 2026-09-06, on the nobara-pc box, opencode drove a 1.5B model through two attempts of about 173 s each and got a tool call printed as plain prose both times, not a real one. The server was working.

The second is memory. On 2026-09-07, on a 16 GB MacBook, a 7B model whose load-time check read "79% of budget" pushed the machine into heavy swapping once opencode's first request arrived.

## What goinfer does

`goinfer-serve check` drives a server that is already running, the way a harness would, and prints one line per feature. One of them, `tools, harness-scale`, sends a dozen-tool schema shaped like opencode's own "build" agent. It reports `ok` or `skip`. On a checkpoint that passes:

```
tools, OpenAI .............  ok    call get_weather({"city": "Paris"}) → result → answer in 2 turns
tools, harness-scale ......  ok    call get_weather({"city": "Paris"}) among 12 tools
```

On a 1.5B model, from the 2026-09-07 MacBook record, the same row reads:

```
tools, harness-scale ......  skip  model did not call the tool under a harness-scale (12-tool)
                                    schema — model answered without calling the tool
                                    (finish_reason="stop") — too small for tool use, or its
                                    template has no tool section
```

`goinfer-chat fit <model>` is the memory half. It prints one plan per backend compiled into the binary. This is the shape it printed for a 12 GB checkpoint on nobara-pc (2026-09-18):

```
cpu     RESIDENT       dense 1.46 GB + experts 11.12 GB + KV@8192 0.75 GB fits 54.60 GB free
cuda    EXPERT-CACHED  dense 1.46 GB + 13/32 experts 4.52 GB + KV@8192 0.75 GB fits 6.88 GB free
```

## How it works

`check` is a client. It starts nothing, so it exercises whatever is serving, with the flags you chose. The tools row makes one chat request with `temperature` 0 and a dozen function definitions, several with nested parameters, and looks for a well-formed call to `get_weather`. A model that answers in prose is a `skip`, not a failure, because that is a fact about the checkpoint. A `FAIL` is reserved for a broken server route, and only a `FAIL` gives a non-zero exit code.

`fit` loads the checkpoint like any other command, so it counts real bytes at the real quantization, and it plans for a context of 8192 unless you pass `-ctx`. It is not free: it took 32.4s on that 12 GB model. What it saves is the surprise. It does not start serving, and the 2026-09-18 run recorded no swap growth.

The Models page's tool-calling line is a field on each registry checkpoint, filled from a recorded `check` run, else "not yet measured".

## What was measured

The claim is narrow: where the check and opencode both ran on the same setup, the row matched what opencode did.

| Machine | Checkpoint | `tools, harness-scale` | What opencode did | Date |
|---|---|---|---|---|
| nobara-pc, RTX 2070 SUPER | Qwen2.5-Coder 1.5B | `skip`, reproduced after the row was added | printed a fake JSON call as prose, twice | 2026-09-06 (opencode 1.18.29) |
| nobara-pc | qwen2.5-coder-0.5b | `skip` | no turn completed (first request was 9,917 tokens, over the context) | 2026-09-07 check, 2026-09-18 opencode |
| MacBook, M1 Pro 16 GB | Qwen2.5-Coder 1.5B, CPU | `skip` | not run on this model | 2026-09-07 |
| MacBook, M1 Pro 16 GB | Qwen2.5-Coder 3B | `skip` | not run on this model | after 2026-09-08 |
| nobara-pc, CUDA, `-ctx 16384` | Qwen2.5-7B-Instruct q4_k_m | `ok`, all 8 checks passed | real `Read` and `Glob` calls, correct answer, two turns | 2026-09-08 (v0.17.2) |

For the working row, the load took 26.2 s and VRAM stayed at 6,824 / 8,192 MiB for the whole exchange. opencode's two turns used 7,165 and 7,399 input tokens. The two MacBook rows have no opencode run on that model, so they show the check's answer, not a match. The 0.5B row is consistent with the check, but the agent's first request was already larger than the context window.

The 2026-09-06 opencode run came first. The harness-scale row was added because of it, so that row is a reproduction, not a forecast. The 7B run is the only `ok`.

## Use it

```
goinfer-serve -model coder=<path>/qwen2.5-7b-instruct-q4_k_m.gguf -quant int4 -backend cuda -ctx 16384
goinfer-serve check http://127.0.0.1:8080
```

`check` takes `-model` (default: the first model the server lists), `-api-key` (or `GOINFER_API_KEY`), `-long-prompt` (default 2000 words, 0 skips that row) and `-timeout`.

To size a model before serving it:

```
goinfer-chat fit <file.gguf|dir|hf:owner/repo[:quant]>
goinfer-chat fit <model> -ctx 32768
```

An explicit `-ctx` is refused if it does not fit, not shrunk; `-measure` adds a real short decode on the best backend, which is a second load.

The opencode config and the reasoning for the working setup are in `docs/integrations/opencode.md`. The Claude Code page (`docs/integrations/claude-code.md`) is measured separately, on 2026-09-02, before this row existed.
