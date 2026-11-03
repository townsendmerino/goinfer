---
title: "Find out before your agent does"
area: "Agents"
order: 19
summary: "serve check sends a dozen-tool schema to a running model before you set up opencode, and fit sizes a checkpoint for your machine first. Both are measured."
stand: "Two commands answer the expensive questions before an agent asks them: will this checkpoint fit here, and will it still call tools when the schema looks like a real agent's."
measured: 2026-09-18
reviewed: 2026-09-29
facts:
  - {label: "will it call tools", value: "`goinfer-serve check`, row `tools, harness-scale`: `ok` or `skip`"}
  - {label: "will it fit", value: "`goinfer-chat fit <model>`"}
  - {label: "`skip` means", value: "pick another checkpoint"}
  - {label: "measured working setup", value: "Qwen2.5-7B-Instruct q4_k_m, CUDA, RTX 2070 SUPER 8 GB"}
doesnt:
  - title: "It doesn't guarantee a long agent session."
    text: "The check sends one request at temperature 0 (always the most likely token) with a short prompt, and the harness-scale row stops after the model's first turn (the call). opencode's real turns on the working setup were 7,165 and 7,399 input tokens. An `ok` means the checkpoint can call a tool among 12; it does not say it will hold up over many turns."
  - title: "A skip is an answer about the checkpoint, not a fault to fix."
    text: "The record's reason is \"too small to tool-call under a real schema\". There is no flag that turns a skip into an ok. goinfer's opencode guide advises you to stop and pick a different checkpoint."
  - title: "The Models page's tool results are thin, and dated."
    text: "Only one of the five checkpoints in goinfer's download registry has a measured harness-scale result (qwen2.5-coder-0.5b, skip, 2026-09-07); the other four say \"not yet measured\". That result predates the 2026-09-24 change that holds tool calls on Qwen models to a grammar, and it has not been re-measured since."
  - title: "The evidence is a handful of runs."
    text: "Two checkpoints (a 1.5B and a 7B) were run through both the check and opencode, on one Linux PC. The smaller ones were checked but not always then driven by opencode, as the table says."
  - title: "fit is not a promise that the load will be gentle."
    text: "fit loads the checkpoint itself to count bytes (32.4s for gpt-oss-20b). On 2026-09-18 it reported that model fitting comfortably, and then a real load of it pushed the machine into swap, although 37+ GB was free. goinfer's load-time memory check was changed afterwards. The records do not say whether fit's own report now accounts for the same short-lived memory peak during loading."
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

The first is size. A checkpoint (one model's weights, as a file) that calls a tool when offered one function can stop doing it when offered a dozen. On 2026-09-06, on a Linux PC (Ryzen 7 3700X, on its CPU), opencode drove Qwen2.5-Coder 1.5B through two attempts of about 173 s each. Both times the model printed a tool call as plain text, which opencode could not act on. The server was working.

The second is memory. On 2026-09-07, on a MacBook Pro (M1 Pro, 16 GB), Qwen2.5-7B-Instruct (q3_k_m) passed the load-time memory check at "79% of budget". Then opencode's first request arrived and pushed the machine into heavy swapping. That check has since been rewritten; [It refuses rather than swaps](/different/07-refuses-rather-than-swaps/) describes how goinfer checks memory now.

## What goinfer does

`goinfer-serve check` sends requests to a server that is already running, the way an agent program (a harness) would, and prints one line per feature. One of them, `tools, harness-scale`, sends a dozen tool definitions shaped like the ones opencode's own "build" agent sends. It reports `ok` or `skip`. On a checkpoint that passes (the 7B in the table below):

```
tools, OpenAI .............  ok    call get_weather({"city": "Paris"}) → result → answer in 2 turns
tools, harness-scale ......  ok    call get_weather({"city": "Paris"}) among 12 tools
```

On Qwen2.5-Coder 1.5B, on the MacBook Pro (2026-09-07), the same row reads:

```
tools, harness-scale ......  skip  model did not call the tool under a harness-scale (12-tool)
                                    schema — model answered without calling the tool
                                    (finish_reason="stop") — too small for tool use, or its
                                    template has no tool section
```

`goinfer-chat fit <model>` is the memory half. It prints one plan per backend compiled into the binary. This is what it printed for gpt-oss-20b, a 12 GB mixture-of-experts checkpoint, on the Linux PC (2026-09-18):

```
cpu     RESIDENT       dense 1.46 GB + experts 11.12 GB + KV@8192 0.75 GB fits 54.60 GB free
cuda    EXPERT-CACHED  dense 1.46 GB + 13/32 experts 4.52 GB + KV@8192 0.75 GB fits 6.88 GB free
```

`RESIDENT` means the whole model fits in that device's memory. `EXPERT-CACHED` means the GPU holds the shared weights and room for 13 of each layer's 32 experts, and fetches other experts when a token needs them; [A 26B model on an 8 GB card](/different/08-a-26b-model-on-an-8gb-card/) explains that mode. `KV@8192` is the KV cache, the model's working memory for the conversation, sized for 8192 tokens.

## How it works

`check` is a client. It starts nothing, so it tests whatever is serving, with the flags you chose. The tools row makes one chat request with `temperature` 0 and a dozen function definitions, several with nested parameters, and looks for a well-formed call to `get_weather`. A model that answers in prose is a `skip`, not a failure, because that is a fact about the checkpoint. A `FAIL` is reserved for a broken server route, and only a `FAIL` gives a non-zero exit code.

`fit` loads the checkpoint like any other command, so it counts real bytes at the real quantization (the number format the weights are stored in), and it plans for a context of 8192 tokens unless you pass `-ctx`. It is not free: it took 32.4s on that 12 GB model. What it saves is the surprise. It does not start serving, and the 2026-09-18 fit run itself recorded no swap growth.

The [Models page](/models/) shows a tool-calling line for each checkpoint in goinfer's download registry. It is filled from a recorded `check` run, or else says it has not been measured.

## What was measured

The claim is narrow: where the check and opencode both ran on the same setup, the row matched what opencode did.

| Machine | Checkpoint | `tools, harness-scale` | What opencode did | Date |
|---|---|---|---|---|
| Linux PC, on its CPU | Qwen2.5-Coder 1.5B | `skip`, reproduced after the row was added | printed a fake JSON call as prose, twice | 2026-09-06 (opencode 1.18.29) |
| Linux PC | qwen2.5-coder-0.5b | `skip` | no turn completed (first request was 9,917 tokens, over the context) | 2026-09-07 check, 2026-09-18 opencode |
| MacBook Pro (M1 Pro, 16 GB) | Qwen2.5-Coder 1.5B, CPU | `skip` | not run on this model | 2026-09-07 |
| MacBook Pro (M1 Pro, 16 GB) | Qwen2.5-Coder 3B | `skip` | not run on this model | 2026-09-07 |
| Linux PC, CUDA, `-ctx 16384` | Qwen2.5-7B-Instruct q4_k_m | `ok`, all 8 checks passed | real `Read` and `Glob` calls, correct answer, two turns | 2026-09-08 (v0.17.2) |

For the working row, the load took 26.2 s and GPU memory use stayed at 6,824 / 8,192 MiB for the whole exchange. opencode's two turns used 7,165 and 7,399 input tokens. The two MacBook rows have no opencode run on that model, so they show the check's answer, not a match. The 0.5B row is consistent with the check, but the agent's first request was already larger than the context window.

The 2026-09-06 opencode run came first. The harness-scale row was added because of it, so for that model the row reproduced a known failure rather than predicting one. The 7B run is the only `ok`. All of these runs predate 2026-09-24, when goinfer began holding tool calls on Qwen models to a grammar ([Tool calls that can't come out malformed](/different/03-tool-calls-that-cant-be-malformed/)).

## Use it

```
goinfer-serve -model coder=<path>/qwen2.5-7b-instruct-q4_k_m.gguf -quant int4 -backend cuda -ctx 16384
goinfer-serve check http://127.0.0.1:8080
```

`check` takes `-model` (default: the first model the server lists), `-api-key` (or `GOINFER_API_KEY`), `-long-prompt` (default 2000 words, 0 skips that row) and `-timeout` (default 10 minutes).

To size a model before serving it:

```
goinfer-chat fit <file.gguf|dir|hf:owner/repo[:quant]>
goinfer-chat fit <model> -ctx 32768
```

If you pass `-ctx` and it does not fit, fit refuses it rather than shrinking it. `-measure` also times a short real generation on the best backend, which is a second load.

The opencode config and the reasoning for the working setup are in [the opencode guide](https://github.com/townsendmerino/goinfer/blob/main/docs/integrations/opencode.md). [The Claude Code guide](https://github.com/townsendmerino/goinfer/blob/main/docs/integrations/claude-code.md) was measured separately, on 2026-09-02, before this row existed.
