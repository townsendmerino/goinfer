---
title: "Faster, with the same words"
area: "Speed"
order: 14
summary: "On Metal, guessing ahead from text already in context: 2.082× plain decode on copy-heavy requests, 1.068× on chat, and every temperature-0 reply unchanged."
stand: "When the reply is likely to repeat text already in the prompt, goinfer guesses the next few tokens and checks them all in one step. A wrong guess costs a little time; it never changes a word."
measured: 2026-09-28
reviewed: 2026-09-29
facts:
  - {label: "copy-heavy, Metal 1.5B", value: "2.082× plain decode"}
  - {label: "chat, Metal 1.5B", value: "1.068×"}
  - {label: "every reply", value: "identical to plain decode"}
  - {label: "turn it on", value: "`--spec ngram`"}
doesnt:
  - title: "It doesn't help ordinary chat much."
    text: "On the 1.5B, chat read 1.068×, and on the 7B 1.014×. There is little to copy in a fresh answer. The only bar set for chat was not to lose (at least 0.97×), and it did not lose."
  - title: "Today it gives up batching under load."
    text: "A server run with `--spec` gives the GPU model to one generation at a time, so it loses the batching of several conversations. With four clients on Metal it got 0.610× the batched server's total throughput on copy traffic and 0.489× on chat. Use it for one user or an agent loop, not a shared server. A mode that speculates only while a generation is alone and joins the batch when others arrive (`-spec-adaptive`) went into the code as an opt-in on 2026-09-29. Its speed under load has not been measured yet."
  - title: "Same words is a greedy promise."
    text: "Greedy decoding (temperature 0) always takes the model's top-scoring token. There the reply is identical to plain decode's, and that is what was graded. For sampled requests (temperature above 0), the server's own flag text promises the same distribution of replies, not the same tokens as a plain run."
  - title: "The best case was measured, on one small model."
    text: "The copy workload hands the model about 960 tokens of a Go file and asks for it back verbatim, which is what this kind of guessing is best at. It was graded on the 1.5B, on one MacBook Pro (M1 Pro, 16 GB), at one client. The 7B is reported, not graded."
  - title: "Tool calls and structured output are not sped up on the GPU."
    text: "With the model on the GPU, a request held to a grammar (a tool call, or `response_format`) runs plain decode. goinfer's grammar-aware speculation runs only on the CPU backend. Both measured workloads were free text."
figures:
  - {text: "2.082", source: "docs/measurements/metal-spec-step-verify-2026-09-27.md"}
  - {text: "1.068", source: "docs/measurements/metal-spec-step-verify-2026-09-27.md"}
  - {text: "0.977", source: "docs/measurements/metal-spec-step-verify-2026-09-27.md"}
  - {text: "0.956", source: "docs/measurements/metal-spec-step-verify-2026-09-27.md"}
  - {text: "1.846", source: "docs/measurements/metal-spec-step-verify-2026-09-27.md"}
  - {text: "1.014", source: "docs/measurements/metal-spec-step-verify-2026-09-27.md"}
  - {text: "109", source: "docs/measurements/metal-spec-step-verify-2026-09-27.md"}
  - {text: "52.7", source: "docs/measurements/metal-spec-step-verify-2026-09-27.md"}
  - {text: "0.108", source: "docs/measurements/metal-spec-step-verify-2026-09-27.md"}
  - {text: "0.235", source: "docs/measurements/metal-spec-step-verify-2026-09-27.md"}
  - {text: "0.71–0.96", source: "docs/measurements/metal-spec-step-verify-2026-09-27.md"}
  - {text: "0.97", source: "docs/measurements/metal-spec-step-verify-2026-09-27.md"}
  - {text: "1.25", source: "docs/measurements/metal-spec-step-verify-2026-09-27.md"}
  - {text: "1.066", source: "docs/measurements/metal-spec-step-verify-2026-09-27.md"}
  - {text: "macOS 26.6.2", source: "docs/measurements/metal-spec-step-verify-2026-09-27.md"}
  - {text: "0.979", source: "docs/measurements/spec-vs-batching-metal-2026-09-27.md"}
  - {text: "0.928", source: "docs/measurements/spec-vs-batching-metal-2026-09-27.md"}
  - {text: "0.610", source: "docs/measurements/spec-vs-batching-metal-2026-09-27.md"}
  - {text: "0.489", source: "docs/measurements/spec-vs-batching-metal-2026-09-27.md"}
  - {text: "3 / 18", source: "docs/measurements/spec-vs-batching-metal-2026-09-27.md"}
  - {text: "15 / 72", source: "docs/measurements/spec-vs-batching-metal-2026-09-27.md"}
  - {text: "24 / 24", source: "docs/measurements/spec-vs-batching-metal-2026-09-27.md"}
  - {text: "156", source: "docs/measurements/spec-vs-batching-metal-2026-09-27.md"}
  - {text: "157", source: "docs/measurements/spec-vs-batching-metal-2026-09-27.md"}
  - {text: "14/14", source: "docs/measurements/spec-vs-batching-metal-2026-09-27.md"}
  - {text: "11/14", source: "docs/measurements/spec-vs-batching-metal-2026-09-27.md"}
sources:
  - "docs/measurements/metal-spec-step-verify-2026-09-27.md"
  - "docs/measurements/spec-vs-batching-metal-2026-09-27.md"
  - "docs/spec/02-cache-ngram.md"
  - "docs/server.md"
  - "scripts/bench_spec_copy.py"
  - "internal/serveapp/openai.go"
---

## The problem

A model writes one token per step, and each step reads all of its weights. A step that checks eight tokens costs much less than eight separate steps, because the weights are read once either way. So if you could guess the next several tokens correctly, you would get several tokens for about the price of one. Guessing ahead and then checking is called speculative decoding.

Some traffic is easy to guess. A code edit echoes the file around it, a retrieval answer quotes a passage, and an agent loop repeats a system prompt and earlier tool results.

The catch is that a guess must not change the answer. A speculation scheme that makes replies faster and different is a different model.

## What goinfer does

With `--spec ngram`, goinfer looks at the last few tokens of the reply (an n-gram is a short run of tokens). It finds where they occurred earlier in the prompt or the reply, and guesses that the tokens which followed there come next. It then runs the model once over all the guessed tokens, keeps the ones that match what the model itself would have picked, and discards the rest. The reply is the one plain decoding (one token per step, no guessing) would produce.

It needs no second model: the guesses are a lookup over text goinfer already holds.

To try it, start a server with the flag and ask for a passage back, as the benchmark does. The benchmark sends about 3,500 characters of Go and wraps them in a Go code fence.

```sh
goinfer-serve -model bench=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -backend metal --embed-int4=false --spec ngram

curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "bench", "temperature": 0, "max_tokens": 256,
  "messages": [{"role": "user", "content": "Here is part of a Go file:\n\n<paste Go code here>\n\nRewrite the code above EXACTLY, character for character, with no changes and no commentary."}]}'
```

## How it works

The index covers the prompt and everything generated so far. A controller decides how many tokens to guess each round. It compares the expected number of tokens gained against the cost of the step that checks the guesses (the verify step). It guesses none when the odds are poor, so a round with no guess costs what plain decode does.

The verify step is where Metal was the problem. Each guessed token is one row of the verify step. The first Metal verify reused the ordinary decode kernels (GPU programs), and the record measured each extra row at 0.71–0.96 of a token's cost, which left almost nothing to save. On 2026-09-27 the verify moved onto the kernels built for [batching several conversations](/different/13-batching-same-answer/), which run several rows in one pass and give bit-identical results. There, with eight rows in a step, each extra row costs 0.108 to 0.235 of a token on the 1.5B and the 7B, at conversation depths from 128 to 2048 tokens.

## What was measured

One MacBook Pro (M1 Pro, 16 GB, macOS 26.6.2), on 2026-09-27. The models were Qwen2.5-Coder 1.5B (graded, meaning held to bars set before the run) and Qwen2.5 7B Instruct (reported), both 4-bit q4_k_m files loaded at int4. One client, greedy (temperature 0). Ratios are speculation on against plain decode.

| Workload | 1.5B (graded) | 7B (reported) | Earlier build, 1.5B |
|---|---|---|---|
| Copy-heavy: a Go file section returned verbatim | 2.082× | 1.846× | 0.977× |
| Chat: six-turn conversations | 1.068× | 1.014× | 0.956× |

On the 1.5B, copy traffic ran at 109 tok/s (tokens per second) against 52.7 tok/s plain. The run had three rounds. Each round ran plain decode, speculation on the earlier build and speculation on the new build, in a rotated order, and each measurement waited for the machine to be idle first. The bars were set in writing before the run ([pre-registered](/different/16-numbers-with-receipts/)): every reply equal to plain decode's (a must-pass), copy at least 1.25×, chat at least 0.97×. All passed. The chat median hides one odd round: the first plain chat measurement was slow, so the three chat ratios span 1.066 to 1.317. The record leaves that unexplained and notes that the two clean rounds read 1.066 and 1.068.

The record is [on GitHub](https://github.com/townsendmerino/goinfer/blob/main/docs/measurements/metal-spec-step-verify-2026-09-27.md). To reproduce it from a clone of the repo, run each benchmark as shown, then again without speculation (`--key plain --serve-args=-embed-int4=false`):

```sh
GOINFER_SERVE_CPU=<serve binary> BENCH_W7_MODEL=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  python3 scripts/bench_spec_copy.py copy.json --key spec --clients 1 --serve-args='-spec=ngram -embed-int4=false'
GOINFER_SERVE_CPU=<serve binary> BENCH_W7_MODEL=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  python3 scripts/bench_w7_plain.py chat.json --clients 1 --engines goinfer --backend metal --key spec --fixed-nonce --serve-args='-spec=ngram -embed-int4=false'
```

## What we got wrong

**It was not faster.** That morning, an earlier build measured 0.979× plain decode on copy traffic and 0.928× on chat, on the 1.5B with one client. The verify cost above is why, and the record set the idea aside for Metal. That evening's graded run measured the same earlier build again, at 0.977× and 0.956×, beside the reworked verify in the table.

**Later turns came back different.** That same morning measurement compared every reply with plain decode's. The first reply of each conversation matched. Later turns mostly did not: 3 / 18 replies matched at one client and 15 / 72 at four, while the single-turn copy requests matched on all 30. The flag's help text and the architecture notes promised identical output, so this was a bug against our own claim.

The cause was not the speculation arithmetic. The speculation loop runs a round's last token through the model only as the first row of the next round. When a reply stopped at `max_tokens`, that token was never run, and the cache was left one token short. So the next turn reused 156 cached positions where plain decode reused 157. ([Turn nine in under half a second](/different/09-turn-nine-in-under-half-a-second/) explains how a turn reuses the cache.) That one token was then read in through Metal's batched prompt path (prefill), whose arithmetic is not bit-identical to decode's, and every later position attended to it.

The fix runs that last token through the model when the generation ends, as plain decode does (commit [`97615930`](https://github.com/townsendmerino/goinfer/commit/97615930)). Through the server, 24 / 24 turns at four clients then matched, with the same reuse counts and an unchanged speed. The same trailing-token gap turned up in CUDA's block drafter (speculation that uses a small draft model, `--drafter`), where a later turn's tokens diverged. On a Linux PC (Ryzen 7 3700X, RTX 2070 SUPER 8 GB, NVIDIA driver 595.91.07), on 2026-09-28, the two-turn tests pass 14/14 with the fix and fail 11/14 with it reverted.

## Use it

- `--spec ngram` on `goinfer-serve` (and `goinfer-chat`). Default off. It falls back to plain decode for a request it cannot speculate on.
- On Metal it verifies on the batched-step kernels.
- On Metal, the default `--embed-int4` currently sends a load to the CPU; pass `--embed-int4=false` ([docs/quantization.md, Known issue](https://github.com/townsendmerino/goinfer/blob/main/docs/quantization.md)). These measurements predate that default, which dates from 2026-09-28.
- Best for one user, an editor or an agent loop with copy-heavy edits. Leave it off for a server shared by several clients.
