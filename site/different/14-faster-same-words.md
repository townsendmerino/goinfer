---
title: "Faster, with the same words"
area: "Speed"
order: 14
summary: "Lossless n-gram speculation on Metal: 2.082× plain decode on copy-heavy requests, 1.068× on chat, and every greedy reply identical to plain decode."
stand: "When the reply is likely to repeat text already in the prompt, goinfer guesses the next few tokens and checks them all in one step. A wrong guess costs a little time; it never changes a word."
measured: 2026-09-28
reviewed:
facts:
  - {label: "copy-heavy, Metal 1.5B", value: "2.082× plain decode"}
  - {label: "chat, Metal 1.5B", value: "1.068×"}
  - {label: "every reply", value: "identical to plain decode"}
  - {label: "turn it on", value: "--spec ngram"}
doesnt:
  - title: "It doesn't help ordinary chat much."
    text: "On the 1.5B, chat read 1.068×, and on the 7B 1.014×. There is little to copy in a fresh answer. The record graded chat only for not losing (at least 0.97×), and it did not lose."
  - title: "Today it gives up batching under load."
    text: "A server run with --spec takes the GPU model for one generation at a time. With four clients on Metal it read 0.610× the batched aggregate on copy traffic and 0.489× on chat. Use it for one user or an agent loop, not a shared server. A \"speculate when alone, batch under load\" mode is registered as a candidate; it is not built."
  - title: "Same words is a greedy promise."
    text: "At temperature 0 the reply is identical to plain decode's, and that is what was graded. For sampled requests the server's own flag text promises the same distribution, not the same tokens as a plain run."
  - title: "The best case was measured, on one small model."
    text: "The copy workload hands the model about 960 tokens of a Go file and asks for it back verbatim, which is what n-gram drafting is best at. It was graded on the 1.5B on one MacBook, at one client. The 7B is reported, not graded."
figures:
  - {text: "2.082", source: docs/measurements/metal-spec-step-verify-2026-09-27.md}
  - {text: "1.068", source: docs/measurements/metal-spec-step-verify-2026-09-27.md}
  - {text: "0.977", source: docs/measurements/metal-spec-step-verify-2026-09-27.md}
  - {text: "0.956", source: docs/measurements/metal-spec-step-verify-2026-09-27.md}
  - {text: "1.846", source: docs/measurements/metal-spec-step-verify-2026-09-27.md}
  - {text: "1.014", source: docs/measurements/metal-spec-step-verify-2026-09-27.md}
  - {text: "109", source: docs/measurements/metal-spec-step-verify-2026-09-27.md}
  - {text: "52.7", source: docs/measurements/metal-spec-step-verify-2026-09-27.md}
  - {text: "0.108", source: docs/measurements/metal-spec-step-verify-2026-09-27.md}
  - {text: "0.235", source: docs/measurements/metal-spec-step-verify-2026-09-27.md}
  - {text: "0.71–0.96", source: docs/measurements/metal-spec-step-verify-2026-09-27.md}
  - {text: "0.97", source: docs/measurements/metal-spec-step-verify-2026-09-27.md}
  - {text: "1.25", source: docs/measurements/metal-spec-step-verify-2026-09-27.md}
  - {text: "1.066", source: docs/measurements/metal-spec-step-verify-2026-09-27.md}
  - {text: "macOS 26.6.2", source: docs/measurements/metal-spec-step-verify-2026-09-27.md}
  - {text: "0.979", source: docs/measurements/spec-vs-batching-metal-2026-09-27.md}
  - {text: "0.928", source: docs/measurements/spec-vs-batching-metal-2026-09-27.md}
  - {text: "0.610", source: docs/measurements/spec-vs-batching-metal-2026-09-27.md}
  - {text: "0.489", source: docs/measurements/spec-vs-batching-metal-2026-09-27.md}
  - {text: "3 / 18", source: docs/measurements/spec-vs-batching-metal-2026-09-27.md}
  - {text: "15 / 72", source: docs/measurements/spec-vs-batching-metal-2026-09-27.md}
  - {text: "24 / 24", source: docs/measurements/spec-vs-batching-metal-2026-09-27.md}
  - {text: "156", source: docs/measurements/spec-vs-batching-metal-2026-09-27.md}
  - {text: "157", source: docs/measurements/spec-vs-batching-metal-2026-09-27.md}
  - {text: "14/14", source: docs/measurements/spec-vs-batching-metal-2026-09-27.md}
  - {text: "11/14", source: docs/measurements/spec-vs-batching-metal-2026-09-27.md}
sources:
  - docs/measurements/metal-spec-step-verify-2026-09-27.md
  - docs/measurements/spec-vs-batching-metal-2026-09-27.md
  - docs/spec/02-cache-ngram.md
  - docs/server.md
  - scripts/bench_spec_copy.py
---

## The problem

A model writes one token per step, and each step reads all of its weights. A step that checks eight tokens costs much less than eight separate steps, because the weights are read once either way. So if you could guess the next several tokens correctly, you would get several tokens for about the price of one.

Some traffic is easy to guess. A code edit echoes the file around it, a retrieval answer quotes a passage, and an agent loop repeats a system prompt and earlier tool results.

The catch is that a guess must not change the answer. A speculation scheme that makes replies faster and different is a different model.

## What goinfer does

With `--spec ngram`, goinfer looks at the last few tokens of the reply, finds where they occurred earlier in the prompt or the reply, and guesses that the tokens which followed there come next. It then runs the model once over all the guessed tokens, keeps the ones that match what the model itself would have picked, and discards the rest. The reply is the one plain decoding would produce.

It needs no second model: the guesses are a lookup over text goinfer already holds.

To try it, start a server with the flag and ask for a passage back, as the benchmark does:

```sh
goinfer-serve -model bench=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -backend metal --embed-int4=false --spec ngram
# request: "Here is part of a Go file: ... Rewrite the code above EXACTLY, character for character, with no changes and no commentary."
```

## How it works

The index covers the prompt and everything generated so far. A controller decides how many tokens to guess each round. It compares the expected number of tokens gained against the verify step's cost, and it guesses none when the odds are poor, so a round with no draft costs what plain decode does.

The verify step is where Metal was the problem. The first Metal verify reused production's decode kernels, and the record measured each extra verified row at 0.71–0.96 of a token, which left almost nothing to save. On 2026-09-27 the verify moved onto the batched-step kernels from the concurrency work, which run several rows in one pass and give bit-identical results. There, an extra row at eight rows costs 0.108 to 0.235 of a token on the 1.5B and the 7B, across depths 128 to 2048.

## What was measured

One MacBook Pro (M1 Pro, 16 GB, macOS 26.6.2), Qwen2.5-Coder 1.5B int4 (graded) and Qwen2.5 7B Instruct (reported), one client, greedy, 2026-09-27. Ratios are speculation on against plain decode.

| Workload | 1.5B (graded) | 7B (reported) | Earlier build, 1.5B |
|---|---|---|---|
| Copy-heavy: a Go file section returned verbatim | 2.082× | 1.846× | 0.977× |
| Chat: six-turn conversations | 1.068× | 1.014× | 0.956× |

On the 1.5B, copy traffic ran at 109 tok/s against 52.7 tok/s plain. Three rounds were run with the arm order rotated, each cell idle-gated. The bars were registered before the run: every reply equal to plain decode's (a hard gate), copy at least 1.25×, chat at least 0.97×. All passed. The chat median hides one odd round: the first plain chat cell was slow, so the three chat ratios span 1.066 to 1.317. The record leaves that unexplained and notes that the two clean rounds read 1.066 and 1.068.

Reproduce it with `scripts/bench_spec_copy.py` (copy) and `scripts/bench_w7_plain.py` (chat).

## What we got wrong

**It was not faster.** That morning, an earlier build measured 0.979× plain decode on copy traffic and 0.928× on chat, on the 1.5B with one client. The verify cost above is why, and the record parked the idea for Metal. That evening's graded run measured the same earlier build again, at 0.977× and 0.956×, beside the reworked verify in the table.

**Later turns came back different.** That same measurement compared every reply with plain decode's. The first reply of each conversation matched. Later turns mostly did not: 3 / 18 replies matched at one client and 15 / 72 at four, while the single-turn copy requests matched on all 30. The flag's help text and the architecture notes promised identical output, so this was a bug against our own claim.

The cause was not the speculation arithmetic. The speculation loop forwards a round's last token only as the first row of the next round. When a reply stopped at `max_tokens`, the cache was left one token short, so the next turn reused 156 positions where plain decode reused 157. That one token was then prefilled through Metal's batched prefill, whose arithmetic is not bit-identical to decode's, and every later position attended to it.

The fix forwards that last token when the generation ends, as plain decode does (commit 97615930). Through the server, 24 / 24 turns at four clients then matched, with the same reuse counts and an unchanged rate. The same trailing-token gap turned up in the CUDA block drafter, where a later turn's ids diverged. On nobara (2026-09-28) a two-turn test passes 14/14 with the fix and fails 11/14 with it reverted.

## Use it

- `--spec ngram` on `goinfer-serve` (and `goinfer-chat`). Default off. It falls back to plain decode for a request it cannot speculate on.
- On Metal it verifies on the batched-step kernels. Pass `--embed-int4=false` for now: the 2026-09-28 default sends a Metal load to the CPU (docs/quantization.md, "Known issue"). These measurements predate that default.
- Best for one user, an editor or an agent loop with copy-heavy edits. Leave it off for a server shared by several clients.
