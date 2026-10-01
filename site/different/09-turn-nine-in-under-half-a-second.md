---
title: "Turn nine in under half a second"
area: "Speed"
order: 9
summary: "goinfer keeps several conversations' history ready, so a new turn reads only what was added. On one CUDA replay, turn 9 of an agent session began in 319.5 ms."
stand: "In a long conversation, each new turn only has to process what you just added. goinfer keeps up to four conversations' history processed at once, so it is not read again."
measured: 2026-09-27
reviewed: 2026-09-29
facts:
  - {label: "conversations kept", value: "4 by default (--kv-sessions)"}
  - {label: "turn 1, nothing reused, 7B on CUDA", value: "651.2 ms to first token"}
  - {label: "turn 9, same conversation", value: "319.5 ms"}
  - {label: "turn off", value: "--kv-sessions 0"}
doesnt:
  - title: "It doesn't keep every conversation."
    text: "The default is four, and a fifth conversation replaces the one used least recently. Each kept conversation costs its own KV-cache memory (about 117 MB for the 1.5B at a 4k context on Metal, 896 MB per slot for the 7B at an 8192-token context on CUDA), and goinfer's memory check lowers the count to what fits. On an 8 GB card the 7B at 8192 got 2 of 4 slots, and four clients taking turns read 55.8 against 55.5 tok/s, no better than one slot."
  - title: "It doesn't reuse text that changed."
    text: "Only the shared start of the prompt is reused. Edit an early message, change the system prompt or the tool list, and everything after the change is prefilled again. A client that rewrites the history between turns gets no reuse. The cache lives in the server process, so a restart starts cold. --session-dir saves sessions to disk across a restart, but the docs describe it for the CPU session cache, not for the GPU slots."
  - title: "It doesn't help every model family equally."
    text: "Recurrent and hybrid families (Gated DeltaNet, Mamba-2, LFM2) keep some or all of their memory as a running state that each token updates. They keep one slot, and can only continue from exactly where the last turn ended. An edited message, a shorter prompt or an identical resend is prefilled from scratch, because that state cannot be rewound."
  - title: "On CPU, a warm turn is not always bit-identical to a cold one."
    text: "With the default fast prefill attention, a new stretch of 512 tokens or more is prefilled by a kernel whose rounding depends on where the prompt was split. So at temperature 0, a token the model scores as a near-tie can differ from a one-shot run of the same prompt. GOINFER_CPU_FAST_ATTENTION=0 makes reuse exact, at a cost the docs measure at 1.43x on a cold 2048-token turn."
  - title: "The turn-nine figure is one run on one machine."
    text: "It is a single replay of a 7B model on CUDA, built from code with uncommitted changes, and timed as a one-token request. The Metal records count tokens prefilled and whole-turn time, not time to first token. A bigger agent prompt can miss half a second: an earlier record's second turn, with 25 tool schemas, read 0.58 s."
figures:
  - {text: "651.2", source: "docs/benchmarks.md"}
  - {text: "319.5", source: "docs/benchmarks.md"}
  - {text: "251.3", source: "docs/benchmarks.md"}
  - {text: "191.2", source: "docs/benchmarks.md"}
  - {text: "219.8", source: "docs/benchmarks.md"}
  - {text: "1922", source: "docs/benchmarks.md"}
  - {text: "1668", source: "docs/benchmarks.md"}
  - {text: "727", source: "docs/benchmarks.md"}
  - {text: "0.58", source: "docs/integrations/claude-code.md"}
  - {text: "0.42", source: "docs/integrations/claude-code.md"}
  - {text: "9.13", source: "docs/integrations/claude-code.md"}
  - {text: "26, 27, 24, 20, 27", source: "docs/measurements/concurrency-mc1-2026-09-26.md"}
  - {text: "187, 342, 494, 642, 797", source: "docs/measurements/concurrency-mc1-2026-09-26.md"}
  - {text: "176, 331, 483, 631, 786", source: "docs/measurements/concurrency-mc1-cuda-2026-09-27.md"}
  - {text: "0.69×", source: "docs/measurements/concurrency-mc0-2026-09-26.md"}
  - {text: "27.4", source: "docs/measurements/concurrency-mc0-2026-09-26.md"}
  - {text: "43.3", source: "docs/measurements/concurrency-mc0-2026-09-26.md"}
  - {text: "117 MB", source: "docs/server.md"}
  - {text: "896 MB", source: "docs/measurements/concurrency-mc1-cuda-2026-09-27.md"}
  - {text: "55.8 against 55.5", source: "docs/measurements/concurrency-mc1-cuda-2026-09-27.md"}
  - {text: "1.43x", source: "docs/server.md"}
  - {text: "156", source: "docs/measurements/spec-vs-batching-metal-2026-09-27.md"}
  - {text: "157", source: "docs/measurements/spec-vs-batching-metal-2026-09-27.md"}
sources:
  - "docs/benchmarks.md"
  - "docs/integrations/claude-code.md"
  - "docs/measurements/concurrency-mc0-2026-09-26.md"
  - "docs/measurements/concurrency-mc1-2026-09-26.md"
  - "docs/measurements/concurrency-mc1-cuda-2026-09-27.md"
  - "docs/measurements/spec-vs-batching-metal-2026-09-27.md"
  - "docs/server.md"
  - "internal/serveapp/sessions.go"
  - "internal/serveapp/banner.go"
---

## The problem

Before a model writes its first word, it reads the whole prompt: the system prompt, the tool list, and every earlier turn. That step is called prefill, and its cost grows with the length of the conversation. In a chat or an agent loop, each turn resends everything so far plus a little more. If the server reads it all again each time, turn nine costs far more than turn one, though you added only a few lines.

An earlier goinfer record (2026-09-02) shows the shape. It ran Qwen2.5-7B-Instruct on a Linux PC (Ryzen 7 3700X, RTX 2070 SUPER 8 GB, NVIDIA driver 595.91.07), with 25 tool schemas in the prompt. With reuse switched off, turns 1 to 3 took 8.86 s, 9.01 s and 9.13 s. With it on, they took 8.80 s, 0.58 s and 0.42 s.

## What goinfer does

Prefill leaves behind the model's working memory for every token it read, called the KV cache (two vectors per token in each layer). goinfer keeps that cache for recent conversations. When a new request arrives, it finds the conversation whose stored tokens are the start of the new prompt, and processes only the tokens after that.

The record for this page replays a ten-turn coding-agent session (read a file, grep, edit, run tests, and so on) against Qwen2.5-7B-Instruct on the Linux PC's GPU. Turn 1 has 727 prompt tokens and nothing to reuse: first token in 651.2 ms. By turn 9 the prompt is 1922 tokens, 1668 of them already in the GPU's KV cache, and the first token arrives in 319.5 ms. Turns 2 to 10 span 191.2 to 319.5 ms.

## How it works

On the CPU, a small list of sessions holds each conversation's tokens and KV cache. A new prompt takes the session that shares the longest start with it. That match must be longer than what the session shares with the other sessions, so a system prompt that every conversation starts with does not pull in someone else's conversation. The session is cut back to the shared point, and only the rest is prefilled.

A model held entirely in GPU memory (Metal, CUDA, WebGPU) has no such list. Instead it reserves up to `--kv-sessions` KV-cache slots on the GPU, one conversation each. A request uses the slot that already holds the start of its prompt. A new conversation takes an empty slot, or else the least recently used one.

Either way, the response's `usage.prefill_reused_tokens` says how many prompt tokens were skipped.

## What was measured

The first table is the source of the title. It is the agent-turn replay on 2026-09-09 on the Linux PC (RTX 2070 SUPER, driver 595.91.07), built from goinfer [`b22e080e`](https://github.com/townsendmerino/goinfer/commit/b22e080e) plus uncommitted changes. The model is Qwen2.5-7B-Instruct from a Q4_K_M file, served as int4 (weights quantized to 4 bits), held entirely on the GPU, and greedy (always taking the most likely token). Every turn forces a named tool call, and a forced tool call does not stream: nothing arrives until the call is complete. So "time to first token" here is the wall-clock time of a request for one token.

<table>
<thead><tr><th>Turn</th><th>Prompt tokens</th><th>Reused</th><th>Time to first token</th></tr></thead>
<tbody>
<tr><td>1 (cold)</td><td>727</td><td>0</td><td>651.2 ms</td></tr>
<tr><td>2</td><td>937</td><td>726</td><td>251.3 ms</td></tr>
<tr><td>3</td><td>1068</td><td>936</td><td>223.1 ms</td></tr>
<tr><td>4</td><td>1235</td><td>1067</td><td>236.1 ms</td></tr>
<tr><td>5</td><td>1353</td><td>1234</td><td>191.2 ms</td></tr>
<tr><td>6</td><td>1442</td><td>1352</td><td>197.0 ms</td></tr>
<tr><td>7</td><td>1506</td><td>1441</td><td>191.4 ms</td></tr>
<tr><td>8</td><td>1669</td><td>1505</td><td>257.0 ms</td></tr>
<tr><td>9</td><td>1922</td><td>1668</td><td>319.5 ms</td></tr>
<tr><td>10</td><td>2030</td><td>1921</td><td>219.8 ms</td></tr>
</tbody>
</table>

The second measurement asks whether several conversations at once still each keep their own history. Each client (one conversation) sent six plain turns, greedy, with the clients taking turns. It ran on Metal (a MacBook Pro, M1 Pro, 16 GB, Qwen2.5-Coder 1.5B int4, 2026-09-26) and on CUDA (the Linux PC, same model, 2026-09-27). The figure is how many tokens each of turns 2 to 6 prefilled. With four slots it was 26, 27, 24, 20, 27 at 1, 2 and 4 clients on both backends. The previous build had one slot, and at 4 clients it prefilled 187, 342, 494, 642, 797 tokens on Metal and 176, 331, 483, 631, 786 on CUDA: every turn read the whole history again. Running several conversations' next tokens together in one step is a separate feature, covered in [Batching that doesn't change the answer](/different/13-batching-same-answer/).

## What we got wrong

**The CPU cache thrashed for three days.** A 2026-09-23 change to the matching rule let two conversations taking turns grab each other's session. Each reused only the 7-token start that the chat template puts on every prompt, so each turn threw away the other conversation's history, and the list never grew past one session. On the MacBook Pro, the record measured 0.69× the one-client throughput at two clients. It was found on 2026-09-26, by the CPU comparison run for the GPU work. The fix ([`9308696e`](https://github.com/townsendmerino/goinfer/commit/9308696e)) took two-client CPU serving from 27.4 to 43.3 tokens per second (tok/s).

**The GPU paths had one slot.** At two clients, Metal reused 7 tokens per turn after the first. Keeping several slots on the GPU fixed that (Metal on 2026-09-26, CUDA and WebGPU on 2026-09-27).

**One token short.** `--spec ngram` turns on speculative decoding: goinfer guesses the next few tokens from text already in the prompt and checks them in one step ([Faster, with the same words](/different/14-faster-same-words/)). With it on Metal (the MacBook Pro, 2026-09-27), a reply that stopped at `max_tokens` left the cache one token short. So the next turn reused 156 positions where plain decode (generating one token per step) reused 157. That one token then went through the prefill kernel, whose arithmetic differs slightly from decode's, and every later turn's reply came out different. The fix ([`97615930`](https://github.com/townsendmerino/goinfer/commit/97615930)) runs that last token through the model at the end of the reply, as plain decode does.

## Use it

Start a server as in the replay (the 7B on an NVIDIA GPU):

```sh
goinfer-serve -model coder=~/models/qwen2.5-7b-instruct-q4_k_m.gguf -quant int4 -backend cuda -ctx 16384
```

- `--kv-sessions N` sets how many conversations stay warm (default 4; 0 turns reuse off). The startup banner says what you got: "session reuse: on (4 conversations kept prefilled)" on CPU, and "session reuse: on the GPU cache, N conversations kept resident" on a GPU model. If the memory check allowed fewer slots than you asked for, the banner gives both numbers.
- To see reuse, send a conversation, then send it again with one more message on the end. The second response's `usage.prefill_reused_tokens` counts the prompt tokens it did not have to read again.
- Keep the start of the prompt identical from turn to turn: system prompt and tool list first, new content last.
- On Metal, an explicit `--embed-int4` sends a load to the CPU, which is why it defaults off there (docs/quantization.md). The Metal records above were measured without it.
- To repeat the replay from a checkout of the repo, with a CUDA build of the server and the 7B at `~/models/qwen2.5-7b-instruct-q4_k_m.gguf`: `GOINFER_SERVE_CUDA=<path to the server binary> python3 scripts/bench_peer_transcript.py out.json --model 7B --backend cuda`.
