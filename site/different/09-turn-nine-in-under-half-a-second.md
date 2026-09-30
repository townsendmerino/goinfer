---
title: "Turn nine in under half a second"
area: "Speed"
order: 9
summary: "goinfer keeps several conversations prefilled, so a later turn prefills only what is new: on one CUDA replay, turn 9 of a coding-agent session began in 319.5 ms."
stand: "In a long conversation, each new turn only has to process what you just added. goinfer keeps up to four conversations prefilled at once, so the history is not read again."
measured: 2026-09-27
reviewed: 2026-09-29
facts:
  - {label: "kept prefilled", value: "4 conversations by default (--kv-sessions)"}
  - {label: "turn 1, cold, 7B on CUDA", value: "651.2 ms to first token"}
  - {label: "turn 9, warm, same session", value: "319.5 ms"}
  - {label: "turn off", value: "--kv-sessions 0"}
doesnt:
  - title: "It doesn't keep every conversation."
    text: "The default is four, and the least recently used one is evicted for a fifth. Each kept conversation costs its own KV memory (about 117 MB for the 1.5B at a 4k context on Metal, 896 MB per slot for the 7B at 8192 on CUDA), and the memory guard clamps the count to what fits. On an 8 GB card the 7B at 8192 got 2 of 4 slots, and four clients taking turns read 55.8 against 55.5 tok/s, no better than one slot."
  - title: "It doesn't reuse text that changed."
    text: "Only the shared start of the prompt is reused. Edit an early message, change the system prompt or the tool list, and everything after the change is prefilled again. A client that rewrites the history between turns gets no reuse. The cache also lives in the server process: a restart starts cold unless --session-dir is set, and the docs describe that flag for the CPU session cache, not for the GPU slots."
  - title: "It doesn't help every model family equally."
    text: "Recurrent and hybrid families (Gated DeltaNet, Mamba-2, LFM2) keep one slot, and can only continue from exactly where the last turn ended. An edited message, a shorter prompt or an identical resend prefills from scratch. Their state cannot be rewound."
  - title: "On CPU, a warm turn is not always bit-identical to a cold one."
    text: "With the default fast prefill attention, a new suffix of 512 tokens or more is prefilled by a kernel whose arithmetic is not split-invariant, so at temperature 0 a near-tie token can differ from a one-shot run of the same prompt. GOINFER_CPU_FAST_ATTENTION=0 makes reuse exact, at a cost the docs measure at 1.43x on a cold 2048-token turn."
  - title: "The turn-nine figure is one run on one box."
    text: "It is a single replay on a 7B model on CUDA, on a tree the record marks dirty, timed as a one-token request. The Metal records count tokens prefilled and whole-turn time, not first-token time. A bigger agent prompt can miss half a second: an earlier record's second turn, with 25 tool schemas, read 0.58 s."
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

An earlier goinfer record (2026-09-02) shows the shape on a 7B model on CUDA, with 25 tool schemas in the prompt. With reuse switched off, turns 1 to 3 took 8.86 s, 9.01 s and 9.13 s. With it on, they took 8.80 s, 0.58 s and 0.42 s.

## What goinfer does

goinfer keeps recent conversations prefilled. When a new request arrives, it finds the conversation whose stored tokens are the start of the new prompt and processes only the tokens after that.

The record for this page replays a ten-turn coding-agent session (read a file, grep, edit, run tests, and so on) against a 7B model on an RTX 2070 SUPER. Turn 1 has 727 prompt tokens and nothing to reuse: first token in 651.2 ms. By turn 9 the prompt is 1922 tokens, 1668 of them already in the GPU's KV cache, and the first token arrives in 319.5 ms. Turns 2 to 10 span 191.2 to 319.5 ms.

## How it works

On the CPU path, a small list of sessions holds each conversation's tokens and KV cache. A new prompt takes the session that shares the longest prefix with it, provided that beats what the session shares with the other sessions, so a common system-prompt lead does not hijack someone else's conversation. The session is rewound to the shared point and only the rest is prefilled.

A GPU-resident model (Metal, CUDA, WebGPU) has no such list. Instead it holds up to `--kv-sessions` KV slots on the device. A request binds the slot that already holds its prefix. A new conversation takes an empty slot, or else the least recently used one.

Either way, the response's `usage.prefill_reused_tokens` says how many prompt tokens were skipped.

## What was measured

The first table is the source of the title: the agent-turn replay on 2026-09-09, on `nobara-pc` (RTX 2070 SUPER, driver 595.91.07), Qwen2.5-7B-Instruct Q4_K_M as int4, CUDA resident, greedy. "Time to first token" is the wall clock of a one-token request, because forced tool calls do not stream.

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

The second measurement asks whether several conversations at once still get their own history. Six plain turns per client, greedy, on Metal (M1 Pro, 16 GB, Qwen2.5-Coder 1.5B int4, 2026-09-26) and on CUDA (`nobara-pc`, same model, 2026-09-27). The figure is how many tokens each of turns 2 to 6 prefilled. With four slots it was 26, 27, 24, 20, 27 at 1, 2 and 4 clients on both backends. The previous build had one slot, and at 4 clients it prefilled 187, 342, 494, 642, 797 tokens on Metal and 176, 331, 483, 631, 786 on CUDA: every turn read the whole history again.

## What we got wrong

**The CPU cache thrashed for three days.** A 2026-09-23 change to the matching rule (`e1c867f6`) let two interleaved conversations take each other's session. Each reused only the 7-token chat-template lead, and the list never grew past one session. The record measured 0.69× the one-client throughput at two clients. It was found on 2026-09-26 by the control run for the GPU work, and the fix took two-client CPU serving from 27.4 to 43.3 tok/s.

**The GPU paths had one slot.** At two clients, Metal reused 7 tokens per turn after the first. Multiple resident slots fixed that.

**One token short.** With `--spec ngram` on Metal, a reply that stopped at `max_tokens` left the cache one token short, so the next turn reused 156 positions where plain decode reused 157. That one token went through the batched prefill kernel, whose arithmetic differs from decode's, and every later turn diverged. It was fixed by forwarding the token at the end of the generation.

## Use it

```sh
goinfer-serve -model coder=~/models/qwen2.5-7b-instruct-q4_k_m.gguf -quant int4 -backend cuda -ctx 16384
```

- `--kv-sessions N` sets how many conversations stay warm (default 4; 0 turns reuse off). The startup banner says what you got: "session reuse: on (4 conversations kept prefilled)" on CPU, "session reuse: on the GPU cache, N conversations kept resident" on a GPU model, with the memory guard's clamp named.
- Keep the start of the prompt identical from turn to turn: system prompt and tool list first, new content last.
- On Metal, pass `--embed-int4=false` for now, or the model decodes on the CPU (docs/quantization.md, "Known issue"). The Metal records above predate that default.
- To repeat the replay: `python3 scripts/bench_peer_transcript.py out.json --model 7B --backend cuda`.
