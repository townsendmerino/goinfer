---
title: "A 26B model on an 8 GB card"
area: "Memory"
order: 8
summary: "Gemma 4 26B-A4B runs on an 8 GB card by streaming its experts from host RAM: 39.3 tokens/s on 2026-09-29. An architecture comparison, not like-for-like."
stand: "The 26B mixture-of-experts model does not fit an 8 GB card, so goinfer keeps its experts in host memory and copies the ones each token needs to the GPU. This page gives the one figure measured for that, and what it does not show."
measured: 2026-09-25
reviewed: 2026-09-29
facts:
  - {label: "card", value: "RTX 2070 SUPER, 8 GB, driver 595.91.07"}
  - {label: "model", value: "Gemma 4 26B-A4B, goinfer's int4 .giw bundle"}
  - {label: "generation rate, 2048-token context", value: "39.3 tok/s on 2026-09-29"}
  - {label: "peak host RAM", value: "20.0 GiB"}
doesnt:
  - title: "It doesn't fit in a small amount of RAM."
    text: "The experts live in pinned host memory, so the machine needs the RAM to hold them. goinfer's peak resident memory in the record was 20.0 GiB, against 16.4 GiB for Ollama and 16.2 GiB for llama.cpp, on the same model and card. The Linux PC it ran on has 62.7 GB of RAM."
  - title: "It isn't a like-for-like win over Ollama."
    text: "goinfer ran its own int4 .giw bundle. Ollama and llama.cpp ran the Q4_K_M GGUF and put some layers on the CPU. The record labels this an architecture comparison and keeps it out of every claim about a model family. An earlier record measured Ollama at about 24.5 tok/s on another file of this model (Google's q4_0 QAT GGUF), so the gap depends on which file is used."
  - title: "Its speed is set by the copy to the card."
    text: "The rate is limited by copying experts over PCIe, not by the kernels or by the mixture-of-experts design. No card large enough to hold the whole model was measured for comparison. The project's capability matrix says so: capacity-bound, not a kernel or MoE deficiency. Before the copy was overlapped with compute, the same model measured 16.12 tok/s and 17.62 tok/s."
  - title: "It doesn't work this way on a 16 GB Mac."
    text: "On Metal, experts are read from disk as they are needed. The first three attempts on a MacBook Pro (M1 Pro, 16 GB), on 2026-09-20 and 2026-09-22, each pushed the machine into swap and were killed by hand or by a script. A later run on the 16 GB MacBook on 2026-09-24, after a fix for a bug that copied the whole weight file into memory, used 8 slots and a 512-token context and generated at 5.98 to 6.19 tok/s. That was four 32-token runs, two for each of two memory settings, with no decision rule set in advance."
  - title: "It doesn't apply to every mixture-of-experts model."
    text: "On CUDA, only experts stored as int4 are streamed; experts in other formats stay fully on the GPU. Only the CUDA and Metal backends take the flag. The code and tests cover Gemma 4, gpt-oss and the Qwen3.6 MoE, but this page has a measured generation rate for Gemma 4 26B-A4B alone. The copy overlap speeds up generation only; the record says prefill (reading the prompt) is untouched."
figures:
  - {text: "595.91.07", source: "docs/measurements/peer-claim-2026-09-25.md"}
  - {text: "39.3", source: "docs/measurements/peer-sweep-2026-09-29.md"}
  - {text: "37.3", source: "docs/measurements/peer-sweep-2026-09-29.md"}
  - {text: "22.3", source: "docs/measurements/peer-sweep-2026-09-29.md"}
  - {text: "1.763", source: "docs/measurements/peer-sweep-2026-09-29.md"}
  - {text: "20.0 GiB", source: "docs/measurements/peer-sweep-2026-09-29.md"}
  - {text: "16.4 GiB", source: "docs/measurements/peer-sweep-2026-09-29.md"}
  - {text: "16.2 GiB", source: "docs/measurements/peer-sweep-2026-09-29.md"}
  - {text: "1.815", source: "docs/measurements/peer-sweep-2026-09-29.md"}
  - {text: "40.2", source: "docs/measurements/peer-sweep-2026-09-29.md"}
  - {text: "27.7", source: "docs/measurements/peer-sweep-2026-09-29.md"}
  - {text: "22.2", source: "docs/measurements/peer-claim-2026-09-25.md"}
  - {text: "24.5", source: "docs/benchmarks.md"}
  - {text: "16.12", source: "docs/benchmarks.md"}
  - {text: "17.62", source: "docs/benchmarks.md"}
  - {text: "76.1%", source: "docs/benchmarks.md"}
  - {text: "82.2%", source: "docs/benchmarks.md"}
  - {text: "11.4", source: "docs/benchmarks.md"}
  - {text: "714", source: "docs/benchmarks.md"}
  - {text: "62.7 GB", source: "docs/benchmarks.md"}
  - {text: "30.36", source: "docs/measurements/moe-streaming-decode-overlap-ceiling-2026-09-22.md"}
  - {text: "38.66", source: "docs/measurements/moe-streaming-decode-overlap-ceiling-2026-09-22.md"}
  - {text: "1.271", source: "docs/measurements/moe-streaming-decode-overlap-ceiling-2026-09-22.md"}
  - {text: "5.98", source: "docs/measurements/m26-alias-fork-collapse-2026-09-24.md"}
  - {text: "6.19", source: "docs/measurements/m26-alias-fork-collapse-2026-09-24.md"}
sources:
  - docs/measurements/peer-sweep-2026-09-29.md
  - docs/measurements/peer-claim-2026-09-25.md
  - docs/benchmarks.md
  - docs/measurements/moe-streaming-decode-overlap-ceiling-2026-09-22.md
  - docs/measurements/m26-alias-fork-collapse-2026-09-24.md
  - docs/measurements/metal-moe-autopager-m26-2026-09-20.md
  - docs/measurements/peer-sweep-2026-09-29.md
  - docs/capability-matrix.json
  - README.md
---

## The problem

Gemma 4 26B-A4B is a mixture-of-experts (MoE) model. Each MoE layer holds 128 small sub-networks, called experts, and a router picks 8 of them for each token. So only about 4 billion of the 26 billion parameters do work on any one token. Stored as int4 (4-bit weights), the experts add up to about 11.4 GB. That does not fit an 8 GB card.

The usual answer is CPU offload: some layers stay in the card's memory (VRAM) and the rest run on the CPU. Ollama and llama.cpp do this. goinfer takes another route. Without the flag below, a model that does not fit runs on the CPU instead, and goinfer says why.

## What goinfer does

With `-moe-cache-experts`, the small part of the model that every token needs stays on the GPU. The experts sit in host memory, and the experts each token needs are copied to the GPU before they run. Every expert still runs on the GPU, so no expert work is done on the CPU.

This is the command that produced the number below. The benchmark script ([`scripts/bench_peer.py`](https://github.com/townsendmerino/goinfer/blob/main/scripts/bench_peer.py)) starts it, using the CUDA build of the server:

```sh
goinfer-serve -model bench=$HOME/models/gemma4-26b-int4.giw -backend cuda -moe-cache-experts -ctx 2048
```

`bench=` is the name that requests use for the model. The `.giw` file is goinfer's own int4 bundle of the model: weights converted ahead of time into the layout goinfer reads (the page [Starts without converting the model again](/different/18-starts-in-a-hundredth/) explains the format). The record does not say how this particular file was built.

## How it works

Each layer keeps its full set of experts in pinned (page-locked) host memory, which the GPU can copy from directly. On the GPU there is a small block of slots. It is sized at load time to the free VRAM the runtime measures. In one logged load it lowered 64 requested slots to 29, because only 3.4 GB was free. Each layer keeps an LRU cache over its slots. The router picks 8 experts. Those already in a slot are used as they are, and the missing ones are copied over PCIe.

Trained routers pick a few experts far more often than the rest, so most picks are already in a slot. An earlier record, from 2026-08-27, put that hit rate at 76.1% with 30 slots and 82.2% with 40 slots. The experts one token routes to add up to about 714 MB. Every one that is not in a slot is copied over PCIe, and that copy is the wall.

A copy overlap, on by default, hides part of it. While a missing expert copies on a second GPU queue (a CUDA stream), the GPU runs the layer's dense branch (the part every token uses) and the experts that are already in slots. A comparison on 2026-09-22, on the same card and model file, switched the overlap on and off between 48-token generations in one process, for 8 pairs. The output scores (logits) were bit-identical with it on and off. The median speedup over the pairs was 1.271×, and the rate over all runs went from 30.36 to 38.66 tokens per second (tok/s). Those runs fed the model synthetic input instead of a prompt, so they measure the effect of the overlap, not a served rate. The overlap is off when CUDA graphs are on (an optional mode, off by default).

## What was measured

- **Machine:** a Linux PC (Ryzen 7 3700X, RTX 2070 SUPER 8 GB, NVIDIA driver 595.91.07), running Nobara 44.
- **Model:** Gemma 4 26B-A4B. **Date:** 2026-09-29 (a re-run of the 2026-09-25 measurement at one build, `754f12d3`).
- **How:** each engine ran as a server, with a 2048-token context and a 128-token prompt. Decoding was greedy: the model takes its most likely token every time, so a run repeats exactly. One run was 8 completions of 64 tokens each. The rate is in tok/s, timed from the first streamed token. Each engine did three runs, taken in turn with the other engines.

| Engine | What it ran | Generation rate |
|---|---|---|
| goinfer | int4 `.giw`, all experts streamed to the GPU | 39.3 tok/s (39.4, 39.3, 37.3) |
| Ollama v0.32.5 | Q4_K_M GGUF, layers offloaded to the CPU | 22.3 tok/s (22.3, 22.3, 22.3) |
| llama.cpp build 427291b | Q4_K_M GGUF, layers offloaded to the CPU | 27.7 tok/s |

The ratio to Ollama is 1.763, the median of the three pairs of runs, and every pair was above 1.67. It is graded ambiguous rather than ahead: goinfer's own three runs spread 5.3% (its third read 37.3), over the 5% bound set before the run. On 2026-09-25 the same measurement read 40.2 tok/s against 22.2 (1.815). **This is an architecture comparison, not a like-for-like one.** goinfer ran its own int4 bundle. The other two ran a different file, the Q4_K_M GGUF (a common 4-bit format). The record keeps this result out of every claim it makes about a model family. Read it as "these two designs, on this card", not as one engine being faster.

Three cautions from the record:

- Every completion repeats the same greedy prompt, so the expert cache sees the same routing each time. The rates are flat across completions, but a benefit carried over from the discarded warm-up run cannot be ruled out.
- The goinfer server was built from commit [`411e7fc4`](https://github.com/townsendmerino/goinfer/commit/411e7fc4). That is the commit where the plan for this run, and the rules for reading its result, were set in writing before the run (pre-registered). [Numbers with their receipts](/different/16-numbers-with-receipts/) explains the method.
- The figure is dated. A re-run of this measurement, part of a re-run of the whole comparison table at one commit, is queued in [the 2026-09-29 sweep plan](https://github.com/townsendmerino/goinfer/blob/main/docs/measurements/peer-sweep-2026-09-29.md), and its result will replace this one.

For context, the same model measured 16.12 tok/s at 30 slots and 17.62 tok/s at 40 slots on 2026-08-27, before the overlap. That was a Go test, not the server, and the GPU waited for every copy to finish. It is history, not a comparison.

## Use it

- `-moe-cache-experts` (off by default) turns the streaming on. It is a load flag in `goinfer-serve` and in `goinfer-chat`.
- `-moe-cache-slots N` sets the number of slots per layer. On CUDA it is an upper bound. 0, the default, asks for all of them and lowers the count to fit free VRAM.
- `-ctx N` sets the context length in tokens. It matters here, because the KV cache (the attention state kept for every token in the context) takes VRAM before the slots are sized. On the 2026-08-27 record, halving the context from 4096 to 2048 raised the slots from 30 to 40.
- `-require-backend` makes the server exit at start-up if a model did not reach the fast GPU path, rather than run on the CPU quietly.
- `GOINFER_MOE_DMA_OVERLAP=0`, set before the server starts, turns the overlap off, for comparison.
- Once the server is up, send a request: `curl localhost:8080/v1/chat/completions -d '{"model":"bench","messages":[{"role":"user","content":"hi"}]}'`.
- The README uses Google's q4_0 GGUF instead. It fetches the file with `goinfer-chat pull gemma-4-26b-a4b` and serves it with `goinfer-serve -backend cuda -moe-cache-experts -model ~/models/gemma-4-26B_q4_0-it.gguf`. The 39.3 figure was measured on the `.giw` bundle, not on that file.
