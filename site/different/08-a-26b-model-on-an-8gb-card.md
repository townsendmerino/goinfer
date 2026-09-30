---
title: "A 26B model on an 8 GB card"
area: "Memory"
order: 8
summary: "Gemma 4 26B-A4B decodes on an 8 GB card by streaming experts from host RAM: 40.2 tok/s on 2026-09-25, an architecture comparison, not like-for-like."
stand: "The 26B mixture-of-experts model does not fit an 8 GB card, so goinfer keeps its experts in host memory and copies the ones each token needs to the GPU. This page gives the one figure measured for that, and what it does not show."
measured: 2026-09-25
reviewed: 2026-09-29
facts:
  - {label: "card", value: "RTX 2070 SUPER, 8 GB, driver 595.91.07"}
  - {label: "model", value: "Gemma 4 26B-A4B, goinfer's int4 .giw bundle"}
  - {label: "decode rate, context 2048", value: "40.2 tok/s on 2026-09-25 (may be replaced)"}
  - {label: "peak host RAM", value: "25.5 GB"}
doesnt:
  - title: "It doesn't fit in a small amount of RAM."
    text: "The experts live in pinned host memory, so the machine needs the RAM for them. The record's peak resident set was 25.5 GB for goinfer, against 17.2 GB for Ollama and 17.0 GB for llama.cpp on the same model and card."
  - title: "It isn't a like-for-like win over Ollama."
    text: "goinfer ran its own int4 .giw bundle. Ollama and llama.cpp ran the Q4_K_M GGUF and placed layers on the CPU. The record labels this an architecture comparison and keeps it out of every family-level claim. An earlier record measured Ollama at about 24.5 tok/s on a different file (Google's q4_0 QAT GGUF), so the gap depends on the checkpoint."
  - title: "It isn't as fast as a card that holds the model."
    text: "The rate is limited by copying experts over PCIe, not by the kernels or by MoE. The registry entry says so: capacity-bound, not a kernel or MoE deficiency. Before the copy was overlapped with compute, the same model measured 16.12 tok/s and 17.62 tok/s."
  - title: "It doesn't work this way on a 16 GB Mac."
    text: "Metal pages experts from disk, and the record's first three attempts on a 16 GB M1 Pro (2026-09-20 and 2026-09-22) each pushed the machine into swap and were killed by hand or by a script. A later run on 2026-09-24, with 8 slots and context 512, decoded at 5.98 to 6.19 tok/s, two runs per arm and not a registered benchmark."
  - title: "It doesn't apply to every mixture-of-experts model."
    text: "The CUDA streaming path takes expert weights held as int4 (other kinds stay fully resident), and only the CUDA and Metal backends take the flag. The code and tests cover Gemma 4, gpt-oss and the Qwen3.6 MoE, but this page has a measured decode rate for Gemma 4 26B-A4B alone. The overlap changes decode only; the record says prefill is untouched."
figures:
  - {text: "40.2", source: "docs/measurements/peer-claim-2026-09-25.md"}
  - {text: "22.2", source: "docs/measurements/peer-claim-2026-09-25.md"}
  - {text: "27.6", source: "docs/measurements/peer-claim-2026-09-25.md"}
  - {text: "1.815", source: "docs/measurements/peer-claim-2026-09-25.md"}
  - {text: "25.5", source: "docs/measurements/peer-claim-2026-09-25.md"}
  - {text: "17.2", source: "docs/measurements/peer-claim-2026-09-25.md"}
  - {text: "17.0", source: "docs/measurements/peer-claim-2026-09-25.md"}
  - {text: "595.91.07", source: "docs/measurements/peer-claim-2026-09-25.md"}
  - {text: "24.5", source: "docs/benchmarks.md"}
  - {text: "16.12", source: "docs/benchmarks.md"}
  - {text: "17.62", source: "docs/benchmarks.md"}
  - {text: "76.1%", source: "docs/benchmarks.md"}
  - {text: "82.2%", source: "docs/benchmarks.md"}
  - {text: "11.4", source: "docs/benchmarks.md"}
  - {text: "714", source: "docs/benchmarks.md"}
  - {text: "30.36", source: "docs/measurements/moe-streaming-decode-overlap-ceiling-2026-09-22.md"}
  - {text: "38.66", source: "docs/measurements/moe-streaming-decode-overlap-ceiling-2026-09-22.md"}
  - {text: "1.271", source: "docs/measurements/moe-streaming-decode-overlap-ceiling-2026-09-22.md"}
  - {text: "5.98", source: "docs/measurements/m26-alias-fork-collapse-2026-09-24.md"}
  - {text: "6.19", source: "docs/measurements/m26-alias-fork-collapse-2026-09-24.md"}
sources:
  - docs/measurements/peer-claim-2026-09-25.md
  - docs/benchmarks.md
  - docs/measurements/moe-streaming-decode-overlap-ceiling-2026-09-22.md
  - docs/measurements/m26-alias-fork-collapse-2026-09-24.md
  - docs/measurements/metal-moe-autopager-m26-2026-09-20.md
  - docs/measurements/peer-sweep-2026-09-29.md
  - docs/capability-matrix.json
---

## The problem

Gemma 4 26B-A4B is a mixture-of-experts model. Each MoE layer holds 128 experts, and each token uses 8 of them, so only about 4 billion of the 26 billion parameters do work on any one token. The int4 experts add up to about 11.4 GB, which does not fit an 8 GB card.

The usual answer is CPU offload: some layers stay in VRAM and the rest run on the CPU. Ollama and llama.cpp do this. goinfer tries another route. Without the flag below, a model that does not fit declines to the CPU path and says why.

## What goinfer does

With `-moe-cache-experts`, the small part of the model that every token needs stays on the GPU. The experts sit in host memory, and each token's routed experts are copied to the GPU before they run. Every expert still executes on the GPU, so there is no CPU compute for them.

The command that produced the number below, from the benchmark harness (`scripts/bench_peer.py`), with the CUDA build of the server:

```sh
goinfer-serve -model bench=$HOME/models/gemma4-26b-int4.giw -backend cuda -moe-cache-experts -ctx 2048
```

## How it works

This is the "C-prime" cache in the code. Per layer, the full set of experts stays in pinned host memory. On the GPU there is a small block of slots, sized at load time to the free VRAM the runtime measures (in one logged load it capped 64 requested slots to 29, because only 3.4 GB was free). Each layer keeps an LRU cache over its slots. The router picks 8 experts. Those already in a slot are used as they are, and the missing ones are copied over PCIe.

Trained routers reuse a few experts far more than the rest, so most picks hit. The earlier record put the hit rate at 76.1% with 30 slots and 82.2% with 40 slots. About 714 MB of experts still crossed PCIe per token before any reuse, and that copy is the wall.

The DMA overlap, on by default, hides part of it. While a missing expert is copying on a second queue, the GPU runs the dense branch and the experts that were already resident. The A/B in `moe-streaming-decode-overlap-ceiling-2026-09-22.md` gave bit-identical logits with the overlap on and off, and a paired median speedup of 1.271×, from 30.36 to 38.66 tok/s pooled. The code turns the overlap off under CUDA graphs.

## What was measured

Machine for the first three rows: nobara-pc, RTX 2070 SUPER 8 GB, driver 595.91.07, Ryzen 7 3700X, Nobara 44. Model: Gemma 4 26B-A4B. Context 2048, greedy decoding, three runs of 8 completions of 64 tokens each, timed from the first streamed token, on 2026-09-25.

| Engine | What it ran | Decode rate |
|---|---|---|
| goinfer | int4 `.giw`, all experts streamed to the GPU | 40.2 tok/s (40.2, 40.2, 40.1) |
| Ollama v0.32.5 | Q4_K_M GGUF, layers offloaded to the CPU | 22.2 tok/s (22.2, 22.2, 22.2) |
| llama.cpp `427291b` | Q4_K_M GGUF, layers offloaded to the CPU | 27.6 tok/s |

The ratio to Ollama is 1.815. **This is an architecture comparison, not a like-for-like one.** goinfer ran its own int4 bundle and the other two ran a different file, and the record keeps it out of any claim about model families. Read it as "these two designs, on this card", not as one engine being faster.

Three cautions from the record. Every completion repeats the same greedy prompt, so the expert cache sees the same routing each time. The rates are flat across completions, but a benefit carried over from the discarded warm-up run cannot be ruled out. The build was made from the pre-registration commit `411e7fc4`. And the figure is dated: a re-run of this cell at one pinned commit is queued in `peer-sweep-2026-09-29.md` and its result will replace this one.

For context, the same model measured 16.12 tok/s at 30 slots and 17.62 tok/s at 40 slots on 2026-08-27, before the overlap. That was a test harness with synchronous copies, not the server, so it is history rather than a comparison.

## Use it

- `-moe-cache-experts` (default off) turns the streaming on. It is a load flag in `goinfer-serve` and in `goinfer-chat`.
- `-moe-cache-slots N` sets the per-layer slot count. On CUDA it is an upper bound. 0 asks for all and lowers itself to fit free VRAM.
- `-ctx N` sets the context. It matters here: on the 2026-08-27 record, halving the context from 4096 to 2048 freed VRAM for more slots.
- `-require-backend` makes the server exit at start-up if a model did not reach the fast GPU path, rather than run on the CPU quietly.
- `GOINFER_MOE_DMA_OVERLAP=0` turns the overlap off, for comparison.
- The README's example loads the GGUF directly: `goinfer-serve -backend cuda -moe-cache-experts -model ~/models/gemma-4-26B_q4_0-it.gguf`. The 40.2 figure was measured on the `.giw` bundle, not that file.
