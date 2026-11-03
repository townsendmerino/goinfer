---
title: "Batching that doesn't change the answer"
area: "Concurrency"
order: 13
summary: "Up to four conversations share each GPU step, and every reply is the one it would get alone. Measured on Metal, CUDA and CPU; a lone request is unchanged."
stand: "When several conversations are active, goinfer generates their next tokens together in one step. Each conversation gets the same reply it would get running alone."
measured: 2026-09-27
reviewed: 2026-09-29
facts:
  - {label: "Metal, 1.5B, 4 clients", value: "1.593× the one-at-a-time build"}
  - {label: "CUDA, 7B, 4 clients", value: "1.826×"}
  - {label: "CPU, 7B, 4 clients", value: "2.19× the unbatched CPU path"}
  - {label: "a lone request", value: "unchanged"}
doesnt:
  - title: "It isn't continuous batching or paged attention."
    text: "Those are the techniques large serving engines use to pack many requests into GPU memory. Here, four conversations run at once by default, each in its own full-size KV cache slot: about 117 MB for the 1.5B at a 4k-token context on Metal. The rest wait in a bounded queue (`--max-queue`, default 8); past that the server answers HTTP 429. goinfer's own docs say it is not a serving engine."
  - title: "A lone request is not faster."
    text: "With one conversation active, its tokens take the same code path as before. The records measure 1.002× and 1.004× on Metal, and 1.000× and 1.000× on the CPU, for the median turn and the slowest (99th-percentile) turn. That is the goal, not a gain."
  - title: "It doesn't cover every model."
    text: "Plain dense models batch. Mixture-of-experts (MoE) models, recurrent and hybrid models (such as Gated DeltaNet), vision models, LoRA adapters and speculative decoding keep one generation at a time, and WebGPU runs one at a time. On Metal, an explicit `--embed-int4` sends a load to the CPU, where it does not batch; it defaults off there (docs/quantization.md)."
  - title: "Two clients gain little, and small models on the CPU are not batched."
    text: "At two clients on Metal, the 7B gained only 1.018× over one-at-a-time serving. A later change lifted that to 1.121× over the build before it. On the CPU, forcing batching on for a small model (0.5B) read 0.73× the unbatched path, so `--cpu-batch auto` does not batch models under 2 GiB."
  - title: "Identical means identical on one machine."
    text: "The guarantee is that batching does not change a reply on that machine and build. goinfer's positioning page says bit-identity (the same bits, not just the same words) holds within a machine and OS version, not between them."
figures:
  - {text: "1.593", source: "docs/measurements/concurrency-mc3-2026-09-26.md"}
  - {text: "117 MB", source: "docs/server.md"}
  - {text: "25.4", source: "docs/measurements/concurrency-mc3-7b-w7-2026-09-27.md"}
  - {text: "76.2", source: "docs/measurements/concurrency-mc3-2026-09-26.md"}
  - {text: "121.5", source: "docs/measurements/concurrency-mc3-2026-09-26.md"}
  - {text: "7.09", source: "docs/measurements/concurrency-mc3-2026-09-26.md"}
  - {text: "4.67", source: "docs/measurements/concurrency-mc3-2026-09-26.md"}
  - {text: "1.002", source: "docs/measurements/concurrency-mc3-2026-09-26.md"}
  - {text: "1.004", source: "docs/measurements/concurrency-mc3-2026-09-26.md"}
  - {text: "macOS 26.6.2", source: "docs/measurements/concurrency-mc3-2026-09-26.md"}
  - {text: "1.785", source: "docs/measurements/concurrency-mc3-7b-w7-2026-09-27.md"}
  - {text: "45.5", source: "docs/measurements/concurrency-mc3-7b-w7-2026-09-27.md"}
  - {text: "0.592", source: "docs/measurements/concurrency-mc3-7b-w7-2026-09-27.md"}
  - {text: "1.018", source: "docs/measurements/concurrency-mc3-7b-w7-2026-09-27.md"}
  - {text: "1.121", source: "docs/measurements/concurrency-mc3-s4-2026-09-27.md"}
  - {text: "153.1", source: "docs/measurements/concurrency-mc3-s3-2026-09-27.md"}
  - {text: "117.8", source: "docs/measurements/concurrency-mc3-s3-2026-09-27.md"}
  - {text: "1.261", source: "docs/measurements/concurrency-mc3-s3-2026-09-27.md"}
  - {text: "1.380", source: "docs/measurements/concurrency-mc3-cuda-2026-09-27.md"}
  - {text: "218.0", source: "docs/measurements/concurrency-mc3-cuda-2026-09-27.md"}
  - {text: "300.6", source: "docs/measurements/concurrency-mc3-cuda-2026-09-27.md"}
  - {text: "1.826", source: "docs/measurements/concurrency-mc3-cuda-2026-09-27.md"}
  - {text: "74.03", source: "docs/measurements/concurrency-mc3-cuda-2026-09-27.md"}
  - {text: "135.16", source: "docs/measurements/concurrency-mc3-cuda-2026-09-27.md"}
  - {text: "99.8%", source: "docs/measurements/concurrency-mc3-cuda-2026-09-27.md"}
  - {text: "595.91.07", source: "docs/measurements/concurrency-mc3-cuda-2026-09-27.md"}
  - {text: "2.19", source: "docs/measurements/concurrency-mc3c-step2-2026-09-27.md"}
  - {text: "5.44", source: "docs/measurements/concurrency-mc3c-step2-2026-09-27.md"}
  - {text: "11.92", source: "docs/measurements/concurrency-mc3c-step2-2026-09-27.md"}
  - {text: "1.000", source: "docs/measurements/concurrency-mc3c-step2-2026-09-27.md"}
  - {text: "0.73", source: "docs/measurements/concurrency-mc3c-step2-2026-09-27.md"}
sources:
  - "docs/measurements/concurrency-mc3-2026-09-26.md"
  - "docs/measurements/concurrency-mc3-cuda-2026-09-27.md"
  - "docs/measurements/concurrency-mc3c-step2-2026-09-27.md"
  - "docs/measurements/concurrency-mc3-7b-w7-2026-09-27.md"
  - "docs/measurements/concurrency-mc3-s3-2026-09-27.md"
  - "docs/measurements/concurrency-mc3-s4-2026-09-27.md"
  - "docs/tasks/task-concurrency-2026-09.md"
  - "docs/server.md"
  - "docs/positioning.md"
---

## The problem

A model reads all of its weights to produce one token. On a GPU that read is most of the cost, so a step that produces one token for one conversation leaves most of the hardware's arithmetic idle.

If two people chat with the same server, the plain answer is to serve one, then the other. On a MacBook Pro (M1 Pro, 16 GB) with Qwen2.5-Coder 1.5B, four clients served that way got 76.2 tok/s (tokens per second) in total, and the slowest turns (the 99th percentile) took 7.09 s. Each person waits behind three others.

The obvious fix is to put every conversation's next token into one step. The risk is the reply. Batched matrix code often adds numbers in a different order. That can change the last bits of a logit (the model's score for a candidate next token), and so change which token wins. A server that answers differently depending on who else is connected is hard to trust.

## What goinfer does

With several conversations active, goinfer generates their next tokens in one shared step. Each conversation's reply is the same token sequence it would get if it ran alone. On Metal and CUDA this is on by default for a dense model (the usual kind, where every weight is used for every token) held on the GPU, up to four conversations at once. On the CPU, the default `--cpu-batch auto` does the same for models with at least 2 GiB of weights.

You can see it work without changing any client. Start a server, send four requests at the same time, and read the concurrency line the server prints after load. The model file is Qwen2.5-Coder 1.5B in GGUF format, quantized to about 4 bits per weight (q4_k_m). The request bodies are placeholders.

```sh
goinfer-serve -model bench=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -backend metal --embed-int4=false
# banner: concurrency: 4 generations at once, each on its own resident KV slot, decode tokens batched (-max-concurrent)

for p in "Name a fruit." "Name a river." "Name a planet." "Name a metal."; do
  curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"bench\",\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":64,\"temperature\":0}" &
done; wait
```

With `temperature` 0 (greedy: always take the top-scoring token), each reply matches the reply that same request gets when sent alone.

## How it works

Each running conversation holds its own slot on the GPU for its KV cache, the model's stored state for the tokens it has already read. So its history stays on the GPU between turns; [Turn nine in under half a second](/different/09-turn-nine-in-under-half-a-second/) covers that reuse. A coordinator collects the tokens that are ready at a token boundary. When at least two are waiting, it runs them as one batched step, and it waits a few milliseconds for stragglers. When only one is waiting, it calls the ordinary single-sequence code. That is why a lone request does not change.

On Metal, inside a batched step, the per-sequence work (normalisation, position encoding, the KV write and attention) runs the same GPU kernels a single request runs. The big matrix multiplies run once for all rows, with kernels tested to reproduce the single-sequence arithmetic bit for bit. The Metal record's test runs four sequences for twelve steps against the ordinary one-at-a-time computation and finds no differing logits.

Reading a new prompt into the cache is called prefill. A long prompt that arrives while others are generating is prefilled in 512-token pieces, with a step for the others between pieces, so a newcomer does not stall everyone. On the CPU, concurrent generations join their next tokens into one batched step, so each weight is read once for all of them.

## What was measured

Each row is a 4-client run of the same workload (described below). It compares the new build with the previous one-at-a-time build. The figure is the median of three pairs of runs, alternating old and new builds, unless noted.

| Machine | Model | Ratio at 4 clients | Total tok/s, old to new | Record |
|---|---|---|---|---|
| MacBook Pro, M1 Pro 16 GB, macOS 26.6.2 (Metal) | Qwen2.5-Coder 1.5B q4_k_m | 1.593× | 76.2 to 121.5 | [2026-09-26](https://github.com/townsendmerino/goinfer/blob/main/docs/measurements/concurrency-mc3-2026-09-26.md) |
| same MacBook (Metal) | Qwen2.5 7B Instruct q4_k_m | 1.785× | 25.4 to 45.5 | [2026-09-27](https://github.com/townsendmerino/goinfer/blob/main/docs/measurements/concurrency-mc3-7b-w7-2026-09-27.md) |
| Linux PC (Ryzen 7 3700X, RTX 2070 SUPER 8 GB), NVIDIA driver 595.91.07 (CUDA) | Qwen2.5-Coder 1.5B q4_k_m | 1.380× | 218.0 to 300.6 | [2026-09-27](https://github.com/townsendmerino/goinfer/blob/main/docs/measurements/concurrency-mc3-cuda-2026-09-27.md) |
| same Linux PC (CUDA), one pair, reported but not held to the bars | Qwen2.5 7B Instruct q4_k_m | 1.826× | 74.03 to 135.16 | same record |
| same Linux PC, Ryzen 7 3700X (CPU), against the earlier unbatched CPU path | Qwen2.5 7B Instruct q4_k_m | 2.19× | 5.44 to 11.92 | [2026-09-27](https://github.com/townsendmerino/goinfer/blob/main/docs/measurements/concurrency-mc3c-step2-2026-09-27.md) |

The workload is the repo's [chat benchmark script](https://github.com/townsendmerino/goinfer/blob/main/scripts/bench_w7_plain.py). Each client holds a six-turn conversation of 128 greedy tokens per turn, and every measurement starts a fresh server. The pass bars were set in writing before the timed runs ([pre-registered](/different/16-numbers-with-receipts/)). They required identical replies on every turn, the same reuse of each conversation's cached history in both builds, at least 1.2× the old build's total throughput, a slowest turn no worse than the old build's, and a lone request within 1.05×. All passed. On the Metal 1.5B measurement the slowest turns (99th percentile) fell from 7.09 s to 4.67 s, and on the Metal 7B to 0.592× the old build's. The CUDA record counts the steps: at 4 clients, 99.8% of generated tokens ran in steps that carried all four conversations.

Two things the table does not settle. First, the Metal 7B row was measured on a later build than the Metal 1.5B row. That build adds batching for sampled requests (temperature above 0), a cheaper batched step and the chunked prefill above. The 1.5B has a later figure of its own, 153.1 tok/s against the previous build's 117.8 to 121.6 (1.261×). But that compares two batched builds, not batched against one-at-a-time, so it is not multiplied into the 1.593× row. The newest direct 1.5B ratio against one-at-a-time serving is the 1.593× record.

Second, the CPU row's baseline is the earlier CPU path, whose workers already ran conversations concurrently, each unbatched. It is not one-at-a-time serving.

## Use it

- `-max-concurrent N` (default 4) sets how many generations run at once. `1` restores strict one-at-a-time serving. It is capped by `-kv-sessions` (default 4), which also sets the number of GPU KV cache slots.
- `-cpu-batch auto|on|off` (default `auto`) batches CPU generations, for models of at least 2 GiB of weights.
- `-prefill-chunk N` (default 512; 0 prefills whole) sets the piece size for a newcomer's prompt.
- `-max-queue N` (default 8) bounds each model's waiting requests. A full queue answers HTTP 429.
- On Metal, an explicit `--embed-int4` sends a load to the CPU, where it does not batch, so it defaults off there ([docs/quantization.md](https://github.com/townsendmerino/goinfer/blob/main/docs/quantization.md)). From 2026-09-28 to 2026-09-30 the default was on, and a plain Metal load did not batch. CUDA and WebGPU keep the int4 table resident (checked 2026-09-30).
- The banner's concurrency line says which mode is running.
- To reproduce a row, run the benchmark from a clone of the repo against a server binary, once per build: `GOINFER_SERVE_CPU=<serve binary> python3 scripts/bench_w7_plain.py out.json --clients 4 --engines goinfer --backend metal --fixed-nonce`. Use `--backend cuda` or `--backend cpu` for the other rows.
