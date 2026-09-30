---
title: "Batching that doesn't change the answer"
area: "Concurrency"
order: 13
summary: "Up to four conversations share each GPU step, and every reply is the one it would get alone. Measured on Metal, CUDA and CPU; a lone request is unchanged."
stand: "When several conversations are active, goinfer decodes their next tokens together in one step. Each conversation gets the same reply it would get running alone."
measured: 2026-09-27
reviewed:
facts:
  - {label: "Metal, 1.5B, 4 clients", value: "1.593× the one-at-a-time build"}
  - {label: "CUDA, 7B, 4 clients", value: "1.826×"}
  - {label: "CPU, 7B, 4 clients", value: "2.19× the workers"}
  - {label: "a lone request", value: "unchanged"}
doesnt:
  - title: "It isn't continuous batching or paged attention."
    text: "Four conversations run at once, each in its own full-size KV slot (about 117 MB for the 1.5B at a 4k context on Metal). The rest wait in a bounded queue (--max-queue, default 8, then a 429). goinfer's own docs say it is not a serving engine."
  - title: "A lone request is not faster."
    text: "With one conversation active, its tokens take the same code path as before. The record measures 1.002× and 1.004× on Metal, and 1.000× and 1.000× on CPU, for the median and slowest turn. That is the goal, not a gain."
  - title: "It doesn't cover every model."
    text: "Plain dense families batch. MoE, recurrent and hybrid families (Gated DeltaNet), vision models, adapters and speculative decoding keep one generation at a time, and WebGPU runs one at a time. On Metal, the default --embed-int4 currently sends a model to the CPU, so pass --embed-int4=false (docs/quantization.md, \"Known issue\")."
  - title: "The gain shrinks with fewer clients and bigger models."
    text: "At two clients on Metal, the 7B gained only 1.018× over one-at-a-time serving. A later change lifted that to 1.121× over the build before it. On a small CPU model, forcing batching on read 0.73× the plain workers, so --cpu-batch auto leaves models under 2 GiB on the workers."
  - title: "Identical means identical on one machine."
    text: "The guarantee is that batching does not change a reply on that machine and build. goinfer's positioning page says bit-identity holds within a machine and OS version, not between them."
figures:
  - {text: "1.593", source: docs/measurements/concurrency-mc3-2026-09-26.md}
  - {text: "117 MB", source: docs/server.md}
  - {text: "25.4", source: docs/measurements/concurrency-mc3-7b-w7-2026-09-27.md}
  - {text: "76.2", source: docs/measurements/concurrency-mc3-2026-09-26.md}
  - {text: "121.5", source: docs/measurements/concurrency-mc3-2026-09-26.md}
  - {text: "7.09", source: docs/measurements/concurrency-mc3-2026-09-26.md}
  - {text: "4.67", source: docs/measurements/concurrency-mc3-2026-09-26.md}
  - {text: "1.002", source: docs/measurements/concurrency-mc3-2026-09-26.md}
  - {text: "1.004", source: docs/measurements/concurrency-mc3-2026-09-26.md}
  - {text: "macOS 26.6.2", source: docs/measurements/concurrency-mc3-2026-09-26.md}
  - {text: "1.785", source: docs/measurements/concurrency-mc3-7b-w7-2026-09-27.md}
  - {text: "45.5", source: docs/measurements/concurrency-mc3-7b-w7-2026-09-27.md}
  - {text: "0.592", source: docs/measurements/concurrency-mc3-7b-w7-2026-09-27.md}
  - {text: "1.018", source: docs/measurements/concurrency-mc3-7b-w7-2026-09-27.md}
  - {text: "1.121", source: docs/measurements/concurrency-mc3-s4-2026-09-27.md}
  - {text: "153.1", source: docs/measurements/concurrency-mc3-s3-2026-09-27.md}
  - {text: "117.8", source: docs/measurements/concurrency-mc3-s3-2026-09-27.md}
  - {text: "1.261", source: docs/measurements/concurrency-mc3-s3-2026-09-27.md}
  - {text: "1.380", source: docs/measurements/concurrency-mc3-cuda-2026-09-27.md}
  - {text: "218.0", source: docs/measurements/concurrency-mc3-cuda-2026-09-27.md}
  - {text: "300.6", source: docs/measurements/concurrency-mc3-cuda-2026-09-27.md}
  - {text: "1.826", source: docs/measurements/concurrency-mc3-cuda-2026-09-27.md}
  - {text: "74.03", source: docs/measurements/concurrency-mc3-cuda-2026-09-27.md}
  - {text: "135.16", source: docs/measurements/concurrency-mc3-cuda-2026-09-27.md}
  - {text: "99.8%", source: docs/measurements/concurrency-mc3-cuda-2026-09-27.md}
  - {text: "595.91.07", source: docs/measurements/concurrency-mc3-cuda-2026-09-27.md}
  - {text: "2.19", source: docs/measurements/concurrency-mc3c-step2-2026-09-27.md}
  - {text: "5.44", source: docs/measurements/concurrency-mc3c-step2-2026-09-27.md}
  - {text: "11.92", source: docs/measurements/concurrency-mc3c-step2-2026-09-27.md}
  - {text: "1.000", source: docs/measurements/concurrency-mc3c-step2-2026-09-27.md}
  - {text: "0.73", source: docs/measurements/concurrency-mc3c-step2-2026-09-27.md}
sources:
  - docs/measurements/concurrency-mc3-2026-09-26.md
  - docs/measurements/concurrency-mc3-cuda-2026-09-27.md
  - docs/measurements/concurrency-mc3c-step2-2026-09-27.md
  - docs/measurements/concurrency-mc3-7b-w7-2026-09-27.md
  - docs/measurements/concurrency-mc3-s3-2026-09-27.md
  - docs/measurements/concurrency-mc3-s4-2026-09-27.md
  - docs/tasks/task-concurrency-2026-09.md
  - docs/server.md
  - docs/positioning.md
---

## The problem

A model reads all of its weights to produce one token. On a GPU that read is most of the cost, so a step that produces one token for one conversation leaves most of the hardware's arithmetic idle.

If two people chat with the same server, the plain answer is to serve one, then the other. On the MacBook (an M1 Pro), four clients served that way read 76.2 tok/s in total, and the slowest turns (the 99th percentile) took 7.09 s. Each person waits behind three others.

The obvious fix is to put every conversation's next token into one step. The risk is the reply. Batched matrix code often adds numbers in a different order, which can change the last bits of a logit and so a token. A server that answers differently depending on who else is connected is hard to trust.

## What goinfer does

With several conversations active, goinfer decodes their next tokens in one shared step, and each conversation's reply is the same token sequence it would get if it ran alone. On Metal and CUDA this is on by default for a dense resident model, up to four at once.

You can see it work without changing any client. Start a server, send four requests at the same time, and read the concurrency line the server prints after load. The request bodies here are placeholders.

```sh
goinfer-serve -model bench=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -backend metal --embed-int4=false
# banner: concurrency: 4 generations at once, each on its own resident KV slot, decode tokens batched (-max-concurrent)

for p in "Name a fruit." "Name a river." "Name a planet." "Name a metal."; do
  curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"bench\",\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":64,\"temperature\":0}" &
done; wait
```

With `temperature` 0, each reply matches the reply that same request gets when sent alone.

## How it works

Each running conversation holds its own KV slot on the GPU, so its history stays resident between turns. A coordinator collects the tokens that are ready at a token boundary. When at least two are waiting, it runs them as one batched step, and it waits a few milliseconds for stragglers. When only one is waiting, it calls the ordinary single-sequence code, which is why a lone request does not change.

On Metal, inside a batched step, the per-sequence work (normalisation, position encoding, the KV write and attention) runs kernels whose code is the single-request kernels' own. The big matrix multiplies run once for all rows, with kernels tested to reproduce the single-sequence arithmetic bit for bit. The Metal record's test runs four sequences for twelve steps against the ordinary forward pass and finds no differing logits.

Long prompts that arrive while others are decoding are prefilled in 512-token pieces, with a decode step between pieces, so a newcomer does not stall everyone. On the CPU, concurrent generations join their decode tokens into one batched forward, so each weight is read once for all of them.

## What was measured

Every row is a 4-client run of the same workload (below), the new build against the previous one-at-a-time build, median of three interleaved pairs unless noted.

| Machine | Model | Ratio at 4 clients | Aggregate tok/s | Record (date) |
|---|---|---|---|---|
| MacBook Pro, M1 Pro 16 GB, macOS 26.6.2 (Metal) | Qwen2.5-Coder 1.5B q4_k_m | 1.593× | 76.2 to 121.5 | `concurrency-mc3-2026-09-26.md` (2026-09-26) |
| same MacBook (Metal) | Qwen2.5 7B Instruct q4_k_m | 1.785× | 25.4 to 45.5 | `concurrency-mc3-7b-w7-2026-09-27.md` (2026-09-27) |
| nobara, RTX 2070 SUPER 8 GB, driver 595.91.07 (CUDA) | Qwen2.5-Coder 1.5B q4_k_m | 1.380× | 218.0 to 300.6 | `concurrency-mc3-cuda-2026-09-27.md` (2026-09-27) |
| same nobara (CUDA), one pair, reported not gated | Qwen2.5 7B Instruct q4_k_m | 1.826× | 74.03 to 135.16 | same record |
| nobara, Ryzen 7 3700X (CPU), against the earlier workers | Qwen2.5 7B Instruct q4_k_m | 2.19× | 5.44 to 11.92 | `concurrency-mc3c-step2-2026-09-27.md` (2026-09-27) |

The workload is `scripts/bench_w7_plain.py`: each client holds a six-turn conversation of 128 greedy tokens per turn, against a fresh server for every cell. The gates were written before the timed runs. They required identical replies on every turn, identical prefix reuse, at least a 1.2× aggregate, a slowest turn no worse than the old build's, and a lone request within 1.05×. All passed. On the Metal 1.5B cell the slowest turns (99th percentile) fell from 7.09 s to 4.67 s, and on the Metal 7B to 0.592× the old build's. The CUDA record counts the steps: at 4 clients, 99.8% of decode tokens ran in four-row steps.

Two things the table does not settle. The 7B Metal row was measured on a later build than the 1.5B row: it includes sampled-token batching, a cheaper step and chunked prefill. The 1.5B has a later figure of its own, 153.1 tok/s against the previous build's 117.8 to 121.6 (1.261×), but that compares two batched builds, not the one-at-a-time build, so it is not multiplied into the 1.593× row. The newest direct 1.5B ratio against one-at-a-time serving is the 1.593× record.

The CPU row's baseline is workers that already ran concurrently, not one-at-a-time serving.

## Use it

- `-max-concurrent N` (default 4) sets how many generations run at once. `1` restores strict one-at-a-time serving. It is capped by `-kv-sessions` (default 4), which also sets the number of GPU KV slots.
- `-cpu-batch auto|on|off` (default `auto`) batches CPU decode tokens, for models of at least 2 GiB of weights.
- `-prefill-chunk N` (default 512; 0 prefills whole) sets the piece size for a newcomer's prompt.
- `-max-queue N` (default 8) bounds each model's waiting requests.
- On Metal, pass `--embed-int4=false` for now. The 2026-09-28 default sends a Metal load to the CPU, where it does not batch. CUDA and WebGPU have not been checked for this.
- The banner's concurrency line says which mode is running. `scripts/bench_w7_plain.py` reproduces a row.
