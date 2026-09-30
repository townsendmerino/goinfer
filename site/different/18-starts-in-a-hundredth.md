---
title: "Starts without converting the model again"
area: "Loading"
order: 18
summary: "After a one-time conversion, a model loads by mapping a file: 1.39 s against 9.16 s for a 7B on the CPU, with almost no heap. Generation speed is unchanged."
stand: "The first time you serve a .gguf, goinfer converts it once into a sidecar file next to it. After that it maps that file instead of converting the model again on every start."
measured: 2026-09-29
reviewed: 2026-09-29
facts:
  - {label: "CPU load, 7B", value: "1.39 s against 9.16 s"}
  - {label: "Go heap after load, 7B", value: "+1 MB against +4577 MB"}
  - {label: "generation speed", value: "unchanged (1.0011×)"}
  - {label: "first start", value: "converts once, writing about the model's size to disk"}
doesnt:
  - title: "It doesn't skip the first-load conversion."
    text: "The first start of each combination of model, quantization and target converts the `.gguf` and writes a file about the model's size beside it: on the Linux PC's CPU, 18 s and 1155 MB for the 1.5B, 87 s and 4548 MB for the 7B; on CUDA the first start took 40.0 s and 120.3 s. If free disk space is short, the server refuses to start and names `-direct-load`."
  - title: "A fast load is not a fast first token."
    text: "Loading maps the file, and each page of weights is read from the operating system's page cache when it is first used. The measured cache was warm, because the files had been written minutes earlier. A cold cache reads from disk and was not measured. On CUDA the weights still have to be copied to the GPU: 3.0–3.2 s for the 1.5B and 10.41–12.22 s for the 7B."
  - title: "The 7B's load got slower in the current format, and why is not known."
    text: "On version 15 of the file format (v15), which is not yet in a release, the 1.5B still loads in 0.00 s but the 7B takes 1.39 s. On the earlier v12 it took 0.01 s. Its heap stays at +1 MB, so the time is not a copy into memory. No profile was taken, so the cause is not established."
  - title: "It doesn't make generation faster."
    text: "The sidecar and the direct load read the same quantized weights. Generation speed came out at 1.0057× and 1.0011× the direct load's on the two models, inside the noise measured by a control that changed nothing. This is a start-up and memory change only."
  - title: "It was measured on two models and one machine."
    text: "Qwen2.5-Coder 1.5B and Qwen2.5 7B, on one Linux PC, on the CPU and with CUDA. Neither record measured WebGPU, a cold page cache, any model above 7B, or macOS load time. A sidecar is keyed to the model, the quantization and the target (a CPU architecture, CUDA, and so on), and an older sidecar is rebuilt once after a format change."
figures:
  - {text: "1.39 s", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "9.16 s", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "2.53 s", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "0.00 s", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "+1184 MB", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "+4577 MB", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "+1 MB", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "+0 MB", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "1.0057", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "1.0011", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "1.0003", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "0.9957", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "18 s", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "87 s", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "1155 MB", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "4548 MB", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "40.0 s", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "120.3 s", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "3.0–3.2 s", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "10.41–12.22 s", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "10.22–10.42 s", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "32.05–32.85 s", source: "docs/measurements/sidecar-v15-2026-09-29.md"}
  - {text: "0.01 s", source: "docs/measurements/cpu-giw-vs-direct-2026-09-24.md"}
  - {text: "5.61 s", source: "docs/measurements/cpu-giw-vs-direct-2026-09-24.md"}
  - {text: "16.49 s", source: "docs/measurements/cpu-giw-vs-direct-2026-09-24.md"}
  - {text: "+1262 MB", source: "docs/measurements/cpu-giw-vs-direct-2026-09-24.md"}
  - {text: "+4961 MB", source: "docs/measurements/cpu-giw-vs-direct-2026-09-24.md"}
  - {text: "1.0007", source: "docs/measurements/cpu-giw-vs-direct-2026-09-24.md"}
  - {text: "1.0016", source: "docs/measurements/cpu-giw-vs-direct-2026-09-24.md"}
  - {text: "327 MB", source: "docs/measurements/sidecar-default-2026-09-22.md"}
  - {text: "1572 MB", source: "docs/measurements/sidecar-default-2026-09-22.md"}
sources:
  - "docs/measurements/sidecar-v15-2026-09-29.md"
  - "docs/measurements/cpu-giw-vs-direct-2026-09-24.md"
  - "docs/measurements/sidecar-default-2026-09-22.md"
  - "docs/giw-bundles.md"
  - "CHANGELOG.md"
  - "internal/prequant/prequant.go"
---

## The problem

A `.gguf` file stores a model in a format made to be shared between programs. goinfer computes with its own 4-bit (int4) weight layout. So serving a `.gguf` directly means reading the file and converting every weight into that layout in memory (re-quantizing it). It does that on every start.

That has two costs. It takes time: on the CPU, 2.53 s for a 1.5B model and 9.16 s for a 7B. And it holds the whole converted model on the Go heap: +1184 MB for the 1.5B and +4577 MB for the 7B. On a machine with modest memory, that heap is memory the operating system cannot use for anything else, and on a laptop it can push the system into swap. ([It refuses rather than swaps](/different/07-refuses-rather-than-swaps/) covers what goinfer does when a load will not fit.)

## What goinfer does

On Linux and macOS, serving a `.gguf` now goes through a sidecar: a converted copy kept next to the original file. The first start converts the model once and writes a `.giw` file beside the `.gguf`. Later starts map that file instead of converting anything. `-direct-load` turns this off. On Windows, the direct load is still the default.

```sh
goinfer-serve -model ~/models/qwen2.5-7b-instruct-q4_k_m.gguf
# first start: "stream-weights: transcoding … (int4, one-time — minutes + ~model-size on disk)…"
# every later start: maps the sidecar instead of converting
```

Nothing else changes for the user. The sidecar is a cache: delete it and it is rebuilt.

## How it works

A `.giw` holds the weights already quantized, in the layout the decoder reads, with its arrays padded so a reader can use them where they sit. Loading maps the file into memory and does no work up front, and the operating system reads pages in as the model uses them. So the heap after load is close to zero. The weights live in the page cache, which the system can reclaim if it needs the space. The padding came with version 12 of the format (v12, commit [`cdae727d`](https://github.com/townsendmerino/goinfer/commit/cdae727d), 2026-09-24). It lets the reader use the arrays of scale factors that come with 4-bit weights in place, instead of copying them.

The file is specific to a model, a quantization and a target, and the target is part of its name (for example `.int4.cpu-amd64.giw` or `.int4.cuda.giw`). The CPU and CUDA sidecars for one model are separate files. The release's one-file chat binaries carry the same kind of file inside them ([One file, model inside](/different/11-one-file-model-inside/)).

## What was measured

The question was whether the sidecar slows decoding (generating tokens, one at a time) on Linux. The plan was set in writing before the run (pre-registered; [Numbers with their receipts](/different/16-numbers-with-receipts/) explains the method). It compared three configurations in one process: a direct load, a sidecar load, and a second direct load. The second direct load is a control: it changes nothing, so any difference it shows is noise. The plan required every greedy token stream (the model always taking its most likely token) to be identical across the three, and fixed in advance the bounds a result had to meet.

It was first run on 2026-09-24 on version 12 of the file format, and repeated on 2026-09-29 on version 15, the format the code now writes, with fresh sidecars and the current defaults. Both runs were on a Linux PC (Ryzen 7 3700X, RTX 2070 SUPER 8 GB), on the CPU backend at int4, with an idle machine. The models were Qwen2.5-Coder 1.5B and Qwen2.5 7B, both q4_k_m GGUF files (a 4-bit format) read from local disk. Each generation read a 128-token prompt and decoded 24 tokens. Load figures are the change in one process after each load; "Go heap in use" is `runtime.MemStats.HeapInuse`.

| Model | Load, 2026-09-29 (v15) | Go heap in use | Load, 2026-09-24 (v12) | Go heap |
|---|---|---|---|---|
| 1.5B direct | 2.53 s | +1184 MB | 5.61 s | +1262 MB |
| 1.5B sidecar | 0.00 s | +0 MB | 0.00 s | +0 MB |
| 7B direct | 9.16 s | +4577 MB | 16.49 s | +4961 MB |
| 7B sidecar | 1.39 s | +1 MB | 0.01 s | +1 MB |

The identity check passed in both runs: every generation's greedy token stream was identical across the three configurations. On 2026-09-29, decode speed, sidecar over direct, read 1.0057 on the 1.5B and 1.0011 on the 7B (the median of seven paired runs); the control read 1.0003 and 0.9957 (2026-09-24: 1.0007 and 1.0016). So the difference is inside the noise.

The 7B's sidecar load is the one figure that moved: 1.39 s on v15 against 0.01 s on v12. It missed the bound set before the run (0.05 s), which is why this page's title changed. The heap did not move, so the time is not spent copying weights into memory; what it is spent on was not measured. The direct loads got faster over the same period, so the sidecar's advantage on the 7B is now 9.16 s against 1.39 s rather than 16.49 s against 0.01 s.

On CUDA, in the same run on the same PC through the server: the sidecar loads in 3.0–3.2 s (1.5B) and 10.41–12.22 s (7B), against 10.22–10.42 s and 32.05–32.85 s direct, with the same output text on all 16 servers per model. The first start there converts once: 40.0 s for the 1.5B and 120.3 s for the 7B.

macOS was measured in an earlier record, for memory only, on the 16 GB MacBook on 2026-09-22. On the 1.5B, the anonymous memory footprint was 327 MB with the sidecar against 1572 MB direct. That was before the v12 format.

## Use it

- On by default on Linux and macOS. Opt out with `-direct-load` or `GOINFER_GGUF_DIRECT=1`.
- To convert ahead of time, on a machine you choose, from a clone of the repo: `go run ./cmd/prequant -o model.int4.giw -quant int4 model.gguf`. Then pass the `.giw` to `--model`.
- A `.giw` is built for one target, so build it for the machine that will run it. `-target` picks it (`cpu`, `cuda`, `metal`, `webgpu` and others), and the default is the CPU of the machine doing the build.
- Serving a `.giw` you built yourself does not use the cache logic: it is loaded as given.
