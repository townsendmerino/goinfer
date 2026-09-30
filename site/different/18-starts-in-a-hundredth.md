---
title: "Starts in a hundredth of a second"
area: "Loading"
order: 18
summary: "After a one-time conversion, a model loads by mapping a prebuilt file: 0.01 s against 16.49 s for a 7B on the CPU, with almost no heap. Decode is unchanged."
stand: "The first time you serve a .gguf, goinfer converts it once into a sidecar file next to it. After that it maps that file instead of re-quantizing the model on every start."
measured: 2026-09-24
reviewed: 2026-09-29
facts:
  - {label: "CPU load, 7B", value: "0.01 s against 16.49 s"}
  - {label: "Go heap after load, 7B", value: "+1 MB against +4961 MB"}
  - {label: "decode speed", value: "unchanged (1.0016×)"}
  - {label: "first start", value: "converts once, about the model's size on disk"}
doesnt:
  - title: "It doesn't skip the first-load conversion."
    text: "The first start of each model, quant and target converts the .gguf and writes a file about the model's size beside it: 20.9 s and 1.29 GB for the 1.5B, 102.1 s and 5.18 GB for the 7B on the test machine. If free disk is short, serve refuses to start and names -direct-load."
  - title: "A fast load is not a fast first token."
    text: "Loading maps the file; the weight pages are read from the page cache when first used. The record's cache was warm, because the files had been written minutes earlier. A cold cache reads from disk and was not measured. On CUDA the weights still have to be copied to the GPU: 2.6–2.8 s for the 1.5B and 10.0–11.0 s for the 7B."
  - title: "It doesn't make decoding faster."
    text: "The sidecar and the direct load read the same quantized weights. Decode speed came out at 1.0007× and 1.0016× on the two models, inside the noise of a do-nothing control. This is a start-up and memory change only."
  - title: "It was measured on two models and one machine."
    text: "Qwen2.5-Coder 1.5B and Qwen2.5 7B, on one Linux box, on the CPU and CUDA. The record did not measure WebGPU, a cold page cache, any model above 7B, or macOS. The CUDA check was not pre-registered as its own rule."
  - title: "A sidecar belongs to one target, and its format changes."
    text: "It is keyed to the model, quant and target (a CPU architecture, or CUDA, and so on). These figures are for the v12 format. The CPU int4 format has since moved on (v15, unreleased), an older sidecar is rebuilt once, and load time and heap were not re-measured on it."
figures:
  - {text: "0.01 s", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "0.00 s", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "5.61 s", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "16.49 s", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "+1262 MB", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "+4961 MB", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "+1 MB", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "+1339 MB", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "+2545 MB", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "+24 MB", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "+26 MB", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "1.0007", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "1.0016", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "0.9995", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "1.0035", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "1.29 GB", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "5.18 GB", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "20.9 s", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "102.1 s", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "2.6–2.8 s", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "8.0–8.2 s", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "10.0–11.0 s", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "26.2–26.8 s", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "29.7 s", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "110.4 s", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "327 MB", source: docs/measurements/sidecar-default-2026-09-22.md}
  - {text: "1572 MB", source: docs/measurements/sidecar-default-2026-09-22.md}
  - {text: "0.6", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
  - {text: "08:52–09:02", source: docs/measurements/cpu-giw-vs-direct-2026-09-24.md}
sources:
  - docs/measurements/cpu-giw-vs-direct-2026-09-24.md
  - docs/measurements/sidecar-default-2026-09-22.md
  - docs/giw-bundles.md
  - CHANGELOG.md
  - internal/prequant/prequant.go
---

## The problem

A `.gguf` file stores a model in a format made to be shared between programs. goinfer runs its own int4 layout, so serving a `.gguf` directly means reading the file and re-quantizing every weight into memory. It does that on every start.

That has two costs. It takes time: on the CPU, 5.61 s for a 1.5B model and 16.49 s for a 7B. And it holds the whole converted model on the Go heap, +1262 MB for the 1.5B and +4961 MB for the 7B. On a machine with modest memory, that heap is memory the operating system cannot use for anything else, and on a laptop it can push the system into swap.

## What goinfer does

On Linux and macOS, serving a `.gguf` now goes through a sidecar. The first start converts the model once and writes a `.giw` file beside it, and later starts map that file instead of converting anything. `-direct-load` turns it off.

```sh
goinfer-serve -model ~/models/qwen2.5-7b-instruct-q4_k_m.gguf
# first start: "stream-weights: transcoding … (int4, one-time — minutes + ~model-size on disk)…"
# every later start: maps the sidecar and loads in a fraction of a second
```

Nothing else changes for the user. The sidecar is a cache: delete it and it is rebuilt.

## How it works

A `.giw` holds the weights already quantized in the layout the decoder reads, with its arrays padded so a reader can use them where they sit. Loading maps the file into memory and does no work up front, and the operating system reads pages as the model uses them. So the heap after load is close to zero, and the weights live in the page cache, which the system can reclaim if it needs the space. The padding is the v12 format (commit `cdae727d`, 2026-09-24), which lets the reader use the scale arrays in place instead of copying them.

The file is specific to a model, a quant and a target, and the target is part of its name. The CPU and CUDA sidecars for one model are separate files.

## What was measured

The question was whether the sidecar costs decode speed on Linux. It was registered first, with a do-nothing arm (a second direct load, to measure noise), a hard gate that every greedy token stream must be identical across arms, and a rule for slower, no cost, faster or ambiguous.

The run was on nobara (Ryzen 7 3700X, CPU backend, int4), 2026-09-24 08:52–09:02, with the box idle (load under 0.6). Models were Qwen2.5-Coder 1.5B q4_k_m and Qwen2.5 7B q4_k_m from local disk. The sidecars were v12 files, 1.29 GB and 5.18 GB. Load figures are the change in one process after each load.

| Model | Load | Go heap in use | RssAnon | RssFile |
|---|---|---|---|---|
| 1.5B direct | 5.61 s | +1262 MB | +1339 MB | +0 |
| 1.5B sidecar | 0.00 s | +0 MB | +0 MB | +24 MB |
| 7B direct | 16.49 s | +4961 MB | +2545 MB | +0 |
| 7B sidecar | 0.01 s | +1 MB | ≈0 | +26 MB |

The gate passed: every generation's greedy stream was identical across the three arms. Decode speed, sidecar over direct, read 1.0007 on the 1.5B and 1.0016 on the 7B (median of seven paired runs). The do-nothing arm read 0.9995 and 1.0035, so the difference is inside the noise, and the registered verdict is "no cost".

On CUDA, checked separately: the sidecar loads in 2.6–2.8 s (1.5B) and 10.0–11.0 s (7B), against 8.0–8.2 s and 26.2–26.8 s direct, with byte-identical output. The first start there converts once: 29.7 s for the 1.5B and 110.4 s for the 7B.

macOS was measured in an earlier record, for memory only (2026-09-22): on the 1.5B the anonymous footprint was 327 MB with the sidecar against 1572 MB direct. That was before the v12 format.

Were the Linux figures taken after the v12 format change? Yes. The record names v12 weights and a v3 bundle, and the change landed earlier the same morning. They were not re-measured on the later format.

## Use it

- On by default on Linux and macOS. Opt out with `-direct-load` or `GOINFER_GGUF_DIRECT=1`.
- To convert ahead of time, on a machine you choose: `go run ./cmd/prequant -o model.int4.giw -quant int4 model.gguf`, then pass the `.giw` to `--model`. A `.giw` is built for one target, so build it for the machine that will run it.
- Serving a `.giw` you built yourself does not use the cache logic: it is loaded as given.
