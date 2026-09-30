---
title: "It refuses rather than swaps"
area: "Memory"
order: 7
summary: "If a model won't fit, goinfer says so and by how much before it loads, watches swap while it loads, and retries dense models with weight streaming."
stand: "goinfer prices a load against the memory the machine has free right now, and refuses with the arithmetic if it will not fit. A swap tripwire backs that up. The tripwire's limit is written down."
measured: 2026-09-24
reviewed: 2026-09-29
facts:
  - {label: "before loading", value: "says what it needs and what is free"}
  - {label: "during loading", value: "swap tripwire, +512 MB over its baseline"}
  - {label: "dense model too big", value: "one automatic retry with weight streaming"}
  - {label: "measured on", value: "one 16 GB MacBook"}
doesnt:
  - title: "It doesn't hold the machine on its own."
    text: "The tripwire fired at +0.72–0.80 GB of swap growth on the gpt-oss-20b positive control, inside its +1 GB bound, but swap kept rising to +1.69 GB before an outside kill switch stopped the process. The likely cause is workers still finishing layers they had already started. That was inferred from timing, not traced. The env-vars doc still advises an external backstop when the fit guard is bypassed."
  - title: "It doesn't watch every load, and it doesn't stop running work."
    text: "The load-time abort applies to a .gguf built directly on the heap, not to a sidecar, streamed or safetensors load. The serving half only refuses new requests with a 503. Requests already running are left to finish."
  - title: "Its estimate can be wrong either way."
    text: "The guard prices weights, KV cache and the mapped source file against 70% of what is free, and a threshold that came from one measured failure, not a sweep. If it cannot read free memory it lets the load through. A sidecar .giw load prices only KV and scratch, because its weights are file-backed."
  - title: "It has only been measured on one machine."
    text: "The records are from a 16 GB MacBook, on the CPU and Metal backends. The swap probe reads macOS and Linux; anywhere else it reports unknown and never trips. We have no record of the tripwire running on Linux or another OS."
  - title: "It doesn't stream a big MoE for you."
    text: "The automatic retry is for dense models. A mixture-of-experts .gguf that doesn't fit is refused, because a manual run of one on the CPU streaming path once ran 2h10 with zero completions on this Mac. You can still pass -stream-weights yourself."
figures:
  - {text: "+7,819 MB", source: "docs/measurements/cold-user-2026-09-06.md"}
  - {text: "8.9 GB", source: "docs/tasks/task-first-hour.md"}
  - {text: "5.0 GB", source: "docs/tasks/task-first-hour.md"}
  - {text: "41.6 GB", source: "docs/measurements/swap-tripwire-2026-09-22.md"}
  - {text: "5.2 GB", source: "docs/measurements/swap-tripwire-2026-09-22.md"}
  - {text: "3.7 GB", source: "docs/measurements/swap-tripwire-2026-09-22.md"}
  - {text: "+0.72–0.80 GB", source: "docs/measurements/swap-tripwire-2026-09-22.md"}
  - {text: "+1.69 GB", source: "docs/measurements/swap-tripwire-2026-09-22.md"}
  - {text: "+1 GB", source: "docs/measurements/swap-tripwire-2026-09-22.md"}
  - {text: "22.9 GB", source: "docs/measurements/swap-tripwire-2026-09-22.md"}
  - {text: "100/100", source: "docs/measurements/swap-tripwire-negative-controls-2026-09-24.md"}
  - {text: "1,346.12 MB", source: "docs/measurements/swap-tripwire-negative-controls-2026-09-24.md"}
  - {text: "16–21%", source: "docs/measurements/sidecar-default-2026-09-22.md"}
  - {text: "0.944", source: "docs/tasks/task-gpu-paths-2026-09.md"}
  - {text: "5.6%", source: "docs/tasks/task-gpu-paths-2026-09.md"}
  - {text: "+512 MB", source: "docs/server.md"}
sources:
  - "docs/tasks/task-never-swap-2026-09.md"
  - "docs/tasks/task-first-hour.md"
  - "docs/tasks/task-gpu-paths-2026-09.md"
  - "docs/tasks/task-fit-to-hardware.md"
  - "docs/measurements/cold-user-2026-09-06.md"
  - "docs/measurements/swap-tripwire-2026-09-22.md"
  - "docs/measurements/swap-tripwire-negative-controls-2026-09-24.md"
  - "docs/measurements/sidecar-default-2026-09-22.md"
  - "docs/server.md"
  - "docs/env-vars.md"
  - "CHANGELOG.md"
  - "decoder/fitguard.go"
  - "internal/modelload/modelload.go"
  - "internal/swapguard/swapguard.go"
---

## The problem

A model that is bigger than the memory you have free does not fail with an error. The operating system starts writing memory to disk (swap) to make room, and the machine gets slow for everything, not just the model.

On 2026-09-06, on a 16 GB MacBook, an early goinfer (v0.16.0) was asked to load a 35B mixture-of-experts checkpoint. It printed nothing about size, and swap grew by +7,819 MB in five seconds. The report's summary of it: the tool never told the user it would not fit, the machine did.

Earlier, on 2026-09-04/05, a plain-CPU load of gpt-oss-20b on the same MacBook had taken swap to 22.9 GB.

## What goinfer does

Before it allocates the model, goinfer adds up what the load needs and compares that with the memory available at that moment. If the total is over budget it refuses and prints the sum. This is the refusal from a real run on the MacBook, on 2026-09-22:

```
decoder: gpt-oss-20b-MXFP4.gguf needs ~24.3 GB resident at quant int4 + 6.0 GB KV + 11.3 GB
reading the checkpoint (the .gguf stays mapped resident for the whole load) = 41.6 GB; this
machine currently has 5.2 GB of memory available (budget 3.7 GB = 70% of that).
  Loading it would page to swap rather than run, so it was NOT loaded.
```

The message goes on to name `-stream-weights`, a smaller model or quant, and `GOINFER_NO_FIT_GUARD=1` to load anyway. Swap did not move during that refusal.

Two things have changed since that run. A plain .gguf on macOS and Linux now loads through a memory-mapped sidecar by default, so a load like this one often meets a smaller check first, on the KV cache and scratch memory. And a dense model that is refused gets one automatic retry with weight streaming, with a note on stderr saying so. `--fit=off` turns that retry off and keeps the refusal. It does not turn the guard off.

## How it works

**Estimate.** The guard prices resident weights at the chosen quant, the KV cache at the context, and, for a plain .gguf, the source file that stays mapped during the load. It compares the sum with 70% of the memory available now, read live rather than from the machine's total. That last part came from a live Mac re-run before v0.17.2, where a budget based on total RAM let a load swap.

**Tripwire.** The server samples swap-used every two seconds and remembers its first reading. If swap grows by more than 512 MB over that baseline, a direct .gguf load aborts with a message naming the growth and the priced terms. While serving, new requests get a 503 until swap stays back within the threshold for 30 seconds. It keys on swap, not on the process's memory, because macOS reports what survived reclaim, not what was asked for.

**Retry.** A dense .gguf that is refused is transcoded once to a sidecar .giw and paged from disk, instead of held whole in memory.

## What was measured

| Date | Machine | What | Result | Record |
|---|---|---|---|---|
| 2026-09-06 | 16 GB MacBook, v0.16.0 | 35B MoE, before any guard | +7,819 MB swap in 5 s, no warning | `cold-user-2026-09-06.md` |
| before v0.17.2 (tagged 2026-09-08) | 16 GB MacBook, 7B int4 | third live re-run of the guard | refused in under 5 seconds, zero Swapouts: 8.9 GB of weights against a 5.0 GB budget | `task-first-hour.md` |
| 2026-09-09 | the MacBook, CPU | dense retry, 7B q4_k_m, 7 of 28 layers resident | streamed 0.944 of resident decode, a 5.6% cost (6 runs per arm) | `task-gpu-paths-2026-09.md` |
| 2026-09-22 | the MacBook, CPU | gpt-oss-20b with the fit guard bypassed | tripped at +0.72–0.80 GB, peaked at +1.69 GB, outside kill needed | `swap-tripwire-2026-09-22.md` |
| 2026-09-22 | the MacBook, CPU | sidecar against direct load, two models | anonymous memory 16–21% of direct, swap +0 MB | `sidecar-default-2026-09-22.md` |
| 2026-09-24 | M1 Pro, 16 GB | 1.5B on CPU and 7B on Metal, the guard should stay quiet | 100/100 completions each, 0 trips, swap flat at 1,346.12 MB | `swap-tripwire-negative-controls-2026-09-24.md` |

The 2026-09-22 row is the honest one. The tripwire fired on time and the machine recovered at once, but the load was not held under its own bound without help.

## Use it

- `goinfer-chat fit <file.gguf>` prints the plan per backend. It loads the checkpoint, so it is not free.
- `-stream-weights` pages weights from a sidecar, and `-weight-cache` sets the budget in GB. `-ctx` pins a smaller context.
- `--fit=off` keeps the refusal instead of retrying. `GOINFER_NO_FIT_GUARD=1` skips the guard, and then the tripwire is the last defence.
- `GOINFER_SWAP_GUARD` takes a threshold in MB, or `off`. The banner line `swap guard: armed, threshold +512 MB over baseline` says it is on.
- `-direct-load` puts a .gguf back on the heap path the load-time tripwire watches.
