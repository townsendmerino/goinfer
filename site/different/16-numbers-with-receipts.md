---
title: "Numbers with their receipts"
area: "Measurement"
order: 16
summary: "Every speed names its machine, checkpoint, quant, versions, date and machine state, and comes from both engines run over their own HTTP, interleaved."
stand: "A speed figure is only useful if you can see what produced it. goinfer's benchmark rows carry that, and the page says where it has lost and where it has been wrong."
measured: 2026-09-28
reviewed:
facts:
  - {label: "each row names", value: "machine, checkpoint and quant, versions, date, thermal state"}
  - {label: "both engines", value: "over their own HTTP, interleaved, restarted between cells"}
  - {label: "a claim withdrawn", value: "0.5B 1.78×, 2026-08-09"}
  - {label: "covers", value: "single-user, single-stream speed"}
doesnt:
  - title: "It doesn't cover more than a few machines."
    text: "The rows come from two computers: a MacBook Pro (M1 Pro, 16 GB) and a Linux box (Ryzen 7 3700X, RTX 2070 SUPER 8 GB). CUDA rows are tied to one NVIDIA driver, 595.91.07, and a new driver means a re-anchor. The site shows speeds for no other hardware."
  - title: "It doesn't cover anything but single-user speed."
    text: "Peer rows are one client and one stream, decode rate or time to first token. The multi-client results (\"Batching that doesn't change the answer\") compare goinfer with its own earlier builds, not with another engine. Nothing here says how any engine behaves under a crowd, or that answers are equally good."
  - title: "It doesn't remove noise."
    text: "Two builds of identical code gave a spread of 1.64% in one registry entry, and all six readings were below 1. A ratio inside the 0.97 to 1.03 band is called level, never a win. The noise registry lists 16 gaps where a gate has no measured spread to lean on, including goinfer's own drift between sessions."
  - title: "It compares against one peer version at its defaults."
    text: "The peer rows use Ollama v0.32.5 and llama.cpp 427291b as shipped. A newer version may differ. One row, the 26B model, is an architecture comparison because the two engines load different files."
  - title: "It hasn't kept us from errors."
    text: "The noise registry found three older records that overstate how tight their own spread was, and a \"3.5% typical drift\" that was generalised from one observation. The rules catch a mistake when someone checks."
figures:
  - {text: "1.78", source: docs/legacy-benchmarks.md}
  - {text: "320.1", source: docs/legacy-benchmarks.md}
  - {text: "269.4", source: docs/legacy-benchmarks.md}
  - {text: "1587.1", source: docs/benchmarks.md}
  - {text: "350", source: docs/benchmarks.md}
  - {text: "4.5", source: docs/benchmarks.md}
  - {text: "1.0456", source: docs/benchmarks.md}
  - {text: "1.0909", source: docs/benchmarks.md}
  - {text: "130", source: docs/benchmarks.md}
  - {text: "7 of 24", source: docs/benchmarks.md}
  - {text: "0.81%", source: docs/measurements/noise-registry.md}
  - {text: "1.64%", source: docs/measurements/noise-registry.md}
  - {text: "0.978–0.996", source: docs/measurements/noise-registry.md}
  - {text: "112.4", source: docs/measurements/noise-registry.md}
  - {text: "116.5", source: docs/measurements/noise-registry.md}
  - {text: "466", source: docs/measurements/noise-registry.md}
  - {text: "0.15%", source: docs/measurements/noise-registry.md}
  - {text: "0.74%", source: docs/measurements/noise-registry.md}
  - {text: "3.12%", source: docs/measurements/noise-registry.md}
  - {text: "5.03%", source: docs/measurements/noise-registry.md}
  - {text: "16 gaps", source: docs/measurements/noise-registry.md}
  - {text: "253.1 / 252.9 / 252.9", source: docs/measurements/peer-claim-2026-09-25.md}
  - {text: "195.1 / 195.1 / 195.0", source: docs/measurements/peer-claim-2026-09-25.md}
  - {text: "1.297 1.296 1.297", source: docs/measurements/peer-claim-2026-09-25.md}
  - {text: "0.97", source: docs/measurements/peer-claim-2026-09-25.md}
  - {text: "1.03", source: docs/measurements/peer-claim-2026-09-25.md}
  - {text: "9 AHEAD", source: docs/measurements/peer-claim-2026-09-25.md}
  - {text: "1.05–1.30", source: docs/measurements/peer-claim-2026-09-25.md}
  - {text: "0.58–0.75", source: docs/measurements/peer-claim-2026-09-25.md}
  - {text: "0.80", source: docs/measurements/peer-claim-2026-09-25.md}
  - {text: "595.91.07", source: docs/measurements/peer-claim-2026-09-25.md}
  - {text: "427291b", source: docs/measurements/peer-claim-2026-09-25.md}
sources:
  - docs/benchmarks.md
  - docs/legacy-benchmarks.md
  - docs/measurements/noise-registry.md
  - docs/measurements/peer-claim-2026-09-25.md
  - scripts/bench_peer.py
---

## The problem

A benchmark number without its conditions cannot be argued with later. goinfer has two examples of its own.

The first was a claim that goinfer's 0.5B model was 1.78× the peer's speed. It came from two columns produced by different methods. goinfer's column was an in-process Go benchmark, which times the kernels and never drives a peer. The peer's column was Ollama's HTTP server, timed end to end. The ratio divided a kernel throughput by an end-to-end one. Measured server to server on 2026-08-09, goinfer's 0.5B at 128 greedy read 320.1 tok/s against Ollama v0.32.5's 269.4. The peer's figure reproduced. Ours did not, because it was never measuring the same thing. We withdrew the claim rather than correct it.

The second passed every rule we had. A 3020-token prefill timing came back 1587.1 s where three later measurements of the same thing gave about 350 s, off by 4.5×. It was dated, named its machine and used the right checkpoint. Most likely an abandoned prefill from a killed client was still using a core. It reached three documents and a release note before another instrument disagreed.

## What goinfer does

A number enters a benchmark table only if it names the same machine for both engines, the same checkpoint and quant, greedy decoding with a fixed seed, pinned versions, the date, a thermal note, and a checkpoint read from local disk. Since the prefill mistake, it also records the machine's load beside the number.

The harness is `scripts/bench_peer.py`. It drives each engine over its own HTTP server and times the decode rate from the first streamed token, so prefill is excluded on both sides and neither engine's own accounting is trusted. Cells are interleaved, with a server restart between them. It stamps a header with the driver, kernel, goinfer commit, peer version, and the load average and GPU temperature at the start of each cell. It refuses to start on a box that is not idle. It checks that both engines hold the same weights tensor by tensor, not by file hash, because repacking changes the hash.

A published row looks like this: 1.5B, depth 128, RTX 2070 SUPER, driver 595.91.07, 2026-09-25. goinfer's three runs read 253.1 / 252.9 / 252.9 tok/s, Ollama's 195.1 / 195.1 / 195.0, and the three pairs 1.297 1.296 1.297.

## How it works

Three habits carry most of the weight.

**Count tokens from the engine's own `usage`, not from streamed chunks.** A chunk is not a token, so counting chunks under-reads, and the error does not cancel in a ratio. Across 130 recorded cells, goinfer read 1.0000 to 1.0456 tokens per chunk and Ollama 1.0000 to 1.0909.

**A short reply is not a slow one.** On an older prompt, replies ended early and voided 7 of 24 decode cells in one sweep. Which cells stopped was noise, so the prompt now asks for a long reply, and a cell is void if a completion returns too few tokens.

**Pre-register, then grade every pair.** The peer-claim record's bars were committed before any timed run and not edited after. A cell is level if every pair sits between 0.97 and 1.03, ahead only if every pair is above 1.03, and it is decided by all its pairs, never the mean.

"A cross-session ratio is not a ratio" comes from the noise registry. In it, on the CUDA box, the same binary restarted in one session gives an RMS of 0.81%. Two builds whose code for that cell is identical give 1.64%, and all six readings sit below 1 (0.978–0.996): a build effect. Across sessions, 466 same-build peer pairs have a median difference of 0.15%. For the 426 CUDA pairs among them the RMS is 0.74%, the p99 3.12% and the maximum 5.03%. goinfer's own between-session drift has one observation, a sampled phi3-mini cell that read 112.4 then 116.5 tok/s. So a ratio is only trusted when both sides ran in one interleaved session.

## What was measured

The peer-claim sweep (2026-09-25) tested "goinfer is as fast as Ollama or faster". The record's verdict: not earned. Ollama v0.32.5 and llama.cpp 427291b at their defaults, greedy decode at depth 128 unless noted.

| Machine | Result against Ollama |
|---|---|
| RTX 2070 SUPER, driver 595.91.07 (CUDA, 12 greedy decode cells) | 9 AHEAD, 1 level, 2 void, 0 behind. Where ahead: 1.05–1.30× |
| M1 Pro 16 GB (Metal, 9 cells) | 1 ahead, 6 behind (every 2048 and 3900 cell: 0.58–0.75×), 2 void |
| Ryzen 7 3700X (CPU, 3 cells) | 0.5B behind, 0.80×; 1.5B and 7B void |

Void cells are kept and named. Two of the twelve CUDA cells are void because Ollama ended its reply early. The losses are in the same record as the wins.

## Use it

- To compare goinfer with a peer, run `bench_peer.py`. The results file carries its own header:
  `GOINFER_SERVE=/path/to/cuda-serve OLLAMA_BIN=~/ollama-0325/bin/ollama python3 scripts/bench_peer.py results.json`
- To compare two goinfer builds, add a second binary and interleave it in the same sweep: `BENCH_ENGINES=goinfer,goinfer_old,ollama`, with `GOINFER_SERVE_CPU_OLD` and `GOINFER_SERVE_CPU` set.
- `scripts/bench_compare.sh` runs in-process benchmarks and never drives a peer. Use it for goinfer against goinfer, and do not put its output beside a peer's number.
- The rules are in `docs/benchmarks.md` (Methodology) and the spreads in `docs/measurements/noise-registry.md`. `scripts/power.py` reads that registry and says how many runs a gate needs.
