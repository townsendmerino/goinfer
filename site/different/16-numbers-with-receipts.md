---
title: "Numbers with their receipts"
area: "Measurement"
order: 16
summary: "Every speed names its machine, checkpoint, quantization, versions, date and machine state, and both engines are timed over their own HTTP servers, in turn."
stand: "A speed figure is only useful if you can see what produced it. goinfer's benchmark rows carry that, and the page says where it has lost and where it has been wrong."
measured: 2026-09-28
reviewed: 2026-09-29
facts:
  - {label: "each row names", value: "machine, checkpoint and quantization, versions, date, thermal state"}
  - {label: "both engines", value: "over their own HTTP servers, taking turns, restarted between runs"}
  - {label: "a claim withdrawn", value: "0.5B 1.78×, 2026-08-09"}
  - {label: "covers", value: "single-user, single-stream speed"}
doesnt:
  - title: "It doesn't cover more than a few machines."
    text: "The rows come from two computers: a MacBook Pro (M1 Pro, 16 GB) and a Linux PC (Ryzen 7 3700X, RTX 2070 SUPER 8 GB). CUDA rows are tied to one NVIDIA driver, 595.91.07, and after a driver change they must be deliberately re-measured before they can be compared. The site shows speeds for no other hardware."
  - title: "It doesn't cover anything but single-user speed."
    text: "Rows against another engine are one client and one stream: generation speed or time to first token. The multi-client results (\"Batching that doesn't change the answer\") compare goinfer with its own earlier builds, not with another engine. Nothing here says how any engine behaves under a crowd, or that answers are equally good."
  - title: "It doesn't remove noise."
    text: "In the project's noise registry, two builds of identical code gave a spread of 1.64%, and all six readings were below 1. A ratio inside the 0.97 to 1.03 band is called level, never a win. The registry lists 16 gaps where a check has no measured spread to lean on, including goinfer's own drift between sessions."
  - title: "It compares against one version of each other engine, at its defaults."
    text: "The rows against other engines use Ollama v0.32.5 and llama.cpp build 427291b as shipped. A newer version may differ. One row, the 26B model, compares two designs rather than like for like, because the two engines load different files."
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
  - {text: "1.045", source: "docs/measurements/peer-sweep-2026-09-29.md"}
  - {text: "1.474", source: "docs/measurements/peer-sweep-2026-09-29.md"}
  - {text: "0.905", source: "docs/measurements/peer-sweep-2026-09-29.md"}
  - {text: "1.059", source: "docs/measurements/metal-decode-gemv-r18b-2026-09-26.md"}
  - {text: "1.039", source: "docs/measurements/metal-decode-gemv-r18b-2026-09-26.md"}
  - {text: "1.032", source: "docs/measurements/metal-decode-gemv-r18b-2026-09-26.md"}
  - {text: "1.185", source: "docs/measurements/metal-decode-gemv-r18b-2026-09-26.md"}
  - {text: "1.113", source: "docs/measurements/metal-decode-gemv-r18b-2026-09-26.md"}
  - {text: "1.064", source: "docs/measurements/metal-decode-gemv-r18b-2026-09-26.md"}
sources:
  - "docs/measurements/metal-decode-gemv-r18b-2026-09-26.md"
  - docs/benchmarks.md
  - docs/legacy-benchmarks.md
  - docs/measurements/noise-registry.md
  - docs/measurements/peer-claim-2026-09-25.md
  - "docs/measurements/peer-sweep-2026-09-29.md"
  - scripts/bench_peer.py
---

## The problem

A benchmark number without its conditions cannot be argued with later. goinfer has two examples of its own.

The first was a claim that goinfer ran a 0.5B model (0.5 billion parameters) at 1.78× the speed of Ollama, another inference engine. It came from two columns produced by different methods. goinfer's column was an in-process Go benchmark, which times the inner compute loops (the kernels) and never drives another engine. Ollama's column was timed end to end through its HTTP server. The ratio divided a kernel throughput by an end-to-end one.

It was then measured server to server on 2026-08-09, on the GPU of a Linux PC (Ryzen 7 3700X, RTX 2070 SUPER 8 GB), with both engines run the same way: the 0.5B at depth 128 (128 tokens of context), with greedy decoding (always the most likely token). goinfer read 320.1 tok/s (tokens per second) against Ollama v0.32.5's 269.4. Ollama's figure reproduced. Ours did not, because it was never measuring the same thing. We withdrew the claim rather than correct it.

The second passed every rule we had. A 3020-token prefill timing (the time to read a 3020-token prompt before the reply starts) came back 1587.1 s where three later measurements of the same thing gave about 350 s, off by 4.5×. It was dated, named its machine and used the right checkpoint. Most likely a prompt from an earlier client, killed before it finished, was still being processed on one core. It reached three documents and a release note before another measurement disagreed.

## What goinfer does

A number enters a benchmark table only if it names all of these:

- the same machine for both engines;
- the same checkpoint (the model's weights file) and the same quantization (the number format the weights are stored in);
- greedy decoding with a fixed seed;
- pinned versions of both engines;
- the date and a thermal note;
- a checkpoint read from local disk.

Since the prefill mistake, it also records the machine's load beside the number.

The harness is [`scripts/bench_peer.py`](https://github.com/townsendmerino/goinfer/blob/main/scripts/bench_peer.py). It drives each engine through its own HTTP server. It times the generation rate from the first streamed token, so prompt processing is excluded on both sides and neither engine's own accounting is trusted. Runs of the two engines are interleaved, with a server restart between them. The results file starts with a header naming the GPU driver, the OS kernel, the goinfer commit and the other engine's version. It also records the load average and GPU temperature at the start of each measurement. The harness refuses to start on a machine that is not idle. It checks that both engines hold the same weights tensor by tensor (each weight array compared), not by file hash, because repacking a file changes its hash.

A published row looks like this: the 1.5B model at depth 128, on the RTX 2070 SUPER with NVIDIA driver 595.91.07, on 2026-09-25. goinfer's three runs read 253.1 / 252.9 / 252.9 tok/s, Ollama's 195.1 / 195.1 / 195.0, and the three paired ratios 1.297 1.296 1.297.

## How it works

Three habits carry most of the weight.

**Count tokens from the engine's own `usage` field, not from streamed chunks.** A chunk is not a token. One chunk can carry several tokens, so counting chunks under-reads, and the error does not cancel in a ratio. Across 130 recorded measurements, goinfer read 1.0000 to 1.0456 tokens per chunk and Ollama 1.0000 to 1.0909.

**A short reply is not a slow one.** With an older prompt, replies ended early, and 7 of 24 decode measurements in one run could not be counted. Which ones stopped early was noise. So the prompt now asks for a long reply, and a measurement is not counted if a completion returns too few tokens.

**Set the bars in writing before the run (pre-registered), then grade every pair.** For the 2026-09-25 comparison with Ollama, the bars were committed before any timed run and not edited after. A pair is one goinfer run divided by the Ollama run beside it. A measurement is level if every pair sits between 0.97 and 1.03, and ahead only if every pair is above 1.03. It is decided by all its pairs, never the mean.

"A cross-session ratio is not a ratio" comes from the project's [noise registry](https://github.com/townsendmerino/goinfer/blob/main/docs/measurements/noise-registry.md), which records how far repeated measurements of the same thing spread. On the Linux PC's GPU, the same binary restarted within one session gives an RMS spread of 0.81%. Two builds whose code for the 0.5B model is identical give 1.64%, and all six readings sit below 1 (0.978–0.996): the build itself has an effect. Across sessions (separate sittings), 466 pairs of restarts of one engine build have a median difference of 0.15%. Every one of those pairs is another engine's (Ollama or llama.cpp's server), not goinfer's. For the 426 CUDA pairs among them the RMS is 0.74%, the p99 3.12% and the maximum 5.03%. goinfer's own drift between sessions has been observed once: a phi3-mini measurement with sampling on read 112.4 tok/s in one session and 116.5 in another. So a ratio is only trusted when both engines ran in one interleaved session.

## What was measured

The pre-registered comparison of 2026-09-25 tested "goinfer is as fast as Ollama or faster". The record's verdict: not earned. Ollama v0.32.5 and llama.cpp build 427291b ran at their defaults. The models were Qwen2.5-Coder 0.5B and 1.5B and Qwen2.5-7B, as 4-bit GGUF files (q4_K_M), with the same weights on both engines. Decoding was greedy.

| Machine | What was measured | Result against Ollama |
|---|---|---|
| Linux PC: RTX 2070 SUPER, driver 595.91.07 (CUDA) | 3 models at depths 128, 2048, 3900 and 8000: 12 measurements | 9 AHEAD, 1 level, 2 not counted, 0 behind. Where ahead: 1.05–1.30× |
| MacBook Pro: M1 Pro, 16 GB (Metal) | 3 models at depths 128, 2048 and 3900: 9 measurements | 1 ahead, 6 behind (every measurement at depth 2048 and 3900: 0.58–0.75×), 2 not counted |
| Linux PC: Ryzen 7 3700X (CPU only) | 3 models at depth 128: 3 measurements | 0.5B behind, 0.80×; 1.5B and 7B not counted |

Measurements that were not counted are kept and named. Each one is a reply that ended early, from goinfer or from the other engine. Two of the twelve CUDA measurements were not counted because Ollama ended its reply early. The losses are in the same record as the wins.

The Metal row is dated, and a page like this one should say when it has moved. Three kernel changes landed on the Mac on 2026-09-25 and 2026-09-26. The same comparison was then repeated on the same MacBook, interleaved in one session like the table, but not pre-registered, so it is reported and decides nothing. It put goinfer ahead of Ollama at every depth it measured: the 1.5B at 1.059×, 1.039× and 1.032× (depths 128, 2048 and 3900), and the 7B at 1.185×, 1.113× and 1.064× ([the R18b record](https://github.com/townsendmerino/goinfer/blob/main/docs/measurements/metal-decode-gemv-r18b-2026-09-26.md)). A pre-registered re-run of the whole table at one build is the step that would replace the row.

The Linux rows have been re-run, all at one build (`754f12d3`), on 2026-09-29, under the same bars plus two added checks written down first: a reply that ends early is judged against half the requested tokens, and a GPU measurement that quietly ran on the CPU is not counted. On CUDA goinfer was ahead of Ollama in all 12 measurements, 1.045× to 1.474×, with none left out. On the CPU the 0.5B was still behind (0.905×), and the 1.5B and 7B, left out in the table above, were level and ahead. The same re-run found a slowdown in goinfer's own top-p sampling, which is how a table like this earns its keep ([the re-run's record](https://github.com/townsendmerino/goinfer/blob/main/docs/measurements/peer-sweep-2026-09-29.md)). The Mac half of the re-run has not been done yet.

Everything here is one client at a time. The multi-client results in [Batching that doesn't change the answer](/different/13-batching-same-answer/) compare goinfer with its own earlier builds, not with another engine.

## Use it

- To compare goinfer with another engine, run `bench_peer.py` on an idle machine, with the checkpoints under `~/models` on local disk. The results file carries its own header:
  `GOINFER_SERVE=/path/to/cuda-serve OLLAMA_BIN=~/ollama-0325/bin/ollama python3 scripts/bench_peer.py results.json`
- To compare two goinfer builds, add the older binary and interleave it in the same run:
  `GOINFER_SERVE_CPU_OLD=/path/to/old-serve GOINFER_SERVE_CPU=/path/to/new-serve BENCH_ENGINES=goinfer,goinfer_old,ollama BENCH_BACKENDS=cpu python3 scripts/bench_peer.py results.json`
- `scripts/bench_compare.sh` runs in-process benchmarks and never drives another engine. Use it for goinfer against goinfer, and do not put its output beside another engine's number.
- The rules are in [`docs/benchmarks.md`](https://github.com/townsendmerino/goinfer/blob/main/docs/benchmarks.md) (Methodology) and the spreads in the noise registry. `scripts/power.py` reads that registry and says how many runs a comparison needs to resolve its bar.
