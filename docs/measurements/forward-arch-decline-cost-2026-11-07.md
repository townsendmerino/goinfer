# What a decline costs on the Mac: Metal against the CPU, same session (RF0, 2026-11-07)

**Status: a measurement of a proxy, taken to correct a figure; not a gate, not registered.** RF0 asked what each decline costs. A declined
family has no GPU arm to time, so the price of a decline can only be read off a family that runs on both. This record times the Qwen2.5
dense models, which do, and it is the first same-session CPU-against-Metal reading in the repository for the question. It says nothing about
the declined families themselves (Phi-3/Phi-4, Granite-4.0-H, LFM2.5, Llama 4, Laguna, Spark-X2.5, Ling 3.0), and nothing about CUDA.

## Result

Decode throughput, goinfer's own CPU path against its Metal path, same checkpoint, same quantisation class, phase A's 129-token prompt (the harness's "depth 128"; `BENCH_DEPTHS=none` skips only its depth curve), greedy,
`scripts/bench_peer.py` interleaved A B B A (`BENCH_ABBA`), tokens per second:

| model | CPU (median of the cell means) | Metal (median of the cell means) | Metal / CPU | cells (CPU, Metal) |
|---|---|---|---|---|
| Qwen2.5-Coder 0.5B Q4_K_M | 118.6 | 211.6 | **1.78×** | 4, 4 |
| Qwen2.5-Coder 1.5B Q4_K_M | 57.7 | 100.6 | **1.74×** | 4, 4 |
| Qwen2.5 7B Q4_K_M | 18.8 | 33.8 | **1.80×** | 2, 2 |

- **The block-by-block ratios** (Metal block against the CPU block of the same position): 0.5B 1.79, 1.78, 1.78, 1.84; 1.5B 1.76, 1.75, 1.69, 1.75; 7B 1.81, 1.78. Every block is between 1.69 and 1.84.
- **The spread inside a cell** (two runs of eight completions of 64 tokens) is under 1% on every cell but two (Metal 0.5B block 3 reads 211.7 and 219.0; CPU 7B block 0 reads 18.9 and 18.5); no cell was void; the harness re-ran one cold-load start (load 4.9 and 4.1 against its cap of 4.0) in place, as its rule says, and the 7B CPU cell it started is the one recorded.
- **The ratio does not grow with the model.** From 0.5B to 7B it stays at 1.7 to 1.8. The CPU path reaches roughly 88 GB/s of weight traffic on the 7B (18.8 tok/s × 4.68 GB), the Metal path roughly 158 GB/s (33.8 × 4.68), which is the 1.80× of the table.

## What it corrects

`docs/tasks/task-forward-architecture-2026-10.md` quoted "a model on the CPU decodes about 15× slower" in six places (§0, the §2 table, §4.2, and three gate lines). No record sits behind that figure. On this Mac, for these three sizes, the CPU is **1.7 to 1.8× slower**, not 15×. The 15× may describe the CUDA box (a 16-thread Ryzen against the card); nothing in the repository measures that pair in one session, and this record does not. The three summary statements now say what was measured, that it is a Mac proxy and that CUDA is unmeasured; the three gate lines under OP2 keep their historical text with a dated correction beside it, because they were registered under the 15× and a Mac ratio does not justify a CUDA decision.

What the correction changes: a family leaving the CPU fallback on the Mac gains about 1.8× on decode for a model like these, which is still the largest single change available to a user of that family but far from an order of magnitude. It also weakens the premise of those gate lines ("the do-nothing arm is the CPU, about 15× slower, so the direction cannot be in doubt"): at 1.8× on a dense model, a first kernel for a new family can be slower than the CPU (OP6's first Metal kernels on Phi-3 were 4.7× slower than the loop until #722), so the direction is not settled by the ratio.

## What ran

Date 2026-11-07, start 17:01 and 17:10 local, about 7 and 3 minutes of cells; by day, under the timing lock (the harness holds it exclusive), after a build of both binaries under the shared lock. No other timed job held the lock.
- **Machine:** MacBook Pro, Apple M1 Pro, 16 GB, macOS 26.6.2, go1.27.0 darwin/arm64. Load average 2.0 to 2.7 and free memory 5.8 to 6.6 GB at the first cells' starts, instant idle gate (busy 5% to 9% against its 10% cap). Thermal: no warning level recorded.
- **Binaries:** `cmd/serve` for the CPU arm and `backend/metal/cmd/serve` for the Metal arm, built from `c75b012e` (main `912eae66` plus documentation), `GOFLAGS` unset, `go.work` from `scripts/make_work.sh`.
- **Checkpoints,** all from `~/models` on this machine: `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` (491 MB), `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` (1,117 MB), `qwen2.5-7b-instruct-q4_k_m.gguf` (4,683 MB). Both arms ran int4 (`decode_path` reads `cpu (int4)` with the row4 layout on the CPU arm); the harness checked that the arms share a quant class.
- **Commands,** from the `c75b012e` tree:
  `GOINFER_SERVE_CPU=… GOINFER_SERVE_METAL=… BENCH_ENGINES=goinfer BENCH_BACKENDS=cpu,metal BENCH_MODELS=0.5B,1.5B BENCH_PHASES=A BENCH_DEPTHS=none BENCH_ABBA=2 BENCH_REQUIRE_BACKEND=1 python3 scripts/bench_peer.py results.json`, then the same with `BENCH_MODELS=7B BENCH_ABBA=1`. `BENCH_REQUIRE_BACKEND=1` makes the Metal arm refuse a load that would fall back to the CPU.
- **Sampling:** greedy (`temperature` 0, `seed` 1), 8 completions of 64 tokens per run, two runs per cell, all 1,024 tokens per cell streamed in 1,024 chunks, token gate ok on every cell.
- The raw JSON and run logs are in the session's log directory outside the repository (a gitignored location is not a citable destination); the table above is computed from them.

## What this does not show

- **Not the declined families.** Their CPU paths include the very features that made them decline (a short convolution, a Mamba or delta-net scan, an own-forward loop); their CPU speed against a hypothetical GPU path is unknown. The ratio for a hybrid or recurrent family could differ.
- **One machine, one depth, int4 only.** Not a curve over depth, not int8, not f16 KV; no nobara reading.
- **7B has two cells per arm** (one A B B A quad) against four for the smaller models.
- **The CPU path got faster than the last record.** `docs/benchmarks.md` and `docs/measurements/cpu-decode-roofline-2026-09-23.md` report Mac CPU decode of 46.9, 19.4 and 5.0 tok/s for the same three sizes on 2026-09-23; this session reads 118.6, 57.7 and 18.8. A cross-session comparison is not a ratio, so this record does not attribute the difference; it is why the Mac CPU fallback is cheaper than the older records suggest.
