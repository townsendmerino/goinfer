# Linux CPU decode, per-component roofline — and the fused gate+up+SwiGLU fork/join it pointed at

Follows `cpu-decode-attribution-2026-09-22-linux.md`, `cpu-worker-pool-2026-09-22-linux.md` (KILLED: "the remaining MLP matmul cost
is not a dispatch problem … a kernel/memory-system question for aikit") and `cpu-peer-reanchor-2026-09-22.md` (0.651× / 0.739× / 0.817× of
Ollama). The question this answers: **where, in bytes and milliseconds, is the remaining gap — and is it the MLP matmuls?** The answer
is partly, and the part that is not the MLP matmuls is bit-identically recoverable.

**Provenance.** `nobara-pc`, Ryzen 7 3700X (8c/16t, one NUMA node, L3 2×16 MiB), Nobara 44, THP `always`, goinfer `ec6fec95` for the
per-component split (before the change) and `d88ef960` for the served sweep (with it), aikit v1.47.0, `GOAMD64=v1`, Backend `cpu`,
`Quant: "int4"`, greedy, 128-token prompt, 24 decode tokens per split run. Models: the same three q4_K_M GGUFs from `~/models` on local
NVMe as every CPU row on this page. Ollama v0.32.5 (`~/ollama-0325`), CPU-forced. Served cells start behind the harness's ≤1.0 load gate (0.73–0.97
recorded); in-process runs were back-to-back on an otherwise idle box — the first split run started at a 1-min load of 2.74 left by the
preceding test, and its two repeats agree with it to ±0.2 ms.

## 1. The ceiling, measured independently (`scripts/readbw.c`, `…-readbw-2026-09-23.log`)

The 30.5 GB/s "read ceiling" every earlier record quotes had no instrument I could find in the tree. An AVX2 read-only stream over 2 GiB (60× L3),
median of 7:

| GB/s | 1 thr | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| contiguous slice per thread | 23.4 | **30.9** | 29.9 | 28.6 | 27.6 |
| 960-byte chunks, round-robin (a column-sharded matmul's shape) | 21.3 | 20.2 | 28.7 | 29.0 | 27.0 |
| … with software prefetch | 21.5 | 30.1 | 30.2 | 28.6 | 27.4 |

The ceiling is real (**~30 GB/s, reached with 2–4 threads; 27–29 with all 16**), the access pattern costs nothing, and prefetch buys
nothing — the hardware streamer already covers a row-sized stream. THP is `always`, which makes the page-size hypothesis unlikely for anonymous
heap weights (`AnonHugePages` for the loaded weights was not checked).
One thread can pull 21–23 GB/s by itself; goinfer's isolated single-thread matmul streams ~8.75 GB/s, so the kernel is *compute*-limited
per thread and DRAM only binds with several threads.

## 2. Real bytes per token (`TestCPUDecode_weightBytesPerToken`, `…-bytes-2026-09-23.log`)

From the loaded model's own storage, not a size formula. **Every int4 matrix is 0.625 B/param, not 0.5**: `MatmulBTW4A8Into` takes
`wScales []float32`, one f32 per 32-group, adding 0.125 B/param. The LM head is int8 (1.003 B/param).

| 1.5B, per token | params | MB | |
|---|---:|---:|---|
| q/k/v | 88.1M | 55.1 | |
| o | 66.1M | 41.3 | |
| gate+up | 770.7M | 481.7 | |
| down | 385.4M | 240.8 | |
| LM head (tied embedding, int8) | 233.4M | 234.0 | |
| **total** | | **1052.9** | 0.5B: 360.4 MB · 7B: 4623.9 MB |

**Correction to the earlier record.** `TestR9_mlpMatmulIsolated` computes bandwidth as `macs × 0.5` with "scales ignored"; the
"19.5 GB/s of weights (64% of the ceiling)" that record and the worker-pool record quote was **real DRAM traffic of ~24 GB/s (≈80%)**.
The isolated matmuls were never as far from the ceiling as they read.

## 3. The 1.5B token, per component (DECODE SPLIT ×3 runs, `…-split-2026-09-23.log`; all three within ±0.2 ms)

| component | ms/token | MB | **GB/s** | vs ~28 GB/s streaming |
|---|---:|---:|---:|---|
| gate+up matmuls | 20.01 | 481.7 | 24.1 | 86% |
| down matmul | 11.95 | 240.8 | 20.1 | 72% |
| LM head | 8.65 | 234.0 | 27.1 | at the ceiling |
| o matmul | 2.41 | 41.3 | 17.2 | 61% |
| q/k/v matmuls | 4.83 | 55.1 | 11.4 | 41% |
| **matmuls, total** | **47.85** | **1052.9** | **22.0** | 79% |
| SwiGLU activation (serial, scalar f64 `exp`) | 3.65 | — | — | not a stream |
| attention core (rope, KV, scores/softmax/AV) | 2.59 | — | — | |
| residual / norms / sample | ~0.4 | — | — | |
| **forward** | **54.43** | | 19.3 avg | |

The two smallest projections are fork/join-shaped (k and v are 256 columns each: 0.25 MB, ~16 columns per worker) — 172 µs/layer for
1.97 MB against ~70 µs at the ceiling. The activation runs serial because fanning it out separately loses
(`cpu-decode-attribution-2026-09-22-linux.md`): it is 3.65 ms of scalar float64 `exp`, 6.7% of the token, no DRAM traffic at all.

## 4. Where the gap to Ollama actually is: achieved bandwidth, not bytes

Ollama's per-token stream is its whole file for the tied-embedding 0.5B/1.5B (the embedding *is* the head): 491.4 MB and 1117.3 MB;
for the untied 7B ~4377 MB (the 4683 MB file less the ~306 MB token-embedding table it only looks a row up in — an estimate).
Same-session served tok/s (§6) × those bytes:

| | goinfer streams | goinfer GB/s (new) | Ollama streams | **Ollama GB/s** |
|---|---:|---:|---:|---:|
| 0.5B | 360 MB | 16.9 | 491 MB | **28.1** |
| 1.5B | 1053 MB | 20.4 | 1117 MB | **26.6** |
| 7B | 4624 MB | 23.1 | ~4377 MB | **~26.3** |

goinfer moves **fewer** bytes per token than Ollama on the 0.5B (0.73×) and 1.5B (0.94×), and ~6% more on the 7B; **Ollama runs at
86–92% of the 30.5 GB/s ceiling on every size** (llama.cpp's 27.1 tok/s on the 1.5B in the 2026-09 peer matrix is 30.3 GB/s — the ceiling
itself). At Ollama's 26.6 GB/s the 1.5B's 1053 MB would take ~39.6 ms = ~25.3 tok/s, i.e. **~1.06× Ollama**. So the remaining CPU-decode gap
is not format or bytes: goinfer's matmuls average 22.0 GB/s (79% of the ~28 GB/s all-thread streaming rate) and ~12% of the token streams
nothing at all.

Reading the premise this work started from — "the remaining cost is in the MLP matmuls, not dispatch" — against this:
**half right.** The MLP matmuls are the largest bucket (32 of 54 ms) but gate+up already stream at 86% of the ceiling; the structural
losses are the small q/k/v/o projections (fork/join-limited, 11–17 GB/s), `down` (20.1, unexplained), and the serial activation. What the
killed worker pool showed is true and unchanged — a spinning pool loses because the *serial stretches* it steals cores from are 6.6 ms of the
token; this work removes one of them instead.

## 5. The change: one fork/join for gate+up+SwiGLU (`decoder/cpu_gateup_fused.go`)

Each worker computes gate **and** up for its own column chunk and then applies `swiglu` to that chunk, so a layer pays one barrier
instead of two and the activation's exp runs in parallel inside it. Everything is per-column or elementwise: **bit-identical, not close.**

**Gates.** (i) `TestGatedMLPFusedGateUp_bitIdentical`: every element `!=`-compared against the unfused path across widths that do and do
not divide N (1,2,3,5,8,16,default), ragged N (517), tiny N; **mutation-checked** — dropping the last column and skipping the activation
each go red. (ii) `TestCPURoofline_fusedGateUp_logitsBitIdentical`: on the real 0.5B / 1.5B / 7B, 48 decode steps × ~152k logits, captured
through the sampler's own `LogitProcessor` in both arms and compared with `!=`: **~21.9M values, 0 differ** (a matching argmax proves little —
CLAUDE.md's LFM2 lesson). (iii) the paired harness fails on any greedy-token divergence; none occurred. (iv) `TestForwardN_matchesSequential`,
`TestSpeculativeGreedyParity`, `TestDecodeParityInt4`, `TestSession_reuseParity`, and the goldens-gated `refresh_parity_hashes.sh`
(62 goldens passed / 0 skipped / 0 failed on amd64) all green.

**Decision rule — and how it was chosen (disclosed).** The rule applied is the Linux-fix bar already on record in
`cpu-decode-attribution-2026-09-22-linux.md` for R9's two shipped fixes: *bit-identical AND a paired in-process ABBA ≥ 3% on the 1.5B at
depth 128 ships; 1.5–3% parks; no other size regresses > 2%*. **I selected that rule after the 0.5B and 1.5B A/Bs were in and before the 7B
ran**, not before any number existed; the other bar on record for this class of lever — R-06's ≥15% ship — would have *parked* this
(1.066×). The fused path is the same kind of change as R9's activation fix (which shipped at 1.189× on the R9 bar), which is why I used R9's,
but it is a choice between two bars made with data in hand, so it is the owner's to overrule.

**Paired ABBA, in-process, one loaded model, 5 pairs, median** (`…-ab-2026-09-23.log`):

| | fused vs plain | fused on top of R-06 batch | R-06 batch alone (plain → batch) |
|---|---:|---:|---:|
| 1.5B | **1.066×** (1.039–1.073) | 1.049× (1.044–1.053) | 1.053× (four pairs 1.049–1.054, one OFF spike) |
| 0.5B | **1.113×** (1.089–1.147) | 1.103× (1.067–1.159) | 1.039× |
| 7B | **1.029×** (1.028–1.030) | 1.023× (1.019–1.029) | not run |

Passes on the 1.5B (≥3%) and no size regresses. It ships **default-on on non-arm64 only** (`fusedGateUpDefault`,
`cpu_tuning_{arm64,other}.go`): measured on amd64 alone; arm64 stays off until the Mac measures it (an mmap'd `.giw` is canonical and would
otherwise take the path unmeasured). `GOINFER_CPU_FUSED_GATEUP=0` forces the old path.

## 6. Served, same session, interleaved (`…-served-2026-09-23.{log,json}`)

`scripts/bench_peer.py`, phase A, depth 128, greedy, 64 tokens, n=2 runs/cell, idle gate, goinfer `d88ef960` (new) vs `ec6fec95` (the do-nothing arm,
built in a worktree) vs Ollama v0.32.5; tokens/chunk 1.000 (Ollama 7B 1.018):

| | goinfer new | goinfer old | Ollama | **new/old** | **new/Ollama** | old/Ollama |
|---|---:|---:|---:|---:|---:|---:|
| 0.5B | 46.9 | 37.8 | 57.1 | **1.241×** | **0.821×** | 0.662× |
| 1.5B | 19.4 | 17.1 | 23.8 | **1.135×** | **0.815×** | 0.718× |
| 7B | 5.0 | 4.9 | 6.0 | **1.020×** | **0.833×** | 0.817× |

**The served gain on the 0.5B and 1.5B is larger than the in-process A/B predicted (1.241× vs 1.113×, 1.135× vs 1.066×) and I have not found
why.** The 7B agrees (1.020× vs 1.029×). Candidates, none tested: the served path's goroutine/GC environment differs from the test's; the old
arm's two barriers cost more under the HTTP server's concurrent goroutines. Both readings clear the bar; the served number is the one to quote for
"how far from Ollama", the in-process one for what the change does to a token. Run-to-run: old/Ollama moved 0.739 → 0.718 on the 1.5B versus
the 2026-09-22 record (the ~3.5% session drift this repo already documents).

## 7. What is left, ranked by measured upside (1.5B, ms/token; not built)

1. **Matmul efficiency, 22.0 → ~26.6 GB/s (Ollama's whole-token rate): up to ~8 ms.** Concentrated in `down` (20.1 GB/s — same matrix size as
   gate/up but N=1536 columns × K=8960; *why* it streams 17% slower per byte is unexplained), `o` (17.2) and q/k/v (11.4).
2. **R-06's q/k/v batch: measured 1.053× / 1.039× (≈2.7 ms), bit-identical, still parked by its own ≥1.15× rule.** The roofline now gives it a
   mechanism (a 172 µs/layer q/k/v dispatch against ~70 µs at the ceiling) and the two levers stack (1.5B plain 56.46 → batch+fused 51.31 ms =
   ~1.10×; 0.5B ~1.17×; 7B ~1.04× — means across two paired blocks, not one interleaved measurement). Its marginal on top of the fused path was
   not measured. Flipping it is the owner's call under the bar it was registered against.
3. **f16 scales for W4A8**: −82 MB/token (−7.8% of bytes ≈ 3–4 ms) — a kernel + `.giw` format change in aikit, not bit-identical to today's
   weights. Not started.
4. The LM head is at the ceiling (27.1 GB/s); only fewer bytes help, and int4 there is a quality trade this repo has measured against.

## Not established

- Why the served gain exceeds the in-process one on 0.5B/1.5B (§6).
- Why `down` is slower per byte than gate/up (§7.1); the S-02 per-worker timestamps inside a real token are still not taken.
- Ollama's 7B per-token bytes (§4) are an estimate from the file size; the 0.5B/1.5B figures are exact (tied embeddings).
- arm64: the fused path is off there and unmeasured.
- The A/B baseline drifts between blocks (plain 1.5B read 55.3–56.5 ms across runs); only paired ratios are quoted.
