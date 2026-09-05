# CPU prefill vs Ollama — at parity, and past it on the marginal (2026-09-05)

**goinfer is 1.54× behind at K=512 and 0.91× at K=3900 — i.e. AHEAD at depth. The whole-curve
marginal ratio is 0.86×, goinfer faster. This supersedes the 2026-09-01 row (2.98× / 1.80×),
which was taken on aikit v1.31.0, before the S-01 register-blocked int4 tile.**

The same sweep was run twice on one box in one session, against the same peer, differing only in
which goinfer binary served: a Sep-1 build on aikit v1.31.0 and a current build on v1.34.0. That
makes the improvement a measurement rather than a subtraction across sessions.

## Provenance

| | |
|---|---|
| box | MacBook, Apple M1 Pro, 8 cores (6P+2E), 16 GB, macOS 26.6.2 |
| goinfer | `3b20f74` (post-tile arm) — built to `~/bench-cur/serve-cpu-v1.34.0` |
| | Sep-1 binary `~/bench-cur/serve-cpu` (pre-tile arm), linking aikit v1.31.0 |
| aikit | **v1.34.0** post-tile / **v1.31.0** pre-tile (verified with `go version -m` on each binary, not assumed from dates) |
| peer | **Ollama v0.32.5** (homebrew `/opt/homebrew/bin/ollama`) |
| model | qwen2.5-coder **1.5B** instruct **q4_K_M**, the same GGUF both sides |
| quant | goinfer **int4**, peer **q4_K_M** — weight-matched |
| CPU forcing | Ollama `num_gpu: 0` |
| harness | `scripts/bench_peer_prefill.py --backend cpu --models 1.5B --depths 512,1024,2048,3900 --n 4`, engines interleaved with a server restart between |
| box load | **loadavg 3.3–3.8, NOT idle** — ordinary desktop. Both engines carried it, so the RATIO is the trustworthy quantity and the absolute tok/s is not |

## The row

| K | pre-tile (v1.31.0) | post-tile (v1.34.0) | tile gain | Ollama | vs Ollama, pre | vs Ollama, post |
|---|---|---|---|---|---|---|
| 512 | 67.6 tok/s | **141.7** | 2.10× | 218.6 / 209.5 | 3.10× behind | **1.54× behind** |
| 1024 | 67.2 | **137.0** | 2.04× | 177.5 / 181.7 | 2.70× | **1.30×** |
| 2048 | 67.7 | **133.6** | 1.97× | 151.5 / 157.4 | 2.32× | **1.13×** |
| 3900 | 63.3 | **118.8** | 1.88× | 108.6 / 114.6 | 1.81× | **0.91× — AHEAD** |

Ollama figures are the post-tile-run / pre-tile-run values; they differ by 2–6% between the two
sweeps, which is session drift and is exactly why the engines are interleaved and why each arm
carries its own peer measurement rather than sharing one.

## Marginal rates — read these, not the fit

The linear fit is `LINEAR_FIT_INVALID` on CPU for both engines: TTFT is superlinear in K, so the
fitted intercept reads negative and `marginal_tok_s` is not a constant. The local slopes:

| segment | goinfer post-tile | goinfer pre-tile | Ollama |
|---|---|---|---|
| 519 → 1031 | 132.7 tok/s | 66.7 | 148.1 |
| 1031 → 2067 | 130.3 | 68.3 | 131.9 |
| 2067 → 3919 | **105.8** | 59.0 | **82.3** |

Whole-curve marginal ratio: **0.86× post-tile** (goinfer ahead), 1.69× pre-tile (behind).

## Why the old row can be trusted as a baseline even as it is replaced

The pre-tile arm re-measured today reproduces the 2026-09-01 row to within 4%: **3.10× at K=512
against the recorded 2.98×, and 1.81× at K=3900 against 1.80×**. So the two arms here are
comparable to each other AND to the historical row. The old number was right; it is stale, not
wrong.

## What changed, and the honest limit on attributing it

The dominant known change is aikit's **S-01 register-blocked W4A8 tile** (`dotW4A8Row4Tile4x4`),
which measures **2.88× single-core** on the kernel and lands at 1.88–2.10× here. But the two arms
are STACK-level builds four days apart, so they conflate the tile with whatever else changed in
goinfer over that window. This is not an aikit-isolated A/B, and should not be quoted as one.

The kernel-to-end-to-end compression (2.88× → ~2.0×) is **not** fork/join: aikit measured prefill
fan-out at 5.4× on six workers at M=512, so the barrier is amortised at prefill batch sizes. The
residue is the non-matmul prefill remainder — serial f32 transcendentals (aikit S-06), attention,
norms, sampler.

## Also superseded by this: the int4/int8int8 inversion

The 2026-09-01 note recorded **int8int8 CPU prefill 25–33% FASTER than int4**, because the W4A8
unpack cost more at M>1 than the halved weight bytes saved. The tile removed that unpack
repetition and the ordering has flipped: int4 now beats int8int8 by 1.10–1.14×. Do not carry the
old inversion forward.

## Cache health

goinfer repeat/fresh 1.01–1.04 across all depths (no prompt caching). Ollama 0.05 → 0.00 (caches
hard, but every prompt here carries a unique prefix so it misses — the comparison is prefill
against prefill, not prefill against a cache lookup).

Raw: `/tmp/peer-cpu-v1.34.0.json`, `/tmp/peer-cpu-v1.31.0.json`.
