# CPU prefill vs Ollama — the first peer number this repo has had (2026-09-01)

**goinfer is 2.98× behind at K=512 and 1.80× behind at K=3900 — the gap NARROWS with depth,
because Ollama's CPU prefill degrades faster than ours. At the deepest interval the marginal
rates are 56.5 vs 87.8 tok/s.**

This is the opposite shape to the CUDA prefill re-anchor taken the same day, where goinfer's
marginal cost rises and Ollama's is flat.

## Provenance

| | |
|---|---|
| box | MacBook, Apple M1 Pro, 8 cores, 16 GB, macOS 26.6.2 |
| goinfer | `9d8e382` |
| peer | **Ollama v0.32.5** (homebrew) — the same version §B8 anchors CUDA decode against |
| model | qwen2.5-coder **1.5B** instruct **q4_K_M**, the SAME GGUF file on both sides (`ollama create` from `~/models/...`) |
| quant | goinfer **int4**, peer **q4_K_M** — weight-matched (see the correction below) |
| CPU forcing | Ollama `num_gpu: 0` — **not optional**; without it Ollama uses Metal and the "CPU" row is a CPU-vs-GPU comparison nobody can see |
| harness | `scripts/bench_peer_prefill.py --backend cpu`, 4 distinct prompts per cell, medians, engines interleaved with a server restart between |

## Result

TTFT rate = `prompt_tokens / TTFT`, what an interactive caller feels:

| K | goinfer | Ollama | ratio |
|---|---|---|---|
| 512 | 68.4 tok/s | 203.7 | **2.98×** |
| 1024 | 68.7 | 187.9 | **2.74×** |
| 2048 | 67.6 | 145.4 | **2.15×** |
| 3900 | 61.9 | 111.2 | **1.80×** |

**Marginal** cost per token (local slopes between adjacent depths — the overhead-free view):

| interval | goinfer | Ollama |
|---|---|---|
| 519→1031 | 69.1 tok/s | 173.7 |
| 1031→2067 | 66.6 | 118.2 |
| 2067→3919 | **56.5** | **87.8** |

**Scaling across 512→3900 (a 7.62× depth increase):** goinfer **8.35×**, Ollama **13.37×**.

## What it says

**The gap closes with depth, monotonically: 2.98 → 2.74 → 2.15 → 1.80.** goinfer's rate is
comparatively flat (68.4 → 61.9) while Ollama's falls hard (203.7 → 111.2). At the margin the
ratio is 1.55× (56.5 vs 87.8), and the scaling numbers say Ollama degrades ~1.6× faster than we do
over the same range.

**Both engines are superlinear on CPU**, which the harness detected rather than assumed: the
least-squares fit returns a *negative* fixed overhead for both (−1.7 s goinfer, −4.3 s Ollama),
which is impossible and is the signature of fitting a line to convex data. The harness flags it
`LINEAR_FIT_INVALID` and the local slopes above are the honest read.

**Ollama caches prompts aggressively on CPU** — repeat/fresh 0.05, 0.02, 0.01, 0.00 across the
four depths, i.e. a repeated prompt is up to 100× faster than a fresh one at K=3900. Every request
in this sweep carries a unique prefix. Reusing one prompt per cell, as a decode harness does, would
have compared our real prefill against the peer's cache lookup and produced a number roughly two
orders of magnitude wrong at depth.

## A correction made mid-measurement

The first CPU sweep ran goinfer at **`int8int8`** — chosen to mirror `benchmarks.md` §A's own
absolute table — against Ollama's native **q4_K_M**. That is 8-bit weights against 4-bit: **not a
peer comparison**, and the repo's own discipline is that both sides must run the same weights. It
was re-run at int4.

The int8 numbers are kept here because their difference is itself a finding:

| K | goinfer int8int8 | goinfer int4 |
|---|---|---|
| 512 | 90.7 tok/s | 68.4 |
| 1024 | 89.1 | 68.7 |
| 2048 | 84.7 | 67.6 |
| 3900 | 78.6 | 61.9 |

**int8int8 prefill is ~25–33% FASTER than int4 on CPU**, despite carrying twice the weight bytes —
the W4A8 unpack costs more at M>1 than the bandwidth saves. That is actionable on its own (a
prefill-heavy CPU workload should prefer int8int8) and it is why the mismatched first sweep
*flattered* nothing: it made goinfer look better than the weight-matched comparison allows.

## Not claimed

- **One model, one machine, one platform.** 0.5B and 7B are unmeasured on this row; so is x86.
- **Decode is untouched by this.** §B8's decode rows stand — the 2026-09-01 prefill changes (A3,
  P18, P19) all gate on `K ≥ 512` or on the batched-prefill loop, so decode at K=1 cannot reach
  them.
- TTFT includes each engine's fixed per-request overhead. Both fits reject a constant overhead
  here, so no floor is subtracted and the marginal slopes carry the caveat.
- No claim about *quality* — this measures rate, and the greedy continuations are not compared.
