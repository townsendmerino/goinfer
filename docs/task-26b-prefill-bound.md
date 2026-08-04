# 26B batched-prefill bound — DEFERRED, not cancelled (bound recorded so nobody re-derives it)

Status: **deferred** in favour of the attention lever (2026-08-03). The build is cheap when
funded — the bound and the mixed-M analysis below are the whole design. Deferred over the scoped
correctness-proof option (C) specifically because the attention rewrite touches `attn_batched`,
which the 26B prefill path consumes: a correctness proof built now would be re-gated after the
attention change.

## What the build is

Extend `PrefillLast` past its dense-only guard for **gemma4moe** layers: attention + the dense
FFN branch batched at M=len, the routed-expert branch sequential per token, per-token join. The
mixed-M analysis (already done) established that the join, the seven Gemma norms, and
`layer_scalar` are all per-row and batch cleanly. The real new work is batched versions of what
the current dense path guards out — **sandwich norms, K=V, attention softcap, per-layer
geometry** — plus the sequential expert loop and the join. Days-level, because gemma4moe is the
most complex forward in the backend.

## The bound (real shapes, gemma-4-26b-a4b)

Config: hidden 2816, 30 layers, 16 heads / 8 KV, head_dim 256 (qDim 4096, kvDim 2048), 128
experts top-8, moe_intermediate 704, dense intermediate 2112, all layers gemma4moe,
sliding_window 1024. Does not fit 8 GB — runs via host↔VRAM expert streaming.

Weight bytes at int4 (0.5 B/weight), per prompt token:

**Batchable (attention + dense branch), per layer:**
- q 2816×4096 = 11.53M, k 2816×2048 = 5.77M, v 5.77M, o 4096×2816 = 11.53M
- dense gate 2816×2112 = 5.95M, up 5.95M, down 2112×2816 = 5.95M
- = 52.45M weights/layer → ×30 = **1.573 G = 786 MB**

**Sequential (routed experts), per token, per layer:**
- top-8 × (gate 2816×704 + up 2816×704 + down 704×2816) = 8 × 5.95M = 47.6M/layer → ×30 =
  **1.428 G = 714 MB**

**Split: 52% batchable / 48% sequential.** Amdahl ceiling (batchable → 0) = 1500 / 714 =
**2.10×**.

## Realistic speedup vs prompt length

Model: batched proj ~5.8× faster than sequential M=1 (measured on the 1.5B: ~0.75 vs ~4.37
ms/token), experts unchanged (~3.97 ms/token, still per-token), attention an O(M²) term shared by
both paths (scaled ~2.86× from the 1.5B for the 26B's heavier heads: 16h×256 vs 12h×128, 30 vs 28
layers).

`T_seq/token ≈ 8.34 ms`, `T_bat/token ≈ 4.72 ms` (both excluding attention `A(M)`):

| prompt M | A(M) est. | T_seq | T_bat | speedup |
|---|---|---|---|---|
| 128 | 33 ms | 1101 ms | 637 ms | **~1.73×** |
| 512 | 469 ms | 4739 ms | 2886 ms | **~1.64×** |
| 2048 | 7450 ms | 24530 ms | 17116 ms | **~1.43×** |

**Materially under 1.5× at 2048, and shrinking with M** — the fixed 48% sequential-expert floor
plus the growing O(M²) attention cap it well below the 2.10× Amdahl ceiling. This is **plumbing
(a real ~1.6–1.7× at short/medium prompts, which is where the crossover lives), not a TTFT fix.**

Caveat on the numbers: the 786/714 split is exact from the shapes; the `BW_m1 ≈ 180 GB/s` and the
~5.8× proj factor are estimates, and `A(M)` is scaled from the 1.5B rather than measured on the
26B (loading the 26B is a ~5-minute host-pinned-alloc cost, not worth it for a bound). The
*direction* — ~1.7× at short prompts shrinking toward ~1.4× at 2048 — is robust to those
estimates; the Amdahl structure fixes it.

## Why attention was ranked over this

Attention is ~33× off its compute ceiling (1.5B M=2048: ~360 GMAC / 4.5 TMAC/s ≈ 80 ms ideal vs
2605 ms measured) against the GEMV's remaining ~2×, it is 66.9% of prefill at 2048, it is the
same work for both prefill paths (so this 26B build does not touch it), and it benefits every
family's long-context prefill. See `docs/task-prefill-attention.md`.

## Decode, and why the 26B is slow (capacity, not MoE)

The prefill bound above is only half the story; decode has the same root cause. The 26B-A4B decodes
at **16.98 tok/s** (capture-free, `benchmarks.md` §B4) — slow *only* because ~11.4 GB of experts do
not fit the 8 GB card, so the routed experts stream host→VRAM over **PCIe (~12 GB/s, ~30× slower than
VRAM)** every token. MoE is cheap by design (only ~4B of 26B parameters activate per token), so a
26B-A4B that FIT VRAM would decode *faster* than a dense 7B. The bottleneck is capacity, not the MoE
architecture and not the kernels — put it on a 16 GB+ card and it beats the dense 7B. This is why
"other MoE models run faster" reduces to "other MoE models fit," and why no kernel/IMMA work is
pointed at the 26B: the fix is memory.

## Operational lesson carried forward

Profile the unit before designing the fix, now that the profiler exists. The batched-prefill GEMV
gap was mis-attributed four times by inference before ncu stated it directly (see
`docs/benchmarks.md` §B2). The attention work starts with an `ncu` profile of `attn_batched`, not
a design.
