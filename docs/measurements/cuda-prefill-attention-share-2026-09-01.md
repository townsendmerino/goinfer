# CUDA prefill: attention is 55% at K=3900 and rising (2026-09-01)

**Step 0 of the fused-attention item, answered. Attention's share of CUDA prefill runs 5.3% → 55.0%
across K=128 → 3900, so a fused (FlashAttention-style) schedule has real headroom on DENSE models
at depth — and roughly none on MoE, where the expert FFN dominates. The deciding variable is model
class, not backend.**

## Provenance

| | |
|---|---|
| box | `nobara-pc`, RTX 2070 SUPER, driver **595.91.07**, Nobara 44 |
| goinfer | `1152c93` + the depth-override patch (committed with this) |
| model | qwen2.5-coder **1.5B** instruct q4_k_m, **int4**, DENSE (no MoE) |
| harness | `cuda/prefill_decomp_test.go` (`TestPrefillDecomp`) — the same instrument that produced the 2026-08-04 figure |
| method | category boundaries are stream syncs (`r.prof`), so the category sum runs over the pipelined wall time; that over-count is the price of per-kernel attribution and is reported in `catSum` |

## The result

| K | gemv | attention | glue | catSum | **attention %** |
|---|---|---|---|---|---|
| 128 | 74.9 ms | 4.5 ms | 5.5 ms | 84.9 ms | 5.3% |
| 512 | 307.7 ms | 53.4 ms | 12.8 ms | 373.9 ms | 14.3% |
| 2048 | 1.260 s | 827 ms | 35.1 ms | 2.123 s | **39.0%** |
| 3900 | 2.396 s | **3.007 s** | 62.5 ms | 5.466 s | **55.0%** |

**The instrument validated itself before the new cell was read.** K=2048 returns **39.0%** against
the **39%** recorded on 2026-08-04 — across a driver/distro re-anchor (595.58.03 → 595.91.07,
Nobara 43 → 44) and a month of changes. A number that reproduces after that much moved underneath
it is worth more than a fresh one.

The depth axis was added for this question: the fixed list stopped at 2048, attention is O(K²)
while the weight matmuls are O(K), so a share read at 2048 systematically understates the regime
the item is about.

## What it answers

**Step 0 — are scores materialized or fused? MATERIALIZED, and deliberately.** The tile in
`attendBatchedHeads` is over QUERY ROWS only; `scores` is `tile × nKeys`, the full key extent per
row. The code forbids the fused schedule in as many words:

> No key-dimension split happens here and none may: that would re-associate the softmax denominator
> and the AV fold, the exact thing acc64 exists to prevent.

and states G20's purpose as *"The point is memory, not speed"* — bounding scratch growth so the
worker pool can still fan out, not avoiding traffic. So the N-wide score row makes three trips
through memory per tile (written by QKᵀ, read-and-rewritten by softmax, read again by scores·V),
which is exactly the traffic a fused schedule removes. **The item does not close at zero cost.**

**The prize, priced.** Amdahl at K=3900 with attention at 55.0%: a *perfect* fusion caps at
**2.22×**; a plausible 2.5× on the attention term gives **~1.49×** end-to-end. That is a real
lever, and it is bought with the A1 guarantees (the running-max rescale re-associates, so a fused
path is not bit-identical — the same category as `--cpu-fast-attention`).

## The correction this measurement forced

An earlier reading of this item priced it off the 2026-09-01 full-model profile, where attention is
**17.4%** of prefill — concluding a 1.21× ceiling and a near-dead item. **That over-generalised a
MoE measurement.** Set side by side:

| model class | backend | K | attention share |
|---|---|---|---|
| **MoE** (Mellum2, 28L) | Mac CPU, f32+fanout | 8192 | **17.4%** (moeMLP 42.1%) |
| **dense** (1.5B) | Mac CPU, acc64 | 8192 | ~70% (the A3 trigger) |
| **dense** (1.5B) | CUDA | 3900 | **55.0%** |

Attention is dominant at depth on dense models on BOTH backends. The 17.4% is a property of a
model whose expert FFN takes 42%, not of the CPU backend or of prefill in general. **The deciding
variable for this item is model class.**

## Why this matters beyond the item

Today's CUDA prefill re-anchor found goinfer's marginal cost per prefill token **rising** with
depth — 0.377 → 0.932 ms/tok across the ladder — while Ollama's is **flat** at 0.064 → 0.063
(`cuda-prefill-reanchor-2026-09-01.md`). A rising marginal cost is the O(K²) attention term, and
this decomposition measures that term going 5.3% → 55.0% over the same range. A flat marginal cost
is what a tiled/fused attention produces.

So three independent measurements taken today agree: **the fused schedule is the mechanism behind
the CUDA prefill gap**, not the format/tensor-core question that §7's fork was argued on. That
fork's DEFERRED verdict is unaffected — it was about a constant factor — but the lever it declined
is not the one this points at.

## Not claimed

- **One dense model, one card.** MoE prefill on CUDA is not measured here; the Mac MoE profile
  says the expert FFN would dominate there, but that is a different backend.
- Depth stops at 3900 (`cudaCtxCap`).
- No fused kernel exists to measure. The 2.22× is an Amdahl ceiling from a measured share, not a
  measured win, and the distinction is exactly what this repo retracted twice today.
- The category sum exceeds pipelined wall time by construction (stream syncs at the boundaries), so
  the shares are of `catSum`, not of the wall clock.
