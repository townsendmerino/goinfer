# Task — batched-prefill attention (the long-context lever)

Status: **DESIGN REVISED (2026-08-04) — the clean ~1.3× is NOT bit-identical-buildable on Turing.**

> **The 64 KB wall (design-first finding, before writing the kernel).** Bit-identity pins the softmax
> denominator to `attn_batched`'s exact 128-strided partition + tree reduce (`ls += e` over keys
> {t, 128+t, …}). Preserving that order under tiling forces **Bk=128** (thread ↔ within-tile position ⇒
> the identical partition). But a Bk=128 K-tile at hd=128 = 128·128·4 = **64 KB — it maxes sm_75 shared
> alone**; no room for a V tile, and materializing `sc[Bq·nWin]` for the V-sum doesn't fit either. So
> sharing BOTH K and V across the query block (the full ~1.3×) is impossible bit-identically on Turing
> (Bk=128 K+V = 128 KB > 64 KB). **Achievable bit-identical:** stage K only for max/denom (shared across
> Bq), stage V only for the V-sum (which re-reads K from global to recompute scores, no materialized
> `sc`) — **~1.15–1.2×**, a big 3-pass, two-staging, causal-per-row, bit-identity-critical kernel. And
> prefill is already past its usability threshold (2.1 s TTFT) and cannot reach Ollama-parity regardless
> (§7 dp4a/IMMA ceiling), so ~1.15× does not change the competitive story. **Fund only as a focused
> campaign for its own sake, not a release lever.** The larger win needs Path 2 (tolerance-gated flash),
> which abandons bit-identity — a stated, separate decision.

Same gates and bit-identity discipline as the GEMV work. Audited PTX untouched; new kernel in its own
file. (Original scope preserved below.)

## Why (measured, not assumed)

`TestPrefillDecomp` (real qwen2.5-coder-1.5b) splits batched prefill into GEMV / attention / glue:

| prompt N | attention share |
|---|---|
| 128 | 8% |
| 512 | 25% |
| 2048 | **56%** |

Attention overtakes the GEMV past ~1–2k tokens. `attn_batched` is a naive O(M²) kernel: `GridX = nH`,
`GridY = M`, each `(head, query)` block runs the M=1 online-softmax over that query's causal window. At
M=2048 that is 2.6 s of the 4.6 s prefill. This lever is **attributed and needs no further measurement
to justify starting** — but the *mechanism* below is a hypothesis to confirm by profiling when the work
begins (the same `TestGemvBatchedBandwidth`-style isolation), not a conclusion to build on. This lane
has recorded three plausible-mechanism-as-conclusion attributions already; profile first.

## Profiled bound (ncu, 1.5B shape, M=2048) — CONFIRMED traffic, not latency

Measured `attn_batched` in isolation (nH=12, hd=128, M=2048, full attention): 92 ms, **1.6% of
compute peak**. ncu:

- **L1/TEX Cache Throughput 98.86% — saturated.** DRAM 0.45% (idle), L2 32.8%, Compute 11.5%.
- No Eligible 95.83%, 0.11 eligible warps/scheduler, **145.7 cyc/instr stalled on the L1 instruction
  queue (MIO throttle)** — this is *throughput* saturation of the memory pipe, NOT the scoreboard
  latency stall the GEMV had. Occupancy fine (86.8% achieved).
- Breakdown: **global K/V loads dominate — 305M load instructions, 3.7 B sectors, 21.96% bytes/sector**
  (4.5× wasted). The QK score pass splits threads over KEYS, so at a given `d` the 32 lanes read 32
  different keys at stride `kvDim` → fully uncoalesced. Shared score traffic (104M) is secondary.

**This is the opposite of the A-staging trap.** The GEMV was L1TEX-*latency*-bound with already-efficient
loads, so shared staging (a traffic fix) bought only 1.2×. Attention is L1TEX-*throughput*-saturated with
4.5× wasted sectors AND O(M) redundant re-reads (DRAM idle → the re-reads are L1-served transactions, not
DRAM bandwidth). Both are traffic the tiling fix removes: a coalesced load into shared cuts the 21.96%,
and sharing a staged K/V tile across a query block cuts the O(M) redundancy. The design below is
confirmed against the hardware, not assumed.

## Bit-identity — the constraint, stated up front

The current `attn_batched` is **bit-identical** to the M=1 `attention()` per row (gated by
`TestAttnBatched_bitIdentical`): same three-pass online softmax — (1) max over the key range, (2)
`exp(score − max)` and sum, (3) `Σ weight·V` — in the same per-query reduction order.

**A true flash attention is NOT bit-identical to this.** Flash fuses the three passes into one with a
running max `m_i` and running denominator `l_i`, rescaling partial outputs by `exp(m_old − m_new)`. The
result is mathematically equal but the floating-point accumulation order differs. That matters here
beyond the attention output itself: attention writes `ctx`, which flows `ctx → o-proj → residual → next
layer's norm → QKV → rope_kv`, so a non-bit-identical prefill attention changes the **K/V of every later
layer** — it would fail both the all-layers×all-rows KV gate and the 64-token byte-identical decode gate
(decode continues from a subtly different cache).

So there are two paths, and the choice must be explicit:

1. **Bit-identical (preferred, matches the rest of this lane).** Keep each query's exact reduction
   ORDER over keys; only change where K/V are read from and stream them once. A block handles `Bq`
   queries × one head and iterates key tiles `[k0, k0+Bk)`, loading each K/V tile into shared memory
   ONCE (coalesced — the load fixes the profiled 21.96% bytes/sector) and reusing it across all `Bq`
   queries (this removes the O(M) redundancy). The tension the profile forces: the current kernel
   materializes `sc[nWin]` scores per query between its passes, and `Bq × nWin` scores do not fit
   shared (nWin up to 2048). Resolution that stays bit-identical: **do not materialize scores — use
   two streaming passes over the key tiles.** Pass 1 streams all tiles and computes each query's `max`
   (reduction over keys, in order). Pass 2 streams the tiles again and RECOMPUTES each score `Q·K`
   (identical value — same Q, same K, same dot order), does `exp(score−max)`, accumulates the
   denominator and the `Σ weight·V`, all over keys in the same order as M=1. Re-dotting Q·K is compute
   (the kernel is at 11.5% compute — nearly free) and avoids the score buffer. This is flash-STYLE
   (tiled, streaming) but **without online rescaling** — the max is global before any `exp`, so the
   float order is byte-identical to `attention()`. Shared budget = one K+V tile (`Bk × hd × 2 × 4`;
   Bk=32–64) + small per-query reductions; independent of nWin. Keeps all three gates green.

2. **Tolerance-gated flash (only if path 1 is insufficient).** A real online-rescaling flash, which
   abandons bit-identity for prefill attention. Then the e2e gate must move from byte-identical decode +
   bit-identical KV to a cosine/near-tie gate (the repo's 3% rule, calibrated by a break-to-verify
   control table — the same discipline the MoE/GLM parity gates used). Decode itself still uses the exact
   M=1 `attention()`, so only the prompt's cache is approximate. This is a real option with a stated
   price; do not take it silently.

Default to path 1. Only fall to path 2 if profiling shows path 1 cannot reach the K/V-traffic floor
within the shared-memory budget.

## Gates (unchanged from the GEMV work)

- `TestAttnBatched_bitIdentical` — the new kernel reproduces M=1 `attention()` bit-for-bit, all rows,
  across startPos × window (path 1). Path 2 replaces this with a calibrated cosine floor.
- `TestPrefillLast_e2e` — KV bit-identical all layers × all rows, last-token logits bit-identical,
  64-token greedy decode byte-identical (path 1). Path 2 moves these to near-tie.
- New kernel in its own `.cu`; `moe.ptx` and the audited PTX untouched.

## Ordering

GEMV activation-staging fix → 26B non-expert half → this. The crossover is served by the GEMV; this is
the long-context regime and can wait behind the product-urgent number.
