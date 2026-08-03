# Task — batched-prefill attention (the long-context lever)

Status: **SCOPED, NOT STARTED.** Do not begin until the GEMV activation-staging fix lands (the
crossover — the product-urgent number — is a GEMV story; this is a separate, longer-context regime).
Same gates and bit-identity discipline as the GEMV work. Audited PTX untouched; new kernel in its own
file.

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

## Expected mechanism (to confirm, not assume)

Likely the same shape as the GEMV: **memory re-reads.** Each key/value at cache position `s` is read by
every query `m ≥ s` (causal), so the K/V cache is re-read O(M) times from L2/DRAM. The cache is **f32**
(4× the bytes of a quantized one), and per layer it is `nKeys × kvDim × 4` bytes re-read ~M times. If a
profile shows K/V read bandwidth near the L2 ceiling with the FLOP rate far below peak, that confirms
it, and the fix is to stage K/V tiles in shared memory so each is read once per query-block, not once
per query. Confirm before committing to that fix.

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

1. **Bit-identical (preferred, matches the rest of this lane).** Keep the exact three-pass, per-query
   reduction order; only change *where K/V are read from*. Tile queries: a block handles `Bq` queries ×
   one head and streams K/V tiles through shared memory, each query reducing over its own causal window
   in the same order as M=1. This shares each K/V load across `Bq` queries → up to `Bq×` less K/V
   traffic, while every query's arithmetic is unchanged. Shared-memory budget is the tension (K+V tile =
   `Bktile × hd × 2 × 4` bytes; hd up to 128 → keep `Bktile` modest, e.g. 32–64). This is the GEMV's own
   "read once, not N times" move, applied to K/V, and it keeps all three gates green.

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
