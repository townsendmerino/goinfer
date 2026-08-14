# P1 — KV re-gather cost, and the strided-matmul A/B (decode)

The decode attention (`decoder/forwardn.go` `attendBatchedHeads`) re-copies every key row and
re-transposes every value row of the KV cache into contiguous scratch on every token, because
aikit's matmul primitives need contiguous rows. This records what that costs and whether
aikit's new `linalg.MatmulBTAcc64Strided` (read the cache in place, no gather) is worth adopting.

Model: qwen2.5-coder, int8int8 (the shipping decode quant, `useAcc64`). Box: Apple M1 Pro.
aikit primitive under test: `MatmulBTAcc64Strided`, bit-identical to `MatmulBTAcc64` on a
packed/transposed copy (aikit `TestMatmulBTAcc64Strided_bitIdenticalToPacked`).

## Step 0 — the cost (execution-evidence, warm steady-state)

Measured with a counter inside the gather loop asserting it ran at the configured context
(`gatherRows == nKeys·nKV·layers·steps`), not merely that a long context was configured.

| model | context | gather ms/step | **share of per-token time** |
|---|---|--:|--:|
| 0.5B | 4k | 7.0 | 3.4% |
| 0.5B | 8k | 15.2 | 8.8% |
| **1.5B** | **4k** | **17.9** | **10.0%** |

The audit estimate was ~10–15% at 4k+. It did NOT hold at 0.5B (too small to be
bandwidth-bound — 207 ms/token ≈ 1 GB/s, far below the M1's ~200 GB/s), but at representative
size (1.5B, bandwidth-bound) it is 10.0% at 4k and climbs with context. The measured number
replaces the estimate.

## Step 4 — the A/B (interleaved, one session, warm-up discarded)

qwen2.5-coder-1.5B, ctx=4097, nKV=2, layers=28, hd=128, **GQA group=6**. Four arms interleaved
on one prefilled cache; 6 visits, first discarded; median ms/step over 5 retained visits ×
4 steps. Bit-identity gated separately (24 greedy tokens byte-identical, gather vs strided).

| arm | median ms/step | Δ vs baseline |
|---|--:|--:|
| baseline (gather) | 179.8 | — |
| K strided (QKᵀ) | 185.0 | **+2.9% — slower** |
| V strided (scores·V) | 173.0 | **−3.8%** |
| both | 171.8 | −4.4% |

**Decision: adopt the strided read for scores·V (V); keep the K re-copy for QKᵀ.**

Why the split — and why the isolated microbenchmark (aikit
`BenchmarkAttnStridedVsPacked`, which showed K a small win and V a large one) was
misleading: it did not model the GQA group. In real attention the per-head gather is done
ONCE and reused across the `group=6` query heads. So:

- **K re-copy is cheap and its packed buffer is reused cache-friendly 6×.** Replacing it with a
  strided read that re-walks the interleaved cache per group member is a net LOSS (+2.9%).
- **The V transpose is an expensive strided-WRITE serial pass** (`vt[d·nKeys+s]`, stride nKeys,
  write-allocate thrashing) with no comparable reuse benefit, so removing it wins (−3.8%) even
  though the strided V read is done per group member.

The net (~4%) is smaller than the 10% gather share because the strided read is not free — it
recovers the transpose but not the whole gather, and K would give it back.

## Sequencing (owed)

1. **aikit** ships `MatmulBTAcc64Strided` in a release (it is on aikit main; committed 7bca814).
2. **goinfer** bumps aikit and adopts it for scores·V ONLY, gated by the bit-identity token
   check and re-measured with this same interleaved A/B before it lands. A longer-context and a
   larger model (7B, group=7) are worth a confirming point — both should favour V more, since
   the transpose grows with context and the model is more bandwidth-bound.

_The step-0 instrumentation and the A/B harness were a labeled-temporary prototype (aikit under
a local `replace`); reverted after measuring. This doc is the durable record._
