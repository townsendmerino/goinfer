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

**Apple M1 Pro (arm64):**

| arm | median ms/step | Δ vs baseline |
|---|--:|--:|
| baseline (gather) | 179.8 | — |
| K strided (QKᵀ) | 185.0 | **+2.9% — slower** |
| V strided (scores·V) | 173.0 | **−3.8%** |
| both | 171.8 | −4.4% |

## The sign flips on x86-64 — the decision is ARCH-DEPENDENT

goinfer re-ran the identical interleaved A/B (same function, model, context, quant; floor
characterized in advance from an A/A pass; **both polarities** run, since the transpose arm's
`vt` write pollutes cache for whichever arm runs next and a symmetric A/A floor can't see that)
on **AMD Ryzen 7 3700X (x86-64)** and got the OPPOSITE sign for V:

| shape (group) | floor (A/A) | forward | reversed | mean | transpose → strided | **abs. penalty** |
|---|--:|--:|--:|--:|--:|--:|
| 1.5B / 4096 (6) | 0.03% | +40.06% | +42.34% | **+41.20%** | 735 → 1030 ms | **295 ms** |
| 7B / 4096 (7)   | 0.82% | +23.52% | +23.46% | **+23.49%** | 1705 → 2106 ms | **401 ms** |
| 1.5B / 512 (6)  | 0.77% | +12.11% | +14.66% | **+13.38%** | 168 → 188 ms | **20 ms** |
| 0.5B / 512 (7)  | 0.47% | +5.21% | — | +5.21% | — | — |

Both polarities agree in sign at every shape — the effect follows the path, not the interleave
slot. The 7B point is the confirming shape the original sequencing asked for, and it lands against
BOTH standing predictions:

- **P1's original expectation — "7B should favour V more"** (the transpose grows with context, the
  model is more bandwidth-bound) — is refuted. 7B is **+23.5% SLOWER**.
- **The cache-line account's first phrasing — "7B should be worse than 1.5B's +41%"** — is refuted
  AS PHRASED, and the phrasing was the error. The account predicts LINE-BYTES, which map to
  absolute time; it says nothing about a percentage whose denominator is free to move. Read
  absolutely, the penalty is monotone — **20 → 295 → 401 ms** — exactly as the model claims. At 7B
  the step time nearly TRIPLES (weight streaming for 7B params dominates), so the same-or-larger
  absolute cost divides down into a smaller percentage.

**Consequence for anyone reading a percentage here:** on x86-64 the strided read's cost tracks
`nKeys × group × line-bytes` and is essentially independent of model size, so **a bigger model
HIDES this regression rather than easing it.** 7B's milder-looking 23% is the denominator moving,
not the defect shrinking — its absolute penalty is the largest of the three. Percentages are the
wrong unit for comparing this cost across model sizes.

The kernel is not in question: `MatmulBTAcc64Strided` is byte-exact (bit-identity gate passed on 48 greedy
tokens + the int8-KV and f32-KV global-cache branches, mutation-verified). Same operand, opposite
sign, so **the decision is a property of the machine, not the algorithm.**

### The governing variable: `bElemStride` relative to the cache-line size

The strided V read has `bElemStride = kvDim` — 256 floats (1 KB) at 1.5B, 512 at 7B. A strided
read is competitive ONLY when consecutive `k` share a cache line; at `bElemStride = kvDim` they
never do, so every element read lands on its own line. Line-bytes touched at 1.5B/4096:
**14.68 MB (transpose) vs 201.33 MB (strided), ~13.7×.** What differs between the two boxes is
whether that 13.7× is survivable:

- **arm64 / M1 Pro:** 128 B lines, ~200 GB/s unified memory, stronger prefetch → the strided read
  is absorbed and the transpose (an expensive strided *write*) is the bigger cost → V wins (−3.8%).
- **x86-64 / Ryzen:** 64 B lines (half the useful bytes per line), weaker prefetch → the 13.7×
  line-bytes dominate → V loses, and worse as context spills to DRAM (+5% → +13% → +40% as
  nKeys grows from 512 to 4096). **64 B-line machines are the worst case for this stride.**

### Decision, correctly scoped

- **On arm64 / M1: adopt strided scores·V (−3.8%). Keep the K re-copy (+2.9%).**
- **On x86-64: do NOT adopt strided scores·V — it is a +40% decode regression at 4k.** Keep the
  transpose. (K stays a re-copy on both.)

The earlier decision line read as portable; it is not. `bElemStride ≥ cache-line-in-floats` is
the precondition, and it fails on any 64 B-line ISA with a kvDim-strided read. A consumer must
scope this to its target ISA, or default to the packed/transpose path and treat strided as an
arm64-only opt-in.

Why the isolated microbench (aikit `BenchmarkAttnStridedVsPacked`) was misleading in BOTH
directions: P1 first caught that it missed GQA amortization (it timed one transpose : one matmul,
not one transpose : `group` matmuls). It ALSO cannot see cache-line geometry — it ran on M1 only,
so it never exposed the x86-64 sign flip. A microbench that fixes neither is not evidence for a
portable decision; only the on-target end-to-end A/B is.

## Unexplained — flagging, not asserting

The M1 baseline is **179.8 ms/step**; the same model, context, and quant on the Ryzen box is
**~735 ms/step — ~4×**. More than ISA differs between these environments (thread count, memory
config, build). It does not affect either ratio (both are within-box interleaved A/Bs), but it
matters for which absolute number is representative of a shipping target — worth resolving before
either box's ms/step is quoted as "decode speed".

## Sequencing (revised)

1. **aikit** shipped `MatmulBTAcc64Strided` in **v1.18.0** (byte-exact, additive; nothing to
   revert — the primitive is correct, only the adoption is ISA-scoped).
2. **goinfer** must NOT ship strided scores·V on x86-64. Either gate the adoption on arm64, or
   keep the transpose everywhere until a per-ISA dispatch exists. Re-measure any new ISA on target
   with this interleaved A/B (floor characterized in advance, both polarities) before adopting.

## goinfer's disposition (2026-08-14) — DECLINED, transpose kept on BOTH ISAs

Not adopted, and the arm64 gate deliberately NOT built. `attendBatchedHeads` keeps the V_headᵀ
transpose and the K re-copy on every ISA; goinfer stays on aikit v1.17.1 (the v1.18.0 bump is
reverted — nothing here consumes `MatmulBTAcc64Strided`). Three reasons, in order of weight:

1. **The arm64 win is −3.8%, and collecting it costs a permanent second code path** that must be
   kept bit-identical to the first forever, on an ISA the goinfer box cannot measure. The two
   paths ARE bit-identical today — gated end to end, see below — but that is a property to be
   re-established at every future edit, on hardware not present.
2. **The ~4× baseline discrepancy is unresolved, and it sits directly underneath the number that
   would justify the gate.** M1 179.8 ms/step vs Ryzen 735 ms/step, same model/context/quant.
   Until that is explained, −3.8% is a delta on a baseline nobody has reconciled.
3. **Nothing is lost by waiting.** The prototype, the bit-identity gate and the A/B harness are
   preserved on branch `strided-v-scoresv` (`6ada6ec`, unmerged). Revisiting needs only an arm64
   box and a re-run; the decision is cheap to reverse and expensive to un-ship.

**What was established and is worth keeping regardless of the decision:**

- The adoption is CORRECT, not merely rejected on speed. Bit-identity was gated end to end — 48
  greedy tokens byte-identical to the transpose, plus the int8-KV and f32-KV global cache branches
  at 24 tokens each — and the gate is MUTATION-VERIFIED: reading the neighbouring KV head
  (in-bounds, different values) fails it at token 0, and the control passes.
- **A mutation that the code neutralises downstream proves nothing.** The first attempt (`bOff+1`)
  only tripped an out-of-bounds panic — it tested bounds, not identity, while looking like a
  passing check. Worth repeating because it is the failure mode of mutation-checking itself.
- The A/B harness runs **both polarities** by construction. Interleaving controls drift, but the
  transpose arm's `vt` write pollutes cache for whichever arm runs NEXT, and a symmetric A/A floor
  cannot see that asymmetry. One-polarity interleaving cannot separate "this path is slower" from
  "slot 1 is slower because slot 0 trashed the cache".
- Interleaving is only sound BECAUSE the paths are bit-identical: the token trajectory is the same
  whichever arm produced each step, so the arms cannot drift into different work. The identity gate
  is what licenses the measurement design, not just the correctness claim.

_The step-0 instrumentation and the M1 A/B harness were a labeled-temporary prototype (aikit
under a local `replace`); reverted after measuring. This doc is the durable record; the x86-64
numbers are goinfer's on the Ryzen box._
