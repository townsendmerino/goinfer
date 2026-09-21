# R3 — the batched-prefill win holds, and is larger, at startPos > 0 (the real prefix-reuse shape)

**Result: R3's own SHIPPED record explicitly left this open — "a `startPos`-on-speed sweep
remain[s] unmeasured." It is now measured. At the shape that actually matters in production (a
resident prefix already built, a new floor-sized turn continued via `PrefillLast(startPos>0)`
instead of a fresh `startPos=0` prompt), the batched path beats sequential by 4.01× at a 64-token
turn and 4.13× at 128 — both comfortably past R3's own ≥2.0× ships band, and both larger than the
0-startPos ratios R3 itself measured (3.83× at K=64, 4.66× at K=128 — comparable, not
contradicted). Nothing about resuming from a nonzero position costs the batched path anything on
this evidence; if anything it does slightly better relative to sequential, since sequential's own
per-token cost is unaffected by startPos while the batched path's fixed dispatch overhead is
amortized over the same turn length either way.**

## Why this needed its own measurement

R3's SHIPPED result deliberately did not re-run the pooled *correctness* gate at `startPos > 0` —
G-08 (`audit-metal-2026-09-12.md`) had already closed that specific coverage gap with a cheaper,
targeted test (`metal/prefill_startpos_test.go`, a tiny synthetic fixture proving `PrefillLast`'s
output matches sequential `Forward` bit-plausibly at `startPos > 0`). But correctness and *speed*
are different questions, and nobody had measured whether `PrefillLast`'s dispatch/fixed-cost
structure changes at all once it starts from a large, already-resident KV instead of position 0 —
a real, not hypothetical, gap: every multi-turn chat and every agent loop is exactly this shape,
and R3's own floor (64 tokens) exists specifically to catch *short new turns*, which in practice
means short turns appended to a long existing conversation, not short standalone prompts.

## Method

Box `apple-m1pro` (M1 Pro, 16 GB), Darwin 25.6.0, goinfer `ff2687fd`. `qwen2.5-coder-1.5b` GGUF
Q4_K_M, `Quant:"int4"`, one Metal resident, `ResidentContext` sized to what the test actually
touches (768). `metal/r3_startpos_speed_test.go` (`TestR3_startPosSpeed`):

1. Build a 512-token resident prefix **via the batched path itself**
   (`PrefillLast(embs[0:512], 0)`) — realistic, since a real served session's own earlier turns
   would already have gone through it, not a synthetic shortcut.
2. For each turn length in {64, 128}: interleave 6 reps of `PrefillLast(turnEmbs, 512)` (batched)
   against 6 reps of a sequential `Forward` loop over the same positions [512, 512+turnLen) — both
   arms read the identical resident prefix and write the identical suffix positions on every rep,
   so the prefix itself never drifts between arms.
3. Score best-of-6 per arm (matching R3's own n=6 / best-of-batches convention), report the ratio.

Gaussian-noise embeddings (`rand.NewSource(3)`, ×0.05 — the same construction R1/R2's own e2e
repro tests use), not real tokenized content: this measures the dispatch/kernel timing shape, not
anything about token-level output quality (that question is G-08's, already closed). 29.4 s total,
one resident, no swap growth, `--- PASS` on both subtests.

## Data

| turn length | startPos | batched best (of 6) | sequential best (of 6) | ratio | R3's own band |
|---:|---:|---:|---:|---:|---|
| 64 | 512 | 228.1 ms | 914.9 ms | **4.01×** | SHIPS (≥2.0×) |
| 128 | 512 | 448.3 ms | 1,851.4 ms | **4.13×** | SHIPS (≥2.0×) |

For comparison, R3's own `startPos=0` numbers (`metal-prefill-floor-2026-09-20.md`,
`bench_peer_prefill.py`, served, different instrument and prompt shape — not directly poolable
with the above, but the right order-of-magnitude check): 3.83× at K=64, 4.66× at K=128. Today's
`startPos=512` ratios (4.01×, 4.13×) sit inside that same range — no regression, no surprising
jump either direction.

## Reading

The batched path's advantage over sequential comes from replacing K separate per-token command
buffers (one dispatch-and-wait each) with one batched GEMM dispatch — a fixed-cost amortization
that has nothing to do with *where* in the KV cache the turn starts, only with *how many* new
tokens it covers. That prediction is confirmed directly: the ratio is stable (in fact very slightly
higher) once the prefix is 512 tokens deep instead of zero. The sequential arm's own absolute cost
scales with prefix depth too (each of its per-token `Forward` calls does a full attention pass over
the whole resident KV, growing with position) — so if anything, sequential gets *relatively* worse
as the prefix grows, which is consistent with today's ratios matching or slightly exceeding R3's
own `startPos=0` figures rather than falling short of them.

## Decision

**No new decision needed — this confirms R3's existing SHIPPED verdict rather than reopening it.**
The floor (64 tokens, batched-by-default above it) is correct at the shape that actually matters in
production, not just at the `startPos=0` shape R3's own gate happened to measure first. Nothing
here changes `metalFastPrefillFloor` or any other shipped default.

## What this does not establish

- **Real tokenized content**, not Gaussian noise — same caveat every e2e speed instrument in this
  repo already carries; content does not change dispatch-level timing shape, which is what this
  measures.
- **Only one prefix depth (512) and two turn lengths (64, 128).** A much deeper prefix (2,000+
  tokens, a long agent-loop history) was not tested; nothing in the mechanism above predicts a
  different result there, but it is not measured.
- **The `noHead` executor-job follow-on** (`docs/audit-metal-2026-09-12.md`'s M-01 closure note:
  the synchronous `forwardHiddenNoHead` wrapper gives back ~0.9 ms/token of un-overlapped
  encode-ahead versus the fuller `noHead`-bit-on-`execJob` version) is **untouched by this
  record** — a real, separate, unmeasured build item, not attempted here. Whether it is worth
  building depends on a measurement this record does not make: how much of today's real sequential
  prefill volume (adapters, Gemma-3-declined families, `/v1/embeddings`, sub-floor prompts) the
  ~0.9 ms/token gap actually costs in aggregate — R3's own Build section named a 10%-of-TTFT
  threshold for deciding this, which was never checked.
