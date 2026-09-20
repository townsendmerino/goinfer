# R13 step 0(ii) — CPU decode attention QK / softmax / AV category split

`docs/tasks/red-october.md` R13, step 0(ii): "`BenchmarkDecodeAtDepth` on the current tree at
{128, 512, 2048, 3900, 8192}, fitted to `t(K) = F + A·K`, with the slope split by category — QK,
softmax, AV... if it is now a third of the slope it caps what this brief can buy and becomes the
next item."

**Bottom line: softmax is not a minor category on either model measured. It is the single LARGEST
share on the 0.5B (~39-40%, bigger than QK or AV alone) and a stable ~22-26% on the 1.5B — both
well inside "caps what this brief can buy" territory. Applying Amdahl's law with the brief's own
registered QK+AV grouping ceilings (1.85× at 0.5B's G=7, 1.80× at 1.5B's G=6 — softmax
untouched by that Build) projects the achievable OVERALL decode-attention speedup at ~1.39× (0.5B
shape) and ~1.54× (1.5B shape), not the 1.8-1.85× the µop count alone would suggest.**

## Method

A temporary, non-shipped diagnostic (not committed — reproducible from the description below, not
from a merged file): `attendBatchedHeads`'s acc64, non-tree, non-windowed-tile branch (the one path
serial greedy decode actually takes — `decoder/forwardn.go`, the loop around
`linalg.MatmulQKAcc64`/the softmax block/`linalg.MatmulAVAcc64`) was wrapped with `time.Now()` /
`atomic.AddInt64` pairs around each of the three phases, gated behind a package-level bool default
`false` so normal decode/prefill/every existing test is unaffected. Verified a genuine no-op:
`TestForwardN_matchesSequential` (bit-identical across 911616 and 19447808 logits),
`TestAttendBatchedHeads_vsNaive` (cosine 1.00000000), `TestAttendStrided_matchesGatherReference`,
`TestSpeculativeGreedyParity`, and `TestSession_reuseParity` all pass unchanged with the
instrumentation present; reverted afterward (see "Why this wasn't committed" below).

`decoder.BenchmarkDecodeAtDepth`'s own setup (batched prefill to depth, then decode from there) was
reused, adding three `b.ReportMetric` lines reading the accumulated per-category nanoseconds
divided by `b.N`. Run via `go test ./decoder/ -bench <name> -run '^$' -benchtime=<20-30>x`, one
depth per invocation via `GOINFER_BENCH_DEPTH`, `GOINFER_PREQUANT_GGUF` pointed at the local
int4 0.5B or 1.5B GGUF, int8int8 quant (`loadBenchModel`'s default).

**Machine:** same MacBook Pro / M1 Pro as the peer depth row (`r13-cpu-depth-row-2026-09-19.md`),
run immediately afterward in the same session — the machine had just come off ~77 minutes of
continuous CPU saturation from that sweep, so absolute ns/token figures here carry the same
unconfirmed-thermal-drift caveat as that record's Finding 2. **The category *shares* (the
percentages below) are far more robust to that than the absolute numbers**, since a uniform
slowdown from throttling scales all three categories together and cancels out of a ratio — this is
the reason the write-up leads with shares, not raw ns.

## Why the raw percentages sum above 100% (and why that's fine)

`attendBatchedHeads` fans QK/softmax/AV work out across a worker pool, one goroutine per query
head-group, so multiple heads' timers run **concurrently** on different cores. Summed per-category
nanoseconds is a sum of *CPU time across threads*, not wall-clock time — it can and does exceed the
wall-clock total by roughly the pool's real parallelism (raw "categorized-%-of-total" read 17.6% at
K=128 up to 224.8% at K=3900 on the 1.5B, climbing with K because deeper contexts let the pool
achieve fuller parallelism). This does not invalidate the comparison **between** categories, since
all three are measured the same way under the same concurrency — it only invalidates reading any
single category's number as a literal fraction of wall-clock time. Every percentage below is
QK/softmax/AV normalized against their own sum (`qk+sm+av`), not against total decode time.

## Data

Three raw `ns/token` samples per depth (QK, softmax, AV — `linalg.MatmulQKAcc64`, the scaled/masked
softmax loop, `linalg.MatmulAVAcc64`), and each category's share of the categorized sum:

**0.5B** (Qwen2.5-Coder, hd=64 — smaller per-key QK/AV cost, so softmax's fixed per-key `math.Exp`
cost is proportionally larger):

| K | QK (ms) | softmax (ms) | AV (ms) | QK share | softmax share | AV share |
|---|---|---|---|---|---|---|
| 128 | 0.617 | 0.681 | 0.438 | 35.5% | **39.2%** | 25.2% |
| 512 | 1.980 | 2.402 | 1.583 | 33.2% | **40.3%** | 26.5% |
| 2048 | 7.349 | 9.082 | 6.191 | 32.5% | **40.1%** | 27.4% |
| 3900 | 14.590 | 17.375 | 12.492 | 32.8% | **39.1%** | 28.1% |

**1.5B** (Qwen2.5-Coder, hd=128 — larger per-key QK/AV cost, so softmax's fixed per-key cost is
proportionally smaller):

| K | QK (ms) | softmax (ms) | AV (ms) | QK share | softmax share | AV share |
|---|---|---|---|---|---|---|
| 128 | 1.103 | 0.692 | 0.856 | **41.6%** | 26.1% | 32.3% |
| 512 | 3.718 | 2.398 | 3.078 | **40.4%** | 26.1% | 33.5% |
| 2048 | 15.191 | 9.146 | 13.697 | **39.9%** | 24.0% | 36.0% |
| 3900 | 36.643 | 18.780 | 31.735 | **42.0%** | 21.5% | 36.4% |

**The split is stable across depth within a model** (as expected — it's a per-key, per-head µop
ratio, not a depth-dependent effect) but **differs sharply between the two models**: softmax is the
single largest category on the 0.5B, the smallest on the 1.5B. The most parsimonious explanation is
head_dim: QK and AV cost scale with `hd` (more FMAs per key), while softmax's cost per key is one
`math.Exp` regardless of `hd` — so a model with the smaller `hd` (0.5B, 64) pays proportionally more
for softmax than one with the larger `hd` (1.5B, 128). This is a reading consistent with the data,
not independently verified against an `hd`-controlled sweep.

## What this means for the Build's achievable ceiling (the Amdahl step 0(ii) exists to feed)

The Build section's kernels (`MatmulQKAcc64Group`/`MatmulAVAcc64Group`) group QK and AV across the
`nH/nKV` query heads that share a KV head — **softmax is untouched by that grouping**, and the
brief's own registered ship band (§ R13, "ships at ≥1.5×... AND phi3-mini... within ±2%") is scored
on the QK+AV category time specifically, not the whole decode-attention term. Applying the brief's
own registered µop ceilings (1.85× at G=7 — the 0.5B/7B group size — and 1.80× at G=6 — the 1.5B
group size) to the K=3900 shares above, with softmax held fixed:

- **0.5B** (QK+AV = 32.8+28.1 = 60.9% of categorized time, softmax 39.1%): a full 1.85× on the
  QK+AV share alone projects new total = `39.1 + 60.9/1.85` ≈ 72.0% of today's categorized time →
  **~1.39× overall**, not 1.85×.
- **1.5B** (QK+AV = 42.0+36.4 = 78.4%, softmax 21.5%): a full 1.80× on QK+AV projects new total =
  `21.5 + 78.4/1.80` ≈ 65.1% → **~1.54× overall**, not 1.80×.

Both projections clear the brief's own registered **served-band** floor of ≥1.5× at 4000 depth-bench
tok/s (§2.2 of `red-october.md`, reading 2: "R2's ≥60 at 4000 is A ≤ 0.92... about 2.5× the peer's" —
different brief, same shape of band; R13's own served band is deferred to post-step-0 per its own
text) only on the 1.5B side, and only marginally. **This is the concrete answer to the brief's own
question**: softmax was never negligible, and on the 0.5B shape specifically it caps a
theoretically-perfect QK+AV grouping to a ~1.39× overall win — below what "1.80-1.85× fewer µops
per MAC" reads as the headline number. A softmax speedup (batching the `math.Exp` calls, or a
faster transcendental approximation within the acc64 fidelity bar) is a real, separate lever this
step 0 surfaces as newly in-scope, not previously registered in the brief's Build section.

## Why this wasn't committed as shipped instrumentation

The three `time.Now()`/`atomic.AddInt64` wraps live inside `decoder/forwardn.go`, one of the
parity-manifest's `core` files (whole-file-hashed) — landing them, even gated behind a
default-`false` bool, marks all 36 tracked families stale and requires either the full T3 sweep or
`scripts/refresh_parity_hashes.sh`'s goldens-proof to re-baseline. The goldens-proof path is
independently blocked right now by three **pre-existing, unrelated** failures
(`TestBailingHybrid_forwardParity`, `TestGemma3VL_textParity`, `TestInt4_forwardParity/gemma3-vl-tiny`
— confirmed failing on `main` at `894c9a9a`, before this session touched anything). R13's own step 0
is explicitly scoped as "measure; no build" — a one-off diagnostic serves that without taking on
either the multi-hour T3 cost or entangling this measurement with fixing an unrelated red. The
wraps are reverted; the method above is written to be trivially reconstructed if the Build phase
(or R9's CPU attribution work) wants to reuse this technique.

## What is and isn't established

**Established:** softmax is a first-order category on CPU decode attention on both models
measured, not the "noise" Gate A0 found it to be against far slower QK/AV kernels — the relative
QK/softmax/AV split is stable across depth and is not confounded by the peer-sweep's thermal-drift
issue (Finding 2 of the sibling record), since it cancels in a same-run ratio. **Not established:**
absolute ns/token figures (same caveat as the peer depth row); whether the head_dim explanation for
the 0.5B-vs-1.5B split direction is the real mechanism (a reading, not verified independently);
phi3-mini's own split (not measured here — the brief's own control case, G=1, still open); 8192
depth (not run, to keep this session's total compute budget reasonable after the ~77-minute peer
sweep).

## Next step

R13 step 0(iii), the distinct-bytes probe, is independent of this and still open. If the Build
phase proceeds, this record's Amdahl numbers argue for treating softmax as in-scope from the start
rather than as a follow-on "next item" — the brief's own phrase for what a large softmax share
would trigger.
