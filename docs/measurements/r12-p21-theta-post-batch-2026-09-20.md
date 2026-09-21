# R12 P21 — Metal `ForwardN` batching was already shipped; freshly measured, still killed

**Result: R12's own citation ("`ForwardN` is a loop, not one command buffer") is stale — the
batching shipped 2026-09-16 (`a1640a6a`), a day before the "Θ=0.96, 20 cells" figure the brief's
own Standing section cites. Freshly re-measured today: real, measurable improvement over the
pre-batching numbers (0.5B: 1.020-1.048 → 0.860-0.861; 1.5B: 1.006-1.019 → 0.948-0.958), but every
cell still lands above the registered 0.8 kill line. The build is done; the decision this brief
registered for it is now actually applicable: KILLED, not "not started."**

## Why this needed checking

`docs/tasks/red-october.md`'s matrix (§1, "Speculative decode" row) and R12's own "Build (P21)"
section both describe `ForwardN` on Metal as still a per-token loop, citing
`docs/measurements/theta-per-backend-2026-09-01.md`'s original finding (Θ 1.006-1.048,
`T(n)/T(1)` exactly linear to n=16 — "what a loop predicts exactly"). That measurement is real and
was correct *as of 2026-09-01*. It is simply older than the code.

## What actually shipped, and when

`git log -S "sequences all N token forward steps inside a SINGLE Metal command buffer" --
metal/backend.go` finds exactly one commit: `a1640a6a` ("perf(decoder,gpu,metal): bump aikit to
v1.44.0; batch/pipeline forward optimizations", 2026-09-16). Its own commit message states
plainly: "Metal's `ForwardN` now batches all [N token forward steps into a single command
buffer]." `metal/backend.go`'s `ForwardN` calls `a.r.ForwardBatch(embeddings, startPos)` today
(confirmed by direct read, not assumed), and `VerifyPath()` reports `"batched layer-major
single-command-buffer"` for every non-paged-MoE family. The one place still describing the old
behavior in comment form is `metal/theta_probe_test.go`'s own doc comment — worth a follow-up fix,
out of scope for this record.

**The dates matter.** `a1640a6a` landed 2026-09-16. Red-october's own cited Metal Θ=0.96 figure is
dated "2026-09-17 sweep, 20 cells" — one day *after* the batching shipped. Today's 1.5B/depth-128
cell reproduces that figure almost exactly (0.958 vs the cited 0.96), which is strong corroborating
evidence that the 2026-09-17 sweep, whatever it was, *already measured the batched path* and
red-october's text simply never connected that number to the (differently-titled, aikit-bump-bundled)
commit that produced it — the same class of stale cross-reference as R9's own R-06 correction
earlier today, not a coincidence of phrasing.

## Method

Box: `apple-m1pro` (M1 Pro, 6P+2E, 16 GB). goinfer `d3ab9223`. `metal/theta_probe_test.go`'s
existing `TestThetaProbe_Metal` (`GOINFER_THETA_PROBE=1`), unmodified — the same instrument
`theta-per-backend-2026-09-01.md` used, so the two are directly comparable. Definition unchanged:
seed `depth` positions, time `ForwardN` over a width ladder {1,2,3,4,6,8,12,16}, Θ = least-squares
slope of T(n) / T(1). Models: qwen2.5-coder 0.5B and 1.5B, depths 128 and 512 (the same four cells
the original record used).

## Data

| model | depth | T(1) | Θ (today, post-batch) | Θ (2026-09-01, pre-batch) | T(16)/T(1) |
|---|---:|---:|---:|---:|---:|
| 0.5B | 128 | 6,407 µs | **0.861** | 1.020 | 13.95 |
| 0.5B | 512 | 7,429 µs | **0.860** | 1.048 | 13.86 |
| 1.5B | 128 | 14,018 µs | **0.958** | 1.006 | 15.39 |
| 1.5B | 512 | 15,152 µs | **0.948** | 1.019 | 15.22 |

`T(16)/T(1)` moved from ~16.1-16.8 (pre-batch, matching a plain loop exactly) to ~13.9-15.4
(post-batch) — real sub-linearity, confirming the command-buffer amortization is genuinely
engaging, not a measurement artifact.

**The improvement is much larger on the smaller model** (0.5B: Θ down 0.16-0.19, a 15-18%
reduction) **than the larger one** (1.5B: Θ down 0.05-0.07, a 5-6% reduction) — the same
fixed-cost-amortized-over-a-larger-token shape as today's earlier R-06/7B finding
(`w4a8-batch-7b-2026-09-20.md`): a per-command-buffer fixed cost matters proportionally less
against a token that's already more expensive on its own.

## Reading against the registered band

R12's own band: **Θ ≤ 0.6 ships; 0.6-0.8 parked; above 0.8 killed** ("batching did not move the
marginal verify cost, and the Metal small-M GEMM is the reason"). All four cells today (0.860,
0.860, 0.958, 0.948) are above 0.8. **Killed, on every cell measured** — not close to the parked
band, let alone shipping, despite the real and consistent improvement over the pre-batching
numbers. The brief's own parenthetical reasoning ("the Metal small-M GEMM is the reason") reads as
still correct: batching removed the K−1 command-buffer boundaries (the effect this record
confirms), but did not touch the underlying M=1-shaped GEMV kernel each verify token still runs
through, which is the larger remaining cost the brief itself already named as the likely blocker.

## Decision

**P21 is closed, not open.** The build already shipped; this record supplies the measurement the
brief's own decision rule needed to actually apply it. Speculative decoding on Metal stays
declined-by-default (Θ well above the ship/park bands at every model size and depth tested).
Re-opening it needs a different lever than command-buffer batching — the brief's own hint (the
small-M GEMM) is the natural next candidate, a genuinely different question from P21's own scope.

## What this doesn't establish

- Larger models (7B) or deeper contexts (2048+) — only the two sizes/depths the original record
  used were re-run, for direct comparability. A larger model would be informative given today's
  size-dependent pattern, but wasn't measured here.
- Whether the small-M GEMM really is the remaining blocker — named as the likely reason by the
  brief's own text, not independently verified by this record.
- CUDA/CPU — unaffected by this finding; their own Θ figures (CUDA 0.155-0.251, CPU 0.456-0.532)
  predate and are unrelated to Metal's `ForwardN` change.
