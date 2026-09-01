# Task (goinfer + aikit): A1 — restructure decode attention for speed, changing no numerics

> **For:** Claude Code, in `~/tmcode/goinfer` (with sibling `~/tmcode/aikit` — kernel work lands
> there). Written 2026-08-23. This is the implementation work order for **option A1 only** of
> `docs/task-attention-decode-cost.md` — read that doc first; it carries the measurements, the
> invariant enumeration, and the closed Gate A0. **A2/A3 are NOT in scope.** The deliverable is
> the three A1 moves, measured after each, with every bit-identity gate green and **zero golden
> churn** — plus results appended to the campaign doc, negatives included at full value.

## The one constraint that is the whole design

Every attention output must be produced by **the exact same f64 fold, in the exact same order,
as today**. Not "within tolerance" — bit-identical. The guarantee this preserves (spec-decode
verify == sequential greedy; decode == prefill; MoE router stability) is enumerated in the
campaign doc's § "The invariant, precisely". Concretely:

- Parallelism may only split **independent outputs** across workers/registers — heads, layers'
  KV groups, individual QK scores (one dot per key), individual AV dims (one fold per dim).
  Never split, chunk, or reassociate any single output's reduction. This is
  `docs/task-decode-splitkv-attention.md:36`'s principle, applied on CPU.
- Gates that must pass unchanged, with no golden regeneration of any kind:
  `TestForwardN_matchesSequential`, `TestSpeculativeGreedyParity`, the full goinfer parity
  suite, and aikit's own `linalg` suite. If a change needs a golden updated, the change is
  wrong — stop and re-read this section.
- New kernels get **exact-equality tests** (`==` on every output, not rel-err) against the
  current path (`MatmulBTAcc64Strided`), across random + stress shapes covering every residue
  of nKeys and hd modulo the interleave width. `TestAttendStrided_matchesGatherReference` is
  the house pattern to copy.

## What is already measured — build against it, do not re-derive

From the campaign doc (Gate A0, closed GO, reproduced within noise):

- Attention is ~11.10 ms/token at 1.5B, depth 130 (404.1 µs/layer isolated), at ~1.0 ns/MAC —
  serial f64 FMA-chain speed. Component split: **QK 47.1%, AV 48.9%**, softmax 3.6%, RoPE 0.5%.
  QK and AV are co-equal: a change that fixes only one caps below 2x.
- **Single-threaded** — `attendBatchedHeads` (`decoder/forwardn.go:585`) is plain nested loops.
  aikit's `MatmulBTAcc64Strided` has a `parallelCols` path, but these calls (~16,640 MACs at
  depth 130) sit ~1000x below its threshold — the second confirmed instance of the
  `int4ParThreshold` bug class. **The fix is NOT lowering that threshold** (over-parallelizing
  tiny calls is the already-learned failure mode); it is parallelizing across the 672
  independent per-head calls per token — move (a).
- AV reads `vals` with element stride `kvDim` (decoder/forwardn.go:761) — one cache line per f64 MAC.
- Depth curve (before-baseline, keep for the after-comparison): depth 128 → 10.93 ms; 8192 →
  858.8 ms (attention = 95% of decode, ~1.15 tok/s). Per-key cost drifts +19.5% over that
  range (cache-tier effect).

## The three moves — suggested build order (c) → (b) → (a), measuring after each

**(c) AV loop-order fix — new aikit kernel, likely the simplest win.** Replace the AV call's
strided fold with keys-outer/dims-inner: hd (=128) f64 accumulators, one pass over V rows read
contiguously. Each dim's accumulator sees keys in today's exact ascending order ⇒ bit-identical
by construction, while the 1KB-strided element reads become sequential row streaming. This is
also the move that should claw back part of the +19.5% depth drift. Check first whether
`MatmulBTAcc64`/`Strided` are pure Go or asm and match that convention; either way the fold
order per output is the contract, not the implementation language.

**(b) QK output-interleaving — new aikit kernel variant.** QK's per-key dots are independent
outputs whose chains currently serialize on f64 FMA latency. Run 4 (or 8 — measure) keys' dots
as concurrent accumulator chains in one pass; each dot's internal d-order stays sequential
ascending. Handle the nKeys tail scalar, and test every nKeys mod-width residue. AV after (c)
already has 128 independent accumulators in flight, so (b) is primarily QK's move.

**(a) Thread across independent chains — goinfer-side, wraps (b)+(c), depth-aware.** At K=1
the (kvh, g) loop bodies are independent per query head: ~34 µs each at depth 130, growing
linearly with depth — comfortably above goroutine fan-out overhead. Distribute the 12 heads
across workers (prefer aikit's existing parallel-width machinery / a persistent pool over
per-call goroutine spawning; default worker count should respect the P-core finding — 6, not
GOMAXPROCS=8, per the mac-cpu measurement doc). Two sub-points:

- **Scratch is currently shared.** `scr.attnBatchBufs` hands out one set of qh/kh/vt/sc/ch
  buffers; per-worker instances are required. Audit the local-ring path in
  `decoder/attention.go` too — it deliberately defers the KV write past the read; threading
  must not reorder that. KV reads themselves are read-only and safe to share.
- **Depth-aware second axis.** At large depth the per-call work crosses the threshold where
  within-call splits become economic: QK across key-tiles (independent scores), AV across
  dim-tiles (independent folds). Gate this on measured per-call MACs using the same
  threshold-style machinery aikit already has — measure the crossover, don't guess a constant.
  This is what lifts the ceiling past heads-only ~6x in exactly the depth regime where
  attention is 95% of decode.

## Measurement protocol (the campaign already paid for these lessons)

- Quiet box; no concurrent sessions (the STREAM/swap incident).
- Isolated benchmark: the plain `b.Run`-subtest shape that produced the baseline — **not**
  nested `testing.Benchmark()` calls (the recorded runaway-calibration hang).
- After each move: the isolated depth cells 128/512/2048/8192 against the baseline above.
- After all three: `bench_peer` end-to-end (decode-only, warm-up discarded, greedy, server
  restart between cells), 0.5B + 1.5B, depth ~128 and one long-context cell.

## Acceptance (from the campaign doc, restated)

- Attention component **≥3x** faster at depth ~130 (11.10 ms → ≤3.7 ms isolated).
- **All gates green, zero golden changes** — this is pass/fail, not best-effort.
- `bench_peer` 1.5B decode consistent with the campaign table (~17.0 → ≥21 tok/s from this
  lever alone; the combined-lever rows belong to the W4A8 campaign, not this task).
- Depth-8192 before/after reported (no hard target promised — measure and record; heads-parallel
  plus the depth-aware axis should deliver more than the depth-130 factor there).
- Results, including any move that measures negative, appended to
  `docs/task-attention-decode-cost.md` in the same style as its Gate A0 sections.

## Not in scope — do not drift into these

- **A2** (M-independent-order f32 kernel) and **A3** (gated fast path) — their own gates, later.
- **Tiling** for the residual cache-tier drift — named non-goal in the campaign doc.
- Softmax/exp changes, RoPE changes, anything on the W4A8/matmul side, GPU/Metal attention.
- Lowering aikit's parallelization threshold for tiny calls — second-instance finding; the fix
  is across-chains, not within tiny calls.
- Prefill/verify (M=K) fast-pathing beyond what (b)/(c) give for free — they share the kernels,
  and that shared speedup is welcome, but no M>1-specific work here.

## Done looks like

Three moves landed (or a documented negative for any that don't pay), each with its isolated
depth-cell measurement; the exact-equality kernel tests and all existing gates green with zero
golden churn; the before/after `bench_peer` table and the refilled acceptance numbers appended
to `docs/task-attention-decode-cost.md`; aikit and goinfer suites green; both working trees
clean and committed, aikit version bumped/tagged per its own release convention if its API
grew (the new kernels are additive — no existing signature changes expected).
