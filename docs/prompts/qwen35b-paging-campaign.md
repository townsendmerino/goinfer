# Task (goinfer): paged-MoE decode — attack the 70.5%, with the diagnostic as the before

> **For:** Claude Code, in `~/tmcode/goinfer`, on the 16 GB M1 Pro. Written 2026-08-24, from
> the 35B diagnostic (`0081d3e`, recorded in `docs/task-zeno-compare.md`): steady-state
> ~1.2-1.4 tok/s; split **paged MoE ~70.5%** (I/O-dominated: ~25 GB read for one 79-token
> request — 1.5x the 16.36 GB expert pool — against a 6 GB budget), DeltaNet ~19%, LM head
> ~7%, attention ~3.5%. This campaign is the MoE share only. **DeltaNet is the named
> follow-on, not this task.** The pager is shared with the gemma4 26B config (16.98 tok/s,
> 81.6% hit) — that cell is a regression gate here, not a target.

> **PRIOR-ART CORRECTION (added 2026-08-24, before any work started):** this brief went out
> without a prior-art sweep of `docs/task-moe-streaming.md` — an existing, dated investigation
> of this exact `expertPager`/`SpanCache` machinery. That gap invalidated two of the four levers
> below (see the inline notes at Step 0.1 and Lever 2/4). Caught at zero cost by a check-before-
> spending pause, not by running the dead work. See `[[qwen35-paging-campaign-rescoped]]`
> (memory) for the full account. **Every future perf-campaign brief in this thread gets a
> prior-art section before it's finalized — that's the lesson this one is paying for.**
>
> **What survives, rescoped:** Step 0.1 and Lever 4 collapse to citations (below). Lever 2 is
> already the status quo on darwin (nothing to build). The read-rate probe below confirmed
> page-fault-driven reads ARE queue-depth-1-bound (0.32→1.79 GB/s at QD8, 5.5x) — but a direct
> admit-time I/O-vs-compute split (`docs/task-zeno-compare.md`, "the admit-time I/O-vs-compute
> split, measured directly") then found the MoE bucket is **86% compute, only 14% I/O** — so that
> 5.5x barely moves the total (1.09x end-to-end).
>
> **REVISED VERDICT (2026-08-24, after a fifth, finer split):** the int32-per-group GEMV was named
> as "the next opening fact" above on the strength of "compute-dominated" alone — but that left
> compute a location, not an attribution (Francis's own catch: ~465ms of compute over the touched
> bytes was ~3GB/s, an order of magnitude under the canonical kernel's own ~40GB/s). A finer split
> (router/gather/GEMV-by-shape/shared, `docs/task-zeno-compare.md` "compute was a location, not an
> attribution") found 91.6% of compute is genuinely GEMV time, ruled out the threshold-bug-class
> hypothesis by direct A/B (forcing serial made it SLOWER, not faster), and found the real
> mechanism by reading the code: paged `.giw` MoE experts never get the arm64 split-half kernel's
> load-time repack (`decoder/weightmat.go`'s own documented, deliberate carve-out — a paged
> expert's repacked heap copy would pin memory paging exists to bound) — so every per-token expert
> GEMV runs the OLDER, canonical kernel, never the shipped 1.6-1.75x-faster one. **CAMPAIGN CLOSED
> on BOTH the read-path/owned-buffer-pread lever AND the int32-per-group GEMV rewrite as the next
> step — neither is what the data points to.** The lever the split actually picks: wire the
> already-shipped, already-bit-identical split-half kernel to the paged path — projected ~1.3x
> end-to-end, bigger than the pread lever's own 1.09x, zero golden churn (vs. the GEMV rewrite's
> full T3 re-validation for an unmeasured payoff). The pread architecture is not dead, just
> reframed: its value is as the plumbing that lets a paged fill be repacked in place (an owned
> copy, not an mmap alias), not as an I/O speedup on its own.

## Step 0 — three measurements that decide which levers matter (before changing anything)

The diagnostic's numbers imply ~316 MB/token faulted (25 GB / 79) at an effective ~0.6 GB/s —
so BOTH the bytes and the rate are candidate levers, and their relative sizes decide the
campaign. Measure:

1. **Expert-usage skew — ALREADY MEASURED, cite don't re-derive.** `docs/task-moe-streaming.md`'s
   Lever 2 section already ran this on a real 35B-A3B trace: hottest 10% of experts absorb 72%
   of accesses, hottest 25% absorb 94%, half the universe never touched. Use these numbers for
   the ceiling math below instead of re-histogramming.
2. **Effective read throughput and shape.** Fault sizes and whether expert reads are
   contiguous or scattered; compare the observed ~0.6 GB/s against the SSD's sequential rate
   (measure it once, same file). The gap between those two numbers is the read-path lever's
   entire size.
3. **Bytes genuinely touched per token** (dense + shared + routed experts, at real top-k) —
   the diagnostic derived ~316 MB *faulted*; this pins the demanded number the hit rate is
   computed against, and bounds the best case: at 100%-hit the MoE share collapses to its
   compute, which the stub split already measured.

**Write the ceiling math into the doc before building:** given measured skew, budget B, and
SSD rate R, the achievable faulted-bytes/token and hence tok/s ceiling for this box. If that
ceiling comes out under ~4 tok/s, say so before spending the week — a physics-bounded result
honestly recorded beats an optimization campaign against a wall.

**The ceiling math's premise is corrected too:** "budget B" is not a real constraint on this
darwin box — the pager's budget is bookkeeping only (`aikit/mmap/madvise_darwin.go`'s eviction
call is a no-op here; see the prior-art note above). The actual constraint is the 16.36 GB pool
against macOS's Unified Buffer Cache under real system memory pressure, and misses are
structural given that pool size on a 16 GB Mac. The ceiling math should be built around
**effective read rate** and **overlap**, not a budget sweep.

## The levers, cheapest first — each behind its own measurement

1. **Budget.** 6 GB is conservative on a box where the process sits at 2.7 GB resident. Sweep
   6 / 9 / 12 GB: hit rate, tok/s, peak footprint, and macOS memory-pressure state at each
   (the 16 GB Mac safety story is why the budget exists — raising it is a measured tradeoff
   against pressure/jetsam, not a free knob; record pressure, don't just tok/s).
   **CAVEAT (prior art):** on darwin the budget doesn't enforce a physical cap at all (see
   above) — expect this sweep to move the pager's own hit/miss bookkeeping without necessarily
   moving real disk I/O by the same amount, since actual residency is decided by macOS's UBC
   under system pressure, not by this number. Still worth running (it's cheap and the
   footprint/pressure side is real), but don't expect it to be the campaign's answer.
2. **Eviction policy — ALREADY THE STATUS QUO on darwin, nothing to build.** `MADV_DONTNEED` is
   a no-op on this platform (`aikit/mmap/madvise_darwin.go`, confirmed current at v1.26.0) — it
   does NOT purge the OS page cache, contrary to this lever's original framing (which was the
   same wrong assumption the diagnostic's own saved memory made, corrected together
   2026-08-24). There is no "evict without DONTNEED" experiment to run here: that's already
   what happens. Skip this lever entirely.
3. **Read path.** If Step 0 shows scattered small reads well below SSD sequential rate:
   expert-sized contiguous reads, and prefetch overlap — the router picks a layer's experts
   *before* their matmuls run, so `WILLNEED`/read-ahead for the chosen experts can overlap
   fault I/O with the current layer's compute. Bounded, mechanical, and multiplicative with
   1-2.

   **Prerequisite probe (added 2026-08-24, run this FIRST, before any buffer-ownership work):**
   page-fault-driven reads are effectively queue-depth-1 I/O — a fault stalls the touching
   thread, so the SSD never sees real parallelism. This plausibly explains the whole observed
   ~0.6 GB/s on its own. Time reading the SAME real expert set three ways: (a) fault-driven mmap
   touch (today's path), (b) single-threaded `pread`, (c) async `pread` at queue depth 4-8.
   Three numbers, an afternoon. If QD8 reaches multi-GB/s, that gap IS the campaign — and
   plausibly explains Zeno's own "disk offloading" edge, since that description matches
   owned-buffer streaming exactly. This sizes `task-moe-streaming.md`'s Lever 1 (owned-buffer
   `pread`, the only route to a firm RAM cap on darwin) before committing to building it —
   design note for when it IS built: owned buffers change the paged path's buffer source from
   mmap aliases to copied slabs, but the plumbing phase's carve-out already keeps paged tensors
   on the canonical kernel, so this is a pager-internal change only — the kernel side is
   untouched, same numerics-inert gates apply (bit-identical, gemma4 cell the regression gate).
4. **Skew-aware retention — ALREADY BUILT, MEASURED, AND FALSIFIED. Do not rebuild.**
   `docs/task-moe-streaming.md`'s Lever 2 section replayed LRU/MRU/LFU/LFU-aging over a real
   35B-A3B trace: plain LFU is strictly worse than LRU (re-fault-restarts-at-count-1
   pathology); LFU-aging only beats LRU in a narrow 3-4 GB regime nobody runs interactively; at
   every realistic budget (≥6-8 GB) all policies converge to LRU because "on a stationary
   skewed signal, recency is a sufficient statistic for frequency." This lever is closed with
   no code, per that doc's own verdict — do not re-derive it.

## Correctness

Paging changes move bytes around; they must be numerics-inert. Gates: the paged
identical-token checks — **verified non-vacuous**, i.e. the test provably exercises eviction
and re-fault (the plumbing phase's vacuous-test lesson applies verbatim); the qwen35-family
goldens; and the **gemma4 paged cell re-run — 16.98 tok/s and 81.6% hit must not regress**,
since every lever here touches the pager both configs share. `-race` on the pager suite if
any concurrency is touched.

## Acceptance

- Projection band derived from Step 0's ceiling math before building, results held to it.
- Working target: **≥3x end-to-end (≥~4 tok/s)** at the 35B config, with the ceiling math
  saying how much more is physically available on this box; stretch is wherever that math
  puts gemma4-class paging behavior.
- Each lever measured separately (hit rate, faulted bytes/token, tok/s, pressure) then
  combined — the diagnostic's table refilled as the after.
- Results appended to `docs/task-zeno-compare.md`; the Zeno Phase 1 question gets re-answered
  there once the number lands (Part B still gated on Francis regardless).

## Not in scope

- **DeltaNet (~19%)** — its own follow-on task once this lands; never perf-visited, deserves
  its own diagnostic-then-fix pass.
- Top-k or routing changes (quality surface), expert quantization changes, prefill, Zeno
  install/Part B, the `.giw` format kind, gemma4-specific tuning beyond the regression gate.
