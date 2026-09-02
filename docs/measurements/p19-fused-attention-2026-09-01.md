# P19 — the fused schedule CLEARS its bar, but only when parallelised over rows (2026-09-01)

> **THIS PAGE REVERSED ITSELF THE SAME DAY. An earlier version closed the item on a 0.690×
> measurement. That number was real and the verdict drawn from it was wrong**: it composed fusion
> over a COLUMN-parallel matmul, which fusion forfeits by construction. The control this page's
> own caveat called for was then run, and with the parallelism moved to QUERY ROWS — both arms,
> same worker count, same serial inner primitive — fusion wins **1.73–1.81×**. The close is
> withdrawn. The history is kept below rather than rewritten, because the mistake is the lesson.

**Measured, at fixed precision, at production shapes:**

| configuration | materialized | best fused | ratio |
|---|---|---|---|
| serial | 52.9 ms | 51.3 ms | **1.031×** — wash |
| column-parallel (composed over `MatmulBT`) | 37.8 ms | 54.0 ms | **0.700×** — loses |
| **row-parallel, 8 workers, serial inner** | **15.7–16.2 ms** | **9.0–9.1 ms** | **1.73–1.81× — CLEARS** |

Pre-registered bar: ≥1.30× clears, <1.10× closes. Row-parallel clears it on both runs.

Correctness holds throughout — cosine **1.000000000**, max|diff| ~1e-8 at every block size and in
every configuration — so the schedule computes the right thing and the question was only ever
whether it is faster.

## Provenance

| | |
|---|---|
| box | `nobara-pc`, Ryzen 7 3700X, Nobara 44 (CPU measurement; no GPU involved) |
| goinfer | `89e4d51` + this harness |
| harness | `decoder/p19_fused_attn_test.go` (`GOINFER_P19=1`) |
| shapes | kt=256, hd=128, nKeys=8192 — production's `attnRowTile(K=8192, nKeys=8192)` |
| precision | **f32 both arms.** Fixed by the item's own rule: "a win that only appears with f32 enabled is A3's win, not this item's" |
| load | 0.13 / 0.04 / 0.01 at start |

Gathers sit outside the timed region: both arms pay identical ones, so including them would only
dilute the effect. Neither arm masks; identical in both, so the ratio is unaffected.

## Pre-registered bar (written before the kernel existed)

	>= 1.30x  -> clears its own bar, prototype in production
	<  1.10x  -> close the item
	1.10-1.30 -> ambiguous, parks

Correctness stated in advance as a tolerance, not bit-identity, because the running-max rescale
re-associates by construction: **cosine ≥ 0.9999**.

## First pass — the arms that produced the withdrawn verdict

| | materialized | best fused | ratio |
|---|---|---|---|
| **serial** (stable) | 52.9 ms | 51.3 ms (kb=512) | **1.031×** |
| **column-parallel** (composed over `MatmulBT`) | 37.4 ms | 54.1 ms (kb=256) | **0.690×** |

Per block size, parallel arm:

| kb | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| ratio | 0.633× | 0.690× | 0.515× | 0.519× |
| cosine | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 |

### Why it lost there, which a control established rather than guessed

**The serial control was run because a kernel A/B in this repo has already published a ratio that
was mostly core count** — G24's first pass read 17.6× against a documented ~3.7× because it raced
parallel f32 against serial acc64. The same trap was available here and in the opposite direction:
`MatmulBT` fans out over its **N output columns**, and the two arms present very different N. The
materialized QKᵀ has N=nKeys=**8192**; the fused one has N=kb, **128–1024**.

Serially the schedule is a **wash** — 1.031×. So on a CPU it is not removing meaningful traffic:
the `kt × nKeys` score block at 8 MiB is streamed well enough by the blocked matmul that avoiding
it buys nothing measurable.

In parallel the materialized arm gains **1.41×** (52.9 → 37.4 ms) while the fused arm gains
**nothing** (51.3 → 54.1 ms, i.e. slightly worse). Fusion forfeits the column parallelism by
construction, because it exists precisely to make the score block small.

**Instability, recorded rather than smoothed:** the parallel materialized arm read 45.3 ms then
37.4 ms across two runs — 21% apart — moving the headline ratio 0.852× → 0.690×. The serial arm was
stable (52.9 vs 53.2). So the serial 1.031× is the citable reading and the parallel figure is
directional. Both are below the close bar, which is why the verdict does not turn on which one is
quoted.

## The row-parallel control — why the first verdict was wrong

The first pass attributed fusion's loss to `MatmulBT` fanning out over N OUTPUT COLUMNS:
materialized presents N=8192, fused presents N=kb (128–1024). That was correct as far as it went,
and this page said so — then drew a verdict from it anyway. **A schedule that is bad at exploiting
one parallelism axis has not been shown to be bad.**

So both arms were re-run parallelised over QUERY ROWS, at the same worker count, each worker using
`MatmulBT` as a **serial** inner primitive through its own pinned Workspace. Kernel quality
identical, parallelism model identical, schedule the only variable:

| | serial | 8 workers, row-parallel | scaling |
|---|---|---|---|
| materialized | 52.9 ms | 15.7–16.2 ms | **3.7×** |
| fused (kb=256–512) | 51.3 ms | 9.0–9.1 ms | **7.3×** |

**Serially the schedule is neutral, so fusion is not saving arithmetic.** The win is entirely in
how the two scale, and the reason is the working set: materialized hands each worker an
`[n, nKeys]` score array — 1.4 MB at these shapes — while fused hands it `[n, kb]`, ~44 KB. Eight
workers streaming 1.4 MB arrays become bandwidth-bound; eight workers on 44 KB blocks stay in
cache. **That is the FlashAttention argument, and it is invisible without the parallel arm.**

**A flaw in the favourable result, caught before it was quoted.** The first row-parallel run read
2.056×, but it allocated the materialized arm's 1.4 MB score buffer INSIDE the timed region while
the fused arm allocated far less — charging the losing arm for allocation. With both arms' scratch
hoisted out, the honest figure is **1.73–1.81×**. Recorded because a result that moves 2.06 → 1.75
when you fix your own instrument is exactly the kind that should be reported with its correction
attached.

**Sequencing against A3, which the item required.** The column-parallel arm is the *pre-*A3 f32
shape; row-parallel is analogous to the head fan-out A3 shipped on 2026-09-01 (measured 3.27× at
the kernel). So the 1.75× here is a win **on top of** A3's, not a re-measurement of it — which is
what the item asked for when it said the two "must not be measured as one arm or neither is
attributable".

## What is closed, and what is NOT

**~~CLOSED: the portable / pure-Go / CPU form of the lever~~ — WITHDRAWN.** The original wording
was: *"The portable claim is the thing measured here, and it does not hold on CPU."* It does hold,
on the axis the first pass did not test. The item's framing — *"FA-2/FA-3 lean on Hopper
warp-specialization and TMA; none of that ports to pure Go. The portable claim is the I/O schedule,
not the kernels"* — is **supported**: the I/O schedule ports, and pays 1.73–1.81× in pure Go once
the parallelism is on the right axis.

**Still NOT tested: the CUDA form.** The 55%-of-prefill share that made this item interesting was
measured on CUDA (`cuda-prefill-attention-share-2026-09-01.md`), where the score matrix goes to HBM
and the arithmetic differs again. The CPU result now makes a CUDA fused kernel *more* interesting
rather than less, but it is not evidence about it.

**~~Also not tested: an implementation that does not sit on top of `MatmulBT`~~ — TESTED, and it
overturned the verdict.** That caveat named the exact confound and was then resolved by the
row-parallel control above. Worth keeping visible: the caveat was written honestly, and if it had
been left as a caveat rather than run, this item would have been closed wrongly on a real number.

**Still not tested: a hand-written SIMD fused inner kernel.** The row-parallel arms still use
`MatmulBT` as their inner primitive, deliberately — writing a scalar Go inner loop would have
measured my kernel quality against aikit's tuned one and told us nothing about scheduling. A fused
kernel that also owns its own SIMD could do better or worse than 1.75×.

## The correction this run also produced

The CPU attention share that made this item look near-dead earlier in the day (17.4%) was a **MoE**
figure. That was corrected before this ran — dense models are 55% (CUDA, K=3900) and ~70% (Mac CPU
acc64, K=8192) — so this item was measured on its merits rather than dismissed on a
model-class error. The negative result stands on this measurement, not on that mistake.
