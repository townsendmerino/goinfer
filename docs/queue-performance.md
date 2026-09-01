# Performance queue

> **EMPTY as of 2026-08-31.** Every item that was here is closed, refuted, withdrawn, or has moved
> to the track that owns it. This file is live — new performance work is filed here — it just has
> nothing in it right now.
>
> **The closed record is [`docs/completed/queue-performance.md`](./completed/queue-performance.md)**
> (2,245 lines, G15–G24 · P1–P16 · A1–A11 and the A9-\* series). It moved rather than being deleted
> because the reasoning is the point, negative results included. Per `docs/README.md`'s archival
> rule, `completed/` is not scanned by the citation lint, so archiving it also retired its citations
> from the live gate — that is deliberate, not an oversight. Pages that linked to
> `docs/queue-performance.md` still resolve here.

## What is open

One item, and it is an OBSERVATION rather than a work item — nothing here is claimed as a
bottleneck to go fix.

- **P17 · `TestSamplingThroughputGate` has ~1% headroom on arm64 and fails under suite load** —
  **DIAGNOSED AND FIXED 2026-08-31. The bar was NOT moved.**

  **The question was: is full-vocab selection genuinely load-sensitive, or was the gate set with no
  headroom on this machine class? Neither, quite.** The gate is a RATIO, and both its level and its
  variance come from the DENOMINATOR (temp-only) — which is not the thing under test. Measured, three
  consecutive isolated runs at V=262144:

  | | temp-only (denominator) | temp+top_p (numerator) | ratio |
  |---|---|---|---|
  | x86 | 1.356 / 1.694 / 1.994 ms — **47% spread** | 6.54 / 6.72 / 6.93 ms — 6% | 4.83 / 3.97 / 3.48 |
  | arm64 | 558.7 / 564.6 / 558.6 µs — 1% | 2.682 / 2.685 / 2.680 ms — **0.2%** | 4.80 / 4.76 / 4.80 |

  The numerator — the "full-vocab selection" the gate claims to protect — is the most stable
  quantity in the measurement, on both machines. The ratio moves because the baseline moves, which
  is what the gate's own comment already warned it does ("a gate whose baseline moves is measuring
  two things at once", written when P2b made that denominator 4.7x faster and the bound had to be
  raised).

  **Fix: best-of-3 instead of a single `testing.Benchmark` mean.** A mean tracks jitter upward — it
  has a floor but no ceiling — and the quantity is floored, so the minimum is the right estimator.
  Under full-suite load, the exact condition that produced the original failure: **5.13x FAIL →
  4.82x PASS.** x86 run-to-run spread cut from 1.35x to 0.39x.

  **A prediction I made and the measurement refuted**, recorded because the correction is the
  useful part: I expected best-of-N to make the ratio machine-independent (predicted x86 4.83 vs
  arm64 4.80). It does not — x86 settles near 4.0 and arm64 near 4.83, a real 0.7x gap on the same
  code. The comment in the test said the wrong thing for one commit and now says the measured one.

  **Residual, NOT fixed:** the bound is effectively set by whichever machine runs hottest on this
  ratio (arm64, ~3% margin), and the denominator remains an optimization target — any future
  speedup to temp-only re-tightens this gate for everyone. The structural answer is to gate the
  NUMERATOR, which is stable to 0.2%; the reason that was not done here is that an absolute
  numerator bound is machine-dependent (2.7 ms arm64 vs 6.5 ms x86), which is the problem the ratio
  was chosen to avoid. Filing that trade rather than resolving it.

  Measured while landing G1, on a tree whose diff contains **no sampler file** — the benchmarked
  code is byte-identical to `main`, so this is not a regression from that work:

  | context | V=262144 temp+top_p ÷ temp-only | vs gate 5.0x |
  |---|---|---|
  | inside the full `./decoder/` suite | **5.13x** | **FAIL** |
  | isolated, `-count=1`, four runs | 4.81 / 4.95 / 4.79 / 4.97 | pass, by 0.6-4% |

  So the gate passes alone and fails under concurrent load, with a mean margin under 3%. Two
  traps worth naming before anyone picks this up:

  - **`go test` CACHES a passing result.** The first three "repeats" here returned byte-identical
    timings (566997 / 2759883 ns/op) because only the first actually ran. Any repeat-measurement
    of this gate needs `-count=1`, or it measures nothing and looks stable while doing it.
  - **Do NOT raise the bar to 5.5x to make this green.** Re-baselining because a number moved is
    how a regression gets blessed; a bar moves only with a mechanism. The open question is whether
    full-vocab selection is genuinely load-sensitive (a cache-pressure story) or whether the gate
    was set with no headroom on this machine class in the first place. Answer that first.

## Where the last items went

| item | disposition |
|---|---|
| **P16** · re-anchor to Nobara 44 / driver `595.91.07` | **DONE 2026-08-31.** Six legs were already re-anchored on 2026-08-27; the seventh (the v0.11.0 qualification) was **retired** rather than re-measured — its numbers were §B6/§B7 by code-identity, and both are current. |
| **P14** · the CPU gap is kernel arithmetic | **DONE.** Items 1+2 refuted (centering), item 3 built, wired, measured (+2.10% decode) and **parked default-OFF** against its pre-registered 4% bar. |
| **A2** · 26B documentation correction | **DONE 2026-08-31**, published as `docs/benchmarks.md` §B4.2. |
| **A10** · the ~150 MiB allocation floor | **CLOSED 2026-08-12** — fully decomposed, nothing unattributed. |
| **D3b** · the expert-cache default | **SHIPPED 2026-08-20** (`8f3c5e7`). Lives in `docs/queue-release.md`. |
| **P10** · DSpark / DFlash block drafters | **MOVED to the speculation track** — see below. Not finished; not performance-queue debt. |
| **P15** · DFlash 2 | **MOVED to the speculation track** — see below. |

## P10 and P15 moved rather than closed, and the distinction matters

They are **not done**, and nothing here should be read as saying they are. They left this queue
because they were never really performance-queue items: the rest of this queue was *find a
bottleneck, measure it, fix or refute it*, and those two are an ongoing research program with
pre-registered kill-gates.

Their substance already lived in [`docs/spec/08-dspark-dflash.md`](./spec/08-dspark-dflash.md)
(~25.5k words — the kill-gates, the increment log, the licensing correction, the Metal verdict);
the entries here were a second, thinner copy that could drift from it. The spec track owns them,
[`docs/spec/README.md`](./spec/README.md) indexes them, and
[`docs/spec/experiments.md`](./spec/experiments.md) is the dated run log.

**Open state, carried over so it is not lost in the move:**

- **P10 · increment 4.** Kill-gates 1 and 2 cleared 2026-08-15 (6.78 tok/verify against a ≥3.0
  bar). Remaining: gate 3 — end-to-end **≥1.3× vs plain resident decode on ≥1 GPU backend** — and
  gate 4's mixed-workload width router. The **Metal leg is measured not-ready**: ~1.13× ceiling
  even at `draft_ms=0`, and `PrefillLast` is not bit-identical, so the lossless contract cannot be
  met there. `gpt-oss` is blocked on a missing harmony chat template, not on the seam.
- **P15 · DFlash 2.** Filed 2026-08-20, **gates before code**, not started. Sequenced to land
  **before** P10's gate-4 width router — doing gate 4 first would mean redoing it.

## Filing a new item

Read `docs/completed/queue-performance.md` first — it is 2,245 lines of what was already tried,
including the negative results, and several of its entries exist because something was rebuilt that
had already been measured and rejected. Then follow the same discipline the archive does: state the
mechanism, pre-register the decision rule and its ambiguous band, include the do-nothing arm, and
record a negative with the same care as a win.

**One recurring defect that archive documents four times over, worth knowing before you add
anything:** an entry gets its resolution *appended* while the stale conclusion is left standing in
verdict position, so the item reads as open long after it closed. A2 (a pre-registration answered
four days earlier), D3b (shipped eleven days earlier), A10 (resolved, header still said OPEN) and
P16 (a stale-list four items out of date) all failed this way. **When you close something, correct
the sentence a scanner stops at — not only the body.**
