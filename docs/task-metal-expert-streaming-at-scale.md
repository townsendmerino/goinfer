# Metal expert streaming at scale — the qwen3_5_moe 35B-A3B measurement

**Status: MEASURED 2026-08-28.** The lane's first real number. Before this, every claim about
Metal expert streaming rested on a 4-expert fixture (correctness) and on gemma4's numbers
(a different model, a different pager wiring).

## What was measured

Qwen3.5-35B-A3B decoding through the **generic** Metal expert pager on a 16 GB Mac — the pager
that `357b7db` generalized past gemma4 and that `1f1496a` gave pread staging.

## Provenance (per the benchmarks.md methodology bar)

| field | value |
|---|---|
| machine | MacBookPro18,3 · Apple M1 Pro · 16 GB · macOS 26.6.2 (Darwin 25.6.0) |
| checkpoint | `~/models/Qwen3.5-35B-A3B-Q4_K_M.int4.giw` — 22.1 GB, **internal SSD** (never `/Volumes`, never `/srv/models`) |
| shape | nE=256, topK=8, 40 paged MoE layers, vocab 248320, W4A8 int4 |
| goinfer | `1f1496a` · go1.27.0 darwin/arm64 |
| decode | greedy self-feeding (argmax → next token), seed token 7, deterministic |
| tokens | 120 timed + 1 warm-up; steady state = median of last 80 |
| slots | `GOINFER_METAL_MOE_SLOTS=32` (≈2.5 GB of slots across 40 layers) |
| thermal | no thermal or performance warning recorded (`pmset -g therm`) |
| harness | `metal/qwen35_35b_paged_test.go` |
| pre-registration | reproduced verbatim in the appendix below, written before any run |

Every run's routing trajectory was **identical** — 13508 stages / 24892 hits / 12548 evictions /
64.8% hit-rate in all four Metal runs — so the arms are matched observations, not pooled ones.

## Result 1 — it runs, at ~2.0 tok/s

**A 35B-A3B decodes through the Metal pager at 1.97–2.02 tok/s steady-state on a 16 GB box**,
holding ~1.9 GB RSS. That is the number the lane did not have.

## Result 2 — pread generalizes, and by MORE than gemma4 measured

Counterbalanced A,B,B,A (pread, byte-copy, byte-copy, pread) to separate the effect from
page-cache warming:

| run | arm | steady-state tok/s | p90 s/tok | major faults/stage | staging share |
|---|---|--:|--:|--:|--:|
| 1 (coldest) | pread | **1.967** | 0.695 | 0.0 | 62% |
| 2 | byte-copy | 0.595 | 3.401 | 98.5 | 73% |
| 3 | byte-copy | 0.641 | 2.908 | 98.5 | 71% |
| 4 (warmest) | pread | **2.022** | 0.683 | 0.0 | 62% |

**pread / byte-copy = 3.23× (range 3.07–3.40).** Pre-registered CONFIRMED threshold was 1.10×.

The order effect is 2.8% within the pread pair and 7.7% within the byte-copy pair — an order of
magnitude smaller than the effect, and pread won from both the coldest and the warmest slot.

**The mechanism is the same one gemma4 found, and it is unambiguous:** major faults collapse from
**98.5 per stage to 0.0** (1,330,594 → 147). The mmap demand-fault page-in is simply gone.

This is **larger than gemma4's measured 1.26×**, and the reason is structural: this model's experts
are ~1.5 MB against gemma4's ~3.19 MB, and there are 256 per layer rather than 128 — so a token
stages more, smaller spans, and per-stage fault overhead is a bigger share of the total. pread's
win scales with stage count, not with bytes.

## Result 3 — the sobering one: Metal paging beats CPU paging by only ~1.2–1.4×

The CPU pager (`StreamWeights`, `decoder/moepaging.go`) run in the **same session, same box, same
checkpoint, same trajectory**:

| arm | budget | steady-state tok/s |
|---|--:|--:|
| Metal paged (pread) | ~2.5 GB of slots | **1.967, 2.022** |
| CPU paged | 6 GB | 1.726, 1.516 |
| CPU paged | 2.5 GB (memory-matched) | 1.391 |

- vs CPU at 6 GB: **1.23×**
- vs CPU memory-matched at 2.5 GB: **1.43×**

**This is much less than the lane's own framing implied.** The Atlas and the queue quoted CPU-paged
at "1.3–1.4 tok/s", which would have made the Metal number look like ~1.4–1.5×. But 1.3–1.4 is the
figure `task-zeno-compare.md` had *already* flagged as noise-contaminated and superseded with 1.605
tok/s. Measured fresh today the CPU arm is 1.52–1.73 at 6 GB — and my harness reproducing 1.605's
neighbourhood is a check that the harness is sound, not a coincidence to wave at.

Two honest caveats on this comparison, neither of which applies to Result 2:
- It is **adjacent, not per-token interleaved** — the two pagers cannot cheaply share a process.
- The CPU arm's own spread is **13.9%** across two runs, against a 23% effect. Two runs is thin.

So the defensible statement is: *Metal expert streaming is modestly ahead of CPU paging, and clearly
ahead when memory-matched — but it is not the step change the lane's framing assumed.*

## What this says about the next lever

Even with pread, **staging is still 62% of a token**. The remaining cost is no longer fault
latency — that is gone — it is *stage count*: 13508 stages against a 64.8% hit rate. The lever is
therefore residency policy (more slots, better retention), not faster staging. Note that prior art
already closed the obvious version of that: `task-moe-streaming.md` measured LRU/LFU/LFU-aging on a
real 35B trace and found plain LRU wins at every realistic budget. So the honest next question is
**how many slots fit**, not **which policy** — an N-sweep, which this harness takes via
`GOINFER_PAGED_N`.

## Reproduce

```
GOINFER_HEAVY_TESTS=1 go test -count=1 -tags goinfer_testhooks ./metal/ \
  -run 'TestQwen35_35B_pagedRuns|TestQwen35_35B_cpuPagedBaseline' -v -timeout 60m
```
`-count=1` is **required**: without it Go's test cache returns `ok (cached)` in ~1s for a repeated
arm, which is a void result that looks exactly like a fast run. That happened on the first attempt
here and silently voided two of the four counterbalance arms.

Raw run logs are archived on the measuring box under a `goinfer-bench-logs` directory in the
home directory (machine-local, so deliberately described rather than cited as a path — a path
outside the repo resolves for nobody else).

## Appendix — the pre-registration, verbatim

Written before any run, so the decision rule could not be fitted to the result.

> **Question.** Does Qwen3.5-35B-A3B (nE=256, topK=8, 40 MoE layers, 22.1 GB int4 .giw) decode
> through the GENERIC Metal expert pager on a 16 GB M1 Pro, and at what steady-state rate?
>
> **Primary metric.** Steady-state decode rate = median of per-token wall times over the LAST TWO
> THIRDS of the run, after warm-up. Median (not mean) because paging produces a heavy right tail;
> the tail is reported separately as p90, never silently averaged away.
>
> **Arms.** A: Metal paged, pread staging (the path ported 2026-08-28). B: Metal paged, mmap
> byte-copy (`GOINFER_MOE_PREAD=0`) — the do-nothing arm for pread. C: CPU-paged prior art,
> 1.3–1.4 tok/s — cited, not claimed against.
>
> **Decision rules.**
> 1. pread generalizes: A/B ≥ 1.10× → CONFIRMED at scale. pread does not: A/B ≤ 1.03× → the
>    gemma4 1.26× is gemma4-specific, say so. **AMBIGUOUS BAND: 1.03× < A/B < 1.10× → PARKED**,
>    no claim either way.
> 2. Ordering is counterbalanced A,B,B,A. The macOS unified buffer cache is system-wide and the
>    file (22.1 GB) exceeds RAM (16 GB), so run 1 is cold and runs 2–4 are partially warm. Only
>    warm-vs-warm pairs are compared; the cold run is reported separately and never pooled.
> 3. NO ratio against the CPU-paged 1.3–1.4 is claimed unless a CPU arm is measured in the same
>    session. Absent that, both numbers are reported side by side with their provenance and the
>    comparison is explicitly marked as not-yet-made.
> 4. Any run whose model path does not start with `$HOME/models` is VOID.

Rule 3 is why a CPU arm was measured rather than the 1.3–1.4 figure being borrowed — and doing so
is what turned the expected ~1.4–1.5× into the actual ~1.2–1.4×.
