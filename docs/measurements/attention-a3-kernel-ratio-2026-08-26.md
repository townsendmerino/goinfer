# A3 kernel ratio — is an f32 attention path worth its divergence flag?

**Answer: yes.** The acc64 attention kernels are **8.14× slower than f32** at long-context prefill
shapes, which is ~2.2× worse than the "~3.7×" the code comments assume. At the 70% attention share
measured in the K=8192 profile, that is **~2.59× end-to-end prefill**. Queue entry **G24**.

**Box:** M1 Pro, dense qwen2.5-coder-1.5b shapes, goinfer `53e1c6d`. Instrument:
`decoder/g23_attnkernel_test.go` (`GOINFER_G24=1`), best-of-5.

## The number

Shapes are what G20's tiling actually calls at an 8k prompt: `kt=256, hd=128, nKeys=8192`.

| | acc64 (direct strided read) | f32 + the gathers acc64 skips | ratio |
|---|---|---|---|
| QK | 57.2 ms | 10.7 ms | **5.32×** |
| AV | 142.6 ms | 13.8 ms | **10.33×** |
| combined | 199.8 ms | 24.5 ms | **8.14×** |

Amdahl on the profile's 70% attention share ⇒ **2.59×** end-to-end long-context prefill.

## What makes this number trustworthy — three things that were nearly wrong

**1. The arms compute the same thing, and it is asserted, not assumed.** QK agreement:
**cosine 1.000000000, maxAbs 3.34e-06**. The test *fails* below 0.999999 rather than reporting a
ratio between two different computations.

**2. Parallelism is equalized — the first run was not, and it lied by 2×.** `MatmulBT` fans out via
`parallelCols`; the acc64 kernels are plain serial loops. The first pass compared serial acc64
against parallel f32 and reported **17.6×**. That was disbelieved *because* it exceeded the
documented 3.7× so implausibly, and the cause was found by reading both kernels. With `MatmulBT`
forced serial the honest ratio is 8.14×.

**3. The f32 arm pays the work acc64 avoids.** `MatmulQK/AVAcc64` read K/V directly by stride,
"skipping a kh gather entirely" and "skipping a vt gather+transpose". Both gathers are INSIDE the
timed f32 arm. Omitting them would have flattered f32 by work the real path cannot avoid.

## The residual mystery, stated rather than used

The kernel comments say acc64 is "~3.7× slower than f32". Measured here: **8.14×**. Not a
contradiction to wave away — the likeliest explanation is that 3.7× was taken at DECODE shapes,
and acc64's strided direct read degrades at `nKeys=8192` in a way A1's move (c) fixed for decode
and nobody re-measured at prefill depth. **The 2.59× estimate does not depend on resolving this**
(it uses the measured ratio, not the documented one), but anyone quoting 8.14× as a general acc64
penalty should measure their own shapes first.

## Separately measured, deliberately NOT multiplied in

f32 additionally parallelizes *within* a matmul: **4.01×** on QK alone. This is **not additive**
with the ratio above, because the real prefill path already fans acc64 out across heads (G16/G20's
worker pool). Reported so the two effects are not silently conflated into a headline.

## What this does and does not license

It clears the bar fixed **before** the measurement was taken (G24: "near 3.7× the flag's surface is
earned, near 1.3× it is not"). It does **not** license shipping f32 attention as a default: A1's
guarantees — spec-decode verify == sequential greedy, and decode == prefill — hold only for acc64,
which is why A3 is an off-by-default, documented divergence in the `--metal-fast-prefill` mould.

**Build constraint carried forward:** the f32 branch is single-threaded by construction (its
per-kv-group `kh`/`vt` gather is shared mutable state). A3 must gather once per kv-group and then
fan the group's query heads across workers reading it, or a kernel 8× faster still loses to
parallel acc64.

> **⚠ WRONG — see the retraction below.** The f32 branch was measured at **1.67× utilization**, not
> 1.0×: `MatmulBT` already fans out over output columns. And the fix does NOT need to "gather once
> per kv-group and share it" — every worker slot already owns a full-size `kh`/`vt` pair, budgeted
> by `prefillAttnWorkers` all along, so each worker gathers privately and shares nothing. Built and
> measured 2026-09-01: 1.92× end-to-end at K=4096.

---

## Reproduced 2026-09-01, and what is left after it

Re-run on the same machine (M1 Pro), same shapes (`kt=256, hd=128, nKeys=8192`), same controls —
both arms serial, the f32 arm paying the `kh` and `vt` gathers acc64 skips:

| | acc64 | f32 + gathers | ratio |
|---|---|---|---|
| QK | 55.3 ms | 10.3 ms | 5.37× |
| AV | 140.5 ms | 13.1 ms | 10.73× |
| **combined** | **195.8 ms** | **23.4 ms** | **8.37×** |

**8.37× against 8.14× recorded** — the finding is stable across a week and across the changes since.
The QK arms still agree at cosine 1.000000000, so the ratio is still between two computations that
produce the same answer.

### The decision rule was not close

Pre-registered: *near 3.7× the surface is earned, near 1.3× it is not.* 8.37× is not near the
boundary — it is 2.3× past the upper anchor. Nothing about this measurement is a judgement call.

### The ceiling, derived from the profile share rather than raced

Attention is ~70% of long-context prefill (K=8192, dense 1.5B: `MatmulAVAcc64` 51.1% +
`MatmulQKAcc64` 18.7%). Amdahl on that share:

| f32 configuration | attention speedup | end-to-end ceiling |
|---|---|---|
| serial (what ships) | 8.37× | **2.61×** |
| + intra-matmul fan-out | ~16.7× | **2.93×** |

**Measured end-to-end was 2.28×**, against a 2.61× ceiling for the serial path — 87% of it. That
agreement is the useful part: it says the profile share is right and there is no large unexplained
term hiding between the kernel and the wall clock.

### The remaining lever, now quantified

The entry's closing condition said "if it is built, the f32 path must also be made to fan out —
otherwise a faster kernel loses to parallel acc64 anyway". That is measured here:

    QK f32 + gather, serial MatmulBT     10.3 ms
    QK f32 + gather, PARALLEL MatmulBT    2.7 ms    → 3.81× forfeited today

So the f32 path is leaving ~3.8× on the QK side because its per-kv-group gather is shared mutable
state and cannot be split as written. Feeding that back through the profile share moves the ceiling
2.61× → 2.93×, i.e. **~13% more end-to-end**, not another 3.8×. Worth knowing before anyone funds
the gather rework expecting the kernel number.

> ## ⚠ RETRACTED 2026-09-01 — the three paragraphs above are wrong, and the rework measured 1.92×
>
> **`~13% more end-to-end` is withdrawn. `3.81× forfeited today` is withdrawn. The
> `2.61× → 2.93×` ceiling pair is withdrawn.** The fan-out was built and measured:
> **1.40× at K=1024, 1.58× at K=2048, 1.92× at K=4096**, end-to-end, bit-identical
> — `docs/measurements/a3-f32-attention-fanout-2026-09-01.md`.
>
> **The defect is one line above: `QK f32 + gather, PARALLEL MatmulBT  2.7 ms`.** That row was
> read as a hypothetical the shipping path forfeits. It is not hypothetical — it is what already
> ships. `MatmulBT` fans out internally over its N output columns (`parallelCols`) above
> `parThreshold = 1<<24` MACs; the tile shape here is ~268M MACs and nothing in goinfer moves that
> threshold. The 10.3 ms arm is the ARTIFICIAL one, produced by this page's own
> `SetParallelThreshold(1<<62)` control. So the "3.81× forfeited" was a comparison between the real
> path and a control, labelled as a comparison between today and tomorrow.
>
> Measured directly rather than inferred (utilization = CPU time ÷ wall time, which no Amdahl
> argument can talk past): f32 as shipped **1.67×**, acc64 **4.75×**, f32 with linalg forced serial
> **1.00×**. The "single-threaded by construction" line at the top of this section is therefore also
> wrong, and was wrong when written — it is true of the HEAD LOOP, not of the work.
>
> **The `2.28× measured vs 2.61× ceiling — 87% of it` agreement above is a coincidence, and that is
> the part worth carrying.** The ceiling was computed from a SERIAL-vs-SERIAL kernel ratio; the
> 2.28× was measured on the PARALLEL production path. Two different parallelism states either side
> of one division sign. The agreement then read as confirmation that "there is no large unexplained
> term hiding between the kernel and the wall clock" — when in fact ~58% of the f32 arm was serial
> code (gather, per-row softmax, scatter) that no amount of column-level fan-out could reach. A
> close agreement between a model and a measurement is not evidence the model is right when the two
> are not measuring the same configuration.
>
> This is the same shape as `bench_compare.sh` output divided by a peer number, and the same shape
> as the 4-layer slice that read 3.11× and measured 1.52× at full depth. Third instance; the
> pattern is *a ratio measured under one set of controls, projected onto a system running under
> another*.

### Status note

A3's question is ANSWERED and was answered on 2026-08-26; this is a reproduction, not a first
measurement. The G24 entry in `docs/completed/queue-performance.md` carries its answer at the top
and its pre-registration below, so reading it downward reads as "waiting on a measurement" — which
is how it was re-opened. The answer is the first line of that entry.

Also overtaken: the rule weighed whether a *permanently-supported, non-bit-identical flag* was
earned. `82dda2a` (2026-08-31) made f32 prefill the DEFAULT above a 512-token floor, so that cost is
now paid by default and the question this measurement settles is how much it buys, not whether to
pay it.
