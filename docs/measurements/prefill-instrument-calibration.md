# Instrument calibration — `BenchmarkPrefillLong` (f32, dense, batched prefill)

**Recorded before any comparison sample exists.** This is the prefill instrument's characterization
in its own right, run so the A/B's floor and warm-up discard come from **separate data** rather than
from the samples they will judge. Deriving a threshold from the data it judges is what separating the
decode 8-sample pass prevented; this is the same separation for prefill.

## Configuration

`BenchmarkPrefillLong`, Qwen2.5-Coder-0.5B-Instruct (dense), **f32** via `GOINFER_BENCH_QUANT=`,
`GOINFER_PREFILL_LEN=512`, `-benchtime 2x`, `-count 8`, single session.
Box: AMD Ryzen 7 3700X, GOMAXPROCS 16, linux/amd64. aikit v1.17.1.

## Capability — established by EXECUTION, not configuration

Configuration recorded for completeness: `canBatchN(512)=true`, f32 weights. That is what was
*intended*. What was **observed**, from a CPU profile of the run:

| witness | flat | cumulative | what it establishes |
|---|---|---|---|
| `linalg.blockedFill` | 0.07s | **50.78s — 42.65%** | the reworked f32 blocked path **executed** |
| `linalg.blockRows3x4` | 7.29s | **46.09s — 38.71%** | it ran at **M>1** (returns immediately unless `iEnd-i >= 3`) |
| `linalg.dotFMA3x4` | 38s | 31.91% | the 3×4 register-blocked kernel inside it |
| `linalg.dot8ColsInto` | 0.11s | 4.29s — 3.60% | the new f32 eight-column kernel |

Both witnesses are far from zero, so the zero-disambiguation rule did not need to fire.

## SENSITIVITY — what this instrument can and cannot resolve

**PROFILED ARM: `aikit v1.17.1`.** The figures below come from the main checkout at HEAD, which
resolves v1.17.1. **The v1.16.0 arm is unprofiled** at the time of writing. That matters because the
fraction is a property of ONE BUILD, and the rework changes the very code whose share is being
measured — the two arms can legitimately differ.

**Pending, and it changes the bound rather than confirming it:** profile the **v1.16.0** arm and, if
the fractions differ materially, **use the SMALLER of the two** for the dilution bound. The bound
goes into the conclusion's wording, so it must be the conservative one — a larger reworked-path share
makes the instrument look more sensitive than it is. Until that is done, the bound below is
provisional and marked as such.

**The reworked path is ~43% of runtime on v1.17.1, not all of it.** The other ~47% is
`MatmulBTAcc64` / `dotF32Acc64` — a different function the rework does not touch.

So an effect inside the reworked code is **diluted ~2.3× at the benchmark level** (provisional, from
the v1.17.1 arm): a 10% change there appears as ~4.3% here. **A flat result therefore means "no effect
larger than ~2.3× the floor within the reworked path", not "no effect".** Stated before the run so it
cannot be quietly dropped from how a flat result is worded.

## The delta and its uncertainty are reported WHATEVER the branch

The floor is a **practical-significance threshold, not a detection limit**, and the two must not be
conflated. With rel-sd 0.227% and 12 retained samples per arm, the standard error of the difference
in **means** is 0.227% × √(2/12) ≈ **0.093%** — so the 0.6% floor sits at roughly **6 standard
errors**. (For the pre-registered **median** statistic the standard error is ≈1.25× that under
normality, ≈0.12%; both are reported so the statistic and its uncertainty match.)

**Consequence, fixed here:** the flat branch reads *"no effect exceeding the declared threshold"*,
**never** *"no difference"* — and the measured delta and its uncertainty are printed alongside the
branch in every case. A real 0.3% effect would be correctly below the floor and **invisible if only
the branch were recorded**. The same applies to the other branches: the number and its uncertainty go
in whichever way it lands.

## Samples

In run order (tok/s): `81.90, 82.34, 81.96, 81.89, 81.79, 82.17, 82.18, 82.12`

| n | mean | median | sd | rel-sd | min | max | range |
|---|---|---|---|---|---|---|---|
| 8 | 82.0438 | 82.0400 | 0.1866 | **0.227%** | 81.79 | 82.34 | 0.670% of mean |

## Warm-up form

**There is none at this shape.** Sample 1 — the post-load sample — sits **−0.86 sd** from the mean of
the rest: not an outlier. This differs from the decode instrument, where the first sample after a
load was plainly cold (0.8733 against a ~0.94 plateau).

## Decisions, fixed here and applied to both arms identically

**FLOOR = 0.6%. The inherited 2.0% does NOT survive contact and is replaced.**
The prefill instrument is **3.6× quieter** than decode (rel-sd 0.227% vs 0.82%). Keeping 2.0% would
be ~8.8σ — an instrument able to miss a real 1.5% effect while reporting "flat". The floor is derived
by the **same method and the same multiplier as decode's** (≈2.4σ, chosen above 2σ so one stray
sample cannot manufacture a result): 2.4 × 0.227% = 0.545%, rounded to **0.6%**.

**WARM-UP DISCARD = 1 per visit — and NOT because warm-up was found.** The characterization found
none. It is retained because 0b observed **one** process start and the A/B has several, so one
in-noise post-load sample is thin evidence for "no warm-up across visits". It costs samples, never
favours an arm (applied identically), and the sample count is raised to compensate. Recorded this way
so nobody later reads the discard as evidence of a warm-up effect that was not observed.

**Statistic: median of retained samples per arm.** Same as decode.
