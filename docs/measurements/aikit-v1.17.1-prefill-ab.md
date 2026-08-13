# A/B pre-registration — aikit v1.16.0 vs v1.17.1, PREFILL

**Written before any prefill sample exists.** No prefill measurement has been taken on this box for
either version.

## Why this one is the critical path

It gates two things at once, and the second is the expensive one:

1. **A goinfer tag cannot honestly be cut without it.** `linalg/matmul_blocked.go` is unchanged
   between v1.17.0 and v1.17.1, so v1.17.0's f32 blocked-matmul rework — `dot8ColsInto` replacing
   `Dot8x4`'s 32-partial round trip, plus `blockRows3x4` ahead of the single-row path on amd64 — is
   **live in both versions and measured in neither.** Decode is measured and flat; prefill is the
   phase that rework actually lives on. A release characterizing one phase while silently carrying
   an unmeasured change to another is a claim by omission.
2. **A tag numbered v0.13.0 or above fires C3**, the Metal consumer window — the largest completely
   uncovered surface in the project, which has already **sunk once**.

**CONDITIONAL, and I stated it too strongly first.** C3's trigger is *"the next goinfer RELEASE TAG
(≥ v0.13.0) that carries an AIKIT BUMP"* — the version floor is part of it. A **v0.12.1** patch
carrying this bump would **not** fire C3, and would not owe E1's parity backfill either (E1 reserves
that for v0.13.0). So what P10 gates depends on a numbering decision nobody has taken yet:

| next tag | P10 owed? | C3 fires? | E1 backfill owed? |
|---|---|---|---|
| `v0.12.1` (patch) | **yes** — the f32 rework ships either way | no | no |
| `v0.13.0` | **yes** | **yes** | **yes** |

P10 is owed under both, which is why it is the critical path regardless. The rest is a decision.

## STEP 0 — the instrument must first be shown CAPABLE, then CHARACTERIZED

Two prerequisites, both before the first comparison sample, and the first was nearly missed.

**0a. Capability — proven by EXECUTION, not by configuration.** Configuring f32 weights and a
`canBatchN` architecture establishes what was *intended*; it does not establish that `blockedFill`
ran or that it ran at M>1. Those diverge exactly the way `loadBenchModel()`'s hardcoded
`int8int8` made them diverge — a correct-looking invocation measuring the wrong kernel.

**The assertion is on observed execution:**

- **`blockedFill` executed** — non-zero samples in a CPU profile of the run (`-cpuprofile`, then
  `go tool pprof -top`). Taken from inside the function, and it needs no change to aikit or to
  goinfer's production code.
- **M was greater than 1** — non-zero samples in **`blockRows3x4`**, which returns immediately
  unless `iEnd-i >= 3` (`linalg/rowblock_amd64.go`). Its presence in the profile is direct evidence
  that the multi-row batched path ran, as opposed to N calls at M=1. The v1.16.0 arm has no
  `blockRows3x4`; its equivalent witness is `blockedFill` plus `Dot8x4`.
- **Both checked non-zero BEFORE the characterization begins.**

**A ZERO MUST BE DISAMBIGUATED BEFORE IT IS READ AS A RESULT.** A zero sample count means *either*
"did not execute" *or* "executed below the sampler's resolution", and **a profile cannot tell you
which**. That is the absence-of-signal class applied to profiling: absence of samples is not absence
of execution.

The two witnesses are not equally informative, and they get different treatment:

- **`blockedFill` on a long f32 prefill should dominate the profile.** A zero there is genuinely
  informative — at that weight, "below the sampler's resolution" is not a plausible explanation.
- **`blockRows3x4` is much narrower** (a 3-row inner sweep). **Its zero is NOT evidence of M=1 on its
  own** — it is equally consistent with a sampling miss.

So on any zero, before concluding anything: **raise `runtime.SetCPUProfileRate` (or `-cpuprofilerate`)
and re-check, or add an explicit counter and read that instead.** Only a zero that survives a
higher-resolution re-check, or a counter reading zero, supports "did not execute". Only then is "no
instrument exists for this yet" the honest outcome — and it is a conclusion reached, not a default.

`b.Logf("canBatchN=%v")` and the quant setting are recorded as *configuration*, alongside the
execution evidence, never in place of it.

**Why the obvious instrument cannot see the change.** `BenchmarkPrefillLong` calls
`loadBenchModel()`, which hardcodes `Options{Quant: "int8int8"}`. Traced through goinfer:

    int8 weights -> matmulInto -> linalg.MatmulBTW8A8Into        (never reaches blockedFill)
    f32  weights -> matmulInto -> matmul -> linalg.MatmulBT -> blockedFill   <- the reworked code

So an int8int8 prefill run exercises the f32 blocked matmul **not at all**, and would have produced
a confident flat result measuring nothing. That is exercised-but-never-triggered, inside the
pre-registration written to prevent it. The instrument therefore needs:

- **f32 weights** (not int8int8, not int4) — otherwise `blockedFill` is never entered;
- **a `canBatchN` architecture** — `prefillLogits` routes MoE and Gemma4 down the sequential
  per-token path, so DeepSeek-V2-Lite (the decode A/B's model) would run at M=1 and miss the batched
  shape the rework targets. A dense model (e.g. Qwen2.5-Coder-1.5B) is required, and
  `b.Logf("canBatchN(%d)=%v")` must be **asserted true in the run log**, not assumed.

**Capability is proven before calibration**: one run of each arm, confirming `canBatchN=true` and a
non-trivial wall time at f32. If the instrument cannot be made to enter `blockedFill`, the honest
outcome is "no instrument exists for this yet", not a flat result.

**0b. Characterization, as its own artifact.** A calibration pass on the prefill instrument —
**same shape as the 8-sample decode one**: repeated samples in one session on one arm, recording the
warm-up form (is the first sample cold, and by how much), the discard count that follows from it, and
the spread (sd, relative sd, min/max).

**This is recorded as instrument calibration in its own right, before any comparison sample exists.**
The 2.0% floor is inherited from the *decode* instrument and is an assumption here. Either it
survives contact with the prefill characterization, or a measured floor replaces it — and **either
way the number predates the first comparison sample.**

**Why this cannot be folded into the A/B run.** Deriving the floor from the comparison data is
deriving the threshold from the data it judges. Separating the 8-sample pass is exactly what kept the
decode floor honest, and "re-derive the floor if the spread exceeds it" quietly gives that back if
the re-derivation reads the A/B's own samples. Characterize first, fix the floor, then compare.

## Harness — the same discipline as the decode A/B, for the same reason

Interleaved **a/b/a/b in one session on one box**. Two worktrees at the same goinfer commit with only
`go.mod` differing, so the goinfer source is identical between arms and the aikit version is the only
variable. `GOWORK=off`, no `replace`, aikit from the module cache.

**Comparison: v1.16.0 against v1.17.1**, because `matmul_blocked.go` is identical between v1.17.0 and
v1.17.1 — the rework is live in both, so v1.16.0 is the only baseline that predates it.

**Warm-up discard: 1 per visit, and NOT because warm-up was found** — 0b found none (sample 1 at
−0.86 sd, not an outlier). It is retained because 0b observed one process start and the A/B has
several. Applied identically to both arms, so it cannot favour either.

**Floor: 0.6%** — 0b ran, the inherited 2.0% did **not** survive it, and a measured floor replaced
it before the first comparison sample. The prefill instrument is 3.6× quieter than decode (rel-sd
0.227% vs 0.82%); 2.0% would have been ~8.8σ, able to miss a real 1.5% effect while reporting flat.
Derived by decode's own method and multiplier (2.4σ). Full record:
`docs/measurements/prefill-instrument-calibration.md`.

**Statistic: median of the retained samples per arm.**

**The 5% session-drift figure applies here too** (`decoder/decode_bench_test.go`): a sequential
before/after would be dominated by it. Interleaving is not optional.

## Branches, fixed now — including the flat one

**In every branch, the measured delta AND its standard error are reported alongside the verdict.**
The floor is a practical-significance threshold (~6 standard errors), **not** a detection limit, so
"flat" means *no effect exceeding the declared threshold* — never *no difference*. A real 0.3% effect
would be correctly below the floor and invisible if only the branch were recorded.

1. **Within ±0.6%** → **flat.** The f32 blocked-matmul rework is **below this instrument's noise
   floor on prefill**. That is the recorded answer and it discharges the obligation — it is a
   result, not an absence of one, and it does *not* become "prefill is unmeasured" in the release
   notes.
2. **v1.17.1 slower by ≥0.6%** → a prefill regression carried by the rework, still live upstream.
   Reported upstream with the same discipline as the decode one: direction, magnitude, method, and
   an explicit statement of what was not isolated.
3. **v1.17.1 faster by ≥0.6%** → the rework does what it was aimed at. **Scoped exactly as a loss
   would be** — one model, one prompt length, one box, prefill only — and it does **not** enter
   CHANGELOG, docs or release notes on this evidence alone.

## Known weakness, stated in advance

**0b has run and the floor is fixed at 0.6% from its data.** The sequencing held: the number
predates the first comparison sample.

The remaining weakness is **sensitivity, not noise**. The reworked path is ~43% of this benchmark's
runtime, so an effect inside it is diluted ~2.3× here: a flat result means *no effect larger than
about 1.4% within the reworked code*, not *no effect*. That bound is stated in every wording of the
result.

---

# Result — appended after the run. BRANCH 3: FASTER

Raw, in run order. **Bold = the pre-registered discard** (first sample of each visit).

| visit | samples (tok/s) |
|---|---|
| 1 · `v1.16.0` | **78.39**, 78.63, 78.54, 78.38, 78.19 |
| 1 · `v1.17.1` | **82.07**, 82.24, 82.07, 82.35, 81.97 |
| 2 · `v1.16.0` | **78.29**, 78.02, 78.55, 79.51, 79.35 |
| 2 · `v1.17.1` | **82.16**, 82.05, 81.93, 82.04, 81.99 |
| 3 · `v1.16.0` | **78.01**, 78.11, 78.03, 78.57, 78.73 |
| 3 · `v1.17.1` | **81.93**, 83.06, 82.88, 82.80, 82.41 |

| arm | n | median | mean | sd | min | max |
|---|---|---|---|---|---|---|
| `v1.16.0` | 12 | 78.5450 | 78.5508 | 0.4771 | 78.02 | 79.51 |
| `v1.17.1` | 12 | 82.1550 | 82.3158 | 0.3936 | 81.93 | 83.06 |

## The delta and its uncertainty — printed regardless of branch, as pre-registered

| quantity | value |
|---|---|
| **median delta** | **+3.6100 tok/s = +4.488%** |
| mean delta | +3.7650 tok/s = +4.681% |
| analytic SE (mean diff) | 0.1786 tok/s = 0.222% → the mean delta is **21.1 SE** |
| **bootstrap 95% CI (median diff)** | **[+4.240%, +5.259%]** (percentile, B=20000, seed 20260812) |

**Bootstrap vs analytic — they agree, and that is worth stating.** Bootstrap half-width 0.510%
against an analytic 1.96·SE half-width of 0.435%: the bootstrap is slightly wider, as expected for
a median, with no sign of the heavy tails or skew that would have made the normality-derived factor
misleading. The median was bootstrapped rather than scaled by 1.25 **because it was chosen for
robustness**, and estimating its uncertainty by assuming normality would have assumed away the
property it was chosen for. Here the two happen to agree; the point is that this was checked.

**Verdict: BRANCH 3 — v1.17.1 is FASTER on prefill by ~4.5%**, far outside the 0.6% floor
(~20 floor-widths, 21 analytic SE). The arms **do not overlap at all**: `v1.16.0` spans
[78.02, 79.51], `v1.17.1` spans [81.93, 83.06]. Per-visit medians are ordered consistently across
all three rounds (78.46 / 78.95 / 78.34 against 82.16 / 82.02 / 82.84).

## Branch 3 gets the SAME scoping a loss would — as pre-registered

**One model** (Qwen2.5-Coder-0.5B, dense), **one prompt length** (512), **one quantization** (f32),
**one box**, **prefill only**. It does **not** enter the CHANGELOG, the docs or the release notes on
this evidence, and no user-facing claim is made from it. It is a measurement, referenced from P10.

## Converting to a within-path figure — and the conservative direction FLIPS on this branch

Both arms are now profiled, and the reworked path's share **differs materially between them**:

| arm | `blockedFill` cumulative |
|---|---|
| `v1.16.0` | **52.18%** |
| `v1.17.1` | **42.65%** |

The share *dropping* is itself coherent with the result: the rework makes that path cheaper, so it
takes a smaller slice of a smaller total.

**The pre-registered rule said "use the smaller fraction", and that rule was written for the FLAT
branch, where a smaller share means a larger admissible hidden effect — genuinely conservative.
This result landed on branch 3, where the direction reverses:** dividing by the smaller share yields
a *larger* claimed within-path speedup, which is the *less* conservative reading of a win.

So both are given, with the conservative end named:

- ÷ 52.18% (v1.16.0 share) → **~8.6% within the reworked path** ← **conservative for this claim**
- ÷ 42.65% (v1.17.1 share) → ~10.5% within the reworked path

**The figure to quote is ~8.6%, and the honest range is 8.6–10.5%.** Applying the letter of a rule
written for a different branch would have inflated the claim.

**Divisor provenance:** measured **under a CPU profiler**, applied to **unprofiled** A/B runs.
Profiling perturbs what it measures, so this is a rough divisor — adequate for converting a
benchmark-level figure into an order-of-magnitude statement about the reworked code, and not more
than that.

## What this does and does not settle

It **discharges P10's obligation**: prefill is measured, on both versions, and the f32 blocked-matmul
rework is a **prefill win** on this shape rather than an unmeasured risk.

It says nothing about **other prompt lengths, other models, MoE or Gemma4 architectures** (which
route down the sequential per-token path and never reach the batched shape), **non-f32
quantizations**, or **any other box**.

**Read alongside P9:** the same aikit change set cost ~3% of *decode* at v1.17.0 (from the int8
kernel, since reverted) and gains ~4.5% of *prefill* at v1.17.1 (from the f32 rework, still live).
Those are different phases, different kernels, and different releases — the pairing is a summary of
two measurements, not a trade-off anyone chose.
