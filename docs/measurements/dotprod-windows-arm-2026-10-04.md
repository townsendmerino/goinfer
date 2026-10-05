# DotProd versus the base kernels on Azure Cobalt 100 (the Windows ARM question): pre-registration, 2026-10-04

**Status: run 1 done and NOT the registered measurement (instrument defects, below); run 2 pending.** The rule and bands below were committed before any run and are not edited after one. Corrections and run results are appended as dated sections at the end, with this text unchanged above them, except that the "Arms" bullet carries an inline correction.

## The question

aikit v1.56.0 probes DotProd on Windows on ARM, so cores that have it (Snapdragon X, Azure Cobalt 100) run the SDOT kernels and the row4 int4 layout instead of the base SMULL/SADALP kernels and the canonical layout.
How much faster, if at all, is that, on the projections a decode and a prefill actually run? The hardware-coverage task left this as "unmeasured" (`docs/tasks/task-hardware-coverage-2026-10.md`).

## What can and cannot be measured

- **Machine.** GitHub's free arm64 runners, `ubuntu-24.04-arm` and `windows-11-arm`: **both Azure Cobalt 100 (Neoverse N2)**, 4 vCPU, shared VMs. Kernel speed does not depend on the OS, so the Linux run answers the kernel question and the Windows run
  confirms it on Windows (scheduler, heap-read model). **Snapdragon X (Oryx cores) is NOT measured and a Cobalt result does not transfer to it.** There is no way to measure it without owning one.
- **Instrument.** `TestDotProdVsBase_AB` (aikit `linalg/dotprod_ab_arm64_test.go`, workflow `dotprod-ab`, on demand): the arms run back to back in ONE process (hasDotProd is a package flag read at call time), round after round, alternating
  which arm goes first. Not a stored baseline, and not two processes minutes apart.
- **Arms.** `base`: canonical weights, hasDotProd=false (**corrected after run 1: for int8 that is the NEON SMULL/SADALP kernel, but for int4 W4A8 it is the pure-Go scalar reference `dotW4A8Scalar`, because aikit has no non-DotProd int4 kernel; this text originally said SMULL/SADALP for both**). `dot`: the PRODUCTION DotProd path: int4 as the repacked-only row4 layout (what goinfer's loaders build on arm64), int8 as is, hasDotProd=true. `dot-canon` (int4 only): canonical layout with SDOT, for attribution.
  The headline is base/dot. `dot-canon` is reported, not graded.
- **Cells.** Qwen2.5 0.5B (896/128/4864), 1.5B (1536/256/8960) and 7B (3584/512/18944); projections q, k, o, gate, down; int4 (W4A8) and int8 (W8A8); M=1 (decode) at both models and M=64 (prefill) for 0.5B and 1.5B only (a 7B M=64 call is seconds);
  workers=1 (the kernel alone) and workers=default (production fan-out). 100 cells. Random weights from a fixed seed: the speed of these kernels does not depend on the values.
- **Per cell.** 21 rounds, each arm timed over a block of at least 20 ms after a warm call following the switch; the speedup of a round is t_base / t_arm. Before timing, base and dot must produce the same result to 1e-4 relative (checked red by planting a different weight in the dot arm: 1.58 relative, the test fails).
- **What this is not.** A kernel-level result gives DIRECTION, not size: this repo measured a kernel microbenchmark's served effect at 0.05-1.72x of it (`docs/measurements/test-efficiency-2026-09/te5b-concordance-2026-09-28.md`). **No whole-token or served claim comes from it.** A shared VM has no idle gate, so by `docs/benchmarks.md`'s
  provenance rules **no figure from this enters a benchmarks table**: it is recorded here as exploratory, shared runner.
- **The QEMU smoke run** used to check the harness ran the same code with emulated DotProd; its ratios are instruction-count artefacts and are **not a result and not quoted anywhere**.

## The decision rule (the constants are in aikit `dotprod_ab_stats_test.go`, unit-tested at their boundaries)

Per cell, on the 21 per-round speedups of `dot` over `base`:

| verdict | needs ALL of |
|---|---|
| **FASTER** | median >= 1.05; at least 80% of rounds above 1; the median of the rounds where `base` ran first AND the median of the rounds where `dot` ran first both >= 1.03 |
| **SLOWER** | median <= 0.95; at most 20% of rounds above 1; both order-half medians <= 0.97 |
| **AMBIGUOUS (noisy)** | (Q3-Q1)/median > 0.15, checked first: too noisy to call either way |
| **AMBIGUOUS** | everything else: the 0.95-1.05 band, and any cell whose win depends on going second (a cache effect, not a kernel) |

**Overall claim "DotProd is faster on Cobalt 100's kernels"** holds only if at least **75% of the M=1 (decode) cells are FASTER and NO cell, at any M, is SLOWER.** A SLOWER cell is reported by name and shape either way: it is the do-nothing arm's finding that the base kernel wins there.
**Parked band:** a result that fails the claim without any SLOWER cell is "ambiguous, parked": it is reported as what it is, with the cell table, and no sentence says DotProd is faster. **Not done:** re-running until a cell passes, dropping a noisy cell, moving a band after seeing data, or quoting the best cell.

## How it is run, and what gets recorded

`gh workflow run dotprod-ab -R townsendmerino/aikit` (21 rounds), both runners, once. If a runner fails for infrastructure reasons it is re-run as the same run; a second measurement on another day is a separate, labelled record, not a replacement.
The Result section will carry, per runner: aikit commit, runner image version, the CPU line, the 1-minute load average at start, run IDs, the SUMMARY line verbatim, every cell's line (or the artifact's path), and the verdict under the rule above.

## Deviation 1: run 1 was not the registered measurement (written after run 1, before run 2)

**Run 1** (aikit workflow run 37265393092, aikit `d4f5eff`, both runners, 2026-10-04 21:53 PDT):

| runner | SUMMARY line, verbatim |
|---|---|
| ubuntu-24.04-arm (Cobalt 100, linux/arm64) | `60 cells, 59 FASTER, 0 SLOWER, 1 other; overall claim "DotProd is faster on this core" = true` |
| windows-11-arm (Cobalt 100, windows/arm64) | `60 cells, 13 FASTER, 0 SLOWER, 47 other; overall claim "DotProd is faster on this core" = false` |

**Why neither is graded as the registered result.**
1. **The Windows run is invalid, not ambiguous.** 88 of its 150 cell lines contain `NaN` or `+Inf` speedups (the Linux log has none): the clock read ZERO elapsed over sub-millisecond blocks. The 13/47 above is the rule applied to garbage.
2. **Both runs broke the pre-registered "at least 20 ms per timed block".** The harness sized the block on the `base` arm and used that call count for every arm, so the fast arm's block was `20 ms / its speedup`: about 0.7 ms for int4 (a ~28x arm) on Linux, and below Windows' clock tick there. This was found by reading the NaN lines and then the code, AFTER seeing the results; the defect is an instrument fault and does not depend on which way it pushed the verdict (the Linux run met the claim, the Windows run did not).
3. **The `base` arm's description was wrong for int4** (corrected inline above). It changes what the int4 ratio means, not whether it is real: a core without DotProd runs scalar Go for every int4 matmul, which is what the ~28x measures.

**What run 1 showed, labelled as the out-of-spec observation it is** (Linux only; median over the per-cell medians of the production `dot` arm, 100 cells, all but one FASTER by the rule, the other `AMBIGUOUS (noisy)` at IQR/median 0.17): int4 decode (M=1) 28.1x (range 23.0-28.4), int4 prefill (M=64) 35.6x (29.7-37.4), int8 decode 1.61x (1.43-1.86), int8 prefill 4.00x (2.82-4.56). The int4 figures are large enough that a 0.7 ms block cannot have manufactured them; the int8 decode figure is the one a short block could move.

**Decision, made before run 2 and independent of its result.** Run 2 is the registered measurement. **The rule, the bands, the cells and the 21 rounds are unchanged.** The instrument is fixed in aikit: each arm is calibrated to its own call count so EVERY arm's block is at least 20 ms; Windows reads `QueryPerformanceCounter` instead of its coarse `time.Now`; and the test now FAILS (rather than prints) if any arm's timing is non-positive or non-finite, or if any timed block is under 90% of 20 ms, and logs the clock and its smallest tick. Plumbing was checked under QEMU with emulated DotProd (its ratios are artefacts and are not quoted). If run 2 disagrees with run 1, both are reported and the registered one (run 2) is graded. Run 1's logs are kept as workflow artifacts of run 37265393092.

## Result

*Run 2 not yet run.*
