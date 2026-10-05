# DotProd versus the base kernels on Azure Cobalt 100 (the Windows ARM question): pre-registration, 2026-10-04

**Status: PRE-REGISTERED, NOT RUN.** The rule below is committed before any run and is not edited after one. The result goes in a "Result" section at the end, with this text unchanged above it.

## The question

aikit v1.56.0 probes DotProd on Windows on ARM, so cores that have it (Snapdragon X, Azure Cobalt 100) run the SDOT kernels and the row4 int4 layout instead of the base SMULL/SADALP kernels and the canonical layout.
How much faster, if at all, is that, on the projections a decode and a prefill actually run? The hardware-coverage task left this as "unmeasured" (`docs/tasks/task-hardware-coverage-2026-10.md`).

## What can and cannot be measured

- **Machine.** GitHub's free arm64 runners, `ubuntu-24.04-arm` and `windows-11-arm`: **both Azure Cobalt 100 (Neoverse N2)**, 4 vCPU, shared VMs. Kernel speed does not depend on the OS, so the Linux run answers the kernel question and the Windows run
  confirms it on Windows (scheduler, heap-read model). **Snapdragon X (Oryx cores) is NOT measured and a Cobalt result does not transfer to it.** There is no way to measure it without owning one.
- **Instrument.** `TestDotProdVsBase_AB` (aikit `linalg/dotprod_ab_arm64_test.go`, workflow `dotprod-ab`, on demand): the arms run back to back in ONE process (hasDotProd is a package flag read at call time), round after round, alternating
  which arm goes first. Not a stored baseline, and not two processes minutes apart.
- **Arms.** `base`: canonical weights, hasDotProd=false. `dot`: the PRODUCTION DotProd path: int4 as the repacked-only row4 layout (what goinfer's loaders build on arm64), int8 as is, hasDotProd=true. `dot-canon` (int4 only): canonical layout with SDOT, for attribution.
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

## Result

*Not run.*
