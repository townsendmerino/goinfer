# Why the decoder tests outgrew their budgets (v0.25.0 → after; 2026-11-22)

<!-- citations-at: 36dbfbaf1 -->

The question comes from `docs/releases/v0.25.0-checklist.md` §5 item 3. At v0.25.0 (36dbfbaf1), the parity sweep's
`./decoder/ ./tokenizer/` cell ran past its 3-hour timeout on the amd64 CUDA box (16 cores, 62 GB). At v0.24.0 (ccdfb3ad6,
2026-11-05) the same cell took about 1 h 43 min. quick -full's decoder cells also overran the 45-minute cell budget. This is a
read of the per-test elapsed times the gates already recorded, plus by-name probes on a third machine (an 8-core Intel laptop
running Linux). No timed run was made, and no number below is a benchmark. The probe times on the laptop are indicative only:
one run each, no interleaving.

## Evidence

- Then: the v0.24.0 release gates' parity cell (`go test -json`, the same box), held under the release's log directory on that
  box, next to the v0.24.0 quick run's per-cell JSON. That quick run's `decoder` and `decoder [checkpoint]` cells stopped at a
  20-minute cell limit (`FAIL … 1200.0s`), so it is not a complete baseline. **The parity cell is the only complete then/now
  pair**, and it is serial (the sum of per-test elapsed is 103.3 min against a wall of about 1 h 43 min).
- Now: the v0.25.0 parity cell's JSON and timeout dump (`../release-v0.25.0/parity-decoder-timeout-dump.txt`), and quick -full's
  cells re-run alone at 120 min (`../release-v0.25.0/quick-full-linux/decoder-120m.json.gz`,
  `../release-v0.25.0/quick-full-linux/decoder-checkpoint-120m.json.gz`, `../release-v0.25.0/quick-full-darwin/decoder-checkpoint-120m.json.gz`).
- `elapsed-nobara.csv.gz` (here): every top-level test's elapsed and result for the parity cell then and now, and for quick -full's
  decoder cell now against the parity cell then.

## What grew

Parity cell, sum of top-level elapsed: **103.3 min → 145.3 min, plus `TestProgram_L_everyParameterIsLive` still running at
36.5 min when the 3-hour timeout fired** (it took 24.4 min at v0.24.0). New tests add 1.4 min. The rest is tests that existed
then and got slower, almost all of them in one class:

| test | then (s) | now (s) | × | same work? |
|---|---:|---:|---:|---|
| `TestSeams_gates` | 318 | 1734 | 5.4 | yes: 64 → 65 fixtures (flint-tiny added), the same layers rebuilt |
| `TestProgram_P_plants` | 243 | 1114 | 4.6 | yes: the same plants on the same fixtures, plus flint-tiny |
| `TestSeams_plantsGoRed` | 137 | 702 | 5.1 | yes |
| `TestSeams_subCaptureRefusedWhereGateSCannotRun` | 64 (failed then) | 309 | 4.8 | yes |
| `TestProgram_L_everyParameterIsLive` | 1463 | ≥ 2187 (timed out) | ≥ 1.5 | yes; 1427 s in quick -full's checkpoint cell and 22 min alone (`../release-v0.25.0/parity-rerun-program-l-36dbfbaf.log`) |
| `TestAdmissionCensus` | 12.4 | 153 | 12.3 | yes: 2,711 → 2,750 census lines; 11.7 s in quick -full's checkpoint cell |
| `TestPlanAgreesWithAdmission` | 11.9 | 153 | 12.8 | yes: 1,890 → 1,917 cells |
| `TestCPUBatch_everyEligibleFixtureBitIdentical` | 125 | 320 | 2.6 | **no**: 36 → 108 subtests, each as fast as before |
| `TestNgramSpecMLA_parity` | 135 | 300 | 2.2 | not examined |
| DecisionCensus trio (`TestDecisionCensus`, `_secondProcess`, `_twoRunsAgree`) | 0.1 (failed) | 233 | — | **new in effect**: at v0.24.0 they failed at once reading a fixture's `.cache` directory |

The real-checkpoint tests between them did not move: `TestPrefillChunked_bitIdenticalReal0p5B` took 1593 s then and 1592 s
now, in the same process and the same hour as the slowed tests. So the box was not throttled or shared, and the CPU kernels did
not regress.

quick -full, on the same box at 36dbfbaf1, run alone at 120 min: the `decoder` cell's top-level elapsed sums to 80.2 min. The
same seam and program tests account for 62 min of it (`TestSeams_gates` 1680 s, `TestProgram_P_plants` 1097 s,
`TestSeams_plantsGoRed` 682 s, `TestSeams_subCaptureRefusedWhereGateSCannotRun` 299 s), then `TestPlanAgreesWithAdmission`
156 s, the DecisionCensus trio 309 s and `TestCPUBatch` 305 s. The `decoder [checkpoint]` cell (60.8 min) is mostly two tests
that were already there: `TestPrefillChunked_bitIdenticalReal0p5B` (1580 s; at v0.24.0 it was still running when the 20-minute
limit cut the cell) and `TestProgram_L_everyParameterIsLive` (1427 s).

On the M1 Pro Mac, quick -full's `decoder [checkpoint]` cell went from 16.2 to 21.8 min: `TestPrefillChunked_bitIdenticalReal0p5B`
714 → 1001 s and the two G40T0 tests 36 → 54 s each. The same test did not move on the CUDA box. The Mac took macOS 26.7.3 on
2026-11-19, between the two runs. That is not examined further here. The Mac's `decoder` cell has no complete v0.24.0 baseline:
it failed after 88 s then.

## Why

**1. The seam and program gates spend nearly all their time on two checkpoints that are not tiny fixtures.** `programFixtures`
loads every directory under `testdata` whose name *contains* "tiny". That includes `tinyllama-awq` and `tinyllama-gptq`: real
TinyLlama 1.1B checkpoints (22 layers, 2048 hidden, 5632 FFN), each 4.2 GB of f32 weights once loaded at `Quant: "f32"`. Both are
gitignored, so they exist only on boxes that pulled them, and CI never loads them. Per fixture, from the log timestamps of
`TestSeams_gates`: `tinyllama-gptq` took 140 s then and 804 s now, `tinyllama-awq` 138 → 790 s, `gpt2` 21 → 84 s. Those three are
96% of the test now. Every other fixture together is under 75 s, each 0.1–10 s, and each also about 4× slower. The other
`programFixtures` callers (`TestSeams_*`, `TestProgram_P_plants`, `TestProgram_L_everyParameterIsLive`) run the same fixture
set. `tinymistral-248m` matches the same substring but does not load as a text model today.

**2. That work did not get slower in the code. It got slower inside the suite process on that box, and the mechanism is not
pinned down.** On the Intel laptop, with `tinyllama-awq` copied in, at ccdfb3ad6 against the current main (d0da204c4, the
same decoder code as 36dbfbaf1 for these paths):

- Loading the fixture: identical (4.2 GB of f32 heap, 154 f32 matrices, at both commits). Loading every fixture
  `programFixtures` finds: identical (50–51 fixtures, 4.2 GB). A loaded model dropped without `Close` is fully collected at both.
- One 24-token forward on it: 8.1 s and 8.2 s.
- `TestSeams_subCaptureRefusedWhereGateSCannotRun` by name, `-tags goinfer_testhooks` as the gates build it: 21.1 s and 21.2 s
  (untagged: 20.4 s and 25.0 s). Run after the 98 tests that precede `TestCPUBatch` in the quick -full decoder cell, in one
  process: 19.8 s.

So the 5× on the CUDA box is not in the forward, the load or a test hook. What is known about it:

- It shows only in the long processes (the parity cell, and quick -full's `decoder` cell run alone). The same tests in a short
  process run at the old speed: `TestProgram_L_everyParameterIsLive` takes 22 min alone and 24 min in quick -full's checkpoint
  cell, but over 36 min at the end of the parity cell, where it ran alone (it is `t.Parallel`, and the dump shows every serial
  test done). `TestAdmissionCensus` takes 11.7 s in the checkpoint cell and 153 s in the parity cell.
- It hits allocation-heavy work (tiny-model forwards, the census and plan walks: 4–12×) and spares compute-bound work (the
  real-checkpoint prefill tests: 1.00×).
- The process carries about 8 GB more mid-suite than it did. The `[g40-kv]` footprint line reads 6.1 → 14.4 GB in the parity cell
  (13.9 GB in quick -full's decoder cell), and `TestMadvise_dontneedFreesRSS` reads VmRSS 9.2 → 17.8 GB.
- The suite runs under a soft `GOMEMLIMIT` of 8 GiB and calls `debug.FreeOSMemory` every 3 s (`decoder/suite_main_test.go`:
  `boundSuiteMemory`, `startMemReclaimer`). Both are unchanged since v0.24.0. Their comments assume a live heap of a few MB.
  `programFixtures` alone holds 8.4 GB live (the two TinyLlamas) on that box, above the limit.

The leading hypothesis is GC pressure: a larger retained heap, or one held over the soft limit, makes every collection and
allocation dearer for the allocation-heavy tests. On the laptop, holding the one 4.2 GB fixture over a 3 GiB limit slowed the
subCapture test 1.4–2×, at both commits, which is short of 5×. What retains the extra 8 GB is not identified. Nothing measured
here explains why v0.24.0, with the same two fixtures loaded together, did not show it. Settling it takes one run on the CUDA
box, which this record did not make: the decoder cell with `GODEBUG=gctrace=1` (and `GOINFER_MEM_PROBE=1`), compared against the
same cell with `GOINFER_TEST_MEMLIMIT=off`.

**3. `TestCPUBatch_everyEligibleFixtureBitIdentical` runs three times the cells.** 871092d2f (2026-11-09, P4a: "the CPU batcher
reads f16 and int8 KV caches") added a KV-precision axis (`""`, `f16`, `i8`) to every fixture × quant cell: 36 → 108 subtests.
The per-subtest times are unchanged (qwen3-1.7b: 15.9 s then, 16.0 / 16.0 / 15.9 s now). The heavy cells are the real
checkpoints in `testdata` (qwen3-1.7b, llama3.2-1b, the TinyLlamas, qwen2.5-0.5b), not the tiny fixtures.

**4. The DecisionCensus trio is new work, not a slowdown.** At v0.24.0 all three failed in under 0.1 s (`read …/.cache: is a
directory`). 270e59f9f ("census skips directories", 2026-11-05) and 6d0e55f5e / 632530317 (2026-11-06: the census reads its
declared fixtures) made them run. They now take 233 s in the parity cell, 309 s in quick -full's decoder cell and 570 s on the Mac.

## Proposals

Per contributor. The budgets, and any change to what a gate covers, are the owner's call.

| contributor | now (parity, CUDA box) | proposal | kind |
|---|---:|---|---|
| TinyLlama AWQ/GPTQ (and gpt2) in `programFixtures` | ~1594 s of `TestSeams_gates`; the bulk of the 4 other callers (not split per fixture) | match "tiny" as a name token (`llama-tiny`, `tiny-qwen2-moe` and `gptoss_tiny.gguf` stay; `tinyllama-*` and `tinymistral-*` go). AWQ and GPTQ keep their own parity tests (`TestAWQ_parity`, `TestGPTQ_parity`) | test fix, but it drops two quant formats from gates H/C/S/P/L on the boxes that have them, so it goes to the owner |
| the unexplained 4–12× in long processes | ~50 min of the parity cell | one gctrace run of the decoder cell on the CUDA box, by day, before choosing a fix | diagnosis |
| `TestCPUBatch` KV axis | 320 s | run the `f16` / `i8` cells on the tiny fixtures only; real checkpoints keep the plain cache | test fix, but it narrows coverage |
| DecisionCensus trio | 233–570 s | none: new coverage | budget |
| `TestProgram_L` / `TestPrefillChunked_bitIdenticalReal0p5B` in the checkpoint cell | 1427 + 1580 s | move one of the two into its own cell | cell split |
