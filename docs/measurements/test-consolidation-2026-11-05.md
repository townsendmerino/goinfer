# Test suite consolidation, 2026-11-04 to 11-05

Francis's brief: review the `*_test.go` files package by package alongside `go test -race -cover` output; convert sprawling,
repetitive tests to table-driven ones; add `t.Parallel()` where safe; add fuzz tests for parsers and decoders; delete brittle
tests that assert only implementation details. Branch `test-consolidation-2026-11`. Ordinary engineering under CLAUDE.md's
"Rigor matched to stakes": no registration, each changed test shown red on a planted defect.

The prior art is [`test-audit-2026-10-15.md`](test-audit-2026-10-15.md) (slow tests classed shrink, share, parallel, duplicate
or leave; the `-race` subset rule) and [`ci-tail-2026-10-27.md`](ci-tail-2026-10-27.md) (the census moved to its own shards).
This audit builds on both.

## Sources

- **The tree:** a `go/ast` census of every test file: 5,120 tests, 18 fuzz targets, 144 benchmarks, 238,613 lines of tests;
  `t.Parallel` in 5 tests. A second census compared every pair of `Test` functions by token shingles twice, once with
  identifiers erased (structure) and once with them kept (the same calls); a copy-paste pair scores high on both.
- **CI:** job durations of 60 successful `ci.yml` runs on `main`, 2026-11-02 to 11-04; per-test times from the slowest-tests
  tables of run 41910585116; coverage from `coverage.yml` run 41910584864, which runs without `-race`.
- **The Mac (M1 Pro):** per-test times without `-race` for the packages changed here, under the shared timing lock. No
  whole-tree `-race -cover` run: the v0.24.0 release gates held the Mac on 11-04, and CI's logs already carry both.

## Where the time goes

CI's wall time is set by the race-free root jobs, not the `-race` shards:

| job | minutes, 10 runs sampled | bounded by |
|---|---|---|
| `root-windows` | 12.2 to 19.7 | `decoder` (1,016 s), `cmd/gate` (516 s), `internal/runtime/sampling` (405 s) |
| `root-darwin` | 7.9 to 18.4 | `decoder` (930 s) |
| `root-windows-arm64` | 11.1 to 15.0 | `decoder` |
| `test-decoder-d` | 5.4 to 11.4 | `TestDecisionCensus_secondProcess`, 496 s under `-race` |
| `test-decoder-a` | 5.7 to 11.1 | `TestDecisionCensus`, 368 s under `-race` |
| `test-rest` | 5.7 to 11.8 | `TestSolvePlacement_monotone`, 209 s under `-race` |

- **The decision census is the largest single cost.** One build is about 22 s on a quiet M1 Pro and 38 s under load, and a
  whole-package run built it four times: twice in `TestDecisionCensus` (G-MM1-C2 wants two in one process) and once each in
  `_secondProcess`'s parent and child.
- **Its CPU is allocation, not decisions.** A profile of one run: `runtime.lock2` 33% of samples (heap-span contention),
  `mallocgc` 20%, and `(*Model).Plan` rebuilding its program (`buildProgram`, `layerOps`) on every bisection step 24%.
- **A GC setting does not help.** The census at `GOGC` 100, 400, 1600, interleaved: 400 cut CPU 22% but left wall time
  unchanged (68 s against 68 s) and doubled peak memory (1.2 to 2.4 GB); 1600 reached swap. Not adopted.
- **`t.Parallel` across the board does not pay.** Of the 1,169 `decoder` tests that finished before a timing run was stopped (the census and `Real0p5B` excluded), 35 take 189 of
  their 215 s on the Mac.
  Most of those 35 set env or a seam through a helper (`loadBenchModel`, `setKnob`, `useProbe`), or read a global counter.

## What changed

Each change is shown red on a planted defect. The planted defects were applied with `go test -overlay`, so the tree never
held one.

| change | before | after | planted defect, red |
|---|---|---|---|
| `TestDecisionCensus_secondProcess` reuses `TestDecisionCensus`'s total when both run in one process | 4 census builds per whole-package run | 3; alone, as on `test-decoder-d`, it builds its own as before | a child printing another total, red both ways |
| `TestSolvePlacement_monotone` walks its 1,500 generated loads across `GOMAXPROCS` goroutines, keeping the lowest break | 17.1 s | 7.2 s; 41 s under `-race` with no race; the same 919 of 1,500 draws and the same first break | a solver that refuses above 40 GiB free |
| `t.Parallel` on three `decoder` tests and four `constrain` property tests that write no package state | 116.8 s under `-race` (`decoder` three) | 54.7 s at `-parallel 4`; `constrain` four 2.7 to 1.2 s | `-race -count=2` over each set: no race |
| eight tiny-fixture parity tests on one helper, `textParity`, each keeping its name, checks and manifest row | 8 bodies of 63 to 83 lines | 8 wrappers | logits, argmax, continuation, the Generate path, the per-family check, a missing golden: each red on exactly the tests that use it |
| other true duplicates: the CUDA and Metal planted-gap tests, the stream-transcode cancellations, the fit estimators, the VL resident-cap and image-reuse tests (helpers, names kept where docs cite them); the Cohere prefill pair, the default-on flag pair, the Anthropic ignored-block pair (tables) | | | a plant through each helper |
| `TestPrefillCoverageAudit` fails when its table leaves out a family; it asserted nothing | | | an unknown family |
| three source-text tests become behaviour tests (below) | | | each one's own defect; a rename that broke the old match now passes |

Before and after the duplicate merges, the same tests pass and skip and all 19 figures they log are identical. Test files: 1,073
lines removed and 714 added in changed files, and 422 in new files (mostly fuzz targets).

**Correction to an interim report:** `TestAdapterProbe` was listed with the tests that cannot fail. It is load-bearing:
`cmd/gate`'s `detectWebGPU` runs it and parses its `ADAPTER_PROBE:` line. It stays. `TestSpike_capabilities` stays too: it
documents that it never fails and is a tool run by hand (`docs/tasks/parked/task-kernel-survey-2026-10.md`).

## Fuzzing

`fuzz-weekly.yml` named `internal/giw` and `internal/serveapp`, both since moved, so 6 of its 15 rows would have failed on the
next Monday, and `FuzzFencedToolCall`, `FuzzToolCallsGrammar` and `FuzzPreprocess` were never listed. The matrix now has 31 rows,
and `TestFuzzWeekly_listsEveryTarget` holds it to the targets in the tree both ways.

Thirteen new targets cover the untrusted parsers that had none. Where a property exists beyond "no panic", the target asserts
it, and the seeds alone (which every `go test` replays) catch the defect:

| target | property | planted production defect |
|---|---|---|
| `FuzzParseIDsBlob` | a parsed blob rewrites as itself; one that verifies carries every signed field of the sealed one | the flags, or the namespace, left out of the MAC |
| `FuzzParseWAV` | `Samples()` equals the decoded length (the refusal before decode relies on it) | a header under-reporting by one |
| `FuzzParseInput` (batch files) | an accepted line has a `custom_id` | an empty one accepted |
| `FuzzParseRef` | an accepted repo is one owner and one name that stays under the cache | `ParseRef` without `validRepo` |
| `FuzzEncode` | ids in the vocabulary; ASCII without the added-token parse round-trips | |
| `FuzzParseQwen3PreprocessConfig` | an accepted config preprocesses to the grid it reports | |
| `FuzzParse` (clef), `FuzzReadGGUFMeta`, `FuzzLoadTekken`, `FuzzServeEmbeddingsInput`, `FuzzServeSystemOne`, `FuzzSplitThink`, `FuzzLoadConfig` | no panic, and for `SplitThink` no output longer than the input | |

`TestParse_refusesNestingDeepEnoughToOverflowTheStack` pins clef's guard: its parser recurses, and without `json.Valid` in front
of it a balanced 2 MiB-deep document is a fatal stack overflow.

**One product bug, fixed.** `ParseQwen3PreprocessConfig` checked only that sizes were positive. At the base commit,
`temporal_patch_size` 2^50 panicked at the first image (`makeslice: len out of range`), a `patch_size * merge_size` that wraps to
zero returned empty pixel values with no error, and `patch_size` and `merge_size` of 2^15 allocated until the process was killed
after 38 s. The parser now bounds them (changelog fragment `2026-11-05-qwen3-preprocessor-config-limits.md`).

## Brittle tests

The census found 27 tests that match a string in source text, some of them on fixtures `cmd/gate` writes itself. Those on
production code mostly say why: the behaviour needs a GPU, a device OOM, or a race
that would be flaky in the passing direction (`TestStartDrain_savesSessionsUnderTheLRULock`,
`TestDeclineAdvice_wiredIntoBuildResident`, `TestEnsureAttnWide_guardsOnTheLastPipeline`). Those stay.

Three could be behaviour tests today, and now are. `TestAnthropic_decodeUsesTheSharedShaping` decodes malformed bodies through
`decodeAnthropicJSON`. `TestRequestLog_wiredIntoTheGenerationRoutes` builds the binary's route table: its premise, that the mux
was built inside the entry point, no longer held. `TestDecoderEmbedder_boundComesFromTheModelNotZero` constructs the embedder.
Their scan rows in `scripts/touched_scans.py` go.

No test was deleted outright. The census found no test that exercises only a mock: the tests that never reach production code
are kernel tests on real devices, real-binary tests and layout guards.

## For Francis to decide

1. **The largest duplication is in gates.** 45 near-identical tests: 17 `*Real_gate`, 14 `*_forwardParity` and 14 `*_textParity`,
   3,345 lines beyond one copy each. 32 of them are in `testdata/gate_ledger.json`, which keys a person's confirmation on the
   test's own body (`gateFuncSource`), so a rewrite voids those confirmations, and a shared helper's body would not be hashed at
   all. Converting them needs a re-promote of each and a ledger key that covers the helper. The 8 non-ledger `_textParity` tests
   are converted here.
2. **arm64 census golden.** `TestDecisionCensus` builds the census twice on every arm64 host (`root-darwin`,
   `root-linux-arm64`, `root-windows-arm64`) and then skips: the golden has only the amd64 class. Recording an arm64 class would
   turn three skips into checks, at the cost of a Mac capture whenever the census moves.
3. **`Plan` rebuilds its program on every call.** It is about 24% of the census's CPU, and a memo would be a production change,
   so it was not made here.
4. **The census reads whatever fixtures are on disk.** Cloning three gitignored fixtures moved it from 29,637 to 31,155 groups.
   That is why the second-process test does not compare a lone child with the golden.

## Coverage, from CI (statements, without `-race`)

`decoder` 74.9%, `internal/app/serveapp` 72.7, `tokenizer` 70.3, `multimodal` 57.8, `chat` 93.2, `constrain` 91.9,
`internal/runtime/giw` 85.0, `pull` 85.7, `internal/batchio` 90.6, `internal/clef` 90.8, `internal/app/chatapp` 48.9,
`internal/runtime/scheduler` 41.6, `internal/app/pullcmd` 7.0. Packages with no tests of their own (`internal/codec`,
`internal/runtime/f16`, `hybridattn`, `mamba2`, `prequant`) were not traced to the tests that cover them from outside.

## Notes

- **This audit's load overlapped the v0.24.0 `quick -full` gate on 11-04.** Its `decoder [checkpoint]` cell hit its 20-minute
  timeout, and `TestPrefillAttnRowTileInvariance` reported `context deadline exceeded` partway through its cases. The gate's
  other reds were deterministic and not load: the `gpt2/.cache` directory in the fixture copy, missing prose seeds, and
  `TestPlan` figures.
- **The 2^15 config probe swapped the Mac** during that gate (load 72), because it was run without a memory bound.
- **Three tests skip on CI because their fixture is gitignored:** `TestGranite_textParity`, `TestQwen3VL_textParity` and
  `TestGenerateGemma4VL_residentContextCapPublishesBudgetClamped`. They ran here on cloned fixtures.
