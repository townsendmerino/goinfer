<!-- citations-at: 6213c6e9443ac4c6daa0d42bf589fea4843d330f -->
# H1: the CUDA heavy tier, test by test: keep, shrink or move (2026-11-23)

Analysis only. No test was run and none is changed here. **Every move, shrink and the time budget in §5 are proposals
for the maintainer**, item by item. Per "Rigor matched to stakes", this is ordinary engineering: no registration and no
timed run. The numbers are existing readings, and each one names its source.

## 1. The question and the sources

H1 asks which heavy tests over about 2.5 minutes should stay in the release gate's heavy tier, which should shrink, and
which should move to an on-demand measure tier. It also asks for a draft per-test time budget.

**Only the CUDA box has a heavy tier.** `cmd/gate/gpu.go:gpuGate.heavyTier` runs `./backend/cuda/` and
`./backend/cuda/kernels/` under `GOINFER_HEAVY_TESTS=1`. The main group runs one process per model family
(`cmd/gate/heavy_groups.go:heavyFamilies`), then the drain group, then each fresh-process test alone. The Metal gate has no
heavy tier. Its suite cell, with `GOINFER_HEAVY_TESTS=1`, took 113.5 s in all at v0.25.0
(`docs/measurements/release-v0.25.0/gate-gpu-metal-36dbfbaf.log`, § "2. Metal suite"). So **no Metal test reaches 2.5
minutes, and every saving below is CUDA's; the Mac's is 0.** The Mac's long real-checkpoint tests run in `gate quick
-full`, not in this tier. They are listed in §6 for context only.

**Sources, in order of use:**

- **G (the v0.25.0 heavy run).** The v0.25.0 release run of `gate gpu` on the CUDA box (RTX 2070 SUPER, driver 595.112.07)
  at 36dbfbaf1, 2026-11-21. Its log is `docs/measurements/release-v0.25.0/gate-gpu-cuda-36dbfbaf.log` and its record
  `docs/measurements/gpu-gate/cuda-20261121T222959Z-36dbfbaf.json`. It is the newest full heavy run committed.
  - The log does not print per-test durations. It prints a heartbeat each minute per process: the test in flight and its
    age.
  - So a G figure is a **bound**. The lower bound is the test's largest in-flight age, and the true duration is under
    that plus 60 s.
  - The list is complete. A test of 2.5 minutes or more is in flight at two or more heartbeats, so it is shown at an age
    of at least 1.5 minutes. Every such test is in the tables below.
  - A process with one test has an **exact** wall in the record's `heavy_processes`.
- **B23 (the 2026-10-14 heavy run).** The exact top-level elapsed times in
  `docs/measurements/b23-cuda-heavy-split-2026-10-15.md` § "3. The CUDA test audit", from the 2026-10-14 heavy run. That
  run is older, and tests added since are absent from it. It is quoted beside G as corroboration, never in its place.
- **Not read: the v0.25.0 run's own per-test streams.** They hold the exact Elapsed of every test and stay on the CUDA
  box, in its gate log folder for 2026-11-21, one `gate_gpu_cuda-heavy-<group>.json` per process. The box's timing lock
  was held by a timed run all through this analysis, so nothing was copied. A follow-up can replace every G bound below
  with an exact figure (§7).

**The brief's headline figures, checked against G:**

| figure | value in G |
|---|---|
| listed heavy tests | 855 (770 in `./backend/cuda/`, 85 in its `kernels/` package) |
| the whole `gate gpu` run | 4 h 35 min (provenance 17:55 UTC to the record at 22:29 UTC); this is the brief's "4.6 h" |
| the heavy main-group processes | 14,047 s, 3.9 h (the sum of `heavy_processes[].wall_s`) |
| KV10 in the gate | 715 s, exact (the `kv10` process holds one test) |
| the ten slowest tests | at least 84 min by G's lower bounds, so at least 36% of the 3.9 h |

- The brief's "883" and "48%" are not reproduced from G. Their source is not named, and the exact streams would settle
  them.
- KV10's 58 minutes is the KV10 campaign's own run at three depths (`docs/measurements/kv10-2026-11/README.md`). The gate
  runs two depths (`cmd/gate/gpu.go:heavyGroupEnv`) and took 715 s.

## 2. The tests at 2.5 minutes or more (G's lower bound)

Columns:

- **min (G)** is the lower bound, with the upper bound after the slash. "=" marks an exact process wall.
- **B23** is the 2026-10-14 exact time in seconds.
- **Guards?** is what the body can fail on, read from its `t.Error` and `t.Fatal` calls, not from its name. "Measures"
  means it fails only on an error (load, I/O, a decline it cannot run past) and logs its result.
- **Campaign** is the archived task doc, or the item's state. Several task docs now live in the private notes checkout,
  not in this tree. They are named here in prose.

| # | test | group | min (G) | B23 s | guards? | campaign | verdict |
|---|---|---|---|---|---|---|---|
| 1 | `TestKSO1C_longContextQuality` | small | 19.4 / 20.4 | n/a | **guards**: on each of two repo documents (900 tokens), the native lane's mean KL to f32 past position 256 must be at most 1.05× the int8 lane's | KS-O1 (O1C), done; its kernel-survey doc is parked | **shrink** |
| 2 | `TestP1r_nemotronRangeSurvey` | nemotron-h | 13.0 / 14.0 | n/a | barely: it fails on a NaN in K/V or on max\|K\|,\|V\| ≥ 32,752. The recorded worst is 44.67 (`docs/measurements/p1r-nemotron-f16-2026-11/README.md`), so the bar has about 730× headroom | P1r, done | **shrink** (was move; §8) |
| 3 | `TestKV10_longContextFidelity` | kv10 | = 11.9 (715 s) | n/a | **measures**: `kv10Judge` fails only on a NaN or Inf in the logits or the metric. The bands are `GATE_READING` lines and the drift is a `GATE_NOTICE` | KV10, **active** (the meaning of its two cosine FAILS is the maintainer's open decision) | **keep** (was shrink; §8) |
| 4 | `TestS10DeepstackPrefillCUDA_real` | qwen3-vl-2b | 8.4 / 9.4 | 538 | **guards** (G-S10g): per image, the worst cosine must be within 0.005 of the text control's minimum, and every argmax split must be a near-tie | `docs/completed/task-multimodal-support-2026-10.md`, archived | **shrink** |
| 5 | `TestGemma4BidirSpanPrefill_real26BDecode` | gemma4-26b | 7.2 / 8.2 | n/a | **measures** (fails only on errors; logs the first-difference step) | same doc (S19), archived | **move** |
| 6 | `TestGemma4BidirSpanPrefill_real26B` | gemma4-26b | 6.0 / 7.0 | n/a | **guards** (G-S19c): per image, the resident span prefill's last argmax equals the CPU bidirectional prefill's (3 images) | same, archived | **shrink** |
| 7 | `TestGemma4EModel_realE4BF32Dump` | gemma4-e4b | 5.3 / 6.3 | n/a | **measures**: a CPU f32 dump written outside the repo; it depends on `..._realE4BAnchorDump` having run first | same (S1/S6), archived | **move** |
| 8 | `TestKSO1C_towardTheFile` | small | 5.2 / 6.2 | n/a | **guards** (O1C-G6): per model, native KL / int8 KL must be at most 1.0, on the 0.5B and on Phi-3 | KS-O1, done | **shrink** |
| 9 | `TestPrefillGateVsReferenceCUDA` | 7b-8b | 4.6 / 5.6 | 307 | **guards**: the prefill-gap §3 criteria A/B/C at K 512 and 1024 against the CPU f32 reference. The fast-prefill default rests on it | `docs/completed/task-prefill-gap.md`, archived; the gate is still load-bearing | **keep** (optional trim) |
| 10 | `TestPrefillLongPrompt` | fresh process | 4.0 / 5.0 | 248 | **measures**: a batched decline is printed and skipped past, and `PrefillPath` is printed, not asserted | `docs/measurements/prefill-chunking-d7-2026-09-04.md`, done | **shrink** (was move; §8) |
| 11 | `TestGemma4_26BActQuantSensitivity` | gemma4-26b | 3.8 / 4.8 | n/a | **measures** by default (it guards only with `GOINFER_S19_FLOATACT=1`, which the tier does not set). It is CPU-only, although it sits in `backend/cuda` | S19, archived | **move** |
| 12 | `TestDefaultVerifyWidth_sweep` | qwen3-4b | 3.5 / 4.5 | 246 | **measures** (tok/s per width; 288 generations) | `docs/measurements/default-verify-width-2026-08-25.md`, done | **move** |
| 13 | `TestGemma4EModel_realE4BNonInferiority` | gemma4-e4b | 3.0 / 4.0 | 224 | **guards** (G-E4B-C1b, registered margins against the 1.5B reference) | S1/S6, archived | **keep** |
| 14 | `TestGridVisionCUDA_real` | qwen3.6-35b (by label) | 3.0 / 4.0 | 240 | **guards** (G-S2b): per image, the worst merged-token cosine against aikit's CPU tower must be at least 0.9999 | multimodal S4, archived | **shrink** |
| 15 | `TestGemma4_26BDecodeLocalize` | gemma4-26b | 2.9 / 3.9 | n/a | **measures** by default. With `GOINFER_S19_CPUFLOAT=1` it is the registered G-S19f gate, but the tier does not set that variable | S19, archived (G-S19f read PASS) | **keep, armed** (was move; §8) |
| 16 | `TestS17LeverA_wholeTower` | gemma3-4b | 2.7 / 3.7 | 198 | **measures** lever A. It fails only on the profiler hook's own checks, which `TestS17ProfileCUDA` also makes | multimodal S17, archived | **move** |
| 17 | `TestGlmOcrCUDAResidentRealImage` | glm-ocr | 2.5 / 3.5 | 184 | **guards** (O3), on three images: resident against CPU, prefill cosine ≥ 0.99 with equal argmax, and 64 greedy tokens identical | `docs/completed/task-glm-ocr-2026-10.md`, archived | **shrink** |
| 18 | `TestKV11TradeCurve` | gemma4-26b | 2.5 / 3.5 | n/a | **measures** (fails on a decode error, or on fewer than two baseline cells). It says it is a timed run to be held under the exclusive timing lock, and the tier does not hold that lock. Each void cell can add a minute's sleep, up to four | KV11, blocked on KV7 | **move** |
| 19 | `TestKV0_continuationEffect_real1p5B` | small | 2.5 / 3.5 | 165 | **guards** one thing: exact-class prefill equals decode bit for bit. `TestKV0_continuationEffect_tiny` runs the same body and assertion; the continuation effect itself is only logged | KV0, done | **move** |

## 3. Borderline: under 2.5 minutes by G's lower bound, over it by the upper bound or by B23

| test | group | min (G) | B23 s | guards? | verdict |
|---|---|---|---|---|---|
| `TestSiglipCUDA_speed` | gemma3-4b | 2.1 / 3.1 | 181 | **measures** ("no bar"; about two thirds of its time is the CPU arm) | **move** |
| `TestGemma4EModel_realE2BNonInferiority` | gemma4-e2b | 2.0 / 3.0 | 149 | **guards** (G3c, registered) | **keep** |
| `TestSpecPagerInteraction` | fresh process | 2.0 / 3.0 | 181 | barely: one instrument check (the off arm's distinct experts per stage within ±2% of topK). R1 to R3 are only logged. The question is closed (`docs/measurements/spec-x-pager-2026-09-02.md`) | **move**, with the instrument check folded into `TestPagerDeterminism` |
| `TestKSO1C_realPhi3NonInferiority` | small | 2.0 / 3.0 | n/a | **guards** (O1C-G5) | **keep** |
| `TestWalk_plantsGoRed` | small, and the parity cell | 1.7 / 2.7 | n/a | **guards**: each of three plants must turn at least one fixture's walk red. It has no heavy check, so it runs in both cells | **shrink** |
| `TestKV3_batchedPrefillPastTheSpanBound_tiny` | small, and the parity cell (2.2 / 3.2 there) | 1.7 / 2.7 | 158 | **guards**: batched against per-token past the span bound, with a control | **shrink**: run it in one cell |
| `TestKSH6_marginParts26B` | gemma4-26b | 1.6 / 2.6 | n/a | **measures** ("a reading, not a gate"). `TestKSH6C_marginPriced26B` guards the margin | **move** |
| `TestGlmOcrCUDAResidentFullBudget` | glm-ocr | 1.6 / 2.6 | 169 | **guards** (6,156-token chunking and the position delta; no cheaper test holds it) | **keep** (optional trim) |
| `TestGemma4EModel_realE4BAnchorDump` | gemma4-e4b | 1.6 / 2.6 | n/a | **measures** (an exploratory dump; at about 36 GB, the largest host RSS of its process) | **move** |
| `TestKV2_scheduleIndependence_real1p5B` | small | 1.6 / 2.6 | n/a | **guards**, but `TestKV2_scheduleIndependence_tiny` holds the same schedules in three arms | **move** |
| `TestBlockSpec_twoTurnsMatchPlain` | qwen3-4b | 1.6 / 2.5 | n/a | **guards** (the real-kernel two-turn check; only here) | **keep**, unchanged (§8) |

## 4. Each verdict: what it saves and what it loses

Savings use G's lower bound: a move saves at least that much. A shrink's "after" is a **projection by case count, not a
reading**. Every shrink keeps the same assertion unless it says otherwise.

### Move

> **Superseded in part by §8.** Of the moves below, P1r and `TestPrefillLongPrompt` became shrinks and `TestGemma4_26BDecodeLocalize` a keep. §9 is the standing list.

The proposal before review: fifteen tests, at least 56 minutes.

**The measure-only tests.** Each runs on demand through the measure tier, its record or its run script, and none asserts
a result that a regression could turn red:

- the 26B decode diagnostic (5), the E4B f32 and anchor dumps (7, and `..._realE4BAnchorDump` in §3), `TestPrefillLongPrompt`
  (10), `TestGemma4_26BActQuantSensitivity` (11), `TestDefaultVerifyWidth_sweep` (12), `TestGemma4_26BDecodeLocalize` (15),
  `TestS17LeverA_wholeTower` (16), `TestKV11TradeCurve` (18), `TestSiglipCUDA_speed` and `TestKSH6_marginParts26B`.

What each move loses:

- **5, 11 and 15** lose no assertion the tier runs. The defect they found is held on tiny by `TestGemma4MoE_floatParity`
  and by the float-activation kernel tests in `backend/cuda/kernels/` (G-S19e).
  - 15 has an alternative: set `GOINFER_S19_CPUFLOAT=1` in `cmd/gate/gpu.go:heavyGroupEnv`, so that it becomes the real
    G-S19f guard on the 26B. That is a gate-definition change, so it is the maintainer's choice. **The recommendation is
    the move**, since the tiny parity holds the contract.
- **7 and the anchor dump:** moving both also drops the gemma4-e4b process's peak host RSS from about 45 GB, because the
  dump is the largest test in it.
- **10** loses nothing it asserts. "Batched does not decline at long M" is asserted on tiny by
  `TestKV3_batchedPrefillPastTheSpanBound_tiny`. If a real-model guard is wanted instead, the shrink is to drop the
  sequential arm and make the decline at 8,012 an error.
- **18** is also wrong to run in the tier as it stands: its own comment requires the exclusive timing lock, and the tier
  holds the lock shared.

**The narrow guards whose contract a cheaper test already holds:**

- **P1r (2)**, at least 13 minutes. It is a range survey with a NaN check. Moving it loses the NaN-in-K/V check on the
  real Nemotron. `TestNemotronResidentCUDA_tier1` stays in the same process. Alternative: shrink it to 1 prompt (a
  projected 1.3 minutes), which keeps the identical assertion given the headroom.
- **KV0 on the 1.5B (19)** and **KV2 on the 1.5B (§3):** each tiny twin runs the same body and assertion. The move loses
  the real-checkpoint instance only.
- **`TestSpecPagerInteraction`:** fold its one assertion (distinct experts per stage equals topK on the off arm) into
  `TestPagerDeterminism`, which already loads the same 35B. That saves a 35B load and a fresh process.

### Shrink

> **Superseded in part by §8.** KV10 and `TestBlockSpec_twoTurnsMatchPlain` became keeps. §9 is the standing list.

The proposal before review: ten shrinks, plus one optional trim (the last row).

| test | shrink | now (G, min) | projected | loses |
|---|---|---|---|---|
| KSO1C long context (1) | one document instead of two | ≥ 19.4 | ≈ 10 | the second text. The bar is relative (native against int8), so one text still tests it |
| KV10 (3) | `GOINFER_KV10_DEPTHS` = `3900` in the heavy tier; 65,536 goes to the measure tier (or the weekly run) | = 11.9 | 13 s at 3,900 (KV10 record, depth table) | the NaN check at 65,536. It guards nothing else there. **KV10 is active and its gate depths are a gate definition: the maintainer's call** |
| S10 real (4) | `ctlPrompts` 12 to 4 for the three images registered against the 4-prompt control | ≥ 8.4 | ≈ 4 (52 CPU prefills to 28) | nothing in coverage. But a smaller control has a higher minimum (the test's own comment), so **the bar tightens and could turn green red**. Check the S16 registration's 4-prompt readings first |
| 26B span prefill (6) | one image instead of three | ≥ 6.0 | ≈ 2 | two images' argmax. The tiny `TestGemma4BidirSpanPrefill_matchesCPU` and `_plantedDefects` hold the contract more strictly |
| KSO1C toward the file (8) | 4 prompts instead of 8 | ≥ 5.2 | ≈ 2.6 | half the prompts. Both models stay, since the 0.5B has the thinner margin (0.91×) |
| grid vision real (14) | two images per family (one square, one tall) | ≥ 3.0 | ≈ 1.5 | two aspect ratios per family. The tiny `TestGridVisionCUDA_matchesCPU` holds the bar |
| GLM-OCR real image (17) | the invoice only (the longest prompt), and 16 generated tokens instead of 64 | ≥ 2.5 | ≈ 1 | the table and formula images |
| walk plants (§3) | stop each plant at the first fixture that differs; run the legacy loop once rather than once per plant | ≥ 1.7, ×2 cells | seconds | nothing: the assertion is "at least one fixture differs". Check that the legacy path is plant-independent first |
| KV3 tiny past the span bound (§3) | run it in one cell, not both; optionally 1 seed per length | ≥ 1.7 + 2.2 | one run | nothing with the dedupe; with 1 seed, two seeds per length |
| two turns match plain (§3) | load the plain model and the spec pair once and reuse them across N | ≥ 1.6 | ≈ 0.6 (9 loads to 3) | the fresh-load-per-arm isolation. The resident must be reset between cases |
| prefill gate vs reference (9), optional | drop the ungraded K=256 cell and the 1.5B's K=3,900 confirmation cell | ≥ 4.6 | not projected (no per-cell times) | nothing graded |

### Keep

These have registered real-checkpoint bars that nothing cheaper holds at that fidelity:

- `TestPrefillGateVsReferenceCUDA` (9)
- `TestGemma4EModel_realE4BNonInferiority` (13)
- `TestGemma4EModel_realE2BNonInferiority`
- `TestKSO1C_realPhi3NonInferiority`
- `TestGlmOcrCUDAResidentFullBudget`
- `TestBlockSpec_twoTurnsMatchPlain` (unchanged, after §8)

One possible trim: the E4B and E2B gates each re-measure the same 1.5B reference. They run in different processes, so
sharing the reference is a design change, not a constant. Fewer prompts would change a registered rule, so it is not
proposed.

### A structural finding: the `small` process re-runs the parity cell

- The heavy partition's `small` group is the complement of the family groups. It is 620 tests and took 4,673 s: 78 minutes,
  the largest process in the tier.
- A test with no heavy check runs in it and also in the parity cell (2b, 1,268 s). Two of §3's tests (`TestWalk_plantsGoRed` and
  `TestKV3_batchedPrefillPastTheSpanBound_tiny`) were seen in flight in both cells.
- How much of `small` is tests the parity cell already ran is **not known** without the per-test streams. It is the
  largest single lever in the tier.
- The fix belongs in `cmd/gate`: restrict `small` to tests that reach a heavy gate, or exclude the parity cell's tests from
  it. It is a gate-definition change, so it is the maintainer's call.

## 5. Draft: a per-test time budget (the threshold is the maintainer's)

> **Decided, Francis, 2026-11-23 (via the coordinator).** The draft below asked about a budget per test. He chose **one
> budget per heavy main-group process, with no warnings: 20 minutes, and a process over its budget fails the heavy tier**,
> naming the process, its wall and its budget. `small` is the one exception, at 60 minutes, because its wall after the overlap
> fix (§4, the structural finding) is not yet measured. Its review is the 2026-11-28 weekly heavy run: set `small`'s own line
> at its measured wall plus 25%, or split `small` in two and drop the exception.
>
> **In cmd/gate:** the constant `heavyProcessBudget` and the table `heavyBudgetExceptions`, beside `heavyGroupEnv`. Each
> entry carries its budget, reason and review; `heavyBudgets` refuses an entry without them, and the tier then does not
> run. `gradeHeavyBudget` grades each process. The budget is written into the run's heavy input, so a record from before
> it replays unchanged.
>
> **Basis:** the v0.25.0 walls (§1's record, `heavy_processes`): `small` 77.9 minutes, the 26B 31.8, `gemma3-4b` 20.1,
> every other process 15.1 or less. After the moves and shrinks in §9 the largest process other than `small` projects to
> about 15 to 17 minutes, so 20 leaves about 20 to 30%. `small` projects to about 41 to 62 minutes after the overlap fix,
> depending on how much of the parity cell's 21 minutes it sheds. The per-test draft below is kept as the record of the
> question; per-test times (`slow_tests`) were not adopted with it.

**Where it lives.**

- `cmd/gate/event.go:results` keeps each top-level test's final `Elapsed` beside its action. Today it keeps the action
  only, which is why the record has no per-test times.
- `cmd/gate/grade_gpu_heavy.go:heavyProcLines` compares each test's time against the budget. A test over budget becomes a
  notice in the verdict's notices block, as `GATE_NOTICE` lines already do. The same check covers the drain and
  fresh-process cells.
- `cmd/gate/gpu_record.go:heavyProcRecord` gains a `slow_tests` list: the name and seconds of every top-level test that
  took **60 s or more**, independent of the budget. Every test a 2.5-minute triage needs is then in the committed record
  with its exact time (§8, point 5).
- The budget is a constant beside `heavyGroupEnv`. An exception list names any test allowed past it, with the reason.

**Options (recommendation first):**

1. **Warn at 5 minutes per test, never fail.**
   - Pro: it cannot redden a release over cost, and it makes cost visible in every record.
   - Con: a warning can be ignored.
2. **Warn at 5 minutes; fail at 15 minutes unless the test is on the exception list.**
   - Pro: a new 20-minute survey cannot slip in unnoticed.
   - Con: a slow box or thermal state can turn a correct release red over time alone. The exception list needs upkeep.
3. **A budget per process (for example 30 minutes), not per test.**
   - Pro: it fits how the tier is partitioned.
   - Con: it hides which test grew.

After the moves and shrinks that stand (§9), the slowest tests left are KV10 at 11.9 minutes (exact) and KSO1C long
context at about 10 minutes (projected). So option 2's 15-minute line would hold every test that remains without
exceptions.

## 6. Context: the Mac's long tests are in `quick -full`, not this tier

From the v0.25.0 `quick -full` streams. These are exact test2json Elapsed figures:

- **The Mac** (`docs/measurements/release-v0.25.0/quick-full-darwin/decoder-checkpoint-120m.json.gz`): 173 tests, 21.8
  minutes in all. One test is over 2.5 minutes: `TestPrefillChunked_bitIdenticalReal0p5B`, 16.7 minutes.
- **The CUDA box, same cell** (`.../quick-full-linux/decoder-checkpoint-120m.json.gz`):
  `TestPrefillChunked_bitIdenticalReal0p5B` 26.3 minutes, `TestProgram_L_everyParameterIsLive` 23.8, and
  `TestPrefillAttnRowTileInvariance` 2.3.
- **The CUDA box, the decoder cell** (`.../quick-full-linux/decoder-120m.json.gz`): `TestSeams_gates` 28.0,
  `TestProgram_P_plants` 18.3, `TestSeams_plantsGoRed` 11.4, `TestCPUBatch_everyEligibleFixtureBitIdentical` 5.1,
  `TestSeams_subCaptureRefusedWhereGateSCannotRun` 5.0, `TestPlanAgreesWithAdmission` 2.6, and
  `TestDecisionCensus_twoRunsAgree` 2.1.

These are outside H1's scope and are not triaged here. The same method would apply to them.

## 7. Open

- **Replace each G bound with the exact Elapsed** from the v0.25.0 per-test streams on the CUDA box, when its timing lock
  is free. That settles the brief's "883" and "48%", and the size of the `small`/parity overlap.
- Each verdict above is **the maintainer's decision, item by item.**

## 8. Review answers (Gemini, on the PR), item by item

The verdict cells in §2 and §3 already carry the changes below. Where §4 says otherwise, §8 and §9 stand.

**1. "The registered 26B fidelity gates pass vacuously in the heavy tier; the proposal removes them instead of arming
them." Right for #15. Changed: keep it, and arm it.**

- `TestGemma4_26BDecodeLocalize` is the registered G-S19f gate only under `GOINFER_S19_CPUFLOAT=1`. The tier does not set
  that variable, so today it runs about 3 minutes and asserts nothing.
- The tiny `TestGemma4MoE_floatParity` holds the float-decode contract, but not at the 26B's depth, nor at its window
  past step 95.
- **Proposal:** set `GOINFER_S19_CPUFLOAT=1` in `cmd/gate/gpu.go:heavyGroupEnv`. Only this test reads the variable. Its
  runtime armed is not measured.
- It is a gate-definition change, so it needs the maintainer's word with the rest.
- Not changed: #11 and #5 stay moves.
  - **#11, `TestGemma4_26BActQuantSensitivity`,** guards only under `GOINFER_S19_FLOATACT=1`, and then only a float-path
    miss count. That is a coverage counter, not a fidelity bar, and the tiny parity test holds it. Running it means a
    26B CPU load (about 4 minutes) for one counter.
  - **#5, the decode diagnostic,** has no gate under any setting.

**2. "#10 `TestPrefillLongPrompt` skips past batched declines, and moving it loses real-model long-prompt memory-fit
coverage." Right in part. Changed: shrink, not move.**

- The test fails if the 7B Q4_K_M does not load resident at `-ctx 8192` on the 8 GB card. That is a real fit check.
- It does not fail if the batched prefill declines: a decline is printed and passed over.
- **The shrink:**
  - drop the sequential arm (its 14,668 single-token forwards are most of the time);
  - make a batched decline at the longest length (8,012) a `t.Errorf`.
- Projected about 1 minute, saving about 3. The test then guards both the fit and the long batched prefill on a real model.
- **Risk:** if 8,012 legitimately declines at v0.25.0, this turns red. The v0.25.0 stream on the CUDA box says which, so
  read it before the change lands.

**3. "The shrinks and moves for #2, #3 and `TestSpecPagerInteraction` weaken the 64K-context, fp16-overflow and
pager+spec guards." Right for #3 and #2; refuted for `TestSpecPagerInteraction`.**

- **#3 KV10: changed to keep both depths** (3,900 and 65,536) until KV10 closes.
  - At 65,536 it is the only real-checkpoint run in the tier at 64K context. Its NaN/Inf check there is the only
    assertion that depth has.
  - KV10 is still active, and the meaning of its cosine FAILS is the maintainer's open decision.
  - Revisit the depth when KV10 closes. Running only the f16 arm at 65,536 would halve the cost, but it needs a test change
    and gives up the f32 control.
- **#2 P1r: changed from move to shrink, 10 prompts to 2.**
  - The overflow assertion (max|K|,|V| < 32,752, and no NaN) stays on the real Nemotron.
  - With the recorded 730× headroom, two prompts cannot plausibly move the verdict. Projected about 2.6 minutes.
- **`TestSpecPagerInteraction`: refuted, so it stays a move.**
  - Its only result assertion is the off arm's instrument check (distinct experts per stage equals topK). That check is
    folded into `TestPagerDeterminism`, which already loads the same 35B with the pager.
  - Its R1 to R3 readings of pager and speculation together are logged, never asserted. Moving it drops no pager+spec
    guard, because none runs today.
  - What it does lose is a smoke run of `NewBlockSpec` on a paged 35B. Its own record found that on CUDA no model both
    needs the pager and may speculate (`docs/measurements/spec-x-pager-2026-09-02.md`), so that path is not one a request
    reaches.

**4. "Shrinking `TestBlockSpec_twoTurnsMatchPlain` to shared loads risks state leaking across cases." Right. Changed:
keep it unchanged.**

- The fresh load per arm is the isolation the test relies on. A reset between cases is a second contract that this test
  would then have to trust.
- It saves about a minute, which does not pay for that.

**5. "slow_tests recorded only over 5 minutes won't give exact times for the next triage." Right. Changed.**

- The record's `slow_tests` now lists every top-level test of **60 s or more**, independent of the budget line (§5).
- A 2.5-minute triage then needs nothing but the committed record.

**6. "The PR's numbers disagree with the record." Right: there were two disagreements. Fixed.**

- §4 said "eleven" shrinks where the PR said ten. The eleventh was `TestPrefillGateVsReferenceCUDA`'s optional trim, which
  §2 and the keep list mark as keep. The record now counts ten shrinks plus one optional trim.
- `TestBlockSpec_twoTurnsMatchPlain` appeared both as a shrink and in the keep list. It is now a keep (point 4).
- The PR body is rewritten from §9, and both now carry the same table and totals.

## 9. Proposed as approved: the moves and shrinks that stand after §8

- Minutes are G's lower bounds for a move. For a shrink they are a projection by case count.
- The Mac saves 0 on every row, since it has no heavy tier.
- **Bold** rows are ones the review changed.
- **Approved on the PR, not yet carried out.** Each is a separate change, and nothing in this record implements one.

| test | proposal | CUDA min saved |
|---|---|---|
| **`TestP1r_nemotronRangeSurvey`** | **shrink: 2 prompts instead of 10 (was move)** | ≈ 10.4 |
| `TestKSO1C_longContextQuality` | shrink: one document | ≈ 9.4 |
| `TestGemma4BidirSpanPrefill_real26BDecode` | move (measures) | ≥ 7.2 |
| `TestGemma4EModel_realE4BF32Dump` | move (measures) | ≥ 5.3 |
| `TestS10DeepstackPrefillCUDA_real` | shrink: control 12 to 4 (a stricter bar; check S16 first) | ≈ 4.4 |
| `TestGemma4BidirSpanPrefill_real26B` | shrink: one image | ≈ 4.0 |
| `TestGemma4_26BActQuantSensitivity` | move (measures; CPU only) | ≥ 3.8 |
| `TestDefaultVerifyWidth_sweep` | move (measures) | ≥ 3.5 |
| **`TestPrefillLongPrompt`** | **shrink: drop the sequential arm, a decline at 8,012 fails (was move)** | ≈ 3.0 |
| `TestS17LeverA_wholeTower` | move (measures) | ≥ 2.7 |
| `TestKSO1C_towardTheFile` | shrink: 4 prompts | ≈ 2.6 |
| `TestKV11TradeCurve` | move (a timed run that needs the exclusive lock) | ≥ 2.5 |
| `TestKV0_continuationEffect_real1p5B` | move (its tiny twin holds the assertion) | ≥ 2.5 |
| `TestSiglipCUDA_speed` | move (measures) | ≥ 2.1 |
| `TestSpecPagerInteraction` | move; its one check is folded into `TestPagerDeterminism` | ≥ 2.0 |
| `TestWalk_plantsGoRed` | shrink: stop at the first differing fixture | ≈ 1.7 (and again in the parity cell) |
| `TestKV3_batchedPrefillPastTheSpanBound_tiny` | shrink: run in one cell, not two | ≥ 1.7 |
| `TestKSH6_marginParts26B` | move (a reading) | ≥ 1.6 |
| `TestGemma4EModel_realE4BAnchorDump` | move (measures; about 36 GB RSS) | ≥ 1.6 |
| `TestKV2_scheduleIndependence_real1p5B` | move (its tiny twin holds the assertion) | ≥ 1.6 |
| `TestGridVisionCUDA_real` | shrink: 2 images per family | ≈ 1.5 |
| `TestGlmOcrCUDAResidentRealImage` | shrink: the invoice, 16 tokens | ≈ 1.5 |
| | **12 moves, 10 shrinks** | **≈ 77 (CUDA); 0 (Mac)** |

**Turned into keep by the review:**

- **`TestKV10_longContextFidelity`**: both depths, until KV10 closes.
- **`TestGemma4_26BDecodeLocalize`**: kept and armed with `GOINFER_S19_CPUFLOAT=1`, which is a gate-definition change.
- **`TestBlockSpec_twoTurnsMatchPlain`**: unchanged.

**Kept as before:** `TestPrefillGateVsReferenceCUDA` (its K=256 and K=3,900 trim is optional), the E4B and E2B
non-inferiority gates, `TestKSO1C_realPhi3NonInferiority` and `TestGlmOcrCUDAResidentFullBudget`.

**Still the maintainer's, beyond these rows:** the per-test budget (§5), and the fix for the `small`/parity overlap (§4).
