# B23: the CUDA heavy tier, one process per model family (2026-10-15)

Owner decision 2026-10-15, B23 option 1 (`docs/queue-engineering.md`): split the CUDA resident heavy suite so each `go test`
process holds one large model family, log each process's `VmHWM`, and aim for no process above about 20 GB. No test's assertions
change. The partition and its rules are in `docs/code-notes/cmd-gate.md`, "cudaHeavy: one model family per process".

## 1. The result

**Grouping cannot meet the 20 GB goal: single tests exceed it on their own.** Four tests read over 20 GB as the first, or nearly
the first, test in their process, so their own footprint is that size and no partition by test can lower it.

| test | family | VmHWM, alone or nearly | why it counts as alone |
|---|---|---|---|
| `TestGemma4_26B_1bBound` | Gemma 4 26B (loads the 49 GB bf16 `gemma-4-26b-a4b-it` safetensors) | 51,077 MB RSS | first test in its process |
| `TestGptOssResidentParityCUDA` | gpt-oss-20b (loads the MXFP4 GGUF twice, CUDA and a CPU reference) | 44,651 MB VmHWM | the only test in its process |
| `TestPagerDeterminism` | Qwen3.6-35B-A3B int4 `.giw` | 41,805 MB RSS | second, after a vision-tower test that peaked at 3,931 MB |
| `TestGemma4EModel_realE4BAnchorDump` | Gemma 4 E4B | 35,852 MB RSS | second, after a test that peaked at 16,806 MB, so an upper bound |

`TestCohereRealResidentParityCUDA` (aya-expanse-8b and command-r7b, both bf16 safetensors, in one test) read 37,262 MB, fifth in
its process after tests that peaked at 15,925 MB: an upper bound, likely over 20 GB alone, not shown.

**Owner decision 2026-10-15, option (a): accept these four as they are, each under a per-test memory budget**, and merge the
split. A budget is the measured reading plus about 5%, the run-to-run spread of the suite's VmHWM in this box's ABBA (37,229 to
38,361 MB, `docs/measurements/kv2-cuda-2026-10-15.md` §7), rounded up to the next 1,000 MB. Figures are MB as the gate prints them
(kB / 1024). A reading over budget is a regression to explain before it is accepted.

| test | family | budget | basis |
|---|---|---|---|
| `TestGemma4_26B_1bBound` | Gemma 4 26B | 54,000 MB, and see below | 51,077 MB RSS, first test in its process; the process's VmHWM 51,085 MB |
| `TestGptOssResidentParityCUDA` | gpt-oss-20b | 47,000 MB | 44,651 MB VmHWM, the only test in its process |
| `TestPagerDeterminism` | Qwen3.6-35B-A3B | 44,000 MB | 41,805 MB RSS, after a test that peaked at 3,931 MB |
| `TestGemma4EModel_realE4BAnchorDump` | Gemma 4 E4B | 38,000 MB | at most 35,852 MB RSS, after a test that peaked at 16,806 MB (an upper bound) |

**The 26B's reading is a floor, not its demand.** Its process sits at `user-1000.slice`'s `memory.high` of 53,248 MiB, against
which the desktop and the page cache also count. With the desktop up, reclaim swapped 24,064 MB of the process's own pages (peak
VmSwap), so its resident peak was capped by the slice. Expect that process to swap whenever the desktop is logged in. B24
(`docs/queue-engineering.md`) is the follow-up that would lower it.

The four groups not run by day (qwen3-4b, qwen3-vl-2b, gemma3-4b, small) get their readings from the next `gate gpu` run, which
logs every process's VmHWM and peak VmSwap. No separate run is made for them.

## 2. The readings (day, tier 1)

**Instrument.** `go run ./cmd/gate cuda-heavy -group <label>` at `0c472899`, each group alone under `timing_lock.py run`, with
`GOINFER_HEAVY_TESTS=1` and `GOINFER_ISOLATE_DEFER=1` as the gate sets them. The gate samples the `go test` child's `cuda.test`
binary from `/proc/<pid>/status` every 0.5 s (`procPeak`). VmHWM is the binary's high-water mark; "largest RSS in flight" is the
largest VmRSS sampled while a test was the oldest one running. It includes what earlier tests in the process left resident, so it
bounds a test from above and equals it for the first test.

**Context.** nobara-pc, RTX 2070 SUPER 8 GB, 62 GB RAM, `user-1000.slice` `memory.high` 53,248 MiB. The desktop was logged in
(`kwin_wayland` held a 17 MiB CUDA context). System swap in use was about 11.5 GB before and after the runs. Runs 17:05 to 17:57 PDT.

| group | tests (pass / skip) | wall | VmHWM | peak VmSwap | largest RSS in flight |
|---|---|---|---|---|---|
| gemma4-26b | 8 / 2 | 815 s | 51,085 MB | **24,064 MB** | `TestGemma4_26B_1bBound` 51,077; `TestResidentLaunchVRAMProbe` 49,930; `TestGraphsDecode26B` 47,646 |
| qwen3.6-35b | 3 / 1 | 417 s | 45,279 MB | 4,368 MB | `TestQwen36_35B_cache` 45,211; `TestPagerDeterminism` 41,805; `TestGridVisionCUDA_real` 3,931 |
| gpt-oss-20b | 1 / 0 | 58 s | 44,651 MB | 1,276 MB | `TestGptOssResidentParityCUDA` 43,822 |
| 7b-8b | 11 / 10 | 666 s | 37,687 MB | 0 MB | `TestCohereRealResidentParityCUDA` 37,262; `TestFlashDecodeRowsBitIdentical` 22,535; `TestBuildScratchAccounting_plantedDefect` 15,925 |
| gemma4-e4b | 2 / 1 | 345 s | 36,052 MB | 82 MB | `TestGemma4EModel_realE4BAnchorDump` 35,852; `TestGemma4EModel_realE4BNonInferiority` 16,806 |
| gemma4-e2b | 3 / 0 | 171 s | 18,947 MB | 0 MB | `TestGemma4EModel_realE2BPLEHostCost` 18,905; `TestGemma4EModel_realE2BNonInferiority` 12,792 |
| qwen25vl-3b | 7 / 0 | 201 s | 12,109 MB | 0 MB | `TestQwen25VLMRoPEPrefillResidentReal_gate` 11,301 |
| glm-ocr | 8 / 0 | 362 s | 6,396 MB | 0 MB | `TestGlmOcrCUDAResidentRealImage` 6,396 |

The skips are the fresh-process tests the main group defers (`ISOLATED-SKIP`) and tests that need an opt-in variable or an asset
this box does not hold, as in the gate. The 26B process swapped 24 GB and ran `TestGraphsDecode26B` in 357 s against 182 s in the
2026-10-14 release gate, so its later tests' timings are not citable.

**Not measured by day:** qwen3-4b (27 tests, about 12 min of test time in the 2026-10-14 gate's logs), qwen3-vl-2b (7, about
15 min), gemma3-4b (21, about 19 min) and small (392, about 30 min). Each is over the 10-minute day limit, together about 76 min. By the
owner's decision their readings come from the next `gate gpu` run, which records every process's VmHWM and peak VmSwap in its
output and in the record's `heavy_processes`.

Logs: `~/goinfer-logs/b23-2026-10-15/` on nobara (`run-<group>.log`, the gate's `gate_gpu_cuda-heavy-<group>.json` streams, and
`run-day.sh`, the chain that ran them).

## 3. The CUDA test audit: the 15 slowest `./cuda/` tests

From existing readings only (no run). Each time is the test's top-level elapsed in the log named. A = the night queue's
`kv3-cuda-suite-a.log`, B = `kv3-cuda-suite-b.log` (both `~/goinfer-logs/night/runs/2026-10-14/`); G = the release gate's heavy
tier, `~/goinfer-logs/gate-gpu-night-2026-10-14/` (`gate-gpu.log` and its per-process streams, named below). Where a test appears in
more than one, the gate's reading is used. Family is the group the partition puts it in.

| # | test | s | log | family |
|---|---|---|---|---|
| 1 | `TestS10DeepstackPrefillCUDA_real` | 538 | G, `gate_gpu_cuda-heavy-M-Z.json` (B: 514) | qwen3-vl-2b |
| 2 | `TestPrefillGateVsReferenceCUDA` | 307 | G, `heavy-M-Z` (B: 309) | 7b-8b (and the 1.5B) |
| 3 | `TestPrefillLongPrompt` | 248 | G, `iso-TestPrefillLongPrompt` (A: 250) | 7b-8b |
| 4 | `TestDefaultVerifyWidth_sweep` | 246 | G, `heavy-A-L` (A: 246) | qwen3-4b + DFlash drafter |
| 5 | `TestGridVisionCUDA_real` | 240 | G, `heavy-A-L` (A: 229) | vision towers only; in qwen3.6-35b by its label |
| 6 | `TestGemma4EModel_realE4BNonInferiority` | 224 (FAIL) | G, `heavy-A-L` (A: skipped) | gemma4-e4b |
| 7 | `TestS17LeverA_wholeTower` | 198 | G, `heavy-M-Z` (B: 198) | gemma3-4b (and GLM-OCR, Qwen3.5-0.8B towers) |
| 8 | `TestGlmOcrCUDAResidentRealImage` | 184 | G, `heavy-A-L` (A: 176) | glm-ocr |
| 9 | `TestGraphsDecode26B` | 182 | G, `heavy-A-L` (A: 195) | gemma4-26b |
| 10 | `TestSpecPagerInteraction` | 181 | G, `iso-TestSpecPagerInteraction` (A: 170) | qwen3.6-35b |
| 11 | `TestSiglipCUDA_speed` | 181 | G, `heavy-M-Z` (B: 174) | gemma3-4b |
| 12 | `TestGlmOcrCUDAResidentFullBudget` | 169 | G, `heavy-A-L` (A: 167) | glm-ocr |
| 13 | `TestKV0_continuationEffect_real1p5B` | 165 | G, `heavy-A-L` (A: 171) | small (1.5B) |
| 14 | `TestKV3_batchedPrefillPastTheSpanBound_tiny` | 158 | G, `heavy-A-L` (A: 157) | small (tiny fixture) |
| 15 | `TestGemma4EModel_realE2BNonInferiority` | 149 | G, `heavy-A-L` (A: 145) | gemma4-e2b |

The release gate's `heavy-A-L` stream also holds `TestGemma4EModel_realE4BQATNonInferiority` at 210 s; that test was since folded
into `TestGemma4EModel_realE4BNonInferiority` (`025c0f6b`), so it is not listed. Of the KV2 night's `heavy-g1-cuda.log`
(`~/goinfer-logs/kv2-cuda-2026-10-15/night-2026-10-15-heavy/`), only `TestCohereRealResidentParityCUDA` (207 s there, 102 s in the
gate) would reach the list, and its reading came from a run that swapped (`docs/measurements/kv2-cuda-2026-10-15.md` §7).
