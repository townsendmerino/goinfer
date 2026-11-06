# The asset registry from a test two levels down (nobara-sb01, 2026-11-06)

`decoder/assets.go:repoRoot` was the constant `".."`. The GPU backends moved to `backend/<name>/` on 2026-11-03, so from `backend/cuda` and `backend/metal` the registry opened
`../testdata/assets.json`, failed, and every `decoder.AssetPathForTest` call skipped (`asset registry unreadable`). A skipped test prints `ok`; nothing went red. Found 2026-11-06 while reading the MEM11 gate (`../mem11-cuda-2026-11/README.md`).

- **Red:** `red.log`: `TestAssetRegistry_resolvesFromEveryCallerDir` (decoder) chdirs into every directory that has a test calling the registry and loads it; `backend/cuda` and `backend/metal` FAIL, `decoder` passes.
- **Change:** `assetRoot()` walks up from the working directory to the nearest ancestor holding `testdata/assets.json` and returns it as `..`, `../..`; `repoRoot` stays `..` for decoder's own golden paths. `loadAssetRegistry` and the `$REPO/` expansion use it. `green.log`: the four registry tests pass, three caller directories.
- **Parity manifest:** not stale (`TestParityManifest*`, `TestParityKey*` pass); `assets.go` is not a parity input, so there is no refresh. `touched.log`: 12 checks green.
- **`unskipped.txt`:** every test function that asks the registry directly, by package, with its build tags and asset env. Two more reach it through `requireCUDAResidentMRoPE`
  (`TestRopeKVMRoPEBatched_degenerateMatchesScalarKernel`, `TestRopeKVMRoPEBatched_matchesCPUReference`). Most also need `GOINFER_HEAVY_TESTS=1`; without it they still skip, by design.
  Not changed by this PR: what each test then does on its first real run. CUDA ones are read in `reading/` (a later commit); the Metal ones are `backend/metal`'s nine, for the Mac.

## CUDA read, registered 2026-11-06 05:36:26 PST before it runs

`run-cuda-read.sh`: the CUDA tests above, with `GOINFER_HEAVY_TESTS=1`, one `go test` process per model family (mrope, qwen35vl, qwen25vl, glmocr, gptoss, cohere, drafters), the exclusive timing lock for the whole script, nobara's card. **Cap:** the group timeouts sum to 2 h (5+10+10+15+15+25+40 minutes); a group that hits its timeout is reported as a timeout and the read goes on. Expected about 1 to 1.5 h. `TestLoRACostMeasure` (`realckpt`, a measurement, not a gate) is not in the read.
**Rule:** the test's own verdict. A SKIP is counted and read (why). Each red is classified with evidence as a test bug, a stale golden or a product failure, and a product failure goes to the coordinator, not into this PR. Nothing is re-skipped to make a group green. No bar moves.

## CUDA read, 2026-11-06 05:36 to 05:57 PST: 26 of 26 PASS, no skip, no red

`reading-first.log`, `reading-first/<group>.log` (go test -v). The exclusive timing lock, nobara's 2070 SUPER, `GOINFER_HEAVY_TESTS=1`; wall 20 min 39 s (05:36:42 to 05:57:21) against the 2 h cap (the registered expectation was 1 to 1.5 h; the 4B/drafter tests take 13 to 30 s each, and the longest, `TestDefaultVerifyWidth_sweep`, 244 s).

| group | tests (top-level) | result | go test wall |
|---|---|---|---|
| mrope | `TestRopeKVMRoPEBatched_*` (3: the two listed and an interleaved-mode sibling) | 3 PASS | 23.9 s |
| qwen35vl | `TestQwen35VLReal_residentImagePrefillMatchesCPU` (A, B, C) | PASS | 16.0 s |
| qwen25vl | `TestQwen25VLResidentReal_gate`, `TestQwen25VLMRoPEPrefillResidentReal_gate` | 2 PASS | 27.5 s |
| glmocr | `TestGlmOcrCUDAContextFit`, `...ResidentRealImage` (invoice, table, formula), `...ResidentFullBudget` | 3 PASS | 353.2 s |
| gptoss | `TestGptOssResidentParityCUDA` | PASS | 36.9 s |
| cohere | `TestCohereRealResidentParityCUDA` (aya-expanse-8b, command-r7b) | PASS | 98.8 s |
| drafters | 15: block spec, width sweep, resident drafter (fuse, extend, block, streamed-vs-held, scratch), DFlash loop (lossless, gate3), drafter vs off, flash-decode lane, R14 argmax A/B and profile, two-turn | 15 PASS | 671.0 s |

`TestLoRACostMeasure` (`realckpt`, a measurement) was not run, as registered. Zero `--- SKIP` lines and zero `--- FAIL` in any log. So the tests that skipped silently for three days all pass on their first real run: **nothing to classify, no product red, no test bug, no stale golden.** This reads the tests, not the numbers inside them: the sweeps and A/B tests print figures that were not compared to any earlier record here.

The nine `backend/metal` functions are un-skipped too and were not run (sb11m, on the Mac).

## Review fixes and the second CUDA read, registered 2026-11-06 06:23:37 PST before it runs

Review of this PR (no product bug found) asked for four fixes, all in this commit: (1) `AssetPathForTest` and the decoder gates' `assetPath` fail the test (`t.Fatal`) when the registry cannot be read or parsed, and still skip for absent weights (`TestAssetPathForTest_unreadableRegistryFailsAndMissingWeightsSkip`, red in `red-fatal.log`, green in `green-fatal.log`); (2) the four `TestAssetRegistry_*` scan rules in `scripts/touched_scans.py` also trigger on `decoder/assets.go` (and the new one on `testdata/assets.json`); (3) `run-cuda-read.sh` makes `reading/` itself and takes the exclusive timing lock itself (it re-runs under `timing_lock.py` unless a holder's token is in the environment); (4) `assetRoot()` looks in the working directory first, with a repository-root case in `TestAssetRegistry_resolvesFromEveryCallerDir` (red in `red-root.log`, green in `green2.log`). `touched2.log`: 12 checks green.
**The CUDA read is run again** on the final tree through the amended script (the first read, above, ran the earlier `assetRoot`, and the script's lock step is new): same seven groups, same timeouts, same rule, cost about 21 minutes, expected done by about 21 minutes after the start below. The first read stays on record.

### The second CUDA read, 2026-11-06 06:23:37 to 06:43:42 PST: 26 of 26 PASS again, no skip, no red

`reading.log`, `reading/<group>.log`, through the amended script (it took the exclusive lock itself and made `reading/`), on the final tree: mrope 3, qwen35vl 1, qwen25vl 2, glmocr 3, gptoss 1, cohere 1, drafters 15, all PASS, zero `--- SKIP` and zero `--- FAIL`; wall 20 min 5 s. Nothing to classify. The first read is `reading-first.log` and `reading-first/`.

## The Metal read, relayed from the Mac session (mac-sb11m, 2026-11-06; not run or re-checked by nobara-sb01)

Reported: MacBookPro18,3, 16 GB, an archive of this PR's head `35ba1b35`, one test per invocation, exclusive timing lock, on AC, `~/models` only; the files are `~/goinfer-logs/pr709-metal-2026-11-06/` on the Mac (`README.md` is the table, one log per test). Their account:

| test | result |
|---|---|
| `TestKSO1_referenceMatchesAikitPhi3` | PASS (within 1e-6 of aikit's dequantizer on 32 Q5_K and 17 Q6_K tensors) |
| `TestKSO1_laneBytesPhi3` | PASS |
| `TestQ4KLane_realPhi3NonInferiority` | PASS (+4.39 points) |
| `TestKSC1_phi3Speed` | PASS (0.530) |
| `TestKSO1a_phi3Speed` | PASS (0.809, bit-identical) |
| `TestKSO1_speed` | **FAIL as registered**: O1-G7 KILL (0.9806 and 1.0145), the KILL the owner overrode on 2026-10-24 |
| `TestOneSlotStep_phi3PrefillAB` | PASS (it only reports): median 0.213, the step route 4.7x slower than sequential, sent to the coordinator as a product reading |
| `TestGptOssResidentParityReal20B`, `TestGptOssResidentMemGuardDeclines` | not run: the checkpoint is not on that Mac |

So the Metal side of this fix un-skips no correctness failure; the one FAIL is a registered speed gate whose KILL was already overridden, and the 0.213 is a reading about a product route, not a harness result. Four of the nine need a second switch beyond the asset (`unskipped.txt`).
