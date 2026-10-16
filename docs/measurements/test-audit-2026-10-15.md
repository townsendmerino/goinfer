# Test audit, 2026-10-15: where the test time goes, and the cheap fixes

<!-- citations-at: 69e6160c -->

Owner's brief, 2026-10-15:

- measure without running anything heavy;
- class each slow test as shrink, share, parallel, duplicate or leave;
- fix the shrink, share and parallel cases with every assertion kept, each shown to go red on a planted break;
- list the duplicates and the release-night checks for a decision.

Branch `test-audit-2026-10`, from main `69e6160c`. The prior art is `docs/completed/task-ci-speed-2026-09.md` (C0–C9): the decoder
sharded across four jobs, the sampler sweeps strided under `-race`, the Go test cache. This audit builds on it and does
not redo it.

## Sources

- **CI, per test:** `scripts/ci_test_summary.py`'s "SLOWEST 40 TOP-LEVEL TESTS" in every test job's log.
  - Main's last green code run: `38812659448` (`5bb78148`).
  - PR #29's green run: `38824958928`, the current tree with G40 and KV2.
  - A test from a package marked `(cached)` shows the time recorded when its result was cached, not time paid in that
    run. It is marked "replay" below.
- **CI, per job and step:** the Actions API. The slow KV2 run is `38810497681`.
- **The Mac, per test, without `-race`:**
  - `go test -json` of the packages below, by day, under the timing lock;
  - the `gpu` module with CI's own command, `go test -short -tags 'gpu goinfer_testhooks' ./gpu/...`.
  - No model was loaded; the logs are in `~/goinfer-logs/test-audit-2026-10-15/`.
- **Heavy and night-only tests:** the night runner's logs and `SUMMARY.md` files, 2026-10-08 to 2026-10-15.

## CI: what bounds the wall time

A code push's CI took 7.4 to 9.2 minutes in the recent runs. The jobs run in parallel; the slowest set the wall.

| job | wall | bounded by |
|---|---|---|
| `gpu-darwin` | 8.2–8.4 min (test step 6.6) | the `gpu` package alone: 392 s on CI, against 100 s for the same command on an M1 Pro |
| `test-rest` (`-race`) | 6.9–7.2 min | `internal/whisper` 356 s, `multimodal` 259 s, `internal/serveapp` 228 s; `internal/kvmgr` when it is not cached (below) |
| `root-windows` | 7.0–7.3 min | `internal/sampling` 278 s, `cmd/gate` 218 s |
| `root-darwin` | 5.5–6.0 min | `internal/sampling` 237 s |
| `root-windows-arm64` | 5.6–6.2 min | as `root-windows` |

- **What makes `gpu-darwin` slow:** the GitHub macOS runner's GPU.
  - The same 264 tests (98 skip under `-short`) take 392 s there and 100 s on the Mac; every dispatch and readback costs
    more on the runner's virtualised Metal device.
  - The largest single test is `TestGumbelDeviceAgreesWithHost` (23.7 s on the Mac), then
    `TestDecodeRunnerW4A8_geometries` (9.8 s).
- **The 22.4-minute `gpu-darwin` on KV2's run `38810497681` was the checkout, not the tests:**
  - `actions/checkout` took 14.0 minutes, and its test step 6.9, the usual.
  - C0 measured no checkout tail across 28 runs in September, so this is a new, single observation. It is worth watching,
    not acting on.
- **The 16.6-minute `test-rest` on the same run was `internal/kvmgr` under `-race`:**
  - `TestOracle_pool` takes about 20 minutes under the race detector (1,233.6 s, the replayed figure), against 21 s
    without it.
  - It is paid whenever `internal/kvmgr` changes, which KV2's pushes did, and is otherwise cached.

## The 20 slowest tests

CI seconds, the slowest reading across the two runs. `-race` marks a job run under the race detector. "Local" is the Mac
without `-race`.

| # | test | module / package | CI s | job | what it checks | class | done |
|---|---|---|---|---|---|---|---|
| 1 | `TestOracle_pool` | root / `internal/kvmgr` | 1,233.6 (replay; ~20 min when paid) | test-rest, `-race` | the KV manager against an oracle: 8 seeds × 20,000 random operations, a full check after each | parallel | yes |
| 2 | `TestPolicy_plantedDefectsAreRed` | `internal/whisper` | 187.9 | test-rest, `-race` | each planted long-form policy defect changes some result against transformers | shrink (stop at the first difference) | yes |
| 3 | `TestSampleFromTopK_matchesFullPath` | `internal/sampling` | 151.4 | decoder-a, `-race` (strided) | the device top-K sampler draws what the full-row sampler draws, every cell | parallel | yes |
| 4 | `TestWebUI_appGateInBrowser` | `internal/serveapp` | 117.9 | test-rest | the shipped web UI in Chrome: streaming, Copy on both clipboard paths, re-render | leave: a real browser is the point | — |
| 5 | `TestQwenPreprocess_inputPixelLimit_M15` | `multimodal` | 94.2 | test-rest, `-race` | an image over the input-pixel limit is refused before its pixels are decoded | shrink | yes, and an assertion added (below) |
| 6 | `TestTopFilterLogits_MatchesReference` | `internal/sampling` | 80.3 | decoder-c, `-race` (strided) | the bounded selection equals the reference bit for bit over a seed sweep | already parallel and strided (C8) | — |
| 7 | `TestCapGlmOcrPixels_throughPreprocess` | `multimodal` | 74.1 | test-rest, `-race` | GLM-OCR's pixel cap reaches the patch grid through the real preprocess | shrink | yes |
| 8 | `TestPolicy_tinyMatchesTransformers` | `internal/whisper` | 68.8 | test-rest, `-race` | every long-form configuration equals transformers' | leave: every configuration is the assertion; see duplicates | — |
| 9 | `TestTimestamps_plantedDefectsAreRed` | `internal/whisper` | 58.6 | test-rest, `-race` | each planted timestamp defect changes some case | shrink (stop at the first red case) | yes |
| 10 | `TestPrefillChunked_bitIdentical` | `decoder` | 47.0–78.6 | decoder-tail, `-race` | G40-1: a chunked CPU prefill is bit-identical to one pass | leave: a registered gate's sizes (G40-1); shrinking it is an amendment | — |
| 11 | `TestSampleFromTopK_deviceRoundedZ` | `internal/sampling` | 40.8 | decoder-c, `-race` | a perturbed Z gives the same tokens | leave: one sampler's draw sequence | — |
| 12 | `TestHelp_namesTheModelCache` | `internal/serveapp` | 36.4 | test-rest | the real `--help` of both binaries names the cache directory | leave: building and running the binaries is the point | — |
| 13 | `TestSample_DrawIdentity` | `internal/sampling` | 33.3 (replay) | decoder-b, `-race` | the optimised draw equals the reference for a fixed seed | leave for now: small, sharded on its own job | — |
| 14 | `TestVoxtral_tinyPlantedDefectsAreRed` | `decoder` | 32.6–59.0 | decoder-tail, `-race` | each planted Voxtral audio-path defect is red on some criterion | leave: one loaded model shared by the defects, and each defect's margins are recorded by design | — |
| 15 | `TestWhisperFeatures_plantedDefectsAreRed` | `multimodal` | 29.4 | test-rest, `-race` | each planted log-mel defect clears the bar, its margin logged | parallel | yes |
| 16 | `TestVoxtralPrompt_wiring` | `internal/serveapp` | 28.8 | test-rest, `-race` | serve's Voxtral prompt and features, bit for bit, and a generation through it | leave: one composition | — |
| 17 | `TestQuick_neverDropsAGoListDependent` | `cmd/gate` | 25.1 | test-rest | `gate quick`'s selection against an independent per-binary `go list` | leave: the independent listing is the point | — |
| 18 | `TestSessionLRU_sharesAPreamble` | `internal/serveapp` | 23.8 | test-rest, `-race` | KV2's CPU copy through serve's session LRU | leave: through the caller by design | — |
| 19 | `TestGemma4Preprocess_matchesTransformers` | `multimodal` | 20.6 | test-rest, `-race` | Gemma 4's preprocess equals transformers' pixels bit for bit on four images | leave: the four images are the evidence | — |
| 20 | `TestBackendTagGuardFailsBuild` | `cmd/serve` | 18.7 (replay) | test-rest | a build without a backend tag fails | leave: the build is the point | — |

**Also slow, outside the CI top 20:**

- `gpu` (`gpu-darwin`, from the local run):
  - `TestGumbelDeviceAgreesWithHost`, 23.7 s on the Mac: **parallel, done** (the host references);
  - `TestDecodeRunnerW4A8_geometries`, 9.8 s: leave, one geometry per kernel shape.
- `internal/sampling`:
  - `TestGumbelDraw_goodnessOfFit` (14.6 s local): **parallel, done**;
  - `TestGumbelDraw_equivalentToLegacyDistribution` (17.0 s): **parallel, done**;
  - `TestGumbelDraw_largeVocabTailAndHead` (12.6 s): leave, one draw stream;
  - `TestSamplingThroughputGate` (15.8 s): leave, a timing is the point.

## The fixes

Every assertion is kept. Draw counts, statistics and margins are compared with the run before the change, and each
sped-up test was shown red on a planted break, then restored. Times are the Mac's, without `-race`; under `-race` they
are noted where measured.

| test | change | before | after | the same work? | planted break → red |
|---|---|---|---|---|---|
| `TestOracle_pool` | the 8 seeds as parallel subtests (each its own manager and oracle; the package has no goroutines or locks) | package 21.3 s | 7.5 s; 200 s under `-race` locally, against CI's 1,233.6 | same walks and checks; no data race under `-race` | `unref` that skips the last release: "block 0 has 1 references, 0 sequences hold it" at seed 1, operation 12 |
| `TestPolicy_plantedDefectsAreRed` | stop at the first differing configuration, in a fixed order | 11.8 s | 2.0 s | the assertion is "some configuration differs" | the "conditioning uncut" defect disabled: red, "the gate cannot see it" |
| `TestTimestamps_plantedDefectsAreRed` | stop at the first red case | 4.6 s | 1.6 s | the assertion is "some case differs" | the pair-rule defect disabled: red |
| `TestSampleFromTopK_matchesFullPath` | the 24 (vocab, shape) cells as parallel subtests | 42.5 s | 14.1 s | 102,336 draws, 88,538 served, both before and after | a device token off by one on 1 id in 97: red, 1,006 of 102,336 differ |
| `TestGumbelDraw_goodnessOfFit` | the three temperatures as parallel subtests, logits drawn first in the old order | 14.6 s | about 5 s | the same z at every temperature (+0.63, −0.71, −0.33) | the draw at 1.15× the temperature: red at all three |
| `TestGumbelDraw_equivalentToLegacyDistribution` | the legacy counts drawn beside the new ones | 17.0 s | 14.3 s | the same z (+0.57, df 198) | the same plant: red, z = +834 |
| `TestGumbelDeviceAgreesWithHost` | each group's device draws first, then the host references across the cores, compared in order | 23.7 s | 18.3 s | 12,000 draws, 0 mismatches, both | the kernel's 1/T at 1.002×: red, mismatches that are not near-ties |
| `TestQwenPreprocess_inputPixelLimit_M15` | the over-limit image 6000×6000 (36 MP) instead of 12000×12000 (144 MP); the fixture asserted over the limit | 1.7 s; 61.8 s under `-race` | 0.4 s; 15.7 s under `-race` | the same refusal | below |
| `TestCapGlmOcrPixels_throughPreprocess` | the page 2400×3200 (7.7 MP) instead of 3000×4000 (12 MP), still over the 4.82 MP ceiling | 1.7 s; 27.3 s under `-race` | 1.2 s; 18.6 s under `-race` | the same three grid assertions | the cap ignored: red, "a 2 MP cap gave 4,727,520 px" |
| `TestWhisperFeatures_plantedDefectsAreRed` | each defect as a parallel subtest (aikit's defect hook takes the defect as a parameter) | 2.2 s | about 1 s | the same six margins, to the digit | defect 3 computed as no defect: red, "stayed under the bar" |

**Package totals, the Mac, without `-race`:**

| package | before | after |
|---|---|---|
| `internal/sampling` | 126.9 s | 86.3 s |
| `internal/whisper` | 21.8 s | 9.0 s |
| `multimodal` | 8.7 s | 7.0 s |
| `internal/kvmgr` | 21.3 s | 7.5 s |
| `gpu`, `-short` | 99.8 s | 83.6 s |

**A blind spot found and closed: M15 could not see its own guard.**

- The input-pixel check runs twice in `multimodal/qwen_preprocess.go:QwenPreprocess`: on the header, before decode, and
  again after decode, with the same message.
- With the header check removed, the test stayed green, at 144 MP as well as 36: the post-decode check refused the image
  after allocating its canvas, which is the cost M-15 exists to prevent.
- The test now also asserts that the refusal allocates under 16 MiB. With the header check removed it is red:
  144,118,760 bytes allocated.

## Duplicates, for the owner's decision (none deleted)

1. **Pure-computation packages run twice: under `-race` in `test-rest`, and without it in every `root-*` job.**
   - `internal/whisper` (356 s under `-race`), `multimodal` (259 s) and `internal/kvmgr` (about 20 minutes when not
     cached) start no goroutines in the tests named above, so the race detector has nothing to find in them.
   - The tests run in full, without `-race`, in `root-darwin`, `root-windows`, `root-windows-arm64` and
     `root-linux-arm64`.
   - The precedent is C8, which strides the sampler sweeps under `-race`. `TestOracle_pool` already reads
     `KVMGR_ORACLE_OPS`.
   - **Options:** stride these under `-race` as C8 does, or leave them. This is the largest remaining lever on
     `test-rest`.
   - **The new parallel subtests are the exception:** they do share a process. One `-race` run of each (done here, no
     race) is the evidence they are safe.
2. **`TestPolicy_tinyMatchesTransformers` and `TestPolicy_plantedDefectsAreRed`** both run the policy comparison. The
   planted test now stops at the first difference, so the overlap is one configuration per defect. Kept.

## Release-night checks (left alone)

| check | where | time | source |
|---|---|---|---|
| `go run ./cmd/gate quick -full -base <previous tag>` | both boxes | about 10 min on the Mac: the Metal suite 4 min 12 s, the real-checkpoint cells 3 min 44 s | CLAUDE.md's gate tiers, measured 2026-10-13; the v0.23.0 records in `docs/measurements/quick-full/` |
| `go run ./cmd/gate gpu` (the Metal device gate) | the Mac | 3 min actual, queued at 25 | night `SUMMARY.md` 2026-10-13 and 2026-10-14 (`op1-gate-gpu-metal`, `kv1-gate-gpu-metal`, `kv2-gate-gpu-metal`) |
| `go run ./cmd/gate parity` (the real T3 sweep) | nobara | hours | RELEASING.md; the record `docs/measurements/parity-sweep/parity-20261015T093055Z-695ce16c.json` |
| the peer matrix | both boxes | night | RELEASING.md |
| the two env-gated correctness gates (`docs/measurements/b13-dark-gates-2026-10-06/run-dark-gates.sh`) | the Mac | 62 s warm, about 6 min cold | CLAUDE.md's gate tiers |

**Night-only tests, the slowest on record** (measurements and speed gates, where the time is the point):

| test | time |
|---|---|
| `TestGridTowersWebGPU_patchSweep` | 413.5 s |
| `TestQ4KLane_realPhi3NonInferiority` | 306.3 s |
| `TestGridVisionWebGPU_speedGate` | 292.1 s |
| `TestQwen25VisionWebGPU_speedGate` | 264.4 s |
| `TestMC3Step_w8BitIdentical` | 255.9 s |
| `TestVisionEncoderF32_speedGate` | 211.3 s |
| `TestHLChainAB` | 189.0 s |

**The night estimates run high:** the Metal device gate is queued at 25 minutes and takes 3. Tighter estimates would
fit more into a night; that is a queueing note, not a test change.

## The `cuda` module (nobara's; not changed)

CI only builds and vets it (the `cuda` job, 1.2–1.5 minutes): its tests need the GPU and run on nobara, whose timings
are not on the Mac. Nothing in it was changed.

## Next, not done

- **Batching `TestGumbelDeviceAgreesWithHost`'s device draws** into fewer submits would cut `gpu-darwin` further. It
  touches the WebGPU Gumbel state's API, so it is not a test-only change.
- **The `-race` duplicates above:** the owner's decision.
