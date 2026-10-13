# `gate quick`: which cuda tests are slow, and which of them may be deferred (2026-10-13)

Owner request: measure which tests of the `cuda` module take 4 s or more in a `gate quick` run's `test_cuda.json`, add the
untagged kernel sweeps among them to `quickModule.Slow` (`cmd/gate/quick_select.go`), and re-check the `gpu` list on
Vulkan (it was measured on the Mac).

## Instrument

The 19 `test_cuda.json` files `gate quick` left on nobara-pc between 2026-10-05 and 2026-10-13 (the cuda cell, `go test -json
-short -tags 'cuda goinfer_testhooks' ./cuda/`, a different tree each time). A test's time is the `Elapsed` of its `pass`
event; a top-level test only (subtests are inside it).

## Read

- 495 tests recorded, 43 (subtests included) with a passing run of 4 s or more. The slowest top-level tests, by median over
  the runs that reached them: `TestAttnVit_logicAndBMIdentity` 22.4 s, `TestForcedSMShapes_decodeIsIdentical` 20.5 s,
  `TestForcedFreeVRAM_residentDecisions` 21.2 s, `TestGemma4MoEScaled_residentParity` 17.2 s, `TestWindowedKV_*` 5 to 13 s each.
- **Every one of those carries `goinfer_testhooks`**, so `gate gpu`'s `cuda-suite` (tags `cuda` only) does not run it: they have
  no second instrument and are not deferred. This is the Metal rule (`Slow` names kernel sweeps that load no model through the
  decoder and are untagged, so `gate gpu` runs them).
- Of the 40 test files that carry `cuda` and not `goinfer_testhooks` (85 top-level tests), the ones that reached 4 s:

  | test | latest run | median over runs | max | runs |
  |---|---|---|---|---|
  | `TestTowerBase_attention` | 8.2 s | 8.6 s | 12.3 s | 13 |
  | `TestGemmMMA_vsExact` | 3.0 s | 3.4 s | 4.6 s | 19 |

  **Added to `Slow`: `TestTowerBase_attention` only.** `TestGemmMMA_vsExact` is under 4 s in the latest run and at the median,
  so it stays in the quick cell.
- What the deferral buys on a decoder-only diff: about 8 s of a cuda cell that costs minutes. Small, and it is all the rule
  allows: the large tests are tagged.

## The gpu list on Vulkan

`go test -tags 'gpu goinfer_testhooks' -short -json ./gpu/...` on nobara-pc (RTX 2070 SUPER, Vulkan), GPU otherwise idle, one run:
the gpu package passed in **234 s** (257 top-level tests, 95 skipped).

| test | Vulkan | in `Slow`? |
|---|---|---|
| `TestContextClose_finalizerRace` | **50.5 s** | no, until now |
| `TestContextClose_lateWrapperClose` | **48.5 s** | no, until now |
| `TestDecodeRunnerW4A8_geometries` | 12.3 s | yes (Mac-measured) |
| `TestGumbelDeviceAgreesWithHost` | 9.3 s | yes (Mac-measured) |
| `TestGemma4MoE_webgpuPlantedDefects`, `TestPrefillInt4_matchesWalk`, `TestEmbedInt4_residentDecisionDoesNotChange` | 7 to 10 s | no: tagged `goinfer_testhooks` |

- **The Mac-measured list holds on Vulkan.** Both of its tests are over 4 s here.
- **Two tests it lacked are the largest in the package:** 150 rounds of creating a WebGPU device each, about 100 s of the 234 s. They
  load no model and are built with the `gpu` tag alone, so `gate gpu`'s `webgpu-suite` runs them; they are added to `Slow`.
- The remaining tests of 4 s or more carry `goinfer_testhooks` (the same rule as the cuda list: no second instrument, not deferred).
- One run on one box. It says which tests are long here, not by how much each varies.
