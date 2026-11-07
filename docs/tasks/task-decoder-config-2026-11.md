# Task: take Config, Options and Architecture out of decoder? A split proposal (DC1–DC4) — 2026-11

> **Status 2026-11-07: DC1 done (DC1a #714, DC1b 2026-11-07); DC2/DC3 parked (Francis, 2026-11-06).** Written by mac-fit from a count of the tree at `2fd3a644`. Recommendation: **extract `Config` only (DC1a, DC1b); leave `Options` and `Architecture` where they are (DC2, DC3 parked).** Extraction is not needed for `decoder/fit`, which is finished (`docs/layout.md` "Tier 3").

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| DC0 | done | Francis 2026-11-06: DC1 yes; DC2/DC3 parked | this proposal |
| DC1a | done | `Config`'s cross-file methods exported in place (`headDim`→`EffectiveHeadDim`, `validateX`→`ValidateX`, …), merged in #714 (2026-11-06) | §3; #714 |
| DC1b | done | 2026-11-07 (mbp2-vk5): `decoder/config` holds `Config`, all its methods (`ValidateGemma12`, `ValidateGlmOcr` moved in), `Load`, `ResolveEOSIDs`, `LenientJSON` and `RecommendedSampling`; `decoder.Config` and `decoder.RecommendedSampling` are aliases, `loadConfig`/`resolveEOSIDs` one-line wrappers; layout rule `config-leaf` (red under a plant); `paritykey.BuilderFile` learns `decoder/config/config.go`; program fingerprints and the decision census unchanged; one parity refresh, 12 `deps_hash` lines (the families with no programs; the 31 with programs stay fresh under the builder switch) | §3; §8 |
| DC2 | blocked | `Options` only when a subsystem needs it (it drags `Backend`, `Drafter`, `Knobs`) | §4 |
| DC3 | blocked | no: `Architecture` is the hub (§5); continue `internal/runtime/program` instead | §5 |
| DC4 | done | the aliases in `decoder` are the published API and stay (T3-fit question a); kept, Francis 2026-11-06 | §5 |

## 1. What is there (files naming each type, `2fd3a644`; decoder has 206 non-test and 606 test files)

| Type | Defined | Size | Fields (unexported) | Methods | decoder src / test | outside src / test |
|---|---|---|---|---|---|---|
| `Config` | `config.go` | 329-line struct, 1,462-line file | 130 (0) | 40, 36 of them unexported | 61 / 58 | 4 / 3 |
| `Options` | `model.go` | 178-line struct | 38 (2) | 2 (`Validate`, `withAutoBackend`) | 38 / 355 | 35 / 581 |
| `Architecture` | `arch.go` | 164-line struct | 61 (21) | 57 across 9 files | 92 / 62 | 0 / 0 |

What each drags: `Config` imports only the standard library (`config.go`'s import block) and its fields are plain data plus `RecommendedSampling`. `Options` names `Backend`, `Drafter` and `Knobs`. `Architecture` names 18 types, 12 of them per-family parameter structs (`gemma4Params`, `qwen35Params`, …), holds `func(i int) bool` fields, and is built by `registry.go` and 40 `arch_*.go` files (3,040 lines) from a `Config`.

## 2. What each blocks today

- **The fit adapter** (`fitCheckFor`, `kvPricingFor`, `estimateKVBytes`, `Model.Plan`, `FitQuery`) needs `resolveArchitecture` (the registry), `ResolveKV` and `buildDemandFor` (`Architecture`, `Options`, the backends' demand reports). Extracting `Config` alone does not move it; extracting all three still leaves the registry and the demand reports behind. So the fit adapter does not justify this work, and it stays in `decoder` as a thin adapter.
- **Config loading and validation** (`loadConfig`, `ggufConfig`, `resolveEOSIDs`, 36 per-family validators and normalizers) needs nothing else: a `decoder/config` package would test in seconds without compiling `decoder`. That is the one subsystem this survey finds that `Config` alone unblocks. Its tests today: `config_bounds_test.go`, `config_fuzz_test.go`, `config_nonfinite_test.go`, `config_snapshot_test.go`, `laguna_config_test.go`, `mellum21_config_test.go` and three more.
- **Other Tier 3 subsystems** (KV, weights, the program builders) take `Model` or `Architecture` as well; `Config` alone unblocks none of them. This survey did not look at them one by one.

## 3. Proposed steps (each builds and passes alone)

- **DC1a, in place (~40 files, mechanical, ±300 lines).** Export the methods that cross files: `headDim` (23 files), `rotaryDim` (16), `gemma4GlobalGeometry`/`gemma4PartialRotary`/`gemma4RopeBases` (1), `lagunaFirstKDense`/`lagunaGatePerHead`, the three `normalize*LayerTypes`/`Blocks`, and the `validate*` set called from 34 files (the `arch_*.go`, `registry.go`, `weights.go`, `lora.go`). Same package, so nothing else changes. Gates: whole `./decoder` run (22 min), census hash, parity refresh below.
- **DC1b, the move (~1,500 lines moved, ~9 test files follow, one new package).** `config.go`, the `Config` methods in `gemma12.go` and `arch_glm_ocr.go`, `loadConfig`/`resolveEOSIDs`; `ggufConfig` in `gguf.go` stays and calls `config.Config` (it reads a `GGUFFile`). `decoder/config.go` becomes `type Config = config.Config` (+ the loader wrappers) so `decoder.Config`'s 4 + 3 outside users and every `arch_*.go` compile unchanged. An `internal/layout` rule: `decoder/config` imports no goinfer package.
- **DC2 (parked).** `Options` would move with `Backend`, `Drafter`, `Knobs` (and `withAutoBackend`'s callers in `fit`), with 581 outside test files naming `decoder.Options`: alias mandatory, no consumer today.
- **DC3 (parked, recommend no).** `Architecture` has 21 unexported fields, 57 methods and 154 files naming it; moving it exports or wrappers a large surface and moves the 40 `arch_*.go` the parity key treats as family-owned. The way out of `decoder` for that hub is already under way as data: `internal/runtime/program` (`ActKind`, `MoEConfig` are aliases of it) and `task-forward-architecture-2026-10.md`.

## 4. Order and sizes

DC1a (a day, ~40 files) → DC1b (a day, one package, ~25 files touched with tests and layout data). DC2 and DC3 start only on a named consumer.

## 5. What stays as aliases (DC4)

`decoder.Config`, `decoder.Options`, `decoder.Architecture` and everything in `decoder/fitalias.go` stay as exported names; they are the published API, and `Model.Plan` returns `fit.Plan` under its old name. A `decoder.Config` alias also keeps `config.Config`'s 130 fields addressable as before.

## 6. Risk: parity hashes, fingerprints, census

- **Parity keys.** `paritykey.BuilderFile` and 42 `testdata/parity/*.json` name `decoder/config.go`, `decoder/arch.go`, `decoder/arch_*.go` and their content hashes (`deps_hash`). DC1a edits ~40 of those files, DC1b removes `config.go` from the list. Each re-stales the families it reaches. The sound route is `scripts/refresh_parity_hashes.sh`, which refuses unless the forward goldens that run pass; it is proof only for the families whose goldens run here, the rest wait for the Linux box. VK5's week-3 shadow key is live (SB4): changing the key's file list mid-week needs the coordinator's word.
- **Program fingerprints** (`testdata/program_fingerprints.json`) cover what the builders build, not their source, so a move leaves them equal; that is the check that DC1 changed no numerics.
- **Decision census:** `TestDecisionCensus` hash `706b6759…` on this Mac (the golden compare skips here; the hash equality is the evidence, as in #697). Config is not an input of the census's arithmetic, so it must not move.
- **Layout data:** `internal/layout/imports.json` gains the arrows and a `config-leaf` rule; `decoderRatchet` counts are unaffected (the aliases keep `decoder` imported).

## 7. Decision (Francis, 2026-11-06)

DC1 goes ahead: `Config` only. DC2 (`Options`) and DC3 (`Architecture`) stay parked until a subsystem names a consumer. DC1a is its own PR from `origin/main`, kept to the method renames so it merges around the open decoder PRs; DC1b follows. Each PR: a red-first test where behavior could change (DC1b: a layout rule that `decoder/config` imports no goinfer package), the program fingerprints and the census hash unchanged as the no-numerics proof, one parity refresh at the end, `touched.py` and `cmd/lint`, no auto-merge. Before DC1b changes `paritykey.BuilderFile`, check whether VK5's week-3 shadow key is still running and tell the coordinator, who decides whether DC1b waits for the week to end.

## 8. DC1b, done 2026-11-07

- **The move.** `decoder/config/config.go` (from `decoder/config.go`, with `ValidateGemma12` from `gemma12.go` and `ValidateGlmOcr` from
  `arch_glm_ocr.go`), `nonfinite.go` (from `decoder/config_nonfinite.go`) and `sampling.go` (`RecommendedSampling`, `Applied`, `Normalized` and the
  generation_config reader, from `decoder/recommended_sampling.go`; `ggufSampling` stays in `decoder`, since it reads a `GGUFFile`). `loadConfig` is
  `config.Load`, `resolveEOSIDs` `config.ResolveEOSIDs`. Two config-only test files moved with it (`nonfinite_test.go`, `sampling_test.go`); the rest
  stay in `decoder` and call the wrappers.
- **Red first.** `internal/layout`'s `config-leaf` rule fails `TestLayout_arrows` on a plant that imports `internal/runtime/program` from
  `decoder/config`, and passes without it.
- **No numerics moved.** `TestProgramFingerprint_table` passes unchanged, and `TestDecisionCensus`'s golden compare passes on amd64 (29,637 groups).
- **Parity.** `paritykey.BuilderFile` and decoder's `isBuilder` list `decoder/config/config.go`, which joins `core`. Under the builder switch
  (`docs/measurements/vk5-2026-10.md` §13) only the 12 families with no programs re-stale; one goldens-gated refresh moved their 12 `deps_hash` lines.
- **Comments.** Every comment naming a `Config` method by its old unexported name now names the exported one.
- **Evidence.** Logs in `docs/measurements/dc1b-2026-11-07/`. `touched.py`'s two reds there are not this change's: `cmd/gate`'s
  `TestProcSampler_findsTheTestBinaryUnderItsParent` and `backend/webgpu`'s `TestUploadKV_gemma4MoE/kv-f32/past-the-window` fail the same way on
  `origin/main` on mbp2 (its webgpu device is an Intel iGPU).
- **VK5.** The week-3 shadow week is over (graded early, 2026-11-07); this change went after the builder switch, as §7 asked.

