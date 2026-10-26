# AX1: aikit's towers and GPU backends copied into goinfer (2026-10-26)

Track AX of `docs/tasks/task-subsystems-2026-10.md`, step AX1; the gates are registered in that doc's §7.6. Session
mac-ax01, on the MacBook (arm64, Metal), Go 1.27.3. Everything here is tier 0: no model, no timed run.

## What was copied, and from where

| aikit path | tag (commit) | goinfer path |
|---|---|---|
| `vision` | `v1.75.0` (`38dc8e2b`) | `internal/vision` |
| `audio` | `v1.75.0` | `internal/audio` |
| `gpu` (the device module) | `gpu/v0.35.0` (`0fd8325c`) | split by backend, package `gpudev`: `cuda/internal/gpudev` (the Linux files, the `.cu` sources, PTX and its tooling) and `metal/internal/gpudev` (the darwin files); untagged and `darwin \|\| linux` files in both |
| `gpu/qwencuda`, `gpu/visioncuda` | `gpu/v0.35.0` | `cuda/internal/qwencuda`, `cuda/internal/visioncuda` |
| `gpu/qwenmetal`, `gpu/visionmetal` | `gpu/v0.35.0` | `metal/internal/qwenmetal`, `metal/internal/visionmetal` |
| `gpu/webgpu` | — | retired, not copied (owner, 2026-10-26): nothing imports it, and the `gpu` module has its own WGSL |

Both are the newest aikit tags on 2026-10-26, and aikit main (`3e7c479`) matches them on every copied path (only the
four sub-module `go.mod` pins differ). aikit's README freezes these paths at the same tags (aikit PR #21).

`ax1_copy.py copy` made the copy from `git archive` extractions of the two tags; its edits are import paths, the device
package's clause (`gpu` to `gpudev`, importers aliasing it `gpu`), fixture paths, and the Metal half's cross-backend doc test
reading the CUDA half's files by relative path; it then runs gofmt (a rewritten import can re-sort). The vision tests' aikit fixtures
moved byte for byte to `internal/vision/testdata`, with goinfer's `gemma4-vision-tiny` (aikit does not track it) and the
aikit oracle scripts that pinned the goldens.

## Gates

| gate | result |
|---|---|
| AX1-S (source identity) | PASS: `ax1_copy.py check` recomputes 179 files from the tag snapshots, 0 differ, no extra files; two copied tests edited after the copy are listed, not compared (below) |
| AX1-N (adds only) | PASS: no existing `.go` file changes. Modified: `.gitignore` (the fixture exception), `NOTICE`, `THIRD_PARTY_LICENSES.md`, `cuda/go.mod` and `metal/go.mod` (purego from indirect to direct), `scripts/touched_scans.py` (the moved source-reading tests), and this track's task doc. The root `go.mod` and `go.sum` are unchanged |
| AX1-B (builds) | PASS: ci.yml's cleanliness guard, run from the root as CI runs it (no purego, gocudrv or aikit/gpu in `go list -deps ./...` for linux or darwin, or in `go list -m all`); build and vet of the new root packages on darwin, linux/amd64 and windows (amd64, arm64); `realckpt`; `GOEXPERIMENT=simd` (linux/amd64, vet); the `metal` module with `metal goinfer_testhooks`; the whole `cuda` module with `GOOS=linux` and `cuda goinfer_testhooks`; the `gpu` module with `gpu`; `gofmt -l` empty; CI's staticcheck (2026.2.1, 0.8.1, its canary printing U1000) clean on the new packages, darwin and linux, and on the `metal` and `cuda` ones with their tags |
| AX1-T (tests carry over) | PASS: 0 regressions by name, `ax1_compare.py` |

AX1-T, per package (`go test -v -count=1`, one package per call, under the shared timing lock). The aikit baseline ran
11:43:57 to 11:44:08 PDT and goinfer 11:44:31 to 11:44:39; registered at about 10 minutes a side, measured 11 s and 8 s.

| package | aikit PASS | goinfer PASS | skips (same on both sides unless noted) |
|---|---:|---:|---|
| vision | 68 | 73 | 8 real-checkpoint or env-gated (`ALS_G0`, `ALS_G2`, gemma4 E2B and 26B real, GLM-OCR real and cost sweep, Qwen3 real and DeepStack real); aikit also skipped the 3 gemma4 tiny tests, which pass here because the fixture is present (+2 gemma4 subtests) |
| audio | 15 | 15 | `TestEncoderSpeed_record` (env-gated) |
| gpudev | 59 | 59, then 56 (Metal half) | 2 fence probes, the no-copy and shared-event probes (env-gated), `TestPTXReproducible` (needs NVRTC). After the split, three file lints (`TestKernelFMALint`, `TestKernelFMALint_coversEmbeddedPTX`, `TestGemvQuantFMAHistogram`) live in the CUDA half with the `.cu` and `.ptx` files they read; no device, run by CI's Linux cuda job |
| qwenmetal | 5 | 5 | none |
| visionmetal | 5 | 5 | `TestMetalViTThroughput` (env-gated benchmark) |

Logs: `logs/aikit-*.log`, `logs/goinfer-*.log` (the first layout, 11:44) and `logs/goinfer-split-*.log` (the split, 12:44).
vision, audio, qwenmetal and visionmetal read the same after the split.

## After the first CI run (2026-10-26)

PR #274's first CI run was red in five jobs. Four causes, each reproduced on the Mac where it could be:

- **cmd/gate's quick selector** (`TestQuick_neverDropsAGoListDependent`, root-darwin and root-windows) dropped the cuda
  module's test binaries for an edit to the `.ptx` files of a root-module `internal/gpudev`: they were embedded only by its
  Linux files, and the selector takes a package's files from its owner's listing (the root module, listed for the host).
  The split below removes the case, so the selector is unchanged (the test reads 0 dropped). The gap stays for any future
  root package whose files differ by GOOS and that another module compiles for a different GOOS.
- **touched.py's scan table** lacked the three gpudev tests that read source; they are now in `scripts/touched_scans.py`.
- **`TestEncoder_useResidencySet`** (root-darwin): the hosted runner's virtualized GPU makes no residency set, and the
  test's helpers called `Fatalf` on the parent test from inside subtests (the `panic(nil)`). It now skips when the device
  makes no set, and its helpers fail the subtest. It still passes on the MacBook.
- **`TestVisionCUDA_graphCapture`** (cuda, a GPU-less runner) now skips when there is no CUDA device at all.

The last two change copied files; `ax1_copy.py check` lists them as edited after the copy and compares the rest byte for byte.

## After the second CI run: the device package splits by backend

The second run left one red: lint's cleanliness guard, which keeps purego, gocudrv and aikit's `gpu` out of the root
module's packages and module graph. A root-module `internal/gpudev` put both in. The coordinator asked for the code to
move and the guard to stay. Of the two ways, a nested module would have been a sixth module (`RELEASING.md` is written for
five; the task doc's rule 4 is "packages, not modules"), so the package splits instead: its CUDA half goes to the `cuda`
module and its Metal half to the `metal` module, both of which already require purego and gocudrv. About 330 lines (the
coalesced-copy helper, its test, the upload-batch test, the package doc) are in both halves. The doc-consistency test that
compares the two backends' buffer docs lives in the Metal half and reads the CUDA half's files by relative path.

## Not run here

The CUDA code builds on Linux only. It compiled here with `GOOS=linux`; its device tests run on nobara:

- `cuda/internal/gpudev`: the device tests (`cuda_*_test.go`, `roofline_test.go`), `TestPTXReproducible` (needs NVRTC)
  and the three file lints. From `cuda/`: `go test -v -count=1 -timeout 20m ./internal/gpudev/`.
- `cuda/internal/visioncuda` (`TestVisionCUDA_*`) and `cuda/internal/qwencuda` (`TestQwenCUDA_*`): from `cuda/`,
  `go test -v -count=1 -timeout 20m ./internal/visioncuda/`, then the same for `./internal/qwencuda/`.

Each against aikit at `gpu/v0.35.0` with the same flags as the baseline, compared with `ax1_compare.py`.
The `goexperiment.simd` vision files were vetted for linux/amd64, not run.

## nobara device run (2026-10-26)

The three CUDA packages above, run on nobara: NVIDIA GeForce RTX 2070 SUPER (8 GB), driver 595.91.07, Go 1.27.3
linux/amd64, NVRTC 12.9.86 for `TestPTXReproducible`. goinfer at `origin/main` `f8e248b77` (a detached worktree, with a
`go.work` naming the repo's own modules and aikit at its `go.mod` pin, v1.75.0); the baseline is a `git archive` of aikit
`gpu/v0.35.0` (`0fd8325c`), run with `GOWORK=off` (its sub-modules' `replace` lines point inside the snapshot). Each
`go test -v -count=1 -timeout 20m`, one package per call, aikit then goinfer, under the exclusive timing lock, 15:29 to
15:30 PDT. Load average 0.14, 60 GB available; the timing lock and the night queue were idle.

| package | aikit PASS | goinfer PASS | result |
|---|---:|---:|---|
| `visioncuda` | 4 | 4 | PASS, 0 regressions: `TestVisionCUDA_parityWithCPU`, `_registersBackend`, `_repeatable`, `_graphCapture` |
| `qwencuda` | 4 | 4 | PASS, 0 regressions: `TestQwenCUDA_parityWithCPU`, `_fixtureExercisesBothAttentionKinds`, `_repeatable`, `_allocationFailureIsAnErrorAndTheEncoderRecovers` |
| `gpudev`, as registered | 18 | 18 | **both crash** in `TestCUDA_launchDirect` (`signal: segmentation fault (core dumped)`), so each run stops after the same 18 passes; `ax1_compare.py` reads 0 regressions |
| `gpudev`, `-skip '^TestCUDA_launchDirect$'` | 68 | 65 | `ok` on both sides; the 3 tests absent in goinfer are by design (below) |

No test skipped on either side; no test failed.

**`TestCUDA_launchDirect` segfaults in aikit at the tag as well as in goinfer**, the same way, alone (`-run` of that one
test, with `GOTRACEBACK=all`: the process dies with no Go traceback, so the fault is outside Go code). It is not a
difference the copy introduced, and it was not localized or fixed here. It is the device test of `Queue.LaunchDirect`, which
the CUDA resident's launch path calls (`docs/measurements/ks-a2-2026-10/README.md`), so a crash in it on the one CUDA card
is worth its own look.

The three tests aikit passes that goinfer's `cuda/internal/gpudev` does not run:

- `TestOutputContract_newBufferDocsAgreeWithBackends` and `TestOutputContract_hostWritersStateTheirs` live in the Metal
  half (`metal/internal/gpudev/outputcontract_doc_test.go`), as the split section above says.
- `TestPTXReproducible/gemv_w8a8` rebuilds `anncuda`'s PTX; `anncuda` stays in aikit (SD2,
  `docs/tasks/task-subsystems-2026-10.md`). The other four PTX files (`gemv_quant`, `roofline`, `smoke`, `vit`) reproduce
  byte for byte on both sides.

Logs: `logs/nobara-aikit-*.log` and `logs/nobara-goinfer-cuda-internal-*.log`; `*-skip-launchDirect.log` are the reruns
without that test, `*-launchDirect.log` the single-test runs.
