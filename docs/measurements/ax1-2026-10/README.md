# AX1: aikit's towers and GPU backends copied into goinfer (2026-10-26)

Track AX of `docs/tasks/task-subsystems-2026-10.md`, step AX1; the gates are registered in that doc's §7.6. Session
mac-ax01, on the MacBook (arm64, Metal), Go 1.27.3. Everything here is tier 0: no model, no timed run.

## What was copied, and from where

| aikit path | tag (commit) | goinfer path |
|---|---|---|
| `vision` | `v1.75.0` (`38dc8e2b`) | `internal/vision` |
| `audio` | `v1.75.0` | `internal/audio` |
| `gpu` (the device module) | `gpu/v0.35.0` (`0fd8325c`) | `internal/gpudev` (package `gpudev`) |
| `gpu/qwencuda`, `gpu/visioncuda` | `gpu/v0.35.0` | `cuda/internal/qwencuda`, `cuda/internal/visioncuda` |
| `gpu/qwenmetal`, `gpu/visionmetal` | `gpu/v0.35.0` | `metal/internal/qwenmetal`, `metal/internal/visionmetal` |
| `gpu/webgpu` | — | not copied: nothing imports it, and the `gpu` module has its own WGSL (retirement proposed) |

Both are the newest aikit tags on 2026-10-26, and aikit main (`3e7c479`) matches them on every copied path (only the
four sub-module `go.mod` pins differ). aikit's README freezes these paths at the same tags (aikit PR #21).

`ax1_copy.py copy` made the copy from `git archive` extractions of the two tags; its edits are import paths, the device
package's clause (`gpu` to `gpudev`, importers aliasing it `gpu`), and fixture paths. The vision tests' aikit fixtures
moved byte for byte to `internal/vision/testdata`, with goinfer's `gemma4-vision-tiny` (aikit does not track it) and the
aikit oracle scripts that pinned the goldens.

## Gates

| gate | result |
|---|---|
| AX1-S (source identity) | PASS: `ax1_copy.py check` recomputes 174 files from the tag snapshots, 0 differ, no extra files |
| AX1-N (adds only) | PASS: 176 added files; modified only `go.mod` and `go.sum` (purego and gocudrv, lines added, none removed), `.gitignore` (the fixture exception), `NOTICE`, `THIRD_PARTY_LICENSES.md` |
| AX1-B (builds) | PASS: build and vet of the new packages on darwin, linux/amd64 and windows; `realckpt`; `GOEXPERIMENT=simd` (linux/amd64, vet); the `metal` module with `metal goinfer_testhooks`; the whole `cuda` module with `GOOS=linux` and `cuda goinfer_testhooks`; the `gpu` module with `gpu`; `gofmt -l` empty; CI's staticcheck (2026.2.1, 0.8.1, its canary printing U1000) clean on the new packages, darwin and linux, and on the `metal` and `cuda` ones with their tags |
| AX1-T (tests carry over) | PASS: 0 regressions by name, `ax1_compare.py` |

AX1-T, per package (`go test -v -count=1`, one package per call, under the shared timing lock). The aikit baseline ran
11:43:57 to 11:44:08 PDT and goinfer 11:44:31 to 11:44:39; registered at about 10 minutes a side, measured 11 s and 8 s.

| package | aikit PASS | goinfer PASS | skips (same on both sides unless noted) |
|---|---:|---:|---|
| vision | 68 | 73 | 8 real-checkpoint or env-gated (`ALS_G0`, `ALS_G2`, gemma4 E2B and 26B real, GLM-OCR real and cost sweep, Qwen3 real and DeepStack real); aikit also skipped the 3 gemma4 tiny tests, which pass here because the fixture is present (+2 gemma4 subtests) |
| audio | 15 | 15 | `TestEncoderSpeed_record` (env-gated) |
| gpudev | 59 | 59 | 2 fence probes, the no-copy and shared-event probes (env-gated), `TestPTXReproducible` (needs NVRTC) |
| qwenmetal | 5 | 5 | none |
| visionmetal | 5 | 5 | `TestMetalViTThroughput` (env-gated benchmark) |

Logs: `logs/aikit-*.log` and `logs/goinfer-*.log`.

## After the first CI run (2026-10-26)

PR #274's first CI run was red in five jobs. Four causes, each reproduced on the Mac where it could be:

- **cmd/gate's quick selector** (`TestQuick_neverDropsAGoListDependent`, root-darwin and root-windows) dropped the cuda
  module's test binaries for an edit to `internal/gpudev/testdata/*.ptx`. The ptx files are embedded only by gpudev's Linux
  files, and the selector took a package's files from its owner's listing (the root module, listed for the host) alone.
  `loadGraph` now adds the files another module's listing compiles. On the Mac the test reads 4 dropped without the change
  and 0 with it.
- **touched.py's scan table** lacked the three gpudev tests that read source; they are now in `scripts/touched_scans.py`.
- **`TestEncoder_useResidencySet`** (root-darwin): the hosted runner's virtualized GPU makes no residency set, and the
  test's helpers called `Fatalf` on the parent test from inside subtests (the `panic(nil)`). It now skips when the device
  makes no set, and its helpers fail the subtest. It still passes on the MacBook.
- **`TestVisionCUDA_graphCapture`** (cuda, a GPU-less runner) now skips when there is no CUDA device at all.

The last two change copied files; `ax1_copy.py check` lists them as edited after the copy and compares the rest byte for byte.

## Not run here

The CUDA code builds on Linux only. It compiled here with `GOOS=linux`; its device tests run on nobara:

- `internal/gpudev`, the Linux test files (`cuda_*_test.go`, `roofline_test.go`) and `TestPTXReproducible` (needs NVRTC):
  `go test -v -count=1 -timeout 20m ./internal/gpudev/` from the repo root.
- `cuda/internal/visioncuda` (`TestVisionCUDA_*`) and `cuda/internal/qwencuda` (`TestQwenCUDA_*`): from `cuda/`,
  `go test -v -count=1 -timeout 20m ./internal/visioncuda/`, then the same for `./internal/qwencuda/`.

Each against aikit at `gpu/v0.35.0` with the same flags as the baseline, compared with `ax1_compare.py`.
The `goexperiment.simd` vision files were vetted for linux/amd64, not run.
