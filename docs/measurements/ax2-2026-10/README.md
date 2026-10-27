# AX2: the towers' importers switch to goinfer's copies (2026-10-26)

Track AX of `docs/tasks/task-subsystems-2026-10.md`, steps AX2a to AX2d; the gates are registered in §7.8 (AX2a and
AX2b) before each step's code. Session mac-ax01, MacBook (arm64, Metal), Go 1.27.3. Tier 0 throughout.

## AX2a: the tower-identity instrument

`internal/towerid` (contract in its package comment) runs the twelve tower families on their committed tiny fixtures,
on the CPU or through each device backend's production registry, and fingerprints every output. One test per backend
calls `towerid.Check`: `internal/towerid` (CPU), `metal`, `gpu` (`-tags gpu`) and `cuda` (`-tags cuda`).

| gate | result |
|---|---|
| AX2a-T, CPU | **PASS**: 12 families, 22 cases, two runs equal; 0.2 s a run |
| AX2a-T, Metal | **PASS**: the 6 families Metal declares (SigLIP, Qwen2.5-VL, Qwen3 and its DeepStack form, GLM-OCR, Gemma 4 vision, Gemma 4 audio), 17 cases, every one attached on the device; 0.55 s for both runs |
| AX2a-T, WebGPU | **PASS**: the 5 families WebGPU declares, 15 cases, every one attached on the device; 0.46 s |
| AX2a-T, CUDA | not run here: builds with `GOOS=linux -tags cuda`; nobara runs it with AX2d |
| fails closed | `TestRun_unregisteredDeviceIsAnError`: with no backend linked in, `Run` on metal, cuda and webgpu each errors on SigLIP ("no resident encoder backend") instead of returning CPU bits, which is how a registry split shows up |

The registered cost estimate was under a minute a backend; each measured under a second, plus the test binary's build.

Records at AX2a's head, the base for AX2b-ID: `records-ax2a/{cpu,metal,webgpu}-darwin-arm64.json`. `ax2_compare.py`
compares two records case by case, with no tolerance.

## AX2b: the import switch

Three commits: the mechanical switch (`763b7711`, made by `ax2_rewrite.py`), this record's script and head records, and
`multimodal.SiglipTower` and `GlmOcrTower` for the invoice example and the agent demo.

| gate | result |
|---|---|
| AX2b-N (the mechanical commit is the script) | **PASS**: `ax2_rewrite.py check 763b7711` reproduces all 137 files from the parent, 0 problems. The mapping is aikit's `vision`, `audio` and `gpu/visioncuda` to goinfer's copies |
| AX2b-S (nothing left on aikit's towers) | **PASS** after one catch: `go list -deps -test` of the root (plain, `realckpt goinfer_testhooks`, windows), `metal`, `gpu`, `cuda` (`GOOS=linux`) and `demo/agent` names neither package. The first run found `cuda/s4siglip`'s blank import of aikit's `gpu/visioncuda`. That import compiles and would have registered into aikit's `vision`, which nothing reads after the switch, so that test's `EnableResident` would have failed on nobara. `visioncuda` joined the script's mapping (its copy, `cuda/internal/visioncuda`) and the mechanical commit was remade |
| AX2b-ID (tower identity) | **PASS**: the head's records equal AX2a's on the CPU (22 cases), Metal (17) and WebGPU (15). CUDA and nobara's CPU wait for nobara (relayed) |
| AX2b-M (the gate can go red) | **PASS**, registered fallback form. Leaving any one unnamed-slot registration file on aikit's `vision` does not compile (tried `metal/vl_towers.go`, `gpu/vision_register.go`, `gpu/qwen25_vision.go`, `cuda/vision_register.go`, `cuda/qwen25_vision.go`): each shares the tower types with its package. Planting at the registration instead turned `TestTowerIdentity_metal` red both times. An emptied `vision.RegisterResident` slot gave "siglip on metal: … no resident encoder backend". A renamed Qwen3 registration gave `no "metal" Qwen3.5 tower in this binary (registered: [metal-planted])`. Green again after restoring |
| AX2b-B (builds) | **PASS** locally: build and vet of every module under its tags (root also `realckpt`, cuda for linux, demo/agent through CI's workspace shape); CI's staticcheck clean in all five, canary red; root cleanliness guard 0. `touched.py` (15:34 to 15:41) found one red, `TestEnvVars_docAndCodeAgree`: `internal/towerid` read `GOINFER_TOWER_ID_OUT` from non-test code. Fixed on AX2a's branch (the tests read it and pass the path) and merged up; the test passes. CI's full suite and apidiff run on the PR |

The image and audio parity tests run in CI's suite; the switch changes no golden file.

## AX2c: the metal module on metal/internal/gpudev

One mechanical commit (`ax2cd_rewrite.py metal`): 18 files, the import path only. The first attempt missed
`metal/device.go`'s one-line `import gpu "…"`, so the package's type aliases still named aikit's types and the module did
not compile; the regex was fixed in its own commit before the mechanical one was made.

| gate | result |
|---|---|
| AX2c-N | **PASS**: `ax2cd_rewrite.py metal check` reproduces the 18 files, 0 problems |
| AX2c-S | **PASS**: `go list -deps -test -tags goinfer_testhooks ./...` in `metal` names no `aikit/gpu` |
| AX2c-ID | **PASS**: Metal's record at AX2c's head equals AX2b's, 17 cases (`records-ax2c/`) |
| AX2c-T | **PASS**: the five `metal` packages, base (AX2b's head, 15:45:40 to 15:51:53) and head (15:53:12 to 15:59:25), by name: `metal` 699 and 699 PASS, `metal/internal/gpudev` 56, `qwenmetal` 5, `visionmetal` 5, `metal/kernels` 10, 0 regressions. 12.9 to 13.0 GB available at each start (void below 4 GB). Registered at about 5 minutes a side; measured 6 |
| AX2c-B | **PASS**: build and vet (plain and `goinfer_testhooks`), staticcheck, gofmt |

Seven `metal` tests fail at the base and at the head alike, all for a gitignored fixture this worktree does not have
(`granite-tiny`, `kimi-tiny`, a GPT-2 and a Gemma checkpoint: "read config.json: no such file"); listed and excluded as
registered: `TestGPT2ResidentParityMetal`, `TestHiddenLastResidentParityMetal`, `TestI8_exactKernelModelIsSplit`,
`TestMLAResident_routingForced`, `TestMLAResident_tinyAgainstControl`, `TestMambaKernels_cpuParity`,
`TestMambaKernels_plantsAreRed`. Logs: `logs-ax2c/`.

## AX2d: the cuda module on cuda/internal/gpudev (Mac side)

One mechanical commit (`ax2cd_rewrite.py cuda`): 83 files, the import path only.

| gate | result |
|---|---|
| AX2d-N | **PASS**: `ax2cd_rewrite.py cuda check` reproduces the 83 files, 0 problems |
| AX2d-S | **PASS**: `GOOS=linux go list -deps -test -tags 'cuda goinfer_testhooks' ./...` in `cuda` names nothing under `aikit/gpu` |
| AX2d-B | **PASS**: build and vet for linux/amd64 with `cuda` and `cuda goinfer_testhooks`; staticcheck with CI's `cuda goinfer_testhooks` clean. With plain `cuda` it reports 10 unused helpers (used only under `goinfer_testhooks`), the same 10 as origin/main's, so none is this commit's |
| AX2d-ID, AX2d-T | on nobara (relayed): the commands are below |

### What nobara runs (by day, in its own session, under its timing lock shared)

From a checkout of this branch, with a `go.work` covering `.` and `./cuda`. `B` is AX2b's base (AX2a's head, `7d0727c0`),
`H2` AX2b's head (`67bcf5d4`), `C` AX2c's head (`3367cb58`, the commit before AX2d's mechanical one) and `D` AX2d's
PR head; check each out in turn:

- AX2b-ID and AX2d-ID, at `B`, `H2` and `D`:
  `GOINFER_TOWER_ID_OUT=<dir>/<rev>-cpu-linux-amd64.json go test -count=1 -v -run '^TestTowerIdentity_cpu$' ./internal/towerid/`
  and from `cuda/`: `GOINFER_TOWER_ID_OUT=<dir>/<rev>-cuda-linux-amd64.json go test -count=1 -v -tags cuda -run '^TestTowerIdentity_cuda$' .`;
  then `python3 docs/measurements/ax2-2026-10/ax2_compare.py` on each pair (B against H2, H2 against D).
- AX2d-T, at `C` and at `D`: from `cuda/`, one package per call, `go test -v -count=1 -timeout 60m -tags 'cuda goinfer_testhooks' -skip '^TestCUDA_launchDirect$' <pkg>`
  for every package with tests (`go list -tags 'cuda goinfer_testhooks' -f '{{if or .TestGoFiles .XTestGoFiles}}{{.ImportPath}}{{end}}' ./...`),
  plus `-run '^TestCUDA_launchDirect$' ./internal/gpudev/` alone; compare each package's two logs with
  `python3 docs/measurements/ax1-2026-10/ax1_compare.py`.

## AX2b-ID and AX2d on nobara (2026-10-26, nobara-ax1-cuda, relayed)

**Verdict: PASS, identical before and after.** RTX 2070 SUPER, driver 595.91.07, Go 1.27.3 linux/amd64. The runs held the
timing lock exclusively, from 16:23 to 16:41 PDT. At every start the load average was 0.9 to 3.5 and 58 GB was available.
`kwin_wayland` held a 17 MiB CUDA context, so a desktop session was logged in. Nobara's own write-up is
`nobara-ax2d/pr284-comment.txt`; its records and logs are beside it, copied read-only from nobara with its lock checked
free in the same command.

- **AX2b-ID and AX2d-ID:** the CPU (22 cases) and CUDA (17 cases) records at `7d0727c0`, `67bcf5d4`, origin/main and
  `528ced85` agree, 0 differ in every pair. All 17 CUDA cases record the device path at all four revisions. Re-checked
  here with `ax2_compare.py` on the copied records.
- **AX2d-T:** the before and after results match by name.

  | package | PASS | FAIL | SKIP |
  |---|---:|---:|---:|
  | `cuda` | 792 | 4 | 338 |
  | `cuda/internal/gpudev` | 65 | 0 | 0 |
  | `qwencuda` | 4 | 0 | 0 |
  | `visioncuda` | 4 | 0 | 0 |
  | `s4siglip` | 0 | 0 | 1 (real checkpoint) |

  The four `cuda` failures are the same on both sides, with the same messages and figures, so they are listed and
  excluded as registered: `TestCUDAKVSlots_pricedAgainstWhatIsLeft` (the free-VRAM probe ran twice during the build) and
  `TestResidentDenseBytes_matchesCUDADevice` with its 0.5B and 1.5B subtests (the build's device bytes exceed Plan's
  figure plus the scratch margin). **They are red on origin/main on nobara's card**, independent of Track AX.
  `TestCUDA_launchDirect` segfaults on both sides, as at AX1.

## AX3: no goinfer module requires aikit's gpu module

Gates registered in the task doc's §7.11 (16:54) before the code.

| gate | result |
|---|---|
| AX3-S (the scan) | **PASS**: `internal/movedpaths`'s `TestNoMovedAikitPath` is green on the tree. Before the requires were dropped it listed exactly the `go.mod`/`go.sum` lines AX3 removes. A planted aikit `gpu` import in a test file and a planted `aikit/gpu` require in `metal/go.mod` each turned it red with the file and line; green after each removal. Listed in `scripts/touched_scans.py` |
| AX3-M (the module graphs) | **PASS**: no `go.mod` or `go.sum` names `aikit/gpu` or `aikit/gpu/visioncuda`; `go list -m all` in `metal` and in `cuda` (`GOOS=linux`) names neither |
| AX3-B | **PASS**: build, vet and staticcheck of all five modules under their tags; the root cleanliness guard 0; `TestHardwareCensus` 31 predicates, 31 entries; `cmd/gate`'s `TestModRequire` and `TestRowPins`; `decoder`'s `TestAikitPinsAgree`; `test_touched.py`; the citation lint, exit 0 both with the sibling aikit checkout and as CI runs it (module cache only); `TestTowerIdentity_metal` equal to AX2c's record |
| AX3-G | deferred to the next release tag (`standalone-build.yml`), as registered |

**The citation lint needed no fallback**, contrary to what §7.11 registered. As CI runs it, with only the module cache, the
lint answered 23 lookups from aikit's root module and none from `aikit/gpu`, at origin/main and on this branch alike. No
live document resolves a citation through `aikit/gpu`. A pinned record's older `aikit/gpu` still comes through CI's
`--pinned-modules` step, which reads each pinned commit's own `cuda/go.mod`.

One import was left behind by AX2d: `cuda/kernels/ptx_test.go` arrived with SB1's batch 1 after AX2d's rewrite ran. A
mechanical commit at AX3's start switched it. From AX3 on, `internal/movedpaths` catches the next one.

## Review of #282 (Gemini, relayed 2026-10-27)

Four findings. Two [test] findings held on origin/main; each got a test that was red there and is green after the fix.

- **Whisper and Voxtral could pass vacuously on CUDA.** `towerid`'s `device()` read the runtime Whisper registry, so a CUDA
  build whose Whisper registration went missing would skip both families and still pass. The backends that serve the
  Whisper encoders are now declared in `towerid` (`whisperBackends`), so a missing registration errors in
  `multimodal.UseWhisperDevice`. `TestRun_whisperDeclaredButUnregisteredIsAnError` holds it.
- **`siglip()` did not check residency after `EnableResident`.** A factory returning `(nil, nil)` recorded `device: true` with
  the CPU's bits; the red run recorded hash `2040d686…`, the CPU record's. `vision.Encoder` gains `ResidentEnabled()`, as
  Qwen's encoder has, and `siglip()` checks it. `TestRun_siglipResidentFactoryReturningNothingIsAnError` holds it.
- **[nit] `LoadSiglipTower`'s comment said the WebGPU tower needs quant.** It does not: unquantized, WebGPU builds the f32
  tower (towerid's WebGPU SigLIP case runs it). Corrected.
- **[nit] the ARCHITECTURE package table's dependency column.** Already gone on main: SB13 and SB14 replaced it with the table
  generated from `internal/layout/imports.json`, which lists `internal/audio` and `internal/vision` correctly.

The records still equal AX2b's on the CPU (22 cases), Metal (17) and WebGPU (15); the fixes change no tower code. The CUDA
change compiles here (`GOOS=linux`); nobara's `TestTowerIdentity_cuda` run follows, and it should now record the Whisper
and Voxtral cases (17, as at AX2d) or fail.
