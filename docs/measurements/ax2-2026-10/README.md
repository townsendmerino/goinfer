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
