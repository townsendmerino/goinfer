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
