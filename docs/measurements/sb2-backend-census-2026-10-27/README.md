# SB2 step 2: the backend layers of the admission census, 2026-10-27 (nobara-sb01)

Registered in `docs/tasks/task-subsystems-2026-10.md`, "SB2 step 2", before the code. Nothing here is timed.

## Layer B-cuda

`cuda/sb2_census_test.go:TestBackendAdmissionCensus_cuda`, golden `cuda/testdata/sb2_cuda_census.amd64.txt.gz`.

- 412 cells reach CUDA's build, on layer D's 70 fixtures × 9 option cells. Every cell layer D builds resident on cuda (396) is among
  them. The other 16 are `int4/kv-f16` cells whose layer D load fails after the build, on the KV it asked for.
- Answers: 279 plan (slack 0 on every tiny fixture, since no buffer reaches 2 MiB) and 133 static declines.
- Both builds agree. Wall time is 1.7 s for the two builds, on nobara with no device opened.
- Plant: `moeActG32DeclineOffForTest` turns 42 of 412 cells red, every one of them a cell that takes the per-32 MoE decline, and no
  other cell changes.
- A zero-slack plant (`allocSlackOffForTest`) cannot show red on tiny fixtures, since every slack is already 0, so it is not the
  plant.

## Layer B-webgpu

`gpu/sb2_census_test.go:TestBackendAdmissionCensus_webgpu`, golden `gpu/testdata/sb2_webgpu_census.<goarch>.txt.gz`.

- The refactor moves WebGPU's static gates into `gpu/residency.go:staticResidentDecline`, in their order. These are the five checks
  before the build's lock, plus the quantized-KV layout decline, which read the model alone. `BuildResident` calls it first.
- 218 cells reach the build. 207 pass the static gates, and 11 take the quantized-KV decline (qwen3_5, nemotron, mla at `--kv f16`).
- Plant: `quantKVDeclineOffForTest` turns those 11 red, and no other cell changes.
- The arm64 golden is a copy of the amd64 one. The lines carry no int4-layout text, and CI's gpu-darwin job holds it either way.

### Before and after the refactor

`webgpu-before.txt` is the same census run before the refactor (`webgpu_before_test.go.txt`, removed afterwards). It ran on
nobara's device (RTX 2070 SUPER, driver 595.91.07, Vulkan), through the real `BuildResident`, on the tree at `cb9cf471d`.

- Of the 218 cells, 191 built, 11 took the quantized-KV decline, and 16 failed mid-upload with
  `gpu: MoE residency expert kind "f32" (int8/int4 only)`.
- Compared with the new census: the cell sets are equal. Every one of the 11 declines has the same text, and every built cell or
  mid-upload failure reads `static ok`. That is 0 mismatches in 218.
- The f32 MoE expert check is made inside the upload, so it stays there. Moving it would turn an error into a typed decline, which
  changes what `DecodePath` reports.
- Layer D cannot change: the decoder package does not link the gpu module.
- The same probe, re-run on the refactored tree, gives a byte-identical file: 218 answers, with the real build on the device.

## Not yet censused

- Each backend's memory and device reads inside the build: CUDA's weight and KV fit checks, expert slots, prefill width and SM
  shape, and WebGPU's KV slot clamp.
- Metal, whose census is a Mac session's.
