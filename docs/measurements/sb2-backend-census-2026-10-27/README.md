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
- After main's buffer-lease change to the upload code (`c200a84de`, `94b301608`) was merged in, the probe gave the same file
  again.

## The routed-expert kind check (review of #348)

WebGPU's check that routed experts are int8 or int4 moved from the upload into `staticResidentDecline`. I re-ran the device probe
(`webgpu_before_test.go.txt`) on that tree. Against `webgpu-before.txt`, it differs on exactly the 16 f32 MoE cells:
`err gpu: MoE residency expert kind "f32" (int8/int4 only)` became `err resident path declined: the routed experts are f32, ...`.
Both fall back to the CPU, and the other 202 answers are byte-identical. Layer B-webgpu's golden changes on the same 16 cells.

## Layer B-metal (mac-vk01, 2026-10-27)

`metal/sb2_census_test.go:TestBackendAdmissionCensus_metal`, golden `metal/testdata/sb2_metal_census.arm64.txt.gz`.

- **What the census records.** A fake backend named metal runs `residentPreBuild` (the feature taxonomy, an explicit
  `-ctx` cap, the fit choice and the memory guard with its context shrink), records the answer, and declines. No device is
  opened.
- **The machines.** It runs once per recorded machine with a Metal budget: `m1pro-2026-10-20`, `m1pro-apps-open` and
  `m1pro-near-idle`.
- **Each profile's readings replace the machine's:** `metalLiveAvailable`, `metalWorkingSetCap`, the decoder's metal probe,
  its host-available figure, and physical RAM. Physical RAM became a variable (`physicalRAM`), a test seam with no change
  in behavior.
- **Cells:** 274 reach Metal's build per profile, 822 lines in all, on layer D's 46 fixtures. Every cell layer D builds
  resident on metal is among them.
- **Answers**, the same on all three profiles:
  - 804 `ok ctx=4096 fit=4096 from=4096 why=the default`;
  - 18 `ok ctx=16384 fit=16384 from=16384 why=fit: 1 slot, 0.0 GB KV`. These are `qwen3next-tiny` in six cells, where the
    fit choice raises the context.
- **No line takes a decline.**
  - The decoder declines a missing feature first, so those cells are layer D's lines and do not reach here.
  - No cell pins `-ctx`.
  - No tiny fixture comes near the memory guard on any profile.
- **Both builds agree.** One build of 822 lines takes about 2.1 s on the M1 Pro.
- **The plant departs from the brief.** No line takes a `residentPreBuild` decline, so a `…OffForTest` beside one would
  have nothing to change. The plant is in the decision that varies: `metalFitFreeUnknownForTest` makes the fit choice treat
  the Metal budget as unknown. It turns 18 of 822 lines red, exactly the ones whose context the fit raised, and no other
  line changes.
- **The memory guard is not censused by this layer.** No fixture reaches it on any profile. Censusing it is the next layer:
  a fixture-free table test of `residentMemoryDecline` and `shrinkCtxToFit` over the profiles' figures. It needs
  registering before it is written.
- **The ratchet.** Francis exempted the census files by name (his answer 1, 2026-10-27), so
  `internal/layout:TestBackendTests_decoderImportsOnlyFall` is green with `metal/sb2_census_test.go` named beside the cuda
  and webgpu ones. This layer was rebuilt on main after that landed, and the census matches the golden it was written
  with: no admission answer changed in between.

## Not yet censused

- Each backend's memory and device reads inside the build: CUDA's weight and KV fit checks, expert slots, prefill width and SM
  shape, and WebGPU's KV slot clamp.
- Metal's memory guard (above).
