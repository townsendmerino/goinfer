# SB2 memory reads, step 2: the memory census over the machine profiles (2026-10-29, nobara-sb01)

Gates are registered in `docs/tasks/task-subsystems-2026-10.md`, "SB2 memory reads, step 2", at `33e2d3c51`, before the code.
The work builds on step 1's seam (`cudaBackend.planMemory`, #468). Box: nobara, RTX 2070 SUPER, driver 595.91.07.

## The census

`cuda/sb2_census_test.go:TestBackendMemoryCensus_cuda`, golden `cuda/testdata/sb2_cuda_mem_census.amd64.txt.gz`. It covers every
cell layer B-cuda plans (279) × each profile with `cuda_free_bytes` (`nobara-desktop`, `nobara-no-desktop`): 558 lines. Each
line holds:
- the figures fed, labelled `(profile)` or `(profile-pool)`;
- the planned context;
- the margin;
- the companion reserve;
- the weights verdict;
- `kv: step 3`, since the KV fit reads free VRAM after allocation and is not fed here.

A MoE build whose context is planned pays `moe_route`'s pool before its plan (`memPlan.PoolPaid`). So it is fed the profile's
figure less the profile's `cuda_moe_route_pool_bytes`. That field is now on both nobara profiles: 138,412,032, step 1's G-MP4.

## Gates

- **G-MC1:** built twice in one process, equal; matches the golden; 558 lines = 279 planned cells × 2 profiles (`gmc12.log`).
  Deviceless (`CUDA_VISIBLE_DEVICES=`), 2.9 s.
- **G-MC2:** with `weightsFitDecision`'s margin planted a thousandfold, all 558 lines refuse at the weights (`gmc12.log`).
- **G-MC3, on the card** (`gmc3.txt`, probe `sb2_mcprobe_test.go.txt`, card holding only the compositor `gmc3-card.txt`). The
  registered memory probe was read immediately before each of 412 loads and compared with the real build's readings:
  - the four qwen3next-tiny MoE cells that plan: `weightsOpen` equals the probe to the byte; `plan` and `weights` equal the probe
    less exactly 138,412,032. That is the census's `profile-pool` figure, with zero deviation;
  - the 275 other cells that reach the weights check: the reading equals the probe to the byte, on every cell.

So each figure the census feeds is what the card gives the build, up to the profile's own reading.

## What the census pins, as stated before the run

Both profiles are about 7.6 GB free, and the tiny fixtures' weights are kilobytes. Every verdict is `weights ok`, and the two
profiles' lines differ only in the figures. The census pins the decision path and the figures fed. Profiles of smaller cards,
or cells over real checkpoints, would make it discriminate.
