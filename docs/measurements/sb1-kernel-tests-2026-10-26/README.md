# SB1 step 1, batch 1: CUDA kernel tests out of package `cuda` (2026-10-26, nobara)

The gates are registered in `docs/tasks/task-subsystems-2026-10.md`, "SB1 step 1 — gates". This is the record of batch 1.

- **Tree:** branch `sb1-decoder-imports-2026-10-26` off `origin/main` `bc365a837`; the before-run at `2d8863338` (the ratchet
  commit), the after-run on the moved tree.
- **Box:** nobara, RTX 2070 SUPER, driver 595.91.07. The card held one other context in both runs, the desktop
  compositor's (`kwin_wayland`, 17 MiB, the one `CLAUDE.md` says to count); no other process. The timing lock was free and
  each run held it shared.
- **What moved:** 25 test files, from `cuda/` to `cuda/kernels/`, with 35 `Test` and 16 `Benchmark` functions.
  None imported `decoder` itself; each was tied to it only by package `cuda`'s own imports. `go list -deps -test
  ./cuda/kernels/` (tags `cuda goinfer_testhooks`) names no `goinfer/decoder`.
- **Edits beyond `package kernels`:** the build line drops `goinfer_testhooks` on the eight files that had it; the two
  test-only PTX embeds (`addone`, `gemv_w8a8`) and the sixteen `cuda/kernels.go` embeds the tests use are read from
  `cuda/testdata/*.ptx` by `cuda/kernels/ptx_test.go`; `mustAlloc`, `mustStream` and `gcKVIdentityTab` are copied into
  `cuda/kernels/helpers_test.go` (the table's block size is `kvpage.cuh`'s 64).

## Result: same outcome for every moved test

| run | command | result |
|---|---|---|
| before | `cd cuda && go test -count=1 -v -tags 'cuda goinfer_testhooks' -run '<the 35>' .` | 32 PASS, 3 SKIP, 2.5 s (`batch1-before.log`) |
| after | `cd cuda && go test -count=1 -v -tags cuda -run '<the 35>' ./kernels/` | 32 PASS, 3 SKIP, 2.3 s (`batch1-after.log`) |
| `gate gpu` 2a, as the cell runs it | `go test -count=1 -p 1 -short -v -tags cuda ./cuda/ ./cuda/kernels/` | both `ok`; 129 PASS, 19 SKIP, 50 s wall (`batch1-2a-suite.log`) |

The per-test outcome lists of the before and after runs are identical (`diff` empty). The three skips are the same, with
the same reasons, in both: `TestA13_SingleFailedAllocPoisons` (`GOINFER_A13_SINGLEFAIL` unset), `TestAttnBatchedBandwidth`
and `TestLead3_pinOrderMicrobench` (`GOINFER_HEAVY_TESTS` unset). `go test -list '.*'` finds all 51 names before
(in `./cuda/`) and after (in `./cuda/kernels/`). The benchmarks were listed, not run.

`gate gpu` keeps every moved test: 2a lists `./cuda/kernels/` beside `./cuda/`, and the heavy tier runs
`./cuda/kernels/` whole as one more process (label `kernels`) under `GOINFER_HEAVY_TESTS=1`, reconciled with the
`./cuda/` groups, so the two heavy-gated bandwidth tests still run there. The heavy tier itself was not run for this
batch. CI's cuda job and the night `cuda-device-suite` run `./cuda/...` and needed no change.

Build and vet of the cuda module pass under no tag, `cuda`, and `cuda goinfer_testhooks`; staticcheck (CI's pinned
build) is clean under `cuda goinfer_testhooks`.

## The ratchet

`internal/layout`'s `TestBackendTests_decoderImportsOnlyFall`, cuda: direct 246 (unchanged: no moved file imported
`decoder`), reach 352 → 327. metal and gpu unchanged (245/395, 87/172).

## Files moved

`allocgran_test.go`, `gocudrv_proof_test.go`, `gptoss_act_test.go`, `layernorm_quant_test.go`, `lead3_pin_order_bench_test.go`, `moe_route_test.go`, `quant_vec_bench_test.go`, `quant_vec_exact_test.go`, `rmsnorm_batched_test.go`, `rope_mscale_test.go`, `vnorm_test.go`, `attn_batched_test.go`, `attn_bw_test.go`, `attn_sink_test.go`, `mla_latent_reuse_prototype_test.go`, `mla_test.go`, `prefill_splitkv_test.go`, `rope_pairwise_test.go`, `rope_partial_test.go`, `window_test.go`, `gemv_bw_test.go`, `decode_projection_test.go`, `e2e_decode_test.go`, `a13_singlefail_test.go`, `attn_headdim_test.go`.

# Batch 2 (2026-10-26, 16:41–16:50 PDT)

- **Tree:** before-run in a detached worktree at `e71abd3b0` (batch 1's head, on `main` as #286), after-run on the moved
  tree. Both waited for the timing lock behind another session's timed CUDA run and started after it ended; the card held
  only the compositor's context (`kwin_wayland`, 17 MiB) both times.
- **What moved:** `layernorm_quant_batched_test.go`, `gelu_quant_test.go`, `moe_route_reservation_test.go`,
  `route_gptoss_reservation_test.go` and `attn_fused_vit_test.go`: 6 tests, none importing `decoder`. Each shared one
  small helper with a test that stays: `cosineF32` now lives in `cuda/testhooks_helpers_test.go` for the three vision
  tests, and `cuda/kernels` holds its own copies of `moePTXOrOverride` (still used by `moe_route_demand_test.go`, which
  stays for its drain marker) and `f16rne` (still used by `attn_fused_test.go`). All five drop `goinfer_testhooks`.

| run | result |
|---|---|
| before, `./cuda/`, tags `cuda goinfer_testhooks`, the 6 by name | 5 PASS, 1 SKIP, 13.3 s (`batch2-before.log`) |
| after, `./cuda/kernels/`, tag `cuda`, the 6 by name | 5 PASS, 1 SKIP, 13.1 s (`batch2-after.log`) |
| `TestRouteGptOssGrowsPoolPastMoERoute` alone, before and after | PASS both, the same reservation figures (`batch2-alone.log`) |

Identical per test. The skip is `TestRouteGptOssGrowsPoolPastMoERoute` in both batch runs: it cannot evaluate once
`TestMoERouteFirstLaunchReservation` has reserved the local-memory pool in the same process, and says to run it alone,
which the third row does. All 6 names are listed before and after. Build and vet pass under no tag, `cuda`, and
`cuda goinfer_testhooks`; staticcheck is clean; `go list -deps -test ./cuda/kernels/` still names no `decoder`.

Ratchet, cuda: direct 246 (unchanged), reach 327 → 322.

**What blocks the next kernel tests.** About a dozen more decoder-free kernel tests (`shape_sweep`, `actgroup_kernel`,
`gemv_rn`, `gemm_mma`, `gemv_w4f32`, `gemv_w8a8_batched`, `moe_gemv`, `moe_pack`, `ks_a1`, `kso1c_kernels`, ...) check a
kernel against package `cuda`'s production packers (`packI8`, `packWeight`, `packWeightStack`, `f32tof16`,
`nibblePosFast`, `permuteFast`). They can leave only with the packers, and `packWeight` calls `decoder.Int4F32` and
returns the resident's `hostW`. The attention-digest tests need `prefill.go`'s and `flash_decode.go`'s launch geometry,
and the tower tests need the tower code, both still in package `cuda`.

# The packer move and batch 3 (2026-10-26, 17:40–18:10 PDT)

Francis's decision (17:36): move cuda's weight packers into a decoder-free package behind a bit-identity gate, then free
the kernel tests that needed them. The gate is registered in the task doc, "moving cuda's weight packers".

- **What moved:** the packing bodies of `packWeight`, `packWeightStack`, `packI8`, `packI8Into`, `packInt4Into`,
  `nibblePosFast`, `permuteFast` and `f32tof16`, into `cuda/internal/wpack` (imports aikit `linalg` only). Package `cuda`
  keeps each name as a thin wrapper; `packWeight` and `packWeightStack` still run `planWeight` ahead of each pack, member by
  member. `decoder.Int4F32`'s two calls (`Int4F16`, `linalg.F16ToF32Slice`) are made in `wpack`; no `decoder` file
  changed. `hostW` stays in package `cuda`.
- **Gate:** `cuda`'s `TestWeightPackers_bitIdentity`, 32 SHA-256 hashes (`cuda/testdata/wpack_hashes.txt`), written at
  `91fe36e7b` before any code moved (28 s), reproduced on the unchanged tree (30 s), committed as `064c42989`, then run
  unedited after the move: **all 32 match** (32 s). `packers-identity.log`.
- **Shown red twice,** each plant reverted: rounding `F32ToF16` on bit 0x0800 instead of 0x1000 fails
  `f32tof16/all-2^32` alone (the int4 scales go f16 → f32 → f16, so their low mantissa bits are zero and no rounding
  change reaches them: only the 2^32 sweep pins the rounding); changing `NibblePos` fails all nine int4-layout hashes and
  the two bit packers.
- **Tests moved:** `gemv_w4f32`, `gemv_w8a8_batched`, `moe_gemv` and `shape_sweep` to `cuda/kernels` (device);
  `moe_pack` and `f16_convert` to `cuda/internal/wpack` (they test the packer itself, need no device, and drop the `cuda`
  tag). Each destination has a small test file giving package `cuda`'s names over `wpack`, so the moved bodies are
  unchanged; `f16tof32` is copied into `actgroup_kernel_test.go`, its one remaining user in `cuda`.

| run | result |
|---|---|
| before, `./cuda/`, tags `cuda goinfer_testhooks`, the 10 by name | 10 PASS (`batch3-before.log`) |
| after, `./cuda/kernels/` and `./cuda/internal/wpack/`, tag `cuda` | 10 PASS (`batch3-after.log`) |
| after, `./cuda/internal/wpack/` untagged | 4 PASS |

Identical per test; all 10 names listed before and after. Build and vet of the cuda module pass under no tag, `cuda`,
and `cuda goinfer_testhooks`; staticcheck is clean. Neither destination's test build reaches `decoder`.

Ratchet, cuda: direct 246 (unchanged), reach 322 → 317 (six files out, the identity test in).

**Still in package `cuda`, and why:** `gemv_rn` (its `rnCfg` must match `gemv_w4a8_rn.cu`'s `#define`s, and
`gemm_mma_test.go` uses it); `actgroup_kernel`, `ks_a1` (their helpers feed `kso1c_kernels_test.go`, which loads models);
`gemm_mma` and the attention-digest tests (launch geometry in `prefill.go` and `flash_decode.go`); the tower tests.

# metal and gpu (2026-10-26)

- **gpu:** no test-only move exists. Its WGSL kernels and their dispatch (`New`, `Context`, `ResidentW8A8`, ...) are
  production code in package `gpu`, and every kernel test calls them; the few gpu test files that depend on nothing in
  the package are infrastructure (closers, writable, voidrule). A gpu kernel package means moving the kernel layer
  itself out of package `gpu`, a production split.
- **metal:** 62 test files (67 tests, 20 benchmarks) qualify by the rule cuda's batches used: their only package-metal
  dependencies are `metal/kernels` aliases (`msl.go`, `pack.go`), `device.go`'s aliases, and pure helpers in
  `testshared_test.go` (`cosF`, `argmaxF`, `f16ToF32`, `mustFinite`, `parityStats`, `assertParity`), none imports
  `decoder`, and no test that stays uses what they declare. They were not moved from nobara: metal tests build only for
  darwin, and CI's `metal-darwin` runs one device-free test, so no run here or in CI can show the before and after. Left out
  on purpose: `parity_falsify_test.go` (CI's one metal test), `tower_identity_test.go` (AX's towers),
  `embeddinggemma2_real_test.go` (a real checkpoint). The list, for a Mac session:

  act_quant_bench, attention_bench, attention_prefill_bench, attention, attn_kread_staged_probe, attn_kvwidth_probe,
  attn_m3_probe, cpulayer, deltanet_seq, gemm_mma8_mc3, gemm_smallm_mc3, gemma4_moekernels, gemv, gemv_w4a8_coal_bench,
  gemv_w4a8_moe_bench, gemv_w4a8_sa_bench, gemv_w4a8_sa_bias_resid, gemv_w4a8, gemv_w4f16, gemv_w8a8_amax_bench,
  gemv_w8a8_body, gemv_w8a8_coal_bench, gpt2_kernels, gptoss_route_degenerate, ksd1_split, ksd2_steel, ksd3_timing,
  ksd4_gemm, ksd7_dequant, layer, layernorm_f16, layernorm_quant_bench, metal, mla_kernels, moe_route_bench, moe_route_sg,
  qknorm, qknorm_whole, quant_vec_bench, residency_probe, rmsnorm_f32_bench, rmsnorm_quant_bench, rmsnorm_quant_f16_bench,
  rmsnorm, rope2_bench, rope2_kv, rope2, rope_pairwise, rope_partial, rope, s17_gemm_probe, sa_largek, sa_outlier,
  scale_roundtrip, simdsum_tree_mc3, swiglu_massive, swiglu_quant_bench, swiglu_quant_gptoss_bench, swiglu, tax, vnorm,
  window (each `<name>_test.go`).

# metal: the 62 files, on the Mac (2026-10-26, 21:40–21:50 PDT, mac-ax01)

The gates are registered in `docs/tasks/task-subsystems-2026-10.md`, "SB1 step 1 — metal's batches". Box: the MacBook
(M1 Pro, Metal), Go 1.27.3; about 13 GB available at each run; every run held the timing lock shared; nothing timed (the 20
benchmarks were not run, and the 12 skipped tests are env-gated probes, skipped on both sides for the same reason).

- **What moved:** all 62 files of nobara-sb01's list, from `metal/` to `metal/kernels/`, with 67 `Test` and 20 `Benchmark`
  functions, in four batches. `layer_test.go` went in batch 1 with `cpulayer_test.go`, which declares the `cpuLayer` and `q8row`
  it calls. A `go/ast` reference scan over the list found no other edge between a moved file and one that stays; its three
  other hits were local variables sharing a name.
- **Edits beyond `package kernels`:** `mla_kernels_test.go` imported `metal/kernels`, a cycle inside the package, so the
  import and its three `kernels.` qualifiers went. `metal/kernels/moved_helpers_test.go` gives the moved tests package
  `metal`'s lowercase names for this package's MSL sources (`allKernels`, `prefillKernels`, …) and `packW4A8Row` and
  `f32ToF16`. The copies of `metal`'s pure helpers turned out unused (U1000) and were dropped. Build lines are unchanged.
- **No `decoder` change**, and no test deleted: each of the 87 functions is declared in `metal/kernels` after the move.

| batch | files | tests | after the move (before: the same) | metal reach |
|---|---:|---:|---|---|
| 1 | 17 | 18 | 10 PASS, 8 SKIP | 395 → 378 |
| 2 | 15 | 19 | 18 PASS, 1 SKIP | 378 → 363 |
| 3 | 15 | 14 | 13 PASS, 1 SKIP | 363 → 348 |
| 4 | 15 | 16 | 14 PASS, 2 SKIP | 348 → 333 |
| all, after the helper prune | 62 | 67 | 55 PASS, 12 SKIP, 0 FAIL | 333 |

The before-run took 14.5 s for all 67 tests in `./metal` (tags `goinfer_testhooks`). Each after-run ran its batch by name in
`./metal/kernels` with the same tags and was compared test by test, skip reasons included. `go list -deps -test
./metal/kernels/` names no `goinfer/decoder`. The metal module builds and vets with and without `goinfer_testhooks`.
staticcheck finds the same 12 items in package `metal` as on origin/main, and none in `metal/kernels`. metal's direct count
stays 245: none of the moved files imported `decoder`. Logs: `metal-logs/`.

# Launch geometry out of package cuda (item 1 of 21:42; 2026-10-26, 21:45–22:10 PDT)

Read as `cuda/prefill.go` and `cuda/flash_decode.go`: the relayed `decoder/` paths do not exist (task doc, "launch
geometry out of package cuda"). No `decoder` file changed.

- **What moved:** `faWarps`, `faMaxRows`, `faRowsPerCTA`, `faRun`, `faRowRuns`, `attnFusedBM`, `attnFusedThreads`,
  `attnFusedKPAD`, `attnFusedMinRows`, `imgBlockMaxNWin` and `imgBlocksMaxNWin` into `cuda/internal/launchgeom` (standard
  library only); `f32ToF16RNE` and `f32SliceToF16RNE` into `cuda/internal/wpack`. Package `cuda` keeps every name, as a
  constant, an alias or a thin wrapper. `faRun`'s fields are exported now, so its readers in `flash_decode.go` and
  `attn_digest_test.go` name them `Row0`, `N`, `NKeys0`, `WinStart`, `Per`.
- **Gate 1:** `TestLaunchGeometry_identity`'s 7 hashes, written at `b344ee030` before any code moved (14.5 s) and
  reproduced (14.8 s), match after the move with the test unedited; the packer gate's 32 hashes match too
  (`launchgeom-identity.log`). Shown red: `AttnFusedKPAD` 8 → 16 fails the `constants` hash (reverted).
- **Gate 2:** the whole `./cuda/` package under `cuda goinfer_testhooks` (`gate gpu`'s 2b group, which includes the digest
  tests), before at `b344ee030` (302 PASS, 288 SKIP, 516 s) and after, with `./cuda/kernels/` (515.7 s and 38.1 s):
  every one of the 590 tests the before-run reported has the same outcome after. The after-run's other 47 names are the
  tests batches 1–3 had already put in `cuda/kernels`. `TestAttnKernelDigests` passes on both sides, so every attention
  kernel's output digest still equals `cuda/testdata/attn_digests.json`. (`TestMoERouteFirstLaunchReservation` and
  `TestRouteGptOssGrowsPoolPastMoERoute` skip in a whole-package run by design: each measures a first launch in its
  process, and says to run it alone; batch 2 ran both alone.)
- **Tests moved:** `attn_digest_rig`, `attn_digest`, `attn_softcap_kernel` and `prefill_splitkv_img` to `cuda/kernels`
  (paths to `cuda/testdata` and the `.cu` sources now go through `..`). `mamba_kernels_test.go` stays in `cuda`: it loads
  a model through `decoder` for its reference. It keeps a copy of the part of the digest rig it drives
  (`mamba_kernels_rig_test.go`); a shared rig package would remove the copy, at the cost of renaming every rig call.
- **Ratchet,** cuda: direct 246, reach 317 → 315 (four files out; the identity test and the rig copy in).

# Batch 4: the last test-only moves on cuda and gpu (2026-10-27, nobara-sb01)

- **Registered** in `docs/tasks/task-subsystems-2026-10.md` ("SB1 step 1 — the last test-only moves on cuda and gpu") at
  `a61076287`, before the code. The before-run ran in a detached worktree at that commit and the after-run on the moved tree.
- **Box:** nobara, RTX 2070 SUPER, driver 595.91.07, with Vulkan for gpu. The card held only the desktop compositor's 16 MiB
  context in both runs (`batch4-before-card.txt`, `batch4-after-card.txt`). Both runs held the timing lock shared, after a
  timed tack eval released it.
- **What moved:**
  - `cuda/gemv_rn_test.go` to `cuda/kernels`. Its PTX is read from `../testdata`, and it packs with `wpack.Pack` and
    `wpack.PackI8` (what package `cuda`'s `packWeight` and `packI8` call). `gemm_mma_test.go` keeps a copy of `rnCfg`.
  - `gpu/moe_expert_test.go`, `gpu/moe_test.go` and `gpu/gumbel_test.go` to `gpu/internal/kernels`. The Gumbel test's host
    reference is the same three `internal/sampling` calls `decoder.GumbelDrawForTest` and `decoder.GumbelKeyForTest` make.

| run | result |
|---|---|
| before, `./cuda/` (`cuda goinfer_testhooks`), the 2 by name | `TestGemvRN_bitIdentical` PASS, `TestGemvRNBandwidth` SKIP (`set GOINFER_HEAVY_TESTS=1`) (`batch4-cuda-before.log`) |
| after, `./cuda/kernels/` (`cuda`) | the same (`batch4-cuda-after.log`) |
| before, `./gpu/` (`gpu goinfer_testhooks`), the 4 by name | 4 PASS (`batch4-gpu-before.log`) |
| after, `./gpu/internal/kernels/` | 4 PASS (`batch4-gpu-after.log`) |

- **Outcome:** the same for every moved test, and all six names are listed before and after.
- **Checks:**
  - Build, vet and staticcheck pass for cuda under no tag, `cuda`, and `cuda goinfer_testhooks`, and for gpu under
    `gpu goinfer_testhooks`.
  - Neither destination's test build reaches `decoder`.
- **Ratchet:**
  - cuda: direct 246, reach 315 → 314.
  - gpu: direct 87 → 86, reach 99 → 96.
- **What is left on cuda and gpu is not test-only.** A type-checked scan of every test file lists the package-level
  declarations each file uses and where they are declared. Every other test file that does not import `decoder` is one of
  these:
  - a test of the backend package's production code;
  - a source scan of that package's files: `ptx_consistency`, `kernel_fma_lint`, `pipeline_lint`, and
    `kernel_local_memory`, which also checks `backend.go`;
  - a helper used by tests that load models;
  - a tower test.

  The kernel tests still in package `cuda` need production code moved first. `actgroup_kernel` and `ks_a1` feed
  `kso1c_kernels_test.go`, which reads a real GGUF through `decoder`. `gemm_mma` uses the resident's launch constants
  and `cudaResident`.
