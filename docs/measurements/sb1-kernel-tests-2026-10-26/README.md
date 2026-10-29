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

# metal batch 5 (2026-10-28, 09:51–10:00 PDT, mac-ax01, on battery)

The gates are registered in `docs/tasks/task-subsystems-2026-10.md`, "SB1 step 1 — metal batch 5". Box: the MacBook (M1 Pro,
Metal), Go 1.27.3, on battery (72% → 67%, the house without power): correctness runs only, `go test -p 2`, nothing timed.
Another session (mac-vk5grade) was running Metal tests at the same time.

- **What moved:** `moe_test.go`, `profile_test.go` and `sa_qv_fusion_test.go`, from `metal/` to `metal/kernels/`, 7 `Test`
  functions, with nothing changed but the package clause. `clampI`, declared in `moe_test.go` and used by `gemma_test.go`,
  which stays, is copied next to its user.
- **What stayed:** `prefill_gemm_s2_test.go`, the fourth candidate. Its `gemmS2Kernels` (429 lines of MSL) is also compiled
  by `prefill_decomp_test.go`, so moving it would mean copying a kernel library, not a helper.
- **Runs** (`-v`, tags `goinfer_testhooks`): before, in `./metal`, 8 PASS (`batch5-before.log`, 4.47 GB reclaimable at the
  start; an earlier run at 3.996 GB is kept as `batch5-before-void.log`, void by the 4 GB rule though every test passed);
  after, the 7 moved tests in `./metal/kernels`, 7 PASS (`batch5-after.log`, 5.82 GB), and `TestS2KernelsCompile`, which
  stayed, PASS in `./metal`.
- **Checks:** vet of `metal` and `metal/kernels` with and without `goinfer_testhooks`; `go list -deps -test ./metal/kernels/`
  names no `goinfer/decoder` under either tag set; CI's staticcheck (canary U1000) clean on both packages; `touched.py`
  green (the metal suite 103 s, `metal/kernels` 173 s).
- **Ratchet:** metal reach 333 → 330, direct 245 unchanged (none of the three imported `decoder`); the SB2 census
  exemptions unchanged.
- **What is left on metal:** of the 245 files importing `decoder`, 231 load a model. Two kernel tests,
  `deltanet_test.go` and `qgate_kernels_test.go`, use `decoder`'s own production code as their CPU reference (through
  `decoder/testhooks.go`), so they move once that code leaves `decoder`. The rest of the 88 that reach `decoder` only by
  being in package `metal` are shared helpers for tests that stay, tests needing package `metal`'s backend or tower
  registration, or tests of `metal`'s production code.

# metal batch 7: the Mamba-2 CPU reference out of decoder (2026-10-29, 07:21–07:40 PDT, mac-ax01)

The gates are registered in `docs/tasks/task-subsystems-2026-10.md`, "SB1 step 1 — metal batch 7", at `e31ba9db` before the
code. Box: the MacBook (M1 Pro, Metal), on AC, Go 1.27.3; correctness runs only (`go test -p 2`, holding the timing lock
shared), nothing timed. Branch `sb1-metal-batch7-2026-10-29`, off `rename-hybridattn-2026-10-28` (#421) at `899213bb`.

- **What moved out of `decoder`:** the Mamba-2 mixer between its projections (the causal conv, the selective scan, the gated
  RMSNorm) into `internal/mamba2` (`Params`, `Weights`, `State`, `NewState`, `Mix`, `GatedRMSNormGrouped`), importing only the
  standard library and `internal/hybridattn`'s `Silu` and `Softplusf`. `decoder/mamba2.go:mamba2Step` keeps `in_proj` and
  `out_proj`, the int8 seams and the capture hook, and calls `mamba2.Mix` with `ssmForceF32` as its `f32` argument;
  `mamba2State` is an alias of `mamba2.State`, so the cache reset in `decoder/kvcache.go` names the exported fields.
  `GatedRMSNormGroupedForTest`, `SetMambaCapHook` and `SetSSMForceF32` keep their signatures for `cuda` and `gpu`.
- **What moved to `metal/kernels`:** `TestMambaGNorm_groups` (now against `mamba2.GatedRMSNormGrouped`) and
  `TestActQuant_relu2`, in `metal/kernels/mamba_kernels_test.go`, untagged. `mambaCos` is copied there; the chain tests
  (`TestMambaKernels_cpuParity`, `TestMambaKernels_plantsAreRed`) stay in `metal/mamba_kernels_test.go`, which still loads
  the tiny fixtures through `decoder` for their real per-token inputs.
- **Gate 1, `decoder` bit-identical:** a scratch test (`metal-logs/batch7-hash-scratch_test.go.txt`, not committed) hashed
  `mamba2Seq` on seeded random weights at three geometries with `ssmForceF32` off and on, the grouped norm at each, and the
  logits of `nemotron-tiny` and `granite-tiny` at f32 and int8 over 24 seeded tokens, a `TruncateTo(0)` and 8 more: all
  16 hashes equal before and after. `TestMamba2_parity` (`cosine=1.00000000 max|Δ|=2.98e-07`), `TestGranite_textParity`,
  `TestNemotron_textParity`, the four `TestNemotron3NanoMoE_*`, the four `TestB27_*` that run (`TestB27_cpuFidelitySlice`
  skips for want of its slice, both sides) and `TestTruncateTo_resetsRecurrent` pass before and after with the same figures.
- **Gate 2, the metal tests:** 4 PASS before in `./metal/` (`-tags goinfer_testhooks`, `batch7-before.log`); after, the two
  chain tests PASS in `./metal/` and the two moved tests PASS in `./metal/kernels/` tagged and untagged, every logged figure
  equal (chain worst cosine `1.000000000`, max |diff| `1.19e-07` and `7.15e-07`; plants as before, including granite-tiny's
  logged non-red softplus plant; gated norm `2.38e-07`, its plant `0.792793`; act_quant max |diff| `0`).
- **Void cells, re-run:** the before-run's first attempt started at 1.77 GB reclaimable and was discarded; the after-run's
  `decoder` and `metal` cells started at 3.78 and 3.53 GB (`batch7-after-void.log`; its `metal/kernels` cells, at 4.17 and
  4.01 GB, stand) and were re-run at 4.67 and 4.44 GB (`batch7-after-rerun.log`) with the same outcome, hashes and figures.
- **Checks:** build; vet of the root module (untagged, `goinfer_testhooks`, `realckpt`), `metal`, `gpu` (`gpu
  goinfer_testhooks`) and `cuda` (`cuda goinfer_testhooks`, `GOOS=linux`); CI's pinned `staticcheck` on `decoder`,
  `internal/mamba2`, `internal/layout` and `metal` with and without `goinfer_testhooks` (canary red first; it found
  `mamba2Params.convDim` unused, removed); `go list -deps -test ./metal/kernels/` names no `goinfer/decoder`;
  `internal/layout` green with three new arrows (`decoder → internal/mamba2`, `internal/mamba2 → internal/hybridattn`, test
  `metal/kernels → internal/mamba2`); `scripts/touched.py` green. Parity `deps_hash` refreshed through
  `scripts/refresh_parity_hashes.sh` (34 goldens passed, 0 failed).
- **Ratchet:** unchanged, as registered (metal direct 243, reach 328): `metal/mamba_kernels_test.go` still imports
  `decoder`. Moving its chain tests needs their inputs from somewhere other than a `decoder` forward.

# metal batch 6: DeltaNet and q-gate CPU references out of decoder (2026-10-28, 17:47–18:09 PDT, agy-sb1)

The gates are registered in `docs/tasks/task-subsystems-2026-10.md`, "SB1 step 1 — metal batch 6", at `1214adf5` before the
code. Box: the MacBook (M1 Pro, Metal), Go 1.27.3; correctness runs only (`go test -p 2`, holding the timing lock shared),
nothing timed.

- **What moved out of `decoder`:** the CPU reference functions for Gated DeltaNet (`GatedDeltaNetStep`, `DeltaNetCore`,
  `DeltaNetRecurrence`, `L2NormScaledInto`, `L2NormScaled`, `NegExpAFromLog`, `SetDeltaCapHook`, `HasDeltaCapHook`) and the
  Qwen 3.5 interleaved query/gate split and sigmoid gate (`SplitQGate`, `SplitQGateAlloc`, `QGateContext`) moved to
  `internal/hybridattn` (`internal/hybridattn/deltanet.go`, importing only the standard library and `aikit/linalg`).
  `decoder/deltanet.go`, `decoder/deltanet_n.go`, `decoder/forward_qwen35.go`, and `decoder/testhooks.go`
  call `internal/hybridattn` so there is one implementation; `decoder`'s `*ForTest` hooks stay for `cuda` and `gpu`.
- **What moved to `metal/kernels`:** `metal/deltanet_test.go` (`TestDeltaNetKernels_cpuParity`, `TestDeltaNorm_zeroHead`,
  `TestDeltaNetKernels_mutations`) and `metal/qgate_kernels_test.go` (`TestQGateKernels_cpuParity`,
  `TestQGateKernels_mutations`) moved to `metal/kernels/`, calling `internal/hybridattn` directly and dropping their `decoder`
  import and `goinfer_testhooks` build constraint. `uploadF32` and `cosF32`, declared in `deltanet_test.go` and used by
  tests that stay in `metal/`, are copied to `metal/mamba_kernels_test.go` (as `clampI` was in batch 5).
- **Runs:**
  - `decoder` parity tests (`TestGatedDeltaNet_parity`, `TestDeltaNetCoreN_bitIdentical`, `TestQwen35_forwardParity`,
    `TestQwen3_5_textParity`, `TestQwen3Next_textParity`, `TestOlmoHybrid_forwardParity`): 6 PASS before and after,
    bit-identical (`TestGatedDeltaNet_parity` `maxAbs=2.09e-07 cosine=1.00000000`, `TestOlmoHybrid_forwardParity`
    `cosine=0.9999999999996838`, `TestQwen35_forwardParity` greedy tokens `[162 127 171 212 161 226]`).
  - Metal kernel tests (`-v`, `-p 2`): before in `./metal` with `-tags goinfer_testhooks`, 5 PASS (`batch6-before.log`,
    4.52 GB reclaimable, 58% battery); after in `./metal/kernels`, 5 PASS with `-tags goinfer_testhooks` (`batch6-after.log`,
    4.03 GB reclaimable, 54% charging) and 5 PASS untagged (`batch6-after-untagged.log`, 5.62 GB reclaimable, 54% charging),
    with identical parity figures (`delta_conv` `1.18e-06`, `delta_gates` `2.44e-07`, `delta_rule` `8.16e-06`, `delta_gnorm`
    `2.24e-05`, `qgate` `2.1e-07`).
- **Checks:** build, vet, and `staticcheck` across the root module and `metal` module with and without `goinfer_testhooks`;
  `go list -deps -test ./metal/kernels/` names no `goinfer/decoder`; `internal/layout` tests (`TestLayout_arrows`,
  `TestBackendTests_decoderImportsOnlyFall`, `TestPackageDocs_namedTestsExist`, `TestArchitecture_importAppendix`) all green.
- **Ratchet:** metal direct 245 → 243, reach 330 → 328; the SB2 census exemptions unchanged.

# cuda batch 5: `cuda/deltanet_test.go` against the DeltaNet CPU references (2026-10-29, nobara-sb01)

- **Registered** in `docs/tasks/task-subsystems-2026-10.md` ("SB1 step 1 — cuda batch 5") at `b9d4fe1cc`, before the code.
- **Box:** nobara, RTX 2070 SUPER, driver 595.91.07. Correctness runs only, each holding the timing lock shared; the card's
  compute contexts are recorded before each run.
- **What moved:** `cuda/deltanet_test.go` (`TestDeltaNetKernels_cpuParity`, `TestDeltaNorm_zeroHead`) to `cuda/kernels`. It
  calls the references directly (`NewDeltaNet`, `SetDeltaCapHook`, `DeltaNetStep`, `L2NormScaled`) instead of `decoder`'s
  `*ForTest` forwards, and drops its `decoder` import and `goinfer_testhooks` constraint. `cosF32`, used by 19 tests that
  stay in `cuda/`, is copied to `cuda/mamba_kernels_test.go` (same build constraint).

| run | card at start | result |
|---|---|---|
| before, `./cuda/` (`cuda goinfer_testhooks`) | desktop compositor only (`batch5-before-card.txt`) | 2 PASS (`batch5-before.log`) |
| after, `./cuda/kernels/` (`cuda`, then `cuda goinfer_testhooks`) | another session's `cuda.test`, 2414 MiB (`batch5-after-void1-card.txt`) | **VOID** by the registered rule; 2 PASS each with the before-run's figures (`batch5-after-void1.log`, `batch5-after-testhooks-void1.log`) |
| after, `./cuda/kernels/` (`cuda`), 09:32 PDT, against `internal/hybridattn` | desktop compositor only (`batch5-after-card.txt`) | 2 PASS (`batch5-after.log`) |
| after, `./cuda/kernels/` (`cuda goinfer_testhooks`), 09:32 PDT | desktop compositor only (`batch5-after-testhooks-card.txt`) | 2 PASS (`batch5-after-testhooks.log`) |

- **Outcome:** the same for both moved tests on every valid run, and both names are listed before and after.
- **Per-stage worst figures, the same before and on both valid after-runs:** `delta_conv` cosine 1.000000000, maxAbs/rms `2.37e-06`; `delta_gates` `2.42e-07`;
  `delta_rule` `7.07e-06`; `delta_gnorm` `2.22e-05`; `delta_norm` clean over 16 heads including the zero head.
- **Checks:** `go vet` and `staticcheck` (the canary fires U1000) pass on `./cuda/` and `./cuda/kernels/` under no tag and
  `cuda goinfer_testhooks`; under `cuda` alone staticcheck reports five U1000 findings in files this batch does not touch
  (`cost.go`, `drain_marker_test.go`, `kvpool_helpers_test.go`). `go list -deps -test -tags cuda ./cuda/kernels/` names no
  `goinfer/decoder`. `internal/layout`'s tests pass with the lowered ratchet.
- **Ratchet:** cuda direct 246 → 245, reach 314 → 313; the SB2 census exemptions unchanged.

# cuda batch 6: the activation-group kernel tests (2026-10-29, nobara-sb01)

- **Registered** in `docs/tasks/task-subsystems-2026-10.md` ("SB1 step 1 — cuda batch 6") at `5bae03752`, before the code. The
  `kquantGemv` identity hash was written at `345184685`, the tree before the move, and the before-run ran in a detached worktree
  at that commit.
- **Box:** nobara, RTX 2070 SUPER, driver 595.91.07. Correctness runs only, each holding the timing lock shared, all at 09:32
  PDT with only the desktop compositor on the card (`batch6-*-card.txt`). `GOINFER_KSO1C_OLD_ACTGROUP_PTX` pointed at
  `cuda/testdata/actgroup.ptx` on every run, so the untouched-kernels test ran instead of skipping.
- **What moved:**
  - `kquantGemvRows` (`cuda/resident.go`) to `cuda/internal/launchgeom` as `KQuantGemvRows`, with `KQuantGemvGridX`.
  - `cuda/actgroup_kernel_test.go`, `cuda/ks_a1_test.go` and `cuda/kso1c_kernels_test.go` to `cuda/kernels`. Each reads its
    PTX from `../testdata`, packs with `wpack` and reads `launchgeom`. The GGUF reference test opens the file with
    `embed.OpenGGUFSplitMmap`, the call `decoder.OpenGGUFMmap` makes for a single-file GGUF. `kso1cPlanes` is kept in
    `kso1c_resident_test.go`, which stays. `kso1cModels`, `kso1cLoad1` and `modelPath` are copied to `cuda/kernels`, and the
    moved `f16tof32` is dropped for the identical one already there.

| run | result |
|---|---|
| before, `./cuda/` (`cuda goinfer_testhooks`), the 9 by name | 8 PASS, `TestKSA1_gemvTiming` SKIP (`set GOINFER_KSA1_TIMING=1`) (`batch6-before.log`) |
| after, `./cuda/kernels/` (`cuda`) | the same (`batch6-after.log`) |
| after, `./cuda/kernels/` (`cuda goinfer_testhooks`) | the same (`batch6-after-testhooks.log`) |

- **Outcome:** every test has the same outcome and the same logged lines on all three runs: O1C-G1 worst `|err|/Σ|w·a|`
  Q5_K `1.51e-07`, Q6_K `1.44e-07`, every plant red; 25 untouched cells byte-equal; the 0.5B (12 Q6_K tensors) and Phi-3
  (32 Q5_K, 17 Q6_K) within 1e-6 of aikit.
- **launchgeom:** `TestKQuantGemv_matchesCuDefine` fails under a plant (`KQuantGemvRows = 2`: "actgroup.cu defines KQ_ROWS 1;
  launchgeom has 2", `batch6-launchgeom-plant.log`) and passes on the revert (`batch6-launchgeom-after.log`).
  `TestLaunchGeometry_identity` holds all 9 hashes after the move (`batch6-launchgeom-identity.log`).
- **Checks:** `go vet` and `staticcheck` are clean on `./cuda/`, `./cuda/kernels/` and `./cuda/internal/launchgeom/` under no
  tag and under `cuda goinfer_testhooks`. Under `cuda` alone, staticcheck flags only the pre-existing U1000 findings in files
  this batch does not touch. `cuda/kernels`' test build reaches no `decoder`, and `internal/layout` passes.
- **Ratchet:** cuda direct −1, reach −3: 245/313 → 244/310 on top of cuda batch 5.

# cuda batch 7: `gemm_mma`'s kernel test and the f16 KV converter's test (2026-10-29, nobara-sb01)

- **Registered** in `docs/tasks/task-subsystems-2026-10.md` ("SB1 step 1 — cuda batch 7") at `3875a7cf2`, before the code. The
  `gemmMMA` identity hash was written on the tree before the move, and the before-run ran on that tree.
- **Box:** nobara, RTX 2070 SUPER, driver 595.91.07. Each GPU run held the timing lock shared and started with only the desktop
  compositor on the card (`batch7-before-card.txt`, `batch7-after-card.txt`).
- **What moved:** gemm_w4a8_mma's tile geometry to `cuda/internal/launchgeom` (`GemmMMABM`, `GemmMMABN`, `GemmMMAThreads`,
  `GemmMMAKStep`, `GemmMMAAPad`, `GemmMMAShmem`, `GemmMMAGrid`); `TestGemmMMA_vsExact` to `cuda/kernels`; `TestF32ToF16RNE_ties`
  to `cuda/internal/wpack`. The two selector tests stay in `cuda/gemm_mma_test.go`.

| run | result |
|---|---|
| before, `./cuda/` (`cuda`), the 4 by name | 4 PASS (`batch7-before.log`) |
| after, `TestGemmMMA_vsExact` in `./cuda/kernels/` (`cuda`) | PASS (`batch7-after-kernels.log`) |
| after, the two selector tests in `./cuda/` (`cuda`) | 2 PASS (`batch7-after-cuda.log`) |
| after, `TestF32ToF16RNE_ties` in `./cuda/internal/wpack/` (untagged) | PASS (`batch7-after-wpack.log`) |

- **Outcome:** the same for every test. `TestGemmMMA_vsExact`'s 24 logged lines (six shapes × four M) are identical before and
  after; the worst relative delta is `1.107e-06` against the registered `1e-5`.
- **launchgeom:** `TestGemmMMA_matchesCuDefines` is red under a `GKSTEP` plant (`batch7-launchgeom-plant.log`) and, after
  Gemini's review, under a 129-thread plant that the first form (`GemmMMAThreads/32`) let through
  (`batch7-launchgeom-plant-threads.log`); green after (`batch7-launchgeom-after.log`). `TestLaunchGeometry_identity` holds
  all 10 hashes after the move (`batch7-launchgeom-identity.log`).
- **Checks:** vet and staticcheck clean on `./cuda/`, `./cuda/kernels/`, `./cuda/internal/launchgeom/` and
  `./cuda/internal/wpack/` under no tag and `cuda goinfer_testhooks`; the destinations reach no `decoder`;
  `scripts/test_touched.py` and `internal/layout` pass.
- **Ratchet:** cuda 244/310 → 244/309 (`kvhalf_test.go` leaves; `gemm_mma_test.go` stays).

# gpu batch 5: `gpu/deltanet_test.go` against the DeltaNet CPU references (2026-10-29, nobara-sb01)

- **Registered** in `docs/tasks/task-subsystems-2026-10.md` ("SB1 step 1 — gpu batch 5") at `533fc4101`, before the code; the
  before-run ran in a detached worktree at that commit.
- **Box:** nobara, RTX 2070 SUPER through Vulkan, driver 595.91.07. Each run held the timing lock shared and started with only
  the desktop compositor on the card (`gpu5-*-card.txt`), at 13:37 PDT.
- **What moved:** `gpu/deltanet_test.go` (`TestDeltaRule_cpuParity`, `TestDeltaNorm_cpuParity`) to `gpu/internal/kernels`,
  calling `internal/hybridattn`'s `NewDeltaNet`, `SetDeltaCapHook`, `DeltaNetStep` and `L2NormScaled` instead of `decoder`'s
  `*ForTest` forwards, and dropping its `decoder` import and `goinfer_testhooks` constraint. No production code moved.

| run | result |
|---|---|
| before, `./gpu/` (`gpu goinfer_testhooks`) | 2 PASS (`gpu5-before.log`) |
| after, `./gpu/internal/kernels/` (`gpu`) | 2 PASS (`gpu5-after.log`) |
| after, `./gpu/internal/kernels/` (`gpu goinfer_testhooks`) | 2 PASS (`gpu5-after-testhooks.log`) |

- **Outcome:** every logged line is the same on all three runs, apart from the reference's name in one log label: per-step
  cosines 1.000000000; worst `deltaGates` `2.31e-07`, `deltaRule` `6.03e-06`, `deltaGNorm` `1.32e-05` (maxAbs/rms);
  `deltaNorm` clean over 16 heads.
- **Checks:** the moved test compiles under `gpu` and `gpu goinfer_testhooks`; vet passes untagged and under both; staticcheck
  is clean under `gpu goinfer_testhooks` (under `gpu` alone it reports only U1000 findings in files this batch does not
  touch); `gpu/internal/kernels`' test build reaches no `decoder`; `internal/layout` passes with the regenerated
  `imports.json` and `ARCHITECTURE.md` appendix.
- **Ratchet:** gpu 86/96 → 85/95.


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
| after, `./gpu/internal/kernels/` (`gpu goinfer_testhooks`) | 4 PASS (`batch4-gpu-after.log`) |

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

## Batch 4: review fixes (Gemini on #360)

- **One source for `gemv_w4a8_rn`'s launch geometry.** RN, MT and the block size were written down in three places: production's
  `rnBlockRows` and literal `256`/`8` in `bGemvB`, and an `rnCfg` copy in each of `gemm_mma_test.go` and `cuda/kernels`'
  `gemv_rn_test.go`. They now live in `cuda/internal/launchgeom` as `GemvRNRows`, `GemvRNMT`, `GemvRNThreads` and
  `GemvRNGridX`, and production and both tests read them.
  - Production's grid is unchanged: `(ceil(N/2)+7)/8` is `GemvRNGridX` with 256/32 = 8 warps per block.
  - `launchgeom`'s `TestGemvRN_matchesCuDefines` holds RN and MT to the `#define` defaults in `gemv_w4a8_rn.cu`. Setting
    `GemvRNRows` to 3 turned it red.
  - `TestLaunchGeometry_identity` gains a `gemvRN` hash, and every existing hash is unchanged.
  - No sweep script in the tree edits these defines; the "the sweep seds both" note referred to a run that is not committed.
- **One Gumbel reference.** `internal/sampling` gains `GumbelDraw` and `GumbelKeyAt`. decoder's `GumbelDrawForTest` and
  `GumbelKeyForTest` forward to them, and the moved gpu test calls them directly, so there is no copy.
- **Device runs on the fixed tree** (nobara, compositor-only card):
  - `TestGemmMMA_vsExact`, `TestGemmMMA_selectorKeepsExactBelowFloor`, `TestLaunchGeometry_identity`,
    `TestGemvRN_bitIdentical` and gpu's `TestGumbelDeviceAgreesWithHost` pass.
  - With `GOINFER_HEAVY_TESTS=1`, cuda's `TestGumbelDeviceAgreesWithHost` (through decoder's hook) passes, as do
    `TestGemma2Softcap_prefillBitIdentical`, `TestPrefillChunked_fastKernelsOnEveryChunk` and
    `TestHiddenLastResidentParityCUDA`, which run production's batched prefill.

