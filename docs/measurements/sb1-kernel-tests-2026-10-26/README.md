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
