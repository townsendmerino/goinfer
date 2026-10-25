# SP2 — speculation on a recurrent hybrid: Qwen3.5-9B with DFlash on nobara (2026-10-24)

SP2 of track SP. The gates were registered before any code, in `docs/tasks/task-forward-architecture-2026-10.md` §6a, "SP2 on
nobara" (commit `7d5a96cf`, pushed before the first line of SP2 code). This page records each reading against that rule.

- **The mechanism:** `cuda/recurrent_spec.go` (the arena, `VerifyLastNArgmax`, `FoldVerified`), the twins in `cuda/deltanet.cu`
  (`delta_conv_rows_spec`, `delta_rule_rows_128_spec`, `delta_rule_rows_spec`), and the loop in `decoder/blockspec.go`
  (`BlockSpec.generate`, `Model.specRollbackSafeVia`).
- **The gates:** tier 0 in `cuda/recurrent_spec_test.go` and `decoder/recurrent_spec_test.go`; the 9B in
  `cuda/recurrent_spec_real_test.go`, driven by `run-sp2.sh` (`MODE=identity|cost|speed`).
- **Logs:** `~/goinfer-logs/sp2-2026-10/` on nobara; the JSON lines each 9B run writes are copied under `raw/`.

## The kernels: what changed in `deltanet.ptx`

Rebuilt with `cuda/build_ptx.sh deltanet` (NVRTC 12.9.86; the committed PTX first rebuilt byte-identically at main `d9a1cb4d`,
so the toolchain reproduces it). Every pre-existing kernel's floating-point instruction sequence is unchanged; the PTX of the
row-batched kernels differs only in labels, symbol names and address arithmetic order. Each verify twin's floating-point
sequence equals its committing kernel's (`delta_conv_rows_spec` = `delta_conv_rows`, `delta_rule_rows_128_spec` =
`delta_rule_rows_128`). None of the three new kernels uses local memory.

## Readings

(filled as each gate runs)
