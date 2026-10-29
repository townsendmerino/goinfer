- **SB1 step 1, Metal batch 6 (`internal/program`, `metal/kernels`, `docs/tasks/task-subsystems-2026-10.md` §8 and §9)**:
  the Gated DeltaNet and Qwen 3.5 query/gate CPU reference functions (`GatedDeltaNetStep`, `DeltaNetCore`,
  `DeltaNetRecurrence`, `L2NormScaledInto`, `L2NormScaled`, `NegExpAFromLog`, `SetDeltaCapHook`, `SplitQGate`,
  `QGateContext`) move out of `decoder` into `internal/program`, with `decoder` and `decoder/testhooks.go` calling
  `internal/program` so there is a single implementation (`decoder` bit-identical across its DeltaNet and Qwen 3.5 parity
  tests). `metal/deltanet_test.go` and `metal/qgate_kernels_test.go` move to `metal/kernels` against `internal/program`,
  dropping their `decoder` import and `goinfer_testhooks` tag (5 PASS before and after; `metal` ratchet direct 245 → 243,
  reach 330 → 328).
