- **`internal/hybridattn` (`internal/hybridattn/deltanet.go`, `decoder`, `metal/kernels`, `internal/layout/imports.json`, `docs/ARCHITECTURE.md`)**:
  Renamed `internal/program` to `internal/hybridattn` (`package hybridattn`) for the shared Gated DeltaNet and Qwen 3.5 q-gate operations used by `decoder` and the kernel tests, reserving `internal/program` for the §9 operation-list package.
