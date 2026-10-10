//go:build !arm64

package decoder

// Architecture-conditional CPU decode defaults (see cpu_tuning_arm64.go). Both measured on the
// Ryzen 7 3700X, docs/measurements/cpu-decode-attribution-2026-09-22-linux.md.

// attnGroupedKernels: aikit ships the grouped acc64 attention kernels as NEON only (linalg/attn_acc64_group_other.go
// returns 0 blocks, so the whole group falls to the Go path). On every other architecture the "grouped" path is a
// pure-Go loop that is slower than the per-head path on the 1.5B's 6-heads-per-KV geometry. Off until a port exists.
var attnGroupedKernels = false

// activationFanoutEnabled: the 6-goroutine activation fan-out costs about 3x what the serial loop does per element on
// this box (goroutine wake stagger dwarfs a few thousand scalar silu calls). Serial.
var activationFanoutEnabled = false

// fusedGateUpDefault: the fused gate+up+SwiGLU fork/join (cpu_gateup_fused.go): one barrier per layer instead of two,
// with the activation folded into it. Bit-identical, and a gain on every measured model
// (docs/measurements/cpu-decode-roofline-2026-09-23.md). On.
const fusedGateUpDefault = true

// w4a8BatchDefault: R-06's one fork/join for q/k/v (weightmat.go). Bit-identical, and a small gain on every measured
// model on top of the fused gate+up (docs/tasks/task-cpu-decode-peer-gap-2026-09.md, L2). On.
const w4a8BatchDefault = true
