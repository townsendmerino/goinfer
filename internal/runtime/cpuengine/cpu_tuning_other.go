//go:build !arm64

package cpuengine

// Architecture-conditional CPU decode defaults (see cpu_tuning_arm64.go). Both measured on the
// Ryzen 7 3700X, docs/measurements/cpu-decode-attribution-2026-09-22-linux.md.

// AttnGroupedKernels aikit ships the grouped acc64 attention kernels as NEON only (linalg/attn_acc64_group_other.go
// returns 0 blocks, so the whole group falls to the Go path). On every other architecture the "grouped" path is a
// pure-Go loop that is slower than the per-head path on the 1.5B's 6-heads-per-KV geometry. Off until a port exists.
var AttnGroupedKernels = false

// ActivationFanoutEnabled the 6-goroutine activation fan-out costs about 3x what the serial loop does per element on
// this box (goroutine wake stagger dwarfs a few thousand scalar silu calls). Serial.
var ActivationFanoutEnabled = false

// FusedGateUpDefault the fused gate+up+SwiGLU fork/join (cpu_gateup_fused.go): one barrier per layer instead of two,
// with the activation folded into it. Bit-identical, and a gain on every measured model
// (docs/measurements/cpu-decode-roofline-2026-09-23.md). On.
const FusedGateUpDefault = true

// W4A8BatchDefault one fork/join for q/k/v (weightmat.go). Bit-identical, and a small gain on every measured
// model on top of the fused gate+up. On.
const W4A8BatchDefault = true
