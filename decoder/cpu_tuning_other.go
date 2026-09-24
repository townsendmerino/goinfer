//go:build !arm64

package decoder

// Architecture-conditional CPU decode defaults (see cpu_tuning_arm64.go). Both measured on the
// Ryzen 7 3700X, docs/measurements/cpu-decode-attribution-2026-09-22-linux.md.

// attnGroupedKernels: aikit ships the grouped acc64 attention kernels as NEON only
// (linalg/attn_acc64_group_other.go returns 0 blocks — the whole group falls to the Go path), so
// on every other architecture the "grouped" path is a pure-Go loop that measured 1.18× (depth
// 128) to 2.86× (depth 4096) SLOWER than the per-head path on the 1.5B's 6-heads-per-KV
// geometry. Off until a port exists.
var attnGroupedKernels = false

// activationFanoutEnabled: the 6-goroutine activation fan-out costs ~3× what the serial loop
// does per element on this box (goroutine wake stagger dwarfs 9-19k scalar silu calls) — 1.5B
// 13.5 → 3.9 ms/token, 7B 27.9 → 8.0. Serial.
var activationFanoutEnabled = false

// fusedGateUpDefault: the fused gate+up+SwiGLU fork/join (cpu_gateup_fused.go) — one barrier per
// layer instead of two, with the activation folded into it. Paired ABBA on the Ryzen 7 3700X,
// bit-identical: 1.5B 1.066×, 0.5B 1.113×, 7B 1.029× against the unfused path
// (docs/measurements/cpu-decode-roofline-2026-09-23.md). On.
const fusedGateUpDefault = true
