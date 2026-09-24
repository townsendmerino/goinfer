//go:build arm64

package decoder

// Architecture-conditional CPU decode defaults. Each is a var so a test can A/B it in-process;
// the value is the measured default for this architecture, not a heuristic.

// attnGroupedKernels: aikit's grouped acc64 attention kernels (R13) are NEON assembly
// (linalg/attn_acc64_group_arm64.s); on arm64 the grouped path runs them.
var attnGroupedKernels = true

// activationFanoutEnabled: the SwiGLU/GeGLU fan-out (parallelElementwise) was tuned here with
// silubench (mlp.go); kept on. Not re-measured with the DECODE SPLIT lines yet — see the Linux
// record for what to look at.
var activationFanoutEnabled = true

// fusedGateUpDefault: the fused gate+up+SwiGLU fork/join (cpu_gateup_fused.go) was measured only on
// the amd64 Ryzen box. Off here until the Mac measures it — its weights are usually row4-repacked
// (which the fused path declines anyway), but an mmap'd .giw is canonical and would otherwise turn
// it on unmeasured. GOINFER_CPU_FUSED_GATEUP=1 forces it on for that measurement.
const fusedGateUpDefault = false
