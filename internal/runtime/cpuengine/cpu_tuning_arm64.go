//go:build arm64

package cpuengine

// Architecture-conditional CPU decode defaults. Each is a var so a test can A/B it in-process;
// the value is the measured default for this architecture, not a heuristic.

// AttnGroupedKernels aikit's grouped acc64 attention kernels are NEON assembly
// (linalg/attn_acc64_group_arm64.s); on arm64 the grouped path runs them.
var AttnGroupedKernels = true

// ActivationFanoutEnabled the SwiGLU/GeGLU fan-out (parallelElementwise) was tuned here with
// silubench (mlp.go); kept on. Not re-measured with the DECODE SPLIT lines yet — see the Linux
// record for what to look at.
var ActivationFanoutEnabled = true

// FusedGateUpDefault the fused gate+up+SwiGLU fork/join (cpu_gateup_fused.go) was measured only on
// the amd64 Ryzen box. Off here until the Mac measures it — its weights are usually row4-repacked
// (which the fused path declines anyway), but an mmap'd .giw is canonical and would otherwise turn
// it on unmeasured. GOINFER_CPU_FUSED_GATEUP=1 forces it on for that measurement.
const FusedGateUpDefault = false

// W4A8BatchDefault the one fork/join for q/k/v (weightmat.go) is on for amd64 only. Here it read a
// null on the 7B (docs/measurements/w4a8-batch-7b-2026-09-20.md) and has not been measured on the current tree.
// Off until the Mac measures it; GOINFER_W4A8_BATCH=1 forces it on for that measurement.
const W4A8BatchDefault = false
