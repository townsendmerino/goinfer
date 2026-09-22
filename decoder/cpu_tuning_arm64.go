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
