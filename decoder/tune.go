package decoder

import "github.com/townsendmerino/aikit/linalg"

// DefaultDecodeParallelThreshold is the matmul parallelism crossover (in MACs) for the
// int8 (W8A8) decode path: parallelize the per-token weight matmuls, leaving trivially
// small ops serial. Measured optimum on Apple M1 Pro / Qwen2.5-Coder-0.5B (~68 tok/s vs
// ~51 serial), independently re-confirmed on Ryzen 7 3700X (0.5B 22→36, 1.5B 10.5→15.1
// tok/s vs aikit's conservative 16.78M default).
//
// It is applied PER-WORKSPACE, automatically: newDecodeScratch sets it on the decode
// Workspace (matmulInto path) and matmul()'s free W8A8 branch sets it per-call. So every
// decode stream — library Load, serve, tests, future entry points (sidecar, c-archive
// FFI) — gets it without any startup call, and it is race-free across concurrent streams
// (unlike a process global). The int4 (W4A8) path has its OWN crossover, int4ParThreshold
// = 1<<20 (weightmat.go): the two values genuinely differ — measured separately, on
// different kernels/models, and a same-context sweep confirmed 300K is slightly better for
// small int8 while 1<<20 is the int4/gemma4 optimum. Unify only after a proper joint sweep.
// Hardware-specific (shifts with core count / memory latency); the M1 Pro is Phase 5's rig.
const DefaultDecodeParallelThreshold = 300_000

// SetDecodeParallelThreshold sets aikit/linalg's PROCESS-GLOBAL matmul crossover. It is NO
// LONGER needed for goinfer decode — that is automatic per-Workspace (see above). It remains
// as an optional escape hatch for aikit paths that do NOT go through goinfer's decode
// Workspace (e.g. a direct encoder/embedding call), or to force a global override in a
// measurement sweep. 0 parallelizes every matmul; a large value forces serial. Prefer NOT
// calling it — the per-Workspace default is race-free and needs no wiring.
func SetDecodeParallelThreshold(macs int) { linalg.SetParallelThreshold(macs) }
