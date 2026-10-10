package decoder

import "github.com/townsendmerino/aikit/linalg"

// DefaultDecodeParallelThreshold is the matmul parallelism crossover (in MACs) for the int8 (W8A8) decode path:
// parallelize the per-token weight matmuls, leaving trivially small ops serial. It is hardware-specific (it shifts with
// core count and memory latency); it was measured on Apple M1 Pro and Ryzen 7 3700X, and
// BenchmarkInt8ParThresholdSweep re-checks it.
//
// It is applied per workspace, automatically: newDecodeScratch sets it on the decode Workspace (matmulInto path) and
// matmul()'s free W8A8 branch sets it per call. So every decode stream (library Load, serve, tests, future entry
// points) gets it without any startup call, and it is race-free across concurrent streams (unlike a process global).
// The int4 (W4A8) path has its own crossover, int4ParThreshold = 1<<20 (weightmat.go): the two values differ because
// they were measured separately, on different kernels and models. Unify only after a proper joint sweep.
const DefaultDecodeParallelThreshold = 300_000

// SetDecodeParallelThreshold sets aikit/linalg's PROCESS-GLOBAL matmul crossover. It is NO
// LONGER needed for goinfer decode — that is automatic per-Workspace (see above). It remains
// as an optional escape hatch for aikit paths that do NOT go through goinfer's decode
// Workspace (e.g. a direct encoder/embedding call), or to force a global override in a
// measurement sweep. 0 parallelizes every matmul; a large value forces serial. Prefer NOT
// calling it — the per-Workspace default is race-free and needs no wiring.
func SetDecodeParallelThreshold(macs int) { linalg.SetParallelThreshold(macs) }
