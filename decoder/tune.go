package decoder

import "github.com/townsendmerino/aikit/linalg"

// DefaultDecodeParallelThreshold is the matmul parallelism crossover (in MACs)
// that favors batch=1 decode: parallelize the per-token weight matmuls (the
// batched q/k/v + gate/up projections and the LM head), while leaving trivially
// small ops serial. Measured optimum on Apple M1 Pro / Qwen2.5-Coder-0.5B
// (~68 tok/s vs ~51 serial). It is a reasonable, OVERRIDABLE, hardware-specific
// value — the crossover shifts on x86/AVX2, fewer-core machines, and larger
// models — not a universal constant.
const DefaultDecodeParallelThreshold = 300_000

// SetDecodeParallelThreshold tunes aikit/linalg's matmul parallelism crossover
// for the whole process. Call it once at startup when the workload is
// single-token decode (the chat case). 0 parallelizes every matmul; a large
// value forces serial. goinfer owns this for its decode workload rather than
// relying on aikit's library default, which also serves the encoder / ken and
// must stay conservative.
func SetDecodeParallelThreshold(macs int) { linalg.SetParallelThreshold(macs) }
