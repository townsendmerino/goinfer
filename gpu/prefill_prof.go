//go:build gpu

package gpu

import "time"

// Per-category wall time for the batched prefill path (docs/measurements/webgpu-prefill-decomp-2026-09-22.md), mirroring
// cuda/prefill.go's profTic/profToc: same shape, same accepted trade-off (category boundaries are syncs, so the category sum
// runs a bit over the pipelined wall time, the price of per-kernel attribution). Four categories (the GEMM, the attention,
// the batched norms/rope and the KV write), one more than CUDA's three (gemv/attn/glue).
// History: docs/code-notes/gpu.md#prefillProf.
type prefillProf struct {
	gemm, attn, normsRope, kvWrite time.Duration
}

type prefillProfCat int

const (
	profGemm prefillProfCat = iota
	profAttn
	profNormsRope
	profKVWrite
)

// prefillProfTic syncs the device and returns a start time when profiling is on (c.prefillProf != nil);
// a no-op zero Time otherwise — one nil check, no allocation, matching cudaResident's profTic exactly.
func (c *Context) prefillProfTic() time.Time {
	if c.prefillProf == nil {
		return time.Time{}
	}
	c.device.Poll(true, nil)
	return time.Now()
}

// prefillProfToc syncs the device and adds the elapsed time to the named category (no-op when profiling off).
func (c *Context) prefillProfToc(cat prefillProfCat, t0 time.Time) {
	if c.prefillProf == nil {
		return
	}
	c.device.Poll(true, nil)
	d := time.Since(t0)
	switch cat {
	case profGemm:
		c.prefillProf.gemm += d
	case profAttn:
		c.prefillProf.attn += d
	case profNormsRope:
		c.prefillProf.normsRope += d
	case profKVWrite:
		c.prefillProf.kvWrite += d
	}
}
