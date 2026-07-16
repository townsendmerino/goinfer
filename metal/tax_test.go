//go:build darwin

package metal

import (
	"testing"
	"time"
)

// TestLayerA_bindingTax isolates the per-token binding tax — the Metal analog of the
// CUDA spike's ~5 µs channel-hop. Two numbers matter for the decode loop:
//   - the per-command-buffer round-trip (commit + waitUntilCompleted), paid ONCE per
//     token if a token's whole layer stack is encoded into one command buffer;
//   - the marginal per-encoded-dispatch cost (msgSend to encode one more dispatch).
//
// The doc's warning: "Metal's encoder/commit overheads are different animals — measure."
func TestLayerA_bindingTax(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	const src = `
#include <metal_stdlib>
using namespace metal;
kernel void nop(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) { out[0] = out[0]; }`
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "nop")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	q := d.NewCommandQueue()
	buf := d.NewBufferLen(1)

	best := func(reps int) time.Duration {
		for w := 0; w < 5; w++ { // warm
			q.Run1DBatch(pipe, 1, 1, reps, buf)
		}
		b := time.Hour
		for it := 0; it < 50; it++ {
			t0 := time.Now()
			q.Run1DBatch(pipe, 1, 1, reps, buf)
			if dt := time.Since(t0); dt < b {
				b = dt
			}
		}
		return b
	}

	t1 := best(1)
	t64 := best(64)
	marginal := (t64 - t1) / 63

	t.Logf("per-command-buffer round-trip (1 dispatch, commit+wait): %.1f µs", float64(t1.Nanoseconds())/1e3)
	t.Logf("64 dispatches in one command buffer:                    %.1f µs", float64(t64.Nanoseconds())/1e3)
	t.Logf("marginal per-encoded-dispatch (msgSend):                %.2f µs", float64(marginal.Nanoseconds())/1e3)
	t.Logf("=> a token = 1 command buffer, 1 commit+wait (~%.0f µs fixed) + N dispatches "+
		"(~%.1f µs each) — the commit/wait is amortized over the whole token, not per layer",
		float64(t1.Nanoseconds())/1e3, float64(marginal.Nanoseconds())/1e3)
}
