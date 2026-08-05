//go:build cuda && goinfer_testhooks

package cuda

import (
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// TestCUDA_graphSharedMemHazard isolates the graph-replay-under-contention divergence to a KERNEL
// CLASS. The full-forward gate showed every segment diverges from live launches under concurrent GPU
// load; the bare vadd primitive (no shared memory) did not. The segments' common factor is kernels
// with dynamic shared memory + block reductions. This compares, for the SAME captured chain, graph
// replay vs live issue, for two kernel classes:
//   - elementwise (fScaleVec): dst[i] *= s — no shared memory, no reduction
//   - reduction  (fRms):       rmsnorm+quant — dynamic shared memory + a block reduction
//
// Run it under a concurrent GPU churn (another process). If the reduction chain diverges (graph !=
// live) while the elementwise chain stays identical, the hazard is graph replay of dynamic-shared-mem
// reduction kernels under a concurrent context — a primitive-level property, not the goinfer forward.
func TestCUDA_graphSharedMemHazard(t *testing.T) {
	const dir = "../testdata/gemma4-moe-tiny"
	mc, rf := loadG4MoECache(t, dir, false)
	defer mc.Close()
	r := rf.(*cudaResident)
	Ly := &r.layers[0]

	const K = 16
	seed := make([]float32, r.hidden)
	for i := range seed {
		seed[i] = float32(i%11) - 5
	}

	// compareChain captures `chain`, then runs it live (from seed) and via replay (from seed), and
	// reports whether the outputs of `readback` diverge. Returns (maxAbsDiff, mismatches).
	compareChain := func(name string, chain func() error, readback func() []float32) (float64, int) {
		var live, graphed []float32
		err := r.do(func() error {
			if e := gpu.Upload(r.x, seed); e != nil {
				return e
			}
			if e := chain(); e != nil {
				return e
			}
			if e := r.stream.Sync(); e != nil {
				return e
			}
			live = readback()

			g, e := r.stream.Capture(chain)
			if e != nil {
				return e
			}
			defer g.Close()
			if e := gpu.Upload(r.x, seed); e != nil {
				return e
			}
			if e := g.Replay(); e != nil {
				return e
			}
			if e := r.stream.Sync(); e != nil {
				return e
			}
			graphed = readback()
			return nil
		})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		maxAbs, bad := 0.0, 0
		for i := range live {
			d := float64(live[i] - graphed[i])
			if d < 0 {
				d = -d
			}
			if d > maxAbs {
				maxAbs = d
			}
			if live[i] != graphed[i] {
				bad++
			}
		}
		t.Logf("%s: %d/%d elems differ, maxAbsDiff=%.6g (graph vs live)", name, bad, len(live), maxAbs)
		return maxAbs, bad
	}

	// Elementwise chain: K in-place scales. No shared memory.
	elemChain := func() error {
		for i := 0; i < K; i++ {
			if e := r.launch(r.fScaleVec, LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
				Arg(r.x), gpu.ArgValue(float32(0.9999)), gpu.ArgValue(int32(r.hidden))); e != nil {
				return e
			}
		}
		return nil
	}
	readX := func() []float32 {
		h := make([]float32, r.hidden)
		_ = gpu.Download(r.x, h)
		return h
	}
	// Reduction chain: K rmsnorm+quant launches (dynamic shared memory + block reduction). Each writes
	// the int8 quant r.aq and scale r.aSc from r.x and Ly.preNorm; r.x is left unchanged, so the K
	// launches are independent-but-identical — the output r.aq is deterministic.
	redChain := func() error {
		for i := 0; i < K; i++ {
			if e := r.rms(r.x, Ly.preNorm, r.aq, r.aSc); e != nil {
				return e
			}
		}
		return nil
	}
	readAq := func() []float32 { // read the int8 quant output as raw words → floats for comparison
		h := make([]int32, r.hidden/4)
		_ = gpu.Download(r.aq, h)
		out := make([]float32, len(h))
		for i, v := range h {
			out[i] = float32(v)
		}
		return out
	}

	_, elemBad := compareChain("elementwise(fScaleVec)", elemChain, readX)
	_, redBad := compareChain("reduction(fRms)", redChain, readAq)

	// In isolation both are 0. The diagnostic value is running this under external GPU churn: if
	// redBad>0 while elemBad==0, the hazard is graph replay of shared-mem reduction kernels under a
	// concurrent context.
	t.Logf("SUMMARY: elementwise mismatches=%d, reduction mismatches=%d (run under churn to expose the class)", elemBad, redBad)
	if elemBad != 0 || redBad != 0 {
		t.Errorf("graph replay diverged from live: elementwise=%d reduction=%d", elemBad, redBad)
	}
}
