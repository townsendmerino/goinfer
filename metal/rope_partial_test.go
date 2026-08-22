//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestRopePartial — validates the production (parameterized) rope kernel with PARTIAL rotary
// (rhalf = rotaryDim/2 < hd/2, Phi): dims [0,2*rhalf) rotate as pairs (d, rhalf+d); the tail
// [2*rhalf, hd) must pass through UNCHANGED. Matches decoder/rope.go applyRoPE.
func TestRopePartial(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "rope")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	const nH, hd, pos = 4, 128, 41
	const rotaryDim = 96 // partial: rotate the first 96 dims, leave [96,128)
	rhalf := rotaryDim / 2
	rng := rand.New(rand.NewSource(13))
	x := make([]float32, nH*hd)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	invf := make([]float32, rhalf)
	for i := range invf {
		invf[i] = float32(1.0 / math.Pow(10000, float64(2*i)/float64(rotaryDim)))
	}
	// CPU reference: rotate (d, rhalf+d) for d<rhalf per head; tail untouched.
	ref := append([]float32(nil), x...)
	for h := range nH {
		b := h * hd
		for dd := range rhalf {
			th := float64(pos) * float64(invf[dd])
			c, s := math.Cos(th), math.Sin(th)
			x0, x1 := float64(ref[b+dd]), float64(ref[b+rhalf+dd])
			ref[b+dd] = float32(x0*c - x1*s)
			ref[b+rhalf+dd] = float32(x0*s + x1*c)
		}
	}
	q := d.NewCommandQueue()
	buf := NewBufferFloats(d, x)
	q.Run1D(pipe, nH*rhalf, 64, buf, NewBufferFloats(d, invf), NewBufferU32(d, hd),
		NewBufferU32(d, pos), NewBufferU32(d, uint32(nH*rhalf)), NewBufferU32(d, uint32(rhalf)),
		NewBufferFloats(d, []float32{1.0}))
	got := buf.Floats()

	var maxAbs float64
	tailChanged := false
	for h := range nH {
		for i := range hd {
			gi := h*hd + i
			if dd := math.Abs(float64(got[gi] - ref[gi])); math.IsNaN(dd) || dd > maxAbs {
				maxAbs = dd // propagate NaN so mustFinite can catch degenerate output
			}
			if i >= rotaryDim && got[gi] != x[gi] { // tail must be identical
				tailChanged = true
			}
		}
	}
	mustFinite(t, "partial rope maxAbs", maxAbs)
	if maxAbs > 1e-5 || tailChanged {
		t.Fatalf("partial rope FAIL: maxAbs=%.2e tailChanged=%v", maxAbs, tailChanged)
	}
	t.Logf("partial rope (rotaryDim=%d/hd=%d): maxAbs=%.2e, tail [%d,%d) untouched — PARITY ✓",
		rotaryDim, hd, maxAbs, rotaryDim, hd)
}
