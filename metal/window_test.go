//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestSlidingWindow — validates sliding-window attention: windowed attention (window=W) over N
// keys must equal FULL attention over just the last W keys (same queries). This is exactly the
// Mistral semantics — a query attends only keys[nKeys-W, nKeys). Uses the production attention
// kernel + f16 KV.
func TestSlidingWindow(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "attention")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	const nH, nKV, hd = 4, 2, 128
	const N, W = 50, 16 // 50 keys, window 16 → attend last 16
	kvDim := nKV * hd
	scale := float32(1 / math.Sqrt(float64(hd)))
	rng := rand.New(rand.NewSource(17))
	q := make([]float32, nH*hd)
	for i := range q {
		q[i] = rng.Float32()*2 - 1
	}
	kFull := make([]uint16, N*kvDim)
	vFull := make([]uint16, N*kvDim)
	for i := range kFull {
		kFull[i] = f32ToF16(rng.Float32()*2 - 1)
		vFull[i] = f32ToF16(rng.Float32()*2 - 1)
	}
	run := func(kc, vc []uint16, nKeys, window int) []float32 {
		cq := d.NewCommandQueue()
		out := d.NewBufferLen(nH * hd)
		cq.Run1D(pipe, nH*128, 128, d.NewBufferFloats(q), d.NewBufferU16s(kc), d.NewBufferU16s(vc),
			out, d.NewBufferU32(nH), d.NewBufferU32(nKV), d.NewBufferU32(hd), d.NewBufferU32(uint32(nKeys)),
			d.NewBufferFloats([]float32{scale}), d.NewBufferU32(uint32(window)))
		return append([]float32(nil), out.Floats()...)
	}
	// A: windowed over all N keys.
	outA := run(kFull, vFull, N, W)
	// B: full attention over just the last W keys.
	lastK := append([]uint16(nil), kFull[(N-W)*kvDim:]...)
	lastV := append([]uint16(nil), vFull[(N-W)*kvDim:]...)
	outB := run(lastK, lastV, W, 0)

	var maxAbs float64
	for i := range outA {
		if dd := math.Abs(float64(outA[i] - outB[i])); math.IsNaN(dd) || dd > maxAbs {
			maxAbs = dd // propagate NaN so mustFinite can catch degenerate output
		}
	}
	mustFinite(t, "sliding-window maxAbs", maxAbs)
	if maxAbs > 1e-4 {
		t.Fatalf("sliding-window FAIL: windowed(N=%d,W=%d) != full(last %d): maxAbs=%.2e", N, W, W, maxAbs)
	}
	t.Logf("sliding-window: windowed over %d keys == full over last %d: maxAbs=%.2e — PARITY ✓", N, W, maxAbs)
}
