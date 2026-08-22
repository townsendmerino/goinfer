//go:build darwin && goinfer_testhooks

package metal

import (
	"math/rand"
	"testing"
	"time"
)

// BenchmarkRope2 isolates rope2's own throughput at a realistic GQA dispatch shape (model.go:
// r.pRope2, launch shape (nH*half + nKV*half, 64), dispatched once per layer).
func BenchmarkRope2(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "rope2")
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}

	const nH, nKV, hd = 32, 8, 128
	const half = hd / 2
	const nHhd = nH * hd
	qTotal := uint32(nH * half)
	kTotal := uint32(nKV * half)

	rng := rand.New(rand.NewSource(19))
	rndf := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*2 - 1
		}
		return s
	}
	dX := NewBufferFloats(d, rndf(nH*hd+nKV*hd*2)) // qkv fused: Q + K + V, only Q/K rotated
	dInvf := NewBufferFloats(d, rndf(half))
	uHd := NewBufferU32(d, hd)
	uPos := NewBufferU32(d, 3)
	uQtotal := NewBufferU32(d, qTotal)
	uKtotal := NewBufferU32(d, kTotal)
	uHalf := NewBufferU32(d, half)
	uScale := NewBufferFloats(d, []float32{1.0})
	uNHhd := NewBufferU32(d, uint32(nHhd))

	q_ := d.NewCommandQueue()
	const reps = 8
	run := func() {
		e := q_.Begin()
		for range reps {
			e.Dispatch(pipe, int(qTotal+kTotal), 64, dX, dInvf, uHd, uPos, uQtotal, uKtotal, uHalf, uScale, uNHhd)
		}
		e.End()
	}

	for range 4 {
		run()
	}
	b.ResetTimer()
	best := time.Hour
	for range b.N {
		t0 := time.Now()
		run()
		if dt := time.Since(t0); dt < best {
			best = dt
		}
	}
	b.ReportMetric(float64(best/time.Duration(reps))/float64(time.Nanosecond), "ns/dispatch(best)")
}
