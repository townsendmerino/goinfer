//go:build darwin && goinfer_testhooks

package metal

import (
	"math/rand"
	"testing"
	"time"
)

// BenchmarkAttention isolates decode's attention kernel (metal/kernels.go, model.go: r.pAttn,
// launch shape (nH*128, 128) one threadgroup per Q head) at a realistic long-context depth --
// the V-read loop (`for(uint s=winStart;s<nKeys;s++) a += sc[s]*float(vb[s*kvDim+d])`) is the
// per-token cost this isolates; qk-scoring/softmax are the same kernel but the V-read dominates
// at depth (docs/benchmarks.md: attention ~56% of a token at 2048 ctx).
func BenchmarkAttention(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "attention")
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}

	const nH, nKV, hd, nKeys = 12, 2, 128, 2048 // qwen2.5-1.5b-class shape at a realistic long-ctx depth
	kvDim := nKV * hd
	rng := rand.New(rand.NewSource(5))
	rndf := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64()) * 0.5
		}
		return s
	}
	rndf16 := func(n int) []uint16 {
		s := make([]uint16, n)
		for i := range s {
			s[i] = f32ToF16(float32(rng.NormFloat64()) * 0.5)
		}
		return s
	}

	q := NewBufferFloats(d, rndf(nH*hd))
	kc := NewBufferU16s(d, rndf16(nKeys*kvDim))
	vc := NewBufferU16s(d, rndf16(nKeys*kvDim))
	out := d.NewBufferLen(nH * hd)
	uNH := NewBufferU32(d, uint32(nH))
	uNKV := NewBufferU32(d, uint32(nKV))
	uHd := NewBufferU32(d, uint32(hd))
	uNKeys := NewBufferU32(d, uint32(nKeys))
	uScale := NewBufferFloats(d, []float32{1.0 / 11.3137})
	uWindow := NewBufferU32(d, 0)
	sinks := NewBufferFloats(d, make([]float32, nH))
	uHasSink := NewBufferU32(d, 0)

	q_ := d.NewCommandQueue()
	const reps = 2
	run := func() {
		e := q_.Begin()
		for range reps {
			e.Dispatch(pipe, nH*128, 128, q, kc, vc, out, uNH, uNKV, uHd, uNKeys, uScale, uWindow, sinks, uHasSink)
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
