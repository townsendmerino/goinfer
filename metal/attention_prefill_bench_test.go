//go:build darwin && goinfer_testhooks

package metal

import (
	"math/rand"
	"testing"
	"time"
)

// BenchmarkAttentionPrefill isolates attention_prefill's (metal/prefill.go) own throughput at a
// realistic prompt length and head geometry (qwen2.5-1.5b-class: nH=12, nKV=2, hd=128), one
// dispatch per prefill call (prefill.go: pf.pAttn, grid = M*nH*128).
func BenchmarkAttentionPrefill(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "attention_prefill")
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}

	const M, nH, nKV, hd = 140, 12, 2, 128
	const qDim = nH * hd
	const kvDim = nKV * hd
	const qStride = qDim // Q-only buffer for this throughput benchmark

	rng := rand.New(rand.NewSource(59))
	rndHalf := func(n int) []uint16 {
		s := make([]uint16, n)
		for i := range s {
			s[i] = f32ToF16(rng.Float32()*2 - 1)
		}
		return s
	}
	dQkv := d.NewBufferU16s(rndHalf(M * qStride))
	dKc := d.NewBufferU16s(rndHalf(M * kvDim))
	dVc := d.NewBufferU16s(rndHalf(M * kvDim))
	dOut := d.NewBufferU16s(make([]uint16, M*qDim))
	uNH := d.NewBufferU32(nH)
	uNKV := d.NewBufferU32(nKV)
	uHd := d.NewBufferU32(hd)
	uStartPos := d.NewBufferU32(0)
	uScale := d.NewBufferFloats([]float32{1.0 / 11.3137}) // 1/sqrt(128)
	uQStride := d.NewBufferU32(uint32(qStride))
	uWindow := d.NewBufferU32(0)

	q_ := d.NewCommandQueue()
	const reps = 4
	run := func() {
		e := q_.Begin()
		for range reps {
			e.Dispatch(pipe, M*nH*128, 128, dQkv, dKc, dVc, dOut, uNH, uNKV, uHd, uStartPos, uScale, uQStride, uWindow)
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
