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
	dQkv := NewBufferU16s(d, rndHalf(M*qStride))
	dKc := NewBufferU16s(d, rndHalf(M*kvDim))
	dVc := NewBufferU16s(d, rndHalf(M*kvDim))
	dOut := NewBufferU16s(d, make([]uint16, M*qDim))
	uNH := NewBufferU32(d, nH)
	uNKV := NewBufferU32(d, nKV)
	uHd := NewBufferU32(d, hd)
	uStartPos := NewBufferU32(d, 0)
	uScale := NewBufferFloats(d, []float32{1.0 / 11.3137}) // 1/sqrt(128)
	uQStride := NewBufferU32(d, uint32(qStride))
	uWindow := NewBufferU32(d, 0)

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

// benchAttentionPrefillFused is shared by the short (M=140, apples-to-apples with
// BenchmarkAttentionPrefill) and long-context (M=3900, the depth docs/task-prefill-gap.md §4
// prices L2-Metal against) variants below.
func benchAttentionPrefillFused(b *testing.B, M int) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "attention_prefill_fused")
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}

	const nH, nKV, hd = 12, 2, 128
	const qDim = nH * hd
	const kvDim = nKV * hd
	qStride := qDim

	rng := rand.New(rand.NewSource(59))
	rndHalf := func(n int) []uint16 {
		s := make([]uint16, n)
		for i := range s {
			s[i] = f32ToF16(rng.Float32()*2 - 1)
		}
		return s
	}
	dQkv := NewBufferU16s(d, rndHalf(M*qStride))
	dKc := NewBufferU16s(d, rndHalf(M*kvDim))
	dVc := NewBufferU16s(d, rndHalf(M*kvDim))
	dOut := NewBufferU16s(d, make([]uint16, M*qDim))
	uNH := NewBufferU32(d, nH)
	uNKV := NewBufferU32(d, nKV)
	uHd := NewBufferU32(d, hd)
	uStartPos := NewBufferU32(d, 0)
	uScale := NewBufferFloats(d, []float32{1.0 / 11.3137}) // 1/sqrt(128)
	uQStride := NewBufferU32(d, uint32(qStride))
	uWindow := NewBufferU32(d, 0)
	uM := NewBufferU32(d, uint32(M))

	const sgpt = 4
	numRowTiles := (M + 7) / 8
	total := nH * numRowTiles
	total = (total + sgpt - 1) / sgpt * sgpt * 32

	q_ := d.NewCommandQueue()
	const reps = 4
	run := func() {
		e := q_.Begin()
		for range reps {
			e.Dispatch(pipe, total, sgpt*32, dQkv, dKc, dVc, dOut, uNH, uNKV, uHd, uStartPos, uScale, uQStride, uWindow, uM)
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

// BenchmarkAttentionPrefillFused is the direct M=140 counterpart to BenchmarkAttentionPrefill —
// same geometry, same reps, so ns/dispatch(best) is comparable line for line.
func BenchmarkAttentionPrefillFused(b *testing.B) { benchAttentionPrefillFused(b, 140) }

// BenchmarkAttentionPrefillLongCtx / BenchmarkAttentionPrefillFusedLongCtx: M=3900, the depth
// docs/task-prefill-gap.md §4's "then, and now sized" paragraph prices the fused kernel against
// (S model K=3900, where the exact kernel's per-token cost is least flat).
func BenchmarkAttentionPrefillLongCtx(b *testing.B) {
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
	const M, nH, nKV, hd = 3900, 12, 2, 128
	const qDim = nH * hd
	const kvDim = nKV * hd
	const qStride = qDim
	rng := rand.New(rand.NewSource(59))
	rndHalf := func(n int) []uint16 {
		s := make([]uint16, n)
		for i := range s {
			s[i] = f32ToF16(rng.Float32()*2 - 1)
		}
		return s
	}
	dQkv := NewBufferU16s(d, rndHalf(M*qStride))
	dKc := NewBufferU16s(d, rndHalf(M*kvDim))
	dVc := NewBufferU16s(d, rndHalf(M*kvDim))
	dOut := NewBufferU16s(d, make([]uint16, M*qDim))
	uNH := NewBufferU32(d, nH)
	uNKV := NewBufferU32(d, nKV)
	uHd := NewBufferU32(d, hd)
	uStartPos := NewBufferU32(d, 0)
	uScale := NewBufferFloats(d, []float32{1.0 / 11.3137})
	uQStride := NewBufferU32(d, uint32(qStride))
	uWindow := NewBufferU32(d, 0)

	q_ := d.NewCommandQueue()
	const reps = 2
	run := func() {
		e := q_.Begin()
		for range reps {
			e.Dispatch(pipe, M*nH*128, 128, dQkv, dKc, dVc, dOut, uNH, uNKV, uHd, uStartPos, uScale, uQStride, uWindow)
		}
		e.End()
	}
	for range 2 {
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

func BenchmarkAttentionPrefillFusedLongCtx(b *testing.B) { benchAttentionPrefillFused(b, 3900) }
