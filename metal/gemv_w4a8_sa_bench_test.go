//go:build darwin && goinfer_testhooks

package metal

import (
	"math/rand"
	"testing"
	"time"
)

// BenchmarkGemvW4A8Sa isolates gemv_w4a8_sa's (metal/kernels.go, SA_BODY macro) own throughput
// at a realistic K=4096 (Mistral-7B-class QKV/O-proj/gate-up width, same geometry as
// TestSAGemvLargeK), the highest-call-frequency int4 GEMV family (11 production call sites).
func BenchmarkGemvW4A8Sa(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "gemv_w4a8_sa")
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}

	const N, K = 256, 4096
	rng := rand.New(rand.NewSource(31))
	rndWords := func(n int) []uint32 {
		s := make([]uint32, n)
		for i := range s {
			s[i] = rng.Uint32()
		}
		return s
	}
	rndHalf := func(n int) []uint16 {
		s := make([]uint16, n)
		for i := range s {
			s[i] = f32ToF16(rng.Float32()*0.01 + 0.001)
		}
		return s
	}
	rnd8 := func(n int) []int8 {
		s := make([]int8, n)
		for i := range s {
			s[i] = int8(rng.Intn(255) - 127)
		}
		return s
	}
	dWq := d.NewBufferUint32s(rndWords(N * (K / 8)))
	dSct := d.NewBufferU16s(rndHalf(N * (K / 32)))
	dAq := d.NewBufferInt8(rnd8(K))
	dAsc := d.NewBufferFloats([]float32{0.01})
	dOut := d.NewBufferLen(N)
	uK := d.NewBufferU32(uint32(K))

	q_ := d.NewCommandQueue()
	const reps = 4
	run := func() {
		e := q_.Begin()
		for range reps {
			e.DispatchTG(pipe, N*32, 256, K*2, dWq, dSct, dAq, dAsc, dOut, uK)
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
