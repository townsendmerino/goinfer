//go:build darwin && goinfer_testhooks

package metal

import (
	"math/rand"
	"testing"
	"time"
)

// BenchmarkGemvW4A8Moe isolates gemv_w4a8_moe's (metal/moe.go, shares SA_BODY's UNP8-family
// unpack) own throughput at a realistic gate|up-projection K (moe.go: mo.pGU, launch shape
// (rowsPerExpert*32, 256), dispatched once per selected expert per layer).
func BenchmarkGemvW4A8Moe(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "gemv_w4a8_moe")
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}

	const E, N, K = 8, 64, 4096
	rng := rand.New(rand.NewSource(41))
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
	dWq := NewBufferUint32s(d, rndWords(E*N*(K/8)))
	dSct := NewBufferU16s(d, rndHalf(E*N*(K/32)))
	dAq := NewBufferInt8(d, rnd8(K))
	dAsc := NewBufferFloats(d, []float32{0.01})
	dOut := d.NewBufferLen(N)
	uK := NewBufferU32(d, uint32(K))
	dIdx := NewBufferUint32s(d, []uint32{3})
	uSlot := NewBufferU32(d, 0)
	uRowsPerExpert := NewBufferU32(d, uint32(N))

	q_ := d.NewCommandQueue()
	const reps = 4
	run := func() {
		e := q_.Begin()
		for range reps {
			e.DispatchTG(pipe, N*32, 256, K*2, dWq, dSct, dAq, dAsc, dOut, uK, dIdx, uSlot, uRowsPerExpert)
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
