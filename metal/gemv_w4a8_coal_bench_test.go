//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestGemvW4A8Coal_cpuParity gates gemv_w4a8_coal (metal/kernels.go, W4A8_BODY macro) -- every
// dense down-projection in decode (model.go: r.pGemv, launch shape (H*32, 32) one simdgroup per
// output row, K=intermediate size). Sourced from the real allKernels.
func TestGemvW4A8Coal_cpuParity(t *testing.T) {
	worst, skip := gemvW4A8CoalDrift(t, allKernels)
	if skip {
		return
	}
	if worst.cos < 0.99999 || worst.maxrel > 1e-4 {
		t.Fatalf("gemv_w4a8_coal drifts from the CPU reference: cosine=%.7f maxrel=%.2e", worst.cos, worst.maxrel)
	}
	t.Logf("gemv_w4a8_coal %dx%d vs CPU: cosine=%.9f maxrel=%.2e", gemvW4A8CoalN, gemvW4A8CoalK, worst.cos, worst.maxrel)
}

type gemvW4A8CoalResult struct{ cos, maxrel float64 }

const (
	gemvW4A8CoalN = 512  // sample rows -- enough to exercise every lane/tail shape
	gemvW4A8CoalK = 4096 // realistic intermediate size, multiple of 8 (word) and 32 (group)
)

func gemvW4A8CoalDrift(t *testing.T, src string) (worst gemvW4A8CoalResult, skip bool) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
		return gemvW4A8CoalResult{}, true
	}
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "gemv_w4a8_coal")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	rng := rand.New(rand.NewSource(11))

	quantRow := func(row []float32) ([]int8, float32) {
		mx := float32(0)
		for _, v := range row {
			if a := float32(math.Abs(float64(v))); a > mx {
				mx = a
			}
		}
		sc := mx / 127
		if sc == 0 {
			sc = 1
		}
		q := make([]int8, len(row))
		for i, v := range row {
			r := math.Round(float64(v / sc))
			q[i] = int8(max(min(int(r), 127), -127))
		}
		return q, sc
	}

	a := make([]float32, gemvW4A8CoalK)
	for i := range a {
		a[i] = rng.Float32()*2 - 1
	}
	aq, aSc := quantRow(a)

	nWords := gemvW4A8CoalK / 8
	nGroups := gemvW4A8CoalK / 32
	bq := make([]uint32, gemvW4A8CoalN*nWords)
	bsc := make([]float32, gemvW4A8CoalN*nGroups) // f32 here, cast to half on device below

	row := make([]float32, gemvW4A8CoalK)
	ref := make([]float32, gemvW4A8CoalN)
	for n := range gemvW4A8CoalN {
		for i := range row {
			row[i] = rng.Float32()*2 - 1
		}
		words, scales := packW4A8Row(row)
		copy(bq[n*nWords:(n+1)*nWords], words)
		// Round-trip the scale through f16 -- the kernel reads `bsc` as half, so the CPU
		// reference must use the SAME rounded value, not the wider f32 the packer produced.
		for g := range scales {
			scales[g] = f16ToF32(f32ToF16(scales[g]))
		}
		copy(bsc[n*nGroups:(n+1)*nGroups], scales)

		// Recover the exact quantized nibbles the kernel will unpack (round-trip through the
		// same words), so the CPU reference matches the on-device int4×int8 dot exactly.
		var acc float64
		for wi := 0; wi < nWords; wi++ {
			x := words[wi]
			var gi int
			for j := 0; j < 8; j++ {
				nib := int((x>>(4*uint(j)))&0xF) - 8
				gi += nib * int(aq[wi*8+j])
			}
			acc += float64(gi) * float64(scales[wi/4])
		}
		ref[n] = float32(acc) * aSc
	}

	// device buffers: bsc must be half-precision (matches `device const half* bsc`).
	bscHalf := make([]uint16, len(bsc))
	for i, v := range bsc {
		bscHalf[i] = f32ToF16(v)
	}

	q := d.NewCommandQueue()
	out := d.NewBufferLen(gemvW4A8CoalN)
	q.Run1D(pipe, gemvW4A8CoalN*32, 32,
		d.NewBufferUint32s(bq),
		d.NewBufferU16s(bscHalf),
		d.NewBufferInt8(aq),
		d.NewBufferFloats([]float32{aSc}),
		out,
		d.NewBufferU32(uint32(gemvW4A8CoalK)),
	)
	got := out.Floats()

	var dot, na, nb, maxrel float64
	for n := range gemvW4A8CoalN {
		dot += float64(got[n]) * float64(ref[n])
		na += float64(got[n]) * float64(got[n])
		nb += float64(ref[n]) * float64(ref[n])
		if diff := math.Abs(float64(got[n] - ref[n])); diff > 0 {
			rel := diff / (math.Abs(float64(ref[n])) + 1e-6)
			if rel > maxrel {
				maxrel = rel
			}
		}
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	return gemvW4A8CoalResult{cos: cos, maxrel: maxrel}, false
}

// BenchmarkGemvW4A8Coal isolates gemv_w4a8_coal's own throughput at production dispatch shape.
func BenchmarkGemvW4A8Coal(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "gemv_w4a8_coal")
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}

	rng := rand.New(rand.NewSource(11))
	nWords := gemvW4A8CoalK / 8
	nGroups := gemvW4A8CoalK / 32
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
	dBq := d.NewBufferUint32s(rndWords(gemvW4A8CoalN * nWords))
	dBsc := d.NewBufferU16s(rndHalf(gemvW4A8CoalN * nGroups))
	dAq := d.NewBufferInt8(rnd8(gemvW4A8CoalK))
	dAsc := d.NewBufferFloats([]float32{0.01})
	dOut := d.NewBufferLen(gemvW4A8CoalN)
	uK := d.NewBufferU32(uint32(gemvW4A8CoalK))

	q_ := d.NewCommandQueue()
	const reps = 4
	run := func() {
		e := q_.Begin()
		for range reps {
			e.Dispatch(pipe, gemvW4A8CoalN*32, 32, dBq, dBsc, dAq, dAsc, dOut, uK)
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
