//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestGemvW8A8Coal_cpuParity gates gemv_w8a8_coal (metal/kernels.go) — the int8 LM-head GEMV every
// resident decode step dispatches (model.go's ForwardEmb*/ForwardArgmax, r.pGemvW8, launch shape
// (V*32, 32) one simdgroup per output row). Sourced from the real allKernels (not a private copy,
// unlike gemv_test.go's TestLayerB_gemvW8A8Parity, which tests the UNUSED naive gemv_w8a8 sibling)
// so any future edit here is gated against the actual shipped kernel.
func TestGemvW8A8Coal_cpuParity(t *testing.T) {
	worst, skip := gemvCoalDrift(t, allKernels)
	if skip {
		return
	}
	if worst.cos < 0.99999 || worst.maxrel > 1e-4 {
		t.Fatalf("gemv_w8a8_coal drifts from the CPU reference: cosine=%.7f maxrel=%.2e", worst.cos, worst.maxrel)
	}
	t.Logf("gemv_w8a8_coal %dx%d vs CPU: cosine=%.9f maxrel=%.2e", gemvCoalN, gemvCoalK, worst.cos, worst.maxrel)
}

type gemvCoalResult struct{ cos, maxrel float64 }

const (
	gemvCoalN = 512  // sample rows, not the full vocab -- enough to exercise every lane/tail shape
	gemvCoalK = 4096 // a realistic hidden dim (e.g. a 7-8B-class model's LM head width)
)

func gemvCoalDrift(t *testing.T, src string) (worst gemvCoalResult, skip bool) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
		return gemvCoalResult{}, true
	}
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "gemv_w8a8_coal")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	quantRow := func(rng *rand.Rand, row []float32) ([]int8, float32) {
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
			if r > 127 {
				r = 127
			}
			if r < -127 {
				r = -127
			}
			q[i] = int8(r)
		}
		return q, sc
	}

	rng := rand.New(rand.NewSource(7))
	a := make([]float32, gemvCoalK)
	for i := range a {
		a[i] = rng.Float32()*2 - 1
	}
	aq, aSc := quantRow(rng, a)

	bq := make([]int8, gemvCoalN*gemvCoalK)
	bSc := make([]float32, gemvCoalN)
	for n := range gemvCoalN {
		row := make([]float32, gemvCoalK)
		for k := range row {
			row[k] = rng.Float32()*2 - 1
		}
		q, sc := quantRow(rng, row)
		copy(bq[n*gemvCoalK:(n+1)*gemvCoalK], q)
		bSc[n] = sc
	}

	ref := make([]float32, gemvCoalN)
	for n := range gemvCoalN {
		var acc int32
		for k := range gemvCoalK {
			acc += int32(aq[k]) * int32(bq[n*gemvCoalK+k])
		}
		ref[n] = float32(acc) * aSc * bSc[n]
	}

	q := d.NewCommandQueue()
	out := d.NewBufferLen(gemvCoalN)
	q.Run1D(pipe, gemvCoalN*32, 32,
		NewBufferInt8(d, aq),
		NewBufferFloats(d, []float32{aSc}),
		NewBufferInt8(d, bq),
		NewBufferFloats(d, bSc),
		out,
		NewBufferU32(d, uint32(gemvCoalK)),
	)
	got := out.Floats()

	var dot, na, nb, maxrel float64
	for n := range gemvCoalN {
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
	return gemvCoalResult{cos: cos, maxrel: maxrel}, false
}

// BenchmarkGemvW8A8Coal isolates gemv_w8a8_coal's own throughput at production dispatch shape
// (gemvCoalN*32 threads, threadgroup 32). Same safe-operating-point discipline as
// BenchmarkDeltaRule (metal/deltanet_bench_test.go): bisect reps against the known Metal
// fault-0x10 crash risk before trusting a number, order-alternated best-of-N via best-of-min-time
// across b.N reps.
func BenchmarkGemvW8A8Coal(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "gemv_w8a8_coal")
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}

	rng := rand.New(rand.NewSource(7))
	rnd8 := func(n int) []int8 {
		s := make([]int8, n)
		for i := range s {
			s[i] = int8(rng.Intn(255) - 127)
		}
		return s
	}
	rndf := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*0.01 + 0.001
		}
		return s
	}
	dAq := NewBufferInt8(d, rnd8(gemvCoalK))
	dAsc := NewBufferFloats(d, []float32{0.01})
	dBq := NewBufferInt8(d, rnd8(gemvCoalN*gemvCoalK))
	dBsc := NewBufferFloats(d, rndf(gemvCoalN))
	dOut := d.NewBufferLen(gemvCoalN)
	uK := NewBufferU32(d, uint32(gemvCoalK))

	q_ := d.NewCommandQueue()
	const reps = 2 // starting point mirroring BenchmarkDeltaRule's bisected-safe reps; re-bisect if this crashes
	run := func() {
		e := q_.Begin()
		for range reps {
			e.Dispatch(pipe, gemvCoalN*32, 32, dAq, dAsc, dBq, dBsc, dOut, uK)
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
