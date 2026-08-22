//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestRmsnormF32_cpuParity gates rmsnorm_f32 (metal/kernels.go) sourced from the real allKernels --
// Gemma's sandwich-norm kernel (model.go: r.pRmsF32, launch shape (256, 256), dispatched for
// postAttnNorm/postMLPNorm).
func TestRmsnormF32_cpuParity(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "rmsnorm_f32")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const rfH = 4096
	const eps = 1e-6
	rng := rand.New(rand.NewSource(17))
	x := make([]float32, rfH)
	w := make([]float32, rfH)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
		w[i] = 0.5 + rng.Float32()
	}

	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	rms := float32(1 / math.Sqrt(ss/float64(rfH)+eps))
	ref := make([]float32, rfH)
	for i := range x {
		ref[i] = x[i] * rms * (1.0 + w[i]) // addOne=1
	}

	q := d.NewCommandQueue()
	xBuf := NewBufferFloats(d, x)
	q.Run1D(pipe, 256, 256,
		xBuf,
		NewBufferFloats(d, w),
		NewBufferU32(d, uint32(rfH)),
		NewBufferFloats(d, []float32{eps}),
		NewBufferU32(d, 1), // addOne
	)
	got := xBuf.Floats()

	var dot, na, nb, maxAbs float64
	for i := range rfH {
		dot += float64(got[i]) * float64(ref[i])
		na += float64(got[i]) * float64(got[i])
		nb += float64(ref[i]) * float64(ref[i])
		if d := math.Abs(float64(got[i] - ref[i])); d > maxAbs {
			maxAbs = d
		}
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	mustFinite(t, "rmsnorm_f32 cosine", cos)
	if cos < 0.999999 {
		t.Fatalf("rmsnorm_f32 drifts from CPU reference: cosine=%.9f maxAbs=%.4g", cos, maxAbs)
	}
	t.Logf("rmsnorm_f32 H=%d vs CPU: cosine=%.9f maxAbs=%.4g", rfH, cos, maxAbs)
}

// BenchmarkRmsnormF32 isolates rmsnorm_f32's own throughput at production dispatch shape
// (256 threads, one threadgroup).
func BenchmarkRmsnormF32(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "rmsnorm_f32")
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}

	const rfH = 4096
	rng := rand.New(rand.NewSource(17))
	rndf := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*2 - 1
		}
		return s
	}
	dX := NewBufferFloats(d, rndf(rfH))
	dW := NewBufferFloats(d, rndf(rfH))
	uH := NewBufferU32(d, uint32(rfH))
	uEps := NewBufferFloats(d, []float32{1e-6})
	uAddOne := NewBufferU32(d, 1)

	q_ := d.NewCommandQueue()
	const reps = 8
	run := func() {
		e := q_.Begin()
		for range reps {
			e.Dispatch(pipe, 256, 256, dX, dW, uH, uEps, uAddOne)
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
