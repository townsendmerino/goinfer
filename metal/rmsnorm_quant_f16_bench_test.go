//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestRmsnormQuantF16_cpuParity gates rmsnorm_quant_f16 (metal/prefill.go) sourced from the real
// prefillKernels -- prefill's fused norm+quant of the LAST token's f16 activation feeding the
// int8-pinned LM head (prefill.go: pf.pRmsQ, launch shape (256, 256), dispatched once per prefill
// call).
func TestRmsnormQuantF16_cpuParity(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "rmsnorm_quant_f16")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const rqH = 4096
	const eps = 1e-6
	const addOne = 1
	rng := rand.New(rand.NewSource(53))
	xf16 := make([]uint16, rqH)
	x := make([]float32, rqH)
	w := make([]float32, rqH)
	for i := range x {
		v := rng.Float32()*2 - 1
		xf16[i] = f32ToF16(v)
		x[i] = f16ToF32(xf16[i]) // round-trip so the CPU ref matches the f16 input exactly
		w[i] = 0.5 + rng.Float32()
	}

	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	rms := float32(1 / math.Sqrt(ss/float64(rqH)+eps))
	s := make([]float64, rqH)
	var mx float64
	for i := range x {
		g := 1.0 + w[i]
		s[i] = float64(x[i]) * float64(rms) * float64(g)
		if a := math.Abs(s[i]); a > mx {
			mx = a
		}
	}
	refSc := float32(mx / 127)
	if refSc == 0 {
		refSc = 1
	}
	refQ := make([]int8, rqH)
	for i := range s {
		q := max(min(int(math.Round(s[i]/float64(refSc))), 127), -127)
		refQ[i] = int8(q)
	}

	q := d.NewCommandQueue()
	aqBuf := d.NewBufferBytes(rqH)
	ascBuf := d.NewBufferLen(1)
	q.Run1D(pipe, 256, 256,
		NewBufferU16s(d, xf16),
		NewBufferFloats(d, w),
		aqBuf,
		ascBuf,
		NewBufferU32(d, uint32(rqH)),
		NewBufferFloats(d, []float32{eps}),
		NewBufferU32(d, addOne),
	)
	gotQ := aqBuf.Int8s()
	gotSc := ascBuf.Floats()[0]

	mism := 0
	var dot, na, nb float64
	for i := range rqH {
		if gotQ[i] != refQ[i] {
			mism++
		}
		g := float64(gotQ[i]) * float64(gotSc)
		r := float64(refQ[i]) * float64(refSc)
		dot += g * r
		na += g * g
		nb += r * r
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	mustFinite(t, "rmsnorm_quant_f16 cosine", cos)
	if cos < 0.99999 || mism > rqH/100 {
		t.Fatalf("rmsnorm_quant_f16 drifts from CPU reference: cosine=%.7f int8-mismatch=%d/%d", cos, mism, rqH)
	}
	t.Logf("rmsnorm_quant_f16 H=%d vs CPU: cosine=%.9f, int8 exact=%d/%d", rqH, cos, rqH-mism, rqH)
}

// BenchmarkRmsnormQuantF16 isolates rmsnorm_quant_f16's own throughput at a realistic dispatch
// shape.
func BenchmarkRmsnormQuantF16(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "rmsnorm_quant_f16")
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}

	const rqH = 4096
	rng := rand.New(rand.NewSource(53))
	rndf16 := func(n int) []uint16 {
		s := make([]uint16, n)
		for i := range s {
			s[i] = f32ToF16(rng.Float32()*2 - 1)
		}
		return s
	}
	rndf := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*2 - 1
		}
		return s
	}
	dX := NewBufferU16s(d, rndf16(rqH))
	dW := NewBufferFloats(d, rndf(rqH))
	dAq := d.NewBufferBytes(rqH)
	dAsc := d.NewBufferLen(1)
	uH := NewBufferU32(d, uint32(rqH))
	uEps := NewBufferFloats(d, []float32{1e-6})
	uAddOne := NewBufferU32(d, 1)

	q_ := d.NewCommandQueue()
	const reps = 8
	run := func() {
		e := q_.Begin()
		for range reps {
			e.Dispatch(pipe, 256, 256, dX, dW, dAq, dAsc, uH, uEps, uAddOne)
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
