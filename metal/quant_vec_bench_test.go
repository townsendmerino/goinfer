//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestQuantVec_cpuParity gates quant_vec (metal/kernels.go) sourced from the real allKernels --
// quantizes the attention context output before the O-proj GEMV (model.go: r.pQv, launch shape
// (256, 256), one threadgroup, called once per layer per token).
func TestQuantVec_cpuParity(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "quant_vec")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const qvH = 4096
	rng := rand.New(rand.NewSource(11))
	x := make([]float32, qvH)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	var mx float32
	for _, v := range x {
		if a := float32(math.Abs(float64(v))); a > mx {
			mx = a
		}
	}
	refSc := mx / 127
	if refSc == 0 {
		refSc = 1
	}
	refQ := make([]int8, qvH)
	for i, v := range x {
		q := max(min(int(math.Round(float64(v/refSc))), 127), -127)
		refQ[i] = int8(q)
	}

	q := d.NewCommandQueue()
	aqBuf := d.NewBufferBytes(qvH)
	ascBuf := d.NewBufferLen(1)
	q.Run1D(pipe, 256, 256,
		d.NewBufferFloats(x),
		aqBuf,
		ascBuf,
		d.NewBufferU32(uint32(qvH)),
	)
	gotQ := aqBuf.Int8s()
	gotSc := ascBuf.Floats()[0]

	mism := 0
	var dot, na, nb float64
	for i := range qvH {
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
	mustFinite(t, "quant_vec cosine", cos)
	if cos < 0.99999 || mism > qvH/100 {
		t.Fatalf("quant_vec drifts from CPU reference: cosine=%.7f int8-mismatch=%d/%d", cos, mism, qvH)
	}
	t.Logf("quant_vec H=%d vs CPU: cosine=%.9f, int8 exact=%d/%d", qvH, cos, qvH-mism, qvH)
}

// BenchmarkQuantVec isolates quant_vec's own throughput at production dispatch shape
// (256 threads, one threadgroup).
func BenchmarkQuantVec(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "quant_vec")
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}

	const qvH = 4096
	rng := rand.New(rand.NewSource(11))
	rndf := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*2 - 1
		}
		return s
	}
	dX := d.NewBufferFloats(rndf(qvH))
	dAq := d.NewBufferBytes(qvH)
	dAsc := d.NewBufferLen(1)
	uH := d.NewBufferU32(uint32(qvH))

	q_ := d.NewCommandQueue()
	const reps = 8
	run := func() {
		e := q_.Begin()
		for range reps {
			e.Dispatch(pipe, 256, 256, dX, dAq, dAsc, uH)
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
