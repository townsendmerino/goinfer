//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestSwigluQuant_cpuParity gates swiglu_quant (metal/kernels.go) sourced from the real
// allKernels -- the gated-MLP activation+quant every SwiGLU model (Qwen, Llama, Mixtral, ...)
// dispatches (model.go: r.pSw, launch shape (256, 256), once per layer). Benchmarked at a
// realistic intermediate size (4096) rather than the tiny test fixture's actual 128, since real
// production models are in the thousands.
func TestSwigluQuant_cpuParity(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "swiglu_quant")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const sgI = 4096
	const actSilu = 1
	rng := rand.New(rand.NewSource(29))
	g := make([]float32, sgI)
	u := make([]float32, sgI)
	for i := range g {
		g[i] = rng.Float32()*4 - 2
		u[i] = rng.Float32()*2 - 1
	}

	silu := func(x float64) float64 { return x / (1 + math.Exp(-x)) }
	s := make([]float64, sgI)
	var mx float64
	for i := range g {
		s[i] = silu(float64(g[i])) * float64(u[i])
		if a := math.Abs(s[i]); a > mx {
			mx = a
		}
	}
	refSc := float32(mx / 127)
	if refSc == 0 {
		refSc = 1
	}
	refQ := make([]int8, sgI)
	for i := range s {
		q := max(min(int(math.Round(s[i]/float64(refSc))), 127), -127)
		refQ[i] = int8(q)
	}

	q := d.NewCommandQueue()
	dqBuf := d.NewBufferBytes(sgI)
	dsBuf := d.NewBufferLen(1)
	q.Run1D(pipe, 256, 256,
		d.NewBufferFloats(g),
		d.NewBufferFloats(u),
		dqBuf,
		dsBuf,
		d.NewBufferU32(uint32(sgI)),
		d.NewBufferU32(actSilu),
	)
	gotQ := dqBuf.Int8s()
	gotSc := dsBuf.Floats()[0]

	mism := 0
	var dot, na, nb float64
	for i := range sgI {
		if gotQ[i] != refQ[i] {
			mism++
		}
		gg := float64(gotQ[i]) * float64(gotSc)
		r := float64(refQ[i]) * float64(refSc)
		dot += gg * r
		na += gg * gg
		nb += r * r
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	mustFinite(t, "swiglu_quant cosine", cos)
	if cos < 0.99999 || mism > sgI/100 {
		t.Fatalf("swiglu_quant drifts from CPU reference: cosine=%.7f int8-mismatch=%d/%d", cos, mism, sgI)
	}
	t.Logf("swiglu_quant I=%d vs CPU: cosine=%.9f, int8 exact=%d/%d", sgI, cos, sgI-mism, sgI)
}

// BenchmarkSwigluQuant isolates swiglu_quant's own throughput at a realistic dispatch shape.
func BenchmarkSwigluQuant(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "swiglu_quant")
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}

	const sgI = 4096
	rng := rand.New(rand.NewSource(29))
	rndf := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*4 - 2
		}
		return s
	}
	dG := d.NewBufferFloats(rndf(sgI))
	dU := d.NewBufferFloats(rndf(sgI))
	dDq := d.NewBufferBytes(sgI)
	dDs := d.NewBufferLen(1)
	uI := d.NewBufferU32(uint32(sgI))
	uAct := d.NewBufferU32(1)

	q_ := d.NewCommandQueue()
	const reps = 8
	run := func() {
		e := q_.Begin()
		for range reps {
			e.Dispatch(pipe, 256, 256, dG, dU, dDq, dDs, uI, uAct)
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
