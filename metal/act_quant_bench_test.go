//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestActQuant_cpuParity gates act_quant (metal/kernels.go) sourced from the real allKernels --
// GPT-2/Nemotron's fused activation+quant (non-gated MLP; model.go: r.pActQuant, launch shape
// (256, 256), dispatched once per layer).
func TestActQuant_cpuParity(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "act_quant")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const aqI = 3072 // GPT-2's default 4*n_embd(768)
	const actGeluTanh = 0
	rng := rand.New(rand.NewSource(23))
	u := make([]float32, aqI)
	for i := range u {
		u[i] = rng.Float32()*4 - 2
	}

	geluTanh := func(x float64) float64 {
		return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x)))
	}
	s := make([]float64, aqI)
	var mx float64
	for i, v := range u {
		s[i] = geluTanh(float64(v))
		if a := math.Abs(s[i]); a > mx {
			mx = a
		}
	}
	refSc := float32(mx / 127)
	if refSc == 0 {
		refSc = 1
	}
	refQ := make([]int8, aqI)
	for i := range s {
		q := max(min(int(math.Round(s[i]/float64(refSc))), 127), -127)
		refQ[i] = int8(q)
	}

	q := d.NewCommandQueue()
	dqBuf := d.NewBufferBytes(aqI)
	dsBuf := d.NewBufferLen(1)
	q.Run1D(pipe, 256, 256,
		d.NewBufferFloats(u),
		dqBuf,
		dsBuf,
		d.NewBufferU32(uint32(aqI)),
		d.NewBufferU32(actGeluTanh),
	)
	gotQ := dqBuf.Int8s()
	gotSc := dsBuf.Floats()[0]

	mism := 0
	var dot, na, nb float64
	for i := range aqI {
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
	mustFinite(t, "act_quant cosine", cos)
	if cos < 0.99999 || mism > aqI/100 {
		t.Fatalf("act_quant drifts from CPU reference: cosine=%.7f int8-mismatch=%d/%d", cos, mism, aqI)
	}
	t.Logf("act_quant I=%d vs CPU: cosine=%.9f, int8 exact=%d/%d", aqI, cos, aqI-mism, aqI)
}

// BenchmarkActQuant isolates act_quant's own throughput at production dispatch shape.
func BenchmarkActQuant(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "act_quant")
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}

	const aqI = 3072
	rng := rand.New(rand.NewSource(23))
	rndf := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*4 - 2
		}
		return s
	}
	dU := d.NewBufferFloats(rndf(aqI))
	dDq := d.NewBufferBytes(aqI)
	dDs := d.NewBufferLen(1)
	uI := d.NewBufferU32(uint32(aqI))
	uAct := d.NewBufferU32(0)

	q_ := d.NewCommandQueue()
	const reps = 8
	run := func() {
		e := q_.Begin()
		for range reps {
			e.Dispatch(pipe, 256, 256, dU, dDq, dDs, uI, uAct)
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
