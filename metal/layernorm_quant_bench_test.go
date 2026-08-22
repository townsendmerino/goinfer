//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestLayernormQuant_cpuParity gates layernorm_quant (metal/kernels.go) sourced from the real
// allKernels -- GPT-2/Cohere's fused LayerNorm+quant norm kernel (model.go: r.pLayerNorm, launch
// shape (256, 256)).
func TestLayernormQuant_cpuParity(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "layernorm_quant")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const lnH = 4096
	const eps = 1e-5
	rng := rand.New(rand.NewSource(13))
	x := make([]float32, lnH)
	w := make([]float32, lnH)
	bias := make([]float32, lnH)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
		w[i] = 0.5 + rng.Float32()
		bias[i] = rng.Float32()*0.2 - 0.1
	}

	var sum float64
	for _, v := range x {
		sum += float64(v)
	}
	mean := float32(sum / float64(lnH))
	var ss float64
	for _, v := range x {
		d := float64(v) - float64(mean)
		ss += d * d
	}
	inv := float32(1 / math.Sqrt(ss/float64(lnH)+eps))
	y := make([]float32, lnH)
	var mx float32
	for i := range x {
		y[i] = (x[i]-mean)*inv*w[i] + bias[i]
		if a := float32(math.Abs(float64(y[i]))); a > mx {
			mx = a
		}
	}
	refSc := mx / 127
	if refSc == 0 {
		refSc = 1
	}
	refQ := make([]int8, lnH)
	for i := range y {
		q := max(min(int(math.Round(float64(y[i]/refSc))), 127), -127)
		refQ[i] = int8(q)
	}

	q := d.NewCommandQueue()
	aqBuf := d.NewBufferBytes(lnH)
	ascBuf := d.NewBufferLen(1)
	q.Run1D(pipe, 256, 256,
		NewBufferFloats(d, x),
		NewBufferFloats(d, w),
		NewBufferFloats(d, bias),
		aqBuf,
		ascBuf,
		NewBufferU32(d, uint32(lnH)),
		NewBufferFloats(d, []float32{eps}),
		NewBufferU32(d, 1), // hasBias
	)
	gotQ := aqBuf.Int8s()
	gotSc := ascBuf.Floats()[0]

	mism := 0
	var dot, na, nb float64
	for i := range lnH {
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
	mustFinite(t, "layernorm_quant cosine", cos)
	if cos < 0.99999 || mism > lnH/100 {
		t.Fatalf("layernorm_quant drifts from CPU reference: cosine=%.7f int8-mismatch=%d/%d", cos, mism, lnH)
	}
	t.Logf("layernorm_quant H=%d vs CPU: cosine=%.9f, int8 exact=%d/%d", lnH, cos, lnH-mism, lnH)
}

// BenchmarkLayernormQuant isolates layernorm_quant's own throughput at production dispatch shape
// (256 threads, one threadgroup).
func BenchmarkLayernormQuant(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "layernorm_quant")
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}

	const lnH = 4096
	rng := rand.New(rand.NewSource(13))
	rndf := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*2 - 1
		}
		return s
	}
	dX := NewBufferFloats(d, rndf(lnH))
	dW := NewBufferFloats(d, rndf(lnH))
	dB := NewBufferFloats(d, rndf(lnH))
	dAq := d.NewBufferBytes(lnH)
	dAsc := d.NewBufferLen(1)
	uH := NewBufferU32(d, uint32(lnH))
	uEps := NewBufferFloats(d, []float32{1e-5})
	uHasBias := NewBufferU32(d, 1)

	q_ := d.NewCommandQueue()
	const reps = 8
	run := func() {
		e := q_.Begin()
		for range reps {
			e.Dispatch(pipe, 256, 256, dX, dW, dB, dAq, dAsc, uH, uEps, uHasBias)
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
