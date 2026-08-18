//go:build darwin

// GPT-2's two remaining missing kernels (LayerNorm, non-gated-MLP activation), gated standalone
// against decoder/rmsnorm.go's layerNorm and decoder/mlp.go's nonGatedMLP before any wiring into
// model.go's real dispatch. FeatOutBias's kernel (gemv_w4a8_sa_bias_resid) and FeatLearnedPos
// (no per-layer kernel — it's a dispatch-skip + host-side embedding add) are the other two
// pieces; see gemv_w4a8_sa_bias_resid_test.go and model.go respectively.
package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestLayerNormQuant gates layernorm_quant (kernels.go) against decoder/rmsnorm.go's layerNorm:
// y = (x-mean)/sqrt(var+eps)*w [+ b], then int8-quantized. Checks BOTH the biased case (GPT-2)
// and the bias-free case (Cohere's hasBias=0 path) — the CPU reference itself branches on a nil
// bias, so a test that only exercises one branch could pass with the other silently broken.
func TestLayerNormQuant(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "layernorm_quant")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const H = 768 // GPT-2 small hidden dim
	const eps = 1e-5
	rng := rand.New(rand.NewSource(41))
	x := make([]float32, H)
	w := make([]float32, H)
	b := make([]float32, H)
	for i := range x {
		x[i] = rng.Float32()*4 - 2 // deliberately not zero-mean, so the mean-subtraction actually matters
		w[i] = rng.Float32()*0.5 + 0.75
		b[i] = rng.Float32()*0.4 - 0.2
	}

	// Matches decoder/rmsnorm.go's layerNorm exactly (unexported, so reimplemented here — the
	// same pattern every other metal kernel test uses for its CPU reference).
	ref := func(hasBias bool) []float32 {
		var mean float64
		for _, v := range x {
			mean += float64(v)
		}
		mean /= float64(H)
		var variance float64
		for _, v := range x {
			d := float64(v) - mean
			variance += d * d
		}
		variance /= float64(H)
		inv := 1.0 / math.Sqrt(variance+eps)
		out := make([]float32, H)
		for i, v := range x {
			y := float32((float64(v)-mean)*inv) * w[i]
			if hasBias {
				y += b[i]
			}
			out[i] = y
		}
		return out
	}

	q := d.NewCommandQueue()
	run := func(hasBias bool) []float32 {
		aq := d.NewBufferBytes(H)
		asc := d.NewBufferLen(1)
		hb := uint32(0)
		if hasBias {
			hb = 1
		}
		q.Run1D(pipe, 256, 256, d.NewBufferFloats(x), d.NewBufferFloats(w), d.NewBufferFloats(b),
			aq, asc, d.NewBufferU32(uint32(H)), d.NewBufferFloats([]float32{eps}), d.NewBufferU32(hb))
		sc := asc.Floats()[0]
		out := make([]float32, H)
		for i, v := range aq.Int8s() {
			out[i] = float32(v) * sc
		}
		return out
	}

	cmp := func(name string, got, want []float32) {
		var dot, na, nb float64
		for i := range want {
			dot += float64(got[i]) * float64(want[i])
			na += float64(got[i]) * float64(got[i])
			nb += float64(want[i]) * float64(want[i])
		}
		cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
		mustFinite(t, name, cos)
		if cos < 0.9999 {
			t.Errorf("%s: cosine %.7f < 0.9999", name, cos)
		}
		t.Logf("%s: cosine %.7f", name, cos)
	}
	cmp("LayerNorm with bias (GPT-2)", run(true), ref(true))
	cmp("LayerNorm bias-free (Cohere)", run(false), ref(false))
}

// TestActQuant gates act_quant (kernels.go) against decoder/mlp.go's nonGatedMLP's activation
// stage (GELU-tanh only — glu_act's only relevant branch for GPT-2's "gelu_new"). Values
// deliberately span well past GELU-tanh's saturation region so a wrong or unclamped tanh
// argument (the exact class of bug 38a2b7c fixed for Gemma) would show up here too.
func TestActQuant(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "act_quant")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const I = 3072 // GPT-2 small's 4*768 FFN width
	rng := rand.New(rand.NewSource(43))
	u := make([]float32, I)
	for i := range u {
		u[i] = (rng.Float32()*2 - 1) * 14 // up to |14|, well past GELU-tanh's saturation
	}

	geluTanhF64 := func(x float64) float64 {
		const c = 0.7978845608028654 // sqrt(2/pi)
		return 0.5 * x * (1 + math.Tanh(c*(x+0.044715*x*x*x)))
	}
	want := make([]float32, I)
	var mx float32
	for i, v := range u {
		want[i] = float32(geluTanhF64(float64(v)))
		if a := float32(math.Abs(float64(want[i]))); a > mx {
			mx = a
		}
	}
	refSc := mx / 127
	refQ := make([]int8, I)
	for i, v := range want {
		refQ[i] = int8(max(min(int(math.Round(float64(v/refSc))), 127), -127))
	}

	q := d.NewCommandQueue()
	dq := d.NewBufferBytes(I)
	ds := d.NewBufferLen(1)
	q.Run1D(pipe, 256, 256, d.NewBufferFloats(u), dq, ds, d.NewBufferU32(uint32(I)), d.NewBufferU32(0)) // act=0 (GELU-tanh)
	gotQ := dq.Int8s()
	gotSc := ds.Floats()[0]

	var dot, na, nb float64
	for i := range I {
		gg := float64(gotQ[i]) * float64(gotSc)
		rr := float64(refQ[i]) * float64(refSc)
		dot += gg * rr
		na += gg * gg
		nb += rr * rr
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	mustFinite(t, "act_quant cosine", cos)
	if cos < 0.9999 {
		t.Fatalf("act_quant (GELU-tanh) parity FAIL: cosine=%.7f", cos)
	}
	t.Logf("act_quant GELU-tanh I=%d vs CPU: cosine=%.7f — PARITY", I, cos)
}
