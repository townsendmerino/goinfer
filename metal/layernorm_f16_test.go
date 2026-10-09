//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// G-B1 (docs/tasks/task-metal-pairwise-followups-2026-10.md, Part B): layernorm_f16 against a float64 reference, max abs diff
// <= 4e-3 on inputs in [-2, 2] shifted by a non-zero mean, and rmsnorm_f16 (no centring) on the same input must miss by at
// least 10x. layernorm_quant_f16's dequantized output must land within half a quantization step plus the f16 input rounding.
func TestLayerNormF16_matchesReference(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile prefillKernels: %v", err)
	}
	pipe := func(n string) Pipeline {
		p, err := d.NewComputePipeline(lib, n)
		if err != nil {
			t.Fatalf("pipeline %s: %v", n, err)
		}
		return p
	}
	q := d.NewCommandQueue()
	const M, H = 5, 256
	const eps = float32(1e-5)
	rng := rand.New(rand.NewSource(21))
	x := make([]uint16, M*H)
	w := make([]float32, H)
	for i := range x {
		x[i] = f32ToF16(rng.Float32()*4 - 2 + 0.7) // a non-zero mean, so centring matters
	}
	for i := range w {
		w[i] = 0.5 + rng.Float32()
	}
	ref := make([]float64, M*H)
	for m := range M {
		var s, ss float64
		for i := range H {
			s += float64(f16ToF32(x[m*H+i]))
		}
		mean := s / H
		for i := range H {
			d := float64(f16ToF32(x[m*H+i])) - mean
			ss += d * d
		}
		inv := 1 / math.Sqrt(ss/H+float64(eps))
		for i := range H {
			ref[m*H+i] = (float64(f16ToF32(x[m*H+i])) - mean) * inv * float64(w[i])
		}
	}
	run := func(kernel string) float64 {
		out := NewBufferU16s(d, make([]uint16, M*H))
		q.Run1D(pipe(kernel), M*128, 128, NewBufferU16s(d, x), NewBufferFloats(d, w), out, NewBufferU32(d, H),
			NewBufferFloats(d, []float32{eps}), NewBufferU32(d, 0))
		got := out.U16s()[:M*H]
		mx := 0.0
		for i := range got {
			mx = math.Max(mx, math.Abs(float64(f16ToF32(got[i]))-ref[i]))
		}
		return mx
	}
	ln, rms := run("layernorm_f16"), run("rmsnorm_f16")
	t.Logf("layernorm_f16 max|diff| %.3g (tol 4e-3) | rmsnorm_f16 control %.3g", ln, rms)
	if ln > 4e-3 {
		t.Errorf("layernorm_f16 max|diff| %.3g > 4e-3", ln)
	}
	if rms < 10*4e-3 {
		t.Errorf("the RMSNorm control is within 10x of the tolerance (%.3g): the case cannot tell centring from none", rms)
	}
	// layernorm_quant_f16, one row (the last-token final norm).
	aq := NewBufferInt8(d, make([]int8, H))
	asc := NewBufferFloats(d, []float32{0})
	q.Run1D(pipe("layernorm_quant_f16"), 128, 128, NewBufferU16s(d, x[(M-1)*H:]), NewBufferFloats(d, w), aq, asc,
		NewBufferU32(d, H), NewBufferFloats(d, []float32{eps}), NewBufferU32(d, 0))
	sc := float64(asc.Floats()[0])
	codes := aq.Int8s()[:H]
	mx := 0.0
	for i := range H {
		mx = math.Max(mx, math.Abs(float64(codes[i])*sc-ref[(M-1)*H+i]))
	}
	t.Logf("layernorm_quant_f16 max|dequant - ref| %.3g (step %.3g)", mx, sc)
	if mx > sc/2+1e-3 {
		t.Errorf("layernorm_quant_f16 off by %.3g, more than half a step (%.3g)", mx, sc/2)
	}
}
