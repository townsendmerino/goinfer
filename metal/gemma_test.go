//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// geluTanhRef mirrors decoder/rmsnorm.go geluTanh — the CPU reference computes in f64; the
// kernel in f32 (the same trade the SwiGLU path ships with). Reference kept in f64 here so the
// test measures the real divergence rather than hiding it.
func geluTanhRef(x float32) float32 {
	const c = 0.7978845608028654 // sqrt(2/pi)
	x64 := float64(x)
	inner := c * (x64 + 0.044715*x64*x64*x64)
	return float32(0.5 * x64 * (1 + math.Tanh(inner)))
}

// TestGemma_rmsnormF32 validates the sandwich-norm kernel (in-place RMSNorm of a sublayer
// output, no fused quant) in BOTH addOne modes — Gemma's (1+w) offset vs plain w. CPU order is
// (v*inv)*(1+w[i]): the weight is applied AFTER the normalize.
func TestGemma_rmsnormF32(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "rmsnorm_f32")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	const H = 256
	const eps = 1e-6
	for _, addOne := range []bool{false, true} {
		name := "plain_w"
		if addOne {
			name = "add_one"
		}
		t.Run(name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(17))
			x := make([]float32, H)
			w := make([]float32, H)
			for i := range x {
				x[i] = rng.Float32()*2 - 1
				w[i] = rng.Float32()*0.5 - 0.25 // small, like a real Gemma norm weight
			}
			// CPU reference (f64 accumulation, as decoder/rmsnorm.go does).
			var ss float64
			for _, v := range x {
				ss += float64(v) * float64(v)
			}
			inv := float32(1 / math.Sqrt(ss/float64(H)+eps))
			want := make([]float32, H)
			for i := range x {
				g := w[i]
				if addOne {
					g = 1 + w[i]
				}
				want[i] = x[i] * inv * g
			}
			a := uint32(0)
			if addOne {
				a = 1
			}
			xBuf := d.NewBufferFloats(x)
			q := d.NewCommandQueue()
			q.Run1D(pipe, 256, 256, xBuf, d.NewBufferFloats(w),
				d.NewBufferU32(H), d.NewBufferFloats([]float32{eps}), d.NewBufferU32(a))
			got := xBuf.Floats()
			for i := range want {
				if dd := math.Abs(float64(got[i] - want[i])); dd > 1e-5 {
					t.Fatalf("[%d]=%.6f want %.6f (Δ=%.2e)", i, got[i], want[i], dd)
				}
			}
		})
	}
}

// TestGemma_gluAct validates the act selector in swiglu_quant: ACT_SILU (1) must reproduce the
// shipped SwiGLU exactly, and ACT_GELU_TANH (0) must match the CPU geluTanh. The ordinals are
// decoder.ActKind's iota, so a swap here would silently run Gemma as SwiGLU.
func TestGemma_gluAct(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "swiglu_quant")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	const I = 256
	rng := rand.New(rand.NewSource(29))
	g := make([]float32, I)
	u := make([]float32, I)
	for i := range g {
		g[i] = rng.Float32()*4 - 2 // span the activation's interesting range
		u[i] = rng.Float32()*2 - 1
	}
	silu := func(x float32) float32 { return x / (1 + float32(math.Exp(float64(-x)))) }

	for _, tc := range []struct {
		name string
		act  uint32
		f    func(float32) float32
	}{
		{"gelu_tanh", 0, geluTanhRef}, // ActGeluTanh — Gemma
		{"silu", 1, silu},             // ActSiLU — everything else admitted
	} {
		t.Run(tc.name, func(t *testing.T) {
			// CPU reference: s = act(g)*u, then symmetric int8 quant (scale = maxabs/127).
			s := make([]float32, I)
			var mx float32
			for i := range g {
				s[i] = tc.f(g[i]) * u[i]
				if a := float32(math.Abs(float64(s[i]))); a > mx {
					mx = a
				}
			}
			refSc := mx / 127
			if refSc == 0 {
				refSc = 1
			}
			dq := byteBuf(d, I)
			ds := d.NewBufferLen(1)
			q := d.NewCommandQueue()
			q.Run1D(pipe, 256, 256, d.NewBufferFloats(g), d.NewBufferFloats(u), dq, ds,
				d.NewBufferU32(I), d.NewBufferU32(tc.act))
			if gotSc := ds.Floats()[0]; math.Abs(float64(gotSc-refSc)) > 1e-5 {
				t.Fatalf("scale %.7f want %.7f", gotSc, refSc)
			}
			gotQ := dq.Int8s()
			for i := range s {
				want := int(math.Round(float64(s[i] / refSc)))
				want = clampI(want, -127, 127)
				// ±1 tolerance: the kernel's f32 exp/tanh vs the reference's f64 can land either
				// side of a rounding boundary. A wrong ACT would be off by far more.
				if dd := gotQ[i] - int8(want); dd > 1 || dd < -1 {
					t.Fatalf("[%d] q=%d want %d (act=%s, g=%.4f u=%.4f)", i, gotQ[i], want, tc.name, g[i], u[i])
				}
			}
		})
	}
}
