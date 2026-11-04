//go:build darwin

package kernels

import (
	"math"
	"math/rand"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/mamba2"
)

// TestMambaGNorm_groups is the K gate for the gated norm at a group count the tiny fixtures lack (both have one group): the
// kernel at 4 groups of 32 against the CPU's own grouped norm (mamba2.GatedRMSNormGrouped), and D's norm plant, the
// same input taken as one group over the full width, must fail it.
func TestMambaGNorm_groups(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pNorm, err := d.NewComputePipeline(lib, "mamba_gnorm")
	if err != nil {
		t.Fatal(err)
	}
	const dInner, groups, eps = 128, 4, 1e-5
	rng := rand.New(rand.NewSource(9))
	rnd := func(n int, scale float64) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64() * scale)
		}
		return s
	}
	y, z, w := rnd(dInner, 1), rnd(dInner, 1), rnd(dInner, 0.5)
	// Groups at different scales, so a norm over the full width cannot pass for a per-group one.
	for g := range groups {
		for i := g * dInner / groups; i < (g+1)*dInner/groups; i++ {
			y[i] *= float32(1 + 3*g)
		}
	}
	want := make([]float32, dInner)
	for i := range want {
		want[i] = y[i] * (z[i] / (1 + float32(math.Exp(float64(-z[i])))))
	}
	mamba2.GatedRMSNormGrouped(want, w, groups, eps, false)
	run := func(nGroups int) []float32 {
		dY, dZ, dW, dOut := NewBufferFloats(d, y), NewBufferFloats(d, z), NewBufferFloats(d, w), d.NewBufferLen(dInner)
		q := d.NewCommandQueue()
		e := q.Begin()
		e.Dispatch(pNorm, nGroups*64, 64, dY, dZ, dW, dOut, NewBufferU32(d, uint32(dInner/nGroups)), NewBufferFloats(d, []float32{eps}), NewBufferU32(d, 0))
		e.End()
		return slices.Clone(dOut.Floats())
	}
	c, a := mambaCos(want, run(groups))
	t.Logf("gated norm, %d groups of %d: cosine %.9f, max |diff| %.3g", groups, dInner/groups, c, a)
	if c < 0.99999 || a > 1e-3 {
		t.Errorf("the gated norm at %d groups drifts from the CPU: cosine %.9f, max |diff| %.3g", groups, c, a)
	}
	c, a = mambaCos(want, run(1))
	t.Logf("plant, one group over the full width: cosine %.6f, max |diff| %.3g", c, a)
	if c >= 0.99999 && a <= 1e-3 {
		t.Error("the norm taken over the full width where 4 groups are asked for stayed inside the bar")
	}
}

// actQuantRun runs act_quant (compiled from src) on u at activation ordinal act and returns its int8 codes and scale.
func actQuantRun(t *testing.T, d *Device, src string, u []float32, act uint32) ([]int8, float32) {
	t.Helper()
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	p, err := d.NewComputePipeline(lib, "act_quant")
	if err != nil {
		t.Fatal(err)
	}
	dU, dQ, dS := NewBufferFloats(d, u), d.NewBufferLen(len(u)), d.NewBufferLen(1)
	q := d.NewCommandQueue()
	e := q.Begin()
	e.Dispatch(p, 256, 256, dU, dQ, dS, NewBufferU32(d, uint32(len(u))), NewBufferU32(d, act))
	e.End()
	return slices.Clone(dQ.Int8s()[:len(u)]), dS.Floats()[0]
}

// TestActQuant_relu2 is the K gate for act_quant's squared-ReLU branch (ordinal 2, decoder's ActReLU2): its dequantized output
// against relu(u)² quantized by act_quant's own rule (scale max/127, codes rounded and clamped to ±127), with the plant, plain
// ReLU, failing it. And the I gate for the branch's addition: at ordinals 0 (GELU-tanh) and 1 (SiLU) act_quant's codes and scale
// are bit-identical to the kernel compiled with glu_act in nongated_act's place, act_quant as it was before the branch.
func TestActQuant_relu2(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	rng := rand.New(rand.NewSource(3))
	u := make([]float32, 1027) // not a multiple of 4: the scalar tail runs too
	for i := range u {
		u[i] = float32(rng.NormFloat64() * 2)
	}
	ref := func(f func(float32) float32) []float32 {
		s := make([]float32, len(u))
		var mx float32
		for i, v := range u {
			s[i] = f(v)
			mx = max(mx, float32(math.Abs(float64(s[i]))))
		}
		sc := mx / 127
		out := make([]float32, len(u))
		for i, v := range s {
			out[i] = float32(max(-127, min(127, int(math.Round(float64(v/sc)))))) * sc
		}
		return out
	}
	deq := func(q []int8, sc float32) []float32 {
		out := make([]float32, len(q))
		for i, v := range q {
			out[i] = float32(v) * sc
		}
		return out
	}
	relu2 := func(x float32) float32 { r := max(x, 0); return r * r }
	q, sc := actQuantRun(t, d, allKernels, u, 2)
	c, a := mambaCos(ref(relu2), deq(q, sc))
	t.Logf("act_quant squared ReLU: cosine %.9f, max |diff| %.3g", c, a)
	if c < 0.99999 || a > 1e-3 {
		t.Errorf("act_quant's squared ReLU drifts from relu(u)²: cosine %.9f, max |diff| %.3g", c, a)
	}
	from, to := "if (act == ACT_RELU2) { float r = max(x, 0.0f); return r*r; }", "if (act == ACT_RELU2) { return max(x, 0.0f); }"
	if strings.Count(allKernels, from) != 1 {
		t.Fatalf("the library must contain %q exactly once", from)
	}
	q, sc = actQuantRun(t, d, strings.Replace(allKernels, from, to, 1), u, 2)
	if c, a := mambaCos(ref(relu2), deq(q, sc)); c >= 0.99999 && a <= 1e-3 {
		t.Errorf("the plant, plain ReLU, stayed inside the bar: cosine %.6f, max |diff| %.3g", c, a)
	}
	// I: the old act_quant is the library with glu_act called where nongated_act is.
	start := strings.Index(allKernels, "kernel void act_quant(")
	end := start + strings.Index(allKernels[start:], "\n}\n")
	old := allKernels[:start] + strings.ReplaceAll(allKernels[start:end], "nongated_act(", "glu_act(") + allKernels[end:]
	if start < 0 || end <= start || old == allKernels || strings.Count(allKernels[start:end], "nongated_act(") == 0 {
		t.Fatal("could not rebuild the act_quant before the branch: the identity check would compare a kernel with itself")
	}
	for _, act := range []uint32{0, 1} {
		qn, sn := actQuantRun(t, d, allKernels, u, act)
		qo, so := actQuantRun(t, d, old, u, act)
		if !slices.Equal(qn, qo) || math.Float32bits(sn) != math.Float32bits(so) {
			t.Errorf("act_quant at ordinal %d is not bit-identical to the kernel before the squared-ReLU branch", act)
		}
	}
}

// mambaCos is the cosine and max |diff| of a against b, with an all-zero vector against a nonzero one scored 0, not 1: a kernel
// that writes nothing must not read as a match. A copy of the helper in metal/mamba_kernels_test.go, whose chain tests stay in
// package metal.
func mambaCos(a, b []float32) (cos, maxAbs float64) {
	cos, maxAbs = cosF32(a, b)
	var na, nb float64
	for i := range a {
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if (na == 0) != (nb == 0) {
		cos = 0
	}
	return cos, maxAbs
}
