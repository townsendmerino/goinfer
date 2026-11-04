//go:build cuda

package kernels

import (
	"math"
	"math/rand"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/internal/mamba2"
)

// OP2b on CUDA, gate K (docs/tasks/task-forward-architecture-2026-10.md §4.7, "OP2b on CUDA"): the Mamba-2 gated norm against
// internal/mamba2's grouped norm, and the squared-ReLU quantize against the CPU's rule, on seeded inputs. The chain test over the
// tiny fixtures' real per-token inputs stays in package cuda (cuda/mamba_kernels_test.go), which captures them from a decoder
// forward.

var (
	adMamba    = adMod{"mamba", readPTX("mamba")}
	adNongated = adMod{"nongated_act", readPTX("nongated_act")}
)

// allZero reports whether every element of x is exactly zero.
func allZero(x []float32) bool {
	for _, v := range x {
		if v != 0 {
			return false
		}
	}
	return true
}

// mambaCosNonZero is the cosine and max |diff| of a against b, after asserting that neither is all zeros. A kernel that writes nothing, or a replay of
// a model closed under it, must fail the reading by name and not agree with a zero reference (cosF32 returns 1 for that case).
func mambaCosNonZero(t *testing.T, what string, a, b []float32) (cos, maxAbs float64) {
	t.Helper()
	if allZero(a) {
		t.Fatalf("%s: the reference output is all zeros: the reading is void", what)
	}
	if allZero(b) {
		t.Fatalf("%s: the kernel's output is all zeros: the reading is void", what)
	}
	return cosF32(a, b)
}

// TestMambaGNorm_groups is K for the gated norm at a group count the tiny fixtures lack (both have one group): 4 groups of 32 against the CPU's grouped norm
// (mamba2.GatedRMSNormGrouped), and D's norm plant, the same input taken as one group over the full width, must fail it. It also covers a group size
// that is not a multiple of the block (the 4B's is 896 against a block of 256).
func TestMambaGNorm_groups(t *testing.T) {
	r, _ := newAdRig(t)
	const eps = 1e-5
	rng := rand.New(rand.NewSource(9))
	rnd := func(n int, scale float64) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64() * scale)
		}
		return s
	}
	for _, c := range []struct{ dInner, groups int }{{128, 4}, {896 * 2, 2}} {
		y, z, w := rnd(c.dInner, 1), rnd(c.dInner, 1), rnd(c.dInner, 0.5)
		size := c.dInner / c.groups
		for g := range c.groups { // groups at different scales, so a norm over the full width cannot pass for a per-group one
			for i := g * size; i < (g+1)*size; i++ {
				y[i] *= float32(1 + 3*g)
			}
		}
		want := make([]float32, c.dInner)
		for i := range want {
			want[i] = y[i] * (z[i] / (1 + float32(math.Exp(float64(-z[i])))))
		}
		mamba2.GatedRMSNormGrouped(want, w, c.groups, eps, false)
		run := func(nGroups int) []float32 {
			dY, dZ, dW, dOut := r.f32(y), r.f32(z), r.f32(w), r.zeros(c.dInner)
			r.launch(adMamba, "mamba_gnorm", adCfg(nGroups, 1, 1, 256, 0), gc.Arg(dY), gc.Arg(dZ), gc.Arg(dW), gc.Arg(dOut), adI(c.dInner/nGroups), gc.ArgValue(float32(eps)), adI(0))
			return r.read(dOut, c.dInner)
		}
		cos, a := mambaCosNonZero(t, "gated norm", want, run(c.groups))
		t.Logf("gated norm, %d groups of %d: cosine %.9f, max |diff| %.3g", c.groups, size, cos, a)
		if !(cos >= 0.99999) || !(a <= 1e-3) {
			t.Errorf("the gated norm at %d groups drifts from the CPU: cosine %.9f, max |diff| %.3g", c.groups, cos, a)
		}
		cos, a = mambaCosNonZero(t, "gated norm plant", want, run(1))
		t.Logf("plant, one group over the full width: cosine %.6f, max |diff| %.3g", cos, a)
		if cos >= 0.99999 && a <= 1e-3 { // NaN is not inside the bar, so it reads as red
			t.Errorf("the norm taken over the full width where %d groups are asked for stayed inside the bar", c.groups)
		}
		r.release()
	}
}

// TestRelu2Quant_exact is K for relu2_quant: its packed int8 codes and scale equal, exactly, relu(u)^2 quantized by the rule glu_quant states (scale = max/127,
// codes rounded to nearest even and clamped to +-127). Exact because the activation is one f32 multiply and the quantization is the same arithmetic on both
// sides. I is a multiple of 4, as every real intermediate width is (the kernel packs four codes a word).
func TestRelu2Quant_exact(t *testing.T) {
	r, _ := newAdRig(t)
	rng := rand.New(rand.NewSource(3))
	const I = 1028
	u := make([]float32, I)
	for i := range u {
		u[i] = float32(rng.NormFloat64() * 2)
	}
	sq := make([]float32, I)
	var mx float32
	for i, v := range u {
		v = max(v, 0)
		sq[i] = v * v
		mx = max(mx, sq[i])
	}
	wantScale := mx / 127
	inv := 1 / wantScale
	want := make([]int8, I)
	for i, d := range sq {
		want[i] = int8(max(-127, min(127, int(math.RoundToEven(float64(d*inv))))))
	}
	dU, dScale, dScratch := r.f32(u), r.zeros(1), r.zeros(I)
	dQ := mustAlloc[int32](t, r.cx, I/4)
	r.free = append(r.free, func() { dQ.Close() })
	r.launch(adNongated, "relu2_quant", adCfg(1, 1, 1, 256, 256*4), gc.Arg(dU), adI(0), adI(I), gc.Arg(dQ), gc.Arg(dScale), gc.Arg(dScratch))
	packed := make([]int32, I/4)
	if err := gc.CopyDtoH(r.bg, packed, dQ); err != nil {
		t.Fatal(err)
	}
	got := make([]int8, 0, I)
	for _, w := range packed {
		for b := range 4 {
			got = append(got, int8(w>>(8*b)))
		}
	}
	nonzero := false
	for _, c := range got {
		nonzero = nonzero || c != 0
	}
	if !nonzero {
		t.Fatal("relu2_quant wrote only zero codes: the reading is void")
	}
	if s := r.read(dScale, 1)[0]; s != wantScale {
		t.Errorf("scale %v, want %v", s, wantScale)
	}
	bad := 0
	for i := range want {
		if got[i] != want[i] {
			if bad < 3 {
				t.Errorf("code %d: %d, want %d (u %v)", i, got[i], want[i], u[i])
			}
			bad++
		}
	}
	if bad > 0 {
		t.Errorf("%d of %d codes differ", bad, I)
	}
	r.release()
}
