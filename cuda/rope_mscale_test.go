//go:build cuda

package cuda

import (
	"context"
	"math"
	"math/rand"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestRopeMscale is the CUDA twin of metal/rope_test.go's TestRope_mscale: it proves the
// YaRN attention_factor IN ISOLATION, before any family is declared resident on the strength
// of it.
//
// WHY IN ISOLATION, AND WHY BEFORE THE DECLARATION. `FeatRopeMscale` is a claim that this
// backend can express YaRN's cos/sin scaling. Until 2026-08-31 CUDA's three rope kernels took
// no scale parameter at all — so declaring the feature would have admitted gpt-oss AND (as a
// documented side effect, since CUDA already declares Mellum's other four required features)
// Mellum onto a path that silently ignores the factor. Silently: the kernel is correct
// arithmetic, just the wrong arithmetic, and nothing errors. A gate that runs a whole model
// and checks a cosine can miss a scalar this small; this one cannot, because it compares
// against a scalar reference computed in Go.
//
// The two assertions are deliberately different questions:
//
//	scale == 1.0   must reproduce the PRE-EXISTING unscaled rotation. Every family without
//	               YaRN dispatches this kernel every layer of every token, so a regression in
//	               the no-op path is the expensive failure, not the exotic one.
//	scale == 0.85  must match the scaled reference AND must NOT match the unscaled one. The
//	               second half is what catches a parameter that is accepted and then dropped —
//	               which is exactly what a signature-only change looks like if the multiply is
//	               forgotten, and is indistinguishable from success without it.
func TestRopeMscale(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Skipf("no context: %v", err)
	}
	defer ctx.Close()
	bg := context.Background()

	glmod, err := ctx.LoadModule(gluePTX)
	if err != nil {
		t.Fatalf("glue module: %v", err)
	}
	fRope, err := glmod.Function("rope")
	if err != nil {
		t.Fatalf("rope function: %v", err)
	}

	const nH, hd, pos = 4, 16, 11
	half := hd / 2

	invf := make([]float32, half)
	for i := range invf {
		invf[i] = float32(1.0 / math.Pow(10000, float64(2*i)/float64(hd)))
	}

	// ref mirrors decoder/rope.go applyRoPE: the scale is folded into cos/sin, NOT applied to
	// the rotated output. Those two differ, and the second is the wrong one — the kernel
	// comments say so precisely because it is an easy place to be plausibly wrong.
	ref := func(x0 []float32, scale float32) []float32 {
		x := append([]float32(nil), x0...)
		for head := range nH {
			for d := range half {
				b := head * hd
				th := float64(pos) * float64(invf[d])
				c := float32(math.Cos(th)) * scale
				s := float32(math.Sin(th)) * scale
				a, bb := x[b+d], x[b+half+d]
				x[b+d] = a*c - bb*s
				x[b+half+d] = a*s + bb*c
			}
		}
		return x
	}

	rng := rand.New(rand.NewSource(23))
	x := make([]float32, nH*hd)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	stream := mustStream(t, ctx)

	invB := mustAlloc[float32](t, ctx, half)
	if err := gc.CopyHtoD(bg, invB, invf); err != nil {
		t.Fatalf("upload invf: %v", err)
	}

	run := func(scale float32) []float32 {
		t.Helper()
		xb := mustAlloc[float32](t, ctx, len(x))
		if err := gc.CopyHtoD(bg, xb, x); err != nil {
			t.Fatalf("upload x: %v", err)
		}
		cfg := gc.LaunchConfig{
			GridX: uint32((nH*half + 255) / 256), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1,
		}
		if err := fRope.LaunchOn(bg, stream, cfg,
			gc.Arg(xb), gc.Arg(invB),
			gc.ArgValue(int32(nH)), gc.ArgValue(int32(hd)), gc.ArgValue(int32(pos)),
			gc.ArgValue(scale)); err != nil {
			t.Fatalf("launch (scale=%v): %v", scale, err)
		}
		// Synchronize BEFORE reading back: LaunchOn is asynchronous on the stream, and without
		// this the first readback races the kernel and returns the still-unrotated upload. That
		// failure is not subtle-looking but it IS misleading — it made scale=1 look broken while
		// scale=0.85 passed, i.e. it framed a test bug as a kernel bug.
		if err := stream.Synchronize(bg); err != nil {
			t.Fatalf("sync (scale=%v): %v", scale, err)
		}
		out := make([]float32, len(x))
		if err := gc.CopyDtoH(bg, out, xb); err != nil {
			t.Fatalf("readback (scale=%v): %v", scale, err)
		}
		return out
	}

	cmp := func(name string, got, want []float32) {
		t.Helper()
		var worst float64
		for i := range want {
			if d := math.Abs(float64(got[i] - want[i])); d > worst {
				worst = d
			}
		}
		t.Logf("%s: max|diff| = %.3e", name, worst)
		if worst > 1e-5 {
			t.Errorf("%s: max|diff| %.3e exceeds 1e-5", name, worst)
		}
	}

	cmp("scale=1 (no-op, the path every non-YaRN family takes)", run(1.0), ref(x, 1.0))

	const realScale = float32(0.85)
	got := run(realScale)
	cmp("scale=0.85", got, ref(x, realScale))

	unscaled := ref(x, 1.0)
	for i := range got {
		if math.Abs(float64(got[i]-unscaled[i])) > 1e-4 {
			return // differs from unscaled, as it must
		}
	}
	t.Error("scale=0.85 is indistinguishable from scale=1 — the parameter is accepted and then ignored, " +
		"which is the failure this test exists for: FeatRopeMscale would be declared on a kernel that drops the factor")
}
