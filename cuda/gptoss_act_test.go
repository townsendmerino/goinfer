//go:build cuda && goinfer_testhooks

// Gate for gpt-oss's clamped interleaved-SwiGLU expert epilogue (residency step 2 of 2).
//
// Checked against the CPU reference in decoder/forward_gptoss.go, which is itself matched to
// HF's GptOssExperts:
//
//	gate = clamp(Gate·h + gateBias, max = limit)      // UPPER only
//	up   = clamp(Up·h   + upBias,   [-limit, limit])  // BOTH
//	glu  = gate · sigmoid(alpha · gate)
//	d    = (up + 1) · glu
//
// THE INPUTS DELIBERATELY SATURATE BOTH BRANCHES. A clamp that is symmetric on the gate — the
// tidier-looking mistake — agrees everywhere the activation does not saturate, which is most
// random data. It differs only past the limit, i.e. exactly where a clamp exists to matter.
// So the test drives values well past ±limit on both branches and compares against a
// reference that encodes the asymmetry.
//
// The comparison is on the DEQUANTIZED int8 output, because that is what the down-projection
// GEMV consumes; a tolerance of one quantization step is therefore expected and anything
// larger is a real disagreement.
package cuda

import (
	"context"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

func TestGptOssAct_matchesReference(t *testing.T) {
	const I = 256
	const alpha, limit = float32(1.702), float32(7.0)

	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	cx, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	defer cx.Close()
	bg := context.Background()
	mod, err := cx.LoadModule(gptOssActPTX)
	if err != nil {
		t.Fatalf("LoadModule(gptoss_act): %v", err)
	}
	fn, err := mod.Function("glu_quant_gptoss")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}
	stream := mustStream(t, cx)

	// Spread well past ±limit so BOTH clamps engage on many elements.
	g := make([]float32, I)
	u := make([]float32, I)
	gb := make([]float32, I)
	ub := make([]float32, I)
	for i := range g {
		g[i] = float32(math.Sin(float64(i)*0.37)) * 12 // |g| up to 12 > limit 7
		u[i] = float32(math.Cos(float64(i)*0.29)) * 11
		gb[i] = float32(math.Sin(float64(i)*1.1)) * 0.5
		ub[i] = float32(math.Cos(float64(i)*0.9)) * 0.5
	}
	clampedHi, clampedLo := 0, 0
	for i := range g {
		if g[i]+gb[i] > limit {
			clampedHi++
		}
		if u[i]+ub[i] < -limit {
			clampedLo++
		}
	}
	if clampedHi == 0 || clampedLo == 0 {
		t.Fatalf("inputs do not exercise both clamps (%d over, %d under) — the test would pass "+
			"against a symmetric clamp", clampedHi, clampedLo)
	}
	t.Logf("clamp coverage: %d elements over +limit, %d under -limit", clampedHi, clampedLo)

	dG := mustAlloc[float32](t, cx, I)
	dU := mustAlloc[float32](t, cx, I)
	// Bias table for TWO experts, and the routing index selects expert 1 — so the test
	// exercises the on-device biasRow = idx[slot]*2*I arithmetic rather than only the
	// activation. Expert 0 is filled with a decoy that would be visibly wrong if used.
	const nExp, slot = 2, 0
	biasGU := make([]float32, nExp*2*I)
	for k := range I {
		biasGU[k] = 99        // expert 0 gate bias: decoy
		biasGU[I+k] = -99     // expert 0 up bias: decoy
		biasGU[2*I+k] = gb[k] // expert 1 gate bias (the one idx selects)
		biasGU[3*I+k] = ub[k] // expert 1 up bias
	}
	dBias := mustAlloc[float32](t, cx, len(biasGU))
	dIdx := mustAlloc[int32](t, cx, 1)
	dQ := mustAlloc[int32](t, cx, I/4)
	dScale := mustAlloc[float32](t, cx, 1)
	dScratch := mustAlloc[float32](t, cx, I)
	for _, cp := range []struct {
		d *gc.Buffer[float32]
		h []float32
	}{{dG, g}, {dU, u}, {dBias, biasGU}} {
		if e := gc.CopyHtoD(bg, cp.d, cp.h); e != nil {
			t.Fatalf("H2D: %v", e)
		}
	}
	if e := gc.CopyHtoD(bg, dIdx, []int32{1}); e != nil { // select expert 1
		t.Fatalf("H2D idx: %v", e)
	}

	cfg := gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 4}
	if e := fn.LaunchOn(bg, stream, cfg,
		gc.Arg(dG), gc.Arg(dU), gc.ArgValue(int32(0)), gc.ArgValue(int32(0)), gc.ArgValue(int32(I)),
		gc.Arg(dBias), gc.Arg(dIdx), gc.ArgValue(int32(slot)), gc.ArgValue(alpha), gc.ArgValue(limit),
		gc.Arg(dQ), gc.Arg(dScale), gc.Arg(dScratch)); e != nil {
		t.Fatalf("launch: %v", e)
	}
	if e := stream.Synchronize(bg); e != nil {
		t.Fatalf("sync: %v", e)
	}
	qs := make([]int32, I/4)
	sc := make([]float32, 1)
	if e := gc.CopyDtoH(bg, qs, dQ); e != nil {
		t.Fatalf("D2H q: %v", e)
	}
	if e := gc.CopyDtoH(bg, sc, dScale); e != nil {
		t.Fatalf("D2H scale: %v", e)
	}

	// CPU reference — decoder/forward_gptoss.go's gptOssExpert, activation only.
	want := make([]float32, I)
	for i := range want {
		gx := g[i] + gb[i]
		ux := u[i] + ub[i]
		if gx > limit {
			gx = limit // upper clamp ONLY
		}
		if ux > limit {
			ux = limit
		} else if ux < -limit {
			ux = -limit
		}
		glu := gx * float32(1.0/(1.0+math.Exp(-float64(alpha*gx))))
		want[i] = (ux + 1) * glu
	}

	// Compare dequantized: one step of the symmetric int8 scale is the expected floor.
	step := float64(sc[0])
	worst, at := 0.0, -1
	for i := range want {
		code := int8(uint32(qs[i/4]) >> (8 * (i % 4)))
		got := float64(code) * step
		d := math.Abs(got - float64(want[i]))
		if d > worst {
			worst, at = d, i
		}
	}
	t.Logf("clamped-SwiGLU: scale %.6g, max|diff| %.3e (one int8 step = %.3e)", step, worst, step)
	if worst > 1.5*step {
		t.Errorf("max|diff| %.3e at %d exceeds 1.5 quantization steps (%.3e) — a real disagreement, "+
			"not rounding", worst, at, 1.5*step)
	}
}
