//go:build cuda

package cuda

import (
	"context"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/aikit/linalg"
)

// G-S19e (docs/tasks/task-multimodal-support-2026-10.md, "S19 follow-up step 2"): the float-activation int4 kernels
// against a CPU float64 reference (each int4 row dequantized by WeightMat.Row, dotted with the f32 activation), at the
// 26B-A4B's shapes, so the ragged-tail word walk runs (the down projections' K/8 is not a multiple of 64). Each planted
// defect must turn the comparison red.

const w4f32MaxRelErr = 1e-5

// w4f32Mat is a random int4 weight of N rows by K, quantized by aikit as the loader does.
func w4f32Mat(N, K int, seed uint32) linalg.WeightMat {
	f := make([]float32, N*K)
	s := seed
	for i := range f {
		s = s*1664525 + 1013904223
		f[i] = float32(int32(s>>8)%2000-1000) / 1000
	}
	return linalg.QuantizeInt4(f, N, K, 32)
}

// w4f32Act is a random f32 activation with a few large entries, like a real residual's.
func w4f32Act(K int, seed uint32) []float32 {
	a := make([]float32, K)
	s := seed
	for i := range a {
		s = s*1664525 + 1013904223
		a[i] = float32(int32(s>>8)%2000-1000) / 1000
		if i%97 == 0 {
			a[i] *= 40
		}
	}
	return a
}

// w4f32Ref is the reference row sums; defect plants a wrong reading of the weight: 1 the -8 offset dropped (each value +8),
// 2 each value scaled by the next group's scale.
func w4f32Ref(w *linalg.WeightMat, a []float32, defect int) []float64 {
	N, K := w.Rows(), w.Cols()
	_, s16, group, _ := w.Int4F16()
	ng := K / group
	out := make([]float64, N)
	row := make([]float32, K)
	for n := range N {
		w.Row(n, row)
		var acc float64
		for k := range K {
			v := float64(row[k])
			sc := float64(linalg.F16ToF32(s16[n*ng+k/group]))
			switch defect {
			case 1:
				v += 8 * sc
			case 2:
				q := math.Round(v / sc)
				v = q * float64(linalg.F16ToF32(s16[n*ng+(k/group+1)%ng]))
			}
			acc += v * float64(a[k])
		}
		out[n] = acc
	}
	return out
}

func w4f32RelErr(got []float32, want []float64) float64 {
	var maxDiff, maxRef float64
	for i := range got {
		maxDiff = max(maxDiff, math.Abs(float64(got[i])-want[i]))
		maxRef = max(maxRef, math.Abs(want[i]))
	}
	return maxDiff / maxRef
}

// naturalOrder undoes permuteFast: the planted nibble-order defect.
func naturalOrder(w uint32) uint32 {
	var o uint32
	for i := range 8 {
		o |= ((w >> (4 * nibblePosFast(i))) & 0xf) << (4 * i)
	}
	return o
}

func TestGemvW4F32_matchesCPUFloat(t *testing.T) {
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
	mod, err := cx.LoadModule(gemvW4F32PTX)
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	fn := func(name string) *gc.Function {
		f, err := mod.Function(name)
		if err != nil {
			t.Fatalf("Function(%s): %v", name, err)
		}
		return f
	}
	fDense, fMoE, fWacc := fn("gemv_w4f32"), fn("gemv_w4f32_moe"), fn("gemv_w4f32_moe_wacc")
	stream := mustStream(t, cx)
	up32 := func(v []uint32) *gc.Buffer[uint32] {
		b := mustAlloc[uint32](t, cx, len(v))
		if e := gc.CopyHtoD(bg, b, v); e != nil {
			t.Fatal(e)
		}
		return b
	}
	up16 := func(v []uint16) *gc.Buffer[uint16] {
		b := mustAlloc[uint16](t, cx, len(v))
		if e := gc.CopyHtoD(bg, b, v); e != nil {
			t.Fatal(e)
		}
		return b
	}
	upF := func(v []float32) *gc.Buffer[float32] {
		b := mustAlloc[float32](t, cx, len(v))
		if e := gc.CopyHtoD(bg, b, v); e != nil {
			t.Fatal(e)
		}
		return b
	}
	read := func(b *gc.Buffer[float32], n int) []float32 {
		if e := stream.Synchronize(bg); e != nil {
			t.Fatal(e)
		}
		out := make([]float32, n)
		if e := gc.CopyDtoH(bg, out, b); e != nil {
			t.Fatal(e)
		}
		return out
	}
	cfg := func(N int) gc.LaunchConfig {
		return gc.LaunchConfig{GridX: uint32((N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
	}

	// The dense kernel at the dense branch's two shapes, with and without the planted defects.
	for _, sh := range []struct {
		name string
		N, K int
	}{{"dense gate 2112x2816", 2112, 2816}, {"dense down 2816x2112", 2816, 2112}} {
		w := w4f32Mat(sh.N, sh.K, uint32(sh.N+sh.K))
		h, err := packWeight(&w)
		if err != nil {
			t.Fatal(err)
		}
		a := w4f32Act(sh.K, 7)
		dA, dGs, dOut := upF(a), up16(h.ws16), mustAlloc[float32](t, cx, sh.N)
		run := func(words []uint32) []float32 {
			dW := up32(words)
			defer dW.Close()
			// Cleared before every launch, so an output the kernel did not write reads 0 rather than a previous run's value.
			if e := gc.CopyHtoD(bg, dOut, make([]float32, sh.N)); e != nil {
				t.Fatal(e)
			}
			if e := fDense.LaunchOn(bg, stream, cfg(sh.N), gc.Arg(dW), gc.Arg(dA), gc.Arg(dGs), gc.ArgDevicePtr(0),
				gc.ArgValue(int32(sh.N)), gc.ArgValue(int32(sh.K/8)), gc.ArgValue(int32(sh.K/32)), gc.Arg(dOut), gc.ArgValue(int32(0))); e != nil {
				t.Fatal(e)
			}
			return read(dOut, sh.N)
		}
		got := run(h.wpk)
		for i, v := range got {
			if v == 0 {
				t.Fatalf("%s: output %d is still the zero dst was cleared to: the launch did not write it", sh.name, i)
			}
		}
		ref := w4f32Ref(&w, a, 0)
		e := w4f32RelErr(got, ref)
		t.Logf("%s: max relative error %.3g (bar %g)", sh.name, e, w4f32MaxRelErr)
		if e > w4f32MaxRelErr {
			t.Errorf("G-S19e FAIL %s: max relative error %.3g", sh.name, e)
		}
		nat := make([]uint32, len(h.wpk))
		for i, x := range h.wpk {
			nat[i] = naturalOrder(x)
		}
		for _, d := range []struct {
			name string
			e    float64
		}{
			{"nibble order not permuted", w4f32RelErr(run(nat), ref)},
			{"-8 offset dropped", w4f32RelErr(got, w4f32Ref(&w, a, 1))},
			{"group scale one group off", w4f32RelErr(got, w4f32Ref(&w, a, 2))},
		} {
			t.Logf("  planted %-26s: %.3g (red above %g: %v)", d.name, d.e, w4f32MaxRelErr, d.e > w4f32MaxRelErr)
			if d.e <= w4f32MaxRelErr {
				t.Errorf("G-S19e FAIL %s: planted defect %q stays under the bar (%.3g)", sh.name, d.name, d.e)
			}
		}
		dA.Close()
		dGs.Close()
		dOut.Close()
	}

	// The expert kernels at the experts' two shapes: three experts stacked, slot 0 routed to expert 2; wacc into a
	// non-zero dst with weight 0.37.
	for _, sh := range []struct {
		name string
		N, K int
	}{{"expert gate|up 1408x2816", 1408, 2816}, {"expert down 2816x704", 2816, 704}} {
		mats := []linalg.WeightMat{w4f32Mat(sh.N, sh.K, 11), w4f32Mat(sh.N, sh.K, 12), w4f32Mat(sh.N, sh.K, 13)}
		stack, err := packWeightStack(&mats[0], &mats[1], &mats[2])
		if err != nil {
			t.Fatal(err)
		}
		a := w4f32Act(sh.K, 9)
		dW, dGs, dA := up32(stack.wpk), up16(stack.ws16), upF(a)
		dIdx, dWgt := up32([]uint32{2, 0, 1}), upF([]float32{0.37, 0, 0})
		ref := w4f32Ref(&mats[2], a, 0)
		dOut := mustAlloc[float32](t, cx, sh.N)
		if e := fMoE.LaunchOn(bg, stream, cfg(sh.N), gc.Arg(dW), gc.Arg(dA), gc.Arg(dGs), gc.Arg(dIdx), gc.ArgValue(int32(0)),
			gc.ArgValue(int32(sh.N)), gc.ArgValue(int32(sh.N)), gc.ArgValue(int32(sh.K/8)), gc.ArgValue(int32(sh.K/32)), gc.Arg(dOut)); e != nil {
			t.Fatal(e)
		}
		e := w4f32RelErr(read(dOut, sh.N), ref)
		t.Logf("%s moe: max relative error %.3g", sh.name, e)
		if e > w4f32MaxRelErr {
			t.Errorf("G-S19e FAIL %s moe: max relative error %.3g", sh.name, e)
		}
		if e0 := w4f32RelErr(read(dOut, sh.N), w4f32Ref(&mats[0], a, 0)); e0 <= w4f32MaxRelErr {
			t.Errorf("%s moe: matches expert 0 as well as expert 2 (%.3g): the routing index is not read", sh.name, e0)
		}
		base := w4f32Act(sh.N, 21)
		if e := gc.CopyHtoD(bg, dOut, base); e != nil {
			t.Fatal(e)
		}
		if e := fWacc.LaunchOn(bg, stream, cfg(sh.N), gc.Arg(dW), gc.Arg(dA), gc.Arg(dGs), gc.Arg(dIdx), gc.Arg(dWgt), gc.ArgValue(int32(0)),
			gc.ArgValue(int32(sh.N)), gc.ArgValue(int32(sh.N)), gc.ArgValue(int32(sh.K/8)), gc.ArgValue(int32(sh.K/32)), gc.Arg(dOut)); e != nil {
			t.Fatal(e)
		}
		want := make([]float64, sh.N)
		for i := range want {
			want[i] = float64(base[i]) + 0.37*ref[i]
		}
		e = w4f32RelErr(read(dOut, sh.N), want)
		t.Logf("%s wacc: max relative error %.3g", sh.name, e)
		if e > w4f32MaxRelErr {
			t.Errorf("G-S19e FAIL %s wacc: max relative error %.3g", sh.name, e)
		}
		for _, b := range []interface{ Close() error }{dW, dGs, dA, dIdx, dWgt, dOut} {
			b.Close()
		}
	}
}

// TestGluF32_matchesReference: glu_f32 against glu_quant's activation computed in float64, both activations, on a split
// gate|up buffer (gOff 0, uOff I).
func TestGluF32_matchesReference(t *testing.T) {
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
	mod, err := cx.LoadModule(gemvW4F32PTX)
	if err != nil {
		t.Fatal(err)
	}
	f, err := mod.Function("glu_f32")
	if err != nil {
		t.Fatal(err)
	}
	stream := mustStream(t, cx)
	const I = 704
	gu := w4f32Act(2*I, 5)
	dGU := mustAlloc[float32](t, cx, 2*I)
	dOut := mustAlloc[float32](t, cx, I)
	defer dGU.Close()
	defer dOut.Close()
	if e := gc.CopyHtoD(bg, dGU, gu); e != nil {
		t.Fatal(e)
	}
	for _, act := range []struct {
		name string
		id   int32
		f    func(x float64) float64
	}{
		{"gelu-tanh", 0, func(x float64) float64 {
			return 0.5 * x * (1 + math.Tanh(0.7978845608028654*(x+0.044715*x*x*x)))
		}},
		{"silu", 1, func(x float64) float64 { return x / (1 + math.Exp(-x)) }},
	} {
		if e := f.LaunchOn(bg, stream, gc.LaunchConfig{GridX: uint32((I + 255) / 256), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			gc.Arg(dGU), gc.Arg(dGU), gc.ArgValue(int32(0)), gc.ArgValue(int32(I)), gc.ArgValue(int32(I)), gc.ArgValue(act.id), gc.Arg(dOut)); e != nil {
			t.Fatal(e)
		}
		if e := stream.Synchronize(bg); e != nil {
			t.Fatal(e)
		}
		got := make([]float32, I)
		if e := gc.CopyDtoH(bg, got, dOut); e != nil {
			t.Fatal(e)
		}
		want := make([]float64, I)
		for k := range want {
			want[k] = act.f(float64(gu[k])) * float64(gu[I+k])
		}
		e := w4f32RelErr(got, want)
		t.Logf("glu_f32 %s: max relative error %.3g", act.name, e)
		if e > w4f32MaxRelErr {
			t.Errorf("glu_f32 %s: max relative error %.3g (bar %g)", act.name, e, w4f32MaxRelErr)
		}
	}
}
