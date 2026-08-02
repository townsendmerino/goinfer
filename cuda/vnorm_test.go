//go:build cuda

package cuda

import (
	"context"
	"fmt"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestQKNorm_widths sweeps the per-head qk_norm kernel across head widths — 128 (Qwen3), 256
// (Gemma 3's max), and 512 (Gemma 4's global head, which NO prior model reached, so this width
// has never been exercised for qk_norm). qk_norm and v_norm are the SAME kernel (v_norm reuses
// it with nH=0 / unit weight), and both do a per-head RMS reduction over hd on a fixed 128-thread
// block: 1 element/thread at 128, 4 at 512. That multi-element reduction path is the suspect for
// the two-geometry K=V parity drift (wrong K/V from position 0, compounding through the cache).
// Compares BOTH the weighted path (Q/K with a learned norm) and the scale-less path (V) to the
// CPU RMSNorm oracle, per element (relative error — a reduction bug or a 2x is visible; cosine
// would hide a uniform scale).
func TestQKNorm_widths(t *testing.T) {
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
	qmod, err := ctx.LoadModule(fusedQKVPTX)
	if err != nil {
		t.Fatalf("fusedQKV module: %v", err)
	}
	fQKN, err := qmod.Function("qk_norm")
	if err != nil {
		t.Fatalf("qk_norm: %v", err)
	}
	stream := mustStream(t, ctx)
	const nH, nKV = 4, 2
	const eps = float32(1e-6)

	cpuNorm := func(x, w []float32, rows, hd int) []float32 {
		out := append([]float32(nil), x...)
		for r := 0; r < rows; r++ {
			var ss float64
			for d := 0; d < hd; d++ {
				ss += float64(out[r*hd+d]) * float64(out[r*hd+d])
			}
			inv := float32(1.0 / math.Sqrt(ss/float64(hd)+float64(eps)))
			for d := 0; d < hd; d++ {
				out[r*hd+d] *= inv * w[d]
			}
		}
		return out
	}
	maxRel := func(got, ref []float32) float64 {
		var m float64
		for i := range got {
			if r := math.Abs(float64(got[i]-ref[i])) / (math.Abs(float64(ref[i])) + 1e-6); r > m {
				m = r
			}
		}
		return m
	}

	for _, hd := range []int{128, 256, 512} {
		t.Run(fmt.Sprintf("hd%d", hd), func(t *testing.T) {
			qh := make([]float32, nH*hd)
			kh := make([]float32, nKV*hd)
			qw := make([]float32, hd)
			kw := make([]float32, hd)
			for i := range qh {
				qh[i] = float32(math.Sin(float64(i)*0.11)) * 3
			}
			for i := range kh {
				kh[i] = float32(math.Cos(float64(i)*0.07)) * 2
			}
			for i := range qw {
				qw[i] = 1 + float32(i%5)*0.1
				kw[i] = 1 + float32(i%3)*0.2
			}
			refQ := cpuNorm(qh, qw, nH, hd)
			refK := cpuNorm(kh, kw, nKV, hd)
			unit := make([]float32, hd)
			for i := range unit {
				unit[i] = 1.0
			}
			refV := cpuNorm(kh, unit, nKV, hd)

			dq := mustAlloc[float32](t, ctx, nH*hd)
			dk := mustAlloc[float32](t, ctx, nKV*hd)
			dqw := mustAlloc[float32](t, ctx, hd)
			dkw := mustAlloc[float32](t, ctx, hd)
			du := mustAlloc[float32](t, ctx, hd)
			_ = gc.CopyHtoD(bg, dq, qh)
			_ = gc.CopyHtoD(bg, dk, kh)
			_ = gc.CopyHtoD(bg, dqw, qw)
			_ = gc.CopyHtoD(bg, dkw, kw)
			_ = gc.CopyHtoD(bg, du, unit)

			_ = fQKN.LaunchOn(bg, stream, gc.LaunchConfig{GridX: uint32(nH + nKV), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
				gc.Arg(dq), gc.Arg(dk), gc.Arg(dqw), gc.Arg(dkw),
				gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)), gc.ArgValue(eps), gc.ArgValue(int32(0)))
			_ = stream.Synchronize(bg)
			gotQ := make([]float32, nH*hd)
			gotK := make([]float32, nKV*hd)
			_ = gc.CopyDtoH(bg, gotQ, dq)
			_ = gc.CopyDtoH(bg, gotK, dk)

			dv := mustAlloc[float32](t, ctx, nKV*hd)
			_ = gc.CopyHtoD(bg, dv, kh)
			_ = fQKN.LaunchOn(bg, stream, gc.LaunchConfig{GridX: uint32(nKV), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
				gc.Arg(dv), gc.Arg(dv), gc.Arg(du), gc.Arg(du),
				gc.ArgValue(int32(0)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)), gc.ArgValue(eps), gc.ArgValue(int32(0)))
			_ = stream.Synchronize(bg)
			gotV := make([]float32, nKV*hd)
			_ = gc.CopyDtoH(bg, gotV, dv)

			rq, rk, rv := maxRel(gotQ, refQ), maxRel(gotK, refK), maxRel(gotV, refV)
			t.Logf("hd=%d: maxRelErr Q=%.2e K=%.2e V=%.2e", hd, rq, rk, rv)
			if rq > 1e-3 || rk > 1e-3 || rv > 1e-3 {
				t.Errorf("hd=%d per-head norm diverges from CPU (Q=%.3e K=%.3e V=%.3e) — the %d-elem/thread reduction is wrong at this width", hd, rq, rk, rv, hd/128)
			}
		})
	}
}

// TestVNorm_scaleless isolates Gemma 4's V-norm (attention_k_eq_v) BEFORE it is wired into
// the K=V forward, so a red here is the norm alone — not the skipped projection or the copy
// ordering that land with it (the hd=512 lesson: isolate the new primitive first).
//
// V = v_norm(k) is a SCALE-LESS per-head RMSNorm (rmsNormNoWeight: no learned weight). The
// resident path reuses the qk_norm kernel, which computes x*inv*(addOne ? 1+w : w). Scale-less
// means g==1, which needs addOne=0 AND unit weight (w=1.0). The trap: addOne=1 with w=1.0 gives
// g=2 — a silent 2x on V that scales the whole attention output. cosine can't see a uniform 2x
// (it is scale-invariant), so this asserts per-element RELATIVE error against the CPU oracle —
// where a 2x is an instant, unambiguous 100% miss.
func TestVNorm_scaleless(t *testing.T) {
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
	qmod, err := ctx.LoadModule(fusedQKVPTX)
	if err != nil {
		t.Fatalf("fusedQKV module: %v", err)
	}
	fQKN, err := qmod.Function("qk_norm")
	if err != nil {
		t.Fatalf("qk_norm: %v", err)
	}
	stream := mustStream(t, ctx)

	const nKV, hd = 2, 128
	const eps = float32(1e-6)
	vh := make([]float32, nKV*hd)
	for i := range vh {
		vh[i] = float32(math.Sin(float64(i)*0.13)) * 3 // non-trivial magnitudes so a 2x is visible
	}

	// CPU oracle: rmsNormNoWeight per head (the exact scale-less norm the forward applies to V).
	ref := append([]float32(nil), vh...)
	for r := 0; r < nKV; r++ {
		var ss float64
		for d := 0; d < hd; d++ {
			ss += float64(ref[r*hd+d]) * float64(ref[r*hd+d])
		}
		inv := float32(1.0 / math.Sqrt(ss/float64(hd)+float64(eps)))
		for d := 0; d < hd; d++ {
			ref[r*hd+d] *= inv
		}
	}

	// GPU: qk_norm with nH=0 (no Q pass), k=vB, unit weight, addOne=0 → norms only V, scale-less.
	unit := make([]float32, hd)
	for i := range unit {
		unit[i] = 1.0
	}
	dv := mustAlloc[float32](t, ctx, nKV*hd)
	du := mustAlloc[float32](t, ctx, hd)
	_ = gc.CopyHtoD(bg, dv, vh)
	_ = gc.CopyHtoD(bg, du, unit)
	_ = fQKN.LaunchOn(bg, stream, gc.LaunchConfig{GridX: uint32(nKV), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
		gc.Arg(dv), gc.Arg(dv), gc.Arg(du), gc.Arg(du),
		gc.ArgValue(int32(0)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)), gc.ArgValue(eps), gc.ArgValue(int32(0)))
	_ = stream.Synchronize(bg)
	got := make([]float32, nKV*hd)
	_ = gc.CopyDtoH(bg, got, dv)

	// Per-element relative error — NOT cosine (which a uniform 2x survives).
	var maxRel float64
	for i := range got {
		den := math.Abs(float64(ref[i])) + 1e-6
		if rel := math.Abs(float64(got[i]-ref[i])) / den; rel > maxRel {
			maxRel = rel
		}
	}
	t.Logf("scale-less v_norm vs CPU rmsNormNoWeight: maxRelErr=%.2e (a 2x would read ~1.0)", maxRel)
	if maxRel > 1e-3 {
		t.Fatalf("v_norm maxRelErr %.3e > 1e-3 — likely the addOne/unit-weight convention (a 2x reads ~1.0), not an f32-reduction diff", maxRel)
	}
}
