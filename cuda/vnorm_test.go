//go:build cuda

package cuda

import (
	"context"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

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
