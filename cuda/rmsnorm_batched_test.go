//go:build cuda

package cuda

import (
	"context"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestRmsnormBatched_bitIdentical compares the batched rmsnorm_quant_batched (M=1) against the decode
// rmsnorm_quant HEAD-TO-HEAD at the REAL hidden width (1536) — the comparison no existing gate makes.
// The batched-vs-decode forward gap (TestBatchedVsDecodeGap; 84% stream divergence) was localized past
// the GEMV (bit-identical at real dims) to the RMS by elimination; this pins whether the two RMS
// kernels actually diverge, and at what magnitude. Same input, weight, eps, addOne, blockDim (256) as
// the two production launches (r.rms / bRmsB).
func TestRmsnormBatched_bitIdentical(t *testing.T) {
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

	glMod, err := cx.LoadModule(gluePTX)
	if err != nil {
		t.Fatalf("glue module: %v", err)
	}
	fnDec, err := glMod.Function("rmsnorm_quant")
	if err != nil {
		t.Fatalf("rmsnorm_quant: %v", err)
	}
	pbMod, err := cx.LoadModule(prefillBatchedPTX)
	if err != nil {
		t.Fatalf("prefill_batched module: %v", err)
	}
	fnBat, err := pbMod.Function("rmsnorm_quant_batched")
	if err != nil {
		t.Fatalf("rmsnorm_quant_batched: %v", err)
	}
	stream := mustStream(t, cx)

	for _, H := range []int{1536, 8960} { // real widths (hidden, and the down-proj input)
		const eps = float32(1e-6)
		xf := make([]float32, H)
		wf := make([]float32, H)
		var s uint32 = 0x1234567
		for i := range xf {
			s = s*1664525 + 1013904223
			xf[i] = (float32(s>>8)/float32(1<<24))*2 - 1 // [-1,1)
			s = s*1664525 + 1013904223
			wf[i] = float32(s>>8) / float32(1<<24) // [0,1)
		}
		xd := mustAlloc[float32](t, cx, H)
		wd := mustAlloc[float32](t, cx, H)
		_ = gc.CopyHtoD(bg, xd, xf)
		_ = gc.CopyHtoD(bg, wd, wf)

		run := func(fn *gc.Function, shared int) ([]int32, float32) {
			q := mustAlloc[int32](t, cx, H/4)
			sc := mustAlloc[float32](t, cx, 1)
			cfg := gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(shared)}
			if err := fn.LaunchOn(bg, stream, cfg,
				gc.Arg(xd), gc.Arg(wd), gc.ArgValue(int32(H)), gc.ArgValue(eps), gc.ArgValue(int32(0)),
				gc.Arg(q), gc.Arg(sc)); err != nil {
				t.Fatalf("launch H=%d: %v", H, err)
			}
			if err := stream.Synchronize(bg); err != nil {
				t.Fatalf("sync: %v", err)
			}
			qh := make([]int32, H/4)
			sch := make([]float32, 1)
			_ = gc.CopyDtoH(bg, qh, q)
			_ = gc.CopyDtoH(bg, sch, sc)
			return qh, sch[0]
		}

		qDec, scDec := run(fnDec, (H+256)*4)
		qBat, scBat := run(fnBat, (256+H)*4)

		qMism := 0
		for i := range qDec {
			if qDec[i] != qBat[i] {
				qMism++
			}
		}
		scΔ := math.Abs(float64(scDec - scBat))
		if qMism == 0 && scDec == scBat {
			t.Logf("H=%-4d: rmsnorm_quant == rmsnorm_quant_batched BIT-IDENTICAL", H)
		} else {
			t.Errorf("H=%d: RMS DIVERGES — quant int mism=%d/%d, scaleΔ=%g (decode=%g batched=%g)",
				H, qMism, H/4, scΔ, scDec, scBat)
		}
	}
}
