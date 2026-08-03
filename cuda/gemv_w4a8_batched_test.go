//go:build cuda

package cuda

import (
	"context"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/linalg"
)

// TestGemvW4A8Batched_bitIdentical is milestone 1's gate: the weight-stationary batched GEMV
// (gemv_w4a8_batched) must equal the sequential M=1 GEMV (aikit gemv_w4a8_fwd) BIT-FOR-BIT at every
// output element, for several M — including M=1 (the batched path must reproduce the GEMV exactly; the
// cheapest regression check and the first that must pass) and a non-tile-multiple M (MT=8, so M=13
// exercises the clamped last tile). Reference = the SAME kernel run per activation row, so the diff
// isolates the batching from the arithmetic. Anything but bit-equality means the per-group float
// accumulation order drifted — which is the whole reason there is no IMMA path.
func TestGemvW4A8Batched_bitIdentical(t *testing.T) {
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

	refMod, err := cx.LoadModule(gpu.QuantGEMVPTX)
	if err != nil {
		t.Fatalf("LoadModule(ref): %v", err)
	}
	fnRef, err := refMod.Function("gemv_w4a8_fwd")
	if err != nil {
		t.Fatalf("Function(gemv_w4a8_fwd): %v", err)
	}
	batMod, err := cx.LoadModule(gemvBatchedPTX)
	if err != nil {
		t.Fatalf("LoadModule(batched): %v", err)
	}
	fnBat, err := batMod.Function("gemv_w4a8_batched")
	if err != nil {
		t.Fatalf("Function(gemv_w4a8_batched): %v", err)
	}
	stream := mustStream(t, cx)

	// N not a multiple of 8, K a non-trivial multiple of 32 that is NOT a multiple of 64*8 (so the GEMV's
	// 64-stride main loop AND its 32-stride tail both run — the tail is where a lazy port diverges).
	const N, K, group = 20, 160, 32
	kw, kg, kd4 := K/8, K/group, K/4

	// int4 weight (group-scaled), packed exactly as production (permuteFast + f16 group scales).
	wf := make([]float32, N*K)
	var s uint32 = 12345
	for i := range wf {
		s = s*1664525 + 1013904223
		wf[i] = float32(int32(s>>8)%2000-1000) / 1000
	}
	wm := linalg.QuantizeInt4(wf, N, K, group)
	hw, err := packWeight(&wm)
	if err != nil {
		t.Fatalf("packWeight: %v", err)
	}
	dW := mustAlloc[uint32](t, cx, len(hw.wpk))
	dGs := mustAlloc[uint16](t, cx, len(hw.ws16))
	dBias := mustAlloc[float32](t, cx, N) // zeros: non-null so both paths add bias[n]==0 identically
	defer dW.Close()
	defer dGs.Close()
	defer dBias.Close()
	if e := gc.CopyHtoD(bg, dW, hw.wpk); e != nil {
		t.Fatalf("H2D W: %v", e)
	}
	if e := gc.CopyHtoD(bg, dGs, hw.ws16); e != nil {
		t.Fatalf("H2D gs: %v", e)
	}
	if e := gc.CopyHtoD(bg, dBias, make([]float32, N)); e != nil {
		t.Fatalf("H2D bias: %v", e)
	}

	cfg := gc.LaunchConfig{GridX: uint32((N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}

	for _, M := range []int{1, 8, 13, 100} {
		// M activation rows, each independently int8-quantized (row scale), packed 4 int8/word.
		Apk := make([]uint32, M*kd4)
		aSc := make([]float32, M)
		for m := 0; m < M; m++ {
			af := make([]float32, K)
			for i := range af {
				s = s*1664525 + 1013904223
				af[i] = float32(int32(s>>8)%2000-1000) / 800
			}
			q8, sc := linalg.QuantizeRowsInt8(af, 1, K)
			copy(Apk[m*kd4:(m+1)*kd4], packI8(q8, 1, K))
			aSc[m] = sc[0]
		}
		dA := mustAlloc[uint32](t, cx, M*kd4)
		dAs := mustAlloc[float32](t, cx, M)
		dRef := mustAlloc[float32](t, cx, N)
		dBat := mustAlloc[float32](t, cx, M*N)
		if e := gc.CopyHtoD(bg, dA, Apk); e != nil {
			t.Fatalf("H2D A: %v", e)
		}
		if e := gc.CopyHtoD(bg, dAs, aSc); e != nil {
			t.Fatalf("H2D aScale: %v", e)
		}

		// Reference: gemv_w4a8_fwd per activation row → ref[m][N] (per-row buffers; gc.Buffer has no offset).
		dArow := mustAlloc[uint32](t, cx, kd4)
		dAsRow := mustAlloc[float32](t, cx, 1)
		ref := make([][]float32, M)
		for m := 0; m < M; m++ {
			if e := gc.CopyHtoD(bg, dArow, Apk[m*kd4:(m+1)*kd4]); e != nil {
				t.Fatalf("H2D Arow: %v", e)
			}
			if e := gc.CopyHtoD(bg, dAsRow, aSc[m:m+1]); e != nil {
				t.Fatalf("H2D AsRow: %v", e)
			}
			if e := fnRef.LaunchOn(bg, stream, cfg,
				gc.Arg(dW), gc.Arg(dArow), gc.Arg(dGs), gc.Arg(dAsRow), gc.Arg(dBias),
				gc.ArgValue(int32(N)), gc.ArgValue(int32(kw)), gc.ArgValue(int32(kg)), gc.Arg(dRef), gc.ArgValue(int32(0))); e != nil {
				t.Fatalf("ref launch M=%d m=%d: %v", M, m, e)
			}
			if e := stream.Synchronize(bg); e != nil {
				t.Fatalf("sync: %v", e)
			}
			row := make([]float32, N)
			if e := gc.CopyDtoH(bg, row, dRef); e != nil {
				t.Fatalf("D2H ref: %v", e)
			}
			ref[m] = row
		}
		dArow.Close()
		dAsRow.Close()

		// Batched: one launch over all M columns → bat[M*N].
		if e := fnBat.LaunchOn(bg, stream, cfg,
			gc.Arg(dW), gc.Arg(dA), gc.Arg(dGs), gc.Arg(dAs), gc.Arg(dBias),
			gc.ArgValue(int32(N)), gc.ArgValue(int32(kw)), gc.ArgValue(int32(kg)), gc.ArgValue(int32(M)),
			gc.Arg(dBat), gc.ArgValue(int32(0))); e != nil {
			t.Fatalf("batched launch M=%d: %v", M, e)
		}
		if e := stream.Synchronize(bg); e != nil {
			t.Fatalf("sync: %v", e)
		}
		bat := make([]float32, M*N)
		if e := gc.CopyDtoH(bg, bat, dBat); e != nil {
			t.Fatalf("D2H bat: %v", e)
		}

		mism := 0
		for m := 0; m < M; m++ {
			for n := 0; n < N; n++ {
				if ref[m][n] != bat[m*N+n] {
					if mism < 3 {
						t.Errorf("M=%d [m=%d,n=%d]: batched %v != GEMV %v (bits differ)", M, m, n, bat[m*N+n], ref[m][n])
					}
					mism++
				}
			}
		}
		if mism == 0 {
			t.Logf("M=%3d: batched == sequential GEMV, BIT-IDENTICAL (%d elems)", M, M*N)
		} else {
			t.Fatalf("M=%d: %d/%d elements differ — per-group float accumulation order drifted", M, mism, M*N)
		}
		dA.Close()
		dAs.Close()
		dRef.Close()
		dBat.Close()
	}
}
