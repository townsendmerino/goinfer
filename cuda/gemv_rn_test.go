//go:build cuda

package cuda

import (
	"context"
	"os"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/linalg"
)

// MUST match the #define defaults in gemv_w4a8_rn.cu (the sweep seds both).
const (
	rnRN  = 2
	rnMT  = 16
	rnBLK = 256
)

func rnCfg(N int) gc.LaunchConfig {
	warps := (N + rnRN - 1) / rnRN
	nWarps := rnBLK / 32
	return gc.LaunchConfig{
		GridX: uint32((warps + nWarps - 1) / nWarps), GridY: 1, GridZ: 1,
		BlockX: uint32(rnBLK), BlockY: 1, BlockZ: 1,
	}
}

// TestGemvRN_bitIdentical: gemv_w4a8_rn == the M=1 gemv_w4a8_fwd BIT-FOR-BIT, incl. odd N (the boundary
// where n0+r overruns N — the write/pointer guards) and the K=8960 tail. Exhaustive over N*M.
func TestGemvRN_bitIdentical(t *testing.T) {
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

	refMod, _ := cx.LoadModule(gpu.QuantGEMVPTX)
	fnRef, err := refMod.Function("gemv_w4a8_fwd")
	if err != nil {
		t.Fatalf("ref fn: %v", err)
	}
	rnMod, err := cx.LoadModule(gemvRNPTX)
	if err != nil {
		t.Fatalf("rn module: %v", err)
	}
	fnRN, err := rnMod.Function("gemv_w4a8_rn")
	if err != nil {
		t.Fatalf("rn fn: %v", err)
	}
	stream := mustStream(t, cx)

	for _, sh := range []struct{ N, K int }{
		{20, 1536},  // odd N (n0+r guard)
		{23, 8960},  // odd N + K=8960 tail
		{1536, 512}, // wide
	} {
		N, K := sh.N, sh.K
		const group = 32
		kw, kg, kd4 := K/8, K/group, K/4
		wf := make([]float32, N*K)
		var s uint32 = 12345
		for i := range wf {
			s = s*1664525 + 1013904223
			wf[i] = float32(int32(s>>8)%2000-1000) / 1000
		}
		wm := linalg.QuantizeInt4(wf, N, K, group)
		hw, _ := packWeight(&wm)
		dW := mustAlloc[uint32](t, cx, len(hw.wpk))
		dGs := mustAlloc[uint16](t, cx, len(hw.ws16))
		dBias := mustAlloc[float32](t, cx, N)
		_ = gc.CopyHtoD(bg, dW, hw.wpk)
		_ = gc.CopyHtoD(bg, dGs, hw.ws16)
		_ = gc.CopyHtoD(bg, dBias, make([]float32, N))
		refCfg := gc.LaunchConfig{GridX: uint32((N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
		rc := rnCfg(N)

		for _, M := range []int{1, 8, 13, 100} {
			Apk := make([]uint32, M*kd4)
			aSc := make([]float32, M)
			for m := range M {
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
			dRN := mustAlloc[float32](t, cx, M*N)
			_ = gc.CopyHtoD(bg, dA, Apk)
			_ = gc.CopyHtoD(bg, dAs, aSc)

			dArow := mustAlloc[uint32](t, cx, kd4)
			dAsRow := mustAlloc[float32](t, cx, 1)
			ref := make([][]float32, M)
			for m := range M {
				_ = gc.CopyHtoD(bg, dArow, Apk[m*kd4:(m+1)*kd4])
				_ = gc.CopyHtoD(bg, dAsRow, aSc[m:m+1])
				_ = fnRef.LaunchOn(bg, stream, refCfg,
					gc.Arg(dW), gc.Arg(dArow), gc.Arg(dGs), gc.Arg(dAsRow), gc.Arg(dBias),
					gc.ArgValue(int32(N)), gc.ArgValue(int32(kw)), gc.ArgValue(int32(kg)), gc.Arg(dRef), gc.ArgValue(int32(0)))
				_ = stream.Synchronize(bg)
				row := make([]float32, N)
				_ = gc.CopyDtoH(bg, row, dRef)
				ref[m] = row
			}
			dArow.Close()
			dAsRow.Close()

			if e := fnRN.LaunchOn(bg, stream, rc,
				gc.Arg(dW), gc.Arg(dA), gc.Arg(dGs), gc.Arg(dAs), gc.Arg(dBias),
				gc.ArgValue(int32(N)), gc.ArgValue(int32(kw)), gc.ArgValue(int32(kg)), gc.ArgValue(int32(M)),
				gc.Arg(dRN), gc.ArgValue(int32(0))); e != nil {
				t.Fatalf("rn launch N=%d K=%d M=%d: %v", N, K, M, e)
			}
			_ = stream.Synchronize(bg)
			out := make([]float32, M*N)
			_ = gc.CopyDtoH(bg, out, dRN)

			mism := 0
			for m := range M {
				for n := range N {
					if ref[m][n] != out[m*N+n] {
						if mism < 3 {
							t.Errorf("N=%d K=%d [m=%d,n=%d]: rn %v != GEMV %v", N, K, m, n, out[m*N+n], ref[m][n])
						}
						mism++
					}
				}
			}
			if mism != 0 {
				t.Fatalf("N=%d K=%d M=%d: %d/%d differ", N, K, M, mism, M*N)
			}
			dA.Close()
			dAs.Close()
			dRef.Close()
			dRN.Close()
		}
		t.Logf("N=%4d K=%4d: gemv_w4a8_rn == gemv_w4a8_fwd BIT-IDENTICAL (RN=%d MT=%d)", N, K, rnRN, rnMT)
		dW.Close()
		dGs.Close()
		dBias.Close()
	}
}

// TestGemvRNBandwidth times gemv_w4a8_rn at the gate/up shape vs the 4.41 ms coalesced batched baseline.
func TestGemvRNBandwidth(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1")
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	cx, err := dev.Primary()
	if err != nil {
		t.Skipf("ctx: %v", err)
	}
	defer cx.Close()
	bg := context.Background()
	mod, _ := cx.LoadModule(gemvRNPTX)
	fn, err := mod.Function("gemv_w4a8_rn")
	if err != nil {
		t.Fatalf("fn: %v", err)
	}
	stream := mustStream(t, cx)

	const N, K, group, M = 8960, 1536, 32, 512
	kw, kg, kd4 := K/8, K/group, K/4
	wf := make([]float32, N*K)
	var s uint32 = 999
	for i := range wf {
		s = s*1664525 + 1013904223
		wf[i] = float32(int32(s>>8)%2000-1000) / 1000
	}
	wm := linalg.QuantizeInt4(wf, N, K, group)
	hw, _ := packWeight(&wm)
	dW := mustAlloc[uint32](t, cx, len(hw.wpk))
	dGs := mustAlloc[uint16](t, cx, len(hw.ws16))
	dBias := mustAlloc[float32](t, cx, N)
	dA := mustAlloc[uint32](t, cx, M*kd4)
	dAs := mustAlloc[float32](t, cx, M)
	dOut := mustAlloc[float32](t, cx, M*N)
	defer dW.Close()
	defer dGs.Close()
	defer dBias.Close()
	defer dA.Close()
	defer dAs.Close()
	defer dOut.Close()
	_ = gc.CopyHtoD(bg, dW, hw.wpk)
	_ = gc.CopyHtoD(bg, dGs, hw.ws16)
	_ = gc.CopyHtoD(bg, dBias, make([]float32, N))
	af := make([]float32, M*K)
	for i := range af {
		s = s*1664525 + 1013904223
		af[i] = float32(int32(s>>8)%2000-1000) / 800
	}
	q8, sc := linalg.QuantizeRowsInt8(af, M, K)
	_ = gc.CopyHtoD(bg, dA, packI8(q8, M, K))
	_ = gc.CopyHtoD(bg, dAs, sc)

	cfg := rnCfg(N)
	launch := func() {
		_ = fn.LaunchOn(bg, stream, cfg,
			gc.Arg(dW), gc.Arg(dA), gc.Arg(dGs), gc.Arg(dAs), gc.Arg(dBias),
			gc.ArgValue(int32(N)), gc.ArgValue(int32(kw)), gc.ArgValue(int32(kg)), gc.ArgValue(int32(M)),
			gc.Arg(dOut), gc.ArgValue(int32(0)))
	}
	for range 3 {
		launch()
	}
	_ = stream.Synchronize(bg)
	const reps = 20
	t0 := time.Now()
	for range reps {
		launch()
	}
	_ = stream.Synchronize(bg)
	per := time.Since(t0) / reps
	macs := float64(N) * float64(K) * float64(M)
	t.Logf("RN=%d MT=%d  N=%d K=%d M=%d  |  %v/launch  → %.2f TMAC/s (%.1f%% dp4a peak)  [baseline 4.41 ms]",
		rnRN, rnMT, N, K, M, per, macs/per.Seconds()/1e12, 100*macs/per.Seconds()/1e12/18)
}
