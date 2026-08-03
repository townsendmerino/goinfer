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

// These MUST match the #define defaults compiled into gemv_w4a8_staged.cu (the sweep seds both).
const (
	stagedMT  = 32
	stagedKC  = 64
	stagedBLK = 256
)

func stagedCfg(N int) gc.LaunchConfig {
	nWarps := stagedBLK / 32
	return gc.LaunchConfig{
		GridX: uint32((N + nWarps - 1) / nWarps), GridY: 1, GridZ: 1,
		BlockX: uint32(stagedBLK), BlockY: 1, BlockZ: 1,
		SharedMemBytes: uint32(2 * stagedKC * (stagedMT + 1) * 4),
	}
}

// TestGemvStaged_bitIdentical is the gate: gemv_w4a8_staged must equal the M=1 gemv_w4a8_fwd BIT-FOR-BIT
// at every output element, for several (N,K,M) — including K=8960 (Kwords=1120, a 32-word tail after the
// 64-strides, which the chunking must place in the FINAL chunk exactly as the un-chunked kernel), odd N
// (the n>=N staging-participation guard), and M=1 (the batched path must reproduce the GEMV). Exhaustive
// over all N*M outputs — an accumulation-order error from the chunking cannot hide in a spot check.
func TestGemvStaged_bitIdentical(t *testing.T) {
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
	stMod, err := cx.LoadModule(gemvStagedPTX)
	if err != nil {
		t.Fatalf("LoadModule(staged): %v", err)
	}
	fnSt, err := stMod.Function("gemv_w4a8_staged")
	if err != nil {
		t.Fatalf("Function(gemv_w4a8_staged): %v", err)
	}
	stream := mustStream(t, cx)

	for _, sh := range []struct{ N, K int }{
		{20, 1536},  // odd N (n>=N guard); Kwords=192=3*64, no tail
		{64, 512},   // Kwords=64 = single full chunk
		{24, 8960},  // Kwords=1120 = 17 strides + 32-word TAIL in the final chunk
		{1536, 512}, // wide N
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
		hw, err := packWeight(&wm)
		if err != nil {
			t.Fatalf("packWeight: %v", err)
		}
		dW := mustAlloc[uint32](t, cx, len(hw.wpk))
		dGs := mustAlloc[uint16](t, cx, len(hw.ws16))
		dBias := mustAlloc[float32](t, cx, N)
		_ = gc.CopyHtoD(bg, dW, hw.wpk)
		_ = gc.CopyHtoD(bg, dGs, hw.ws16)
		_ = gc.CopyHtoD(bg, dBias, make([]float32, N))

		refCfg := gc.LaunchConfig{GridX: uint32((N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
		stCfg := stagedCfg(N)

		for _, M := range []int{1, 8, 13, 100} {
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
			dSt := mustAlloc[float32](t, cx, M*N)
			_ = gc.CopyHtoD(bg, dA, Apk)
			_ = gc.CopyHtoD(bg, dAs, aSc)

			// Reference: gemv_w4a8_fwd per row.
			dArow := mustAlloc[uint32](t, cx, kd4)
			dAsRow := mustAlloc[float32](t, cx, 1)
			ref := make([][]float32, M)
			for m := 0; m < M; m++ {
				_ = gc.CopyHtoD(bg, dArow, Apk[m*kd4:(m+1)*kd4])
				_ = gc.CopyHtoD(bg, dAsRow, aSc[m:m+1])
				if e := fnRef.LaunchOn(bg, stream, refCfg,
					gc.Arg(dW), gc.Arg(dArow), gc.Arg(dGs), gc.Arg(dAsRow), gc.Arg(dBias),
					gc.ArgValue(int32(N)), gc.ArgValue(int32(kw)), gc.ArgValue(int32(kg)), gc.Arg(dRef), gc.ArgValue(int32(0))); e != nil {
					t.Fatalf("ref launch: %v", e)
				}
				_ = stream.Synchronize(bg)
				row := make([]float32, N)
				_ = gc.CopyDtoH(bg, row, dRef)
				ref[m] = row
			}
			dArow.Close()
			dAsRow.Close()

			if e := fnSt.LaunchOn(bg, stream, stCfg,
				gc.Arg(dW), gc.Arg(dA), gc.Arg(dGs), gc.Arg(dAs), gc.Arg(dBias),
				gc.ArgValue(int32(N)), gc.ArgValue(int32(kw)), gc.ArgValue(int32(kg)), gc.ArgValue(int32(M)),
				gc.Arg(dSt), gc.ArgValue(int32(0))); e != nil {
				t.Fatalf("staged launch N=%d K=%d M=%d: %v", N, K, M, e)
			}
			_ = stream.Synchronize(bg)
			st := make([]float32, M*N)
			_ = gc.CopyDtoH(bg, st, dSt)

			mism := 0
			for m := 0; m < M; m++ {
				for n := 0; n < N; n++ {
					if ref[m][n] != st[m*N+n] {
						if mism < 3 {
							t.Errorf("N=%d K=%d [m=%d,n=%d]: staged %v != GEMV %v", N, K, m, n, st[m*N+n], ref[m][n])
						}
						mism++
					}
				}
			}
			if mism != 0 {
				t.Fatalf("N=%d K=%d M=%d: %d/%d differ — chunked accumulation order drifted", N, K, M, mism, M*N)
			}
			dA.Close()
			dAs.Close()
			dRef.Close()
			dSt.Close()
		}
		t.Logf("N=%4d K=%4d: staged == gemv_w4a8_fwd BIT-IDENTICAL (M=1/8/13/100)", N, K)
		dW.Close()
		dGs.Close()
		dBias.Close()
	}
}

// TestGemvStagedBandwidth times one staged GEMV at the gate/up shape and reports the achieved MAC rate
// and — the point of the fix — the activation GLOBAL read volume vs the M×K minimum. With staging the
// activation is read once per block (BLK/32 warps share it), so global reads ≈ (N/warpsPerBlock)×M×K,
// down from the un-staged N×M×K. Compare to TestGemvBatchedBandwidth (the un-staged 4.98 ms baseline).
func TestGemvStagedBandwidth(t *testing.T) {
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
		t.Skipf("primary ctx: %v", err)
	}
	defer cx.Close()
	bg := context.Background()
	mod, err := cx.LoadModule(gemvStagedPTX)
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	fn, err := mod.Function("gemv_w4a8_staged")
	if err != nil {
		t.Fatalf("Function: %v", err)
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

	cfg := stagedCfg(N)
	launch := func() {
		_ = fn.LaunchOn(bg, stream, cfg,
			gc.Arg(dW), gc.Arg(dA), gc.Arg(dGs), gc.Arg(dAs), gc.Arg(dBias),
			gc.ArgValue(int32(N)), gc.ArgValue(int32(kw)), gc.ArgValue(int32(kg)), gc.ArgValue(int32(M)),
			gc.Arg(dOut), gc.ArgValue(int32(0)))
	}
	for i := 0; i < 3; i++ {
		launch()
	}
	_ = stream.Synchronize(bg)
	const reps = 20
	t0 := time.Now()
	for i := 0; i < reps; i++ {
		launch()
	}
	_ = stream.Synchronize(bg)
	per := time.Since(t0) / reps

	warpsPerBlock := float64(stagedBLK / 32)
	macs := float64(N) * float64(K) * float64(M)
	actMin := float64(M) * float64(K)
	actStaged := float64(N) / warpsPerBlock * float64(M) * float64(K) // global reads with staging
	sec := per.Seconds()
	t.Logf("MT=%d KC=%d BLK=%d  N=%d K=%d M=%d  |  %v/launch", stagedMT, stagedKC, stagedBLK, N, K, M, per)
	t.Logf("  MACs        = %.1f G → %.2f TMAC/s (dp4a peak ~18 → %.1f%%)", macs/1e9, macs/sec/1e12, 100*macs/sec/1e12/18)
	t.Logf("  act min     = %.2f MB (M×K)", actMin/1e6)
	t.Logf("  act staged  = %.2f GB (N/%.0f × M×K) = %.0f× min → %.2f TB/s", actStaged/1e9, warpsPerBlock, actStaged/actMin, actStaged/sec/1e12)
	t.Logf("  (un-staged baseline was %.2f GB = %d× min, 4.98 ms)", float64(N)*actMin/1e9, N)
}
