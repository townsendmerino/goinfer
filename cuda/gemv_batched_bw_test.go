//go:build cuda

package cuda

import (
	"context"
	"os"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/aikit/linalg"
)

// TestGemvBatchedBandwidth times ONE gemv_w4a8_batched launch in isolation at a representative prefill
// shape (gate/up: N=8960, K=1536, M=512) and derives the achieved MAC rate and the effective activation
// read bandwidth under the kernel's actual access pattern. It settles the attribution ncu would have,
// without ncu: the kernel is weight-stationary, so each of the N output-row warps re-reads the WHOLE
// [M,K] activation → activation traffic = N×M×K, a factor of N over the M×K minimum. If the derived
// activation bandwidth sits near the card's L2 ceiling while the MAC rate sits far below dp4a peak, the
// kernel is activation-read-bound (L2), not compute-bound — and IMMA (a compute-ceiling lever) would
// not help; the fix is staging the A tile so it is read once, not N times.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestGemvBatchedBandwidth -v
func TestGemvBatchedBandwidth(t *testing.T) {
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

	mod, err := cx.LoadModule(gemvBatchedPTX)
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	fn, err := mod.Function("gemv_w4a8_batched")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}
	stream := mustStream(t, cx)

	// gate/up projection shape of qwen2.5-coder-1.5b, batched at M=512.
	const N, K, group, M = 8960, 1536, 32, 512
	kw, kg, kd4 := K/8, K/group, K/4

	wf := make([]float32, N*K)
	var s uint32 = 999
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

	cfg := gc.LaunchConfig{GridX: uint32((N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
	launch := func() {
		_ = fn.LaunchOn(bg, stream, cfg,
			gc.Arg(dW), gc.Arg(dA), gc.Arg(dGs), gc.Arg(dAs), gc.Arg(dBias),
			gc.ArgValue(int32(N)), gc.ArgValue(int32(kw)), gc.ArgValue(int32(kg)), gc.ArgValue(int32(M)),
			gc.Arg(dOut), gc.ArgValue(int32(0)))
	}
	// warm
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
	actMin := float64(M) * float64(K)                 // activation read once (int8 → K bytes/row)
	actActual := float64(N) * float64(M) * float64(K) // kernel's N× re-read
	weightBytes := float64(N) * float64(K) / 2        // int4
	sec := per.Seconds()
	t.Logf("shape N=%d K=%d M=%d  |  %v/launch", N, K, M, per)
	t.Logf("  MACs            = %.1f G   → %.2f TMAC/s  (dp4a peak ~18 → %.1f%%)", macs/1e9, macs/sec/1e12, 100*macs/sec/1e12/18)
	t.Logf("  activation min  = %.2f MB  (M×K, read once)", actMin/1e6)
	t.Logf("  activation N×   = %.2f GB  (kernel re-reads per output row) → %.2f TB/s effective", actActual/1e9, actActual/sec/1e12)
	t.Logf("  weight (int4)   = %.2f MB  → %.2f GB/s", weightBytes/1e6, weightBytes/sec/1e9)
	t.Logf("  ATTRIBUTION: MAC%% low + activation-BW near L2 ceiling ⇒ activation-read-bound, not compute; re-read factor = N = %d", N)
}
