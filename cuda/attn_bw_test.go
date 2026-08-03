//go:build cuda

package cuda

import (
	"context"
	"math"
	"os"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestAttnBatchedBandwidth times attn_batched in isolation at the real qwen2.5-coder-1.5b attention
// shape and M=2048 (nH=12, nKV=2, hd=128, full attention), the ncu target for the attention-lever
// decision. It exists to be profiled: `ncu --kernel-name attn_batched ... /tmp/attnbench
// -test.run TestAttnBatchedBandwidth`. The question it must answer BEFORE any tiling design — is the
// 33×-off-compute a TRAFFIC bound (K/V re-read from L2, which shared-memory tiling fixes) or a LATENCY
// bound (like the GEMV, where the same shared-staging change bought only 1.2×)?
func TestAttnBatchedBandwidth(t *testing.T) {
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
	mod, err := cx.LoadModule(prefillBatchedPTX)
	if err != nil {
		t.Fatalf("module: %v", err)
	}
	fn, err := mod.Function("attn_batched")
	if err != nil {
		t.Fatalf("fn: %v", err)
	}
	stream := mustStream(t, cx)

	const nH, nKV, hd, M = 12, 2, 128, 2048
	const qDim, kvDim = nH * hd, nKV * hd
	const window = 0 // qwen2.5-coder: full attention
	scale := float32(1.0 / math.Sqrt(float64(hd)))

	q := make([]float32, M*qDim)
	kc := make([]float32, M*kvDim)
	vc := make([]float32, M*kvDim)
	var s uint32 = 7
	rnd := func() float32 { s = s*1664525 + 1013904223; return float32(int32(s>>8)%2000-1000) / 1000 }
	for i := range q {
		q[i] = rnd()
	}
	for i := range kc {
		kc[i] = rnd()
		vc[i] = rnd()
	}
	dQ := mustAlloc[float32](t, cx, len(q))
	dKc := mustAlloc[float32](t, cx, len(kc))
	dVc := mustAlloc[float32](t, cx, len(vc))
	dCtx := mustAlloc[float32](t, cx, M*qDim)
	defer dQ.Close()
	defer dKc.Close()
	defer dVc.Close()
	defer dCtx.Close()
	_ = gc.CopyHtoD(bg, dQ, q)
	_ = gc.CopyHtoD(bg, dKc, kc)
	_ = gc.CopyHtoD(bg, dVc, vc)

	maxNWin := M
	if window > 0 && window < maxNWin {
		maxNWin = window
	}
	cfg := gc.LaunchConfig{GridX: nH, GridY: M, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1,
		SharedMemBytes: uint32((maxNWin + 128) * 4)}
	launch := func() {
		_ = fn.LaunchOn(bg, stream, cfg,
			gc.Arg(dQ), gc.Arg(dKc), gc.Arg(dVc), gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)),
			gc.ArgValue(int32(hd)), gc.ArgValue(int32(0)), gc.ArgValue(scale), gc.ArgValue(int32(window)),
			gc.ArgValue(int32(M)), gc.Arg(dCtx))
	}
	for i := 0; i < 3; i++ {
		launch()
	}
	_ = stream.Synchronize(bg)
	const reps = 10
	t0 := time.Now()
	for i := 0; i < reps; i++ {
		launch()
	}
	_ = stream.Synchronize(bg)
	per := time.Since(t0) / reps
	// causal attention MACs: sum_m (m+1) keys × nH × hd × 2 (QK + AV)
	macs := float64(M) * float64(M) / 2 * float64(nH) * float64(hd) * 2
	t.Logf("attn_batched  nH=%d hd=%d M=%d  |  %v/launch  → %.1f GMAC/launch, %.2f TMAC/s (%.1f%% of ~4.5T)",
		nH, hd, M, per, macs/1e9, macs/per.Seconds()/1e12, 100*macs/per.Seconds()/1e12/4.5)
}
