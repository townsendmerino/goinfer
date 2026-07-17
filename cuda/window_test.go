//go:build cuda

package cuda

import (
	"context"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestSlidingWindowAttention is the decisive semantic test for the windowed attention kernel.
// A real checkpoint's window (Mistral 4096, Phi-3-mini 2047) never engages at the short prompts
// the parity gates use — winStart = max(0, nKeys-window) = 0 — so those tests prove only that
// passing a window doesn't break the full-causal path. This proves the window actually windows:
//
//	attention(N keys, window=W)  ==  attention(last W keys, window=0)
//
// i.e. a windowed run over N keys must be IDENTICAL to a full-causal run over just the trailing
// W of them. That is the definition of the sliding window (decoder/kvcache.go WindowStart:
// start = max(pos-window+1, 0)), and it is what the CPU does. Same construction as Metal's
// TestSlidingWindow.
func TestSlidingWindowAttention(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	cx, _ := dev.Primary()
	defer cx.Close()
	bg := context.Background()

	const nH, nKV, hd = 8, 2, 64
	const nKeys, window = 40, 12 // N keys, window W: the last 12 must be all that matters
	const qDim, kvDim = nH * hd, nKV * hd
	const scale = float32(0.125)

	mod, err := cx.LoadModule(gluePTX)
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	fn, err := mod.Function("attention")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}
	stream := mustStream(t, cx)

	var seed uint32 = 12345
	rnd := func() float32 { seed = seed*1664525 + 1013904223; return float32(int32(seed>>8)%2000-1000) / 1000 }
	q := make([]float32, qDim)
	for i := range q {
		q[i] = rnd()
	}
	kFull := make([]float32, nKeys*kvDim)
	vFull := make([]float32, nKeys*kvDim)
	for i := range kFull {
		kFull[i] = rnd()
		vFull[i] = rnd()
	}
	// The trailing `window` keys, as their own contiguous cache.
	start := nKeys - window
	kTail := append([]float32(nil), kFull[start*kvDim:]...)
	vTail := append([]float32(nil), vFull[start*kvDim:]...)

	run := func(k, v []float32, nk, win int) []float32 {
		dq := mustAlloc[float32](t, cx, len(q))
		dk := mustAlloc[float32](t, cx, len(k))
		dv := mustAlloc[float32](t, cx, len(v))
		dc := mustAlloc[float32](t, cx, qDim)
		defer dq.Close()
		defer dk.Close()
		defer dv.Close()
		defer dc.Close()
		_ = gc.CopyHtoD(bg, dq, q)
		_ = gc.CopyHtoD(bg, dk, k)
		_ = gc.CopyHtoD(bg, dv, v)
		nWin := nk
		if win > 0 && nk > win {
			nWin = win
		}
		if e := fn.LaunchOn(bg, stream, gc.LaunchConfig{
			GridX: nH, GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1,
			SharedMemBytes: uint32((nWin + 128) * 4)},
			gc.Arg(dq), gc.Arg(dk), gc.Arg(dv), gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)),
			gc.ArgValue(int32(hd)), gc.ArgValue(int32(nk)), gc.ArgValue(scale),
			gc.ArgValue(int32(win)), gc.Arg(dc)); e != nil {
			t.Fatalf("launch: %v", e)
		}
		if e := stream.Synchronize(bg); e != nil {
			t.Fatalf("sync: %v", e)
		}
		out := make([]float32, qDim)
		_ = gc.CopyDtoH(bg, out, dc)
		return out
	}

	windowed := run(kFull, vFull, nKeys, window) // 40 keys, window 12
	tailFull := run(kTail, vTail, window, 0)     // last 12 keys, full causal

	var maxAbs float64
	for i := range windowed {
		if d := math.Abs(float64(windowed[i] - tailFull[i])); d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("windowed(%d keys, W=%d) vs full(last %d keys): maxAbs=%.6g", nKeys, window, window, maxAbs)
	if maxAbs != 0 {
		t.Fatalf("sliding window is not equivalent to full attention over the trailing W keys "+
			"(maxAbs %.6g) — the window is attending over the wrong span", maxAbs)
	}

	// window=0 and window>=nKeys must both be full causal (the global-layer / short-prompt paths
	// every plain-dense model takes).
	full := run(kFull, vFull, nKeys, 0)
	for _, w := range []int{nKeys, nKeys + 100} {
		got := run(kFull, vFull, nKeys, w)
		for i := range got {
			if got[i] != full[i] {
				t.Fatalf("window=%d over %d keys != full causal — a window >= nKeys must be inert", w, nKeys)
			}
		}
	}
	t.Logf("window=0 and window>=nKeys are both exactly full-causal ✓")
}
