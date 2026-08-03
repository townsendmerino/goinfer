//go:build cuda

package cuda

import (
	"context"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestAttnBatched_bitIdentical gates the correctness crux of the batched prefill: attn_batched must
// equal the sequential M=1 attention() BIT-FOR-BIT at every (row, head, dim), for the two masks that
// only exist at M>1:
//   - CAUSALITY: batched row m attends to keys [0, startPos+m]. The reference runs attention() with
//     nKeys = startPos+m+1 per row; if attn_batched used a single nKeys (or the full length) for all
//     rows, earlier rows would attend to future keys and diverge here. A last-token-logits gate can
//     NEVER catch this (the last row legitimately attends to everything) — this per-row check is the
//     only isolation that does.
//   - SLIDING WINDOW: with window>0 and window < startPos+M, later rows drop their oldest keys while
//     earlier rows (nKeys<=window) keep all of theirs — a per-row winStart. window=0 (dense) and
//     window=5 (kicks in partway) are both gated.
//
// Reference = the SAME attention() run per row with that row's nKeys/window, so any diff isolates the
// batching (per-row masking) from the arithmetic. Same blockDim on both paths keeps the reduce tree
// identical, which bit-identity requires.
func TestAttnBatched_bitIdentical(t *testing.T) {
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

	refMod, err := cx.LoadModule(gluePTX)
	if err != nil {
		t.Fatalf("LoadModule(glue): %v", err)
	}
	fnRef, err := refMod.Function("attention")
	if err != nil {
		t.Fatalf("Function(attention): %v", err)
	}
	batMod, err := cx.LoadModule(prefillBatchedPTX)
	if err != nil {
		t.Fatalf("LoadModule(prefill_batched): %v", err)
	}
	fnBat, err := batMod.Function("attn_batched")
	if err != nil {
		t.Fatalf("Function(attn_batched): %v", err)
	}
	stream := mustStream(t, cx)

	const nH, nKV, hd = 4, 2, 8
	const qDim, kvDim = nH * hd, nKV * hd
	const blockDim = 64
	scale := float32(1.0 / math.Sqrt(float64(hd)))

	var rng uint32 = 987654321
	rnd := func() float32 { rng = rng*1664525 + 1013904223; return float32(int32(rng>>8)%2000-1000) / 1000 }

	// startPos>0 gives every row a causal past; M spans the window boundary. Two window regimes.
	for _, tc := range []struct {
		startPos, M, window int
	}{
		{0, 16, 0}, // fresh prompt, dense
		{7, 16, 0}, // warm cache, dense
		{0, 40, 5}, // fresh prompt, window kicks in after 5 tokens
		{7, 40, 5}, // warm cache + window: winStart differs per row AND spans the cache seam
		{3, 33, 8}, // non-round sizes
	} {
		startPos, M, window := tc.startPos, tc.M, tc.window
		nTot := startPos + M

		kc := make([]float32, nTot*kvDim)
		vc := make([]float32, nTot*kvDim)
		for i := range kc {
			kc[i] = rnd()
			vc[i] = rnd()
		}
		q := make([]float32, M*qDim)
		for i := range q {
			q[i] = rnd()
		}

		dKc := mustAlloc[float32](t, cx, len(kc))
		dVc := mustAlloc[float32](t, cx, len(vc))
		dQ := mustAlloc[float32](t, cx, len(q))
		if e := gc.CopyHtoD(bg, dKc, kc); e != nil {
			t.Fatalf("H2D kc: %v", e)
		}
		if e := gc.CopyHtoD(bg, dVc, vc); e != nil {
			t.Fatalf("H2D vc: %v", e)
		}
		if e := gc.CopyHtoD(bg, dQ, q); e != nil {
			t.Fatalf("H2D q: %v", e)
		}

		// Reference: attention() per row, nKeys=startPos+m+1, its own window shared-mem size.
		dQrow := mustAlloc[float32](t, cx, qDim)
		dCtxRow := mustAlloc[float32](t, cx, qDim)
		ref := make([][]float32, M)
		for m := 0; m < M; m++ {
			nKeys := startPos + m + 1
			winStart := 0
			if window > 0 && nKeys > window {
				winStart = nKeys - window
			}
			nWin := nKeys - winStart
			if e := gc.CopyHtoD(bg, dQrow, q[m*qDim:(m+1)*qDim]); e != nil {
				t.Fatalf("H2D qrow: %v", e)
			}
			cfg := gc.LaunchConfig{GridX: nH, GridY: 1, GridZ: 1, BlockX: blockDim, BlockY: 1, BlockZ: 1,
				SharedMemBytes: uint32((nWin + blockDim) * 4)}
			if e := fnRef.LaunchOn(bg, stream, cfg,
				gc.Arg(dQrow), gc.Arg(dKc), gc.Arg(dVc),
				gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)), gc.ArgValue(int32(nKeys)),
				gc.ArgValue(scale), gc.ArgValue(int32(window)), gc.Arg(dCtxRow)); e != nil {
				t.Fatalf("ref launch sp=%d m=%d: %v", startPos, m, e)
			}
			if e := stream.Synchronize(bg); e != nil {
				t.Fatalf("sync: %v", e)
			}
			row := make([]float32, qDim)
			if e := gc.CopyDtoH(bg, row, dCtxRow); e != nil {
				t.Fatalf("D2H ref: %v", e)
			}
			ref[m] = row
		}
		dQrow.Close()
		dCtxRow.Close()

		// Batched: one launch, GridX=nH, GridY=M. Shared mem sized to the largest row's window.
		maxNWin := nTot
		if window > 0 && window < maxNWin {
			maxNWin = window
		}
		dCtx := mustAlloc[float32](t, cx, M*qDim)
		cfg := gc.LaunchConfig{GridX: nH, GridY: uint32(M), GridZ: 1, BlockX: blockDim, BlockY: 1, BlockZ: 1,
			SharedMemBytes: uint32((maxNWin + blockDim) * 4)}
		if e := fnBat.LaunchOn(bg, stream, cfg,
			gc.Arg(dQ), gc.Arg(dKc), gc.Arg(dVc),
			gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)), gc.ArgValue(int32(startPos)),
			gc.ArgValue(scale), gc.ArgValue(int32(window)), gc.ArgValue(int32(M)), gc.Arg(dCtx)); e != nil {
			t.Fatalf("batched launch sp=%d: %v", startPos, e)
		}
		if e := stream.Synchronize(bg); e != nil {
			t.Fatalf("sync: %v", e)
		}
		bat := make([]float32, M*qDim)
		if e := gc.CopyDtoH(bg, bat, dCtx); e != nil {
			t.Fatalf("D2H bat: %v", e)
		}

		mism := 0
		for m := 0; m < M; m++ {
			for i := 0; i < qDim; i++ {
				if ref[m][i] != bat[m*qDim+i] {
					if mism < 3 {
						t.Errorf("sp=%d win=%d [m=%d,i=%d]: batched %v != M=1 %v", startPos, window, m, i, bat[m*qDim+i], ref[m][i])
					}
					mism++
				}
			}
		}
		if mism == 0 {
			t.Logf("startPos=%2d M=%2d window=%d: attn_batched == M=1 attention, BIT-IDENTICAL (%d rows)", startPos, M, window, M)
		} else {
			t.Fatalf("startPos=%d M=%d window=%d: %d/%d elements differ — per-row causal/window mask drifted", startPos, M, window, mism, M*qDim)
		}
		dKc.Close()
		dVc.Close()
		dQ.Close()
		dCtx.Close()
	}
}
