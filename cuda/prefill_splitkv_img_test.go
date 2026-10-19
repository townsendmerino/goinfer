//go:build cuda

package cuda

import (
	"context"
	"fmt"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestPrefillSplitKV_rowsEqualAttnImgBatched (the image-block twin of KV3 gate G1): with an image-block table, the three row-batched kernels write the
// context attn_img_batched writes, bit for bit at every (row, head, dim). attn_img_batched is the exact batched attention for a Gemma 3 image prompt, and
// like attn_batched it keeps a row's scores in shared memory, so past the bound only the row-batched kernels can run it. The cases put the block at the
// start, in the middle and at the end of the pass, use one block and two, a plain and a sliding-window layer (the window's start comes from the CAUSAL
// count, never the widened one: the block sits more than one window into the sequence in the windowed cases, where the coupled formula goes wrong),
// head dims 64 to 512, launch groups of one row, a ragged count and every row. The image kernel reads an f32 cache, so there is no f16 arm.
//
// Plant: the same launches with the block table withheld (nBlocks 0) must NOT equal the reference, or this test cannot tell a widened row from a causal one.
func TestPrefillSplitKV_rowsEqualAttnImgBatched(t *testing.T) {
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
	imgMod, err := cx.LoadModule(attnImgPrefillPTX)
	if err != nil {
		t.Fatalf("LoadModule(attn_img_prefill): %v", err)
	}
	fnImg, err := imgMod.Function("attn_img_batched")
	if err != nil {
		t.Fatal(err)
	}
	rowsMod, err := cx.LoadModule(prefillSplitKVPTX)
	if err != nil {
		t.Fatalf("LoadModule(prefill_splitkv): %v", err)
	}
	fn := func(name string) *gc.Function {
		f, e := rowsMod.Function(name)
		if e != nil {
			t.Fatalf("Function(%s): %v", name, e)
		}
		return f
	}
	fnSc, fnSm, fnVs := fn("splitkv_scores_rows"), fn("splitkv_softmax_rows"), fn("splitkv_vsum_rows")
	stream := mustStream(t, cx)
	launch := func(f *gc.Function, cfg gc.LaunchConfig, args ...gc.KernelArg) {
		t.Helper()
		if e := f.LaunchOn(bg, stream, cfg, args...); e != nil {
			t.Fatalf("launch: %v", e)
		}
	}
	var rng uint32 = 20261018
	rnd := func() float32 { rng = rng*1664525 + 1013904223; return float32(int32(rng>>8)%2000-1000) / 1000 }

	checked, planted := 0, 0
	for _, g := range []struct{ nH, nKV, hd int }{{4, 2, 64}, {4, 4, 128}, {8, 4, 256}, {4, 1, 512}} {
		for _, tc := range []struct {
			startPos, M, window int
			blocks              [][2]int
		}{
			{0, 120, 0, [][2]int{{10, 50}}},                  // a block near the start, plain layer
			{0, 120, 0, [][2]int{{70, 120}}},                 // a block that ends the pass
			{0, 120, 0, [][2]int{{0, 40}, {80, 110}}},        // two blocks, one at the very start
			{300, 96, 0, [][2]int{{330, 380}}},               // a pass that starts past zero
			{0, 400, 64, [][2]int{{200, 260}}},               // sliding window with the block well over one window in
			{150, 200, 40, [][2]int{{200, 230}, {300, 340}}}, // sliding window, two blocks, later pass
			{0, 160, 100, [][2]int{{20, 70}}},                // a window wider than the block
		} {
			nH, nKV, hd := g.nH, g.nKV, g.hd
			startPos, M, window := tc.startPos, tc.M, tc.window
			qDim, kvDim, nTot := nH*hd, nKV*hd, startPos+M
			scale := float32(1.0 / math.Sqrt(float64(hd)))
			kc, vc := make([]float32, nTot*kvDim), make([]float32, nTot*kvDim)
			for i := range kc {
				kc[i], vc[i] = rnd(), rnd()
			}
			q := make([]float32, M*qDim)
			for i := range q {
				q[i] = rnd()
			}
			flat := make([]int32, 0, 2*len(tc.blocks))
			for _, b := range tc.blocks {
				flat = append(flat, int32(b[0]), int32(b[1]))
			}
			dKc, dVc, dQ := mustAlloc[float32](t, cx, len(kc)), mustAlloc[float32](t, cx, len(vc)), mustAlloc[float32](t, cx, len(q))
			dBlk := mustAlloc[int32](t, cx, len(flat))
			for _, e := range []error{gc.CopyHtoD(bg, dKc, kc), gc.CopyHtoD(bg, dVc, vc), gc.CopyHtoD(bg, dQ, q), gc.CopyHtoD(bg, dBlk, flat)} {
				if e != nil {
					t.Fatalf("H2D: %v", e)
				}
			}
			// pageable host-to-device copies return once staged: the barrier makes the uploads visible to the first launch (see the attn_batched test)
			if e := cx.Synchronize(bg); e != nil {
				t.Fatalf("context sync after the uploads: %v", e)
			}
			maxNWin := nTot
			if window > 0 && window < maxNWin {
				maxNWin = window
			}
			maxNWin = imgBlocksMaxNWin(maxNWin, window, tc.blocks)
			dRef := mustAlloc[float32](t, cx, M*qDim)
			launch(fnImg, gc.LaunchConfig{GridX: uint32(nH), GridY: uint32(M), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((maxNWin + 128) * 4)},
				gc.Arg(dQ), gc.Arg(dKc), gc.Arg(dVc), gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)), gc.ArgValue(int32(startPos)),
				gc.ArgValue(scale), gc.ArgValue(int32(window)), gc.ArgValue(int32(M)), gc.Arg(dRef), gc.ArgDevicePtr(0), gc.Arg(dBlk), gc.ArgValue(int32(len(tc.blocks))))
			if e := stream.Synchronize(bg); e != nil {
				t.Fatal(e)
			}
			ref := make([]float32, M*qDim)
			if e := gc.CopyDtoH(bg, ref, dRef); e != nil {
				t.Fatal(e)
			}

			rows := func(R int, withBlocks bool) []float32 {
				stride := maxNWin + 3
				dSc, dInv, dCtx := mustAlloc[float32](t, cx, R*nH*stride), mustAlloc[float32](t, cx, R*nH), mustAlloc[float32](t, cx, M*qDim)
				defer dSc.Close()
				defer dInv.Close()
				defer dCtx.Close()
				blk, nb := gc.ArgDevicePtr(0), int32(0)
				if withBlocks {
					blk, nb = gc.Arg(dBlk), int32(len(tc.blocks))
				}
				for row0 := 0; row0 < M; row0 += R {
					n := min(R, M-row0)
					launch(fnSc, gc.LaunchConfig{GridX: uint32(nH), GridY: uint32((stride + 127) / 128), GridZ: uint32(n), BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(hd * 4)},
						gc.Arg(dQ), gc.Arg(dKc), gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)), gc.ArgValue(int32(startPos)),
						gc.ArgValue(int32(window)), gc.ArgValue(int32(row0)), gc.ArgValue(scale), gc.Arg(dSc), gc.ArgValue(int32(stride)), blk, gc.ArgValue(nb))
					launch(fnSm, gc.LaunchConfig{GridX: uint32(nH), GridY: uint32(n), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 4},
						gc.Arg(dSc), gc.ArgValue(int32(nH)), gc.ArgValue(int32(startPos)), gc.ArgValue(int32(window)), gc.ArgValue(int32(row0)),
						gc.ArgValue(int32(stride)), gc.Arg(dInv), gc.ArgDevicePtr(0), blk, gc.ArgValue(nb))
					launch(fnVs, gc.LaunchConfig{GridX: uint32(nH), GridY: uint32((hd + 31) / 32), GridZ: uint32(n), BlockX: 32, BlockY: 1, BlockZ: 1},
						gc.Arg(dSc), gc.Arg(dVc), gc.Arg(dInv), gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)), gc.ArgValue(int32(startPos)),
						gc.ArgValue(int32(window)), gc.ArgValue(int32(row0)), gc.ArgValue(int32(stride)), gc.Arg(dCtx), blk, gc.ArgValue(nb))
				}
				if e := stream.Synchronize(bg); e != nil {
					t.Fatal(e)
				}
				out := make([]float32, M*qDim)
				if e := gc.CopyDtoH(bg, out, dCtx); e != nil {
					t.Fatal(e)
				}
				return out
			}
			differ := func(a, b []float32) int {
				n := 0
				for i := range a {
					if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
						n++
					}
				}
				return n
			}
			name := fmt.Sprintf("nH=%d nKV=%d hd=%d start=%d M=%d window=%d blocks=%v", nH, nKV, hd, startPos, M, window, tc.blocks)
			nonzero := 0
			for _, v := range ref {
				if v != 0 {
					nonzero++
				}
			}
			if nonzero < len(ref)/2 {
				t.Fatalf("%s: the reference is mostly zero: the comparison proves nothing", name)
			}
			for _, R := range []int{1, 7, M} {
				if d := differ(rows(R, true), ref); d != 0 {
					t.Errorf("%s R=%d: %d of %d floats differ from attn_img_batched", name, R, d, len(ref))
				}
				checked++
			}
			if d := differ(rows(M, false), ref); d == 0 {
				t.Errorf("VOID %s: the row-batched kernels with the block table withheld equal attn_img_batched, so this case cannot tell a widened row from a causal one", name)
			}
			planted++
			dRef.Close()
			dKc.Close()
			dVc.Close()
			dQ.Close()
			dBlk.Close()
		}
	}
	t.Logf("%d cases bit-identical to attn_img_batched; %d plants (blocks withheld) all differ", checked, planted)
}
