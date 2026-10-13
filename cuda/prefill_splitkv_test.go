//go:build cuda

package cuda

import (
	"context"
	"fmt"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestPrefillSplitKV_rowsEqualAttnBatched (KV3 tiled exact, gate G1): the three row-batched kernels write the context attn_batched
// writes, bit for bit at every (row, head, dim), over head dims 20 to 512 (a multiple of 4, which attn_batched's float4 read of q needs too), MHA and GQA, a causal past,
// a sliding window that opens partway through the rows, a sink, launch groups of one row, of a ragged count and of every row, and
// both KV element types. Every span is under attn_batched's shared-memory bound, which is the only place the two can be compared.
func TestPrefillSplitKV_rowsEqualAttnBatched(t *testing.T) {
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
	t.Run("kv=f32", func(t *testing.T) {
		psRowsCases(t, cx, prefillBatchedPTX, prefillSplitKVPTX, func(r func() float32) float32 { return r() })
	})
	t.Run("kv=f16", func(t *testing.T) {
		// Finite halves of mixed sign and magnitude: both kernels read the same bits, so the values need only be ordinary.
		psRowsCases(t, cx, prefillBatchedKV16PTX, prefillSplitKVKV16PTX, func(r func() float32) uint16 {
			v := r()
			sign := uint16(0)
			if v < 0 {
				sign, v = 0x8000, -v
			}
			return sign | uint16(10+int(v*5.99))<<10 | uint16(math.Float32bits(r())>>9)&0x3ff
		})
	})
}

func psRowsCases[T gc.Supported](t *testing.T, cx *gc.Context, batPTX, rowsPTX []byte, kvElem func(func() float32) T) {
	bg := context.Background()
	batMod, err := cx.LoadModule(batPTX)
	if err != nil {
		t.Fatalf("LoadModule(prefill_batched): %v", err)
	}
	fnBat, err := batMod.Function("attn_batched")
	if err != nil {
		t.Fatal(err)
	}
	rowsMod, err := cx.LoadModule(rowsPTX)
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

	var rng uint32 = 20261013
	rnd := func() float32 { rng = rng*1664525 + 1013904223; return float32(int32(rng>>8)%2000-1000) / 1000 }
	launch := func(f *gc.Function, cfg gc.LaunchConfig, args ...gc.KernelArg) {
		t.Helper()
		if e := f.LaunchOn(bg, stream, cfg, args...); e != nil {
			t.Fatalf("launch: %v", e)
		}
	}
	checked := 0
	for _, g := range []struct{ nH, nKV, hd int }{{4, 2, 64}, {4, 4, 128}, {8, 4, 256}, {4, 1, 512}, {3, 3, 20}} {
		for _, tc := range []struct{ startPos, M, window int }{{0, 40, 0}, {7, 33, 0}, {0, 300, 0}, {5, 200, 37}, {100, 70, 150}, {300, 16, 0}} {
			for _, withSink := range []bool{false, true} {
				nH, nKV, hd := g.nH, g.nKV, g.hd
				startPos, M, window := tc.startPos, tc.M, tc.window
				qDim, kvDim, nTot := nH*hd, nKV*hd, startPos+M
				scale := float32(1.0 / math.Sqrt(float64(hd)))
				kc, vc := make([]T, nTot*kvDim), make([]T, nTot*kvDim)
				for i := range kc {
					kc[i], vc[i] = kvElem(rnd), kvElem(rnd)
				}
				q := make([]float32, M*qDim)
				for i := range q {
					q[i] = rnd()
				}
				dKc, dVc, dQ := mustAlloc[T](t, cx, len(kc)), mustAlloc[T](t, cx, len(vc)), mustAlloc[float32](t, cx, len(q))
				for _, e := range []error{gc.CopyHtoD(bg, dKc, kc), gc.CopyHtoD(bg, dVc, vc), gc.CopyHtoD(bg, dQ, q)} {
					if e != nil {
						t.Fatalf("H2D: %v", e)
					}
				}
				// A host-to-device copy from pageable memory returns once it is staged, and a kernel on another stream is not ordered
				// behind it. The barrier and the read-back make the uploads visible to the first launch, which is the reference's.
				if e := cx.Synchronize(bg); e != nil {
					t.Fatalf("context sync after the uploads: %v", e)
				}
				back := make([]float32, len(q))
				if e := gc.CopyDtoH(bg, back, dQ); e != nil {
					t.Fatal(e)
				}
				for i := range q {
					if back[i] != q[i] {
						t.Fatalf("q did not land on the device (element %d)", i)
					}
				}
				sinkArg := gc.ArgDevicePtr(0)
				var dSink *gc.Buffer[float32]
				if withSink {
					sk := make([]float32, nH)
					for i := range sk {
						sk[i] = rnd() * 3
					}
					dSink = mustAlloc[float32](t, cx, nH)
					if e := gc.CopyHtoD(bg, dSink, sk); e != nil {
						t.Fatal(e)
					}
					sinkArg = gc.Arg(dSink)
					if e := cx.Synchronize(bg); e != nil {
						t.Fatalf("context sync after the sink upload: %v", e)
					}
				}
				maxNWin := nTot
				if window > 0 && window < maxNWin {
					maxNWin = window
				}
				// the reference: attn_batched at the production block width
				dRef := mustAlloc[float32](t, cx, M*qDim)
				launch(fnBat, gc.LaunchConfig{GridX: uint32(nH), GridY: uint32(M), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((maxNWin + 128) * 4)},
					gc.Arg(dQ), gc.Arg(dKc), gc.Arg(dVc), gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)), gc.ArgValue(int32(startPos)),
					gc.ArgValue(scale), gc.ArgValue(int32(window)), gc.ArgValue(int32(M)), gc.Arg(dRef), sinkArg)
				if e := stream.Synchronize(bg); e != nil {
					t.Fatal(e)
				}
				ref := make([]float32, M*qDim)
				if e := gc.CopyDtoH(bg, ref, dRef); e != nil {
					t.Fatal(e)
				}
				for _, R := range []int{1, 7, M} {
					stride := maxNWin + 3 // not the tight width: a row's slice is addressed by the stride, not by its own window
					dSc, dInv, dCtx := mustAlloc[float32](t, cx, R*nH*stride), mustAlloc[float32](t, cx, R*nH), mustAlloc[float32](t, cx, M*qDim)
					for row0 := 0; row0 < M; row0 += R {
						n := min(R, M-row0)
						widest := startPos + row0 + n
						if window > 0 && window < widest {
							widest = window
						}
						launch(fnSc, gc.LaunchConfig{GridX: uint32(nH), GridY: uint32((widest + 127) / 128), GridZ: uint32(n), BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(hd * 4)},
							gc.Arg(dQ), gc.Arg(dKc), gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)), gc.ArgValue(int32(startPos)),
							gc.ArgValue(int32(window)), gc.ArgValue(int32(row0)), gc.ArgValue(scale), gc.Arg(dSc), gc.ArgValue(int32(stride)))
						launch(fnSm, gc.LaunchConfig{GridX: uint32(nH), GridY: uint32(n), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 4},
							gc.Arg(dSc), gc.ArgValue(int32(nH)), gc.ArgValue(int32(startPos)), gc.ArgValue(int32(window)), gc.ArgValue(int32(row0)),
							gc.ArgValue(int32(stride)), gc.Arg(dInv), sinkArg)
						launch(fnVs, gc.LaunchConfig{GridX: uint32(nH), GridY: uint32((hd + 31) / 32), GridZ: uint32(n), BlockX: 32, BlockY: 1, BlockZ: 1},
							gc.Arg(dSc), gc.Arg(dVc), gc.Arg(dInv), gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)), gc.ArgValue(int32(startPos)),
							gc.ArgValue(int32(window)), gc.ArgValue(int32(row0)), gc.ArgValue(int32(stride)), gc.Arg(dCtx))
					}
					if e := stream.Synchronize(bg); e != nil {
						t.Fatal(e)
					}
					got := make([]float32, M*qDim)
					if e := gc.CopyDtoH(bg, got, dCtx); e != nil {
						t.Fatal(e)
					}
					name := fmt.Sprintf("nH=%d nKV=%d hd=%d start=%d M=%d window=%d sink=%v R=%d", nH, nKV, hd, startPos, M, window, withSink, R)
					mism, nonzero := 0, 0
					for i := range got {
						if math.Float32bits(got[i]) != math.Float32bits(ref[i]) {
							if mism == 0 {
								t.Errorf("%s: row %d elem %d: rows %v, attn_batched %v", name, i/qDim, i%qDim, got[i], ref[i])
							}
							mism++
						}
						if ref[i] != 0 {
							nonzero++
						}
					}
					if mism > 0 {
						t.Errorf("%s: %d of %d floats differ", name, mism, len(got))
					}
					if nonzero < len(ref)/2 {
						t.Fatalf("%s: the reference is mostly zero (%d of %d): the comparison proves nothing", name, nonzero, len(ref))
					}
					checked++
					dSc.Close()
					dInv.Close()
					dCtx.Close()
				}
				dRef.Close()
				dKc.Close()
				dVc.Close()
				dQ.Close()
				if dSink != nil {
					dSink.Close()
				}
			}
		}
	}
	t.Logf("%d cases bit-identical to attn_batched", checked)
}
