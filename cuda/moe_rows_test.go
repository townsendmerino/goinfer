//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/aikit/linalg"

	"github.com/townsendmerino/goinfer/decoder"
)

// The grouped expert kernels' gates (moe_rows.cu): every output is the per-pair kernels' value bit for bit.
//
//	go test -count=1 -timeout 10m -tags 'cuda goinfer_testhooks' -run '^TestMoERows_' -v ./cuda/

// TestMoERows_kernelsEqualPerPair: gemv_w4a8_moe_rows and glu_quant_rows against gemv_w4a8_moe and glu_quant, one launch per pair, at the geometry of
// Mellum2.1's experts (gate||up 1792 x 2304, down 2304 x 896: 288 and 112 words, so both the 64-word main loop and a ragged tail run), a tiny K that
// is only the tail, pair counts around the tile (1, 7, 8, 9, 37), permuted activation rows and a permuted output index.
func TestMoERows_kernelsEqualPerPair(t *testing.T) {
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
	moeMod, err := cx.LoadModule(moePTX)
	if err != nil {
		t.Fatal(err)
	}
	fnOld, err := moeMod.Function("gemv_w4a8_moe")
	if err != nil {
		t.Fatal(err)
	}
	glueMod, err := cx.LoadModule(gluePTX)
	if err != nil {
		t.Fatal(err)
	}
	fnGluOld, err := glueMod.Function("glu_quant")
	if err != nil {
		t.Fatal(err)
	}
	rowsMod, err := cx.LoadModule(moeRowsPTX)
	if err != nil {
		t.Fatal(err)
	}
	fnRows, err := rowsMod.Function("gemv_w4a8_moe_rows")
	if err != nil {
		t.Fatal(err)
	}
	fnGluRows, err := rowsMod.Function("glu_quant_rows")
	if err != nil {
		t.Fatal(err)
	}
	stream := mustStream(t, cx)
	var rng uint32 = 20261019
	next := func() uint32 { rng = rng*1664525 + 1013904223; return rng }
	sync := func() {
		t.Helper()
		if e := stream.Synchronize(bg); e != nil {
			t.Fatal(e)
		}
	}

	checked := 0
	for _, g := range []struct{ N, K int }{{1792, 2304}, {2304, 896}, {40, 96}} {
		const nSlots, slot = 3, 1
		kw, kg := g.K/8, g.K/32
		mats := make([]linalg.WeightMat, nSlots)
		ptrs := make([]*linalg.WeightMat, nSlots)
		for e := range mats {
			f := make([]float32, g.N*g.K)
			for i := range f {
				f[i] = float32(int32(next()>>8)%2000-1000) / 700
			}
			mats[e] = linalg.QuantizeInt4(f, g.N, g.K, 32)
			ptrs[e] = &mats[e]
		}
		stack, err := packWeightStack(ptrs...)
		if err != nil {
			t.Fatal(err)
		}
		// the stack has nSlots*N rows; the kernels index slot*N + row
		dW, dGs := mustAlloc[uint32](t, cx, len(stack.wpk)), mustAlloc[uint16](t, cx, len(stack.ws16))
		dSlot := mustAlloc[uint32](t, cx, 1)
		for _, e := range []error{gc.CopyHtoD(bg, dW, stack.wpk), gc.CopyHtoD(bg, dGs, stack.ws16), gc.CopyHtoD(bg, dSlot, []uint32{slot})} {
			if e != nil {
				t.Fatal(e)
			}
		}
		const nRows = 50
		aWords := 2 * kw
		act := make([]int32, nRows*aWords)
		for i := range act {
			var p int32
			for b := range 4 {
				p |= (int32(int8(next()>>24)) & 0xff) << (8 * uint(b))
			}
			act[i] = p
		}
		scale := make([]float32, nRows)
		for i := range scale {
			scale[i] = float32(next()>>10%1000+1) / 40000
		}
		dA, dSc := mustAlloc[int32](t, cx, len(act)), mustAlloc[float32](t, cx, nRows)
		for _, e := range []error{gc.CopyHtoD(bg, dA, act), gc.CopyHtoD(bg, dSc, scale)} {
			if e != nil {
				t.Fatal(e)
			}
		}
		// per-row copies for the reference launches (the old kernel takes whole buffers)
		rowA := make([]*gc.Buffer[int32], nRows)
		rowS := make([]*gc.Buffer[float32], nRows)
		for r := range nRows {
			rowA[r], rowS[r] = mustAlloc[int32](t, cx, aWords), mustAlloc[float32](t, cx, 1)
			if e := gc.CopyHtoD(bg, rowA[r], act[r*aWords:(r+1)*aWords]); e != nil {
				t.Fatal(e)
			}
			if e := gc.CopyHtoD(bg, rowS[r], scale[r:r+1]); e != nil {
				t.Fatal(e)
			}
		}
		for _, nP := range []int{1, 7, 8, 9, 37} {
			aIdx, dIdx := make([]uint32, nP), make([]uint32, nP)
			perm := make([]uint32, nP)
			for i := range perm {
				perm[i] = uint32(i)
			}
			for i := nP - 1; i > 0; i-- {
				j := int(next() % uint32(i+1))
				perm[i], perm[j] = perm[j], perm[i]
			}
			for p := range nP {
				aIdx[p] = next() % nRows // repeats allowed: two pairs may share an activation row
				dIdx[p] = perm[p]
			}
			dAIdx, dDIdx := mustAlloc[uint32](t, cx, nP), mustAlloc[uint32](t, cx, nP)
			dOut := mustAlloc[float32](t, cx, nP*g.N)
			for _, e := range []error{gc.CopyHtoD(bg, dAIdx, aIdx), gc.CopyHtoD(bg, dDIdx, dIdx)} {
				if e != nil {
					t.Fatal(e)
				}
			}
			if e := cx.Synchronize(bg); e != nil {
				t.Fatal(e)
			}
			if e := fnRows.LaunchOn(bg, stream, gc.LaunchConfig{GridX: uint32((g.N + 7) / 8), GridY: uint32((nP + moeRowsTile - 1) / moeRowsTile), GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
				gc.Arg(dW), gc.Arg(dA), gc.Arg(dGs), gc.Arg(dSc), gc.Arg(dSlot), gc.ArgValue(int32(g.N)), gc.ArgValue(int32(kw)), gc.ArgValue(int32(kg)),
				gc.Arg(dAIdx), gc.Arg(dDIdx), gc.ArgValue(int32(nP)), gc.Arg(dOut)); e != nil {
				t.Fatal(e)
			}
			sync()
			got := make([]float32, nP*g.N)
			if e := gc.CopyDtoH(bg, got, dOut); e != nil {
				t.Fatal(e)
			}
			distinct := map[uint32]bool{}
			for p := range nP {
				dRef := mustAlloc[float32](t, cx, g.N)
				if e := fnOld.LaunchOn(bg, stream, gc.LaunchConfig{GridX: uint32((g.N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
					gc.Arg(dW), gc.Arg(rowA[aIdx[p]]), gc.Arg(dGs), gc.Arg(rowS[aIdx[p]]), gc.Arg(dSlot), gc.ArgValue(int32(0)), gc.ArgValue(int32(g.N)),
					gc.ArgValue(int32(g.N)), gc.ArgValue(int32(kw)), gc.ArgValue(int32(kg)), gc.Arg(dRef)); e != nil {
					t.Fatal(e)
				}
				sync()
				want := make([]float32, g.N)
				if e := gc.CopyDtoH(bg, want, dRef); e != nil {
					t.Fatal(e)
				}
				dRef.Close()
				for i, w := range want {
					distinct[math.Float32bits(w)] = true
					if gv := got[int(dIdx[p])*g.N+i]; math.Float32bits(gv) != math.Float32bits(w) {
						t.Fatalf("N=%d K=%d pairs=%d pair %d row %d: grouped %v, per-pair %v", g.N, g.K, nP, p, i, gv, w)
					}
				}
				checked++
			}
			if len(distinct) < g.N/2 {
				t.Fatalf("VOID: N=%d K=%d: the reference holds only %d distinct values, the comparison proves nothing", g.N, g.K, len(distinct))
			}
			dAIdx.Close()
			dDIdx.Close()
			dOut.Close()
		}

		// glu_quant_rows against glu_quant, per row, for both activations
		if g.K == 2304 {
			inter := 896
			gu := 2 * inter
			for _, act := range []int32{0, 1} {
				const nP = 21
				buf := make([]float32, nP*gu)
				for i := range buf {
					buf[i] = float32(int32(next()>>8)%4000-2000) / 300
				}
				dBuf := mustAlloc[float32](t, cx, len(buf))
				dQ, dS, dScr := mustAlloc[int32](t, cx, nP*inter/4), mustAlloc[float32](t, cx, nP), mustAlloc[float32](t, cx, nP*inter)
				if e := gc.CopyHtoD(bg, dBuf, buf); e != nil {
					t.Fatal(e)
				}
				if e := cx.Synchronize(bg); e != nil {
					t.Fatal(e)
				}
				if e := fnGluRows.LaunchOn(bg, stream, gc.LaunchConfig{GridX: nP, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 32 * 4},
					gc.Arg(dBuf), gc.ArgValue(int32(gu)), gc.ArgValue(int32(0)), gc.ArgValue(int32(inter)), gc.ArgValue(int32(inter)), gc.ArgValue(act),
					gc.Arg(dQ), gc.Arg(dS), gc.Arg(dScr)); e != nil {
					t.Fatal(e)
				}
				sync()
				gotQ, gotS := make([]int32, nP*inter/4), make([]float32, nP)
				if e := gc.CopyDtoH(bg, gotQ, dQ); e != nil {
					t.Fatal(e)
				}
				if e := gc.CopyDtoH(bg, gotS, dS); e != nil {
					t.Fatal(e)
				}
				for p := range nP {
					rowBuf := mustAlloc[float32](t, cx, gu)
					rq, rs, rscr := mustAlloc[int32](t, cx, inter/4), mustAlloc[float32](t, cx, 1), mustAlloc[float32](t, cx, inter)
					if e := gc.CopyHtoD(bg, rowBuf, buf[p*gu:(p+1)*gu]); e != nil {
						t.Fatal(e)
					}
					if e := cx.Synchronize(bg); e != nil {
						t.Fatal(e)
					}
					if e := fnGluOld.LaunchOn(bg, stream, gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: uint32(glueQuantThreads), BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(glueQuantThreads * 4)},
						gc.Arg(rowBuf), gc.Arg(rowBuf), gc.ArgValue(int32(0)), gc.ArgValue(int32(inter)), gc.ArgValue(int32(inter)), gc.ArgValue(act),
						gc.Arg(rq), gc.Arg(rs), gc.Arg(rscr)); e != nil {
						t.Fatal(e)
					}
					sync()
					wantQ, wantS := make([]int32, inter/4), make([]float32, 1)
					if e := gc.CopyDtoH(bg, wantQ, rq); e != nil {
						t.Fatal(e)
					}
					if e := gc.CopyDtoH(bg, wantS, rs); e != nil {
						t.Fatal(e)
					}
					if math.Float32bits(gotS[p]) != math.Float32bits(wantS[0]) {
						t.Fatalf("act %d row %d: glu_quant_rows scale %v, glu_quant %v", act, p, gotS[p], wantS[0])
					}
					for i, w := range wantQ {
						if gotQ[p*inter/4+i] != w {
							t.Fatalf("act %d row %d word %d: glu_quant_rows %#x, glu_quant %#x", act, p, i, gotQ[p*inter/4+i], w)
						}
					}
					rowBuf.Close()
					rq.Close()
					rs.Close()
					rscr.Close()
					checked++
				}
				dBuf.Close()
				dQ.Close()
				dS.Close()
				dScr.Close()
			}
		}
		for r := range nRows {
			rowA[r].Close()
			rowS[r].Close()
		}
		dW.Close()
		dGs.Close()
		dSlot.Close()
		dA.Close()
		dSc.Close()
	}
	t.Logf("%d (pair or row) outputs sets bit-identical to the per-pair kernels", checked)
}

// moeRowsResidual loads dir with C' staging on 3 slots (fewer than the experts, so the cache evicts), runs the expert-major prefill and returns every prompt
// row's f32 residual. grouped picks the grouped kernels or the per-pair launches; it reports how many layers took the grouped path.
func moeRowsResidual(t *testing.T, dir string, grouped bool, M int) ([][]float32, int64) {
	t.Helper()
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
	t.Setenv("GOINFER_MOE_CACHE_SLOTS", "3")
	t.Setenv("GOINFER_CUDA_MOE_EXPERT_MAJOR", "1")
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Skipf("cuda load: %v", err)
	}
	defer mc.Close()
	r, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok || !r.cacheExperts {
		t.Skip("the C' resident did not engage")
	}
	if batched, why := r.PrefillPath(); !batched {
		t.Skipf("batched prefill declined: %s", why)
	}
	r.moeRowsOff = !grouped
	_, _, _, _, _, _, vocab := mc.Dims()
	embs := make([][]float32, M)
	var s uint32 = 4242
	for i := range embs {
		s = s*1664525 + 1013904223
		embs[i] = append([]float32(nil), mc.EmbedResidentForTest(int(s>>8)%(vocab-1))...)
	}
	before := cudaMoeRowsRuns
	rows, err := r.ResidualAll(context.Background(), embs, 0)
	if err != nil {
		t.Fatalf("ResidualAll: %v", err)
	}
	return rows, cudaMoeRowsRuns - before
}

func residualDiff(a, b [][]float32) (diff, total int) {
	for m := range a {
		for i := range a[m] {
			total++
			if math.Float32bits(a[m][i]) != math.Float32bits(b[m][i]) {
				diff++
			}
		}
	}
	return
}

// TestMoERows_residualIdenticalToPerPair: with the cache evicting, every prompt row's residual after the last layer is the same bits whether the experts run
// as grouped launches or as per-pair launches, at prompt lengths around the 8-pair tile (the pairs per expert are about M*topK/8). The per-pair path is held
// to the per-row path and to the fully resident build by TestMoEExpertMajorCUDA_residualIdentical.
func TestMoERows_residualIdenticalToPerPair(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	for _, dir := range []string{"../testdata/qwen3moe-tiny", "../testdata/qwen3moe-tiny-k3"} {
		for _, M := range []int{5, 40, 133} {
			if _, err := os.Stat(dir); err != nil {
				t.Skipf("no fixture at %s", dir)
			}
			per, ranPer := moeRowsResidual(t, dir, false, M)
			grp, ranGrp := moeRowsResidual(t, dir, true, M)
			if ranPer != 0 || ranGrp == 0 {
				t.Fatalf("%s M=%d: grouped layers: %d with the grouped path off, %d with it on: the arms are not the ones named", dir, M, ranPer, ranGrp)
			}
			if len(per) != len(grp) || len(per) == 0 {
				t.Fatalf("%s M=%d: rows %d vs %d", dir, M, len(per), len(grp))
			}
			if d, n := residualDiff(per, grp); d != 0 {
				t.Errorf("%s M=%d: grouped differs from per-pair on %d of %d residual elements", dir, M, d, n)
			} else {
				t.Logf("%s M=%d: grouped == per-pair on all %d residual elements (%d grouped layers)", dir, M, n, ranGrp)
			}
		}
	}
}

// TestMoERows_plantedRankSwapIsCaught: the identity gate above must be able to go red. With each pair's output filed under the wrong rank the residuals
// must differ from the per-pair path's; if they did not, TestMoERows_residualIdenticalToPerPair would pass on a grouped path with broken indexing.
func TestMoERows_plantedRankSwapIsCaught(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	const dir = "../testdata/qwen3moe-tiny-k3"
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no fixture at %s", dir)
	}
	per, _ := moeRowsResidual(t, dir, false, 40)
	moeRowsPlantRankSwapForTest = true
	defer func() { moeRowsPlantRankSwapForTest = false }()
	bad, ran := moeRowsResidual(t, dir, true, 40)
	if ran == 0 {
		t.Fatal("VOID: the grouped path never ran")
	}
	if d, n := residualDiff(per, bad); d == 0 {
		t.Errorf("VOID: with every pair filed under the wrong rank the residuals still equal the per-pair path's (%d elements), so the identity gate cannot see an indexing fault", n)
	} else {
		t.Logf("planted rank swap: %d of %d residual elements differ, as they must", d, n)
	}
}
