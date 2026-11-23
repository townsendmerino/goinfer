//go:build cuda

package kernels

import (
	"fmt"
	"math"
	"slices"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestFlashDecodeHD96_againstFloat64 is gate 1 of docs/measurements/fa-hd96-phi3-2026-11-22/README.md: the head-dim-96 flash-decode
// kernels from decode_fa and decode_fa_kv16 against the float64 reference, at the lane's G2-K bar (cosine >= 0.9999, max abs <= 1e-2).
// fa_partial_96 with fa_combine at one query (4 and 16 splits); fa_partial_rows_96_g{G} with fa_combine_rows over verify batches of 1, 5
// and 16 rows, each row against its own reference and bit for bit against the M = 1 lane at that row's depth; and fa_partial_ragged_96 with
// fa_combine_ragged, each listed row bit for bit against the M = 1 lane. GQA groups 1, 2, 4 and 8; windows 0 and 100 (1,500 for the
// ragged rows); plain and peaked scores. The reference is capReference with a cap of 1e30, where cap·tanh(s/cap) is s in float64.
func TestFlashDecodeHD96_againstFloat64(t *testing.T) {
	r, _ := newAdRig(t)
	const hd, noCap, amp = 96, 1e30, 4
	cases, rows, ragged := 0, 0, 0
	m1 := func(m adMod, q gc.KernelArg, kc, vc, kvt gc.KernelArg, nH, nKV, G, winStart, nKeys, nSplit int) []float32 {
		U := nSplit * faWarps
		part, ctx := r.zeros(nH*U*(hd+4)), r.zeros(nH*hd)
		r.launch(m, "fa_partial_96", adCfg(nKV, nSplit, 1, faWarps*32, (G*hd+faWarps*64)*4), q, kc, vc, kvt,
			adI(nH), adI(nKV), adI(winStart), adI(nKeys), gc.ArgValue(adScale(hd)), adI(nSplit), gc.Arg(part))
		r.launch(m, "fa_combine", adCfg(nH, 1, 1, hd, 3*U*4), gc.Arg(part), adI(hd), adI(U), gc.Arg(ctx))
		return r.read(ctx, nH*hd)
	}
	bitsEqual := func(id string, got, want []float32) {
		for j := range want {
			if math.Float32bits(got[j]) != math.Float32bits(want[j]) {
				t.Errorf("%s element %d: %v, the M = 1 lane gives %v", id, j, got[j], want[j])
				return
			}
		}
	}
	for kvi, m := range adFA {
		f16 := kvi == 1
		for _, G := range []int{1, 2, 4, 8} {
			for _, peaked := range []bool{false, true} {
				for _, window := range []int{0, 100} {
					nKV := 2
					nH := nKV * G
					id := func(k string, s ...any) string {
						return fmt.Sprintf("%s/%s/hd96/g%d/peaked=%v/win%d/%v", k, adKVNames[kvi], G, peaked, window, s)
					}
					for _, nKeys := range []int{130, 300, 2100} {
						in := capMakeInputs(1, nKeys, nH, nKV, hd, amp, peaked, 5, f16, 501)
						want, _, _, _ := capReference(in.q, in, nH, nKV, hd, nKeys, window, noCap, nil)
						winStart := 0
						if window > 0 && nKeys > window {
							winStart = nKeys - window
						}
						q, kc, vc := r.f32(in.q), r.kv(in.k, f16), r.kv(in.v, f16)
						for _, nSplit := range []int{4, 16} {
							capCompare(t, id("fa_partial_96", nKeys, "s", nSplit), m1(m, gc.Arg(q), kc, vc, r.kvt(nKeys), nH, nKV, G, winStart, nKeys, nSplit), want)
							cases++
						}
						r.release()
					}
					const nSplit, n0 = 16, 130
					for _, nRows := range []int{1, 5, 16} {
						nTot := n0 + nRows
						in := capMakeInputs(nRows, nTot, nH, nKV, hd, amp, peaked, 5, f16, 601)
						q, kc, vc := r.f32(in.q), r.kv(in.k, f16), r.kv(in.v, f16)
						U := nSplit * faWarps
						part, ctx := r.zeros(faMaxRows*nH*U*(hd+4)), r.zeros(nRows*nH*hd)
						R := faRowsPerCTA(hd, G)
						for _, run := range faRowRuns(n0, nRows, window, nSplit) {
							r.launch(m, fmt.Sprintf("fa_partial_rows_96_g%d", G), adCfg(nKV, nSplit, (run.N+R-1)/R, faWarps*32, (R*G*hd+faWarps*R*64)*4),
								gc.Arg(q), kc, vc, r.kvt(nTot), adI(nH), adI(nKV), adI(run.WinStart), adI(run.NKeys0), adI(run.Per), gc.ArgValue(adScale(hd)),
								adI(nSplit), adI(run.N), adI(run.Row0), gc.Arg(part))
							r.launch(m, "fa_combine_rows", adCfg(nH, run.N, 1, hd, 3*U*4), gc.Arg(part), adI(nH), adI(hd), adI(U), adI(run.Row0), gc.Arg(ctx))
						}
						got := r.read(ctx, nRows*nH*hd)
						for row := range nRows {
							qRow := in.q[row*nH*hd : (row+1)*nH*hd]
							want, _, _, _ := capReference(qRow, in, nH, nKV, hd, n0+row, window, noCap, nil)
							capCompare(t, id("fa_partial_rows_96", nRows, "row", row), got[row*nH*hd:(row+1)*nH*hd], want)
							nKeys := n0 + row
							winStart := 0
							if window > 0 && nKeys > window {
								winStart = nKeys - window
							}
							bitsEqual(id("fa_partial_rows_96 against M=1", nRows, "row", row), got[row*nH*hd:(row+1)*nH*hd],
								m1(m, gc.Arg(r.f32(qRow)), kc, vc, r.kvt(nTot), nH, nKV, G, winStart, nKeys, nSplit))
							rows++
						}
						cases++
						r.release()
					}
				}
			}
		}
		// The ragged step's twin: adCasesFlashRagged's rows (depths on both sides of the lane's floor and past a 1,500-key window, shared
		// slots, a launch list in a non-identity order), each listed row bit for bit against the M = 1 lane over its slot's table.
		for _, G := range []int{1, 4} {
			for _, window := range []int{0, 1500} {
				nKV, nSplit := 2, 16
				nH := nKV * G
				R, kvDim, qDim, nSlot := len(adFARaggedPos), nKV*hd, nH*hd, slices.Max(adFARaggedSlot)+1
				nb := (slices.Max(adFARaggedPos) + 1 + 63) / 64
				tabs, slots, pos := make([]int32, nSlot*nb), make([]int32, R), make([]int32, R)
				for i := range tabs {
					tabs[i] = int32(i)
				}
				for i, p := range adFARaggedPos {
					slots[i], pos[i] = int32(adFARaggedSlot[i]), int32(p)
				}
				q := adRand(R*qDim, 91, 1)
				kc, vc := r.kv(adRand(nSlot*nb*64*kvDim, 92, 1), f16), r.kv(adRand(nSlot*nb*64*kvDim, 93, 1), f16)
				U, K := nSplit*faWarps, len(adFARaggedIdx)
				part, ctx := r.zeros(K*nH*U*(hd+4)), r.zeros(R*qDim)
				r.launch(m, "fa_partial_ragged_96", adCfg(nKV, nSplit, K, faWarps*32, (G*hd+faWarps*64)*4), gc.Arg(r.f32(q)), kc, vc,
					gc.Arg(r.i32(tabs)), adI(nb), gc.Arg(r.i32(slots)), gc.Arg(r.i32(pos)), gc.Arg(r.i32(adFARaggedIdx)), adI(0),
					adI(nH), adI(nKV), adI(window), gc.ArgValue(adScale(hd)), adI(nSplit), gc.Arg(part))
				r.launch(m, "fa_combine_ragged", adCfg(nH, K, 1, hd, 3*U*4), gc.Arg(part), adI(nH), adI(hd), adI(U), adI(0),
					gc.Arg(r.i32(adFARaggedIdx)), gc.Arg(ctx))
				got := r.read(ctx, R*qDim)
				for _, i32 := range adFARaggedIdx {
					i := int(i32)
					nKeys := adFARaggedPos[i] + 1
					winStart := 0
					if window > 0 && nKeys > window {
						winStart = nKeys - window
					}
					s := adFARaggedSlot[i]
					bitsEqual(fmt.Sprintf("fa_partial_ragged_96/%s/g%d/w%d row %d", adKVNames[kvi], G, window, i), got[i*qDim:(i+1)*qDim],
						m1(m, gc.Arg(r.f32(q[i*qDim:(i+1)*qDim])), kc, vc, gc.Arg(r.i32(tabs[s*nb:(s+1)*nb])), nH, nKV, G, winStart, nKeys, nSplit))
					ragged++
				}
				r.release()
			}
		}
	}
	if cases != 288 || rows != 704 || ragged != 32 {
		t.Fatalf("%d cases, %d verify rows and %d ragged rows ran, want 288, 704 and 32: the matrix did not run whole", cases, rows, ragged)
	}
	if !t.Failed() {
		t.Logf("%d hd-96 flash-decode cases (%d verify rows, %d ragged rows) inside the bar against float64, at f32 and f16 KV", cases, rows, ragged)
	}
}
