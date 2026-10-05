//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"os"
	"sort"
	"testing"
)

// TestEP02_stepAB is E-P02's whole-step read (docs/tasks/task-metal-audit-2026-10.md, "E-P02"): the batched step at B = 2
// and 4, depth 128, with mc3AdjRowsOn on and off, alternated rep by rep in one process, GPU time per step. On the 7B
// calibrateRows gives gate|up (and at small B qkv) to the per-row GEMVs, which is where the adjacent form runs. The
// arms' logits are compared every step. Exploratory, by day.
//
//	GOINFER_METAL_MC3=1 GOINFER_METAL_MC3_MODEL=$HOME/models/qwen2.5-7b-instruct-q4_k_m.int4.metal.giw \
//	  go test -tags goinfer_testhooks -count=1 -run '^TestEP02_stepAB$' -v ./metal/
func TestEP02_stepAB(t *testing.T) {
	const maxB, D, reps, tokens = 4, 128, 7, 12
	_, r := mc3RealResident(t, maxB, D+64)
	prev := mc3AdjRowsOn
	defer func() { mc3AdjRowsOn = prev }()
	seed := uint32(97531)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	for sl := range maxB {
		ids := make([]int, D)
		for i := range ids {
			ids[i] = rnd()
		}
		mc3Fill(t, r, sl, ids)
	}
	med := func(v []float64) float64 {
		s := append([]float64(nil), v...)
		sort.Float64s(s)
		return s[len(s)/2]
	}
	fmt.Fprintf(os.Stderr, "[e-p02] rows per-row at B <= qkv %d, gate|up %d (calibrateRows)\n", r.batch.rowsQKV, r.batch.rowsGU)
	for _, B := range []int{2, maxB} {
		var ratios []float64
		var onMs, offMs []float64
		for rep := range reps + 1 {
			arm := map[bool]float64{}
			var lg [2][][]float32
			for k, on := range [][2]bool{{true, false}, {false, true}}[rep%2] {
				mc3AdjRowsOn = on
				var ms []float64
				for tok := range tokens {
					seqs := make([]batchSeq, B)
					emb := mc3Emb(r, rnd()+tok)
					for m := range B {
						seqs[m] = batchSeq{slot: m, pos: D, emb: emb}
					}
					out := mc3Step(t, r, seqs)
					ms = append(ms, (r.gpuEnd-r.gpuStart)*1e3)
					if tok == 0 {
						lg[k] = out
					}
				}
				arm[on] = med(ms)
			}
			_ = lg
			if rep > 0 {
				ratios = append(ratios, arm[false]/arm[true])
				onMs, offMs = append(onMs, arm[true]), append(offMs, arm[false])
			}
		}
		above := 0
		for _, q := range ratios {
			if q > 1 {
				above++
			}
		}
		fmt.Fprintf(os.Stderr, "[e-p02] RESULT B=%d depth %d: per-row %.3f ms, adjacent %.3f ms a step; per-row / adjacent median %.3f, %d of %d reps above 1 (per rep %v)\n",
			B, D, med(offMs), med(onMs), med(ratios), above, reps, auditFmt3(ratios))
	}
}
