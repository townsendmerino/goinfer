//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// TestEP05_deepRowsAB is E-P05's in-process A/B (docs/audit-metal-2026-09-30.md, "E-P05"): an 8-row batched step with
// every row on its own slot at depth D (past attnFADepthFloor, so every row runs flash attention), the deep rows' first
// pass and combine as one multi-row dispatch pair each (mc3FARowsOn, parked off) against one pair per row. Every step's
// logits must be bit-equal between the arms; GPU time per step, arms alternated per rep, 7 reps of 12 steps. Slots are
// filled by the batched prefill pass (their K/V is the pass's, which is all a timing needs). The MC3 checkpoint
// (mc3LoadCheckpoint); GOINFER_EP05_DEPTHS (default 1100,2048,4096); GOINFER_EP05_B (default 8).
//
//	GOINFER_METAL_MC3=1 go test -tags goinfer_testhooks -count=1 -run '^TestEP05_deepRowsAB$' -v ./metal/
func TestEP05_deepRowsAB(t *testing.T) {
	depths := []int{1100, 2048, 4096}
	if v := os.Getenv("GOINFER_EP05_DEPTHS"); v != "" {
		depths = nil
		for _, f := range strings.Split(v, ",") {
			var n int
			fmt.Sscan(strings.TrimSpace(f), &n)
			depths = append(depths, n)
		}
	}
	B := 8
	if v := os.Getenv("GOINFER_EP05_B"); v != "" {
		fmt.Sscan(v, &B)
	}
	maxD := 0
	for _, D := range depths {
		maxD = max(maxD, D)
	}
	_, r := mc3RealResident(t, B, maxD+64)
	prev := mc3FARowsOn
	defer func() { mc3FARowsOn = prev }()
	t0 := time.Now()
	seed := uint32(5550555)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	for _, D := range depths {
		for sl := range B {
			if err := r.useKVSlot(sl); err != nil {
				t.Fatal(err)
			}
			embs := make([][]float32, D)
			for i := range embs {
				embs[i] = mc3Emb(r, rnd())
			}
			if r.PrefillLast(embs, 0) == nil {
				t.Fatalf("depth %d slot %d: the pass declined", D, sl)
			}
		}
		if !r.canUseAttnFAAt(0, D+1) {
			t.Fatalf("depth %d is below the attention floor: the rows would not run flash attention", D)
		}
		auditHB("e-p05", t0, "depth %d: %d slots filled", D, B)
		ms := map[bool][]float64{}
		for rep := range 8 {
			for _, on := range []bool{rep%2 == 0, rep%2 != 0} {
				mc3FARowsOn = on
				var ks []float64
				for st := range 12 {
					emb := mc3Emb(r, 100+st)
					seqs := make([]batchSeq, B)
					for m := range B {
						seqs[m] = batchSeq{slot: m, pos: D, emb: emb}
					}
					out := mc3Step(t, r, seqs)
					ks = append(ks, (r.gpuEnd-r.gpuStart)*1e3)
					if on {
						continue
					}
					mc3FARowsOn = true
					ref := mc3Step(t, r, seqs) // the same step, rows on, for the identity check (not timed)
					mc3FARowsOn = false
					for m := range out {
						for j := range out[m] {
							if math.Float32bits(out[m][j]) != math.Float32bits(ref[m][j]) {
								t.Fatalf("depth %d step %d row %d logit %d: multi-row %v, per-row %v", D, st, m, j, ref[m][j], out[m][j])
							}
						}
					}
				}
				if rep > 0 {
					ms[on] = append(ms[on], auditMedian(ks))
				}
			}
		}
		ratio := make([]float64, len(ms[true]))
		above := 0
		for i := range ratio {
			ratio[i] = ms[false][i] / ms[true][i]
			if ratio[i] > 1 {
				above++
			}
		}
		auditHB("e-p05", t0, "RESULT depth %d B=%d: per-row %.3f ms, multi-row %.3f ms; per-row/multi-row median %.3f, %d of %d reps above 1 (per rep %s); logits bit-equal",
			D, B, auditMedian(ms[false]), auditMedian(ms[true]), auditMedian(ratio), above, len(ratio), auditFmt3(ratio))
	}
}
