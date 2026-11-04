//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math"
	"os"
	"testing"
	"time"
)

// TestR19_stepPrefillAB times R-19 (docs/tasks/task-recompute-audit.md): a short fresh prompt through PrefillLast on the
// step route with every piece's head run (stepPrefillAllHeads, as it was) against R-19's head-less pieces and last-row
// copy, arms alternated rep by rep in one process after a warm-up, last-row logits compared every rep. Prompts of 16
// and 31 tokens (1 and 3 head-less pieces). The checkpoint is mc3LoadCheckpoint's (GOINFER_METAL_MC3_MODEL, else the
// 1.5B). By day, in-process, exploratory.
//
//	GOINFER_R19_AB=1 GOINFER_METAL_MC3=1 go test -tags goinfer_testhooks -count=1 -run '^TestR19_stepPrefillAB$' -v ./metal/
func TestR19_stepPrefillAB(t *testing.T) {
	if os.Getenv("GOINFER_R19_AB") != "1" {
		t.Skip("set GOINFER_R19_AB=1 (times a real checkpoint)")
	}
	a := mc3PrefillResident(t, 2, 1024)
	r := a.r
	if a.VerifyCost() == nil {
		t.Fatalf("no step kernels (batchIneligible: %q)", r.batchIneligible())
	}
	defer func() { stepPrefillAllHeads = false }()
	t0 := time.Now()
	reps := auditReps(7)
	for _, n := range []int{16, 31} {
		embs := make([][]float32, n)
		for i := range embs {
			embs[i] = mc3Emb(r, (i*7919+3)%20000)
		}
		ms := map[bool][]float64{}
		var ref []float32
		for rep := range reps + 1 {
			order := []bool{true, false}
			if rep%2 == 1 {
				order = []bool{false, true}
			}
			for _, all := range order {
				stepPrefillAllHeads = all
				st := time.Now()
				lg, err := a.PrefillLast(context.Background(), embs, 0)
				if err != nil {
					t.Fatal(err)
				}
				dt := float64(time.Since(st).Microseconds()) / 1e3
				if ref == nil {
					ref = append([]float32(nil), lg...)
				}
				for j := range ref {
					if math.Float32bits(ref[j]) != math.Float32bits(lg[j]) {
						t.Fatalf("%d tokens rep %d: logit %d differs between the arms", n, rep, j)
					}
				}
				if rep > 0 {
					ms[all] = append(ms[all], dt)
				}
			}
		}
		ratio := make([]float64, reps)
		above := 0
		for i := range ratio {
			ratio[i] = ms[true][i] / ms[false][i]
			if ratio[i] > 1 {
				above++
			}
		}
		auditHB("r19", t0, "%d tokens: every head %.2f ms, R-19 %.2f ms (medians of %d); RESULT every-head / R-19 median %.3f, %d of %d reps above 1 (per rep %s)",
			n, auditMedian(ms[true]), auditMedian(ms[false]), reps, auditMedian(ratio), above, reps, auditFmt3(ratio))
	}
}
