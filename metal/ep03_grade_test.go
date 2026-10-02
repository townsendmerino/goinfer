//go:build darwin

package metal

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestEP03StepAB is E-P03's grade (docs/tasks/task-metal-audit-2026-10.md, "E-P03: built, pending its grade"): the MC3
// batched step with the fragment FB calibrateFB chose for this model against the same step with FB = 2 everywhere (the
// shipped build), in one process on one resident, at B = 4 and 8 sequences and depths 128 and 2048. Each rep runs both
// arms, alternated, `steps` steps each at the same positions with the same inputs; an arm's time is the median step GPU
// time. Every rep's first-step logits must be equal across the arms (FB changes which rows a simdgroup owns, not a sum).
// If calibration chose FB = 2 everywhere, the arms are the same build and the reading is 1 by construction.
//
//	GOINFER_METAL_MC3=1 GOINFER_METAL_EP03=1 [GOINFER_METAL_MC3_MODEL=...] go test -run '^TestEP03StepAB$' -v ./metal/
func TestEP03StepAB(t *testing.T) {
	if os.Getenv("GOINFER_METAL_EP03") != "1" {
		t.Skip("set GOINFER_METAL_EP03=1: E-P03's timed grade (night-only)")
	}
	const maxB, ctx, steps = 8, 2560, 6
	m, r := mc3RealResident(t, maxB, ctx)
	_ = m
	b := r.batch
	name := filepath.Base(os.Getenv("GOINFER_METAL_MC3_MODEL"))
	auto := [4]int{b.fbQKV, b.fbO, b.fbGU, b.fbD}
	set := func(fb [4]int) { b.fbQKV, b.fbO, b.fbGU, b.fbD = fb[0], fb[1], fb[2], fb[3] }
	defer set(auto)
	t0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[ep03 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	hb("%s: calibrated FB qkv %d, o %d, gate|up %d, down %d", name, auto[0], auto[1], auto[2], auto[3])
	if auto == [4]int{2, 2, 2, 2} {
		hb("%s: calibration chose FB = 2 everywhere; the arms are the same build", name)
	}
	seed := uint32(97531)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	reps := auditReps(9)
	for _, D := range []int{128, 2048} {
		ids := make([]int, D)
		for i := range ids {
			ids[i] = rnd()
		}
		for s := range maxB {
			mc3Fill(t, r, s, ids)
		}
		for _, B := range []int{4, 8} {
			embs := make([][][]float32, steps)
			for st := range embs {
				embs[st] = make([][]float32, B)
				for i := range B {
					embs[st][i] = mc3Emb(r, rnd())
				}
			}
			run := func(fb [4]int) (float64, []float32) {
				set(fb)
				var ts []float64
				var first []float32
				for st := range steps {
					seqs := make([]batchSeq, B)
					for i := range B {
						seqs[i] = batchSeq{slot: i, pos: D, emb: embs[st][i]}
					}
					out := mc3Step(t, r, seqs)
					if st == 0 {
						first = append([]float32(nil), out[0]...)
					}
					ts = append(ts, (r.gpuEnd-r.gpuStart)*1e3)
				}
				return auditMedian(ts), first
			}
			var ratios, ta, tb []float64
			for rep := range reps {
				var a, f float64
				var la, lf []float32
				if rep%2 == 0 {
					a, la = run(auto)
					f, lf = run([4]int{2, 2, 2, 2})
				} else {
					f, lf = run([4]int{2, 2, 2, 2})
					a, la = run(auto)
				}
				for i := range la {
					if la[i] != lf[i] {
						t.Fatalf("B=%d depth %d rep %d: logit %d differs between the calibrated FB and FB = 2 (%v against %v)", B, D, rep, i, la[i], lf[i])
					}
				}
				ratios, ta, tb = append(ratios, f/a), append(ta, a), append(tb, f)
			}
			above := 0
			for _, x := range ratios {
				if x > 1 {
					above++
				}
			}
			hb("%s B=%d depth %d: step %.3f ms calibrated, %.3f ms FB=2 (medians of %d reps x %d steps); logits equal in every rep",
				name, B, D, auditMedian(ta), auditMedian(tb), reps, steps)
			fmt.Fprintf(os.Stderr, "[ep03] %s E-P03 METRIC fb2/calibrated B=%d depth %d: median %.4f (%d of %d reps above 1)\n",
				name, B, D, auditMedian(ratios), above, reps)
		}
	}
}
