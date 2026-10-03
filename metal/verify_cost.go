//go:build darwin

package metal

import (
	"fmt"
	"math"
	"os"
	"sort"
)

// verifyCostCalibrate turns E-P06's load-time measurement on. FALSE until its served grade passes
// (docs/tasks/task-metal-audit-2026-10.md, "E-P06: built, pending its grade"): main carries the build with --spec
// ngram still pricing the verify by stepVerifyCost, as before E-P06. Tests set it to exercise the measurement.
var verifyCostCalibrate = false

// verifyCostDepth is the depth calibrateVerifyCost measures the verify at: the dearest depth stepVerifyCost was taken
// at, so a model's own curve stays as conservative in depth as the constant it replaces and differs only by model.
const verifyCostDepth = 2048

// calibrateVerifyCost measures this model's argmax-only verify cost curve (E-P06, docs/audit-metal-2026-09-30.md), for
// VerifyCost to report instead of stepVerifyCost, which is the 7B's. It runs while the resident is built, so no
// generation is live: production's own token (ForwardEmb) and the step kernels' 2-, 4- and 8-row verify
// (forwardMultiInto, argmax-only, as PrefillLastNArgmax runs it) on slot 0 at verifyCostDepth, the arms interleaved,
// 7 rounds, the first 2 discarded, the GPU-time median of each. The K/V the rows write sits past anything cached
// and is overwritten before it is read. Any error, or a curve verifyCurveFrom refuses, keeps stepVerifyCost.
func (r *resident) calibrateVerifyCost() {
	if r.batch == nil || len(r.kvSlotBufs) == 0 {
		return
	}
	D := min(verifyCostDepth, r.ctxCap-batchMaxSeqs-1)
	if D < 1 {
		return
	}
	defer r.stopExec()
	emb := make([]float32, r.H)
	r.embed.Row(0, emb)
	arms := []int{1, 2, 4, 8} // 1 = production's token
	per := map[int][]float64{}
	for rep := range 7 {
		for k := range arms {
			M := arms[(k+rep)%len(arms)]
			if M == 1 {
				r.ForwardEmb(emb, D)
			} else {
				seqs := make([]batchSeq, M)
				for i := range seqs {
					seqs[i] = batchSeq{slot: 0, pos: D + i, emb: emb}
				}
				if _, _, err := r.forwardMultiInto(seqs, true); err != nil {
					fmt.Fprintf(os.Stderr, "metal: verify cost curve not measured (%v); the shipped curve stays\n", err)
					return
				}
			}
			if rep >= 2 {
				per[M] = append(per[M], r.gpuEnd-r.gpuStart)
			}
		}
	}
	med := func(xs []float64) float64 {
		v := append([]float64(nil), xs...)
		sort.Float64s(v)
		return v[len(v)/2]
	}
	curve, why := verifyCurveFrom(med(per[1]), med(per[2]), med(per[4]), med(per[8]))
	if why != "" {
		fmt.Fprintf(os.Stderr, "metal: verify cost curve refused (%s); the shipped curve stays\n", why)
		return
	}
	r.verifyCost = curve
	fmt.Fprintf(os.Stderr, "metal: verify cost curve at depth %d: 2 / 4 / 8 rows = %.2f / %.2f / %.2f tokens (shipped %.2f / %.2f / %.2f)\n",
		D, curve[2], curve[4], curve[8], stepVerifyCost[2], stepVerifyCost[4], stepVerifyCost[8])
}

// verifyCurveFrom is the cost curve, in single-token steps indexed by rows verified, from the GPU times of a token and
// of 2-, 4- and 8-row verifies: stepVerifyCost's shape, with 3 and 5-7 rows interpolated and 9 an 8-row step plus one
// token, as there. It refuses (why != "") a time that is not positive and finite, and a curve that does not rise from
// above 1, since the controller's choice assumes a row never costs less than nothing.
func verifyCurveFrom(tok, t2, t4, t8 float64) (curve []float64, why string) {
	for _, t := range []float64{tok, t2, t4, t8} {
		if !(t > 0) || math.IsInf(t, 0) {
			return nil, fmt.Sprintf("a GPU time of %g", t)
		}
	}
	c2, c4, c8 := t2/tok, t4/tok, t8/tok
	if !(1 < c2 && c2 <= c4 && c4 <= c8) {
		return nil, fmt.Sprintf("2 / 4 / 8 rows at %.3f / %.3f / %.3f tokens, not rising from above 1", c2, c4, c8)
	}
	lerp := func(a, b float64, f float64) float64 { return a + (b-a)*f }
	return []float64{0, 1, c2, lerp(c2, c4, 0.5), c4, lerp(c4, c8, 0.25), lerp(c4, c8, 0.5), lerp(c4, c8, 0.75), c8, c8 + 1}, ""
}
