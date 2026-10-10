package decoder

import "fmt"

// moePreadRateBytesPerSec is the default pread-rate prior for the paged-MoE working-set prediction: about 3.7 GB/s at
// concurrency 1 on the dev Mac's SSD (docs/completed/task-w4a8-neon-bandwidth.md). It is a default, not a live
// measurement of the machine at hand. Having the guard measure its own rate (one 64 MB pread at load) is not built;
// docs/tasks/task-never-swap-2026-09.md names it.
const moePreadRateBytesPerSec = 3.7e9

// moeSlowTokPerSecThreshold is the refusal rule: below this predicted rate, a paged-MoE load is refused without an
// explicit --accept-slow. A var, not a const, so a test can override it without a fixture whose real byte counts
// predict a rate this low.
var moeSlowTokPerSecThreshold = 2.0

// moeHitRatePrior maps a residency fraction (pager budget / total mapped expert bytes) to a predicted LRU cache hit
// rate, piecewise-linearly between calibration points taken from the only measured hit-rate curve in this tree: CUDA's
// C' expert cache on Gemma 4 26B-A4B int4 (docs/benchmarks.md §B4.1).
//
// It is a borrowed prior, not a measurement of this pager: a different architecture (CUDA host-to-VRAM streaming
// against this pager's mmap or pread) with the same eviction discipline (LRU over routed-expert demand), so the shape
// is expected to transfer and the exact numbers are not. Say "prior" in any message that quotes the result.
//
// The output is clamped to [0, 0.999]: a zero budget predicts a zero hit rate, and a budget covering every expert
// predicts near-certainty rather than 1.0, which would make predictedMoETokPerSec's miss-bytes term vanish and report
// +Inf, a worse failure (an unusable number) than a merely optimistic finite one.
func moeHitRatePrior(residencyFraction float64) float64 {
	type point struct{ x, y float64 }
	pts := [...]point{{0, 0}, {0.125, 0.573}, {0.234, 0.761}, {0.3125, 0.822}, {1.0, 0.999}}
	switch {
	case residencyFraction <= pts[0].x:
		return pts[0].y
	case residencyFraction >= pts[len(pts)-1].x:
		return pts[len(pts)-1].y
	}
	for i := 1; i < len(pts); i++ {
		if residencyFraction <= pts[i].x {
			lo, hi := pts[i-1], pts[i]
			t := (residencyFraction - lo.x) / (hi.x - lo.x)
			return lo.y + t*(hi.y-lo.y)
		}
	}
	return pts[len(pts)-1].y // unreachable given the clamps above; keeps the compiler happy
}

// predictedMoETokPerSec predicts tok/s as 1 / (missBytes / preadRate), where missBytes = activeBytesPerToken x
// (1 - hitRate). Compute time is deliberately omitted: there is no measured I/O-free CPU-paged-MoE per-token cost to
// anchor it, and a confidently wrong estimate is worse than none. That makes this an upper bound on tok/s, not a point
// estimate: I/O dominates a CPU-paged MoE's per-token cost, so the omitted compute term can only make the real rate
// smaller.
//
// activeBytesPerToken is topK x expertBytes + denseCoreBytes; resolving it is the caller's job
// (moeWorkingSetPrediction). hitRate is clamped to [0, 0.999] as moeHitRatePrior clamps its output. It returns 0,
// meaning "don't know", for any input it cannot turn into a positive finite prediction; a caller must not read that as
// zero tok/s.
func predictedMoETokPerSec(activeBytesPerToken int64, hitRate float64) float64 {
	if activeBytesPerToken <= 0 {
		return 0
	}
	switch {
	case hitRate < 0:
		hitRate = 0
	case hitRate > 0.999:
		hitRate = 0.999
	}
	missBytes := float64(activeBytesPerToken) * (1 - hitRate)
	if missBytes <= 0 {
		return 0
	}
	ioSeconds := missBytes / moePreadRateBytesPerSec
	if ioSeconds <= 0 {
		return 0
	}
	return 1 / ioSeconds
}

// moeWorkingSetPrediction resolves activeBytesPerToken and the residency fraction from a
// just-loaded Model and its just-built expert pager, then predicts tok/s via
// moeHitRatePrior + predictedMoETokPerSec. ok is false whenever an input isn't known well enough
// to predict anything (every unknown proceeds, same discipline as every guard in this file) — the
// caller must not read tokPerSec when ok is false.
//
// activeBytesPerToken = (layers x topK) x average-expert-bytes + denseCoreBytes:
//   - average-expert-bytes is p.total / p.nExperts — exact for a uniform-shape MoE (every expert
//     in a real architecture has the same Gate/Up/Down shape; ResidentWeightBytesPaged's own doc
//     comment makes the identical claim for the same reason).
//   - denseCoreBytes is m.ResidentDenseWeightBytes() — every matrix a routed-expert cap cannot
//     shrink (attention, shared expert, router, embeddings). This branch never pages the dense
//     part (only w.arch.MoE != nil triggers expertPager, and the dense weights stay unpaged
//     alongside it), so the WHOLE dense weight set is read every single token at M=1 decode —
//     exactly the quantity a per-token byte budget needs.
func moeWorkingSetPrediction(m *Model) (tokPerSec float64, ok bool) {
	if m == nil || m.w == nil || m.w.arch == nil || m.w.arch.MoE == nil || m.pager == nil {
		return 0, false
	}
	p := m.pager
	if p.nExperts <= 0 || p.total <= 0 {
		return 0, false
	}
	avgExpertBytes := p.total / int64(p.nExperts)
	if avgExpertBytes <= 0 {
		return 0, false
	}
	activeExpertsPerToken := int64(len(m.w.Layers)) * int64(m.w.arch.MoE.TopK)
	if activeExpertsPerToken <= 0 {
		return 0, false
	}
	activeBytesPerToken := activeExpertsPerToken*avgExpertBytes + m.ResidentDenseWeightBytes()
	budget := p.budget()
	if budget <= 0 {
		return 0, false
	}
	residency := float64(budget) / float64(p.total)
	predicted := predictedMoETokPerSec(activeBytesPerToken, moeHitRatePrior(residency))
	if predicted <= 0 {
		return 0, false
	}
	return predicted, true
}

// moeWorkingSetRefusal is the refusal rule: below moeSlowTokPerSecThreshold, a paged-MoE load is refused unless
// acceptSlow. It is pure threshold logic, split from moeWorkingSetPrediction's arithmetic so the refusal path can be
// tested with a synthetic predicted rate. predicted <= 0 ("don't know") never refuses: every unknown proceeds, as with
// every guard in this file.
func moeWorkingSetRefusal(modelName string, predicted float64, acceptSlow bool) error {
	if predicted <= 0 || predicted >= moeSlowTokPerSecThreshold || acceptSlow {
		return nil
	}
	return fmt.Errorf(
		"decoder: %s's paged-MoE working set predicts ~%.2f tok/s (a prior, not a measurement — "+
			"see moeHitRatePrior's own doc comment) below the %.1f tok/s floor. Pass --accept-slow "+
			"(decoder.Options.AcceptSlowMoE) to load it anyway, or raise -weight-cache to shrink "+
			"the miss rate this prediction assumes",
		modelName, predicted, moeSlowTokPerSecThreshold)
}
