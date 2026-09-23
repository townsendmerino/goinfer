package decoder

import "fmt"

// moePreadRateBytesPerSec is S4 item 5's own default pread-rate prior
// (task-never-swap-2026-09.md): docs/completed/task-w4a8-neon-bandwidth.md measured ~3.7 GB/s at
// concurrency 1 on this Mac's SSD. A DEFAULT, not a live measurement — the brief's own "Read
// first" note names the eventual improvement ("until the guard measures its own, one 64 MB pread
// at load"), not built here; this constant is what stands in for it until that lands.
const moePreadRateBytesPerSec = 3.7e9

// moeSlowTokPerSecThreshold is S4 item 5's registered rule (c): below this predicted rate, a
// paged-MoE load is refused without an explicit --accept-slow. A var, not a const, so a test can
// override it without needing a fixture whose real byte counts happen to predict a rate this low
// (a fixture that big is exactly what this session cannot safely load — see S5's own scoping note).
var moeSlowTokPerSecThreshold = 2.0

// moeHitRatePrior maps a residency fraction (pager budget / total mapped expert bytes) to a
// predicted LRU cache hit rate, interpolated from the only measured hit-rate curve in this tree —
// CUDA's C′ expert cache, Gemma 4 26B-A4B int4, benchmarks.md §B4.1 (12.5% residency -> 57.3% hit
// rate, 23.4% -> 76.1%, 31.25% -> 82.2%).
//
// BORROWED, NOT MEASURED FOR THIS PAGER — stated as a PRIOR, per the brief's own wording, not a
// claim about this pager's real behavior: a different architecture (CUDA host<->VRAM streaming vs
// this pager's mmap/pread), though the SAME eviction discipline (LRU over routed-expert demand),
// which is why the SHAPE is expected to transfer even though the exact numbers will not.
//
// Piecewise-linear between the calibration points; clamped to [0, 0.999] so a zero budget predicts
// a zero hit rate (no free lunch) and a budget covering every expert predicts NEAR-certainty
// rather than exactly 1.0 — an exact 1.0 would make predictedMoETokPerSec's miss-bytes term
// vanish and report +Inf, which is a worse failure mode (an unusable number) than a merely
// optimistic finite one.
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

// predictedMoETokPerSec is S4 item 5's own arithmetic: predicted tok/s ~= 1 / (missBytes /
// preadRate). Compute time is deliberately OMITTED — this repo has no measured CPU-paged-MoE
// compute-only (I/O-free) per-token cost to anchor it on, and guessing one wrong would be worse
// than leaving it out (a confidently wrong estimate is worse than none, per this repo's own
// measurement discipline). Omitting it makes this an UPPER BOUND on tok/s (an optimistic
// prediction), not a point estimate: I/O dominates a CPU-paged MoE's per-token cost in every real
// measurement this tree has (R-05, task-recompute-audit.md: "MoE is ~70% of a CPU-paged 35B
// token"), so the omitted compute term can only make the real rate SMALLER, never larger — the
// same "can only get stricter" direction S4 items 1/3 already use for their own margins, applied
// here to a prediction rather than a refusal threshold.
//
// activeBytesPerToken is topK x expertBytes + denseCoreBytes — the caller's job to resolve
// (moeWorkingSetPrediction, below); this function is pure arithmetic over numbers already known.
// hitRate is clamped to [0, 0.999] for the same reason moeHitRatePrior clamps its own output.
// Returns 0 (not a rate) for any input this function cannot turn into a positive, finite
// prediction — "don't know" is what the caller should treat that as, not "zero tok/s".
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

// moeWorkingSetRefusal is S4 item 5's registered rule (c): below moeSlowTokPerSecThreshold, a
// paged-MoE load is refused unless acceptSlow. Pure threshold logic, split out from
// moeWorkingSetPrediction's own real-number arithmetic so the refusal path can be tested with a
// synthetic predicted rate instead of needing a fixture large enough to genuinely predict a slow
// one (this session has no such fixture it can safely load — see S5's own scoping note in this
// same task doc). predicted <= 0 ("don't know") never refuses — same "every unknown proceeds"
// discipline as the rest of this file.
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
