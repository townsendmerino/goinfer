package mem

import "fmt"

// MoEPreadRateBytesPerSec is the default pread-rate prior for the paged-MoE working-set prediction: about 3.7 GB/s at
// concurrency 1 on the reference SSD. It is a default, not a live measurement of the machine at hand.
const MoEPreadRateBytesPerSec = 3.7e9

// MoESlowTokPerSecFloor is the default refusal floor: below this predicted decode rate, a paged-MoE or CPU mapped-MoE
// load is refused without an explicit --accept-slow.
const MoESlowTokPerSecFloor = 2.0

// MoEHitRatePrior maps a residency fraction (pager budget / total mapped expert bytes) to a predicted LRU cache hit
// rate, piecewise-linearly between calibration points taken from the measured hit-rate curve on CUDA's expert cache
// over Gemma 4 26B-A4B int4 (docs/benchmarks.md, "Host↔VRAM MoE streaming", the slot-count sweep).
//
// It is a borrowed prior, not a measurement of the host pager: a different transport with the same eviction discipline
// (LRU over routed-expert demand), so the shape is expected to transfer and the exact numbers are not. Any user-facing
// message quoting the result states that it is a prior.
//
// The output is clamped to [0, 0.999]: a zero budget predicts a zero hit rate, and a budget covering every expert
// predicts near-certainty rather than 1.0, which would make PredictedMoETokPerSec's miss-bytes term vanish and report
// +Inf.
func MoEHitRatePrior(residencyFraction float64) float64 {
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
	return pts[len(pts)-1].y
}

// PredictedMoETokPerSec predicts tok/s as 1 / (missBytes / MoEPreadRateBytesPerSec), where
// missBytes = activeBytesPerToken * (1 - hitRate). Compute time is deliberately omitted: I/O dominates a CPU-paged
// MoE's per-token cost, making this an upper bound on tok/s rather than a point estimate.
//
// hitRate is clamped to [0, 0.999] as MoEHitRatePrior clamps its output. It returns 0, meaning "unknown", for any
// input it cannot turn into a positive finite prediction; callers must not interpret 0 as zero tok/s.
func PredictedMoETokPerSec(activeBytesPerToken int64, hitRate float64) float64 {
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
	ioSeconds := missBytes / MoEPreadRateBytesPerSec
	if ioSeconds <= 0 {
		return 0
	}
	return 1 / ioSeconds
}

// PredictMoEWorkingSet predicts a model's decode rate from its routed experts (totalBytes mapped across nExperts
// experts, budgetBytes of them held resident, nLayers * topK active expert touches per token plus denseCoreBytes read
// every token). ok is false whenever any input is unknown or non-positive, following the guard contract that every
// unknown proceeds.
func PredictMoEWorkingSet(totalBytes int64, nExperts, nLayers, topK int, denseCoreBytes, budgetBytes int64) (tokPerSec float64, ok bool) {
	if nExperts <= 0 || totalBytes <= 0 || nLayers <= 0 || topK <= 0 || budgetBytes <= 0 {
		return 0, false
	}
	avgExpertBytes := totalBytes / int64(nExperts)
	if avgExpertBytes <= 0 {
		return 0, false
	}
	activeExpertsPerToken := int64(nLayers) * int64(topK)
	if activeExpertsPerToken <= 0 {
		return 0, false
	}
	activeBytesPerToken := activeExpertsPerToken*avgExpertBytes + denseCoreBytes
	residency := float64(budgetBytes) / float64(totalBytes)
	predicted := PredictedMoETokPerSec(activeBytesPerToken, MoEHitRatePrior(residency))
	if predicted <= 0 {
		return 0, false
	}
	return predicted, true
}

// MoEWorkingSetRefusal returns the load refusal when predicted falls strictly below floor without acceptSlow.
// predicted <= 0 ("unknown") never refuses: every unknown proceeds.
func MoEWorkingSetRefusal(modelName string, predicted, floor float64, acceptSlow bool) error {
	if predicted <= 0 || predicted >= floor || acceptSlow {
		return nil
	}
	return fmt.Errorf(
		"decoder: %s's paged-MoE working set predicts ~%.2f tok/s (a prior, not a measurement — "+
			"see moeHitRatePrior's own doc comment) below the %.1f tok/s floor. Pass --accept-slow "+
			"(decoder.Options.AcceptSlowMoE) to load it anyway, or raise -weight-cache to shrink "+
			"the miss rate this prediction assumes",
		modelName, predicted, floor)
}
