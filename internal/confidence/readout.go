// Package confidence is the shared readout behind goinfer's confidence surfaces
// (docs/tasks/task-constrained-confidence.md): a softmax over a RESTRICTED set of tokens at a
// position a grammar or a template controls, renormalized over that set and optionally
// temperature-scaled. Per-field confidence on constrained output (C) reads it over the tokens a
// grammar left legal; decisions (D) read it over an option set's label tokens.
//
// A number this package returns is the model's probability over what was allowed, which is
// narrower than the probability that an answer is correct. It is not a calibrated confidence
// unless a fitted temperature says so (the task doc's §2).
package confidence

import "math"

// LogSumExpFinite returns log Σ exp(logits[i]) over the finite entries of logits, and how many
// there are. A constrained decoder masks illegal tokens to −∞, so this is the normalizer of the
// model's distribution restricted to the legal set: p(id) = exp(logits[id] − lse) for a legal
// id. With no finite entry it returns (−∞, 0).
func LogSumExpFinite(logits []float32) (lse float64, legal int) {
	maxv := math.Inf(-1)
	for _, v := range logits {
		if f := float64(v); f > maxv && !math.IsInf(f, 0) && !math.IsNaN(f) {
			maxv = f
		}
	}
	if math.IsInf(maxv, -1) {
		return maxv, 0
	}
	var sum float64
	for _, v := range logits {
		f := float64(v)
		if math.IsInf(f, 0) || math.IsNaN(f) {
			continue
		}
		sum += math.Exp(f - maxv)
		legal++
	}
	return maxv + math.Log(sum), legal
}

// RestrictedLogSoftmax returns the log-probabilities of ids under a softmax taken over ids alone
// (the rest of the vocabulary is ignored), after dividing their logits by temperature
// (temperature ≤ 0 is treated as 1). The result is in ids' order and sums to 1 in probability
// space. An id out of range, or whose logit is not finite, gets −∞ and takes no mass; if no id
// has a finite logit every entry is −∞.
func RestrictedLogSoftmax(logits []float32, ids []int, temperature float64) []float64 {
	if temperature <= 0 {
		temperature = 1
	}
	out := make([]float64, len(ids))
	maxv := math.Inf(-1)
	for i, id := range ids {
		out[i] = math.Inf(-1)
		if id < 0 || id >= len(logits) {
			continue
		}
		f := float64(logits[id])
		if math.IsInf(f, 0) || math.IsNaN(f) {
			continue
		}
		out[i] = f / temperature
		if out[i] > maxv {
			maxv = out[i]
		}
	}
	if math.IsInf(maxv, -1) {
		return out
	}
	var sum float64
	for _, v := range out {
		if !math.IsInf(v, -1) {
			sum += math.Exp(v - maxv)
		}
	}
	lse := maxv + math.Log(sum)
	for i, v := range out {
		if !math.IsInf(v, -1) {
			out[i] = v - lse
		}
	}
	return out
}
