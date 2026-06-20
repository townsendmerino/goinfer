package decoder

// Sampled speculative decoding: lossless rejection sampling for a deterministic
// (point-mass) drafter. The draft proposes one token x with q(x)=1, so the
// Leviathan/Chen accept rule min(1, p(x)/q(x)) reduces to "accept x with
// probability p(x)"; on rejection the correction is drawn from the residual
// (p − q)+ = p with x removed and renormalized. The result reproduces the target
// sampler's distribution exactly (in-distribution lossless), where p is the
// sampler's *actual* draw distribution — temperature + top-k/top-p/min-p.

// distVector returns the sampler's normalized next-token distribution for one
// logit row as a full-vocab vector (0 outside the kept set, summing to 1) — the
// exact distribution Sample draws from. It is history-independent: it does NOT
// apply the repetition/presence/frequency penalties, LogitBias, or LogitProcessor
// (those depend on per-position history, which the speculative verify does not yet
// thread — genNgram rejects them on the sampled path). Greedy (temp ≤ 0) returns
// the argmax as a point mass.
func (s *Sampler) distVector(logits []float32) []float64 {
	if s.p.Temperature <= 0 {
		v := make([]float64, len(logits))
		v[argmax(logits)] = 1
		return v
	}
	probs := softmaxStable(logits, s.p.Temperature)
	if s.p.TopK <= 0 && s.p.TopP <= 0 && s.p.MinP <= 0 {
		return probs // drawFull draws from this directly
	}
	kept := topFilter(probs, s.p.TopK, s.p.TopP, s.p.MinP) // renormalized support
	v := make([]float64, len(probs))
	for _, ip := range kept {
		v[ip.id] = ip.p
	}
	return v
}

// specStep performs one lossless rejection-sampling step for a point-mass draft
// token x against target distribution p (a full normalized vector): accept x with
// probability p[x]; otherwise return a token drawn from the residual (p with x
// removed, renormalized). Returns (token, accepted). Consumes one rng draw to
// accept, plus one more to draw the correction on rejection.
func (s *Sampler) specStep(p []float64, x int) (int, bool) {
	var px float64
	if x >= 0 && x < len(p) {
		px = p[x]
	}
	if s.rng.Float64() < px {
		return x, true
	}
	return s.drawResidual(p, x), false
}

// drawResidual samples from (p − δ_x)+ renormalized — p with the mass at x removed
// and the rest scaled by 1/(1−p[x]). Used for the lossless correction when a draft
// token is rejected. When x carries no mass (p[x]==0, e.g. the drafter proposed a
// token the sampler filtered out) this is just a draw from p.
func (s *Sampler) drawResidual(p []float64, x int) int {
	var px float64
	if x >= 0 && x < len(p) {
		px = p[x]
	}
	denom := 1 - px
	if denom <= 0 { // x was the entire support — no residual mass (reject can't occur)
		return x
	}
	r := s.rng.Float64() * denom
	var cum float64
	for i, pi := range p {
		if i == x {
			continue
		}
		cum += pi
		if r < cum {
			return i
		}
	}
	for i := len(p) - 1; i >= 0; i-- { // float-rounding guard: last token with mass
		if i != x && p[i] > 0 {
			return i
		}
	}
	return x
}

// drawDist draws a token from a full normalized vector via the sampler rng (the
// seed token and the per-round bonus token; equivalent to a plain sampled draw).
func (s *Sampler) drawDist(p []float64) int { return s.drawFull(p) }

// blocksSpecSampling reports whether the sampler uses a per-position,
// history-dependent transform that the sampled speculative path cannot yet
// reproduce (penalties or logit bias). genNgram errors rather than silently
// breaking losslessness.
func (s *Sampler) blocksSpecSampling() bool {
	p := s.p
	repeat := p.RepeatPenalty > 0 && p.RepeatPenalty != 1
	return repeat || p.PresencePenalty != 0 || p.FrequencyPenalty != 0 || len(p.LogitBias) > 0
}
