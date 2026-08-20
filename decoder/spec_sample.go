package decoder

import "slices"

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
	return s.distVectorFrom(logits)
}

// distVectorFrom is distVector's temperature + top-k/p/min-p step over logits that
// have ALREADY had any history-dependent transforms (bias/penalties) applied.
func (s *Sampler) distVectorFrom(logits []float32) []float64 {
	if s.p.Temperature <= 0 {
		v := make([]float64, len(logits))
		v[argmax(logits)] = 1
		return v
	}
	if s.p.TopK <= 0 && s.p.TopP <= 0 && s.p.MinP <= 0 {
		return softmaxStable(logits, s.p.Temperature) // drawFull draws from this directly
	}
	// Same canonical selection the plain sampler uses (topFilterLogits), so the
	// speculative residual stays lossless against the target distribution.
	kept := topFilterLogits(logits, s.p.Temperature, s.p.TopK, s.p.TopP, s.p.MinP, s.vocabBufN(len(logits))) // renormalized support
	v := make([]float64, len(logits))
	for _, ip := range kept {
		v[ip.id] = ip.p
	}
	return v
}

// distVectorHist is distVector with the history-dependent transforms (LogitBias +
// repetition/presence/frequency penalties) applied first, over the explicit token
// history up to — but NOT including — this position. This is the per-position
// context the speculative verify threads so sampled decoding stays lossless even
// with penalties: it reproduces exactly the distribution the plain sampler would
// draw from at that position (where its history is prompt + all tokens emitted so
// far). Falls back to the cheap distVector when no such transform is configured.
func (s *Sampler) distVectorHist(logits []float32, history []int) []float64 {
	if !s.needsHistory() {
		return s.distVectorFrom(logits)
	}
	work := slices.Clone(logits)
	s.applyLogitBias(work)
	if s.penaltiesConfigured() {
		s.applyPenaltiesOver(work, penaltyWindowOf(history, s.p.RepeatLastN))
	}
	return s.distVectorFrom(work)
}

// needsHistory reports whether the sampling distribution depends on the token
// history (penalties or logit bias) — i.e. whether the speculative path must
// thread a per-position history rather than use the plain distVector.
func (s *Sampler) needsHistory() bool {
	return len(s.p.LogitBias) > 0 || s.penaltiesConfigured()
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
	for i, v := range slices.Backward(p) { // float-rounding guard: last token with mass
		if i != x && v > 0 {
			return i
		}
	}
	return x
}

// drawDist draws a token from a full normalized vector via the sampler rng (the
// seed token and the per-round bonus token; equivalent to a plain sampled draw).
func (s *Sampler) drawDist(p []float64) int { return s.drawFull(p) }
