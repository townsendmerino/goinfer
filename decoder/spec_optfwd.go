package decoder

// optFwdEligible reports whether sampled decode may attempt the optimistic-forward overlap:
// resident GPU decode only (the argmax guess needs a Forward call it can race against the real
// sampler; the staged CPU path has no such concurrency-safe primitive), Temperature>0 (T<=0
// already takes the fastGreedy on-device-argmax path, which needs no guess), no LogitProcessor
// (it may rewrite logits arbitrarily before the real pick, so an argmax of the UNPROCESSED
// logits is not a meaningful guess — same restriction fastGreedy already applies), and
// specRollbackSafe (a miss redoes Forward at the same position; recurrent/DeltaNet state and
// wrapped sliding-window rings can't be corrected that way — see forwardn.go).
func (m *Model) optFwdEligible(sp SamplingParams) bool {
	return m.resident != nil && sp.Temperature > 0 && sp.LogitProcessor == nil && m.specRollbackSafe()
}

// OptFwdStats accumulates optimistic-forward telemetry for one Generate run (mirrors SpecStats).
type OptFwdStats struct {
	Guessed int // steps where a guess was attempted (gate was on)
	Hit     int // steps where the sampler's real choice matched the guess
}

// HitRate is Hit/Guessed — the realized argmax-agreement rate this run measured.
func (s *OptFwdStats) HitRate() float64 {
	if s.Guessed == 0 {
		return 0
	}
	return float64(s.Hit) / float64(s.Guessed)
}

// optFwdGate is a binary (speculate / don't) trailing-hit-rate switch, the same EMA shape as
// AdaptiveDepth (spec_adaptive.go) but for an on/off decision rather than a continuous depth.
//
// The enable threshold is pinned to the WORST measured break-even across depth (90.9% at a
// shallow qwen2.5-coder-0.5b context, rounding down to 0.90) rather than a depth-aware curve:
// break-even only rises with depth (toward ~97% at 2048), so a fixed threshold at the shallow
// floor is never a net loss at any depth — it just leaves some upside on the table deep in
// context, where a depth-aware policy could re-enable more aggressively. Not worth the extra free
// parameter for v1 without a measured gpuPos->cost mapping on every backend.
type optFwdGate struct {
	EnableAt  float64 // trailing hit-rate at/above which speculation turns ON (default 0.90)
	DisableAt float64 // trailing hit-rate below which speculation turns OFF (default 0.75)
	Lambda    float64 // EMA retention (default 0.95)

	alpha  float64
	on     bool
	inited bool
}

// The dead band is wide (0.90/0.75), not the tight one a first cut assumed (0.90/0.85): an EMA
// close to a real, sustained 93% hit rate (the measured T=0.7 number, comfortably above the 90.9%
// worst-case break-even) still has enough sampling variance at Lambda=0.9's ~10-sample effective
// window to dip below a threshold just 3 points below the true mean, causing real flapping with no
// underlying regime change (caught by TestOptFwdGate_marginalRateHoldsSteady). Lambda=0.95 (~20
// samples) and a wide dead band down to 0.75 both push in the same direction: tolerate the natural
// noise around a genuinely-profitable rate, while still reliably catching a sustained drop toward
// the T=1.0-like 73% regime, where staying on would be a real loss.
func (g *optFwdGate) ensure() {
	if g.inited {
		return
	}
	if g.EnableAt <= 0 || g.EnableAt >= 1 {
		g.EnableAt = 0.90
	}
	if g.DisableAt <= 0 || g.DisableAt >= 1 {
		g.DisableAt = 0.75
	}
	if g.Lambda <= 0 || g.Lambda >= 1 {
		g.Lambda = 0.95
	}
	g.alpha = 0.95 // optimistic start: speculate, then adapt to the stream's real hit rate
	g.on = true
	g.inited = true
}

// Should reports whether this step should attempt the optimistic guess.
func (g *optFwdGate) Should() bool {
	g.ensure()
	return g.on
}

// Observe folds one step's hit/miss outcome into the trailing estimate and updates on/off with
// hysteresis (a wide dead band between EnableAt and DisableAt, see the struct comment) so ordinary
// EMA sampling noise around a genuinely-profitable rate can't flip the gate; only a sustained shift
// toward an actually-unprofitable regime does.
func (g *optFwdGate) Observe(hit bool) {
	g.ensure()
	v := 0.0
	if hit {
		v = 1.0
	}
	g.alpha = g.Lambda*g.alpha + (1-g.Lambda)*v
	switch {
	case g.alpha >= g.EnableAt:
		g.on = true
	case g.alpha < g.DisableAt:
		g.on = false
	}
}

// Alpha is the current trailing hit-rate estimate (for telemetry).
func (g *optFwdGate) Alpha() float64 { g.ensure(); return g.alpha }

// optFwdResult is what optFwdStep resolves to: the sampled token/info (identical to what plain
// sampler.SampleWithInfo would have returned -- ONLY scheduling differs), and the logits for the
// NEXT decode position, already available without a further Forward call on a hit.
type optFwdResult struct {
	info       SampleInfo
	nextLogits []float32
}

// optFwdStep runs one sampled-decode position's optimistic-forward overlap. logits are the
// ALREADY-HOST-SIDE logits for gpuPos (from the previous Forward/prefill); the caller has NOT
// sampled them yet. This both samples (identically to plain sampler.SampleWithInfo) and resolves
// the next position's logits, replacing what would otherwise be a separate SampleWithInfo call
// (at the top of the decode loop) plus a separate Forward call (at the bottom) with one combined,
// overlapped step.
//
// Hit: the speculative Forward(guess, gpuPos) already wrote the correct KV entry at gpuPos and
// its returned logits ARE the correct next-position logits -- reused directly, no redo.
// Miss: the speculative write at gpuPos was wrong (guess != real choice); Forward is called again
// at the IDENTICAL gpuPos with the correct token's embedding, overwriting it (both backends'
// resident KV is positional -- a second Forward at the same pos replaces the first; TruncateTo is
// a no-op on both because of exactly this, see cuda/resident.go and metal/backend.go). The emitted
// token stream and its logprobs are bit-identical to the non-speculative path either way -- only
// which goroutine computed which logits, and in what order, differs.
func (m *Model) optFwdStep(sampler *Sampler, logits []float32, gpuPos int, gate *optFwdGate, stats *OptFwdStats) (optFwdResult, error) {
	guess := argmax(logits)
	type fres struct {
		logits []float32
		err    error
	}
	ch := make(chan fres, 1)
	go func() {
		l, err := m.resident.Forward(m.embedResident(guess), gpuPos)
		ch <- fres{l, err}
	}()
	info, serr := sampler.SampleWithInfo(logits)
	spec := <-ch
	if serr != nil {
		return optFwdResult{}, serr
	}
	stats.Guessed++
	hit := info.ID == guess
	gate.Observe(hit)
	if hit {
		stats.Hit++
		if spec.err != nil {
			return optFwdResult{}, spec.err
		}
		return optFwdResult{info: info, nextLogits: spec.logits}, nil
	}
	// Miss: the speculative forward's result is discarded regardless of whether IT errored --
	// only the redo (at the real token) matters from here.
	nextLogits, err := m.resident.Forward(m.embedResident(info.ID), gpuPos)
	if err != nil {
		return optFwdResult{}, err
	}
	return optFwdResult{info: info, nextLogits: nextLogits}, nil
}
