package decoder

import ()

// optFwdMaxTemp is the temperature at or below which the optimistic-forward overlap may run; above it the overlap is a
// measured loss. The overlap can only pay back the sampler cost it hides, and its hit rate falls with temperature. Both
// are model-dependent, so the break-even temperature is too, and 0.2 sits below the lowest measured crossover, the
// only safe place for a single constant.
//
// Known cost: on large-vocab models the overlap still wins between 0.2 and their own crossover, and this threshold
// forfeits that. An adaptive per-model gate (docs/spec/10-optfwd-gate.md) was designed and deliberately not built.
// Truncated sampling (top-k, top-p) is unmeasured: it should raise the hit rate and push the crossover up, so the
// gate is probably conservative there, the correct direction to be wrong in until it is measured.
//
// GOINFER_OPTFWD_MAX_TEMP overrides it, for measurement rather than tuning: moving this number without a ladder behind
// it is how the original unconditional default happened. The ladders and the case for not building the gate:
// docs/code-notes/decoder.md#optFwdMaxTemp.
const optFwdMaxTemp = 0.2

// optFwdEligible reports whether sampled decode may attempt the optimistic-forward overlap:
// resident GPU decode only (the argmax guess needs a Forward call it can race against the real
// sampler; the staged CPU path has no such concurrency-safe primitive), Temperature>0 (T<=0
// already takes the fastGreedy on-device-argmax path, which needs no guess), no LogitProcessor
// (it may rewrite logits arbitrarily before the real pick, so an argmax of the UNPROCESSED
// logits is not a meaningful guess — same restriction fastGreedy already applies), and
// specRollbackSafe (a miss redoes Forward at the same position; recurrent/DeltaNet state and
// wrapped sliding-window rings can't be corrected that way — see forwardn.go).
func (m *Model) optFwdEligible(sp SamplingParams) bool {
	return m.resident != nil && sp.Temperature > 0 && sp.Temperature <= m.knobs.optFwdTempCap() &&
		sp.LogitProcessor == nil && m.specRollbackSafe()
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

// optFwdGate is a binary (speculate / don't) trailing-hit-rate switch, the same EMA shape as AdaptiveDepth
// (spec_adaptive.go) but for an on/off decision rather than a continuous depth.
//
// Its thresholds are fitted on one model and are wrong for small-vocab ones (docs/QUEUE.md G27). Break-even hit rate
// rises as the sampler's share of a step falls, because the overlap can only pay back the sampler it hides. EnableAt
// is pinned to the worst break-even measured on a large-vocab model at a shallow context; on a small-vocab model the
// true break-even sits above it, the whole 0.90/0.75 dead band lies below it, and the gate cannot turn off where the
// feature loses. That is why the loss needed a temperature cap (optFwdMaxTemp) rather than being caught here. A fixed
// threshold at the shallow floor is never a net loss at any depth (break-even only rises with depth) but leaves upside
// deep in context; a depth-aware curve needs a measured gpuPos-to-cost mapping on every backend first.
type optFwdGate struct {
	EnableAt  float64 // trailing hit-rate at/above which speculation turns ON (default 0.90)
	DisableAt float64 // trailing hit-rate below which speculation turns OFF (default 0.75)
	Lambda    float64 // EMA retention (default 0.95)

	alpha  float64
	on     bool
	inited bool

	// scratch holds a COPY of the current position's logits for the duration of the overlap. It
	// lives here only because optFwdGate is the one per-Generate value optFwdStep already receives;
	// see the copy in optFwdStep for why it is needed at all.
	scratch []float32
}

// ensure applies the defaults: a wide dead band (EnableAt 0.90, DisableAt 0.75) and Lambda 0.95 (about 20 samples).
// A tight band flaps: an EMA near a real sustained hit rate still has enough sampling variance over a shorter window
// to dip below a threshold a few points under the true mean, with no regime change
// (TestOptFwdGate_marginalRateHoldsSteady). The wide band and the longer window tolerate that noise while still
// catching a sustained drop toward the regime where staying on loses.
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
//
// Limitation: the hysteresis is a one-way latch in practice, and the dead band's upper half is unreachable. Observe is
// called only from optFwdStep, which the caller in model.go invokes only when Should() is true, so once alpha falls
// below DisableAt nothing is observed again and the re-enable branch in Observe cannot run for the rest of that
// Generate. TestOptFwdGate_hysteresis drives Observe unconditionally, a calling convention production never uses, so
// it does not pin this. Left as is deliberately: fix it together with the thresholds (docs/QUEUE.md G27).
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

// optFwdStep runs one sampled-decode position's optimistic-forward overlap. logits are the host-side logits for gpuPos
// (from the previous Forward or prefill); the caller has not sampled them yet. It samples (identically to plain
// sampler.SampleWithInfo) and resolves the next position's logits, replacing a separate SampleWithInfo plus a
// separate Forward with one overlapped step.
//
// Hit: the speculative Forward(guess, gpuPos) already wrote the correct KV entry at gpuPos and its logits are the
// correct next-position logits, reused directly. Miss: Forward is called again at the identical gpuPos with the
// correct token's embedding, overwriting the entry (resident KV is positional, so a second Forward at the same
// position replaces the first, and TruncateTo is a no-op on both backends for the same reason; cuda/resident.go,
// metal/backend.go). The emitted tokens and logprobs are bit-identical to the non-speculative path either way; only
// which goroutine computed which logits differs.
func (m *Model) optFwdStep(sampler *Sampler, logits []float32, gpuPos int, gate *optFwdGate, stats *OptFwdStats) (optFwdResult, error) {
	guess := argmax(logits)

	// Copy the logits before starting the overlap. Without it the feature is silently wrong on CUDA: a resident backend
	// may return a slice that aliases its own reusable host buffer, valid only until the next Forward (cuda/resident.go
	// returns r.logitsHost, a zero-copy view of logitsPinned), so the speculative Forward's device-to-host DMA writes that
	// buffer while SampleWithInfo reads it and the sampler sees a torn mixture of this position's logits and the next
	// one's. `go test -race` cannot see this: the write is a driver DMA into pinned memory, not a Go memory access. Metal
	// returns a per-call slice and was unaffected; the copy covers every backend, including future ones, for one
	// vocab-sized memcpy per forward. Evidence: docs/code-notes/decoder.md#Model.optFwdStep.copy.
	if cap(gate.scratch) < len(logits) {
		gate.scratch = make([]float32, len(logits))
	}
	gate.scratch = gate.scratch[:len(logits)]
	copy(gate.scratch, logits)
	logits = gate.scratch

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
