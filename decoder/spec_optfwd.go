package decoder

import (
	"os"
	"strconv"
)

// optFwdMaxTemp is the temperature at or below which the optimistic-forward overlap is allowed to
// run. ABOVE IT THE FEATURE IS A MEASURED LOSS, and it used to run unconditionally.
//
// WHY A FIXED THRESHOLD, AND WHY 0.2. The overlap can only pay back the sampler cost it hides, and
// its hit rate falls with temperature. Both halves are model-dependent, so the break-even
// temperature is too: MEASURED 2026-08-27 at T ~ 0.26 on phi3-mini (vocab 32064) and T ~ 0.95 on
// qwen2.5-coder-1.5B (151936). 0.2 sits below the LOWER of the two, which is the only safe place
// for a single constant: at the higher crossover phi3-mini pays 2.8-6.8%.
//
// WHAT THIS COSTS, RECORDED SO IT IS NOT REDISCOVERED AS A BUG. On large-vocab models the overlap
// still wins between 0.2 and their own crossover — 6.0% at T=0.4 and 5.1% at T=0.6 on the 1.5B —
// and this threshold forfeits that. An adaptive per-model gate was designed to recover it
// (docs/spec/10-optfwd-gate.md) and DELIBERATELY NOT BUILT: its whole value was those two cells,
// resting on a model-dependence generalised from two models, which is the same shape of error that
// shipped this feature unconditionally in the first place. Revisit if a third model's crossover
// lands somewhere this gate gets badly wrong; 10's pre-registered bar stands.
//
// TRUNCATED SAMPLING IS UNMEASURED. The ladders were temperature-only with no truncation. top_k /
// top_p cut the candidate set, which should RAISE the hit rate and push the crossover up, so this
// gate is probably conservative there — forfeiting a possible win rather than taking a measured
// loss, which is the correct direction to be wrong in until it is measured.
//
// GOINFER_OPTFWD_MAX_TEMP overrides it, for MEASUREMENT rather than tuning: moving this number
// without a ladder behind it is how the original default happened.
const optFwdMaxTemp = 0.2

func optFwdTempCap() float64 {
	if v := os.Getenv("GOINFER_OPTFWD_MAX_TEMP"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return optFwdMaxTemp
}

// optFwdEligible reports whether sampled decode may attempt the optimistic-forward overlap:
// resident GPU decode only (the argmax guess needs a Forward call it can race against the real
// sampler; the staged CPU path has no such concurrency-safe primitive), Temperature>0 (T<=0
// already takes the fastGreedy on-device-argmax path, which needs no guess), no LogitProcessor
// (it may rewrite logits arbitrarily before the real pick, so an argmax of the UNPROCESSED
// logits is not a meaningful guess — same restriction fastGreedy already applies), and
// specRollbackSafe (a miss redoes Forward at the same position; recurrent/DeltaNet state and
// wrapped sliding-window rings can't be corrected that way — see forwardn.go).
func (m *Model) optFwdEligible(sp SamplingParams) bool {
	return m.resident != nil && sp.Temperature > 0 && sp.Temperature <= optFwdTempCap() &&
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

// optFwdGate is a binary (speculate / don't) trailing-hit-rate switch, the same EMA shape as
// AdaptiveDepth (spec_adaptive.go) but for an on/off decision rather than a continuous depth.
//
// THESE THRESHOLDS ARE FITTED ON ONE MODEL AND ARE WRONG FOR SMALL-VOCAB ONES (G27). Break-even
// hit rate is not a constant: it rises as the sampler share falls, because the overlap can only pay
// back the sampler it hides. The 90.9% below was measured on qwen2.5-coder-0.5b — 152k vocab, a
// LARGE sampler share. On phi3-mini (32k, 5.4% share against the 1.5B's 18.2%) the true break-even
// sits ABOVE 0.90, so the entire 0.90/0.75 dead band lies below it and this gate cannot turn off in
// the regime where the feature loses 2.8-6.8%. That is why the loss needed a temperature cap
// (optFwdMaxTemp) rather than being caught here.
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

	// scratch holds a COPY of the current position's logits for the duration of the overlap. It
	// lives here only because optFwdGate is the one per-Generate value optFwdStep already receives;
	// see the copy in optFwdStep for why it is needed at all.
	scratch []float32
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
//
// THE HYSTERESIS ABOVE IS A ONE-WAY LATCH IN PRACTICE, AND THE DEAD BAND'S UPPER HALF IS DEAD CODE
// (G27). Observe is called ONLY from optFwdStep, and the caller in model.go invokes optFwdStep only
// when Should() is true. So the moment alpha falls below DisableAt and this returns false, no
// further outcomes are ever observed: alpha freezes, and the `alpha >= EnableAt` re-enable branch in
// Observe cannot be reached again for the rest of that Generate. It reads as a two-way band and
// behaves as a latch.
//
// TestOptFwdGate_hysteresis DOES NOT CATCH THIS and actively vouches for the two-way reading,
// because it drives Observe in an unconditional loop — a calling convention production never uses.
// It is correct about the component and blind to the composition.
//
// Left as-is deliberately: since T > 0.2 no longer reaches the overlap at all (optFwdMaxTemp), the
// gate barely runs and the latch is unreachable in the regime that mattered. Fix it together with
// the thresholds, not before — see G27 in docs/QUEUE.md.
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

	// COPY THE LOGITS BEFORE STARTING THE OVERLAP. This is not defensive tidiness; without it the
	// feature is silently wrong on CUDA.
	//
	// A resident backend may legitimately return a slice that ALIASES its own reusable host buffer,
	// valid only until the next Forward — cuda/resident.go returns r.logitsHost, "a zero-copy view
	// of logitsPinned". The overlap then has the speculative Forward's device->host DMA writing that
	// exact buffer while SampleWithInfo is reading it, so the sampler sees a torn mixture of this
	// position's logits and the next one's.
	//
	// MEASURED on the CUDA box before this copy existed: same token id, logprob -2.6463 vs -2.6266
	// (feature on vs GOINFER_NO_OPTFWD=1) — and under `-race`, where the timing shifts, the emitted
	// TOKEN STREAM diverged outright. Severity tracking timing is what identified it as a race
	// rather than an arithmetic difference.
	//
	// `go test -race` CANNOT SEE THIS and reported nothing: the write is a driver DMA into pinned
	// memory, not a Go memory access, so the detector is structurally blind to it. That is the
	// reason this comment is long — the next person to touch the overlap will not get a warning.
	//
	// Metal was unaffected because its Forward hands back a per-call slice, which is why the feature
	// verified clean there. The copy fixes every backend, including ones that do not exist yet, and
	// costs one vocab-sized memcpy against a full forward.
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
