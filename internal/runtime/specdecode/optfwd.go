package specdecode

import "github.com/townsendmerino/goinfer/internal/runtime/sampling"

// OptFwdMaxTemp is the temperature at or below which the optimistic-forward overlap may run.
const OptFwdMaxTemp = 0.2

// OptFwdStats accumulates optimistic-forward telemetry for one Generate run.
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

// OptFwdGate is a binary (speculate / don't) trailing-hit-rate switch.
type OptFwdGate struct {
	EnableAt  float64 // trailing hit-rate at/above which speculation turns ON (default 0.90)
	DisableAt float64 // trailing hit-rate below which speculation turns OFF (default 0.75)
	Lambda    float64 // EMA retention (default 0.95)

	alpha   float64
	on      bool
	inited  bool
	scratch []float32
}

func (g *OptFwdGate) ensure() {
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
	g.alpha = 0.95
	g.on = true
	g.inited = true
}

// Should reports whether this step should attempt the optimistic guess.
func (g *OptFwdGate) Should() bool {
	g.ensure()
	return g.on
}

// Observe folds one step's hit/miss outcome into the trailing estimate with hysteresis.
func (g *OptFwdGate) Observe(hit bool) {
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
func (g *OptFwdGate) Alpha() float64 { g.ensure(); return g.alpha }

// OptFwdResult is what OptFwdStep resolves to: the sampled token/info and the next position's logits.
type OptFwdResult[T any] struct {
	Info       T
	NextLogits []float32
}

// OptFwdStep runs one sampled-decode position's optimistic-forward overlap.
func OptFwdStep[T any](forward func(tok, pos int) ([]float32, error), sample func([]float32) (T, error), idOf func(T) int, logits []float32, gpuPos int, gate *OptFwdGate, stats *OptFwdStats) (OptFwdResult[T], error) {
	guess := sampling.Argmax(logits)

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
		l, err := forward(guess, gpuPos)
		ch <- fres{l, err}
	}()
	info, serr := sample(logits)
	spec := <-ch
	if serr != nil {
		return OptFwdResult[T]{}, serr
	}
	stats.Guessed++
	realID := idOf(info)
	hit := realID == guess
	gate.Observe(hit)
	if hit {
		stats.Hit++
		if spec.err != nil {
			return OptFwdResult[T]{}, spec.err
		}
		return OptFwdResult[T]{Info: info, NextLogits: spec.logits}, nil
	}
	nextLogits, err := forward(realID, gpuPos)
	if err != nil {
		return OptFwdResult[T]{}, err
	}
	return OptFwdResult[T]{Info: info, NextLogits: nextLogits}, nil
}
