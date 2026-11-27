package specdecode

import (
	"context"
	"slices"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/internal/runtime/sampling"
)

// ConfidentDrafter reports a confidence for its most recent Draft.
type ConfidentDrafter interface {
	Drafter
	Confidence() float64
}

// OutcomeRecorder is an optional Drafter capability receiving per-round acceptance outcomes.
type OutcomeRecorder interface {
	RecordOutcome(accepted, drafted int)
}

// RouterDrafter fuses drafters by polling each source and selecting the highest effective confidence.
type RouterDrafter struct {
	Sources []Drafter
	Prior   float64 // shrinkage strength: pseudo-rounds of the static α̂ (0 ⇒ routerPrior)

	rate   []float64
	weight []float64
	chosen int
}

// RouterChosen returns the index of the source chosen by the most recent Draft (used by unit tests).
func RouterChosen(r *RouterDrafter) int { return r.chosen }

// RouterEffective returns r's blended confidence score for source i (used by unit tests).
func RouterEffective(r *RouterDrafter, i int, static float64) float64 { return r.effective(i, static) }

const (
	routerPrior  = 4.0
	routerLambda = 0.85
	routerWMax   = 20.0
)

func (r *RouterDrafter) ensure() {
	if r.rate == nil {
		r.rate = make([]float64, len(r.Sources))
		r.weight = make([]float64, len(r.Sources))
	}
}

func (r *RouterDrafter) effective(i int, static float64) float64 {
	w := r.weight[i]
	if w == 0 {
		return static
	}
	prior := r.Prior
	if prior <= 0 {
		prior = routerPrior
	}
	return (w*r.rate[i] + prior*static) / (w + prior)
}

// Draft returns the highest-effective-confidence source's non-empty proposal.
func (r *RouterDrafter) Draft(ctx []int, k int) []int {
	r.ensure()
	var best []int
	bestEff := -1.0
	r.chosen = -1
	for i, s := range r.Sources {
		d := s.Draft(ctx, k)
		if len(d) == 0 {
			continue
		}
		static := 0.0
		if cd, ok := s.(ConfidentDrafter); ok {
			static = cd.Confidence()
		}
		if eff := r.effective(i, static); eff > bestEff {
			best, bestEff, r.chosen = d, eff, i
		}
	}
	return best
}

// RecordOutcome folds the last round's realized acceptance into the chosen source's running rate.
func (r *RouterDrafter) RecordOutcome(accepted, drafted int) {
	if r.chosen < 0 || drafted == 0 {
		return
	}
	r.ensure()
	frac := float64(accepted) / float64(drafted)
	i := r.chosen
	r.rate[i] = routerLambda*r.rate[i] + (1-routerLambda)*frac
	if r.weight[i] < routerWMax {
		r.weight[i]++
	}
}

// GrammarDrafter proposes the grammar's forced byte-run as draft tokens.
type GrammarDrafter struct {
	Mask     *constrain.Masker  // the live grammar mask (advanced over committed tokens)
	Encode   func(string) []int // tokenizer: forced bytes → canonical token ids
	MaxBytes int                // cap on the forced byte-run probed per round (default 64)

	lastForced int
}

// GrammarConf is the calibrated acceptance probability of a forced grammar proposal.
const GrammarConf = 0.20

// Confidence reports the calibrated acceptance probability when the grammar forced bytes.
func (d *GrammarDrafter) Confidence() float64 {
	if d.lastForced > 0 {
		return GrammarConf
	}
	return 0
}

// Draft returns the canonical tokenization of the grammar's forced byte-run, capped at k tokens.
func (d *GrammarDrafter) Draft(ctx []int, k int) []int {
	mb := d.MaxBytes
	if mb <= 0 {
		mb = 64
	}
	bytes := d.Mask.ForcedBytesRun(mb)
	d.lastForced = len(bytes)
	if len(bytes) == 0 {
		return nil
	}
	toks := d.Encode(string(bytes))
	if len(toks) > k {
		toks = toks[:k]
	}
	return toks
}

// GrammarSpecLoop runs grammar-constrained speculation over fw, which already holds prompt[:prefillFrom].
func GrammarSpecLoop(ctx context.Context, fw Forward, isStop func(int) bool, out chan<- int, errOut *error, stats *SpecStats, mask *constrain.Masker, drafter Drafter, prompt []int, prefillFrom, maxTokens, K int, tr Tracer, commit func(int)) {
	traceSource, traceForced := "router", false
	if _, ok := drafter.(*GrammarDrafter); ok {
		traceSource, traceForced = "grammar", true
	}
	tpos := len(prompt)

	seedLogits, err := fw.Prefill(ctx, prompt[prefillFrom:])
	if err != nil {
		*errOut = err
		return
	}
	mask.MaskAt(mask.GrammarClone(), seedLogits)
	cur := sampling.Argmax(seedLogits)

	emit := func(tok int) bool {
		if isStop(tok) {
			return false
		}
		select {
		case <-ctx.Done():
			*errOut = ctx.Err()
			return false
		case out <- tok:
		}
		stats.Emitted++
		return stats.Emitted < maxTokens
	}
	finishTrailing := func(tok int) {
		if *errOut != nil || stats.Emitted < maxTokens || isStop(tok) {
			return
		}
		if _, err := fw.Verify(ctx, []int{tok}); err != nil {
			return
		}
		tpos++
		if commit != nil {
			commit(tok)
		}
	}
	if !emit(cur) {
		finishTrailing(cur)
		return
	}

	hist := slices.Clone(prompt)
	var lookupBuf []int
	var seqBuf []int
	for {
		mask.Commit(cur)
		lookupBuf = append(lookupBuf[:0], hist...)
		lookupBuf = append(lookupBuf, cur)
		draftTok := drafter.Draft(lookupBuf, K)
		kEff := len(draftTok)

		base := tpos
		seqBuf = append(seqBuf[:0], cur)
		seqBuf = append(seqBuf, draftTok...)
		logitsN, err := fw.Verify(ctx, seqBuf)
		if err != nil {
			*errOut = err
			return
		}

		stats.Rounds++
		stats.Drafted += kEff
		gc := mask.GrammarClone()
		accepted := 0
		allAccept := true
		var nextTok int
		for i := range kEff {
			mask.MaskAt(gc, logitsN[i])
			ti := sampling.Argmax(logitsN[i])
			acc := draftTok[i] == ti
			if tr != nil {
				pTop1, pEnt, pTok := TargetDist(logitsN[i], draftTok[i])
				tr(SpecTrace{
					Step: stats.Rounds, Pos: i, Source: traceSource, Forced: traceForced, Token: draftTok[i],
					Streak: i, QTop1: 1, PTop1: pTop1, PEntropy: pEnt, TV: 1 - pTok, AcceptProb: pTok, Accepted: acc,
				})
			}
			if !acc {
				nextTok = ti
				allAccept = false
				break
			}
			accepted++
			gc.Commit(mask.TokenBytes(draftTok[i]))
		}
		if allAccept {
			mask.MaskAt(gc, logitsN[kEff])
			nextTok = sampling.Argmax(logitsN[kEff])
		}
		stats.Accepted += accepted
		evaluated := accepted
		if !allAccept {
			evaluated = accepted + 1
		}
		stats.Evaluated += evaluated
		if rec, ok := drafter.(OutcomeRecorder); ok && kEff > 0 {
			rec.RecordOutcome(accepted, kEff)
		}

		tpos = base + 1 + accepted
		fw.TruncateTo(tpos)
		hist = append(hist, cur)
		if commit != nil {
			commit(cur)
		}
		for i := 0; i < accepted; i++ {
			mask.Commit(draftTok[i])
			hist = append(hist, draftTok[i])
			if commit != nil {
				commit(draftTok[i])
			}
		}
		for i := 0; i < accepted; i++ {
			if !emit(draftTok[i]) {
				return
			}
		}
		cur = nextTok
		if !emit(cur) {
			finishTrailing(cur)
			return
		}
	}
}
