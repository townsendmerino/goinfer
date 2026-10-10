package decoder

import (
	"context"
	"fmt"
	"slices"

	"github.com/townsendmerino/goinfer/constrain"
)

// ConfidentDrafter reports a confidence for its most recent Draft, so RouterDrafter can order sources by predicted
// acceptance per position rather than a fixed priority (docs/spec/03-router-tree.md). Higher is more likely accepted;
// implementations return an acceptance probability in [0,1] on one scale (see ngramAlpha).
type ConfidentDrafter interface {
	Drafter
	Confidence() float64
}

// OutcomeRecorder is an optional Drafter capability: the verify loop reports how the last Draft's proposal fared
// (accepted of drafted) so a router can keep a running per-source accept rate, which corrects a static estimate when
// the live workload drifts from the mix it was calibrated on.
type OutcomeRecorder interface {
	RecordOutcome(accepted, drafted int)
}

// RouterDrafter fuses drafters: each round it polls every source and returns the proposal of the one with the highest
// effective confidence, the source's static estimate (Confidence) shrunk toward its running realized accept rate as
// outcomes accumulate. The static estimate is fit on a fixed workload mix, so on a drifting workload it can mis-rank
// (ngram's, fit on copy-heavy traffic, over-trusts a spurious short match in prose); the running rate corrects that
// within a few rounds. Sources without a Confidence score rank 0; ties keep source order. On a constrained request all
// ride the same masked verify, so the choice affects only speed, never correctness.
type RouterDrafter struct {
	Sources []Drafter
	Prior   float64 // shrinkage strength: pseudo-rounds of the static α̂ (0 ⇒ routerPrior)

	rate   []float64 // per-source EMA of the realized per-round accept fraction
	weight []float64 // per-source observation weight (capped), ramps trust in `rate`
	chosen int       // source index returned by the last Draft (-1 = none)
}

const (
	routerPrior  = 4.0  // static α̂ counts as this many rounds before the running rate leads
	routerLambda = 0.85 // accept-rate EMA decay (~7-round memory) → tracks recent drift
	routerWMax   = 20.0 // cap on observation weight so the prior never fully washes out
)

func (r *RouterDrafter) ensure() {
	if r.rate == nil {
		r.rate = make([]float64, len(r.Sources))
		r.weight = make([]float64, len(r.Sources))
	}
}

// effective blends the source's static confidence with its running accept rate: a
// Bayesian shrinkage toward α̂ that hands over to the observed rate as weight grows.
func (r *RouterDrafter) effective(i int, static float64) float64 {
	w := r.weight[i]
	if w == 0 {
		return static // no outcomes yet ⇒ pure static α̂ (byte-identical to the old router)
	}
	prior := r.Prior
	if prior <= 0 {
		prior = routerPrior
	}
	return (w*r.rate[i] + prior*static) / (w + prior)
}

// Draft returns the highest-effective-confidence source's non-empty proposal, recording
// which source was chosen so RecordOutcome can attribute the round's outcome to it.
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

// RecordOutcome folds the last round's realized acceptance (accepted of drafted, from
// the chosen source's proposal) into that source's running rate (§06 §9 online correction).
func (r *RouterDrafter) RecordOutcome(accepted, drafted int) {
	if r.chosen < 0 || drafted == 0 {
		return
	}
	r.ensure()
	frac := float64(accepted) / float64(drafted)
	i := r.chosen
	r.rate[i] = routerLambda*r.rate[i] + (1-routerLambda)*frac // EMA: tracks recent acceptance
	if r.weight[i] < routerWMax {                              // weight = capped observation count
		r.weight[i]++
	}
}

// GrammarDrafter proposes the grammar's forced byte-run as draft tokens (docs/spec/01-grammar-fused.md). At positions
// where the grammar determines the bytes (inside object keys, enum and const values), it extracts that byte run from
// the masker's current state and retokenizes it canonically: the tokens the model most likely emits there. The
// verifier confirms under the same mask, so it is lossless; acceptance is below 1 only where the model tokenizes the
// forced bytes differently than Encode.
type GrammarDrafter struct {
	Mask     *constrain.Masker  // the live grammar mask (advanced over committed tokens)
	Encode   func(string) []int // tokenizer: forced bytes → canonical token ids
	MaxBytes int                // cap on the forced byte-run probed per round (default 64)

	lastForced int // bytes in the most recent forced run (0 = abstained); for Confidence
}

// grammarConf is the acceptance probability of a forced grammar proposal, on the same scale as ngramAlpha so the router
// compares sources like for like. It is a trace fit (TestGrammarAlphaPredictor) and it is low, about 0.20: the
// drafter's canonical retokenization of the forced bytes usually differs from how the model tokenizes the same bytes
// under the mask, and the mismatch compounds along a run. Read it as a verdict on this drafter, not on grammar
// speculation in principle: free grammar drafting must guess the tokenization, and a tokenizer-aligned forced drafter
// (unbuilt; docs/spec/01-grammar-fused.md) is the headroom. So grammar ranks last as currently built: any n-gram copy
// (ngramAlpha >= 0.70) outranks it, and the router uses it as the floor, since a miss costs almost nothing. Re-fit per
// model and tokenizer.
const grammarConf = 0.20

// Confidence reports the calibrated acceptance probability α̂_grammar when the grammar
// forced something this Draft, 0 when it abstained (the router skips empty proposals).
func (d *GrammarDrafter) Confidence() float64 {
	if d.lastForced > 0 {
		return grammarConf
	}
	return 0
}

// Draft returns the canonical tokenization of the grammar's forced byte-run from the
// masker's current (post-committed) state, capped at k tokens. ctx is unused — the
// grammar state, not the token history, drives this drafter.
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

// validateGrammarSpec gates the grammar-spec path: a mask + drafter, greedy only, and
// a rollback-safe (non-recurrent) family. Shared by the Model and Session entry points.
func validateGrammarSpec(target *Model, mask *constrain.Masker, drafter Drafter, sp SamplingParams) error {
	if mask == nil || drafter == nil {
		return fmt.Errorf("decoder.GenerateGrammarSpeculative: nil mask/drafter")
	}
	if sp.Temperature != 0 {
		return fmt.Errorf("decoder.GenerateGrammarSpeculative: greedy only for now (Temperature must be 0)")
	}
	// The masked verify argmaxes; penalties and logit bias cannot be applied losslessly there, so reject them rather than
	// silently drop them.
	if sp.HistoryDependent() {
		return fmt.Errorf("decoder.GenerateGrammarSpeculative: repetition penalties / logit bias not supported in greedy speculative decoding; use Generate")
	}
	if !target.specRollbackSafe() {
		return fmt.Errorf("decoder.GenerateGrammarSpeculative: recurrent family or staged sliding-window (rollback unsupported); use Generate")
	}
	return nil
}

// GenerateGrammarSpeculative is grammar-masked speculative decode: the drafter proposes tokens (a GrammarDrafter's
// forced byte-run, an n-gram copy, or a RouterDrafter fusing them) and the verify applies the grammar mask at every
// position, so the output is exactly what constrained Generate (sp.LogitProcessor = mask.Process) would produce:
// bit-exact greedy (TestGrammarSpecParity). The grammar forces structural tokens; an n-gram source additionally copies
// free values that echo the context, all validated by the mask.
//
// Limitations: greedy and the CPU staged path only (no sampling, no resident path). The recurrent-family guard
// (specRollbackSafe) applies as for the n-gram path.
func (target *Model) GenerateGrammarSpeculative(ctx context.Context, prompt []int, maxTokens int, mask *constrain.Masker, drafter Drafter, K int, sp SamplingParams) (<-chan int, *Generation, error) {
	if err := validateGrammarSpec(target, mask, drafter, sp); err != nil {
		return nil, nil, err
	}
	if len(prompt) == 0 {
		return nil, nil, fmt.Errorf("decoder.GenerateGrammarSpeculative: empty prompt")
	}
	out := make(chan int)
	stats := &SpecStats{}
	g := &Generation{Spec: stats}
	go func() {
		defer close(out)
		target.genGrammarInto(ctx, out, g, stats, mask, drafter, prompt, 0, maxTokens, K, sp, nil, nil, nil)
	}()
	return out, g, nil
}

// genGrammarInto is the grammar-masked speculative loop, cache/prefill/commit-aware
// (mirrors genNgramInto) so both Model.GenerateGrammarSpeculative (its own cache) and
// Session.GenerateGrammarSpeculative (a warm KV prefix) share it. cache==nil makes its
// own; otherwise the caller's cache must already hold prompt[:prefillFrom], and commit
// is invoked as each confirmed token lands in the cache (so a Session can keep its
// token list mirroring the cache for the next call's prefix match).
func (target *Model) genGrammarInto(ctx context.Context, out chan<- int, g *Generation, stats *SpecStats, mask *constrain.Masker, drafter Drafter, prompt []int, prefillFrom, maxTokens, K int, sp SamplingParams, tr specTracer, cache *KVCache, commit func(int)) {
	if K < 1 {
		K = 8
	}
	// Source label for traces: a pure GrammarDrafter pins forced bytes (forced=true); other drafters ride the same masked
	// verify but are not grammar-forced. Per-round attribution inside a RouterDrafter is not recorded.
	traceSource, traceForced := "router", false
	if _, ok := drafter.(*GrammarDrafter); ok {
		traceSource, traceForced = "grammar", true
	}
	tc := cache
	if tc == nil {
		tc = target.NewCache(len(prompt) + maxTokens + K + 8)
	}
	tpos := len(prompt)

	seedLogits, err := target.prefillLogits(ctx, prompt[prefillFrom:], tc)
	if err != nil {
		g.err = err
		return
	}
	// The grammar masks only GENERATED tokens (not the prompt), starting fresh —
	// matching constrained Generate. cur is the next confirmed token, pending.
	mask.MaskAt(mask.GrammarClone(), seedLogits) // grammar at its initial state
	cur := argmax(seedLogits)

	emit := func(tok int) bool {
		if target.isStop(tok, sp) {
			return false
		}
		select {
		case <-ctx.Done():
			g.err = ctx.Err()
			return false
		case out <- tok:
		}
		stats.Emitted++
		return stats.Emitted < maxTokens
	}
	// finishTrailing forwards cur when the generation ended by reaching maxTokens just after streaming it, so the cache
	// holds the prompt plus every emitted token, as plain constrained decode's does. Otherwise a round's trailing token is
	// forwarded only as the next round's seq[0], and the next turn re-prefills that position, which changes its output on a
	// backend whose batched prefill is not bit-identical to decode (see genNgramInto's finishTrailing). A stop or a cancel
	// takes no forward.
	finishTrailing := func(tok int) {
		if g.err != nil || stats.Emitted < maxTokens || target.isStop(tok, sp) {
			return
		}
		if _, err := target.forwardN(ctx, []int{tok}, tc); err != nil {
			return // the cache stays one token short, which is consistent
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

	// hist = prompt + committed tokens, the context an n-gram source searches
	// (the grammar source ignores it). cur is the pending next token.
	hist := slices.Clone(prompt)
	var lookupBuf []int
	var seqBuf []int
	for {
		// cur was confirmed last round — commit it to the live grammar, then draft
		// from the source(s): grammar's forced byte-run and/or an n-gram copy.
		mask.Commit(cur)
		lookupBuf = append(lookupBuf[:0], hist...)
		lookupBuf = append(lookupBuf, cur)
		draftTok := drafter.Draft(lookupBuf, K)
		kEff := len(draftTok)

		base := tpos
		seqBuf = append(seqBuf[:0], cur)
		seqBuf = append(seqBuf, draftTok...)
		logitsN, err := target.forwardN(ctx, seqBuf, tc)
		if err != nil {
			g.err = err
			return
		}

		// Verify each draft position against the GRAMMAR-MASKED target argmax,
		// rolling a grammar clone forward over the accepted prefix so each position
		// is masked at the right state (== what constrained Generate would emit).
		stats.Rounds++
		stats.Drafted += kEff
		gc := mask.GrammarClone() // grammar state after cur
		accepted := 0
		allAccept := true
		var nextTok int
		for i := range kEff {
			mask.MaskAt(gc, logitsN[i]) // illegal logits → -inf, so the dist is the masked one
			ti := argmax(logitsN[i])
			acc := draftTok[i] == ti
			if tr != nil {
				// accept_prob = masked p(drafted token): for a grammar-forced token it's <1
				// only when the model prefers a different LEGAL tokenization of the same bytes.
				pTop1, pEnt, pTok := targetDist(logitsN[i], draftTok[i])
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
			nextTok = argmax(logitsN[kEff])
		}
		stats.Accepted += accepted
		evaluated := accepted
		if !allAccept {
			evaluated = accepted + 1
		}
		stats.Evaluated += evaluated
		// Tell the drafter how its proposal fared, so a router can shrink each source's static estimate toward its running rate.
		if rec, ok := drafter.(OutcomeRecorder); ok && kEff > 0 {
			rec.RecordOutcome(accepted, kEff)
		}

		// Roll the cache back to cur + accepted; commit the accepted drafts to the
		// live grammar (cur was already committed above). nextTok stays pending.
		tpos = base + 1 + accepted
		tc.TruncateTo(tpos)
		hist = append(hist, cur) // cur is now committed (durable in the cache)
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
