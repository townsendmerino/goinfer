package decoder

import (
	"context"
	"fmt"
	"slices"

	"github.com/townsendmerino/goinfer/constrain"
)

// ConfidentDrafter reports a confidence for its most recent Draft, so RouterDrafter
// can order sources by predicted acceptance per position (03 inc 3) rather than a
// fixed priority. Higher = more likely accepted. (A heuristic stand-in for the §06
// trained α̂; calibration is a follow-up.)
type ConfidentDrafter interface {
	Drafter
	Confidence() float64
}

// RouterDrafter fuses drafters (03 router): each round it polls every source and
// returns the proposal of the most CONFIDENT one (inc 3). This captures the inc-2
// "tree-recoverable" upside without a tree — the measurement showed the linear
// grammar-first priority mis-picked where a long n-gram copy was the reliable choice
// (n-gram 15/15 vs grammar 2/5 under tokenization), and confidence ordering prefers
// the long copy. Sources without a Confidence() score rank at 0; ties keep source
// order. On a constrained request all ride the same masked verify, so the choice
// only affects speed, never correctness.
type RouterDrafter struct {
	Sources []Drafter
}

// Draft returns the most-confident source's non-empty proposal this position.
func (r *RouterDrafter) Draft(ctx []int, k int) []int {
	var best []int
	bestConf := -1.0
	for _, s := range r.Sources {
		d := s.Draft(ctx, k)
		if len(d) == 0 {
			continue
		}
		conf := 0.0
		if cd, ok := s.(ConfidentDrafter); ok {
			conf = cd.Confidence()
		}
		if conf > bestConf {
			best, bestConf = d, conf
		}
	}
	return best
}

// GrammarDrafter proposes the grammar's FORCED byte-run as draft tokens (01
// grammar-fused). At positions where the grammar determines the bytes (inside object
// keys, enum/const values), it extracts that byte run from the masker's current state
// and retokenizes it canonically — the tokens the model most likely emits there. The
// verifier confirms under the same mask, so it is lossless; acceptance is < 1 only
// where the model tokenizes the forced bytes differently than `encode`.
type GrammarDrafter struct {
	Mask     *constrain.Masker  // the live grammar mask (advanced over committed tokens)
	Encode   func(string) []int // tokenizer: forced bytes → canonical token ids
	MaxBytes int                // cap on the forced byte-run probed per round (default 64)

	lastForced int // bytes in the most recent forced run (0 = abstained); for Confidence
}

// grammarConf is the router confidence of a forced grammar proposal. Set so a
// SUBSTANTIAL n-gram copy (suffix match ≥ 4 tokens) outranks it but short/spurious
// n-gram matches (2–3) don't — grammar's forced bytes are reliable structurally but
// risk a tokenization mismatch (measured 2/5), so only a long, confident copy should
// override it. Heuristic; the §06 α̂ would replace it.
const grammarConf = 3.0

// Confidence reports the router score of the last Draft: grammarConf when the grammar
// forced something, 0 when it abstained.
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

// GenerateGrammarSpeculative is grammar-masked speculative decode (01 / 03): the
// drafter proposes tokens (a GrammarDrafter's forced byte-run, an n-gram copy, or a
// RouterDrafter fusing them, 03), and the verify applies the grammar mask at every
// position so the output is exactly what constrained Generate (sp.LogitProcessor =
// mask.Process) would produce — bit-exact greedy (TestGrammarSpecParity). The grammar
// forces structural tokens; an n-gram source additionally copies free values that
// echo the context, all validated by the mask.
//
// First cut: greedy + CPU staged path. Sampling and the resident path are follow-ups;
// the recurrent-family guard (specRollbackSafe) applies as for the n-gram path.
func (target *Model) GenerateGrammarSpeculative(ctx context.Context, prompt []int, maxTokens int, mask *constrain.Masker, drafter Drafter, K int, sp SamplingParams) (<-chan int, *Generation, error) {
	if mask == nil || drafter == nil {
		return nil, nil, fmt.Errorf("decoder.GenerateGrammarSpeculative: nil mask/drafter")
	}
	if sp.Temperature != 0 {
		return nil, nil, fmt.Errorf("decoder.GenerateGrammarSpeculative: greedy only for now (Temperature must be 0)")
	}
	if !target.specRollbackSafe() {
		return nil, nil, fmt.Errorf("decoder.GenerateGrammarSpeculative: recurrent family (rollback unsupported); use Generate")
	}
	if len(prompt) == 0 {
		return nil, nil, fmt.Errorf("decoder.GenerateGrammarSpeculative: empty prompt")
	}
	if K < 1 {
		K = 8
	}

	out := make(chan int)
	stats := &SpecStats{}
	g := &Generation{Spec: stats}
	go func() {
		defer close(out)
		tc := target.NewCache(len(prompt) + maxTokens + K + 8)
		tpos := len(prompt)

		seedLogits, err := target.prefillLogits(prompt, tc)
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
		if !emit(cur) {
			return
		}

		// hist = prompt + committed tokens, the context an n-gram source searches
		// (the grammar source ignores it). cur is the pending next token.
		hist := slices.Clone(prompt)
		for {
			// cur was confirmed last round — commit it to the live grammar, then draft
			// from the source(s): grammar's forced byte-run and/or an n-gram copy.
			mask.Commit(cur)
			draftTok := drafter.Draft(append(slices.Clone(hist), cur), K)
			kEff := len(draftTok)

			base := tpos
			seq := append([]int{cur}, draftTok...)
			logitsN, err := target.forwardN(seq, tc)
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
			for i := 0; i < kEff; i++ {
				mask.MaskAt(gc, logitsN[i])
				ti := argmax(logitsN[i])
				if draftTok[i] != ti {
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

			// Roll the cache back to cur + accepted; commit the accepted drafts to the
			// live grammar (cur was already committed above). nextTok stays pending.
			tpos = base + 1 + accepted
			tc.TruncateTo(tpos)
			hist = append(hist, cur) // cur is now committed
			for i := 0; i < accepted; i++ {
				mask.Commit(draftTok[i])
				hist = append(hist, draftTok[i])
			}
			for i := 0; i < accepted; i++ {
				if !emit(draftTok[i]) {
					return
				}
			}
			cur = nextTok
			if !emit(cur) {
				return
			}
		}
	}()
	return out, g, nil
}
