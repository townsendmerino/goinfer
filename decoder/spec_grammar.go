package decoder

import (
	"context"
	"fmt"

	"github.com/townsendmerino/goinfer/constrain"
)

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
	if len(bytes) == 0 {
		return nil
	}
	toks := d.Encode(string(bytes))
	if len(toks) > k {
		toks = toks[:k]
	}
	return toks
}

// GenerateGrammarSpeculative is grammar-fused speculative decode (01): the grammar's
// forced byte-runs are drafted for free and verified in one target pass, with the
// grammar mask applied at every verified position so the output is exactly what
// constrained Generate (sp.LogitProcessor = mask.Process) would produce — bit-exact
// greedy (TestGrammarSpecParity). The win is on structured / tool-call output, where
// keys and enum/const literals are grammar-determined.
//
// First cut: greedy + CPU staged path. Sampling and the resident path are follow-ups;
// the recurrent-family guard (specRollbackSafe) applies as for the n-gram path.
func (target *Model) GenerateGrammarSpeculative(ctx context.Context, prompt []int, maxTokens int, mask *constrain.Masker, encode func(string) []int, K int, sp SamplingParams) (<-chan int, *Generation, error) {
	if mask == nil || encode == nil {
		return nil, nil, fmt.Errorf("decoder.GenerateGrammarSpeculative: nil mask/encode")
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
	drafter := &GrammarDrafter{Mask: mask, Encode: encode}

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

		for {
			// cur was confirmed last round — commit it to the live grammar, then draft
			// the forced byte-run that follows it.
			mask.Commit(cur)
			draftTok := drafter.Draft(nil, K)
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
			for i := 0; i < accepted; i++ {
				mask.Commit(draftTok[i])
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
