package specdecode

import (
	"context"

	"github.com/townsendmerino/goinfer/internal/runtime/sampling"
)

// SpecStats accumulates CPU speculative-decoding telemetry for one Generate run.
type SpecStats struct {
	Rounds    int // verification passes (one expensive target forwardN each)
	Drafted   int // draft tokens proposed (== Rounds*K)
	Evaluated int // draft positions actually verified (accepts + the one rejection/round)
	Accepted  int // draft tokens the target confirmed (excludes the per-round correction/bonus)
	Emitted   int // tokens streamed to the caller
}

// AcceptanceRate is accepted draft tokens / proposed.
func (s *SpecStats) AcceptanceRate() float64 {
	if s.Drafted == 0 {
		return 0
	}
	return float64(s.Accepted) / float64(s.Drafted)
}

// EvalAcceptanceRate is accepted / evaluated.
func (s *SpecStats) EvalAcceptanceRate() float64 {
	if s.Evaluated == 0 {
		return 0
	}
	return float64(s.Accepted) / float64(s.Evaluated)
}

// TokensPerRound is the mean tokens confirmed per target pass.
func (s *SpecStats) TokensPerRound() float64 {
	if s.Rounds == 0 {
		return 0
	}
	return float64(s.Emitted) / float64(s.Rounds)
}

// DefaultSpecDepth is the draft depth K a draft-model or n-gram speculative call takes when it passes none.
const DefaultSpecDepth = 4

// DefaultGrammarSpecDepth is the grammar-speculative call's K when it passes none.
const DefaultGrammarSpecDepth = 8

// Forward is the narrow forward surface shared by draft-model and grammar speculative loops.
type Forward interface {
	Prefill(ctx context.Context, ids []int) ([]float32, error)
	Verify(ctx context.Context, ids []int) ([][]float32, error)
	Step(ctx context.Context, id int) ([]float32, error)
	TruncateTo(pos int)
}

// DraftSpecLoop runs greedy speculative decoding with a draft model.
func DraftSpecLoop(ctx context.Context, tf, df Forward, isStop func(int) bool, ctxCap func() int, out chan<- int, errOut *error, stats *SpecStats, prompt []int, maxTokens, K int) {
	seedLogits, err := tf.Prefill(ctx, prompt)
	if err != nil {
		*errOut = err
		return
	}
	tpos := len(prompt)
	if _, err := df.Prefill(ctx, prompt); err != nil {
		*errOut = err
		return
	}
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

	if !emit(cur) {
		return
	}
	for {
		kRound := RoundDraftWidth(K, tpos, ctxCap())
		if kRound < 0 {
			return
		}

		draftTok := make([]int, kRound)
		d := cur
		for i := range kRound {
			dl, err := df.Step(ctx, d)
			if err != nil {
				*errOut = err
				return
			}
			draftTok[i] = sampling.Argmax(dl)
			d = draftTok[i]
		}

		base := tpos
		seq := make([]int, 0, kRound+1)
		seq = append(seq, cur)
		seq = append(seq, draftTok...)
		logitsN, err := tf.Verify(ctx, seq)
		if err != nil {
			*errOut = err
			return
		}

		stats.Rounds++
		stats.Drafted += kRound
		accepted := 0
		allAccept := true
		var nextTok int
		for i := range kRound {
			ti := sampling.Argmax(logitsN[i])
			if draftTok[i] == ti {
				accepted++
				continue
			}
			nextTok = ti
			allAccept = false
			break
		}
		stats.Evaluated += accepted
		if !allAccept {
			stats.Evaluated++
		}
		if allAccept {
			nextTok = sampling.Argmax(logitsN[kRound])
		}
		stats.Accepted += accepted

		keep := base + 1 + accepted
		tpos = keep
		tf.TruncateTo(keep)
		switch {
		case allAccept && kRound > 0:
			if _, err := df.Step(ctx, draftTok[kRound-1]); err != nil {
				*errOut = err
				return
			}
		case allAccept:
			if _, err := df.Step(ctx, cur); err != nil {
				*errOut = err
				return
			}
		}
		df.TruncateTo(keep)

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
}
