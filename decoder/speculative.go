package decoder

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/townsendmerino/goinfer/internal/runtime/specdecode"
)

// GenerateSpeculative runs greedy speculative decoding with a draft model.
func (target *Model) GenerateSpeculative(ctx context.Context, prompt []int, maxTokens int, draft *Model, K int, sp SamplingParams) (<-chan int, *Generation, error) {
	if draft == nil {
		return nil, nil, fmt.Errorf("decoder.GenerateSpeculative: nil draft model")
	}
	if sp.Temperature != 0 {
		return nil, nil, fmt.Errorf("decoder.GenerateSpeculative: greedy only (Temperature must be 0); use Generate for sampling")
	}
	if sp.LogitProcessor != nil {
		return nil, nil, fmt.Errorf("decoder.GenerateSpeculative: LogitProcessor (constrained decoding) not supported yet; use Generate")
	}
	if sp.HistoryDependent() {
		return nil, nil, fmt.Errorf("decoder.GenerateSpeculative: repetition penalties / logit bias are not supported in greedy speculative decoding; use Generate")
	}
	if dv, tv := draft.w.Arch.VocabSize, target.w.Arch.VocabSize; dv != tv {
		return nil, nil, fmt.Errorf("decoder.GenerateSpeculative: draft/target vocab mismatch (%d vs %d) — they must share a tokenizer", dv, tv)
	}
	if len(prompt) == 0 {
		return nil, nil, fmt.Errorf("decoder.GenerateSpeculative: empty prompt")
	}
	if K < 1 {
		K = defaultSpecDepth
	}
	if !target.specRollbackSafe() {
		return nil, nil, fmt.Errorf("decoder.GenerateSpeculative: this model has recurrent state (Mamba-2 / Gated DeltaNet) or a staged sliding-window ring cache that speculative rollback cannot losslessly restore; use Generate")
	}
	if err := target.SpecDecodeConflict(); err != nil {
		return nil, nil, fmt.Errorf("decoder.GenerateSpeculative: %w; use Generate", err)
	}

	resident := target.resident != nil && target.DecodeRunnerEligible()
	draftResident := draft.resident != nil && draft.DecodeRunnerEligible()

	out := make(chan int)
	stats := &SpecStats{}
	g := &Generation{Spec: stats}
	leaveExact := target.enterExactAttention()
	target.runGen(ctx, out, g, leaveExact, func(ctx context.Context) {
		defer leaveExact()
		if resident {
			if target.tryClaimResident() {
				defer atomic.StoreInt32(&target.resBusy, 0)
				target.resHost().ForgetIDs()
			} else {
				resident = false
			}
		}
		if draftResident {
			if draft.tryClaimResident() {
				defer atomic.StoreInt32(&draft.resBusy, 0)
				draft.resHost().ForgetIDs()
			} else {
				draftResident = false
			}
		}
		room := len(prompt) + maxTokens + K + 8
		var tf, df Forward
		if resident {
			tf = &residentForward{m: target}
		} else {
			tf = cpuForward{target, target.NewCache(room)}
		}
		if draftResident {
			df = &residentForward{m: draft}
		} else {
			df = cpuForward{draft, draft.NewCache(room)}
		}
		isStop := func(tok int) bool { return target.isStop(tok, sp) }
		specdecode.DraftSpecLoop(ctx, tf, df, isStop, target.ResidentContextCap, out, &g.err, stats, prompt, maxTokens, K)
	})
	return out, g, nil
}
