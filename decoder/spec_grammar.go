package decoder

import (
	"context"
	"fmt"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/internal/runtime/specdecode"
)

func validateGrammarSpec(target *Model, mask *constrain.Masker, drafter Drafter, sp SamplingParams) error {
	if mask == nil || drafter == nil {
		return fmt.Errorf("decoder.GenerateGrammarSpeculative: nil mask/drafter")
	}
	if sp.Temperature != 0 {
		return fmt.Errorf("decoder.GenerateGrammarSpeculative: greedy only for now (Temperature must be 0)")
	}
	if sp.HistoryDependent() {
		return fmt.Errorf("decoder.GenerateGrammarSpeculative: repetition penalties / logit bias not supported in greedy speculative decoding; use Generate")
	}
	if !target.specRollbackSafe() {
		return fmt.Errorf("decoder.GenerateGrammarSpeculative: recurrent family or staged sliding-window (rollback unsupported); use Generate")
	}
	return nil
}

// GenerateGrammarSpeculative is grammar-masked speculative decode.
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
	target.runGen(ctx, out, g, nil, func(ctx context.Context) {
		target.genGrammarInto(ctx, out, g, stats, mask, drafter, prompt, 0, maxTokens, K, sp, nil, nil, nil)
	})
	return out, g, nil
}

func (target *Model) genGrammarInto(ctx context.Context, out chan<- int, g *Generation, stats *SpecStats, mask *constrain.Masker, drafter Drafter, prompt []int, prefillFrom, maxTokens, K int, sp SamplingParams, tr specTracer, cache *KVCache, commit func(int)) {
	if K < 1 {
		K = defaultGrammarSpecDepth
	}
	tc := cache
	if tc == nil {
		tc = target.NewCache(len(prompt) + maxTokens + K + 8)
	}
	isStop := func(tok int) bool { return target.isStop(tok, sp) }
	specdecode.GrammarSpecLoop(ctx, cpuForward{target, tc}, isStop, out, &g.err, stats, mask, drafter, prompt, prefillFrom, maxTokens, K, tr, commit)
}
