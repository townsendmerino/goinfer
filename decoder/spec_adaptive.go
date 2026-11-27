package decoder

import (
	"context"

	"github.com/townsendmerino/goinfer/internal/runtime/specdecode"
)

// GenerateNgramSpeculativeAdaptive is GenerateNgramSpeculative with the fixed K
// replaced by the AdaptiveDepth controller ad (nil ⇒ a fresh default).
func (target *Model) GenerateNgramSpeculativeAdaptive(ctx context.Context, prompt []int, maxTokens int, drafter Drafter, ad *AdaptiveDepth, sp SamplingParams) (<-chan int, *Generation, error) {
	if ad == nil {
		ad = &AdaptiveDepth{}
	}
	if ad.Theta <= 0 {
		ad.Theta = target.verifyTheta()
	}
	if ad.Cost == nil && sp.Temperature <= 0 && target.resident != nil {
		ad.Cost = target.resRuntime().VerifyCost()
	}
	specdecode.EnsureAdaptive(ad)
	if ad.Theta >= 1 {
		if err := validateNgramSpec(target, drafter, sp); err != nil {
			return nil, nil, err
		}
		ch, gen := target.Generate(ctx, prompt, maxTokens, sp)
		return ch, gen, nil
	}
	return target.genNgram(ctx, prompt, maxTokens, drafter, ad.MaxDraft, sp, nil, ad)
}

// verifyTheta returns the Theta for the path this model's speculative verify actually runs on.
func (m *Model) verifyTheta() float64 {
	if m == nil || m.resident == nil || m.be == nil {
		return specdecode.DefaultTheta
	}
	if batched, _, ok := m.resRuntime().VerifyPathInfo(); ok {
		if !batched {
			return specdecode.SequentialVerifyTheta
		}
		return specdecode.ThetaFor(m.be.Name())
	}
	if canStep(m.resident, stepBatch{Pass: passRows, Emb: stepProbe}) {
		if batched, _, ok := m.resPrefill().PrefillPathInfo(); ok && !batched {
			return specdecode.SequentialVerifyTheta
		}
	}
	return specdecode.ThetaFor(m.be.Name())
}
