package decoder

import (
	"context"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/internal/runtime/specdecode"
)

const (
	defaultTheta          = specdecode.DefaultTheta
	sequentialVerifyTheta = specdecode.SequentialVerifyTheta
	grammarConf           = specdecode.GrammarConf
	blockSpecFuseRows     = specdecode.BlockSpecFuseRows
)

var ngramAlphaAnchors = specdecode.NgramAlphaAnchors

func thetaFor(backend string) float64          { return specdecode.ThetaFor(backend) }
func ngramAlpha(matchLen int) float64          { return specdecode.NgramAlpha(matchLen) }
func specRoundDraftWidth(k, pos, cap int) int  { return specdecode.RoundDraftWidth(k, pos, cap) }
func blockSpecRoundWidth(w, m, e, p, c int) int { return specdecode.BlockSpecRoundWidth(w, m, e, p, c) }

func draftSpecLoop(ctx context.Context, tf, df Forward, isStop func(int) bool, ctxCap func() int, out chan<- int, g *Generation, stats *SpecStats, prompt []int, maxTokens, K int) {
	specdecode.DraftSpecLoop(ctx, tf, df, isStop, ctxCap, out, &g.err, stats, prompt, maxTokens, K)
}

func grammarSpecLoop(ctx context.Context, fw Forward, isStop func(int) bool, out chan<- int, g *Generation, stats *SpecStats, mask *constrain.Masker, drafter Drafter, prompt []int, prefillFrom, maxTokens, K int, tr specTracer, commit func(int)) {
	specdecode.GrammarSpecLoop(ctx, fw, isStop, out, &g.err, stats, mask, drafter, prompt, prefillFrom, maxTokens, K, tr, commit)
}

func ngramSpecLoop(ctx context.Context, tg ngramTarget, seedLogits []float32, out chan<- int, g *Generation, stats *SpecStats, drafter Drafter, prompt []int, maxTokens, K int, sampler *Sampler, needHist bool, tr specTracer, ad *AdaptiveDepth, commit func(int)) {
	var samp specdecode.NgramSampler
	if sampler != nil {
		samp = ngramSamplerAdapter{sampler}
	}
	specdecode.NgramSpecLoop(ctx, tg, seedLogits, out, &g.err, stats, drafter, prompt, maxTokens, K, samp, needHist, tr, ad, commit)
}
