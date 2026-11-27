package decoder

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/townsendmerino/goinfer/internal/runtime/specdecode"
)

// GenerateNgramSpeculative is single-model speculative decoding driven by an n-gram Drafter.
func (target *Model) GenerateNgramSpeculative(ctx context.Context, prompt []int, maxTokens int, drafter Drafter, K int, sp SamplingParams) (<-chan int, *Generation, error) {
	return target.genNgram(ctx, prompt, maxTokens, drafter, K, sp, nil, nil)
}

func validateNgramSpec(m *Model, drafter Drafter, sp SamplingParams) error {
	if drafter == nil {
		return fmt.Errorf("decoder.GenerateNgramSpeculative: nil drafter")
	}
	if sp.LogitProcessor != nil {
		return fmt.Errorf("decoder.GenerateNgramSpeculative: LogitProcessor (constrained/tool decoding) not supported yet; use Generate")
	}
	if sp.Temperature <= 0 && sp.HistoryDependent() {
		return fmt.Errorf("decoder.GenerateNgramSpeculative: repetition penalties / logit bias are not supported in greedy speculative decoding; use Generate")
	}
	if !m.specRollbackSafe() {
		return fmt.Errorf("decoder.GenerateNgramSpeculative: this model has recurrent state (Mamba-2 / Gated DeltaNet / LFM2 conv / KDA) or a staged sliding-window ring cache that speculative rollback cannot losslessly restore; use Generate")
	}
	return nil
}

func (target *Model) genNgram(ctx context.Context, prompt []int, maxTokens int, drafter Drafter, K int, sp SamplingParams, tr specTracer, ad *AdaptiveDepth) (<-chan int, *Generation, error) {
	if err := validateNgramSpec(target, drafter, sp); err != nil {
		return nil, nil, err
	}
	if err := target.SpecDecodeConflict(); err != nil {
		return nil, nil, fmt.Errorf("decoder.GenerateNgramSpeculative: %w; use Generate", err)
	}
	if len(prompt) == 0 {
		return nil, nil, fmt.Errorf("decoder.GenerateNgramSpeculative: empty prompt")
	}
	out := make(chan int)
	stats := &SpecStats{}
	g := &Generation{Spec: stats}
	leaveExact := target.enterExactAttention()
	target.runGen(ctx, out, g, leaveExact, func(ctx context.Context) {
		defer leaveExact()
		target.genNgramInto(ctx, out, g, stats, drafter, prompt, 0, maxTokens, K, sp, tr, ad, nil, nil)
	})
	return out, g, nil
}

func (target *Model) genNgramInto(ctx context.Context, out chan<- int, g *Generation, stats *SpecStats, drafter Drafter, prompt []int, prefillFrom, maxTokens, K int, sp SamplingParams, tr specTracer, ad *AdaptiveDepth, cache *KVCache, commit func(int)) {
	if len(prompt) == 0 {
		g.err = fmt.Errorf("decoder.GenerateNgramSpeculative: empty prompt")
		return
	}
	if K < 1 {
		K = defaultSpecDepth
	}
	sampled := sp.Temperature > 0
	var sampler *Sampler
	needHist := false
	if sampled {
		sampler = NewSampler(sp)
		defer sampler.release()
		needHist = sampler.needsHistory()
	}

	resident := cache == nil && target.resident != nil && target.DecodeRunnerEligible()
	adaptive := resident && target.specAdaptive && target.batcher != nil && !sampled
	mc3Slot := -1
	if resident && !adaptive {
		claimed := false
		if target.batcher != nil {
			claimed = target.batcher.ClaimExclusive(&target.resBusy)
		} else {
			claimed = atomic.CompareAndSwapInt32(&target.resBusy, 0, 1)
		}
		if claimed {
			defer atomic.StoreInt32(&target.resBusy, 0)
		} else {
			resident = false
		}
	}

	tc := cache
	if tc == nil && !resident {
		tc = target.NewCache(len(prompt) + maxTokens + K + 8)
	}

	targetVerify := func(seq []int, base int) ([][]float32, error) {
		if resident {
			if adaptive {
				if err := target.resHost().Bind(mc3Slot); err != nil {
					return nil, err
				}
			}
			embs := make([][]float32, len(seq))
			for i, tok := range seq {
				embs[i] = target.embedResident(tok)
			}
			return target.resident.ForwardN(embs, base)
		}
		e := target.cpuEngine()
		return e.ForwardN(ctx, seq, unwrapCache(tc))
	}
	targetVerifyBatch := func(tok, base int) ([][]float32, error) {
		emb := target.embedResident(tok)
		var row []float32
		q := &batchReq{Seq: ResidentBatchSeq{Slot: mc3Slot, Pos: base, Emb: emb}, Solo: func() error {
			if err := target.resHost().Bind(mc3Slot); err != nil {
				return err
			}
			var ferr error
			row, ferr = target.resident.Forward(emb, base)
			return ferr
		}}
		if err := target.batcher.Forward(q); err != nil {
			return nil, err
		}
		if q.Out != nil {
			row = q.Out.Logits
		}
		return [][]float32{row}, nil
	}
	var idsVerify func(seq []int, base int) ([]int, error)
	if resident && !sampled && tr == nil {
		hasG := target.canDecode(outArgmax)
		hasAV := canStep(target.resident, stepBatch{Pass: passRows, Emb: stepProbe, Out: outEveryArgmax})
		if hasG || hasAV {
			var oneRow [1][]float32
			idsVerify = func(seq []int, base int) ([]int, error) {
				if adaptive {
					if err := target.resHost().Bind(mc3Slot); err != nil {
						return nil, err
					}
				}
				if len(seq) == 1 && hasG {
					oneRow[0] = target.embedResident(seq[0])
					sr, e := target.step(stepBatch{Pass: passToken, Emb: oneRow[:], Pos: base, Out: outArgmax})
					return []int{sr.ID}, e
				}
				if !hasAV {
					return nil, fmt.Errorf("decoder: no argmax-only batched verify on this resident")
				}
				embs := make([][]float32, len(seq))
				for i, tok := range seq {
					embs[i] = target.embedResident(tok)
				}
				sr, e := target.step(stepBatch{Pass: passRows, Emb: embs, Pos: base, Out: outEveryArgmax})
				return sr.IDs, e
			}
		}
	}
	targetTruncate := func(keep int) {
		if resident {
			if adaptive {
				_ = target.resHost().Bind(mc3Slot)
			}
			target.resident.TruncateTo(keep)
		} else {
			tc.TruncateTo(keep)
		}
	}

	var seedLogits []float32
	var err error
	var releaseAdaptive func()
	defer func() {
		if releaseAdaptive != nil {
			releaseAdaptive()
		}
	}()
	if resident {
		if adaptive {
			var ok bool
			seedLogits, mc3Slot, releaseAdaptive, ok, err = target.prefillNgramAdaptive(ctx, g, prompt)
			if err != nil {
				g.err = err
				return
			}
			if !ok {
				resident, adaptive = false, false
			}
		}
		if resident && !adaptive {
			reuseFrom := target.resHost().AcquireIn(prompt, nil, 0, nil, kvScopeOf(ctx)).Reuse
			target.resHost().ForgetIDs()
			if seedLogits, err = target.residentPrefillSeed(ctx, prompt, reuseFrom, false); err != nil {
				g.err = err
				return
			}
			g.PrefillReused = reuseFrom
		} else if !resident {
			if tc == nil {
				tc = target.NewCache(len(prompt) + maxTokens + K + 8)
			}
			if seedLogits, err = target.prefillLogits(ctx, prompt[prefillFrom:], tc); err != nil {
				g.err = err
				return
			}
		}
	} else {
		if seedLogits, err = target.prefillLogits(ctx, prompt[prefillFrom:], tc); err != nil {
			g.err = err
			return
		}
	}
	commitResident := func(generated []int) {
		if !resident {
			return
		}
		if adaptive {
			target.batcher.Exclusive(func() {
				if err := target.resHost().Bind(mc3Slot); err == nil {
					target.resHost().CommitIDs(prompt, generated, nil, 0)
				}
			})
			if releaseAdaptive != nil {
				releaseAdaptive()
			}
		} else {
			target.resHost().CommitIDs(prompt, generated, nil, 0)
		}
	}
	var samp specdecode.NgramSampler
	if sampled {
		samp = ngramSamplerAdapter{sampler}
	}
	tg := ngramTarget{
		Verify: targetVerify, IDsVerify: idsVerify, VerifyBatch: targetVerifyBatch, Truncate: targetTruncate,
		CommitResident: commitResident, IsStop: func(tok int) bool { return target.isStop(tok, sp) },
		CtxCap: target.ResidentContextCap, Adaptive: adaptive,
		NeverYield: adaptive && target.knobs.get(knobSpecAdaptiveNeverYield) != "",
		Exclusive:  func(f func()) { target.batcher.Exclusive(f) },
		Claim:      func() bool { return target.batcher.ClaimAdaptiveRound(&target.resBusy) },
		Release:    func() { atomic.StoreInt32(&target.resBusy, 0) },
	}
	specdecode.NgramSpecLoop(ctx, tg, seedLogits, out, &g.err, stats, drafter, prompt, maxTokens, K, samp, needHist, tr, ad, commit)
}

func (target *Model) prefillNgramAdaptive(ctx context.Context, g *Generation, prompt []int) (seedLogits []float32, slot int, release func(), ok bool, err error) {
	if !target.batcher.ClaimAdaptive(&target.resBusy) {
		return nil, -1, nil, false, nil
	}
	var reuseFrom int
	slot = -1
	target.batcher.Exclusive(func() {
		acq := target.resHost().AcquireIn(prompt, nil, 0, target.batcher.BusySlots(), kvScopeOf(ctx))
		reuseFrom, slot = acq.Reuse, acq.Slot
		if slot >= 0 {
			target.batcher.MarkSlot(slot, true)
			target.resHost().ForgetIDs()
		}
	})
	if slot < 0 {
		target.batcher.ReleaseAdaptive()
		return nil, -1, nil, false, nil
	}
	released := false
	release = func() {
		if !released {
			released = true
			target.batcher.MarkSlot(slot, false)
			target.batcher.ReleaseAdaptive()
		}
	}
	target.batcher.PrefillExclusive(func() {
		if err = target.resHost().Bind(slot); err != nil {
			return
		}
		seedLogits, err = target.residentPrefillSeed(ctx, prompt, reuseFrom, false)
	})
	if err != nil {
		release()
		return nil, -1, nil, false, err
	}
	g.PrefillReused = reuseFrom
	return seedLogits, slot, release, true, nil
}
