package decoder

import (
	"fmt"
	"math"
	"sync/atomic"
	"time"

	"github.com/townsendmerino/goinfer/internal/runtime/logs"
)

// pick is decodeTarget.pick for this generation: the next token's sample, from the logits or from what the resident already chose.
func (r *genRun) pick(clk *decodeClock) (SampleInfo, int, bool, error) {
	m := r.m
	r.optResolved = false
	if r.fastNext >= 0 {
		// The resident already picked this token's argmax on-device (greedy fast
		// path): nothing reads logits this step, so there is nothing to process or
		// sample. Identical to the logits path — guarded by ArgmaxEquivalent/GreedyEquivalent.
		return SampleInfo{}, r.fastNext, true, nil
	}
	if r.optFwd && r.optGate.Should() {
		// Optimistic forward (spec_optfwd.go): samples AND resolves gpuPos+1's logits
		// together, overlapping the real sampler with a speculative Forward on a free
		// argmax guess. decodeTiming intentionally does not split this into
		// tProc/tSample/tEmbed — the whole overlapped operation folds into tFwd below
		// (t0 was set at the top of this iteration, untouched here), since sample and
		// forward no longer have a clean sequential boundary to attribute separately.
		res, operr := m.optFwdStep(r.sampler, r.logits, r.gpuPos, r.optGate, r.g.OptFwd)
		if operr != nil {
			return SampleInfo{}, 0, false, operr
		}
		r.optNextLogits = res.nextLogits
		r.optResolved = true
		return res.info, res.info.ID, true, nil
	}
	// Device top-K row from the previous forward: draw from it if the K candidates prove they
	// hold the whole retained set; otherwise read the full row for this token (no randomness
	// has been consumed, so the fallback is invisible) and take the ordinary path below.
	if r.topKRow == nil {
		return SampleInfo{}, 0, false, nil
	}
	var info SampleInfo
	drew := false
	if r.topKPre != nil {
		info, drew = *r.topKPre, true
		r.g.TopKServed++
	} else if r.topKPreFull {
		r.logits = r.topKFullBuf
		r.g.TopKFallbacks++
	} else if inf, ok := r.sampler.SampleFromTopK(*r.topKRow, r.topKVocab); ok {
		info, drew = inf, true
		r.g.TopKServed++
	} else {
		full, ferr := r.topKRow.Full()
		if ferr != nil {
			return SampleInfo{}, 0, false, ferr
		}
		r.logits = full
		r.g.TopKFallbacks++
	}
	r.topKRow, r.topKPre, r.topKPreFull = nil, nil, false
	if decodeTiming {
		clk.sample += time.Since(clk.t0)
	}
	return info, info.ID, drew, nil
}

// advance is decodeTarget.advance for this generation: it runs the forward that commits next and leaves the following step's logits (or token) in the run's fields.
func (r *genRun) advance(next int, needFull bool, clk *decodeClock) error {
	m := r.m
	var err error
	if r.optResolved {
		// optFwdStep already produced (or redid) this position's Forward and its
		// result logits for gpuPos+1 — nothing left to do but advance the position.
		r.logits = r.optNextLogits
		r.gpuPos++
	} else if r.useGPU {
		var emb []float32
		if decodeTiming {
			clk.t0 = time.Now()
			emb = m.embedResidentInto(next, r.embScratch)
			clk.embed += time.Since(clk.t0)
			clk.t0 = time.Now()
		} else {
			emb = m.embedResidentInto(next, r.embScratch)
		}
		r.embScratch = emb
		r.fastNext = -1 // set again below only by a path that picks the next token on-device
		pos := r.gpuPos
		// A device-drawn token takes its draw HERE, once, whichever path then serves it — the sampler's RNG
		// stream advances exactly as it always has.
		var draw *ResidentBatchDraw
		if !needFull && !r.fastGreedy && r.sampleOn {
			seed, d := r.sampler.NextDraw()
			draw = &ResidentBatchDraw{Temperature: r.sp.Temperature, Seed: seed, Draw: d}
		} else if r.batchGreedy && !needFull {
			draw = &ResidentBatchDraw{Greedy: true}
		}
		// residentCall is this token's production resident call.
		viaForward := false // residentCall returned Forward's logits, the resident's shared host buffer
		residentCall := func() error {
			var ferr error
			if needFull {
				// The gated processor needs this position's full logits (it is about to mask them). A chain still open stops
				// first: it has queued the forward for this position, which Forward then rewrites, and it reopens at the
				// next closed step.
				r.stopChain()
				r.logits, ferr = m.resident.Forward(emb, pos)
				viaForward = true
			} else if r.useChain && r.chainTok {
				// The chain fed itself this token: it is the argmax of the chain's previous forward. The first token
				// (from the prompt's seed logits) opens it.
				if !r.chainOpen {
					if ferr = r.chainRF.GreedyChainStart(next, pos); ferr == nil {
						r.chainOpen, r.chainPos = true, pos
					}
				}
				if ferr == nil && pos != r.chainPos {
					ferr = fmt.Errorf("decoder: greedy chain is at position %d, the decode loop at %d", r.chainPos, pos)
				}
				if ferr == nil {
					r.fastNext, ferr = r.chainRF.GreedyChainNext()
					r.chainPos++
				}
			} else if r.sChainRF != nil && r.chainTok {
				// The sampled chain: as the greedy chain, with this token's draw (taken above, as ForwardSample's is).
				if !r.chainOpen {
					if ferr = r.sChainRF.SampleChainStart(next, pos, draw.Temperature, draw.Seed, draw.Draw); ferr == nil {
						r.chainOpen, r.chainPos = true, pos
					}
				}
				if ferr == nil && pos != r.chainPos {
					ferr = fmt.Errorf("decoder: sampled chain is at position %d, the decode loop at %d", r.chainPos, pos)
				}
				if ferr == nil {
					r.fastNext, ferr = r.sChainRF.SampleChainNext(draw.Seed, draw.Draw)
					r.chainPos++
				}
				if ferr == nil {
					r.g.DeviceSampled++
				}
			} else if r.fastGreedy {
				// Greedy fast path: the resident picks the argmax on-device and returns
				// just the id, skipping the full-logits readback.
				r.stepRow[0] = emb
				var sr stepResult
				sr, ferr = m.step(stepBatch{Pass: passToken, Emb: r.stepRow[:], Pos: pos, Out: outArgmax})
				r.fastNext = sr.ID
			} else if r.sampleOn {
				r.stepRow[0] = emb
				var sr stepResult
				sr, ferr = m.step(stepBatch{Pass: passToken, Emb: r.stepRow[:], Pos: pos, Out: outSample,
					Temperature: draw.Temperature, Seed: draw.Seed, Draw: draw.Draw})
				r.fastNext = sr.ID
				if ferr == nil {
					r.g.DeviceSampled++
				}
			} else if r.topKOn {
				r.stepRow[0] = emb
				var sr stepResult
				sr, ferr = m.step(stepBatch{Pass: passToken, Emb: r.stepRow[:], Pos: pos, Out: outTopK, K: r.topKWidth,
					Temperature: r.sp.Temperature, WantZ: r.sampler.topPActive()})
				row := sr.TopK
				if ferr == nil && r.mc3 != nil {
					// Resolve the draw now, while this generation holds the resident (see topKOn). The sampler is this
					// generation's own and nothing else draws from it before the next iteration would have, so the RNG
					// is consumed in the same order as the unbatched path.
					if inf, ok := r.sampler.SampleFromTopK(row, r.topKVocab); ok {
						r.topKPre = &inf
					} else if full, e := row.Full(); e != nil {
						ferr = e
					} else {
						r.topKFullBuf = append(r.topKFullBuf[:0], full...) // the resident's host buffer is shared
						r.topKPreFull = true
					}
				}
				if ferr == nil {
					r.topKRow = &row
				}
			} else {
				r.logits, ferr = m.resident.Forward(emb, pos)
				viaForward = true
			}
			return ferr
		}
		if r.mc3 != nil {
			// Batched decode: submit the token; it runs in a shared step with the other generations' tokens, or — alone — as
			// residentCall on our own slot. A step's row is bit-identical to this token's own call: Forward's
			// logits, or ForwardSample's id for a device-drawn token (the same draw). A greedy token served by a
			// step continues on the full-logits path, which production's fast path equals by its own contract
			// (ArgmaxEquivalent/GreedyEquivalent above).
			solo := func() error {
				if berr := m.residentBind(r.mc3Slot); berr != nil {
					return berr
				}
				if ferr := residentCall(); ferr != nil {
					return ferr
				}
				if viaForward { // copy while this generation still holds the resident (mc3Logits)
					r.mc3Logits = append(r.mc3Logits[:0], r.logits...)
					r.logits = r.mc3Logits
				}
				return nil
			}
			// Alone on the batcher, a chainable token runs the chain with the resident held across tokens; the
			// first token anyone else wants the resident, the chain stops and this token takes the batcher.
			held := false
			if (r.useChain || r.sChainRF != nil) && !needFull {
				r.chainTok = true
				held, err = r.mc3.holdSolo(solo, r.stopChain)
				r.chainTok = false
			}
			q := &batchReq{seq: ResidentBatchSeq{Slot: r.mc3Slot, Pos: pos, Emb: emb, Draw: draw}, solo: solo}
			if !held {
				err = r.mc3.forward(q)
			}
			if !held && err == nil && q.out != nil {
				if draw != nil {
					r.fastNext = q.out.ID
					if !draw.Greedy {
						r.g.DeviceSampled++
					}
				} else {
					r.logits, r.fastNext = q.out.Logits, -1
				}
			}
		} else {
			err = residentCall()
		}
		r.gpuPos++
	} else if r.cb != nil {
		// A batched step's row is bit-identical to this token's own m.forward (decodeMultiStep's contract).
		q := &batchReq{id: next, cache: r.cache, solo: func() error {
			var ferr error
			r.logits, ferr = m.forward(next, r.cache)
			return ferr
		}}
		err = r.cb.forward(q)
		if err == nil && q.out != nil {
			r.logits = q.out.Logits
		}
	} else {
		r.logits, err = m.forward(next, r.cache)
	}
	return err
}

// chooseDecodePath picks this generation's decode path from what the resident offers and the sampler allows: the greedy and sampled fast paths and the chains. It sets the path flags on r and defines stopChain, which generateInto defers.
func (r *genRun) chooseDecodePath() {
	m := r.m
	// Greedy fast path: when the resident can pick the argmax on-device AND the sampler's choice is exactly argmax(raw logits)
	// with nothing else reading them, skip the full-logits readback. Emitted tokens are identical; GOINFER_NO_GREEDY_FASTPATH
	// forces the logits path (escape hatch / A-B check). fastNext >= 0 means "the resident already picked the next token"; the
	// first token still comes from the prefill logits through the sampler.
	//
	// `top_k=1` takes this path too (GreedyEquivalent), at any temperature: monotone scaling preserves ordering and a one-token
	// distribution is deterministic. Both predicates are consulted and neither is widened.
	//
	// The speculative paths are deliberately not affected: they gate on `sp.Temperature <= 0` directly (speculative.go,
	// spec_grammar.go, spec_ngram.go), so `top_k=1` with a temperature stays speculative-ineligible. Making it eligible would be
	// correct (argmax verification reproduces greedy) but is a second behaviour change and must not ride along here.
	//
	// RNG: this path skips the per-token rng.Float64() draw SampleWithInfo would make. That is unobservable: under top_k=1 every
	// step is deterministic, so no later draw's value can depend on the skipped ones and no emitted token can differ. The RNG
	// stream position does advance differently, so nothing may depend on it downstream.
	r.fastNext = -1
	// A GATED processor (SamplingParams.LogitProcessorGate) leaves the fast paths armed: each step
	// then asks the gate, and only a step it opens takes the full-logits path below (needFull).
	r.gated = r.sp.LogitProcessor != nil && r.sp.LogitProcessorGate != nil
	r.procFree = r.sp.LogitProcessor == nil || r.gated
	r.hasGreedy = m.canDecode(outArgmax)
	r.fastGreedy = r.useGPU && r.hasGreedy && r.procFree &&
		(r.sampler.ArgmaxEquivalent() || r.sampler.GreedyEquivalent()) &&
		m.knobs.get(knobNoGreedyFastpath) == ""
	// The greedy chain (GreedyChainAvailable): the same tokens with the next token's forward queued on the device before the host
	// has seen this one. With no processor, or a gated one while its gate is closed (outside the batched decode: a step the gate opens stops the
	// chain and takes the full-logits path below, which the chain would otherwise skip), no adapter, and where the embedding
	// lookup is a plain table row, which the resident's gather reproduces. Under the batched decode it runs only on the tokens this generation
	// decodes alone, with the resident held across them (holdSolo); every other token takes the batcher as before, so fastGreedy
	// stays the path for those.
	r.chainRF = m.resDecode()
	r.useChain = r.useGPU && r.resCost == nil && (r.sp.LogitProcessor == nil || (r.gated && r.mc3 == nil)) && r.lora == nil &&
		(r.sampler.ArgmaxEquivalent() || r.sampler.GreedyEquivalent()) && m.knobs.get(knobNoGreedyFastpath) == "" &&
		m.embedIsTableRow() && r.chainRF.GreedyChainAvailable()
	if r.useChain && r.mc3 == nil {
		r.fastGreedy = false
	}
	r.chainTok = r.mc3 == nil // this token may run the chain: always unbatched, batched only on a held token
	r.chainOpen, r.chainPos = false, 0
	r.stopChain = func() {
		if r.chainOpen {
			r.chainRF.GreedyChainStop()
			r.chainOpen = false
		}
	}
}

// chooseFastPaths picks the optimistic forward, the device top-K and temperature-draw paths and the sampled chain, and sets up the per-step scratch they share.
func (r *genRun) chooseFastPaths() {
	m := r.m
	// Optimistic forward (spec_optfwd.go): sampled decode's (Temperature>0) sibling of the greedy fast path, overlapping the CPU
	// sampler with a speculative Forward instead of skipping it. Excludes the rare overlap with fastGreedy (Temperature>0 AND
	// top_k==1) so the two never drive the same step. GOINFER_NO_OPTFWD forces the plain sequential path (escape hatch / A-B
	// check). Off under the batched decode: optFwdStep drives the resident outside the batcher's exclusive section.
	r.optFwd = r.useGPU && r.mc3 == nil && r.resCost == nil && !r.fastGreedy && !r.useChain && m.optFwdEligible(r.sp) && m.knobs.get(knobNoOptFwd) == ""

	// Device top-K fast path (sampler_topk.go): a filtered sampler (top_k / top_p / min_p at temperature > 0) needs only the K
	// best logits, so the resident reduces the row on-device and the host filters K candidates instead of V. Excluded: anything
	// that needs or rewrites the full row (bias, penalties, logprobs, a LogitProcessor), the greedy and optimistic-forward paths,
	// and backends whose logits are transformed on the host after readback. GOINFER_NO_TOPK_FASTPATH forces the full-row path
	// (escape hatch / A-B check), same convention as GOINFER_NO_GREEDY_FASTPATH.
	r.topKOn = false
	r.topKWidth, r.topKVocab = 0, len(r.logits)
	// Under the batched decode a TopKRow's Full() reads the resident's logits, which a later step of another generation may have overwritten, so
	// the draw is resolved inside the resident call (topKPre / topKPreFull below) while this generation still holds the resident,
	// not at the top of the next iteration. Do not switch it off there: that sends every CUDA top-p request down the
	// full-row path (docs/measurements/topp-regression-2026-09-30.md).
	if r.useGPU && !r.fastGreedy && !r.useChain && !r.optFwd && r.procFree && m.knobs.get(knobNoTopKFastpath) == "" && r.sampler.TopKEligible() {
		if m.canDecode(outTopK) {
			if w, wok := r.sampler.TopKWidth(len(r.logits)); wok {
				r.topKOn, r.topKWidth = true, w
			}
		}
	}
	// Device temperature-only sampling (sampler_gumbel.go): the resident draws the next token by Gumbel-max on-device and returns
	// just the id, reusing the greedy fast path's fastNext mechanism. Same exclusions as the top-K path;
	// GOINFER_NO_SAMPLE_FASTPATH forces the host draw (A/B check, escape hatch).
	r.sampleOn = r.useGPU && !r.fastGreedy && !r.useChain && !r.optFwd && r.procFree && m.knobs.get(knobNoSampleFastpath) == "" &&
		r.sampler.SampleEligible() && m.canDecode(outSample)
	// A greedy generation's batched token asks the step for its argmax id (a Greedy draw) where the resident offers that, instead
	// of its whole logits row; the id is the argmax the sampler would take from that row. Rows run alone keep residentCall's path.
	r.batchGreedy = false
	if r.mc3 != nil && !r.fastGreedy && !r.sampleOn && r.procFree && (r.sampler.ArgmaxEquivalent() || r.sampler.GreedyEquivalent()) {
		r.batchGreedy = residentCapsOf(m.resident).BatchGreedyDraw
	}
	// The sampled chain (SampleChainAvailable): where the device draw serves the token, the same draw with the next token's forward
	// queued on the device first, as the greedy chain does for the argmax. Same exclusions as the greedy chain, and not where
	// ForwardSample itself takes the argmax (a temperature so small 1/T is infinite).

	if r.sampleOn && r.resCost == nil && r.sp.LogitProcessor == nil && r.lora == nil && m.embedIsTableRow() &&
		!math.IsInf(1/r.sp.Temperature, 0) && r.chainRF.SampleChainAvailable() {
		r.sChainRF = r.chainRF
	}
	// Under the batched decode the top-K draw is made inside the resident call (see topKOn above): topKPre is the drawn token when the row
	// held the whole retained set, else topKPreFull says topKFullBuf holds this step's full row, copied before the resident
	// was released. Both are consumed, and cleared, where topKRow is.

	// mc3Logits holds a batched generation's solo-token logits: a resident's Forward returns its one host buffer, reused by every call, and this
	// generation reads the logits after leaving the resident (in its LogitProcessor and its sampler) while another generation's
	// solo token may already be rewriting that buffer.

	if r.optFwd {
		r.optGate = &optFwdGate{}
		r.g.OptFwd = &OptFwdStats{}
	}

}

// bindDecodeExits defines exitDecode and commitResident: the two exits a decode ends through (a batched generation leaves the decoding count; the resident records what its KV now holds).
func (r *genRun) bindDecodeExits() {
	m := r.m
	// commitResident records what the resident KV now holds for this generation (residentCommitIDs). Under the batched decode it binds
	// this generation's slot first, with the resident to itself, and stops counting it as decoding (so no run waits for
	// its token while it commits).
	r.decoding = false
	r.exitDecode = func() {
		if r.decoding {
			r.decoding = false
			r.mc3.exitDecode()
		}
	}
	r.commitResident = func(generated []int) {
		r.stopChain() // every exit commits; nothing queued on the device may outlive the decode loop
		if r.mc3 == nil {
			m.residentCommitIDs(r.prompt, generated, nil, r.lora)
			return
		}
		r.mc3.releaseHold(r.stopChain) // before the exclusive section below, which waits for the resident a hold keeps
		r.exitDecode()
		r.mc3.exclusive(func() {
			if m.residentBind(r.mc3Slot) == nil {
				m.residentCommitIDs(r.prompt, generated, nil, r.lora)
				r.mc3Committed = true
			}
		})
	}
}

// buildTarget assembles the decode target plainDecodeLoop drives from this generation's pick and advance, its held-send grace under the batched decode and the cost counter that counts forwards.
func (r *genRun) buildTarget() {
	r.tg = decodeTarget{logits: func() []float32 { return r.logits }}
	r.tg.pick = r.pick
	if r.mc3 != nil {
		// A kept hold (a chained decode alone under the batched decode) must not keep the resident while the consumer is not reading: a
		// newcomer would wait on this send. Give the consumer a short grace, then let go before blocking.
		r.tg.send = func(next int) bool {
			if !r.mc3.isHolding() {
				return false
			}
			select {
			case r.out <- next:
				return true
			default:
			}
			r.holdGrace.Reset(mc3HoldSendGrace)
			select {
			case r.out <- next:
				r.holdGrace.Stop()
				return true
			case <-r.holdGrace.C:
				r.mc3.releaseHold(r.stopChain)
			case <-r.ctx.Done():
				r.holdGrace.Stop()
			}
			return false
		}
	}
	if r.useGPU {
		r.tg.finish = r.commitResident
	}
	r.tg.advance = r.advance
	// A counted resident decode is one span over the loop: its tokens are the forwards advance ran, one per emitted
	// token.
	r.decoded = 0
	if r.resCost != nil && r.tg.advance != nil {
		adv := r.tg.advance
		r.tg.advance = func(next int, needFull bool, clk *decodeClock) error { r.decoded++; return adv(next, needFull, clk) }
	}
}

// logDecodeTiming prints the per-token decode split when decode timing is on.
func (r *genRun) logDecodeTiming() {
	if decodeTiming && r.clk.n > 0 {
		ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 / float64(r.clk.n) }
		logs.Infof("DECODE TIMING (%d tok, gpu=%v): forward %.1f ms | sample %.2f ms | logitProc %.2f ms | embed %.2f ms /token",
			r.clk.n, r.useGPU, ms(r.clk.fwd), ms(r.clk.sample), ms(r.clk.proc), ms(r.clk.embed))
		d := func(p *int64) float64 { return ms(time.Duration(atomic.SwapInt64(p, 0))) }
		a, ml, hd := d(&dtAttn), d(&dtMLP), d(&dtHead)
		gu, act, ad := d(&dtGU), d(&dtAct), d(&dtActDown)
		qkv, o, aa := d(&dtQKV), d(&dtO), d(&dtAttnAll)
		logs.Infof("DECODE SPLIT: attention %.2f ms | MLP %.2f ms | LM head %.2f ms | residual %.2f ms /token",
			a, ml, hd, ms(r.clk.fwd)-a-ml-hd)
		logs.Infof("DECODE SPLIT MLP: gate+up matmuls %.2f ms | activation %.2f ms | down matmul %.2f ms /token", gu, act, ad-act)
		logs.Infof("DECODE SPLIT ATTN: q/k/v matmuls %.2f ms | rope+KV+scores/softmax/AV core %.2f ms | o matmul %.2f ms /token", qkv, aa-qkv-o, o)
		lastDecodeSplit.fwd, lastDecodeSplit.attn, lastDecodeSplit.mlp, lastDecodeSplit.head = ms(r.clk.fwd), a, ml, hd
		lastDecodeSplit.gu, lastDecodeSplit.act, lastDecodeSplit.down = gu, act, ad-act
		lastDecodeSplit.qkv, lastDecodeSplit.core, lastDecodeSplit.o = qkv, aa-qkv-o, o
	}
}
