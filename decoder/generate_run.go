package decoder

import (
	"fmt"
	"time"
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
			// MC3: submit the token; it runs in a shared step with the other generations' tokens, or — alone — as
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
