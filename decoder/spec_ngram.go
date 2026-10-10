package decoder

import (
	"context"
	"fmt"
	"runtime"
	"slices"
	"sync/atomic"
)

// Drafter is the cheap half of speculative decoding: it proposes continuation tokens for a context, which the target
// model verifies in one batched pass. The target holds the true distribution and decides what commits, so a Drafter
// changes only speed, never output. Design: docs/spec.
type Drafter interface {
	// Draft proposes up to k continuation tokens for the token context ctx
	// (ctx[len-1] is the most recent confirmed token). It returns fewer than k
	// — possibly none — when it has no confident proposal. A miss is cheap: the
	// caller just falls back to one ordinary decode step that round.
	Draft(ctx []int, k int) []int
}

// NgramDrafter is a zero-model "prompt-lookup" drafter: it finds the most recent earlier occurrence of the current
// token suffix within the running context (prompt plus everything generated so far) and proposes the tokens that
// followed it. It costs no model compute and needs no draft model. It earns its keep when the output echoes the input
// (code edits, RAG, agent loops with a fixed system prompt); on novel prose it misses, which is free.
//
// Matching is exact-suffix and greedy-longest: the longest pattern first, the most recent earlier hit. It is a hash-style
// baseline, not a suffix automaton.
type NgramDrafter struct {
	MinMatch int // shortest suffix length trusted as a match (default 2)
	MaxMatch int // longest suffix length probed (default 16); caps the scan

	lastMatch   int   // suffix length of the most recent Draft hit (0 on miss); for SpecTrace
	proposalBuf []int // reused proposal buffer for zero-alloc Draft
}

// DraftInfo is optional per-Draft metadata a Drafter may expose for SpecTrace.
type DraftInfo struct {
	MatchLen int // n-gram suffix-match length behind the last proposal (0 = miss)
}

// TracingDrafter is the optional capability a Drafter implements to feed
// instrumentation. LastDraftInfo is valid only immediately after Draft, and only
// under single-flight use (one in-flight Draft at a time) — which the decode loop
// guarantees per generation. Used by the trace path; never on the hot path.
type TracingDrafter interface {
	Drafter
	LastDraftInfo() DraftInfo
}

// LastDraftInfo reports the suffix-match length behind the most recent Draft.
func (d *NgramDrafter) LastDraftInfo() DraftInfo { return DraftInfo{MatchLen: d.lastMatch} }

// Confidence is the router score of the last Draft: the calibrated acceptance probability ngramAlpha(match length) in
// [0,1], and 0 on a miss. A longer matched suffix is a more specific copy, so it outranks grammar's forced proposal;
// scoring in accept-probability rather than raw match length lets the router compare sources on one scale.
func (d *NgramDrafter) Confidence() float64 {
	if d.lastMatch == 0 {
		return 0
	}
	return ngramAlpha(d.lastMatch)
}

// ngramAlphaAnchors is the calibrated acceptance probability by suffix match length (monotone non-decreasing),
// evaluated by interpolation on the hot path. It was fit offline on copy-heavy traces of one model family
// (TestNgramAlphaPredictor); re-fit it when a new family or workload lands. The measured buckets:
// docs/code-notes/decoder.md#ngramAlphaAnchors.
var ngramAlphaAnchors = [...]struct {
	matchLen int
	alpha    float64
}{{2, 0.70}, {3, 0.86}, {11, 0.92}, {16, 0.97}}

// ngramAlpha interpolates the calibrated acceptance probability for a suffix match of
// the given length (piecewise-linear between anchors, clamped at both ends). Monotone
// non-decreasing by construction (the anchors increase).
func ngramAlpha(matchLen int) float64 {
	a := ngramAlphaAnchors
	if matchLen <= a[0].matchLen {
		return a[0].alpha
	}
	last := len(a) - 1
	if matchLen >= a[last].matchLen {
		return a[last].alpha
	}
	for i := range last {
		if matchLen <= a[i+1].matchLen {
			lo, hi := a[i], a[i+1]
			f := float64(matchLen-lo.matchLen) / float64(hi.matchLen-lo.matchLen)
			return lo.alpha + f*(hi.alpha-lo.alpha)
		}
	}
	return a[last].alpha
}

// Draft implements Drafter via longest-suffix prompt lookup.
func (d *NgramDrafter) Draft(ctx []int, k int) []int {
	d.lastMatch = 0
	minM, maxM := d.MinMatch, d.MaxMatch
	if minM < 1 {
		minM = 2
	}
	if maxM < minM {
		maxM = 16
	}
	n := len(ctx)
	if n < minM+1 || k < 1 {
		return nil
	}
	hi := min(maxM,
		// need at least one earlier token to match against
		n-1)
	for L := hi; L >= minM; L-- {
		pat := ctx[n-L:]
		// Most recent earlier occurrence: the latest start s with s+L <= n-1.
		for s := n - L - 1; s >= 0; s-- {
			if !slices.Equal(ctx[s:s+L], pat) {
				continue
			}
			cont := ctx[s+L:] // tokens that followed this earlier occurrence
			m := min(k, len(cont))
			d.lastMatch = L
			d.proposalBuf = append(d.proposalBuf[:0], cont[:m]...)
			return d.proposalBuf
		}
	}
	return nil
}

// GenerateNgramSpeculative is single-model speculative decoding driven by an n-gram (prompt-lookup) Drafter instead of
// a draft model. Each round the drafter proposes up to K tokens from the running context; the target verifies
// [cur, draft...] in one batched ForwardN, keeps the matching prefix, replaces the first mismatch with its own token,
// and rolls the cache back with TruncateTo.
//
// It is lossless. With Temperature==0 the output is token-identical to plain greedy (TestNgramSpeculativeGreedyParity);
// with Temperature>0 it reproduces the target sampler's distribution by rejection sampling (spec_sample.go). A drafter
// miss degenerates to one plain decode step.
//
// Sampled mode supports temperature, top-k/top-p/min-p and the history-dependent transforms (repetition, presence and
// frequency penalties, LogitBias), threaded per position via distVectorHist. Greedy mode rejects those transforms,
// because argmax cannot apply them losslessly. LogitProcessor (constrained or tool decoding) is unsupported in either
// mode. Generation.Spec carries acceptance telemetry.
func (target *Model) GenerateNgramSpeculative(ctx context.Context, prompt []int, maxTokens int, drafter Drafter, K int, sp SamplingParams) (<-chan int, *Generation, error) {
	return target.genNgram(ctx, prompt, maxTokens, drafter, K, sp, nil, nil)
}

// validateNgramSpec runs the synchronous precondition checks shared by the Model
// and Session speculative entry points.
func validateNgramSpec(m *Model, drafter Drafter, sp SamplingParams) error {
	if drafter == nil {
		return fmt.Errorf("decoder.GenerateNgramSpeculative: nil drafter")
	}
	if sp.LogitProcessor != nil {
		return fmt.Errorf("decoder.GenerateNgramSpeculative: LogitProcessor (constrained/tool decoding) not supported yet; use Generate")
	}
	// Greedy verify is argmax over raw target logits; it cannot apply penalties or logit bias, so honoring them would
	// silently diverge from plain greedy. Sampled mode threads them per position and is fine.
	if sp.Temperature <= 0 && sp.HistoryDependent() {
		return fmt.Errorf("decoder.GenerateNgramSpeculative: repetition penalties / logit bias are not supported in greedy speculative decoding; use Generate")
	}
	if !m.specRollbackSafe() {
		return fmt.Errorf("decoder.GenerateNgramSpeculative: this model has recurrent state (Mamba-2 / Gated DeltaNet / LFM2 conv / KDA) or a staged sliding-window ring cache that speculative rollback cannot losslessly restore; use Generate")
	}
	return nil
}

// genNgram is the async wrapper: it validates, launches genNgramInto in a
// goroutine over a fresh self-owned cache, and returns the stream. Session-backed
// callers use genNgramInto directly to thread their warm KV cache.
func (target *Model) genNgram(ctx context.Context, prompt []int, maxTokens int, drafter Drafter, K int, sp SamplingParams, tr specTracer, ad *AdaptiveDepth) (<-chan int, *Generation, error) {
	if err := validateNgramSpec(target, drafter, sp); err != nil {
		return nil, nil, err
	}
	// Here, not in validateNgramSpec: a Session shares that validator but verifies on its CPU cache,
	// which never reaches the resident and so cannot see a resident divergence. This entry takes
	// the resident path whenever it is eligible, so it is the one that must refuse.
	if err := target.SpecDecodeConflict(); err != nil {
		return nil, nil, fmt.Errorf("decoder.GenerateNgramSpeculative: %w; use Generate", err)
	}
	if len(prompt) == 0 {
		return nil, nil, fmt.Errorf("decoder.GenerateNgramSpeculative: empty prompt")
	}
	out := make(chan int)
	stats := &SpecStats{}
	g := &Generation{Spec: stats}
	leaveExact := target.enterExactAttention() // decode and verify must share one attention tree
	go func() {
		defer close(out)
		defer leaveExact()
		target.genNgramInto(ctx, out, g, stats, drafter, prompt, 0, maxTokens, K, sp, tr, ad, nil, nil)
	}()
	return out, g, nil
}

// specRoundDraftWidth clamps a speculative round's draft width k (verified with cur at absolute position pos as a
// [1+k]-row batch) to what remains under the resident context cap ctxCap, so a verify never makes checkCap refuse the
// whole round. genNgramInto and GenerateSpeculative share it; blockSpecRoundWidth makes the same clamp for block
// speculation. It returns k unchanged when ctxCap <= 0 or there is room, and -1 when there is no room even for cur
// (pos >= ctxCap): the caller's signal to stop cleanly.
func specRoundDraftWidth(k, pos, ctxCap int) int {
	if ctxCap <= 0 {
		return k
	}
	room := ctxCap - pos - 1
	if room < 0 {
		return -1
	}
	if room < k {
		return room
	}
	return k
}

// genNgramInto is the synchronous speculative decode loop, mirroring generateInto: it assumes cache already holds
// prompt[:prefillFrom] (a Session's warm prefix; 0 and a nil cache for a fresh run), prefills the rest, then runs the
// n-gram verify loop streaming each committed token to out. commit, if non-nil, is invoked with each token as its
// forward commits it to the cache (the Session seam); the final pending token is emitted but not committed, exactly
// one behind the cache, so reconciliation matches the cache contents. It does not close out. When cache is non-nil the
// device-resident path is bypassed (sessions use the CPU cache, as generateInto does); tr, if non-nil, receives
// SpecTrace events.
func (target *Model) genNgramInto(ctx context.Context, out chan<- int, g *Generation, stats *SpecStats, drafter Drafter, prompt []int, prefillFrom, maxTokens, K int, sp SamplingParams, tr specTracer, ad *AdaptiveDepth, cache *KVCache, commit func(int)) {
	if len(prompt) == 0 {
		g.err = fmt.Errorf("decoder.GenerateNgramSpeculative: empty prompt")
		return
	}
	if K < 1 {
		K = 4
	}
	sampled := sp.Temperature > 0
	var sampler *Sampler
	needHist := false
	if sampled {
		sampler = NewSampler(sp)
		needHist = sampler.needsHistory() // penalties / logit bias ⇒ thread per-position history
	}
	// dist is the target's next-token sampling distribution at one position. With a
	// history-dependent transform (penalties/bias) it is computed over `ph` (the
	// tokens before this position); otherwise the cheap history-free path. Returns
	// nil in greedy mode (callers use argmax there).
	dist := func(logits []float32, ph []int) []float64 {
		if needHist {
			return sampler.distVectorHist(logits, ph)
		}
		return sampler.distVector(logits)
	}

	// Resident verify on the device when the target has a built GPU-resident decode path and no session cache was passed:
	// the prompt seeds the resident KV, verify is one batched ForwardN, and rollback is the resident no-op TruncateTo.
	// The drafter is pure Go, so this adds no GPU memory. A session's CPU cache forces the staged path.
	resident := cache == nil && target.resident != nil && target.DecodeRunnerEligible()
	// adaptive claims the resident per round instead of once per generation, so MC3's batcher can admit other generations
	// between rounds ("speculate when alone, batch under load"; docs/tasks/task-concurrency-2026-09.md). Greedy only: a
	// sampled generation takes the always-exclusive path, because the yield round needs a device-side greedy verify through
	// the batcher, which does not exist for a sampled draw.
	adaptive := resident && target.specAdaptive && target.batcher != nil && !sampled
	mc3Slot := -1
	if resident && !adaptive {
		// The resident path drives the model's one shared positional KV; two concurrent generations would interleave writes at
		// overlapping positions and corrupt it. Claim it without blocking, as generateInto does: a loser falls back to the
		// staged CPU path on this call's own cache. Released on return.
		//
		// Claim through the batcher's claimExclusive when one exists, not a bare CAS on resBusy: a plain concurrent generation
		// becomes an MC3 holder via bt.claim, which only reads resBusy, so a bare CAS would succeed while a holder is mid-step
		// and corrupt the resident. claimExclusive also requires holders == 0. Production does not combine the two (-spec
		// without -spec-adaptive forces concurrency to 1), but the guard must not depend on that;
		// TestSpecAdaptiveSwitch_offByDefaultUnaffected exercises the combination.
		claimed := false
		if target.batcher != nil {
			claimed = target.batcher.claimExclusive(&target.resBusy)
		} else {
			claimed = atomic.CompareAndSwapInt32(&target.resBusy, 0, 1)
		}
		if claimed {
			defer atomic.StoreInt32(&target.resBusy, 0)
		} else {
			resident = false
		}
	}
	{
		tc := cache
		if tc == nil && !resident {
			tc = target.NewCache(len(prompt) + maxTokens + K + 8)
		}

		tpos := 0
		targetVerify := func(seq []int, base int) ([][]float32, error) {
			if resident {
				if adaptive {
					// Another generation may have bound a different slot since our last round; rebind
					// ours before touching the resident (a no-op when we are already bound).
					if err := target.residentBind(mc3Slot); err != nil {
						return nil, err
					}
				}
				embs := make([][]float32, len(seq))
				for i, tok := range seq {
					embs[i] = target.embedResident(tok)
				}
				return target.resident.ForwardN(embs, base)
			}
			return target.forwardN(ctx, seq, tc)
		}
		// targetVerifyBatch is targetVerify's adaptive "batch under load" side: one token through MC3's coalescer on our own
		// reserved slot. A batched step has no argmax-only fast path, so it always returns one full logits row; greedy only,
		// as adaptive is.
		targetVerifyBatch := func(tok, base int) ([][]float32, error) {
			emb := target.embedResident(tok)
			var row []float32
			q := &batchReq{seq: ResidentBatchSeq{Slot: mc3Slot, Pos: base, Emb: emb}, solo: func() error {
				if err := target.residentBind(mc3Slot); err != nil {
					return err
				}
				var ferr error
				row, ferr = target.resident.Forward(emb, base)
				return ferr
			}}
			if err := target.batcher.forward(q); err != nil {
				return nil, err
			}
			if q.out != nil {
				row = q.out.Logits
			}
			return [][]float32{row}, nil
		}
		// Argmax-only verify (greedy, resident, no tracer): the accept decision only compares draft tokens with the target's
		// argmax, so reading back and arg-maxing a full logits row per verified row is overhead. A one-row round takes the
		// device-argmax fast path Model.Generate uses (ResidentGreedy); a multi-row round takes the batched argmax pass block
		// speculation uses (PrefillLastNArgmax). The ids equal the logits path's argmax (TestPrefillLastNArgmax_matchesPerRow),
		// so output is unchanged. Any error falls back once to the full-logits ForwardN and turns this path off, so a backend
		// whose batched pass declines (MoE) keeps working. nil when not applicable.
		var idsVerify func(seq []int, base int) ([]int, error)
		if resident && !sampled && tr == nil {
			rg, hasG := target.resident.(ResidentGreedy)
			av, hasAV := target.resident.(interface {
				PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error)
			})
			if hasG || hasAV {
				idsVerify = func(seq []int, base int) ([]int, error) {
					if adaptive {
						if err := target.residentBind(mc3Slot); err != nil {
							return nil, err
						}
					}
					if len(seq) == 1 && hasG {
						id, e := rg.ForwardArgmax(target.embedResident(seq[0]), base)
						return []int{id}, e
					}
					if !hasAV {
						return nil, fmt.Errorf("decoder: no argmax-only batched verify on this resident")
					}
					embs := make([][]float32, len(seq))
					for i, tok := range seq {
						embs[i] = target.embedResident(tok)
					}
					return av.PrefillLastNArgmax(embs, base)
				}
			}
		}
		targetTruncate := func(keep int) {
			tpos = keep
			if resident {
				if adaptive {
					if err := target.residentBind(mc3Slot); err != nil {
						// TruncateTo has no error return; a rebind failure here would surface as the next
						// verify call's own error instead (it rebinds too), so this is not silently lost.
						_ = err
					}
				}
				target.resident.TruncateTo(keep)
			} else {
				tc.TruncateTo(keep)
			}
		}

		// Prefill the (divergent suffix of the) prompt; the last token's logits seed cur.
		var seedLogits []float32
		var err error
		if resident {
			// Shared with generateInto (residentPrefillSeed), so batched prefill applies here too.
			//
			// Prefix reuse: residentAcquire and residentAcquireSlot return the reusable prefix reuseFrom, so a --spec or --drafter
			// agent loop does not cold-prefill the whole prompt every round. resIDs is accurate here because commitResident below
			// keeps it so. Compute reuseFrom first, then forget the recorded ids: verify writes positions this function owns, so
			// the list stops describing the resident past this point, and the next turn's commit makes it true again. Forgetting
			// first would only lose reuse. generateInto uses the same order (resident_reuse.go).
			var reuseFrom int
			if adaptive {
				// Our own slot, reserved for the whole generation: busySlots keeps this pick from colliding with another generation's.
				// residentAcquireSlot rebinds the resident to the slot it picks, so the pick and the forget below need the resident to
				// themselves, as generateInto's MC3 prefill does (model.go); neither is safe bare.
				target.batcher.exclusive(func() {
					reuseFrom, mc3Slot = target.residentAcquireSlot(prompt, nil, nil, target.batcher.busySlots())
					if mc3Slot >= 0 {
						target.batcher.markSlot(mc3Slot, true)
						target.residentForgetIDs()
					}
				})
				if mc3Slot < 0 {
					// No free slot: not resident for this call; fall back to the staged CPU path, as the claim above does.
					resident, adaptive = false, false
				}
			}
			if resident && adaptive {
				target.batcher.prefillExclusive(func() {
					// Re-bind: the slot-acquisition section above has ended, so another generation could have rebound the resident to a
					// different slot since (mc3Prefill, model.go, rebinds for the same reason). Skipping this made the prefill write
					// another generation's slot, intermittently (TestSpecAdaptiveSwitch_matchesPlainDecode).
					if err = target.residentBind(mc3Slot); err != nil {
						return
					}
					seedLogits, err = target.residentPrefillSeed(ctx, prompt, reuseFrom, false)
				})
				if err != nil {
					target.batcher.markSlot(mc3Slot, false)
					g.err = err
					return
				}
				g.PrefillReused = reuseFrom
			} else if resident {
				reuseFrom = target.residentAcquire(prompt, nil, nil) // the n-gram target never binds an adapter; MC1: binds the KV slot
				target.residentForgetIDs()
				if seedLogits, err = target.residentPrefillSeed(ctx, prompt, reuseFrom, false); err != nil {
					g.err = err
					return
				}
				g.PrefillReused = reuseFrom
			} else {
				// Reached when adaptive's slot acquisition above found no free slot: tc was never
				// allocated (resident was true when it was sized), so allocate it now, same fallback
				// the non-resident branch below would have taken from the start.
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
		tpos = len(prompt)

		// hist is the confirmed context the drafter searches: the prompt plus every
		// committed token. cur is the next confirmed token (not yet in hist), exactly
		// like plain decode's pending-next-token invariant.
		hist := slices.Clone(prompt)
		// The seed token is the target's own first draw: argmax (greedy) or a sample
		// from the seed distribution (sampled) — exactly what plain decode emits first.
		cur := argmax(seedLogits)
		if sampled {
			cur = sampler.drawTarget(seedLogits, hist) // seed history = prompt
		}

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
		// commitResident records the accepted sequence in the resident so the next turn can reuse its prefix. Every return
		// that follows a successful, stop or cancelled emit calls it, at a point where hist is exactly what the resident holds,
		// never ahead: targetVerify writes a round's [cur, accepted draft...] K/V before this loop streams any of them, and
		// hist is appended in the same step. A targetVerify or prefill error does not call it: the partial write must stay
		// forgotten. The round's trailing token is streamed but forwarded only at the start of the next round, so a return
		// right after that emit leaves hist one token behind (safe; finishTrailing forwards it when the generation ended at
		// maxTokens). Recurrent families never reach here (validateNgramSpec rejects them).
		commitResident := func() {
			if !resident {
				return
			}
			if adaptive {
				// The generation's own resident touches are over by the time this runs (the round loop
				// has returned); MC3 may be mid-run for other generations right now, so rebinding our
				// slot needs the same momentary exclusive access their own prefills take, not a bare
				// unsynchronized call. mc3Slot is only ever valid (>= 0) when adaptive.
				target.batcher.exclusive(func() {
					if err := target.residentBind(mc3Slot); err == nil {
						target.residentCommitIDs(prompt, hist[len(prompt):], nil, nil)
					}
				})
				target.batcher.markSlot(mc3Slot, false)
			} else {
				// The whole-generation exclusive claim means our slot is already the bound one; no residentBind needed.
				target.residentCommitIDs(prompt, hist[len(prompt):], nil, nil)
			}
		}
		// finishTrailing forwards the round's trailing token when the generation ended by reaching maxTokens just after
		// streaming it. Plain decode forwards every token it emits, so its cache holds the prompt plus every emitted token;
		// here a trailing token is otherwise forwarded only as the next round's seq[0], which leaves the cache one short. A
		// short cache makes the next turn re-prefill that position, which changes its output on a backend whose batched
		// prefill is not bit-identical to decode (Metal's f16-MMA prefill past its floor;
		// docs/measurements/spec-vs-batching-metal-2026-09-27.md §4). The forward is the one a one-row round makes; a stop, a
		// cancel or a full context takes none, as in plain decode.
		finishTrailing := func(tok int) {
			if g.err != nil || stats.Emitted < maxTokens || target.isStop(tok, sp) {
				return
			}
			if specRoundDraftWidth(0, tpos, target.ResidentContextCap()) < 0 {
				return
			}
			var err error
			verifyTrailing := func() {
				if idsVerify != nil {
					_, err = idsVerify([]int{tok}, tpos)
				} else {
					_, err = targetVerify([]int{tok}, tpos)
				}
			}
			if adaptive {
				// A one-off, rare call (only when the generation ended exactly at maxTokens): the same
				// momentary exclusive access commitResident takes, not the per-round claimExclusive dance.
				target.batcher.exclusive(verifyTrailing)
			} else {
				verifyTrailing()
			}
			if err != nil {
				return // the cache stays one token short, which is consistent (hist is not advanced)
			}
			tpos++
			hist = append(hist, tok)
			if commit != nil {
				commit(tok)
			}
		}

		if !emit(cur) {
			finishTrailing(cur)
			commitResident()
			return
		}
		var lookupBuf []int
		var seqBuf []int
		var phBuf []int
		// Adaptive state. yielding: the previous round joined MC3's batch instead of speculating. aloneStreak: consecutive
		// rounds this generation was alone; resuming multi-token drafting after a yield takes specAdaptiveResumeStreak of them
		// (hysteresis, so a briefly idle second client does not thrash the switch), while a generation that never yielded keeps
		// drafting every alone round. holdingExclusive tracks whether resBusy is ours, so the safety-net defer below cannot
		// leave it stuck on an error return; the per-round release further down is the fast path.
		const specAdaptiveResumeStreak = 8
		yielding, aloneStreak := false, 0
		holdingExclusive := false
		// neverYield makes this generation wait for exclusive access rather than ever yielding, reproducing plain -spec ngram
		// (one at a time) with -spec-adaptive on. A measurement control, not a shipped policy; see knobSpecAdaptiveNeverYield
		// (docs/tasks/task-concurrency-2026-09.md).
		neverYield := adaptive && target.knobs.get(knobSpecAdaptiveNeverYield) != ""
		if adaptive {
			defer func() {
				if holdingExclusive {
					atomic.StoreInt32(&target.resBusy, 0)
				}
			}()
		}
		for {
			yieldThisRound := false
			if adaptive {
				if target.batcher.claimExclusive(&target.resBusy) {
					holdingExclusive = true
					aloneStreak++
					if yielding && aloneStreak < specAdaptiveResumeStreak {
						atomic.StoreInt32(&target.resBusy, 0)
						holdingExclusive = false
						yieldThisRound = true
					} else {
						yielding = false
					}
				} else if neverYield {
					// Measurement only: wait for the resident rather than joining MC3's batch. A short,
					// bounded spin — claimExclusive is a cheap non-blocking check, and this generation is
					// the only thing in this process choosing to wait this way.
					for !target.batcher.claimExclusive(&target.resBusy) {
						runtime.Gosched()
					}
					holdingExclusive = true
					aloneStreak++
					yielding = false
				} else {
					aloneStreak = 0
					yielding = true
					yieldThisRound = true
				}
			}
			// 1. Draft up to K tokens from the context ending at cur (zero on a miss).
			lookupBuf = append(lookupBuf[:0], hist...)
			lookupBuf = append(lookupBuf, cur)
			// When the depth controller is at 0 for any proposal (its acceptance estimate says even one node is not worth it and no
			// probe is due), the round is a plain decode step, so the drafter's scan of the whole context is skipped.
			var proposed []int
			if !yieldThisRound && (ad == nil || ad.Depth(K) > 0) {
				// A yield round never drafts: it is one plain token through MC3's batch, exactly what a
				// kEff==0 miss round already is below, just verified via the batcher instead of this
				// generation's own exclusive resident access.
				proposed = drafter.Draft(lookupBuf, K)
			}
			// Fixed K verifies the whole proposal; the adaptive controller trims it to
			// the depth its running acceptance estimate still justifies (04).
			draftTok := proposed
			if ad != nil && !yieldThisRound {
				draftTok = proposed[:ad.Depth(len(proposed))]
			}
			// Trim the draft so verifying [cur, draft...] never writes past the resident context cap: verifying past it makes
			// checkCap refuse the whole round with a hard error instead of finishing cleanly like plain decode. Applied after the
			// adaptive-depth clamp, only tightening it. w < 0 means no room even for cur: stop without a verify round.
			if w := specRoundDraftWidth(len(draftTok), tpos, target.ResidentContextCap()); w < 0 {
				commitResident()
				return
			} else if w != len(draftTok) {
				draftTok = draftTok[:w]
			}
			kEff := len(draftTok)
			matchLen := 0
			if tr != nil {
				if td, ok := drafter.(TracingDrafter); ok {
					matchLen = td.LastDraftInfo().MatchLen
				}
			}

			// 2. Verify: one target pass over [cur, draft…] gives the target's argmax
			// after each position in a single weight stream.
			base := tpos
			seqBuf = append(seqBuf[:0], cur)
			seqBuf = append(seqBuf, draftTok...)
			var logitsN [][]float32
			var ids []int
			var err error
			if yieldThisRound {
				// The batch-under-load side of the switch: one token, on the shared decode step, not this
				// generation's own exclusive access. seqBuf == [cur] here (draftTok is nil above), so this
				// is exactly the kEff==0 round the non-adaptive path below already handles, just routed
				// through the batcher instead of a direct resident call.
				logitsN, err = targetVerifyBatch(cur, base)
			} else {
				verifyRound := func() {
					if idsVerify != nil {
						if ids, err = idsVerify(seqBuf, base); err != nil || len(ids) != len(seqBuf) {
							idsVerify, ids = nil, nil // fall back, permanently, to the full-logits path below
						}
					}
					if ids == nil {
						logitsN, err = targetVerify(seqBuf, base)
					}
				}
				if adaptive {
					// resBusy (claimExclusive above) keeps MC3's holders off the resident, but other generations' exclusive sections (slot
					// pick, prefill, commit) wait on the batcher's busy flag and never read resBusy. Verifying without busy let their
					// residentBind-then-prefill interleave with this round's residentBind-then-ForwardN, and each wrote the other's slot.
					// The round therefore takes busy too (docs/measurements/mc4-candidate-cuda-2026-10-01.md).
					target.batcher.exclusive(verifyRound)
				} else {
					verifyRound()
				}
			}
			if holdingExclusive {
				// Held only for this round's verify (adaptive) — everything from here on is host-side
				// bookkeeping and a channel send, not a resident touch.
				atomic.StoreInt32(&target.resBusy, 0)
				holdingExclusive = false
			}
			if err != nil {
				g.err = err
				return
			}

			// 3. Verify each draft position. Greedy: accept while draftTok[i] equals the
			// target's argmax, replacing the first mismatch with the target's token.
			// Sampled: accept draftTok[i] with probability p_i(draftTok[i]) (point-mass
			// q ⇒ accept rate p(x)); on reject, the correction is drawn from the residual
			// (p_i with draftTok[i] removed). Both are lossless; the first reject stops
			// the chain (positions past it are conditioned on a token we didn't take).
			stats.Rounds++
			stats.Drafted += kEff
			accepted := 0
			allAccept := true
			var nextTok int
			// ph is the penalty/bias history up to the current position: prompt +
			// committed + cur, growing by each accepted draft token (nil when no
			// history-dependent transform is active).
			var ph []int
			if needHist {
				phBuf = append(phBuf[:0], hist...)
				phBuf = append(phBuf, cur)
				ph = phBuf
			}
			for i := range kEff {
				var acc bool
				if sampled {
					p := dist(logitsN[i], ph)
					var tok int
					tok, acc = sampler.specStep(p, draftTok[i])
					if tr != nil {
						tr(traceFromDist(stats.Rounds, i, matchLen, draftTok[i], p, acc))
					}
					if !acc {
						nextTok = tok
					}
					if needHist {
						ph = append(ph, draftTok[i]) // advance to the next position's history
					}
				} else {
					var ti int
					if ids != nil {
						ti = ids[i]
					} else {
						ti = argmax(logitsN[i])
					}
					acc = draftTok[i] == ti
					if tr != nil {
						pTop1, pEnt, pTok := targetDist(logitsN[i], draftTok[i])
						tr(SpecTrace{
							Step: stats.Rounds, Pos: i, Source: "ngram", Token: draftTok[i],
							NgramMatch: matchLen, Streak: i, QTop1: 1,
							PTop1: pTop1, PEntropy: pEnt, TV: 1 - pTok, AcceptProb: pTok, Accepted: acc,
						})
					}
					if !acc {
						nextTok = ti
					}
				}
				if acc {
					accepted++
					continue
				}
				allAccept = false
				break
			}
			if allAccept {
				// Bonus token from the position after the last accepted draft — the
				// target's own draw (argmax / sample). Also covers the kEff==0 miss.
				// ph here is prompt+committed+cur+draft[:kEff] — the bonus position's
				// history.
				if sampled {
					nextTok = sampler.drawTarget(logitsN[kEff], ph)
				} else {
					if ids != nil {
						nextTok = ids[kEff]
					} else {
						nextTok = argmax(logitsN[kEff])
					}
				}
			}
			stats.Accepted += accepted

			// evaluated = positions actually checked (accepts plus the one rejection,
			// if any) — the denominator that makes EvalAcceptanceRate comparable to α̅,
			// and the depth controller's observation.
			evaluated := accepted
			if !allAccept {
				evaluated = accepted + 1
			}
			stats.Evaluated += evaluated
			if ad != nil {
				ad.Observe(accepted, evaluated)
			}

			// 4. Roll the cache back to the confirmed length: cur (at base) plus the
			// accepted draft tokens. nextTok stays pending for the next round.
			if yieldThisRound {
				// A yield round drafted nothing (kEff==0, accepted==0): the resident already holds
				// exactly base+1 positions (what targetVerifyBatch just wrote), so there is nothing to
				// roll back. Skip the resident touch — it would need its own synchronization here, and
				// buys nothing — and just advance the host-side position tracker targetTruncate also sets.
				tpos = base + 1
			} else {
				targetTruncate(base + 1 + accepted)
			}

			// 5. Commit to history: cur, then the accepted draft tokens. nextTok is
			// the new cur and is committed at the top of the next round. The same
			// tokens (now resident in the cache at positions base..base+accepted) are
			// reported to the Session via commit, so its token list mirrors the cache.
			hist = append(hist, cur)
			if commit != nil {
				commit(cur)
			}
			for i := 0; i < accepted; i++ {
				hist = append(hist, draftTok[i])
				if commit != nil {
					commit(draftTok[i])
				}
			}

			// 6. Stream the accepted draft tokens, then the correction/bonus.
			for i := 0; i < accepted; i++ {
				if !emit(draftTok[i]) {
					commitResident()
					return
				}
			}
			cur = nextTok
			if !emit(cur) {
				finishTrailing(cur)
				commitResident()
				return
			}
		}
	}
}
