package specdecode

import (
	"context"
	"runtime"
	"slices"

	"github.com/townsendmerino/goinfer/internal/runtime/sampling"
)

// Drafter proposes continuation tokens for a context, which the target model verifies in one batched pass.
type Drafter interface {
	Draft(ctx []int, k int) []int
}

// NgramDrafter is a zero-model prompt-lookup drafter.
type NgramDrafter struct {
	MinMatch int // shortest suffix length trusted as a match (default 2)
	MaxMatch int // longest suffix length probed (default 16); caps the scan

	lastMatch   int
	proposalBuf []int
}

// DraftInfo is optional per-Draft metadata a Drafter may expose for SpecTrace.
type DraftInfo struct {
	MatchLen int // n-gram suffix-match length behind the last proposal (0 = miss)
}

// TracingDrafter is the optional capability a Drafter implements to feed instrumentation.
type TracingDrafter interface {
	Drafter
	LastDraftInfo() DraftInfo
}

// LastDraftInfo reports the suffix-match length behind the most recent Draft.
func (d *NgramDrafter) LastDraftInfo() DraftInfo { return DraftInfo{MatchLen: d.lastMatch} }

// Confidence is the router score of the last Draft in [0, 1], and 0 on a miss.
func (d *NgramDrafter) Confidence() float64 {
	if d.lastMatch == 0 {
		return 0
	}
	return NgramAlpha(d.lastMatch)
}

// NgramAlphaAnchors holds the calibrated acceptance probability by suffix match length.
var NgramAlphaAnchors = [...]struct {
	MatchLen int
	Alpha    float64
}{{2, 0.70}, {3, 0.86}, {11, 0.92}, {16, 0.97}}

// NgramAlpha interpolates the calibrated acceptance probability for a suffix match of matchLen.
func NgramAlpha(matchLen int) float64 {
	a := NgramAlphaAnchors
	if matchLen <= a[0].MatchLen {
		return a[0].Alpha
	}
	last := len(a) - 1
	if matchLen >= a[last].MatchLen {
		return a[last].Alpha
	}
	for i := range last {
		if matchLen <= a[i+1].MatchLen {
			lo, hi := a[i], a[i+1]
			f := float64(matchLen-lo.MatchLen) / float64(hi.MatchLen-lo.MatchLen)
			return lo.Alpha + f*(hi.Alpha-lo.Alpha)
		}
	}
	return a[last].Alpha
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
	hi := min(maxM, n-1)
	for L := hi; L >= minM; L-- {
		pat := ctx[n-L:]
		for s := n - L - 1; s >= 0; s-- {
			if !slices.Equal(ctx[s:s+L], pat) {
				continue
			}
			cont := ctx[s+L:]
			m := min(k, len(cont))
			d.lastMatch = L
			d.proposalBuf = append(d.proposalBuf[:0], cont[:m]...)
			return d.proposalBuf
		}
	}
	return nil
}

// RoundDraftWidth clamps a speculative round's draft width k to what remains under ctxCap.
// Returns -1 when there is no room even for cur (pos >= ctxCap).
func RoundDraftWidth(k, pos, ctxCap int) int {
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

// NgramSampler is the narrow sampling interface NgramSpecLoop uses in sampled mode.
type NgramSampler interface {
	DrawTarget(logits []float32, hist []int) int
	DistVector(logits []float32) []float64
	DistVectorHist(logits []float32, hist []int) []float64
	SpecStep(p []float64, draftTok int) (tok int, accepted bool)
}

// NgramTarget is what NgramSpecLoop asks of the target model.
type NgramTarget struct {
	Verify         func(seq []int, base int) ([][]float32, error)
	IDsVerify      func(seq []int, base int) ([]int, error)
	VerifyBatch    func(tok, base int) ([][]float32, error)
	Truncate       func(keep int)
	CommitResident func(generated []int)
	IsStop         func(tok int) bool
	CtxCap         func() int
	Adaptive       bool
	NeverYield     bool
	Exclusive      func(f func())
	Claim          func() bool
	Release        func()
}

func (tg NgramTarget) truncateSync(keep int) {
	if tg.Adaptive {
		tg.Exclusive(func() { tg.Truncate(keep) })
		return
	}
	tg.Truncate(keep)
}

type ngramAdaptiveState struct {
	yielding         bool
	aloneStreak      int
	holdingExclusive bool
	neverYield       bool
}

const specAdaptiveResumeStreak = 8

func (st *ngramAdaptiveState) acquireRound(tg NgramTarget) bool {
	if !tg.Adaptive {
		return false
	}
	if tg.Claim() {
		st.holdingExclusive = true
		st.aloneStreak++
		if st.yielding && st.aloneStreak < specAdaptiveResumeStreak {
			tg.Release()
			st.holdingExclusive = false
			return true
		}
		st.yielding = false
		return false
	}
	if st.neverYield {
		for !tg.Claim() {
			runtime.Gosched()
		}
		st.holdingExclusive = true
		st.aloneStreak++
		st.yielding = false
		return false
	}
	st.aloneStreak = 0
	st.yielding = true
	return true
}

func (st *ngramAdaptiveState) releaseRound(tg NgramTarget) {
	if st.holdingExclusive {
		tg.Release()
		st.holdingExclusive = false
	}
}

func verifyNgramPositions(
	draftTok []int,
	logitsN [][]float32,
	ids []int,
	sampled, needHist bool,
	sampler NgramSampler,
	ph []int,
	tr Tracer,
	round, matchLen int,
) (accepted int, allAccept bool, nextTok int) {
	kEff := len(draftTok)
	allAccept = true
	dist := func(logits []float32, hist []int) []float64 {
		if needHist {
			return sampler.DistVectorHist(logits, hist)
		}
		return sampler.DistVector(logits)
	}
	for i := range kEff {
		var acc bool
		if sampled {
			p := dist(logitsN[i], ph)
			var tok int
			tok, acc = sampler.SpecStep(p, draftTok[i])
			if tr != nil {
				tr(TraceFromDist(round, i, matchLen, draftTok[i], p, acc))
			}
			if !acc {
				nextTok = tok
			}
			if needHist {
				ph = append(ph, draftTok[i])
			}
		} else {
			var ti int
			if ids != nil {
				ti = ids[i]
			} else {
				ti = sampling.Argmax(logitsN[i])
			}
			acc = draftTok[i] == ti
			if tr != nil {
				pTop1, pEnt, pTok := TargetDist(logitsN[i], draftTok[i])
				tr(SpecTrace{
					Step: round, Pos: i, Source: "ngram", Token: draftTok[i],
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
		if sampled {
			nextTok = sampler.DrawTarget(logitsN[kEff], ph)
		} else if ids != nil {
			nextTok = ids[kEff]
		} else {
			nextTok = sampling.Argmax(logitsN[kEff])
		}
	}
	return accepted, allAccept, nextTok
}

// NgramSpecLoop runs n-gram speculative decoding from seedLogits after prompt.
func NgramSpecLoop(ctx context.Context, tg NgramTarget, seedLogits []float32, out chan<- int, errOut *error, stats *SpecStats, drafter Drafter, prompt []int, maxTokens, K int, sampler NgramSampler, needHist bool, tr Tracer, ad *AdaptiveDepth, commit func(int)) {
	sampled := sampler != nil
	tpos := len(prompt)
	idsVerify := tg.IDsVerify

	hist := slices.Clone(prompt)
	cur := sampling.Argmax(seedLogits)
	if sampled {
		cur = sampler.DrawTarget(seedLogits, hist)
	}

	emit := func(tok int) bool {
		if tg.IsStop(tok) {
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
	finishTrailing := func(tok int) {
		if *errOut != nil || stats.Emitted < maxTokens || tg.IsStop(tok) {
			return
		}
		if RoundDraftWidth(0, tpos, tg.CtxCap()) < 0 {
			return
		}
		var err error
		verifyTrailing := func() {
			if idsVerify != nil {
				_, err = idsVerify([]int{tok}, tpos)
			} else {
				_, err = tg.Verify([]int{tok}, tpos)
			}
		}
		if tg.Adaptive {
			tg.Exclusive(verifyTrailing)
		} else {
			verifyTrailing()
		}
		if err != nil {
			return
		}
		tpos++
		hist = append(hist, tok)
		if commit != nil {
			commit(tok)
		}
	}

	if !emit(cur) {
		finishTrailing(cur)
		tg.CommitResident(hist[len(prompt):])
		return
	}
	var lookupBuf, seqBuf, phBuf []int
	adState := ngramAdaptiveState{neverYield: tg.Adaptive && tg.NeverYield}
	if tg.Adaptive {
		defer adState.releaseRound(tg)
	}
	for {
		yieldThisRound := adState.acquireRound(tg)

		lookupBuf = append(lookupBuf[:0], hist...)
		lookupBuf = append(lookupBuf, cur)
		var proposed []int
		if !yieldThisRound && (ad == nil || ad.Depth(K) > 0) {
			proposed = drafter.Draft(lookupBuf, K)
		}
		draftTok := proposed
		if ad != nil && !yieldThisRound {
			draftTok = proposed[:ad.Depth(len(proposed))]
		}
		if w := RoundDraftWidth(len(draftTok), tpos, tg.CtxCap()); w < 0 {
			tg.CommitResident(hist[len(prompt):])
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

		base := tpos
		seqBuf = append(seqBuf[:0], cur)
		seqBuf = append(seqBuf, draftTok...)
		var logitsN [][]float32
		var ids []int
		var err error
		if yieldThisRound {
			logitsN, err = tg.VerifyBatch(cur, base)
		} else {
			verifyRound := func() {
				if idsVerify != nil {
					if ids, err = idsVerify(seqBuf, base); err != nil || len(ids) != len(seqBuf) {
						idsVerify, ids = nil, nil
					}
				}
				if ids == nil {
					logitsN, err = tg.Verify(seqBuf, base)
				}
			}
			if tg.Adaptive {
				tg.Exclusive(verifyRound)
			} else {
				verifyRound()
			}
		}
		adState.releaseRound(tg)
		if err != nil {
			*errOut = err
			return
		}

		stats.Rounds++
		stats.Drafted += kEff
		var ph []int
		if needHist {
			phBuf = append(phBuf[:0], hist...)
			phBuf = append(phBuf, cur)
			ph = phBuf
		}
		accepted, allAccept, nextTok := verifyNgramPositions(
			draftTok, logitsN, ids, sampled, needHist, sampler, ph, tr, stats.Rounds, matchLen,
		)
		stats.Accepted += accepted
		evaluated := accepted
		if !allAccept {
			evaluated = accepted + 1
		}
		stats.Evaluated += evaluated
		if ad != nil {
			ad.Observe(accepted, evaluated)
		}

		if yieldThisRound {
			tpos = base + 1
		} else {
			tpos = base + 1 + accepted
			tg.truncateSync(tpos)
		}

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

		for i := 0; i < accepted; i++ {
			if !emit(draftTok[i]) {
				tg.CommitResident(hist[len(prompt):])
				return
			}
		}
		cur = nextTok
		if !emit(cur) {
			finishTrailing(cur)
			tg.CommitResident(hist[len(prompt):])
			return
		}
	}
}
