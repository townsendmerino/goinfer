package decoder

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sort"
)

// SamplingParams controls next-token selection. Zero value (Temperature 0)
// is greedy/argmax — deterministic and the right default for parity tests.
type SamplingParams struct {
	Temperature float64 // 0 = greedy; >0 scales logits before softmax
	TopK        int     // 0 = disabled; keep the K highest-prob tokens
	TopP        float64 // 0 = disabled; nucleus — smallest set with cumprob ≥ TopP
	MinP        float64 // 0 = disabled; keep tokens with prob ≥ MinP·maxProb (a relative floor — the now-standard filter)
	Seed        int64   // RNG seed for reproducible sampling

	// Repetition controls discourage repeats by adjusting the logits of tokens
	// seen in the last RepeatLastN tokens (prompt + output — the generation loop
	// seeds the prompt). All three compose.
	//   RepeatPenalty    llama.cpp style: a positive logit is divided by it
	//                    (negative logits multiplied). 0 or 1 = disabled.
	//   PresencePenalty  OpenAI style: flat subtraction if the token is present.
	//   FrequencyPenalty OpenAI style: subtraction proportional to its count.
	RepeatPenalty    float64
	PresencePenalty  float64
	FrequencyPenalty float64
	RepeatLastN      int // penalty window: 0 (or <0) = whole history, n = last n tokens

	// LogitBias adds a per-token offset to that token's logit before sampling
	// (large positive to force, large negative to ban). Keyed by token id.
	LogitBias map[int]float32

	// Logprobs makes SampleWithInfo / Generate report the chosen token's
	// log-probability; TopLogprobs additionally reports that many
	// highest-probability alternatives. Reported over the full-vocab softmax
	// after bias+penalties, at the sampling temperature (1 when greedy).
	Logprobs    bool
	TopLogprobs int

	StopIDs []int // extra ids that end generation (besides config EOS), e.g. <end_of_turn>
	// LogitProcessor, if non-nil, is called each decode step with the ids
	// generated so far and that step's logits, which it may modify in place
	// before the sampler runs — e.g. masking disallowed tokens to -inf for
	// constrained/structured decoding (see the constrain package). It runs after
	// the forward pass and before sampling and the stop check, so a constraint
	// can also gate EOS (mask it until the output is a complete document).
	LogitProcessor func(generated []int, logits []float32)
	// LogitProcessorGate, when set with LogitProcessor, makes the processor lazy: it is asked after every emitted token
	// (with the ids generated so far) whether the next step needs the processor. While it answers false the processor is
	// not called and the step keeps every on-device fast path (greedy argmax, device sampling, device top-K) as if there
	// were no processor; a step it answers true for reads the full logits and runs the processor. It is for a constraint
	// that switches on partway through a turn (a tool call under tool_choice auto, constrain.LazyMasker): an ungated
	// processor costs every token of the turn its fast path, prose included
	// (docs/measurements/tool-union-2026-09-24.md). The gate must be a pure function of generated ids. It is consulted
	// only on the decode loop in model.go; the speculative paths still refuse any LogitProcessor, gated or not, and
	// callers fall back to Generate.
	LogitProcessorGate func(generated []int) bool
}

// TokenLogprob is a token id paired with its log-probability.
type TokenLogprob struct {
	ID      int
	Logprob float64
}

// SampleInfo is a sample's result: the chosen id, plus — when
// SamplingParams.Logprobs is set — its log-probability and the top alternatives.
type SampleInfo struct {
	ID      int
	Logprob float64        // log P(ID); 0 unless Logprobs is set
	Top     []TokenLogprob // up to TopLogprobs entries, prob-descending; nil unless requested
}

// Sampler turns a logit vector into a token id. It owns its RNG so a run is
// reproducible from Seed, and tracks the token history the penalties read.
type Sampler struct {
	p       SamplingParams
	rng     *rand.Rand
	gseed   uint64 // Philox key for the temperature-only Gumbel-max draw (sampler_gumbel.go)
	gdraw   uint64 // index of the next such draw: the counter, so the stream is (seed, step, logits) only
	history []int  // tokens seen so far: prompt (via Observe) + each drawn token

	// Scratch, each [vocab]-sized and grown on demand by the *BufN helpers. Every buffer is consumed synchronously by the
	// call that requested it and never retained, which is what makes reuse safe.
	vocabBuf      []float64     // shared by sampleChunked's e and chunkedZ's tmp, which are never live together (Sample's branches are exclusive per call)
	distBuf       []float64     // distVectorFrom's returned distribution; each speculative verify position consumes it before the next, so one covers a round
	specLogitsBuf []float32     // distVectorHist's history-dependent copy; same lifetime as distBuf
	workBuf       []float32     // SampleWithInfo's bias/penalty-mutated copy of the logits
	candBuf       []int         // topFilterLogits' candidate-id list in its no-filter case
	ipsBuf        []indexedProb // topFilterLogits' indexed-probability pairs

	// histCounts is the per-id occurrence count over the whole of history, maintained incrementally by recordHistory so
	// applyPenalties' unbounded-window case never rescans history. Every history mutation MUST go through recordHistory,
	// or this desyncs from history.
	histCounts map[int]int
}

// NewSampler builds a sampler from params.
func NewSampler(p SamplingParams) *Sampler {
	// MinP > 1 asks for tokens with probability >= 1.2x the maximum, which is nothing: the threshold maxL + T*ln(minP) sits
	// above maxL, the candidate set comes back empty, and the keep-the-top-token clamps would index an empty slice (a
	// panic in the Generate goroutine). Clamp to 1.0 rather than reject: SamplingParams has no error return here, every
	// other degenerate value is clamped or ignored, and 1.0 is the identity for min-p (keep only what ties the max).
	if p.MinP > 1 {
		p.MinP = 1
	}
	return &Sampler{p: p, rng: rand.New(rand.NewSource(p.Seed)), gseed: uint64(p.Seed)}
}

// vocabBufN returns a length-n float64 scratch buffer, reusing the backing array when it is large enough and growing
// it once otherwise.
func (s *Sampler) vocabBufN(n int) []float64 {
	if cap(s.vocabBuf) < n {
		s.vocabBuf = make([]float64, n)
	}
	return s.vocabBuf[:n]
}

// distBufN is vocabBufN's counterpart for distBuf.
func (s *Sampler) distBufN(n int) []float64 {
	if cap(s.distBuf) < n {
		s.distBuf = make([]float64, n)
	}
	return s.distBuf[:n]
}

// specLogitsBufN is vocabBufN's counterpart for specLogitsBuf, a []float32.
func (s *Sampler) specLogitsBufN(n int) []float32 {
	if cap(s.specLogitsBuf) < n {
		s.specLogitsBuf = make([]float32, n)
	}
	return s.specLogitsBuf[:n]
}

// workBufN is vocabBufN's counterpart for workBuf, a []float32.
func (s *Sampler) workBufN(n int) []float32 {
	if cap(s.workBuf) < n {
		s.workBuf = make([]float32, n)
	}
	return s.workBuf[:n]
}

// candBufN is vocabBufN's counterpart for candBuf.
func (s *Sampler) candBufN(n int) []int {
	if cap(s.candBuf) < n {
		s.candBuf = make([]int, n)
	}
	return s.candBuf[:n]
}

// ipsBufN is vocabBufN's counterpart for ipsBuf.
func (s *Sampler) ipsBufN(n int) []indexedProb {
	if cap(s.ipsBuf) < n {
		s.ipsBuf = make([]indexedProb, n)
	}
	return s.ipsBuf[:n]
}

// Observe seeds the penalty history with already-seen tokens (the generation
// loop calls it once with the prompt) so repetition penalties consider them.
// Each Sample then records the token it draws.
func (s *Sampler) Observe(ids ...int) {
	for _, id := range ids {
		s.recordHistory(id)
	}
}

// recordHistory appends id to history and keeps histCounts in sync. Every history mutation must go through it:
// appending to s.history directly would silently desync histCounts from the counts applyPenalties' fast path reads.
func (s *Sampler) recordHistory(id int) {
	s.history = append(s.history, id)
	if s.penaltiesConfigured() && s.p.RepeatLastN <= 0 {
		if s.histCounts == nil {
			s.histCounts = map[int]int{}
		}
		s.histCounts[id]++
	}
}

// Sample returns the chosen token id for the given logits ([VocabSize]).
//
//   - Temperature ≤ 0 is greedy (argmax) — deterministic, ignores the filters.
//   - Temperature > 0: softmax at that temperature, optionally restricted to
//     top-k ∩ top-p ∩ min-p, then a multinomial draw from the seeded RNG.
//
// LogitBias and the repetition/presence/frequency penalties adjust the logits
// first (on a copy — the caller's slice is untouched when any apply).
func (s *Sampler) Sample(logits []float32) (int, error) {
	info, err := s.SampleWithInfo(logits)
	return info.ID, err
}

// ArgmaxEquivalent reports whether this sampler's pick is exactly argmax(raw logits): no logit bias, no penalties, no
// temperature or filtering, and no logprob reporting that would need the distribution. When true (and
// SamplingParams.LogitProcessor is nil), a backend that can compute the argmax on-device may return just the id and
// skip the full-logits readback; the emitted tokens are identical.
//
// This and GreedyEquivalent live next to every logit-touching parameter on purpose: a new parameter that mutates or
// reads logits must be reflected in both, so a fast path can never silently diverge. They are separate predicates,
// not one widened one, so call sites choose which notion of "deterministic" they need (see GreedyEquivalent).
func (s *Sampler) ArgmaxEquivalent() bool {
	return s.p.Temperature <= 0 && len(s.p.LogitBias) == 0 && !s.penaltiesActive() && !s.p.Logprobs
}

// GreedyEquivalent reports whether this sampler's pick is deterministically the argmax token even though a
// temperature is set: the top_k=1 shape. It is true at any temperature: scaling is strictly monotone and preserves
// the logit ordering, and a distribution restricted to one token is deterministic. top_p and min_p at any value are
// safe alongside it, because both cuts keep at least one token (topFilterLogits), so the retained set is exactly the
// top-1. The tie-break agrees: topFilterLogits orders ties by ascending id and argmax uses a strict >, both selecting
// the lowest tied index, which is the contract the device argmax is aligned to (cuda.TestArgmaxTieBreak).
//
// It uses HistoryDependent, not penaltiesActive: penaltiesActive depends on the observed history, so it is false
// before the first Observe and true after, and a backend consults this predicate once per request, before any token
// is observed, and would latch the wrong answer. HistoryDependent asks whether penalties or bias are configured,
// which is stable. Logprobs is excluded as in ArgmaxEquivalent: reporting a distribution needs the logits the fast
// path skips.
func (s *Sampler) GreedyEquivalent() bool {
	return s.p.TopK == 1 && !s.p.HistoryDependent() && !s.p.Logprobs
}

// SampleWithInfo is Sample plus log-probability reporting (when SamplingParams.Logprobs is set): the chosen token's
// logprob and, if TopLogprobs > 0, that many highest-probability alternatives.
func (s *Sampler) SampleWithInfo(logits []float32) (SampleInfo, error) {
	if len(logits) == 0 {
		return SampleInfo{}, fmt.Errorf("decoder.Sample: empty logits")
	}
	// Bias/penalties mutate logits, so work on a copy when either is active;
	// the no-op path (greedy parity) keeps using the caller's slice directly.
	work := logits
	if len(s.p.LogitBias) > 0 || s.penaltiesActive() {
		// work never escapes this call, so the reused workBuf is safe in place of a fresh clone each token.
		work = s.workBufN(len(logits))
		copy(work, logits)
		s.applyLogitBias(work)
		s.applyPenalties(work)
	}

	var info SampleInfo
	if s.p.Temperature <= 0 {
		info.ID = argmax(work)
	} else if s.p.TopK > 0 || s.p.TopP > 0 || s.p.MinP > 0 {
		// Bounded selection in logit space, with no softmax over the whole vocabulary (topFilterLogits).
		info.ID = s.drawFiltered(topFilterLogits(work, s.p.Temperature, s.p.TopK, s.p.TopP, s.p.MinP, s.vocabBufN(len(work)), s.candBufN(len(work)), s.ipsBufN(len(work))))
		if info.ID < 0 {
			// Unreachable: every filter keeps at least the top token and NewSampler clamps the one input that could
			// empty the set. If it happens, argmax is the answer: a valid id beats a -1 propagating out as a token or a
			// panic in this goroutine.
			info.ID = argmax(work)
		}
	} else {
		// Temperature-only, with or without logprobs: Gumbel-max (sampler_gumbel.go). One draw for every variant, so asking
		// for logprobs or a penalty cannot change which token a seed produces.
		info.ID = s.gumbelDraw(work, s.p.Temperature)
	}
	if s.p.Logprobs {
		info.Logprob, info.Top = computeLogprobs(work, info.ID, s.p.Temperature, s.p.TopLogprobs, s.distBufN(len(work)))
	}
	s.recordHistory(info.ID)
	return info, nil
}

// penaltiesActive reports whether any repetition control is configured and
// there is history for it to act on.
func (s *Sampler) penaltiesActive() bool {
	if len(s.history) == 0 {
		return false
	}
	return (s.p.RepeatPenalty > 0 && s.p.RepeatPenalty != 1) ||
		s.p.PresencePenalty != 0 || s.p.FrequencyPenalty != 0
}

// penaltyWindow is the tail of history the penalties consider (RepeatLastN
// tokens; whole history when RepeatLastN ≤ 0).
func (s *Sampler) penaltyWindow() []int {
	n := s.p.RepeatLastN
	if n <= 0 || n >= len(s.history) {
		return s.history
	}
	return s.history[len(s.history)-n:]
}

// applyLogitBias adds each configured per-token offset to its logit.
func (s *Sampler) applyLogitBias(logits []float32) {
	for id, b := range s.p.LogitBias {
		if id >= 0 && id < len(logits) {
			logits[id] += b
		}
	}
}

// penaltiesConfigured reports whether any repetition penalty is set, independent
// of history (penaltiesActive additionally requires a non-empty history). Used by
// the speculative path, which threads its own per-position history.
func (s *Sampler) penaltiesConfigured() bool {
	return (s.p.RepeatPenalty > 0 && s.p.RepeatPenalty != 1) ||
		s.p.PresencePenalty != 0 || s.p.FrequencyPenalty != 0
}

// HistoryDependent reports whether any history-dependent logit transform is configured: repetition, presence or
// frequency penalties or LogitBias. Greedy speculative decoding verifies with argmax over the target's raw logits and
// cannot honor these, so its validators reject them to keep "token-identical to plain greedy" true. The sampled
// speculative path threads them per position instead (see needsHistory, which mirrors this check).
func (p SamplingParams) HistoryDependent() bool {
	return len(p.LogitBias) > 0 ||
		(p.RepeatPenalty > 0 && p.RepeatPenalty != 1) ||
		p.PresencePenalty != 0 || p.FrequencyPenalty != 0
}

// penaltyWindowOf returns the last n tokens of history (the whole history when
// n ≤ 0), the window the penalties consider — for an explicit history rather than
// the sampler's own.
func penaltyWindowOf(history []int, n int) []int {
	if n <= 0 || n >= len(history) {
		return history
	}
	return history[len(history)-n:]
}

// applyPenalties applies repeat, presence and frequency penalties to the logits of tokens in the sampler's own penalty
// window. Repeat scales (llama.cpp); presence and frequency subtract (OpenAI); they compose. The default window
// (RepeatLastN <= 0) is the whole history, which histCounts already tracks incrementally, so that case skips
// rebuilding a counts map every token (O(n^2) over a generation). The windowed case rebuilds: its window is bounded
// and small.
func (s *Sampler) applyPenalties(logits []float32) {
	if !s.penaltiesActive() {
		return
	}
	if s.p.RepeatLastN <= 0 {
		s.applyPenaltiesFromCounts(logits, s.histCounts)
		return
	}
	s.applyPenaltiesOver(logits, s.penaltyWindow())
}

// applyPenaltiesOver applies the penalties over an explicit token window.
func (s *Sampler) applyPenaltiesOver(logits []float32, window []int) {
	counts := map[int]int{}
	for _, id := range window {
		counts[id]++
	}
	s.applyPenaltiesFromCounts(logits, counts)
}

// applyPenaltiesFromCounts is applyPenaltiesOver's counts-already-known half, so applyPenalties' fast path can feed it
// the incrementally maintained histCounts.
func (s *Sampler) applyPenaltiesFromCounts(logits []float32, counts map[int]int) {
	rep := s.p.RepeatPenalty
	for id, c := range counts {
		if id < 0 || id >= len(logits) {
			continue
		}
		if rep > 0 && rep != 1 {
			if logits[id] > 0 {
				logits[id] = float32(float64(logits[id]) / rep)
			} else {
				logits[id] = float32(float64(logits[id]) * rep)
			}
		}
		if s.p.PresencePenalty != 0 {
			logits[id] -= float32(s.p.PresencePenalty)
		}
		if s.p.FrequencyPenalty != 0 {
			logits[id] -= float32(s.p.FrequencyPenalty * float64(c))
		}
	}
}

// computeLogprobs returns log P(chosen) and the topN highest-prob (id, logprob) pairs, over the full-vocab softmax at
// the sampling temperature (1 when greedy: temperature 0 would be a degenerate point mass). dst is the softmax
// scratch: a Sampler passes s.distBufN(len(logits)) to avoid a full-vocab allocation per logprobs request; nil
// allocates fresh.
func computeLogprobs(logits []float32, chosen int, temperature float64, topN int, dst []float64) (float64, []TokenLogprob) {
	t := temperature
	if t <= 0 {
		t = 1
	}
	probs := softmaxStableInto(logits, t, dst)
	lp := math.Log(probs[chosen])
	if topN <= 0 {
		return lp, nil
	}
	if topN > len(probs) {
		topN = len(probs)
	}
	// topKByLogit finds the top-N indices in O(V log N) instead of sorting every id. Ranking by logit agrees with ranking
	// by prob (softmax is strictly monotone in the logit at a fixed positive temperature), so the index set is identical;
	// only the N-element result needs a final sort. Its deterministic ties-toward-smaller-id also replaces sort.Slice's
	// unspecified tie order with a reproducible one.
	idx := topKByLogit(logits, topN)
	sort.Slice(idx, func(a, b int) bool {
		if probs[idx[a]] != probs[idx[b]] {
			return probs[idx[a]] > probs[idx[b]]
		}
		return idx[a] < idx[b] // deterministic: smaller id wins a tie, matching topKByLogit's own convention
	})
	top := make([]TokenLogprob, topN)
	for i, id := range idx {
		top[i] = TokenLogprob{ID: id, Logprob: math.Log(probs[id])}
	}
	return lp, top
}

// drawFiltered samples one id from the renormalized (id, prob) pairs that
// survived top-k/top-p/min-p filtering (the trailing return guards float rounding).
func (s *Sampler) drawFiltered(ips []indexedProb) int {
	// Defence in depth: the filters keep at least the top token and NewSampler clamps the one input that could empty
	// them, but an empty set would still be a -1 index. Return a sentinel the caller converts to argmax instead of
	// panicking in the generation goroutine.
	if len(ips) == 0 {
		return -1
	}
	r := s.rng.Float64()
	var cum float64
	for _, ip := range ips {
		cum += ip.p
		if r < cum {
			return ip.id
		}
	}
	return ips[len(ips)-1].id
}

// drawFull samples one id from a full probability vector by cumulative search.
func (s *Sampler) drawFull(probs []float64) int {
	r := s.rng.Float64()
	var cum float64
	for i, p := range probs {
		cum += p
		if r < cum {
			return i
		}
	}
	return lastWithMass(probs, 0, len(probs))
}

// lastWithMass is the float-rounding fall-through for a cumulative draw: the last index in [lo,hi) whose probability
// is non-zero, or hi-1 when every one is zero. The vector is masked (top-k and top-p zero the excluded tail), so
// returning hi-1 outright would often emit a token the filter removed, which the caller configured to be impossible.
func lastWithMass(p []float64, lo, hi int) int {
	for i := hi - 1; i >= lo; i-- {
		if p[i] > 0 {
			return i
		}
	}
	return hi - 1
}

func argmax(logits []float32) int {
	if len(logits) == 0 {
		return 0
	}
	_ = logits[len(logits)-1]
	best, bi := logits[0], 0
	i := 1
	for ; i+3 < len(logits); i += 4 {
		_ = logits[i+3]
		v0 := logits[i]
		if v0 > best {
			best, bi = v0, i
		}
		v1 := logits[i+1]
		if v1 > best {
			best, bi = v1, i+1
		}
		v2 := logits[i+2]
		if v2 > best {
			best, bi = v2, i+2
		}
		v3 := logits[i+3]
		if v3 > best {
			best, bi = v3, i+3
		}
	}
	for ; i < len(logits); i++ {
		v := logits[i]
		if v > best {
			best, bi = v, i
		}
	}
	return bi
}

// softmaxStable converts logits to probabilities (numerically stable). It always allocates; softmaxStableInto is the
// scratch-reusing sibling for per-token hot paths.
func softmaxStable(logits []float32, temperature float64) []float64 {
	return softmaxStableInto(logits, temperature, nil)
}

// softmaxStableInto is softmaxStable with a caller-owned destination: dst is reused when its capacity fits, grown once
// otherwise. Safe only where the caller consumes the result before requesting the next one (the same
// one-position-at-a-time contract as distBufN and specLogitsBufN, whose buffers it shares).
func softmaxStableInto(logits []float32, temperature float64, dst []float64) []float64 {
	if len(logits) == 0 {
		return nil
	}
	_ = logits[len(logits)-1]
	if temperature <= 0 {
		temperature = 1
	}
	maxv := float64(logits[0])
	for _, v := range logits[1:] {
		if float64(v) > maxv {
			maxv = float64(v)
		}
	}
	var out []float64
	if cap(dst) >= len(logits) {
		out = dst[:len(logits)]
	} else {
		out = make([]float64, len(logits))
	}
	_ = out[len(out)-1]
	var sum float64
	for i, v := range logits {
		e := math.Exp((float64(v) - maxv) / temperature)
		out[i] = e
		sum += e
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// indexedProb pairs a token id with its probability for top-k/top-p/min-p filtering.
type indexedProb struct {
	id int
	p  float64
}

// topFilterLogits applies top-k, then min-p, then top-p to a logit vector, returning the surviving (id,
// renormalized-prob) pairs in descending order.
//
// Tie-break contract: entries with equal probability are ordered by ascending token id. That order feeds the
// cumulative-CDF draw, so it is part of the sampling result (sort.Slice would leave it arbitrary). The test-only
// reference refTopFilter (sampler_selection_test.go) carries the same tie-break and this path is gated bit for bit
// against it. The CUDA argmax reduce returns the lowest index on an exact tie to match (cuda.TestArgmaxTieBreak).
//
// Summation-order contract: every probability sum below, the top-p cumulative and the final renormalization, runs in
// descending probability order. Summing in another order moves the denominator by ULPs and can flip which side of the
// cumulative-p boundary a token lands on, changing the draw. Do not reorder these loops.
//
// Logit-space selection: temperature scaling and exp are monotone, so top-k and min-p are decided on raw logits with no
// softmax over V; exp is applied only to the retained set. top-p's cutoff is defined on the normalized mass, so it
// needs the full-vocabulary denominator Z: one O(V) exp-sum, with no full probability array and no sort.
//
// candScratch and ipsScratch are caller-owned scratch for the no-filter candidate list and the indexed-probability
// pairs. Both are consumed within this call and by the caller immediately after, never retained.
func topFilterLogits(logits []float32, temperature float64, topK int, topP, minP float64, vocabScratch []float64, candScratch []int, ipsScratch []indexedProb) []indexedProb {
	texp := temperature
	if texp <= 0 {
		texp = 1 // defensive; SampleWithInfo only reaches here for temperature > 0
	}
	maxF := logits[0]
	i := 1
	_ = logits[len(logits)-1]
	for ; i+3 < len(logits); i += 4 {
		v0, v1, v2, v3 := logits[i], logits[i+1], logits[i+2], logits[i+3]
		m0 := max(v0, v1)
		m1 := max(v2, v3)
		m := max(m0, m1)
		if m > maxF {
			maxF = m
		}
	}
	for ; i < len(logits); i++ {
		if logits[i] > maxF {
			maxF = logits[i]
		}
	}
	maxL := float64(maxF)
	// Z (full-vocab softmax denominator) is needed ONLY for the top-p cutoff.
	topPActive := topP > 0 && topP < 1
	var Z float64
	if topPActive {
		// Z uses the same fixed-chunk fold as the temperature-only path; regrouping it moves boundary draws, which is a
		// seed-visible change.
		Z = chunkedZ(logits, maxL, texp, vocabScratch)
	}

	// Candidate set: a bounded SUPERSET of the retained set, trimmed by the min-p /
	// top-p cuts below. The most restrictive active selector bounds it in logit space.
	var cand []int
	switch {
	case topK > 0:
		cand = topKByLogit(logits, topK)
	case minP > 0:
		// e_i ≥ minP·e_max ⟺ logit_i ≥ maxL + T·ln(minP)  (e_max = 1 at the argmax).
		thr := maxL + texp*math.Log(minP)
		cand = candScratch[:0]
		for i, v := range logits {
			if float64(v) >= thr {
				cand = append(cand, i)
			}
		}
	case topPActive:
		cand = topPCandidates(logits, texp, maxL, Z, topP)
	default:
		// Only reachable when the sole "filter" is top-p ≥ 1 (i.e. no effective filter);
		// keep every token, matching the reference. Degenerate and rare.
		cand = candScratch[:len(logits)]
		for i := range cand {
			cand[i] = i
		}
	}

	// Materialize e (unnormalized prob) for the candidates, then order by (prob desc,
	// id asc). We MUST sort by e, not by the raw logit: at low temperature many distinct
	// logits underflow to the same e (typically 0), so they are tied in probability and
	// the contract orders those by id — sorting by logit would split that tie wrongly.
	ips := ipsScratch[:len(cand)]
	for i, id := range cand {
		ips[i] = indexedProb{id: id, p: math.Exp((float64(logits[id]) - maxL) / texp)}
	}
	return finishFilter(ips, minP, topP, topPActive, Z)
}

// finishFilter is topFilterLogits' tail, shared with the device top-K path (sampler_topk.go) so the two
// can never diverge after candidate selection: order the candidates by (prob desc, id asc), apply the
// min-p then top-p cuts, drop zero-probability tokens, and renormalize in descending order. ips must
// hold unnormalized e = exp((logit−max)/T) values; Z is the full-vocabulary denominator and is read
// only when topPActive.
func finishFilter(ips []indexedProb, minP, topP float64, topPActive bool, Z float64) []indexedProb {
	slices.SortFunc(ips, func(a, b indexedProb) int {
		switch {
		case a.p > b.p:
			return -1
		case a.p < b.p:
			return 1
		default:
			return a.id - b.id // equal prob → ascending token id (tie-break contract)
		}
	})

	// min-p cut, relative to the most-likely retained token. Idempotent when min-p was
	// the selector; a genuine trim when top-k was.
	if minP > 0 && len(ips) > 0 {
		thresh := minP * ips[0].p
		cut := len(ips)
		for i := range ips {
			if ips[i].p < thresh {
				cut = i
				break
			}
		}
		if cut < 1 {
			cut = 1 // always keep the top token
		}
		ips = ips[:cut]
	}
	// top-p cut: smallest DESCENDING prefix whose normalized mass reaches topP. Compared
	// as Σe ≥ topP·Z (Z is the full-vocab denominator) so it is the exact nucleus without
	// a per-term divide. If the candidate set holds less mass than topP (top-k/min-p
	// already capped it), cut stays len(ips) — keep them all, matching the reference.
	if topPActive {
		target := topP * Z
		var cum float64
		cut := len(ips)
		for i := range ips {
			cum += ips[i].p // p == e here (pre-renormalization)
			if cum >= target {
				cut = i + 1
				break
			}
		}
		if cut < 1 {
			cut = 1
		}
		ips = ips[:cut]
	}
	// Drop unsamplable zero-probability tokens (temperature underflow). They carry no
	// mass, so the draw is unchanged — but their relative order is unobservable, and a
	// top-k that reaches into the zero region would otherwise keep a different subset of
	// them than the reference. Dropping them keeps the returned set exactly the reference's
	// and never returns a token the CDF walk can't reach. (The argmax has e=1, so ≥1 stays.)
	nz := ips[:0]
	for _, ip := range ips {
		if ip.p > 0 {
			nz = append(nz, ip)
		}
	}
	ips = nz

	// Renormalize the retained set in DESCENDING order (summation-order contract).
	var sum float64
	for i := range ips {
		sum += ips[i].p
	}
	for i := range ips {
		ips[i].p /= sum
	}
	return ips
}

// topKByLogit returns the indices of the k highest logits (ties resolved toward the
// smaller id, matching the ascending-id tie-break), via a k-bounded min-heap: O(V·log k),
// no O(V·log V) sort. The returned slice is unordered; the caller sorts it.
func topKByLogit(logits []float32, k int) []int {
	n := len(logits)
	if k >= n {
		out := make([]int, n)
		for i := range out {
			out[i] = i
		}
		return out
	}
	// "worse to keep": lower logit, or on a tie the LARGER id (so ties evict larger ids,
	// keeping smaller). The min-heap root is the worst-kept element.
	h := make([]int, 0, k)
	worse := func(a, b int) bool {
		if logits[a] != logits[b] {
			return logits[a] < logits[b]
		}
		return a > b
	}
	siftUp := func(i int) {
		for i > 0 {
			p := (i - 1) / 2
			if worse(h[i], h[p]) { // child worse than parent → move it toward the root
				h[p], h[i] = h[i], h[p]
				i = p
			} else {
				break
			}
		}
	}
	siftDownRoot := func() {
		i := 0
		for {
			l, r, m := 2*i+1, 2*i+2, i
			if l < len(h) && worse(h[l], h[m]) {
				m = l
			}
			if r < len(h) && worse(h[r], h[m]) {
				m = r
			}
			if m == i {
				break
			}
			h[i], h[m] = h[m], h[i]
			i = m
		}
	}
	for id := range n {
		if len(h) < k {
			h = append(h, id)
			siftUp(len(h) - 1)
		} else if worse(h[0], id) { // candidate better than the worst kept → swap in
			h[0] = id
			siftDownRoot()
		}
	}
	return h
}

// topPCandidates returns a SUPERSET of the top-p nucleus: the top-b logits (by
// topKByLogit) for the smallest b whose retained exp-mass reaches topP·Z. The bound is
// adaptive — it grows until the mass is verified sufficient host-side, so no fixed cutoff
// can truncate the nucleus (amendment: adaptive bound). The caller applies the exact
// descending cut. Typical nuclei are small; the pathological flat case grows to V.
func topPCandidates(logits []float32, texp, maxL, Z, topP float64) []int {
	n := len(logits)
	target := topP * Z
	b := min(32, n)
	for {
		cand := topKByLogit(logits, b)
		var sum float64
		for _, id := range cand {
			sum += math.Exp((float64(logits[id]) - maxL) / texp)
		}
		if sum >= target || b >= n {
			return cand
		}
		b *= 8
		if b > n {
			b = n
		}
	}
}
