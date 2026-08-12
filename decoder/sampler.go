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
	history []int // tokens seen so far: prompt (via Observe) + each drawn token
}

// NewSampler builds a sampler from params.
func NewSampler(p SamplingParams) *Sampler {
	return &Sampler{p: p, rng: rand.New(rand.NewSource(p.Seed))}
}

// Observe seeds the penalty history with already-seen tokens (the generation
// loop calls it once with the prompt) so repetition penalties consider them.
// Each Sample then records the token it draws.
func (s *Sampler) Observe(ids ...int) { s.history = append(s.history, ids...) }

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

// SampleWithInfo is Sample plus log-probability reporting (when
// SamplingParams.Logprobs is set): the chosen token's logprob and, if
// TopLogprobs > 0, that many highest-probability alternatives.
// ArgmaxEquivalent reports whether this sampler's pick is exactly argmax(raw logits): no
// logit bias, no penalties, no temperature/filtering, and no logprob reporting that would
// need the distribution. When true (and SamplingParams.LogitProcessor is nil), a backend that
// can compute the argmax on-device may return just the id and skip the full-logits readback —
// the emitted tokens are identical.
//
// THIS AND GreedyEquivalent BELOW LIVE NEXT TO EVERY LOGIT-TOUCHING PARAMETER ON PURPOSE:
// adding a new parameter that mutates or reads logits must be reflected in BOTH, so a fast path
// can never silently diverge. They are separate predicates, not one widened one, so that call
// sites choose deliberately which notion of "deterministic" they need (see GreedyEquivalent).
func (s *Sampler) ArgmaxEquivalent() bool {
	return s.p.Temperature <= 0 && len(s.p.LogitBias) == 0 && !s.penaltiesActive() && !s.p.Logprobs
}

// GreedyEquivalent reports whether this sampler's pick is deterministically the argmax token even
// though a temperature is set — the `top_k=1` shape. It is TRUE at any temperature, which is not a
// loophole: temperature scaling is strictly monotone, so it preserves the ordering of logits, and a
// distribution restricted to ONE token is deterministic regardless of that token's probability.
// So `top_k=1` at temperature 1.5 emits exactly what greedy emits.
//
// `top_p` / `min_p` at any value are safe alongside it: both cuts clamp at ≥1 retained token
// (topFilterLogits, "always keep the top token"), so the retained set is exactly the top-1 either
// way. The tie-break agrees too — topFilterLogits orders ties by ascending id and argmax() uses a
// strict `>`, both selecting the LOWEST tied index, which is the contract the device argmax was
// aligned to (audit C-14, gate cuda.TestArgmaxTieBreak).
//
// It uses HistoryDependent, NOT penaltiesActive: penaltiesActive() is computed from the sampler's
// observed history, so it is false before the first Observe and true after. A backend consults this
// predicate ONCE per request, before any token is observed, and would latch the wrong answer.
// HistoryDependent asks the question that is actually stable — "are penalties/bias CONFIGURED" —
// which is the right shape for a per-request routing decision.
//
// Logprobs is excluded for the same reason as in ArgmaxEquivalent: reporting a distribution needs
// the logits the fast path is skipping.
func (s *Sampler) GreedyEquivalent() bool {
	return s.p.TopK == 1 && !s.p.HistoryDependent() && !s.p.Logprobs
}

func (s *Sampler) SampleWithInfo(logits []float32) (SampleInfo, error) {
	if len(logits) == 0 {
		return SampleInfo{}, fmt.Errorf("decoder.Sample: empty logits")
	}
	// Bias/penalties mutate logits, so work on a copy when either is active;
	// the no-op path (greedy parity) keeps using the caller's slice directly.
	work := logits
	if len(s.p.LogitBias) > 0 || s.penaltiesActive() {
		work = slices.Clone(logits)
		s.applyLogitBias(work)
		s.applyPenalties(work)
	}

	var info SampleInfo
	if s.p.Temperature <= 0 {
		info.ID = argmax(work)
	} else if s.p.TopK > 0 || s.p.TopP > 0 || s.p.MinP > 0 {
		// Bounded selection in logit space — no softmax over the whole vocabulary
		// (amendment 3). The old path softmaxed all V then full-sorted all V; both
		// are gone for the filtered case, replaced by topFilterLogits below.
		info.ID = s.drawFiltered(topFilterLogits(work, s.p.Temperature, s.p.TopK, s.p.TopP, s.p.MinP))
	} else if s.p.Logprobs {
		// Logprobs needs the full normalized distribution anyway, so the exact path costs nothing extra.
		info.ID = s.drawFull(softmaxStable(work, s.p.Temperature))
	} else {
		// P2b: deterministic PARALLEL normalization over a FIXED chunk count, drawing against
		// unnormalized weights (the divide pass is gone). Same exp values as before; the sum is
		// regrouped by chunk, which shifts Z by ULPs and therefore moves near-boundary draws — the
		// one given-seed change, bundled with the top-p Z pass below.
		//
		// LAZY Z WAS TRIED HERE AND REFUTED (P2, bc59c56): skipping the tail is not possible at these
		// vocabularies (the remainder bound needs an ~11.2-nat gap; real logits give 5.29 at K=32).
		// So the work is done in parallel instead of avoided.
		info.ID = s.sampleChunked(work, s.p.Temperature, s.rng.Float64())
	}
	if s.p.Logprobs {
		info.Logprob, info.Top = computeLogprobs(work, info.ID, s.p.Temperature, s.p.TopLogprobs)
	}
	s.history = append(s.history, info.ID)
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

// HistoryDependent reports whether any history-dependent logit transform is
// configured: repetition/presence/frequency penalties or LogitBias. Greedy
// speculative decoding verifies with argmax over the target's raw logits, so it
// cannot honor these — its validators reject them to keep "token-identical to plain
// greedy" true (M13). The sampled speculative path threads them per-position
// instead (see needsHistory, which mirrors this check).
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

// applyPenalties applies repeat/presence/frequency penalties to the logits of
// tokens in the sampler's own penalty window. Repeat scales (llama.cpp);
// presence/frequency subtract (OpenAI). They compose.
func (s *Sampler) applyPenalties(logits []float32) {
	if !s.penaltiesActive() {
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

// computeLogprobs returns log P(chosen) and the topN highest-prob (id, logprob)
// pairs, over the full-vocab softmax at the sampling temperature (1 when greedy
// — temperature 0 would be a degenerate point mass).
func computeLogprobs(logits []float32, chosen int, temperature float64, topN int) (float64, []TokenLogprob) {
	t := temperature
	if t <= 0 {
		t = 1
	}
	probs := softmaxStable(logits, t)
	lp := math.Log(probs[chosen])
	if topN <= 0 {
		return lp, nil
	}
	ips := make([]indexedProb, len(probs))
	for i, p := range probs {
		ips[i] = indexedProb{id: i, p: p}
	}
	sort.Slice(ips, func(a, b int) bool { return ips[a].p > ips[b].p })
	if topN > len(ips) {
		topN = len(ips)
	}
	top := make([]TokenLogprob, topN)
	for i := range top {
		top[i] = TokenLogprob{ID: ips[i].id, Logprob: math.Log(ips[i].p)}
	}
	return lp, top
}

// drawFiltered samples one id from the renormalized (id, prob) pairs that
// survived top-k/top-p/min-p filtering (the trailing return guards float rounding).
func (s *Sampler) drawFiltered(ips []indexedProb) int {
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
	return len(probs) - 1
}

func argmax(logits []float32) int {
	best, bi := logits[0], 0
	for i, v := range logits[1:] {
		if v > best {
			best, bi = v, i+1
		}
	}
	return bi
}

// softmaxStable converts logits to probabilities (numerically stable).
func softmaxStable(logits []float32, temperature float64) []float64 {
	if temperature <= 0 {
		temperature = 1
	}
	maxv := float64(logits[0])
	for _, v := range logits[1:] {
		if float64(v) > maxv {
			maxv = float64(v)
		}
	}
	out := make([]float64, len(logits))
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

// topFilterLogits applies top-k, then min-p, then top-p to a LOGIT vector,
// returning the surviving (id, renormalized-prob) pairs in descending order.
// It replaces the old topFilter, which softmaxed all V then full-sorted all V.
//
// TIE-BREAK CONTRACT (amendment 1): entries with equal probability are ordered by
// ASCENDING token id. The old path used sort.Slice, which is not stable, so the order
// of tied entries was arbitrary — and since that order feeds the cumulative-CDF draw,
// it was an unspecified part of the sampling result. It is now specified. The test-only
// reference (refTopFilter, sampler_selection_test.go) carries the identical tie-break
// and this path is gated bit-for-bit against it. (Same defect class as the CUDA
// argmax-reduce index tie-break, which is FIXED — audit C-14, c6600fc: argmax_reduce returns the
// lowest index on an exact tie, matching this contract. Gate: cuda.TestArgmaxTieBreak.)
//
// SUMMATION-ORDER CONTRACT (amendment 2): every probability sum below — the top-p
// cumulative and the final renormalization — runs in DESCENDING probability order.
// The denominator is load-bearing: summing in a different order moves it by ULPs and
// can flip which side of the cumulative-p boundary a token lands on, changing the draw.
// Do not reorder these loops. (Same treatment as the Metal reduction widths.)
//
// LOGIT-SPACE SELECTION (amendment 3): temperature scaling and exp are monotone, so
// top-k and min-p are decided on the raw logits with no softmax over V; exp is applied
// only to the (small) retained set. top-p is the one filter whose cutoff is defined on
// the NORMALIZED mass, so it needs the full-vocab softmax denominator Z — computed as a
// single O(V) exp-sum, with no full probability array and no O(V·log V) sort. That Z
// pass is irreducible for an exact nucleus; the sort it replaces is not.
func topFilterLogits(logits []float32, temperature float64, topK int, topP, minP float64) []indexedProb {
	texp := temperature
	if texp <= 0 {
		texp = 1 // defensive; SampleWithInfo only reaches here for temperature > 0
	}
	maxL := float64(logits[0])
	for _, v := range logits[1:] {
		if float64(v) > maxL {
			maxL = float64(v)
		}
	}
	// Z (full-vocab softmax denominator) is needed ONLY for the top-p cutoff.
	topPActive := topP > 0 && topP < 1
	var Z float64
	if topPActive {
		// P2b step 3: the same fixed-chunk fold as the temperature-only path. This Z feeds the
		// nucleus cut, so regrouping it also moves boundary draws — which is exactly why it lands in
		// the SAME release rather than dribbling out later as a second seed change.
		Z = chunkedZ(logits, maxL, texp)
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
		cand = make([]int, 0, 64)
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
		cand = make([]int, len(logits))
		for i := range cand {
			cand[i] = i
		}
	}

	// Materialize e (unnormalized prob) for the candidates, then order by (prob desc,
	// id asc). We MUST sort by e, not by the raw logit: at low temperature many distinct
	// logits underflow to the same e (typically 0), so they are tied in probability and
	// the contract orders those by id — sorting by logit would split that tie wrongly.
	ips := make([]indexedProb, len(cand))
	for i, id := range cand {
		ips[i] = indexedProb{id: id, p: math.Exp((float64(logits[id]) - maxL) / texp)}
	}
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
