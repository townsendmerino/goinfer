package decoder

import "math"

// Sampling from a device-side top-K (R7, docs/tasks/red-october.md).
//
// A resident backend that implements ResidentTopK hands the sampler the K best logits of the row
// instead of all V. For a FILTERED sampler that is enough to reproduce the host path exactly: the
// retained set of top-k / min-p / top-p is a prefix of the (logit desc, id asc) ordering, the
// probabilities are computed from the same float32 logits with the same float64 expression, and
// everything after candidate selection is finishFilter — the code topFilterLogits itself runs.
//
// What K must prove, per token, or the caller falls back to the full row for that token (no RNG has
// been drawn yet, so the fallback is invisible):
//   - top-k:        K ≥ topK (the width is chosen so this always holds);
//   - min-p:        the K-th returned logit is below the min-p threshold, so no qualifying token is
//                   outside the K;
//   - top-p:        the K returned probabilities already carry topP·Z of the mass, so the nucleus
//                   ends inside the K (the same check topPCandidates makes host-side).
//
// TEMPERATURE-ONLY SAMPLING IS NOT SERVED HERE, and cannot be: it draws by inverse CDF in vocabulary
// index order over a full-V normalisation (sampleChunked), and P2b already refuted a truncated-tail
// shortcut. It stays on the full-row path.
//
// top-p's Z comes from the device (an f32 exp sum reduced in f64), not from chunkedZ's f64 sum, so it
// differs from the host's by rounding. It can only matter when the cumulative mass lands within that
// rounding of topP·Z — the same class of given-seed shift P2b accepted for regrouping Z on the host.
// The mismatch rate against the full path is measured and recorded (cuda TestSampledTopKStreamIdentity)
// rather than assumed.

const (
	// topKWidthDefault is the number of candidates requested per token.
	topKWidthDefault = 256
	// topKWidthWide is used for very wide vocabularies (262k+ families), where the nucleus of a flat
	// distribution is larger.
	topKWidthWide = 1024
	// topKWideVocab is the vocab size at which topKWidthWide applies.
	topKWideVocab = 200_000
)

// TopKEligible reports whether this sampler's configuration can be served from a device top-K: a
// filtered sampler (top-k, min-p or a real top-p) at temperature > 0, with nothing that needs or
// rewrites the full row. The decode loop additionally requires no LogitProcessor.
func (s *Sampler) TopKEligible() bool {
	p := s.p
	// penaltiesConfigured, NOT penaltiesActive: active is false until the sampler has history, so at the
	// start of a generation a configured penalty looks inactive — and would switch on mid-stream, after
	// this path had already stopped reading the full row.
	if p.Temperature <= 0 || p.Logprobs || len(p.LogitBias) > 0 || s.penaltiesConfigured() {
		return false
	}
	return p.TopK > 0 || p.MinP > 0 || (p.TopP > 0 && p.TopP < 1)
}

// topPActive mirrors topFilterLogits' own predicate.
func (s *Sampler) topPActive() bool { return s.p.TopP > 0 && s.p.TopP < 1 }

// TopKWidth returns how many candidates to request for a vocab of the given size, and false when the
// configuration cannot be served (an explicit top-k wider than the kernel's maximum).
func (s *Sampler) TopKWidth(vocab int) (int, bool) {
	k := topKWidthDefault
	if vocab >= topKWideVocab {
		k = topKWidthWide
	}
	if s.p.TopK > k {
		k = topKWidthWide
	}
	if s.p.TopK > k {
		return 0, false
	}
	if k > vocab {
		k = vocab
	}
	return k, true
}

// SampleFromTopK draws a token from a device top-K row exactly as SampleWithInfo would from the full
// row, or reports ok=false — without consuming any randomness — when the K candidates cannot prove
// they hold the whole retained set. The caller then reads the full row and calls SampleWithInfo.
// vocab is the full row's length.
func (s *Sampler) SampleFromTopK(row TopKRow, vocab int) (SampleInfo, bool) {
	n := len(row.IDs)
	if n == 0 || len(row.Logits) != n {
		return SampleInfo{}, false
	}
	texp := s.p.Temperature
	if texp <= 0 {
		return SampleInfo{}, false
	}
	topP, minP, topK := s.p.TopP, s.p.MinP, s.p.TopK
	topPActive := s.topPActive()
	if !(topK > 0 || minP > 0 || topPActive) {
		return SampleInfo{}, false
	}
	maxL := float64(row.Logits[0])
	full := n >= vocab // the row IS the whole vocabulary; nothing outside it to prove absent

	var Z float64
	if topPActive {
		Z = row.Z
		if !(Z > 0) {
			return SampleInfo{}, false
		}
	}

	cand := n
	switch {
	case topK > 0:
		if topK > n {
			return SampleInfo{}, false
		}
		cand = topK
	case minP > 0:
		// e_i ≥ minP·e_max ⟺ logit_i ≥ maxL + T·ln(minP), the same threshold topFilterLogits uses.
		thr := maxL + texp*math.Log(minP)
		cand = 0
		for cand < n && float64(row.Logits[cand]) >= thr {
			cand++
		}
		if cand == n && !full {
			return SampleInfo{}, false // the K-th is still above the threshold: more qualify beyond K
		}
	}

	var ips []indexedProb
	if topPActive && topK == 0 && minP == 0 {
		// Nucleus-only. Walk the candidates in row order — (logit desc, id asc), which is (prob desc, id
		// asc) for every prefix that matters, since distinct float32 logits never collide after exp — and
		// stop as soon as the cumulative mass reaches topP·Z. That both PROVES the K carry the nucleus (if
		// the loop runs out first, they might not) and bounds the exp work to the nucleus, which is what
		// this costs on a core that has just woken from the GPU sync: math.Exp measured ~360 ns/call
		// there, so the old "sum all K, then exp all K again" pass was ~190 us per token. cum here is
		// accumulated in the same order finishFilter will use, so its cut lands at the same index. No
		// tie handling past the cut is needed: two distinct float32 logits do not give the same float64
		// exp at any realistic temperature (it takes T around 1e7), so equal probabilities mean equal logits,
		// which the row already orders by ascending id —
		// the order finishFilter sorts to. (A mutation that deleted such a loop survived every test, which
		// is how this was found to be dead.)
		target := topP * Z
		ips = s.ipsBufN(n)
		var cum float64
		m := 0
		for m < n {
			e := math.Exp((float64(row.Logits[m]) - maxL) / texp)
			ips[m] = indexedProb{id: int(row.IDs[m]), p: e}
			cum += e
			m++
			if cum >= target {
				break
			}
		}
		if cum < target && !full {
			return SampleInfo{}, false
		}
		ips = ips[:m]
	} else {
		ips = s.ipsBufN(cand)
		for i := 0; i < cand; i++ {
			ips[i] = indexedProb{id: int(row.IDs[i]), p: math.Exp((float64(row.Logits[i]) - maxL) / texp)}
		}
	}
	ips = finishFilter(ips, minP, topP, topPActive, Z)

	var info SampleInfo
	info.ID = s.drawFiltered(ips)
	if info.ID < 0 {
		info.ID = int(row.IDs[0]) // the argmax: lowest id among the maxima, by the row's tie order
	}
	s.recordHistory(info.ID)
	return info, true
}
