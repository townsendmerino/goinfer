package decoder

import "math"

// Temperature-only sampling by Gumbel-max (docs/tasks/red-october.md R7b, owner decision 2026-09-20).
//
// WHAT IT IS. For a logit row l and temperature T, the token is
//
//	argmax_i ( l_i/T + G_i ),   G_i = -ln(E_i),   E_i = -ln(1 - w_i),   w_i = (h_i + 0.5) / 2^32
//
// where h_i is a 32-bit word from Philox4x32-10 keyed by (seed) with counter (i>>2, draw, draw>>32, 0), lane
// i&3. By the Gumbel-max theorem that is an exact draw from softmax(l/T), with no normalisation, no
// cumulative sum and no per-token dependence on the other tokens — so it is an argmax, which every GPU
// backend can do in one parallel pass with no f64, and which needs no full-vocabulary readback.
//
// WHY THIS REPLACED THE INVERSE-CDF DRAW. The old temperature-only path (sampleChunked, now the reference
// in sampler_chunked_ref_test.go) drew by cumulative search in vocabulary INDEX order over a full
// normalisation. That cannot be reproduced from a top-K and needs f64 exp to reproduce on a device, so the
// server's default sampling shape (temperature 1, no filters) sat at ~0.74 of greedy speed on CUDA. This
// path is ~0.97 there and works on Metal and WebGPU.
//
// THE COST, DISCLOSED: for a given seed this draws DIFFERENT tokens than every earlier release (the
// distribution is unchanged; the stream is not). Speculative sampled decoding keeps drawing from explicit
// probabilities (it needs them for accept/reject) and was never stream-equal to plain decoding — it is
// in-distribution lossless (spec_sample.go).
//
// NOISE RANGE. G is bounded, so a token whose logit gap to the maximum exceeds the range can never win, where
// the exact draw would give it e^-gap. With w in (0,1) at 2^-32 resolution, E spans [1.2e-10, 22.9], so G
// spans [-3.13, 22.9]: a token needs a gap above ~26 nats to be unreachable, i.e. probability below ~5e-12 of
// the maximum's, and the total mass affected is ≲1e-6 even over a 262k vocabulary. (An f32 uniform u would
// cap G at 16.6 and lose ~5e-4; that is why the noise is computed from w = 1-u with log1p.)
//
// DETERMINISM. Philox is integer-exact, so every backend draws identical h_i. The float transform below may
// differ by an ulp or two between backends, which can change the argmax only when the two best keys are
// within that — measured, not assumed (cuda TestGumbelDeviceAgreesWithHost).

// gumbelNoise maps 32 random bits to a standard Gumbel variate.
func gumbelNoise(h uint32) float32 {
	w := (float64(h) + 0.5) * (1.0 / 4294967296.0) // (0,1), never 0 or 1
	e := -math.Log1p(-w)                           // Exp(1); small-E (large-G) end kept precise
	return float32(-math.Log(e))
}

// gumbelKey is one entry's score. The explicit float32 conversions are load-bearing: they forbid the
// compiler fusing the multiply into the add (FMA), which would make the result depend on the CPU.
func gumbelKey(logit, invT float32, h uint32) float32 {
	return float32(logit*invT) + gumbelNoise(h)
}

// gumbelArgmaxRange returns the best (key, index) over logits[lo:hi], -1 if none is comparable. Ties go to
// the lowest index (strict >).
func gumbelArgmaxRange(logits []float32, lo, hi int, invT float32, key [2]uint32, draw uint64) (float32, int) {
	best := float32(math.Inf(-1))
	idx := -1
	d0, d1 := uint32(draw), uint32(draw>>32)
	for b := lo >> 2; b<<2 < hi; b++ {
		r := philox4x32([4]uint32{uint32(b), d0, d1, 0}, key)
		for lane := 0; lane < 4; lane++ {
			i := b<<2 + lane
			if i < lo || i >= hi {
				continue
			}
			if k := gumbelKey(logits[i], invT, r[lane]); k > best {
				best, idx = k, i
			}
		}
	}
	return best, idx
}

// NextDraw returns the (seed, draw index) of the next temperature-only draw and advances it. It is the
// hook a device sampler uses so its draw i is the SAME draw the host would have made: whichever path
// serves a step, the stream is a function of (seed, step, logits) only.
func (s *Sampler) NextDraw() (seed, draw uint64) {
	seed, draw = s.gseed, s.gdraw
	s.gdraw++
	return seed, draw
}

// gumbelDraw is the host reference implementation of the temperature-only draw.
func (s *Sampler) gumbelDraw(logits []float32, temperature float64) int {
	if temperature <= 0 {
		temperature = 1
	}
	invT := float32(1 / temperature)
	seed, draw := s.NextDraw()
	if math.IsInf(float64(invT), 0) { // absurdly small temperature: the argmax, as greedy
		return argmax(logits)
	}
	key := [2]uint32{uint32(seed), uint32(seed >> 32)}
	var keys [numChunks]float32
	var idxs [numChunks]int
	forEachChunk(len(logits), func(c, lo, hi int) {
		keys[c], idxs[c] = gumbelArgmaxRange(logits, lo, hi, invT, key, draw)
	})
	best := float32(math.Inf(-1))
	idx := -1
	for c := range numChunks { // ascending chunks + strict >: the lowest index wins a tie
		if idxs[c] >= 0 && keys[c] > best {
			best, idx = keys[c], idxs[c]
		}
	}
	if idx < 0 {
		return argmax(logits)
	}
	return idx
}

// SampleEligible reports whether this sampler's configuration can be drawn on-device by ResidentSample:
// temperature-only sampling with nothing that needs or rewrites the full row (logprobs, bias, penalties). The
// decode loop additionally requires no LogitProcessor.
func (s *Sampler) SampleEligible() bool {
	p := s.p
	if p.Logprobs || len(p.LogitBias) > 0 || s.penaltiesConfigured() {
		return false
	}
	return s.gumbelServes()
}
