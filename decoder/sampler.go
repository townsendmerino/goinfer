package decoder

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// SamplingParams controls next-token selection. Zero value (Temperature 0)
// is greedy/argmax — deterministic and the right default for parity tests.
type SamplingParams struct {
	Temperature float64 // 0 = greedy; >0 scales logits before softmax
	TopK        int     // 0 = disabled; keep the K highest-prob tokens
	TopP        float64 // 0 = disabled; nucleus — smallest set with cumprob ≥ TopP
	Seed        int64   // RNG seed for reproducible sampling
	StopIDs     []int   // extra ids that end generation (besides config EOS), e.g. <end_of_turn>
	// LogitProcessor, if non-nil, is called each decode step with the ids
	// generated so far and that step's logits, which it may modify in place
	// before the sampler runs — e.g. masking disallowed tokens to -inf for
	// constrained/structured decoding (see the constrain package). It runs after
	// the forward pass and before sampling and the stop check, so a constraint
	// can also gate EOS (mask it until the output is a complete document).
	LogitProcessor func(generated []int, logits []float32)
}

// Sampler turns a logit vector into a token id. It owns its RNG so a run is
// reproducible from Seed.
type Sampler struct {
	p   SamplingParams
	rng *rand.Rand
}

// NewSampler builds a sampler from params.
func NewSampler(p SamplingParams) *Sampler {
	return &Sampler{p: p, rng: rand.New(rand.NewSource(p.Seed))}
}

// Sample returns the chosen token id for the given logits ([VocabSize]).
//
//   - Temperature ≤ 0 is greedy (argmax) — deterministic, ignores top-k/top-p.
//   - Temperature > 0: softmax at that temperature, optionally restricted to
//     the top-k highest-prob tokens and/or the top-p nucleus, then a
//     multinomial draw from the sampler's seeded RNG.
func (s *Sampler) Sample(logits []float32) (int, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("decoder.Sample: empty logits")
	}
	if s.p.Temperature <= 0 {
		return argmax(logits), nil
	}
	probs := softmaxStable(logits, s.p.Temperature)
	if s.p.TopK > 0 || s.p.TopP > 0 {
		return s.drawFiltered(topFilter(probs, s.p.TopK, s.p.TopP)), nil
	}
	return s.drawFull(probs), nil
}

// drawFiltered samples one id from the renormalized (id, prob) pairs that
// survived top-k/top-p filtering (the trailing return guards float rounding).
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

// softmaxStable converts logits to probabilities (numerically stable). Ready
// for the M6 sampling path and the parity harness.
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

// indexedProb pairs a token id with its probability for top-k/top-p filtering.
type indexedProb struct {
	id int
	p  float64
}

// topFilter applies top-k then top-p to a probability vector, returning the
// surviving (id, renormalized-prob) pairs. Used by the M6 sampling path.
func topFilter(probs []float64, topK int, topP float64) []indexedProb {
	ips := make([]indexedProb, len(probs))
	for i, p := range probs {
		ips[i] = indexedProb{id: i, p: p}
	}
	sort.Slice(ips, func(a, b int) bool { return ips[a].p > ips[b].p })
	if topK > 0 && topK < len(ips) {
		ips = ips[:topK]
	}
	if topP > 0 && topP < 1 {
		var cum float64
		cut := len(ips)
		for i, ip := range ips {
			cum += ip.p
			if cum >= topP {
				cut = i + 1
				break
			}
		}
		ips = ips[:cut]
	}
	var sum float64
	for _, ip := range ips {
		sum += ip.p
	}
	for i := range ips {
		ips[i].p /= sum
	}
	return ips
}
