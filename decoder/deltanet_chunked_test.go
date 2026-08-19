package decoder

import (
	"github.com/townsendmerino/aikit/linalg"
	"math"
	"math/rand"
	"testing"
)

// TestGatedDeltaNet_chunkedMatchesSequential proves the chunked scan is
// algebraically equivalent to the parity-proven sequential recurrence: over
// random weights/inputs and several chunk sizes (incl. 1, which must reduce
// exactly to the per-step path), the whole-sequence output matches to fp
// tolerance. This is the self-contained correctness gate for the kernel — no
// checkpoint/golden needed.
func TestGatedDeltaNet_chunkedMatchesSequential(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	uni := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32() - 0.5
		}
		return s
	}

	p := qwen35Params{ConvKernel: 4, KeyHeadDim: 8, ValueHeadDim: 8, NumKeyHeads: 2, NumValueHeads: 4}
	const hidden = 16
	const eps = 1e-6
	nk, nv := p.NumKeyHeads, p.NumValueHeads
	hk, hv := p.KeyHeadDim, p.ValueHeadDim
	keyDim, valueDim := hk*nk, hv*nv
	convDim := 2*keyDim + valueDim

	aLog := make([]float32, nv)
	for i := range aLog {
		aLog[i] = -2 * rng.Float32() // A_log in [-2,0] → gt = exp(g) in (0,1), stable
	}
	w := &deltaNetWeights{
		inProjQKV: linalg.WrapF32(uni(convDim*hidden), convDim, hidden),
		inProjZ:   linalg.WrapF32(uni(valueDim*hidden), valueDim, hidden),
		inProjB:   uni(nv * hidden),
		inProjA:   uni(nv * hidden),
		convW:     uni(convDim * p.ConvKernel),
		dtBias:    uni(nv),
		negExpA:   negExpAFromLog(aLog),
		normW:     uni(hv),
		outProj:   linalg.WrapF32(uni(hidden*valueDim), hidden, valueDim),
	}

	for _, seq := range []int{1, 5, 17, 40} {
		h := make([][]float32, seq)
		for t := range h {
			h[t] = uni(hidden)
		}
		want := gatedDeltaNet(&cpuBackend{}, h, w, p, hidden, eps) // sequential reference

		for _, chunk := range []int{1, 2, 3, 8, seq, seq + 5} {
			got := gatedDeltaNetChunked(&cpuBackend{}, h, w, p, hidden, eps, chunk)
			var maxAbs, dot, na, nb float64
			for t := range seq {
				for j := range hidden {
					g, wv := float64(got[t][j]), float64(want[t][j])
					if ad := math.Abs(g - wv); ad > maxAbs {
						maxAbs = ad
					}
					dot += g * wv
					na += g * g
					nb += wv * wv
				}
			}
			cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12)
			if maxAbs > 1e-4 {
				t.Errorf("seq=%d chunk=%d: maxAbs %.3e > 1e-4", seq, chunk, maxAbs)
			}
			if seq > 0 && cos < 0.99999 {
				t.Errorf("seq=%d chunk=%d: cosine %.8f < 0.99999", seq, chunk, cos)
			}
		}
	}
}
