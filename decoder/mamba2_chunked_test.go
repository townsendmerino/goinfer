package decoder

import (
	"math"
	"math/rand"
	"testing"
)

// TestMamba2_chunkedMatchesSequential proves the chunked Mamba-2 scan is
// algebraically equivalent to the parity-proven sequential recurrence (mamba2Seq):
// over random weights/inputs and several chunk sizes (incl. 1, which must reduce
// exactly to the per-step path), the whole-sequence output matches to fp tolerance.
// Self-contained — no checkpoint/golden needed.
func TestMamba2_chunkedMatchesSequential(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	uni := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32() - 0.5
		}
		return s
	}

	p := mamba2Params{NHeads: 6, HeadDim: 8, DState: 8, NGroups: 2, DConv: 4, Hidden: 16, NormGroups: 2}
	const eps = 1e-6
	convDim, dInner := p.convDim(), p.dInner()

	aLog := make([]float32, p.NHeads)
	for i := range aLog {
		aLog[i] = -2 * rng.Float32() // A = -exp(aLog) ∈ [-1,-0.135]; dA ∈ (0,1), stable
	}
	w := &mamba2Weights{
		inProj:  uni(p.projDim() * p.Hidden),
		convW:   uni(convDim * p.DConv),
		convB:   uni(convDim),
		aLog:    aLog,
		d:       uni(p.NHeads),
		dtBias:  uni(p.NHeads),
		normW:   uni(dInner),
		outProj: uni(p.Hidden * dInner),
	}

	for _, seq := range []int{1, 5, 17, 40} {
		h := make([][]float32, seq)
		for t := range h {
			h[t] = uni(p.Hidden)
		}
		want := mamba2Seq(h, w, p, eps) // sequential reference

		for _, chunk := range []int{1, 2, 3, 8, seq, seq + 5} {
			got := mamba2Chunked(h, w, p, eps, chunk)
			var maxAbs, dot, na, nb float64
			for t := range seq {
				for j := range p.Hidden {
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
			if cos < 0.99999 {
				t.Errorf("seq=%d chunk=%d: cosine %.8f < 0.99999", seq, chunk, cos)
			}
		}
	}
}
