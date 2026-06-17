//go:build gpu

package gpu

import (
	"math"
	"testing"
)

// TestMLAAttn_parity gates Lever C4a: the absorb-path MLA rank-space attention kernel.
// It mirrors decoder.mlaAttentionAbsorb steps 4b+5 — score each cached latent by the
// full latDim dot (qNopeAbs·cn + qRope·krj), per-head softmax, then collapse V to the
// rank-space weighted latent sum wsum[h] = Σ_j p_j·cn_j. A CPU f64 reference over the
// same random qAbs/latent must match the GPU online-softmax kernel (cosine ~1.0). Uses
// DeepSeek-V3-ish dims (rank 512 > the 128-lane width) so the strided score/value paths
// are exercised, plus a small-rank case so the rank ≤ WG path is covered too.
func TestMLAAttn_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	cases := []struct {
		name                    string
		nH, rank, qkRope, nKeys int
	}{
		{"v3_rank512", 8, 512, 64, 40},
		{"small_rank96", 6, 96, 32, 17},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			latDim := tc.rank + tc.qkRope
			scale := float32(1.0 / math.Sqrt(float64(tc.rank+tc.qkRope)))
			qAbs := randMat(tc.nH*latDim, 1)
			lat := randMat(tc.nKeys*latDim, 2)

			// CPU reference (f64): per head, softmax over the latDim dot, V = cn = lat[:rank].
			ref := make([]float32, tc.nH*tc.rank)
			for h := 0; h < tc.nH; h++ {
				sc := make([]float64, tc.nKeys)
				mx := math.Inf(-1)
				for j := 0; j < tc.nKeys; j++ {
					var dot float64
					for d := 0; d < latDim; d++ {
						dot += float64(qAbs[h*latDim+d]) * float64(lat[j*latDim+d])
					}
					sc[j] = dot * float64(scale)
					if sc[j] > mx {
						mx = sc[j]
					}
				}
				var sum float64
				for j := range sc {
					sc[j] = math.Exp(sc[j] - mx)
					sum += sc[j]
				}
				for j := 0; j < tc.nKeys; j++ {
					w := sc[j] / sum
					for c := 0; c < tc.rank; c++ {
						ref[h*tc.rank+c] += float32(w * float64(lat[j*latDim+c]))
					}
				}
			}

			got, err := ctx.MLAAttn(qAbs, lat, tc.nH, latDim, tc.rank, tc.nKeys, scale)
			if err != nil {
				t.Fatalf("MLAAttn: %v", err)
			}
			cos, maxAbs := cosine(got, ref)
			t.Logf("%s: cosine=%.8f maxAbs=%.3e", tc.name, cos, maxAbs)
			if cos < 0.9999 || maxAbs > 1e-3 {
				t.Errorf("MLAAttn diverges: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
			}
		})
	}
}
