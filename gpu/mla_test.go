//go:build gpu

package gpu

import (
	"math"
	"testing"
)

// TestMLALatentStore_parity gates Lever C4b's latent append: kvA-norm the rank latent +
// decoupled-RoPE the key, mirroring decoder.cache.AppendLatent's normalized/roped form
// (cn ‖ krj). Both V3 GPT-J interleave and plain NeoX are covered against a CPU f64
// reference (rmsNorm + mlaRope) at a nonzero position so the rope-at-pos path is real.
func TestMLALatentStore_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const rank, qkRope, pos = 512, 64, 7
	eps := float32(1e-6)
	half := qkRope / 2
	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e4, float64(2*d)/float64(qkRope)))
	}
	for _, interleave := range []bool{true, false} {
		name := "neox"
		if interleave {
			name = "interleave"
		}
		t.Run(name, func(t *testing.T) {
			kvDown := randMat(rank+qkRope, 11)
			normW := randMat(rank, 12)

			// CPU reference: RMSNorm(kvDown[:rank], normW) ‖ mlaRope(kvDown[rank:], pos).
			ref := make([]float32, rank+qkRope)
			var ss float64
			for i := 0; i < rank; i++ {
				ss += float64(kvDown[i]) * float64(kvDown[i])
			}
			inv := float32(1.0 / math.Sqrt(ss/float64(rank)+float64(eps)))
			for i := 0; i < rank; i++ {
				ref[i] = kvDown[i] * inv * normW[i]
			}
			key := append([]float32(nil), kvDown[rank:]...)
			if interleave {
				tmp := make([]float32, qkRope)
				for i := 0; i < half; i++ {
					tmp[i] = key[2*i]
					tmp[half+i] = key[2*i+1]
				}
				copy(key, tmp)
			}
			for d := 0; d < half; d++ {
				theta := float64(pos) * float64(invFreq[d])
				c, s := float32(math.Cos(theta)), float32(math.Sin(theta))
				x1, x2 := key[d], key[half+d]
				ref[rank+d] = x1*c - x2*s
				ref[rank+half+d] = x2*c + x1*s
			}

			got, err := ctx.MLALatentStore(kvDown, normW, invFreq, rank, qkRope, pos, eps, 1.0, interleave)
			if err != nil {
				t.Fatalf("MLALatentStore: %v", err)
			}
			cos, maxAbs := cosine(got, ref)
			t.Logf("%s: cosine=%.8f maxAbs=%.3e", name, cos, maxAbs)
			if cos < 0.9999 || maxAbs > 1e-3 {
				t.Errorf("MLALatentStore diverges: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
			}
		})
	}
}

// TestMLAHeadMatvec_parity gates Lever C4b's per-head block-diagonal matvec — the W_UK
// absorb (a=q with a wider stride than K, so the qk_rope tail is skipped) and the W_UV
// lift (aStride==K). A CPU f64 reference over random per-head a/w must match (cosine ~1.0).
func TestMLAHeadMatvec_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	cases := []struct {
		name              string
		nH, N, K, aStride int
	}{
		{"absorb_WUK", 8, 512, 128, 192}, // aStride=qkHead=qkNope(128)+qkRope(64); K=qkNope
		{"lift_WUV", 8, 128, 512, 512},   // aStride==K=rank
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := randMat(tc.nH*tc.aStride, 21)
			w := randMat(tc.nH*tc.N*tc.K, 22)
			ref := make([]float32, tc.nH*tc.N)
			for h := 0; h < tc.nH; h++ {
				for n := 0; n < tc.N; n++ {
					var s float64
					for k := 0; k < tc.K; k++ {
						s += float64(a[h*tc.aStride+k]) * float64(w[(h*tc.N+n)*tc.K+k])
					}
					ref[h*tc.N+n] = float32(s)
				}
			}
			got, err := ctx.MLAHeadMatvec(a, w, tc.nH, tc.N, tc.K, tc.aStride)
			if err != nil {
				t.Fatalf("MLAHeadMatvec: %v", err)
			}
			cos, maxAbs := cosine(got, ref)
			t.Logf("%s: cosine=%.8f maxAbs=%.3e", tc.name, cos, maxAbs)
			if cos < 0.9999 || maxAbs > 1e-3 {
				t.Errorf("MLAHeadMatvec diverges: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
			}
		})
	}
}

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
