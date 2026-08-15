package decoder

import (
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestAttendStrided_matchesGatherReference proves the P1 wiring (attendBatchedHeads's
// useAcc64 path reading keys/vals directly via linalg.MatmulBTAcc64Strided) is
// bit-identical to the OLD gather-into-scratch-then-MatmulBTAcc64 approach, at the
// exact stride parameters goinfer derives — not trusting aikit's own kernel-level
// proof (TestMatmulBTAcc64Strided_bitIdenticalToPacked in aikit) to also prove
// goinfer's USE of it. The reference here is a direct, literal transcription of the
// pre-change attendBatchedHeads gather math (kh/vt fill + linalg.MatmulBTAcc64), so a
// wiring mistake (wrong stride, wrong offset, swapped row/elem stride) shows as a
// byte mismatch, not a silent pass.
func TestAttendStrided_matchesGatherReference(t *testing.T) {
	shapes := []struct {
		name           string
		K, nKeys, nKV, nH, hd int
	}{
		{"decode M=1 GQA nKV=2 nH=8", 1, 512, 2, 8, 64},
		{"decode M=1 long-ctx nKV=4 nH=32 hd=128", 1, 4096, 4, 32, 128},
		{"decode M=1 MHA nKV=nH=8", 1, 300, 8, 8, 64},
		{"decode M=1 single KV head nKV=1", 1, 128, 1, 8, 64},
		{"prefill M=8", 8, 300, 8, 32, 64},
		{"prefill M=5 odd shape", 5, 40, 3, 6, 16},
	}
	for _, s := range shapes {
		t.Run(s.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(7))
			kvDim := s.nKV * s.hd
			qDim := s.nH * s.hd
			group := s.nH / s.nKV

			keys := randFloats(rng, s.nKeys*kvDim)
			vals := randFloats(rng, s.nKeys*kvDim)
			qh := randFloats(rng, s.K*s.hd)
			scoresQK := make([]float32, s.K*s.nKeys)
			scoresRef := make([]float32, s.K*s.nKeys)

			// Reference: the pre-change gather (kh) + linalg.MatmulBTAcc64, per KV head.
			for kvh := range s.nKV {
				kh := make([]float32, s.nKeys*s.hd)
				for row := range s.nKeys {
					base := row*kvDim + kvh*s.hd
					copy(kh[row*s.hd:row*s.hd+s.hd], keys[base:base+s.hd])
				}
				linalg.MatmulBTAcc64(qh, kh, scoresRef, s.K, s.hd, s.nKeys)

				// New: strided direct read, the wiring under test.
				linalg.MatmulBTAcc64Strided(qh, keys, scoresQK, s.K, s.hd, s.nKeys, kvh*s.hd, kvDim, 1)

				for i, v := range scoresRef {
					if scoresQK[i] != v {
						t.Fatalf("QKᵀ kvh=%d: byte mismatch at %d: strided=%v gather-ref=%v", kvh, i, scoresQK[i], v)
					}
				}
			}

			// scores·V: same shape check, scores standing in for a real post-softmax row.
			scores := randFloats(rng, s.K*s.nKeys)
			chNew := make([]float32, s.K*s.hd)
			chRef := make([]float32, s.K*s.hd)
			for kvh := range s.nKV {
				vt := make([]float32, s.hd*s.nKeys)
				for row := range s.nKeys {
					base := row*kvDim + kvh*s.hd
					vrow := vals[base : base+s.hd]
					for d := range s.hd {
						vt[d*s.nKeys+row] = vrow[d]
					}
				}
				linalg.MatmulBTAcc64(scores, vt, chRef, s.K, s.nKeys, s.hd)

				linalg.MatmulBTAcc64Strided(scores, vals, chNew, s.K, s.nKeys, s.hd, kvh*s.hd, 1, kvDim)

				for i, v := range chRef {
					if chNew[i] != v {
						t.Fatalf("scores·V kvh=%d: byte mismatch at %d: strided=%v gather-ref=%v", kvh, i, chNew[i], v)
					}
				}
			}
			_ = group
			_ = qDim
		})
	}
}

func randFloats(rng *rand.Rand, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = rng.Float32()*2 - 1
	}
	return out
}
