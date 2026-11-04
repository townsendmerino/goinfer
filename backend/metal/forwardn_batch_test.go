//go:build darwin

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestForwardN_BitIdenticalParity verifies that the single-command-buffer batched ForwardN
// produces 100% bit-identical logits to sequential Forward calls across various batch sizes
// and starting positions on a real Metal resident model.
func TestForwardN_BitIdenticalParity(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}

	w := genTinyWeights(rand.New(rand.NewSource(42)))
	dir := t.TempDir()
	writeDense(t, dir, w)

	loadResident := func() *metalResident {
		m, err := decoder.Load(dir, decoder.Options{Quant: "int8int8"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		r, err := buildResident(m)
		if err != nil {
			t.Fatalf("build resident: %v", err)
		}
		return &metalResident{r: r, hidden: r.H}
	}

	// Test across varying batch sizes N and starting context positions.
	testCases := []struct {
		n        int
		startPos int
	}{
		{n: 1, startPos: 0},
		{n: 2, startPos: 0},
		{n: 4, startPos: 0},
		{n: 8, startPos: 0},
		{n: 12, startPos: 0},
		{n: 4, startPos: 6},
		{n: 8, startPos: 4},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("N=%d_startPos=%d", tc.n, tc.startPos), func(t *testing.T) {
			totalLen := tc.startPos + tc.n
			rng := rand.New(rand.NewSource(int64(tc.n*100 + tc.startPos)))

			// Generate synthetic embeddings for the full sequence up to startPos+n.
			allEmbs := make([][]float32, totalLen)
			for i := range allEmbs {
				allEmbs[i] = make([]float32, tmHidden)
				for j := range allEmbs[i] {
					allEmbs[i][j] = float32(rng.NormFloat64()) * 0.1
				}
			}

			// 1. Sequential ground truth:
			seqRes := loadResident()
			defer seqRes.Close()
			// Pre-seed KV cache up to startPos if startPos > 0
			for p := 0; p < tc.startPos; p++ {
				if _, err := seqRes.Forward(allEmbs[p], p); err != nil {
					t.Fatalf("seq pre-seed at %d: %v", p, err)
				}
			}
			seqLogits := make([][]float32, tc.n)
			for i := 0; i < tc.n; i++ {
				pos := tc.startPos + i
				l, err := seqRes.Forward(allEmbs[pos], pos)
				if err != nil {
					t.Fatalf("seq Forward at %d: %v", pos, err)
				}
				seqLogits[i] = append([]float32(nil), l...)
			}

			// 2. Batched single-command-buffer candidate:
			batchRes := loadResident()
			defer batchRes.Close()
			for p := 0; p < tc.startPos; p++ {
				if _, err := batchRes.Forward(allEmbs[p], p); err != nil {
					t.Fatalf("batch pre-seed at %d: %v", p, err)
				}
			}
			verifyEmbs := allEmbs[tc.startPos : tc.startPos+tc.n]
			batchLogits, err := batchRes.ForwardN(verifyEmbs, tc.startPos)
			if err != nil {
				t.Fatalf("batched ForwardN: %v", err)
			}

			if len(batchLogits) != tc.n {
				t.Fatalf("got %d logit rows, want %d", len(batchLogits), tc.n)
			}

			// Verify BIT-EXACT parity for every logit.
			totalLogits := 0
			for i := 0; i < tc.n; i++ {
				if len(batchLogits[i]) != len(seqLogits[i]) {
					t.Fatalf("row %d length mismatch: batch %d vs seq %d", i, len(batchLogits[i]), len(seqLogits[i]))
				}
				for v := range seqLogits[i] {
					sBits := math.Float32bits(seqLogits[i][v])
					bBits := math.Float32bits(batchLogits[i][v])
					if sBits != bBits {
						t.Fatalf("row %d logit[%d] MISMATCH: seq=%f (0x%08x) vs batch=%f (0x%08x)",
							i, v, seqLogits[i][v], sBits, batchLogits[i][v], bBits)
					}
					totalLogits++
				}
			}
			t.Logf("PASS: N=%d, startPos=%d: %d logits bit-identical", tc.n, tc.startPos, totalLogits)
		})
	}
}

// BenchmarkForwardN_BatchVsSeq compares the execution latency of single-command-buffer
// ForwardN vs sequential Forward loop across batch sizes N in [1, 2, 4, 8, 16].
func BenchmarkForwardN_BatchVsSeq(b *testing.B) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		b.Skipf("no metal device: %v", err)
	}

	w := genTinyWeights(rand.New(rand.NewSource(123)))
	dir := b.TempDir()
	writeDense(&testing.T{}, dir, w)

	loadResident := func() *metalResident {
		m, err := decoder.Load(dir, decoder.Options{Quant: "int8int8"})
		if err != nil {
			b.Fatalf("load: %v", err)
		}
		r, err := buildResident(m)
		if err != nil {
			b.Fatalf("build resident: %v", err)
		}
		return &metalResident{r: r, hidden: r.H}
	}

	batchRes := loadResident()
	defer batchRes.Close()

	for _, n := range []int{1, 2, 4, 8, 16} {
		rng := rand.New(rand.NewSource(int64(n)))
		embs := make([][]float32, n)
		for i := range embs {
			embs[i] = make([]float32, tmHidden)
			for j := range embs[i] {
				embs[i][j] = float32(rng.NormFloat64()) * 0.1
			}
		}

		b.Run(fmt.Sprintf("Seq_N%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for p, emb := range embs {
					if _, err := batchRes.Forward(emb, p); err != nil {
						b.Fatalf("seq Forward: %v", err)
					}
				}
			}
		})

		b.Run(fmt.Sprintf("Batch_N%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := batchRes.ForwardN(embs, 0); err != nil {
					b.Fatalf("batched ForwardN: %v", err)
				}
			}
		})
	}
}
