//go:build darwin

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
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
			forwardNMatchesSequential(t, loadResident, tmHidden, tc.n, tc.startPos)
		})
	}
}

// TestForwardN_DeltaNetBitIdenticalParity is the same comparison on a model with Gated-DeltaNet layers (qwen35-tiny, int4): the
// batched ForwardN runs a DeltaNet layer on row m's residual, as sequential Forward does.
func TestForwardN_DeltaNetBitIdenticalParity(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	loadResident := func() *metalResident {
		m, err := decoder.Load("../../testdata/qwen35-tiny", decoder.Options{Quant: "int4"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		r, err := buildResident(m)
		if err != nil {
			t.Fatalf("build resident: %v", err)
		}
		if !slices.ContainsFunc(r.layers, func(L residLayer) bool { return L.delta != nil }) {
			t.Fatal("the fixture builds no DeltaNet layer")
		}
		return &metalResident{r: r, hidden: r.H}
	}
	probe := loadResident()
	hidden := probe.hidden
	probe.Close()
	for _, n := range []int{2, 4, 8} {
		for _, startPos := range []int{0, 4} {
			t.Run(fmt.Sprintf("N=%d_startPos=%d", n, startPos), func(t *testing.T) {
				forwardNMatchesSequential(t, loadResident, hidden, n, startPos)
			})
		}
	}
}

// forwardNMatchesSequential feeds n seeded embeddings of width hidden at consecutive positions from startPos to a fresh resident's
// ForwardN and to another's sequential Forward (both pre-seeded to startPos), and fails on any logit whose bits differ.
func forwardNMatchesSequential(t *testing.T, loadResident func() *metalResident, hidden, n, startPos int) {
	t.Helper()
	totalLen := startPos + n
	rng := rand.New(rand.NewSource(int64(n*100 + startPos)))

	// Synthetic embeddings for the full sequence up to startPos+n.
	allEmbs := make([][]float32, totalLen)
	for i := range allEmbs {
		allEmbs[i] = make([]float32, hidden)
		for j := range allEmbs[i] {
			allEmbs[i][j] = float32(rng.NormFloat64()) * 0.1
		}
	}

	// 1. Sequential ground truth, its KV cache and recurrent state pre-seeded to startPos.
	seqRes := loadResident()
	defer seqRes.Close()
	for p := 0; p < startPos; p++ {
		if _, err := seqRes.Forward(allEmbs[p], p); err != nil {
			t.Fatalf("seq pre-seed at %d: %v", p, err)
		}
	}
	seqLogits := make([][]float32, n)
	for i := 0; i < n; i++ {
		pos := startPos + i
		l, err := seqRes.Forward(allEmbs[pos], pos)
		if err != nil {
			t.Fatalf("seq Forward at %d: %v", pos, err)
		}
		seqLogits[i] = append([]float32(nil), l...)
	}

	// 2. Batched single-command-buffer candidate.
	batchRes := loadResident()
	defer batchRes.Close()
	for p := 0; p < startPos; p++ {
		if _, err := batchRes.Forward(allEmbs[p], p); err != nil {
			t.Fatalf("batch pre-seed at %d: %v", p, err)
		}
	}
	batchLogits, err := batchRes.ForwardN(allEmbs[startPos:startPos+n], startPos)
	if err != nil {
		t.Fatalf("batched ForwardN: %v", err)
	}
	if len(batchLogits) != n {
		t.Fatalf("got %d logit rows, want %d", len(batchLogits), n)
	}

	// Bit-exact parity for every logit.
	totalLogits := 0
	for i := 0; i < n; i++ {
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
	t.Logf("PASS: N=%d, startPos=%d: %d logits bit-identical", n, startPos, totalLogits)
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
