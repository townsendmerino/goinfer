package decoder

import (
	"math"
	"testing"
)

// TestForwardN_matchesSequential checks the batched multi-position forward is
// numerically identical to running forward() one token at a time — the
// invariant the speculative verifier relies on (it must compute exactly the
// logits plain decode would). Skips without the model asset.
func TestForwardN_matchesSequential(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GINFER_PREQUANT_GGUF", err)
	}
	ids := []int{785, 264, 6573, 311, 1438, 279} // K=6 at positions 0..5
	K := len(ids)

	cseq := m.NewCache(K)
	seq := make([][]float32, K)
	for i, id := range ids {
		l, err := m.forward(id, cseq)
		if err != nil {
			t.Fatalf("seq forward %d: %v", i, err)
		}
		seq[i] = append([]float32(nil), l...)
	}

	cbat := m.NewCache(K)
	bat, err := m.forwardN(ids, cbat)
	if err != nil {
		t.Fatalf("forwardN: %v", err)
	}
	if len(bat) != K {
		t.Fatalf("forwardN returned %d logit vectors, want %d", len(bat), K)
	}
	if cseq.Pos() != cbat.Pos() || cbat.Pos() != K {
		t.Fatalf("cache positions: seq=%d batch=%d want %d", cseq.Pos(), cbat.Pos(), K)
	}

	for i := 0; i < K; i++ {
		if argmax(seq[i]) != argmax(bat[i]) {
			t.Fatalf("position %d: argmax seq=%d batch=%d", i, argmax(seq[i]), argmax(bat[i]))
		}
		var maxd float64
		for j := range seq[i] {
			if d := math.Abs(float64(seq[i][j] - bat[i][j])); d > maxd {
				maxd = d
			}
		}
		if maxd > 1e-4 {
			t.Errorf("position %d: max logit diff %.2e vs sequential (want ~0)", i, maxd)
		} else {
			t.Logf("position %d: argmax %d, max logit diff %.2e", i, argmax(bat[i]), maxd)
		}
	}
}

// TestTruncateTo rolls a cache back and confirms re-decoding from the truncated
// point matches a cache that only ever saw the kept prefix.
func TestTruncateTo(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GINFER_PREQUANT_GGUF", err)
	}
	ids := []int{785, 264, 6573, 311, 1438, 279}

	// Cache A: feed all 6, truncate back to 3, then forward a 7th token.
	ca := m.NewCache(8)
	for _, id := range ids {
		if _, err := m.forward(id, ca); err != nil {
			t.Fatal(err)
		}
	}
	ca.TruncateTo(3)
	if ca.Pos() != 3 {
		t.Fatalf("after TruncateTo(3): Pos=%d", ca.Pos())
	}
	la, err := m.forward(99, ca)
	if err != nil {
		t.Fatal(err)
	}

	// Cache B: only ever saw the first 3 tokens, then the same 7th token.
	cb := m.NewCache(8)
	for _, id := range ids[:3] {
		if _, err := m.forward(id, cb); err != nil {
			t.Fatal(err)
		}
	}
	lb, err := m.forward(99, cb)
	if err != nil {
		t.Fatal(err)
	}

	if argmax(la) != argmax(lb) {
		t.Fatalf("post-truncate argmax %d != fresh-prefix argmax %d", argmax(la), argmax(lb))
	}
	for j := range la {
		if math.Abs(float64(la[j]-lb[j])) > 1e-4 {
			t.Fatalf("post-truncate logits differ at %d: %g vs %g", j, la[j], lb[j])
		}
	}
}
