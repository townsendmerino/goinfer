//go:build goinfer_testhooks

package decoder

import (
	"path/filepath"
	"testing"
)

// TestPrefillReferenceRoundtrip guards WritePrefillReferenceForTest/ReadPrefillReferenceForTest's
// binary format against a silent drift between writer and reader — the exact class of bug that
// cost a full run once already (metal/prefill_gate_test.go's reused-buffer aliasing). Fast, not
// heavy: synthetic data, no checkpoint.
func TestPrefillReferenceRoundtrip(t *testing.T) {
	const vocab, n = 7, 3
	seedLogits := []float32{1, 2, 3, 4, 5, 6, 7}
	refTokens := []int{2, 0, 6}
	refLogits := [][]float32{
		{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7},
		{-1, -2, -3, -4, -5, -6, -7},
		{10, 20, 30, 40, 50, 60, 70},
	}

	path := filepath.Join(t.TempDir(), "roundtrip.bin")
	if err := WritePrefillReferenceForTest(path, seedLogits, refTokens, refLogits); err != nil {
		t.Fatalf("write: %v", err)
	}
	gotSeed, gotTokens, gotLogits, err := ReadPrefillReferenceForTest(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(gotSeed) != vocab {
		t.Fatalf("seed len = %d, want %d", len(gotSeed), vocab)
	}
	for i := range seedLogits {
		if gotSeed[i] != seedLogits[i] {
			t.Errorf("seed[%d] = %v, want %v", i, gotSeed[i], seedLogits[i])
		}
	}
	if len(gotTokens) != n {
		t.Fatalf("tokens len = %d, want %d", len(gotTokens), n)
	}
	for i := range refTokens {
		if gotTokens[i] != refTokens[i] {
			t.Errorf("token[%d] = %d, want %d", i, gotTokens[i], refTokens[i])
		}
	}
	if len(gotLogits) != n {
		t.Fatalf("logits rows = %d, want %d", len(gotLogits), n)
	}
	for i := range refLogits {
		if len(gotLogits[i]) != vocab {
			t.Fatalf("logits[%d] len = %d, want %d", i, len(gotLogits[i]), vocab)
		}
		for j := range refLogits[i] {
			if gotLogits[i][j] != refLogits[i][j] {
				t.Errorf("logits[%d][%d] = %v, want %v", i, j, gotLogits[i][j], refLogits[i][j])
			}
		}
	}
}
