package decoder

import "testing"

// TestGGUFInvPermute checks the q/k row un-permutation on a tiny known case:
// one head, head_dim 4 (half=2), in=1. llama.cpp stores rows in interleaved
// order (2j+s); HF wants half-major order (s*half+j). So GGUF [a,b,c,d] →
// HF [a,c,b,d].
func TestGGUFInvPermute(t *testing.T) {
	gguf := []float32{10, 11, 12, 13} // rows a,b,c,d (in=1)
	got := ggufInvPermute(gguf, 4 /*out*/, 1 /*in*/, 1 /*nHead*/)
	want := []float32{10, 12, 11, 13}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("invPermute[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestGGUFInvPermute_bijection: the permutation must be a bijection over rows
// (no row dropped or duplicated), for a non-trivial multi-head shape.
func TestGGUFInvPermute_bijection(t *testing.T) {
	const nHead, hd, in = 4, 8, 3
	out := nHead * hd
	src := make([]float32, out*in)
	for r := range out {
		for c := range in {
			src[r*in+c] = float32(r*100 + c) // row r encodes its index
		}
	}
	got := ggufInvPermute(src, out, in, nHead)
	seen := make([]bool, out)
	for r := range out {
		srcRow := int(got[r*in]) / 100 // recover which source row landed here
		if srcRow < 0 || srcRow >= out || seen[srcRow] {
			t.Fatalf("row %d came from invalid/duplicate source %d", r, srcRow)
		}
		seen[srcRow] = true
		// Whole row moves together.
		for c := range in {
			if got[r*in+c] != float32(srcRow*100+c) {
				t.Errorf("row %d col %d = %v, not a clean row move", r, c, got[r*in+c])
			}
		}
	}
}
