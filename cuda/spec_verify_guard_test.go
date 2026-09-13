//go:build cuda

package cuda

import (
	"strings"
	"testing"
)

// TestDecodeVerifyDivergence_keyedOnTheSpike needs no GPU: it checks the predicate on a zero-value
// resident, which is exactly the state of every resident loaded without GOINFER_SPLITKV_VSUM_SPLIT.
func TestDecodeVerifyDivergence_keyedOnTheSpike(t *testing.T) {
	var r cudaResident
	if err := r.DecodeVerifyDivergence(); err != nil {
		t.Fatalf("stock resident (spike off) reports divergence: %v — speculative decoding would be "+
			"refused on every CUDA model", err)
	}

	// backend.go only sets skVsumSplit when the value parses to > 1; 1 is "no split" and must not
	// refuse either.
	r.skVsumSplit = 1
	if err := r.DecodeVerifyDivergence(); err != nil {
		t.Fatalf("skVsumSplit=1 is not a split, but reports divergence: %v", err)
	}

	r.skVsumSplit = 4
	err := r.DecodeVerifyDivergence()
	if err == nil {
		t.Fatal("spike enabled (S=4) but no divergence reported — spec decode would silently stop " +
			"being lossless")
	}
	// The operator reads this at startup; it has to name the knob that caused it.
	if !strings.Contains(err.Error(), "GOINFER_SPLITKV_VSUM_SPLIT") {
		t.Errorf("error does not name the env var an operator must unset: %v", err)
	}
}
