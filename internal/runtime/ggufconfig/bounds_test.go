package ggufconfig

import (
	"testing"
)

// End to end: laguna's block_count reaches make([]string, numLayers) and must be bounded by ggufLayerCount like
// granitehybrid, nemotron and llama4.
func TestGGUFLayerCount_boundsTheLagunaPath(t *testing.T) {
	if _, err := ggufLayerCount(1 << 36); err == nil {
		t.Fatal("ggufLayerCount accepted 2^36")
	}
	if n, err := ggufLayerCount(48); err != nil || n != 48 {
		t.Fatalf("ggufLayerCount(48) = %d, %v — a real layer count must pass", n, err)
	}
}
