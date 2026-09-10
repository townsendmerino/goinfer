package decoder

import "testing"

// TestGemma4KVSrcAt_handDerived pins gemma4KVSrcAt (decoder/arch.go) — the shared cross-layer-KV
// redirect the resident CUDA bridge (KVSrcAtResident) reads, and the same algorithm
// forward_gemma4.go's own runLayersGemma4FromEmbed closure computes independently — against
// hand-traced expected values on a small, explicit layer_types config, not just "agrees with
// itself". 4 layers, sliding/full/sliding/full, last 2 shared:
//
//	firstShared = 4-2 = 2. Layers 0,1 own KV: layer 0 sliding -> lastSliding=0;
//	layer 1 full -> lastGlobal=1. Layer 2 (sliding, shared) -> kvSrc=lastSliding=0.
//	Layer 3 (full, shared) -> kvSrc=lastGlobal=1.
func TestGemma4KVSrcAt_handDerived(t *testing.T) {
	cfg := &Config{
		ModelType:        "gemma4",
		HiddenDim:        64,
		NumLayers:        4,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          16,
		GlobalHeadDim:    16,
		NumGlobalKVHeads: 2,
		IntermediateDim:  128,
		VocabSize:        256,
		RMSNormEps:       1e-6,
		RoPELocalBase:    10000,
		RoPEGlobalBase:   1000000,
		SlidingWindow:    8,
		LayerTypes:       []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"},
		SharedKVLayers:   2,
	}
	arch, _, err := gemma4Architecture(cfg)
	if err != nil {
		t.Fatalf("gemma4Architecture: %v", err)
	}
	want := []int{0, 1, 0, 1}
	for i, w := range want {
		if got := arch.gemma4KVSrcAt(i); got != w {
			t.Errorf("gemma4KVSrcAt(%d) = %d, want %d", i, got, w)
		}
	}
}

// TestGemma4KVSrcAt_noSharing confirms the SharedKVLayers==0 case (26B-A4B/31B, and every
// non-gemma4 family) returns identity for every layer — the resident bridge must never redirect
// a checkpoint that owns all its own KV.
func TestGemma4KVSrcAt_noSharing(t *testing.T) {
	cfg := &Config{
		ModelType:        "gemma4",
		HiddenDim:        64,
		NumLayers:        4,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          16,
		GlobalHeadDim:    16,
		NumGlobalKVHeads: 2,
		IntermediateDim:  128,
		VocabSize:        256,
		RMSNormEps:       1e-6,
		RoPELocalBase:    10000,
		RoPEGlobalBase:   1000000,
		SlidingWindow:    8,
		LayerTypes:       []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"},
		SharedKVLayers:   0,
	}
	arch, _, err := gemma4Architecture(cfg)
	if err != nil {
		t.Fatalf("gemma4Architecture: %v", err)
	}
	for i := range 4 {
		if got := arch.gemma4KVSrcAt(i); got != i {
			t.Errorf("gemma4KVSrcAt(%d) = %d, want %d (no sharing)", i, got, i)
		}
	}
}

// TestGemma4KVSrcAt_nonGemma4 confirms a non-gemma4 architecture returns identity unconditionally
// (a.gemma4 == nil short-circuit) — KVSrcAtResident must be a safe no-op for every other family.
func TestGemma4KVSrcAt_nonGemma4(t *testing.T) {
	var a Architecture
	a.NumLayers = 4
	for i := range 4 {
		if got := a.gemma4KVSrcAt(i); got != i {
			t.Errorf("gemma4KVSrcAt(%d) = %d, want %d (non-gemma4 arch)", i, got, i)
		}
	}
}
