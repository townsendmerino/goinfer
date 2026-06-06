package decoder

import (
	"math"
	"testing"
)

// TestGemma4Architecture pins the Gemma 4 descriptor against the verified 12B
// config (google/gemma-4-12B-it…/config.json): Gemma-3-style stack PLUS the
// per-layer attention deltas. Pure unit test — no model asset. It's the contract
// the (WIP) forward pass must satisfy; the GGUF loader returns a clear error
// until that lands (see docs/internal/task-gemma4-support.md).
func TestGemma4Architecture(t *testing.T) {
	// Verified 12B (gemma4_unified_text) values.
	cfg := &Config{
		ModelType:            "gemma4_unified_text",
		HiddenDim:            3840,
		NumLayers:            48,
		NumHeads:             16,
		NumKVHeads:           8,
		HeadDim:              256,
		IntermediateDim:      15360,
		VocabSize:            262144,
		RMSNormEps:           1e-6,
		RoPELocalBase:        10000,
		RoPEGlobalBase:       1000000,
		SlidingWindow:        1024,
		SlidingWindowPattern: 6, // 5 local : 1 global
		QueryPreAttnScalar:   256,
		GlobalHeadDim:        512,
		NumGlobalKVHeads:     1,
		AttentionKEqV:        true,
		PartialRotaryFactor:  0.25, // p-RoPE on the global layers
	}

	arch, schema, err := gemma4Architecture(cfg)
	if err != nil {
		t.Fatalf("gemma4Architecture: %v", err)
	}
	if schema == nil {
		t.Fatal("nil tensor schema")
	}

	// Shared Gemma stack (matches gemma3).
	checks := []struct {
		name string
		got  any
		want any
	}{
		{"Name", arch.Name, "gemma4"},
		{"HiddenDim", arch.HiddenDim, 3840},
		{"NumLayers", arch.NumLayers, 48},
		{"NumHeads", arch.NumHeads, 16},
		{"NumKVHeads", arch.NumKVHeads, 8},
		{"HeadDim (local)", arch.HeadDim, 256},
		{"Norm", arch.Norm, NormRMS},
		{"RMSAddOne", arch.RMSAddOne, false}, // Gemma4RMSNorm is plain x*weight (init 1), not x*(1+w)
		{"NormPlacement", arch.NormPlacement, NormSandwich4},
		{"Act", arch.Act, ActGeluTanh},
		{"QKNorm", arch.QKNorm, true},
		{"SlidingWindow", arch.SlidingWindow, 1024},
		{"RoPELocalBase", arch.RoPELocalBase, float64(10000)},
		{"RoPEGlobalBase", arch.RoPEGlobalBase, float64(1000000)},
		{"TiedLMHead", arch.TiedLMHead, true},
		{"FinalLogitSoftcap", arch.FinalLogitSoftcap, float64(0)},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}

	// Gemma 4 text attention uses scale 1.0 (q/k-norm absorb scaling), not
	// Gemma 3's query_pre_attn_scalar^-0.5.
	if arch.AttnScale != 1.0 {
		t.Errorf("AttnScale = %v, want 1.0", arch.AttnScale)
	}
	// EmbedScale = sqrt(hidden).
	if want := math.Sqrt(3840); arch.EmbedScale != want {
		t.Errorf("EmbedScale = %v, want %v", arch.EmbedScale, want)
	}

	// The Gemma 4 per-layer deltas.
	g := arch.gemma4
	if g == nil {
		t.Fatal("arch.gemma4 is nil — descriptor didn't carry the per-layer deltas")
	}
	if g.GlobalHeadDim != 512 {
		t.Errorf("GlobalHeadDim = %d, want 512", g.GlobalHeadDim)
	}
	if g.NumGlobalKVHeads != 1 {
		t.Errorf("NumGlobalKVHeads = %d, want 1", g.NumGlobalKVHeads)
	}
	if g.GlobalRotaryDim != 128 { // 0.25 * 512
		t.Errorf("GlobalRotaryDim = %d, want 128 (0.25*512)", g.GlobalRotaryDim)
	}
	if !g.KVShared {
		t.Error("KVShared = false, want true (attention_k_eq_v)")
	}

	// 5:1 interleave from the pattern: layers 5,11,… global; 0,1,… local.
	if arch.layerIsGlobal == nil {
		t.Fatal("layerIsGlobal not set")
	}
	if !arch.layerIsGlobal(5) {
		t.Error("layer 5 should be global (full attention)")
	}
	if arch.layerIsGlobal(0) {
		t.Error("layer 0 should be local (sliding)")
	}
	if !arch.layerIsGlobal(47) { // last layer global (pattern: (i+1)%6==0)
		t.Error("layer 47 should be global")
	}
}
