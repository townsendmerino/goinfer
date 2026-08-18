package decoder

import "testing"

// TestQwen3Next_descriptor verifies qwen3NextArchitecture against the REAL
// released config (Qwen/Qwen3-Next-80B-A3B-Instruct config.json, fetched
// directly), not a hand-crafted fixture — this specifically catches the two
// real config-shape deltas from qwen3_5_moe: layer_types absent entirely
// (computed from full_attention_interval) and partial_rotary_factor being a
// top-level field with no rope_parameters object at all (unlike qwen3_5_moe,
// which nests it).
func TestQwen3Next_descriptor(t *testing.T) {
	cfg := &Config{
		ModelType: "qwen3_next", HiddenDim: 2048, NumLayers: 48, NumHeads: 16, NumKVHeads: 2, HeadDim: 256,
		IntermediateDim: 5120, VocabSize: 151936, RMSNormEps: 1e-06,
		RoPEGlobalBase: 10000000, PartialRotaryFactor: 0.25, FullAttentionInterval: 4,
		LinearConvKernelDim: 4, LinearKeyHeadDim: 128, LinearValueHeadDim: 128,
		LinearNumKeyHeads: 16, LinearNumValueHeads: 32,
		NumExperts: 512, NumExpertsPerTok: 10, MoeIntermediateSize: 512, SharedExpertIntermediateSize: 512,
	}
	tp := true
	cfg.NormTopKProb = &tp

	arch, schema, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if schema != &qwen35TensorSchema {
		t.Errorf("schema = %p, want &qwen35TensorSchema (rides the existing qwen3_5_moe loader, no new schema)", schema)
	}
	if arch.qwen35 == nil {
		t.Fatal("arch.qwen35 is nil — DeltaNet params not populated")
	}
	if arch.MoE == nil {
		t.Fatal("arch.MoE is nil")
	}

	// The layer-types delta: no layer_types in the real config, computed from
	// full_attention_interval=4. Verified formula: (i+1)%4==0 -> full_attention,
	// 0-indexed, so layer 3 (not layer 4) is the first full-attention layer.
	if len(cfg.LayerTypes) != 48 {
		t.Fatalf("normalizeQwen3NextLayerTypes: LayerTypes has %d entries, want 48", len(cfg.LayerTypes))
	}
	wantFull := map[int]bool{3: true, 7: true, 11: true, 47: true}
	for i, want := range wantFull {
		if got := arch.layerIsGlobal(i); got != want {
			t.Errorf("layerIsGlobal(%d) = %v, want %v", i, got, want)
		}
	}
	for _, i := range []int{0, 1, 2, 4, 5, 6} {
		if arch.layerIsGlobal(i) {
			t.Errorf("layerIsGlobal(%d) = true, want false (linear_attention)", i)
		}
		if !arch.layerIsLinear(i) {
			t.Errorf("layerIsLinear(%d) = false, want true", i)
		}
	}

	// The rotary delta: partial_rotary_factor is top-level (0.25), not nested in
	// rope_parameters — confirm it actually resolved, not silently dropped to 0.
	if want := int(0.25 * 256); arch.RotaryDim != want {
		t.Errorf("RotaryDim = %d, want %d (0.25 * head_dim=256, the top-level partial_rotary_factor)", arch.RotaryDim, want)
	}
	if arch.RoPEGlobalBase != 10000000 {
		t.Errorf("RoPEGlobalBase = %v, want 10000000 (rope_theta, the flat/non-nested path)", arch.RoPEGlobalBase)
	}

	m := arch.MoE
	switch {
	case m.NumExperts != 512:
		t.Errorf("NumExperts=%d, want 512", m.NumExperts)
	case m.TopK != 10:
		t.Errorf("TopK=%d, want 10", m.TopK)
	case m.IntermediateDim != 512:
		t.Errorf("IntermediateDim=%d, want 512", m.IntermediateDim)
	case m.SharedIntermediateDim != 512:
		t.Errorf("SharedIntermediateDim=%d, want 512", m.SharedIntermediateDim)
	case !m.NormTopKProb:
		t.Error("NormTopKProb=false, want true")
	}
	if !arch.RMSAddOne {
		t.Error("RMSAddOne=false, want true (Qwen3NextRMSNorm(Gemma3RMSNorm): pass — inherits Gemma's (1+w))")
	}
	dn := arch.qwen35
	switch {
	case dn.ConvKernel != 4:
		t.Errorf("ConvKernel=%d, want 4", dn.ConvKernel)
	case dn.KeyHeadDim != 128:
		t.Errorf("KeyHeadDim=%d, want 128", dn.KeyHeadDim)
	case dn.ValueHeadDim != 128:
		t.Errorf("ValueHeadDim=%d, want 128", dn.ValueHeadDim)
	case dn.NumKeyHeads != 16:
		t.Errorf("NumKeyHeads=%d, want 16", dn.NumKeyHeads)
	case dn.NumValueHeads != 32:
		t.Errorf("NumValueHeads=%d, want 32", dn.NumValueHeads)
	}
}

// TestQwen3Next_flatRopeNoRopeParameters confirms the real failure mode this
// family would have hit if qwen3NextArchitecture just delegated to
// qwen35Architecture unmodified: parseRopeSpec hard-errors on an empty
// rope_parameters, and the real Qwen3-Next config never carries one at all.
func TestQwen3Next_flatRopeNoRopeParameters(t *testing.T) {
	cfg := &Config{
		ModelType: "qwen3_next", HiddenDim: 64, NumLayers: 4, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateDim: 32, VocabSize: 64, RMSNormEps: 1e-06,
		RoPEGlobalBase: 1000000, FullAttentionInterval: 4,
		LinearConvKernelDim: 4, LinearKeyHeadDim: 8, LinearValueHeadDim: 8,
		LinearNumKeyHeads: 2, LinearNumValueHeads: 4,
		NumExperts: 4, NumExpertsPerTok: 2, MoeIntermediateSize: 16, SharedExpertIntermediateSize: 16,
	}
	// RopeParameters deliberately left nil/empty — matches the real released config.
	if len(cfg.RopeParameters) != 0 {
		t.Fatalf("test setup: RopeParameters should be empty, got %d bytes", len(cfg.RopeParameters))
	}
	if _, _, err := resolveArchitecture(cfg); err != nil {
		t.Fatalf("resolveArchitecture with no rope_parameters (the real config's shape) should succeed via the flat path, got: %v", err)
	}
}
