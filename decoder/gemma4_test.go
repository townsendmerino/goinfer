package decoder

import (
	"math"
	"testing"
)

// TestGemma4Architecture pins the Gemma 4 descriptor against the verified 12B
// config (google/gemma-4-12B-it…/config.json): Gemma-3-style stack PLUS the
// per-layer attention deltas. Pure unit test — no model asset. It's the contract
// the forward pass (runLayersGemma4) satisfies; end-to-end parity is gated by
// TestGemma4_12B_logitParity.
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

// TestGemma4MoEDescriptor pins Phase 1b: enable_moe_block turns on the parallel
// dense+MoE sub-block (Gemma 4 26B-A4B) while the dense E2B/E4B/12B variants leave
// MoE nil and are byte-unchanged. Descriptor-only — the forward is Phase 2.
func TestGemma4MoEDescriptor(t *testing.T) {
	base := func() *Config {
		return &Config{
			ModelType: "gemma4", HiddenDim: 2816, NumLayers: 30, NumHeads: 16, NumKVHeads: 8,
			HeadDim: 256, IntermediateDim: 2112, VocabSize: 262144, RMSNormEps: 1e-6,
			RoPELocalBase: 10000, RoPEGlobalBase: 1000000, SlidingWindow: 1024,
			SlidingWindowPattern: 6, GlobalHeadDim: 512, NumGlobalKVHeads: 2,
			AttentionKEqV: true, PartialRotaryFactor: 0.25, FinalLogitSoftcap: 30,
		}
	}

	// MoE variant: enable_moe_block=true + the 26B-A4B counts.
	moeCfg := base()
	moeCfg.EnableMoeBlock = true
	moeCfg.NumExperts = 128
	moeCfg.TopKExperts = 8
	moeCfg.MoeIntermediateSize = 704
	arch, _, err := gemma4Architecture(moeCfg)
	if err != nil {
		t.Fatalf("gemma4Architecture(moe): %v", err)
	}
	if arch.MoE == nil {
		t.Fatal("enable_moe_block=true but arch.MoE is nil")
	}
	for _, c := range []struct {
		name string
		got  any
		want any
	}{
		{"NumExperts", arch.MoE.NumExperts, 128},
		{"TopK (from top_k_experts)", arch.MoE.TopK, 8},
		{"IntermediateDim (moe width)", arch.MoE.IntermediateDim, 704},
		{"NormTopKProb (unconditional)", arch.MoE.NormTopKProb, true},
		{"RouterPreNorm", arch.MoE.RouterPreNorm, true},
		{"PerExpertScale", arch.MoE.PerExpertScale, true},
		{"IntermediateDim (dense mlp still 2112)", arch.IntermediateDim, 2112},
		{"gemma4 params still set", arch.gemma4 != nil, true},
	} {
		if c.got != c.want {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}

	// Dense variant: enable_moe_block unset ⇒ MoE nil (byte-unchanged descriptor).
	if a, _, err := gemma4Architecture(base()); err != nil || a.MoE != nil {
		t.Fatalf("dense gemma4: MoE=%v err=%v — want nil MoE", a.MoE, err)
	}

	// Validator rejects an out-of-range top_k and a zero expert count.
	bad := base()
	bad.EnableMoeBlock = true
	bad.NumExperts = 4
	bad.TopKExperts = 9 // > num_experts
	bad.MoeIntermediateSize = 704
	if _, _, err := gemma4Architecture(bad); err == nil {
		t.Error("top_k_experts > num_experts should be rejected")
	}
	bad.TopKExperts, bad.NumExperts = 2, 0
	if _, _, err := gemma4Architecture(bad); err == nil {
		t.Error("num_experts=0 should be rejected")
	}
}
