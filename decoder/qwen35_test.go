package decoder

import (
	"encoding/json"
	"testing"
)

// qwen35TestConfig builds the tiny-random qwen3_5_moe config that
// scripts/pin_qwen35_forward.py pins the golden from (testdata/
// qwen35_forward_golden.json): 4 layers as 3 linear + 1 full, GVA v/k=2,
// partial RoPE 0.25, a 4-expert + shared MoE.
func qwen35TestConfig() *Config {
	return &Config{
		ModelType:                    "qwen3_5_moe",
		VocabSize:                    256,
		HiddenDim:                    64,
		NumLayers:                    4,
		NumHeads:                     4,
		NumKVHeads:                   2,
		HeadDim:                      16,
		HiddenAct:                    "silu",
		RMSNormEps:                   1e-6,
		LayerTypes:                   []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearConvKernelDim:          4,
		LinearKeyHeadDim:             16,
		LinearValueHeadDim:           16,
		LinearNumKeyHeads:            2,
		LinearNumValueHeads:          4,
		NumExperts:                   4,
		NumExpertsPerTok:             2,
		MoeIntermediateSize:          32,
		SharedExpertIntermediateSize: 32,
		RopeParameters:               json.RawMessage(`{"rope_type":"default","rope_theta":1000000.0,"partial_rotary_factor":0.25}`),
	}
}

// TestResolveQwen35: the qwen3_5_moe adapter must resolve the real config into
// the right descriptor — the 3:1 linear/full layer dispatch, the Gated DeltaNet
// geometry (GVA), the routed + shared MoE, QK-norm softmax layers, and partial
// RoPE (rotary_dim = 0.25·head_dim).
func TestResolveQwen35(t *testing.T) {
	arch, schema, err := resolveArchitecture(qwen35TestConfig())
	if err != nil {
		t.Fatalf("resolveArchitecture(qwen3_5_moe): %v", err)
	}
	if arch.Name != "qwen3_5_moe" || schema != &qwen35TensorSchema {
		t.Fatalf("wrong adapter: name=%q", arch.Name)
	}

	// Dims + softmax-layer flags.
	if arch.HiddenDim != 64 || arch.NumHeads != 4 || arch.NumKVHeads != 2 || arch.HeadDim != 16 {
		t.Errorf("dims: hidden=%d heads=%d kv=%d hd=%d", arch.HiddenDim, arch.NumHeads, arch.NumKVHeads, arch.HeadDim)
	}
	if !arch.QKNorm || arch.Norm != NormRMS || !arch.RMSAddOne || arch.Act != ActSiLU || arch.NormPlacement != NormPre2 {
		t.Errorf("wrong norm/act flags (RMSAddOne should be true: Qwen3_5MoeRMSNorm is (1+w))")
	}
	// Partial rotary: 0.25 * 16 = 4.
	if arch.RotaryDim != 4 {
		t.Errorf("RotaryDim = %d, want 4 (partial_rotary_factor 0.25)", arch.RotaryDim)
	}

	// Per-layer dispatch: layers 0-2 linear, layer 3 full (softmax).
	for i := 0; i < 3; i++ {
		if !arch.isLinearLayer(i) || arch.isGlobalLayer(i) {
			t.Errorf("layer %d: isLinear=%v isGlobal=%v, want linear", i, arch.isLinearLayer(i), arch.isGlobalLayer(i))
		}
	}
	if arch.isLinearLayer(3) || !arch.isGlobalLayer(3) {
		t.Errorf("layer 3: isLinear=%v isGlobal=%v, want full/softmax", arch.isLinearLayer(3), arch.isGlobalLayer(3))
	}

	// Gated DeltaNet geometry.
	q := arch.qwen35
	if q == nil {
		t.Fatal("qwen35 params nil")
	}
	if q.ConvKernel != 4 || q.KeyHeadDim != 16 || q.ValueHeadDim != 16 || q.NumKeyHeads != 2 || q.NumValueHeads != 4 {
		t.Errorf("DeltaNet geom = %+v, want conv4/k16/v16/nk2/nv4", q)
	}

	// MoE: 4 experts, top-2, expert width 32, shared expert width 32.
	if arch.MoE == nil || arch.MoE.NumExperts != 4 || arch.MoE.TopK != 2 ||
		arch.MoE.IntermediateDim != 32 || arch.MoE.SharedIntermediateDim != 32 {
		t.Fatalf("MoE = %+v, want 4/2/32/shared32", arch.MoE)
	}
	if !arch.MoE.NormTopKProb {
		t.Errorf("NormTopKProb = false, want true (qwen3_5_moe router always renormalizes top-k)")
	}
}

func TestResolveQwen35_rejectsBadConfig(t *testing.T) {
	cases := map[string]func(*Config){
		"layer_types length":   func(c *Config) { c.LayerTypes = []string{"linear_attention"} },
		"GVA not multiple":     func(c *Config) { c.LinearNumValueHeads = 5 }, // 5 % 2 != 0
		"missing DeltaNet dim": func(c *Config) { c.LinearKeyHeadDim = 0 },
		"bad MoE top_k":        func(c *Config) { c.NumExpertsPerTok = 99 },
		"no rope_parameters":   func(c *Config) { c.RopeParameters = nil },
	}
	for name, mut := range cases {
		cfg := qwen35TestConfig()
		mut(cfg)
		if _, _, err := resolveArchitecture(cfg); err == nil {
			t.Errorf("%s: expected error, got nil", name)
		}
	}
}
