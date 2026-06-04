package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// mellumConfigJSON is the architecture-relevant subset of
// JetBrains/Mellum2-12B-A2.5B-Thinking's config.json (verbatim values). The
// layer_types array is filled in by the test (3 sliding : 1 full, ×7 = 28).
const mellumConfigJSON = `{
  "model_type": "mellum",
  "hidden_size": 2304, "num_hidden_layers": 28, "num_attention_heads": 32,
  "num_key_value_heads": 4, "head_dim": 128, "intermediate_size": 7168,
  "vocab_size": 98304, "rms_norm_eps": 1e-6, "hidden_act": "silu",
  "num_experts": 64, "num_experts_per_tok": 8, "norm_topk_prob": true,
  "moe_intermediate_size": 896, "sliding_window": 1024,
  "tie_word_embeddings": false, "attention_bias": false,
  "rope_parameters": {
    "full_attention": {
      "rope_type": "yarn", "rope_theta": 500000.0, "factor": 16.0,
      "original_max_position_embeddings": 8192, "beta_fast": 32.0,
      "beta_slow": 1.0, "attention_factor": 1.2772588722239782
    },
    "sliding_attention": { "rope_type": "default", "rope_theta": 500000.0 }
  }
}`

func mellumTestConfig(t *testing.T) *Config {
	t.Helper()
	var cfg Config
	if err := json.Unmarshal([]byte(mellumConfigJSON), &cfg); err != nil {
		t.Fatalf("parse mellum config: %v", err)
	}
	// 3:1 sliding/full interleave, repeated to 28 layers.
	for i := 0; i < cfg.NumLayers; i++ {
		if (i+1)%4 == 0 {
			cfg.LayerTypes = append(cfg.LayerTypes, "full_attention")
		} else {
			cfg.LayerTypes = append(cfg.LayerTypes, "sliding_attention")
		}
	}
	return &cfg
}

// TestResolveMellum: the mellum adapter must resolve the real config into the
// right descriptor — MoE knobs (incl. the narrower expert width), the 3:1
// sliding/full interleave from layer_types, and the per-attention-type RoPE
// (YaRN on full layers + its mscale, plain RoPE on sliding layers). The RoPE
// tables are checked against the HF-pinned golden.
func TestResolveMellum(t *testing.T) {
	cfg := mellumTestConfig(t)
	arch, schema, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture(mellum): %v", err)
	}
	if arch.Name != "mellum" || schema != &mellumTensorSchema {
		t.Fatalf("wrong adapter: name=%q", arch.Name)
	}

	// Dims + flags.
	if arch.HiddenDim != 2304 || arch.NumHeads != 32 || arch.NumKVHeads != 4 || arch.HeadDim != 128 {
		t.Errorf("dims: hidden=%d heads=%d kv=%d hd=%d", arch.HiddenDim, arch.NumHeads, arch.NumKVHeads, arch.HeadDim)
	}
	if !arch.QKNorm || arch.Norm != NormRMS || arch.RMSAddOne || arch.Act != ActSiLU || arch.NormPlacement != NormPre2 {
		t.Errorf("wrong norm/act flags (QKNorm=%v norm=%v rmsAddOne=%v act=%v placement=%v)",
			arch.QKNorm, arch.Norm, arch.RMSAddOne, arch.Act, arch.NormPlacement)
	}
	if arch.SlidingWindow != 1024 {
		t.Errorf("sliding window = %d, want 1024", arch.SlidingWindow)
	}

	// MoE: 64 experts, top-8, narrower expert width (896, not 7168).
	if arch.MoE == nil || arch.MoE.NumExperts != 64 || arch.MoE.TopK != 8 || !arch.MoE.NormTopKProb {
		t.Fatalf("MoE config wrong: %+v", arch.MoE)
	}
	if arch.MoE.IntermediateDim != 896 {
		t.Errorf("expert width = %d, want 896 (moe_intermediate_size)", arch.MoE.IntermediateDim)
	}

	// Attention interleave: layer_types says layers 3,7,11,… are full (global).
	for _, i := range []int{0, 1, 2, 4} {
		if arch.isGlobalLayer(i) {
			t.Errorf("layer %d classified global, want sliding (local)", i)
		}
	}
	for _, i := range []int{3, 7, 27} {
		if !arch.isGlobalLayer(i) {
			t.Errorf("layer %d classified sliding, want full (global)", i)
		}
	}

	// RoPE tables vs the HF golden: full layers YaRN, sliding layers plain.
	g := loadMellumRopeGolden(t)
	if g == nil {
		return
	}
	closeVec(t, "mellum global (YaRN) inv_freq", arch.ropeInvFreqGlobal, g.Full.InvFreq)
	closeVec(t, "mellum local (plain) inv_freq", arch.ropeInvFreqLocal, g.Sliding.InvFreq)

	// mscale: YaRN attention_factor on the full layers, 1.0 on sliding.
	if ms := arch.ropeMscale(3); ms != g.Full.AttentionFactor {
		t.Errorf("global mscale = %v, want %v", ms, g.Full.AttentionFactor)
	}
	if ms := arch.ropeMscale(0); ms != 1.0 {
		t.Errorf("local mscale = %v, want 1.0", ms)
	}
}

// TestResolveMellum_rejectsBadConfig: a missing rope_parameters / layer_types /
// MoE field is a loud error, not a silent wrong load.
func TestResolveMellum_rejectsBadConfig(t *testing.T) {
	for name, mutate := range map[string]func(*Config){
		"no rope_parameters": func(c *Config) { c.RopeParameters = nil },
		"no moe width":       func(c *Config) { c.MoeIntermediateSize = 0 },
		"bad topk":           func(c *Config) { c.NumExpertsPerTok = 999 },
		"short layer_types":  func(c *Config) { c.LayerTypes = c.LayerTypes[:3] },
	} {
		cfg := mellumTestConfig(t)
		mutate(cfg)
		if _, _, err := resolveArchitecture(cfg); err == nil {
			t.Errorf("%s: expected error, got nil", name)
		}
	}
}

type mellumRopeGolden struct {
	Full struct {
		AttentionFactor float64   `json:"attention_factor"`
		InvFreq         []float64 `json:"inv_freq"`
	} `json:"full"`
	Sliding struct {
		InvFreq []float64 `json:"inv_freq"`
	} `json:"sliding"`
}

func loadMellumRopeGolden(t *testing.T) *mellumRopeGolden {
	t.Helper()
	raw, err := os.ReadFile("../testdata/mellum_rope_golden.json")
	if errors.Is(err, fs.ErrNotExist) {
		t.Log("no mellum_rope_golden.json — skipping RoPE-table check")
		return nil
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g mellumRopeGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	return &g
}
