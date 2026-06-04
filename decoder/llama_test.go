package decoder

import (
	"math"
	"strings"
	"testing"
)

// validLlamaConfig is a tiny validateLlama-clean llama config for descriptor
// tests (no checkpoint needed). Deliberately omits head_dim to exercise the
// hidden/heads fallback, the way many real Llama configs do.
func validLlamaConfig() *Config {
	return &Config{
		ModelType: "llama", VocabSize: 128, HiddenDim: 16, NumLayers: 2,
		NumHeads: 4, NumKVHeads: 2, // GQA; head_dim derived = 16/4 = 4
		IntermediateDim: 32, RMSNormEps: 1e-5, RoPEGlobalBase: 500000,
		HiddenAct: "silu",
	}
}

func TestResolveArchitecture_llama(t *testing.T) {
	a, schema, err := resolveArchitecture(validLlamaConfig())
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	checks := []struct {
		name string
		got  any
		want any
	}{
		{"Name", a.Name, "llama"},
		{"Norm", a.Norm, NormRMS},
		{"RMSAddOne", a.RMSAddOne, false},
		{"NormPlacement", a.NormPlacement, NormPre2},
		{"Act", a.Act, ActSiLU},
		{"QKNorm", a.QKNorm, false}, // the knob that differs from Qwen3
		{"HeadDim", a.HeadDim, 4},   // derived: hidden 16 / heads 4
		{"AttnScale", a.AttnScale, math.Pow(4, -0.5)},
		{"EmbedScale", a.EmbedScale, 0.0},
		{"SlidingWindow", a.SlidingWindow, 0},
		// single-base RoPE: local == global == rope_theta
		{"RoPELocalBase", a.RoPELocalBase, 500000.0},
		{"RoPEGlobalBase", a.RoPEGlobalBase, 500000.0},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
	// Llama carries no QK-norm tensors; the schema must leave them empty so the
	// loader skips them.
	if schema.QNorm != "" || schema.KNorm != "" {
		t.Errorf("llama schema QNorm/KNorm = %q/%q, want empty", schema.QNorm, schema.KNorm)
	}
	if schema.PostAttnNorm != "" || schema.PostMLPNorm != "" {
		t.Errorf("llama schema should be Pre2 (no post-sublayer norms), got post-attn=%q post-mlp=%q",
			schema.PostAttnNorm, schema.PostMLPNorm)
	}
}

// TestLlama_explicitHeadDim: when config.json DOES carry head_dim, it wins over
// the hidden/heads fallback (true for Llama-3 8B: hidden 4096, 32 heads, but
// head_dim is also 128 — they happen to agree; here we make them disagree to
// prove head_dim is authoritative).
func TestLlama_explicitHeadDim(t *testing.T) {
	cfg := validLlamaConfig()
	cfg.HeadDim = 8 // disagrees with hidden/heads = 4
	a, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if a.HeadDim != 8 {
		t.Errorf("HeadDim = %d, want 8 (explicit head_dim wins)", a.HeadDim)
	}
}

// TestLlama_ropeScaling: llama3 scaling (Llama-3.1+/3.2) loads (G4); a null
// object is the no-scaling common case; an unsupported type fails loudly rather
// than loading with wrong positional frequencies.
func TestLlama_ropeScaling(t *testing.T) {
	// llama3 scaling loads and bakes into the inv-freq table.
	cfg := validLlamaConfig()
	cfg.RopeScaling = []byte(`{"rope_type":"llama3","factor":32.0,"low_freq_factor":1.0,"high_freq_factor":4.0,"original_max_position_embeddings":8192}`)
	a, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("llama3 rope_scaling should load: %v", err)
	}
	// Scaling must change the low-frequency end of the table vs no scaling.
	plain := computeInvFreq(cfg.RoPEGlobalBase, a.HeadDim, nil)
	if a.ropeInvFreqGlobal[len(a.ropeInvFreqGlobal)-1] >= plain[len(plain)-1] {
		t.Errorf("llama3 scaling did not lower the low-frequency inv_freq (%v vs unscaled %v)",
			a.ropeInvFreqGlobal[len(a.ropeInvFreqGlobal)-1], plain[len(plain)-1])
	}

	// null rope_scaling (the common case) loads unchanged.
	cfg.RopeScaling = []byte(`null`)
	if _, _, err := resolveArchitecture(cfg); err != nil {
		t.Errorf("null rope_scaling should load, got %v", err)
	}

	// An unsupported scaling type must be rejected loudly.
	cfg.RopeScaling = []byte(`{"rope_type":"yarn","factor":4.0}`)
	if _, _, err := resolveArchitecture(cfg); err == nil || !strings.Contains(err.Error(), "yarn") {
		t.Fatalf("err = %v, want a yarn rejection", err)
	}
}

// TestLlama_rejectsAttentionBias: q/k/v/o bias on the *llama* path isn't wired
// (Qwen2 uses its own adapter which DOES set bias).
func TestLlama_rejectsAttentionBias(t *testing.T) {
	cfg := validLlamaConfig()
	cfg.AttentionBias = true
	_, _, err := resolveArchitecture(cfg)
	if err == nil || !strings.Contains(err.Error(), "attention_bias") {
		t.Fatalf("err = %v, want an attention_bias rejection", err)
	}
}

// TestResolveArchitecture_qwen2: Qwen2 is the llama descriptor + QKVBias, still
// no QK-norm, with the bias-carrying tensor schema.
func TestResolveArchitecture_qwen2(t *testing.T) {
	cfg := validLlamaConfig()
	cfg.ModelType = "qwen2"
	a, schema, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if a.Name != "qwen2" {
		t.Errorf("Name = %q, want qwen2", a.Name)
	}
	if !a.QKVBias {
		t.Errorf("QKVBias = false, want true")
	}
	if a.QKNorm {
		t.Errorf("QKNorm = true, want false (Qwen2 has no QK-norm)")
	}
	if schema.QBias == "" || schema.KBias == "" || schema.VBias == "" {
		t.Errorf("qwen2 schema missing q/k/v bias names: %q/%q/%q", schema.QBias, schema.KBias, schema.VBias)
	}

	// use_sliding_window=true is rejected (full-attention only for now).
	cfg.UseSlidingWindow = true
	if _, _, err := resolveArchitecture(cfg); err == nil || !strings.Contains(err.Error(), "sliding_window") {
		t.Fatalf("err = %v, want a sliding_window rejection", err)
	}
}

// TestResolveArchitecture_mistral: Mistral is the llama descriptor with a
// sliding window on EVERY layer (all-local, unlike Gemma's 5:1).
func TestResolveArchitecture_mistral(t *testing.T) {
	cfg := validLlamaConfig()
	cfg.ModelType = "mistral"
	cfg.SlidingWindow = 4096
	a, schema, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if a.Name != "mistral" {
		t.Errorf("Name = %q, want mistral", a.Name)
	}
	if a.SlidingWindow != 4096 {
		t.Errorf("SlidingWindow = %d, want 4096", a.SlidingWindow)
	}
	for i := 0; i < a.NumLayers; i++ {
		if a.isGlobalLayer(i) {
			t.Errorf("layer %d global, want all-local", i)
		}
	}
	if schema.QBias != "" { // Mistral has no bias (shares llamaTensorSchema)
		t.Errorf("mistral schema should have no QKV bias, got %q", schema.QBias)
	}

	// sliding_window null/0 (Mistral-v0.2+) ⇒ full attention (all global).
	cfg.SlidingWindow = 0
	a2, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture (no window): %v", err)
	}
	if !a2.isGlobalLayer(0) {
		t.Errorf("with sliding_window=0, layers should be global (full attention)")
	}
}

// TestAddBias: the projection-bias add is a plain elementwise sum.
func TestAddBias(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	addBias(x, []float32{10, 20, 30, 40})
	want := []float32{11, 22, 33, 44}
	for i := range x {
		if x[i] != want[i] {
			t.Errorf("addBias[%d] = %v, want %v", i, x[i], want[i])
		}
	}
}
