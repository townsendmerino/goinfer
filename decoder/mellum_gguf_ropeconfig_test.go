package decoder

import (
	"encoding/json"
	"testing"

	"github.com/townsendmerino/aikit/embed"
)

// TestGGUFMellumConfig_attentionFactorSentinelOmitted is N-69 (docs/audit-2026-09-10.md):
// ggufMellumConfig used to synthesize `"attention_factor": <rope.scaling.yarn_attn_factor>`
// unconditionally, including at llama.cpp's own 1.0 "unset" sentinel and at 0 (metadata
// entirely absent, this package's own gf() helper's absent-value default). Passing either
// through as an explicit attention_factor overrides YaRN's computed mscale default
// (0.1·ln(factor)+1) with a no-op or zero instead of leaving the field out so the decoder
// computes it — exactly the trap ggufLagunaConfig already documents and avoids for the same
// GGUF field, just not applied here too.
func TestGGUFMellumConfig_attentionFactorSentinelOmitted(t *testing.T) {
	base := map[string]any{
		"mellum.block_count":                          uint32(2),
		"mellum.embedding_length":                     uint32(64),
		"mellum.attention.head_count":                 uint32(4),
		"mellum.attention.head_count_kv":              uint32(4),
		"mellum.attention.key_length":                 uint32(16),
		"mellum.attention.sliding_window_pattern":     []any{true, false},
		"mellum.rope.freq_base":                       float32(10000),
		"mellum.rope.freq_base_swa":                   float32(10000),
		"mellum.rope.scaling.factor":                  float32(32),
		"mellum.rope.scaling.original_context_length": float32(4096),
		"mellum.rope.scaling.yarn_beta_fast":          float32(32),
		"mellum.rope.scaling.yarn_beta_slow":          float32(1),
	}
	cases := []struct {
		name       string
		attnFactor any // nil = key absent entirely
		wantField  bool
	}{
		{"absent (metadata never set)", nil, false},
		{"llama.cpp's 1.0 unset sentinel", float32(1), false},
		{"0 (same as absent numerically)", float32(0), false},
		{"a real explicit value", float32(1.3465736), true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			meta := map[string]any{}
			for k, v := range base {
				meta[k] = v
			}
			if c.attnFactor != nil {
				meta["mellum.rope.scaling.yarn_attn_factor"] = c.attnFactor
			}
			g := &embed.GGUFFile{Metadata: meta}
			cfg, err := ggufMellumConfig(g)
			if err != nil {
				t.Fatalf("ggufMellumConfig: %v", err)
			}
			var parsed struct {
				FullAttention struct {
					AttentionFactor *float64 `json:"attention_factor"`
				} `json:"full_attention"`
			}
			if err := json.Unmarshal(cfg.RopeParameters, &parsed); err != nil {
				t.Fatalf("parse synthesized rope_parameters: %v (%s)", err, cfg.RopeParameters)
			}
			got := parsed.FullAttention.AttentionFactor != nil
			if got != c.wantField {
				t.Errorf("attention_factor present = %v, want %v (rope_parameters: %s)",
					got, c.wantField, cfg.RopeParameters)
			}
		})
	}
}
