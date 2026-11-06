package decoder

import (
	"testing"
)

// TestGlm4MoeGGUF_parity gates the GLM-4.5 GGUF loader against the SAME HF golden
// the safetensors path uses. The fixture (scripts/pin_glm_tiny_gguf.py) is F32 — no
// quant — so any divergence is the LOADER (the glm4moe.* metadata read, the
// dense/MoE per-layer split, the exp_probs_b router bias, the ungated ffn_*_shexp
// shared expert, the post_attention_norm pre-MLP name), not quant noise. It is the
// on-box stand-in for a real-model GGUF weightDiff (a real GLM won't fit here).
func TestGlm4MoeGGUF_parity(t *testing.T) {
	textParity{
		golden: "../testdata/glm_tiny_text_golden.json", ckpt: "../testdata/glm-tiny.gguf",
		pin:   "scripts/pin_glm_tiny.py and scripts/pin_glm_tiny_gguf.py",
		label: "glm4_moe GGUF parity",
		check: func(t *testing.T, m *Model) {
			if m.w.arch.Name != "glm4_moe" {
				t.Fatalf("arch = %q, want glm4_moe", m.w.arch.Name)
			}
			// The first_k_dense_replace prefix must load as dense (no experts); the MoE
			// layers must carry experts + the e_score_correction_bias.
			if m.w.Layers[0].Experts != nil {
				t.Errorf("layer 0 (dense prefix) has experts; want dense")
			}
			for i := 1; i < len(m.w.Layers); i++ {
				if m.w.Layers[i].Experts == nil {
					t.Errorf("layer %d (MoE) has no experts", i)
				}
				if got := len(m.w.Layers[i].RouterBias); got != 8 {
					t.Errorf("layer %d RouterBias len = %d, want 8 (exp_probs_b)", i, got)
				}
			}
		},
	}.run(t)
}
