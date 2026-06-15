package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// TestGlm4MoeGGUF_parity gates the GLM-4.5 GGUF loader against the SAME HF golden
// the safetensors path uses. The fixture (scripts/pin_glm_tiny_gguf.py) is F32 — no
// quant — so any divergence is the LOADER (the glm4moe.* metadata read, the
// dense/MoE per-layer split, the exp_probs_b router bias, the ungated ffn_*_shexp
// shared expert, the post_attention_norm pre-MLP name), not quant noise. It is the
// on-box stand-in for a real-model GGUF weightDiff (a real GLM won't fit here).
func TestGlm4MoeGGUF_parity(t *testing.T) {
	const golden = "../testdata/glm_tiny_text_golden.json"
	const gguf = "../testdata/glm-tiny.gguf"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_glm_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(gguf); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no GGUF fixture at %s — run scripts/pin_glm_tiny_gguf.py", gguf)
	}
	var g struct {
		PromptIDs       []int     `json:"prompt_ids"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
		NNew            int       `json:"n_new"`
		ContinuationIDs []int     `json:"continuation_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	m, err := Load(gguf, Options{})
	if err != nil {
		t.Fatalf("Load(%s): %v", gguf, err)
	}
	defer m.Close()

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

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("glm4_moe GGUF parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.9999 {
		t.Errorf("last-logit cosine %.6f < 0.9999 (F32 GGUF — should be ~1.0)", cos)
	}

	got := make([]int, 0, g.NNew)
	for range g.NNew {
		id := argmax(logits)
		got = append(got, id)
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
	}
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d (got %v want %v)", i, got[i], g.ContinuationIDs[i], got, g.ContinuationIDs)
			break
		}
	}
}
