package decoder

import "testing"

// llama.cpp converts Mellum as a NEOX-rope architecture (its llama_model_rope_type lists LLM_ARCH_MELLUM with the NEOX
// archs), so a Mellum GGUF's q/k projections are already in HF rotate_half order. Un-permuting them scrambles every head's
// rotary pairs: the model still emits finite, code-shaped text, so only a check on the permutation itself catches it.
func TestGGUFQKPermuted_mellumIsNEOX(t *testing.T) {
	for arch, want := range map[string]bool{"llama": true, "granite": true, "mellum": false, "qwen3": false} {
		if got := ggufQKPermuted(arch); got != want {
			t.Errorf("ggufQKPermuted(%q) = %v, want %v", arch, got, want)
		}
	}
}
