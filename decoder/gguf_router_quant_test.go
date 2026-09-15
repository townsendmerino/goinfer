package decoder

import (
	"os"
	"testing"
)

// TestGGUFRouter_staysF32AtQuant gates M-27: several GGUF MoE loaders quantized the router
// (ffn_gate_inp) via the generic mat() helper instead of routing it through
// streamMat(..., quantNone, ...) like gpt-oss/qwen35 already did — CUDA/Metal residency
// requires an f32 router (quantizing it flips which experts win near a tie, not just rounding
// noise), so every affected family at a non-f32 quant silently declined resident build. This
// fixture's shared generic loadLayer branch covers GLM/Mellum/Qwen3-MoE/DeepSeek2.
func TestGGUFRouter_staysF32AtQuant(t *testing.T) {
	const gguf = "../testdata/glm-tiny.gguf"
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no tiny GLM GGUF at %s — run scripts/pin_glm_tiny_gguf.py", gguf)
	}
	for _, quant := range []string{"int8", "int8int8", "int4", "int4mix"} {
		t.Run("quant="+quant, func(t *testing.T) {
			m, err := Load(gguf, Options{Quant: quant})
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			defer m.Close()
			sawMoE := false
			for i, l := range m.w.Layers {
				if l.Router.Rows() == 0 {
					continue // dense prefix layer, no router
				}
				sawMoE = true
				if k := l.Router.Kind(); k != "f32" {
					t.Errorf("quant=%s layer %d: router kind = %q, want f32", quant, i, k)
				}
			}
			if !sawMoE {
				t.Fatal("test bug: no MoE layer found in the fixture — nothing was checked")
			}
		})
	}
}
