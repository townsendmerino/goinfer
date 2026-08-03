//go:build darwin

package metal

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillNoNaN asserts the f16-MMA prefill produces finite logits across a range of prompt
// lengths — the direct regression gate for the bug where the int8-pinned LM head was run through
// the int4 gemm_w4f16 kernel (weights misread as packed nibbles), yielding NaN logits at EVERY M
// including the minimal single-tile M=8. Complements TestPrefillParity (which pins the value); a
// NaN here is the specific shipped-path failure that a hand-run caught only because no CI ran
// Metal against a checkpoint.
func TestPrefillNoNaN(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_METAL_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not present: %v", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := BuildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	defer r.Close()

	base := []int{785, 12095, 8948, 264, 6236, 1140, 13, 358, 3003, 264, 6236, 1140, 785, 12095}
	// Cover single-tile (8), multi-tile, and a padded length (140 → Mpad 144) so a padding-only
	// regression can't hide behind aligned sizes.
	for _, M := range []int{8, 16, 140, 144, 200} {
		embs := make([][]float32, M)
		for i := 0; i < M; i++ {
			e := make([]float32, r.H)
			r.embed.Row(base[i%len(base)], e)
			embs[i] = e
		}
		out := r.PrefillLast(embs, 0)
		if hasNaN(out) {
			t.Errorf("M=%d: prefill logits contain NaN/Inf", M)
		}
	}
}

func hasNaN(v []float32) bool {
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return true
		}
	}
	return false
}
