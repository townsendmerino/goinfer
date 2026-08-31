//go:build realckpt

// fp8 loader gate — diffed against the SAME model in bf16, read by the already-full-oracle
// qwen3 path.
//
// This is the verification standard mxfp4.go set and gptoss_safetensors_test.go follows: a new
// reader is checked against a validated reader of the same weights, not against intuition. It
// matters more here than usual, because every plausible way to get fp8 wrong is silent —
// a mis-derived e4m3 table, a transposed weight, an inverted `weight_scale_inv`, or block
// indexing off by one all produce finite, smoothly-scaled, completely wrong logits.
//
// Run:
//
//	GOINFER_QWEN3_FP8=~/models/qwen3-0.6b-fp8 GOINFER_QWEN3_BF16=~/models/qwen3-0.6b-bf16 \
//	  go test ./decoder -tags realckpt -run TestFP8Qwen3_vsBF16 -v
package decoder

import (
	"testing"
)

func TestFP8Qwen3_vsBF16(t *testing.T) {
	fp8Dir := assetPath(t, "GOINFER_QWEN3_FP8")
	bf16Dir := assetPath(t, "GOINFER_QWEN3_BF16")

	// Quant "" keeps both sides in f32 after load, so the comparison isolates the LOADER.
	// Quantizing here would fold goinfer's own int8/int4 error into the number and make a
	// loader bug and a quantizer bug indistinguishable.
	m8, err := Load(fp8Dir, Options{Quant: ""})
	if err != nil {
		t.Fatalf("load fp8: %v", err)
	}
	defer m8.Close()

	mb, err := Load(bf16Dir, Options{Quant: ""})
	if err != nil {
		t.Fatalf("load bf16: %v", err)
	}
	defer mb.Close()

	prompt := []int{785, 264, 6573, 311, 1438, 279, 2038, 25}
	fwd := func(m *Model) []float32 {
		t.Helper()
		c := m.NewCache(len(prompt) + 1)
		var lg []float32
		var e error
		for _, id := range prompt {
			if lg, e = m.forward(id, c); e != nil {
				t.Fatalf("forward: %v", e)
			}
		}
		return lg
	}

	l8, lb := fwd(m8), fwd(mb)
	cos := logitCosine(l8, lb)
	a8, ab := argmax(l8), argmax(lb)
	t.Logf("qwen3-0.6b fp8 vs bf16: argmax %d vs %d | logit cosine %.6f", a8, ab, cos)

	// NOT bit-identical, and it should not be: these are different weights. The fp8 file is a
	// block-quantized approximation of the bf16 one, so the residual here is real quantization
	// error, not loader error. What the gate asserts is that the error is SMALL — a wrong
	// table, transpose, or block index does not land near the original, it lands nowhere near
	// it (the mxfp4 mis-unpack that motivated this discipline read cosine 0.081).
	if cos < 0.99 {
		t.Errorf("logit cosine %.6f < 0.99 — too far apart to be quantization error alone; "+
			"suspect the e4m3 table, a transpose, an inverted weight_scale_inv, or block indexing", cos)
	}
	if a8 != ab {
		t.Errorf("argmax differs: fp8=%d bf16=%d", a8, ab)
	}
}
