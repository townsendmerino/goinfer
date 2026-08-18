//go:build darwin && goinfer_testhooks

package metal

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestGPT2ResidentParity is GPT-2's Metal gate — the same residentParity/assertParity harness
// TestDenseResidentParity/TestGemma3ResidentParity use, on the same real checkpoint
// decoder/gpt2_test.go's CPU parity gate already validates (testdata/gpt2, downloaded via
// scripts/pin_gpt2_real.py — 548 MB, small enough not to need requireHeavyModel's opt-in gate).
//
// Dormant until all four features are declared (same pattern as TestGemma3ResidentParity): metal
// ships the LayerNorm/non-gated-MLP/learned-pos/out-bias kernels and the encodeLayer/
// encodeAttention wiring, but does not yet DECLARE them, so gpt2 still declines to CPU — skip
// rather than fail, and residentParity t.Fatals on a decline once the declaration lands (catching
// a silent CPU fallback rather than an honest skip).
func TestGPT2ResidentParity(t *testing.T) {
	f := decoder.ResidentBackendFeatures("metal")
	if !f[decoder.FeatLayerNorm] || !f[decoder.FeatNonGatedMLP] || !f[decoder.FeatLearnedPos] || !f[decoder.FeatOutBias] {
		t.Skip("metal does not declare all four GPT-2 features yet (kernels/wiring dormant) — see docs/queue-correctness.md G7")
	}
	path := "../testdata/gpt2"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no GPT-2 checkpoint at %s — run scripts/pin_gpt2_real.py", path)
	}
	tk, err := tokenizer.Load(path)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	ids, err := tk.Encode(probeText, false) // GPT-2 prepends no BOS
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, derr := tk.Decode(ids)
	if derr != nil {
		t.Fatalf("decode-back: %v", derr)
	}
	t.Logf("prompt %q -> %v -> decodes to %q", probeText, ids, got)

	st := residentParity(t, path, ids, 24)
	assertParity(t, "gpt2", st, 0.95) // dense int4-vs-int8, same bar as TestDenseResidentParity
}

// TestGPT2ForwardArgmax_matchesFullLogits gates the ForwardArgmax fallback specifically —
// residentParity above only exercises the full-logits path (Forward), never ForwardArgmax, so it
// would stay green even if the V%8!=0 fallback added for GPT-2's 50257 vocab were wrong. GPT-2 is
// the first family whose vocab isn't a multiple of 8, so this is the first real exercise of that
// branch (model.go's ForwardArgmax): it must equal argmax(Forward's full logits) at every
// position, matching model_test.go's TestRealModel_parityAndThroughput's own fused-argmax
// assertion for the %8 case.
func TestGPT2ForwardArgmax_matchesFullLogits(t *testing.T) {
	f := decoder.ResidentBackendFeatures("metal")
	if !f[decoder.FeatLayerNorm] || !f[decoder.FeatNonGatedMLP] || !f[decoder.FeatLearnedPos] || !f[decoder.FeatOutBias] {
		t.Skip("metal does not declare all four GPT-2 features yet — see docs/queue-correctness.md G7")
	}
	path := "../testdata/gpt2"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no GPT-2 checkpoint at %s — run scripts/pin_gpt2_real.py", path)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	if r.V%8 == 0 {
		t.Fatalf("test setup no longer exercises the non-%%8 fallback: vocab %d is a multiple of 8", r.V)
	}
	tk, err := tokenizer.Load(path)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	ids, err := tk.Encode(probeText, false)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	tok, pos, mism := ids[0], 0, 0
	for i := range 12 {
		full := r.Forward(tok, pos)
		want := argmaxF32(full)
		got := int(r.ForwardArgmax(tok, pos))
		if got != want {
			mism++
			t.Logf("pos %d: ForwardArgmax=%d != argmax(Forward)=%d", pos, got, want)
		}
		if i+1 < len(ids) {
			tok = ids[i+1]
		} else {
			tok = want
		}
		pos++
	}
	if mism > 0 {
		t.Errorf("ForwardArgmax disagreed with argmax(Forward) at %d/12 positions (vocab=%d, non-%%8 fallback)", mism, r.V)
	}
}
