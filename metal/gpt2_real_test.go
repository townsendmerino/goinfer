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
