package decoder

import (
	"os"
	"testing"
)

// TestGemma4RunLayersFromEmbed_matchesTokenPath proves runLayersFromEmbedPLE (the embed-by-vector seam, P7) is
// behavior-preserving for the ordinary text path: feeding it the SAME already-scaled embedding the Gemma 4 program walk builds, with
// pleTokenID equal to the real token id, must reproduce the Gemma 4 program walk's own output bit-for-bit (they share every line after
// the embedding prologue, forward_gemma4.go).
//
// Uses the real E2B GGUF (the only local fixture with PLE enabled, hidden_size_per_layer_input=256; every tiny synthetic
// fixture in this package has PLE off), so it is heavy-gated like the other real-checkpoint gemma4 gates.
func TestGemma4RunLayersFromEmbed_matchesTokenPath(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("HOME") + "/models/gemma-4-E2B_q4_0-it.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no E2B gguf (%v)", err)
	}
	m, err := Load(path, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	arch := m.w.arch
	if arch.gemma4 == nil || arch.gemma4.HiddenSizePerLayerInput == 0 {
		t.Fatal("fixture has no PLE (hidden_size_per_layer_input=0) — this test needs a PLE checkpoint")
	}

	const tokenID = 7001 // an arbitrary real token id (TestGemma4_logitParity's own golden argmax)
	const padID = 0

	// Reference path: the Gemma 4 program walk itself.
	cacheA := m.NewCache(4)
	wantH, err := m.runLayers(tokenID, cacheA)
	if err != nil {
		t.Fatalf("the Gemma 4 program walk: %v", err)
	}

	// Embed-by-vector path: build the SAME embedding by hand, feed it through
	// runLayersFromEmbedPLE with pleTokenID == tokenID (must match exactly).
	cacheB := m.NewCache(4)
	h := make([]float32, arch.HiddenDim)
	m.w.Embed.Row(tokenID, h)
	es := float32(arch.EmbedScale)
	for i := range h {
		h[i] *= es
	}
	gotH, err := m.runLayersFromEmbedPLE(h, tokenID, cacheB)
	if err != nil {
		t.Fatalf("runLayersFromEmbedPLE: %v", err)
	}
	if len(gotH) != len(wantH) {
		t.Fatalf("len(gotH)=%d, want %d", len(gotH), len(wantH))
	}
	for i := range wantH {
		if gotH[i] != wantH[i] {
			t.Fatalf("runLayersFromEmbedPLE diverges from the Gemma 4 program walk at index %d: got %v want %v", i, gotH[i], wantH[i])
		}
	}
	t.Logf("runLayersFromEmbedPLE reproduces the Gemma 4 program walk bit-for-bit (%d dims)", len(wantH))

	// PAD substitution: SAME h, but pleTokenID = padID (the multimodal-position
	// case). PLE's token-identity term now comes from a different embedding
	// row, so the output must differ from the tokenID case — proving the
	// substitution is actually wired, not a silent no-op.
	cacheC := m.NewCache(4)
	gotPad, err := m.runLayersFromEmbedPLE(h, padID, cacheC)
	if err != nil {
		t.Fatalf("runLayersFromEmbedPLE (pad): %v", err)
	}
	same := len(gotPad) == len(wantH)
	if same {
		same = true
		for i := range wantH {
			if gotPad[i] != wantH[i] {
				same = false
				break
			}
		}
	}
	if same {
		t.Fatal("PLE PAD-substitution had no effect — pleTokenID=padID produced the SAME output as pleTokenID=tokenID (PLE is meant to be token-identity-sensitive)")
	}
	t.Log("PLE PAD-substitution changes the output relative to the real-token-id PLE path, as expected")
}
