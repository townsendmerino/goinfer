package decoder

import "testing"

// TestEmbedInt4Knob gates Options.EmbedInt4 (#3): with int4 quant, the default pins
// the embed/head table to int8 (logit-critical), but the opt-in knob relaxes it to
// int4 — halving the largest resident tensor on a big-vocab small model. Asserts the
// embed table's precision actually changes (int8 → int4) and both models still
// generate. The quality cost is recorded in docs (≈2.3 pts top-1); this test just
// gates that the knob takes effect and is lossless-to-load.
func TestEmbedInt4Knob(t *testing.T) {
	path := prequantGGUF(t)

	pinned, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load pinned: %v", err)
	}
	defer pinned.Close()
	relaxed, err := Load(path, Options{Quant: "int4", EmbedInt4: true})
	if err != nil {
		t.Fatalf("load relaxed: %v", err)
	}
	defer relaxed.Close()

	// Default pins the embed table to int8; the knob makes it int4.
	if _, _, _, ok := pinned.w.Embed.Int8(); !ok {
		t.Errorf("default int4 mode: embed should be int8-pinned, kind=%s", pinned.w.Embed.Kind())
	}
	if _, _, _, ok := relaxed.w.Embed.Int4(); !ok {
		t.Errorf("EmbedInt4: embed should be int4, kind=%s", relaxed.w.Embed.Kind())
	}

	// Untied models also carry a separate head; it follows the same policy.
	if relaxed.w.arch != nil && !relaxed.w.arch.TiedLMHead && relaxed.w.LMHead.Rows() > 0 {
		if _, _, _, ok := relaxed.w.LMHead.Int4(); !ok {
			t.Errorf("EmbedInt4 (untied): LM head should be int4, kind=%s", relaxed.w.LMHead.Kind())
		}
	}

	// Both still produce a token (knob is a load-time precision choice, not a break).
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	if greedyFirst(t, pinned, prompt) < 0 || greedyFirst(t, relaxed, prompt) < 0 {
		t.Fatal("no token generated")
	}
	t.Logf("embed kind: pinned=%s relaxed=%s", pinned.w.Embed.Kind(), relaxed.w.Embed.Kind())
}
