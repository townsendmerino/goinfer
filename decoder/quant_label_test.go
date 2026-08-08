package decoder

import "testing"

// TestGiwInt4LabelNotMix is the T1-6 regression. A .giw baked with `-quant int4` has int4
// projections but an int8-pinned embedding / LM head (the logit-critical default; the
// EmbedInt4 knob relaxes it). quantLabel used to scan those tables, see int4 coexisting with
// int8, and report "int4mix" — while the batched-prefill gate, which inspects only the seven
// int4 projections, correctly batched. So /health showed `decode_path: …(int4mix)` beside
// `prefill_batched: true`, and the label named a quant the bundle is not.
//
// The .giw path is what triggers the inference: a direct Load records the requested quant
// string and returns it verbatim, never inferring. So the round-trip through SerializeWeights
// is load-bearing here, not incidental.
func TestGiwInt4LabelNotMix(t *testing.T) {
	path := prequantGGUF(t)
	m, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load int4: %v", err)
	}
	defer m.Close()

	// Precondition: the default int4 mode DOES pin the embed to int8 — otherwise this test
	// would pass trivially without exercising the mixed-precision-tables condition that broke.
	if k := m.w.Embed.Kind(); k != "int8" {
		t.Fatalf("precondition: default int4 should pin embed to int8, got %q", k)
	}
	// Direct load returns the requested string directly (does not go through quantLabel).
	if got := m.Quant(); got != "int4" {
		t.Errorf("direct load Quant() = %q, want int4", got)
	}

	// The .giw path (m.quant == "") infers via quantLabel — where T1-6 mislabeled as int4mix.
	blob, err := SerializeWeights(m.w, "int4-label-rt")
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	w2, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatalf("deserialize: %v", err)
	}
	// The round-trip must preserve the int8 head — the exact state that tripped the bug.
	if k := w2.Embed.Kind(); k != "int8" {
		t.Fatalf("round-trip changed embed kind to %q", k)
	}
	m2, err := NewModel(w2, "cpu")
	if err != nil {
		t.Fatalf("new model: %v", err)
	}
	if got := m2.Quant(); got != "int4" {
		t.Errorf("giw Quant() = %q, want int4 — an int8-pinned embed/head must not force int4mix (T1-6)", got)
	}
}
