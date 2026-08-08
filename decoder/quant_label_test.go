package decoder

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/internal/giw"
)

// TestGiwRecordsResolvedQuant is the ITEM-4b gate: a v5 .giw baked by the buffer path records the
// resolved quant label in its header, and the reader PREFERS that field over re-inferring; a bundle
// with the field absent (empty) still infers correctly. Both paths, on a real int4 bake.
func TestGiwRecordsResolvedQuant(t *testing.T) {
	path := prequantGGUF(t)
	m, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load int4: %v", err)
	}
	blob, serr := SerializeWeights(m.w, "t4b")
	m.Close()
	if serr != nil {
		t.Fatalf("serialize: %v", serr)
	}
	giwPath := filepath.Join(t.TempDir(), "q05.int4.giw")
	if err := os.WriteFile(giwPath, giw.Write(blob, nil), 0o644); err != nil {
		t.Fatalf("write .giw: %v", err)
	}
	g, err := Load(giwPath, Options{})
	if err != nil {
		t.Fatalf("load .giw: %v", err)
	}
	defer g.Close()

	// Path 1 — field PRESENT: the buffer bake recorded the resolved label and the reader read it.
	if g.w.bakedQuant != "int4" {
		t.Fatalf("v5 buffer bundle should record bakedQuant=int4, got %q", g.w.bakedQuant)
	}
	if got := g.Quant(); got != "int4" {
		t.Errorf("Quant() = %q, want int4", got)
	}
	// Prefer the field over inference: force a value inference would NOT produce and confirm Quant()
	// returns the recorded field, proving it is read, not re-derived.
	g.w.bakedQuant = "int8int8"
	if got := g.Quant(); got != "int8int8" {
		t.Errorf("Quant() = %q — must PREFER the recorded field (int8int8) over inference (int4)", got)
	}

	// Path 2 — field ABSENT (empty, as a pre-v5 or streamed bundle): fall back to inference, which
	// (post-T1-6) correctly yields int4 for this all-int4-body bundle.
	g.w.bakedQuant = ""
	if got := g.Quant(); got != g.w.quantLabel() || got != "int4" {
		t.Errorf("field absent must fall back to inference: Quant()=%q quantLabel()=%q, want int4", got, g.w.quantLabel())
	}
}

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
