package decoder

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
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

// TestGiwInt4LabelNotMix is the T1-6 regression. A .giw baked with `-quant int4` has int4 projections but an int8-pinned
// embedding / LM head (the logit-critical default; the EmbedInt4 knob relaxes it). quantLabel must not scan those tables,
// see int4 coexisting with int8, and report "int4mix": /health would show `decode_path: …(int4mix)` beside
// `prefill_batched: true`, naming a quant the bundle is not.
//
// The .giw path is what triggers the inference: a direct Load records the requested quant string and returns it verbatim,
// never inferring. So the round-trip through SerializeWeights is load-bearing here, not incidental.
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

// moeInt4Tiny loads the committed qwen3_moe tiny at int4 and checks the condition these tests are about: its routers are
// float32 beside int4 experts and projections.
func moeInt4Tiny(t *testing.T) *Model {
	t.Helper()
	m, err := Load("../testdata/qwen3moe-tiny", Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load qwen3moe-tiny at int4: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	kinds := map[string]int{}
	for _, wm := range m.w.bodyMatmulWeights() {
		if wm.Rows() > 0 {
			kinds[wm.Kind()]++
		}
	}
	if kinds["int4"] == 0 || kinds["f32"] == 0 || len(kinds) != 2 {
		t.Fatalf("precondition: want int4 body weights beside float32 routers only, got %v", kinds)
	}
	return m
}

// bakeAndLoad serializes m's weights to a .giw in a temp dir and loads it back with no quant asked for.
func bakeAndLoad(t *testing.T, m *Model, name string) *Model {
	t.Helper()
	blob, err := SerializeWeights(m.w, "label")
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, giw.Write(blob, nil), 0o644); err != nil {
		t.Fatal(err)
	}
	g, err := Load(p, Options{})
	if err != nil {
		t.Fatalf("load %s: %v", name, err)
	}
	t.Cleanup(func() { g.Close() })
	return g
}

// TestQuantLabel_float32BesideInt4IsNotAMix: a MoE keeps its routers float32 at every quant, so an int4 load of one is
// int4, not int4mix, and an explicit --quant int4 is accepted against the .giw baked from it (a sidecar is one).
func TestQuantLabel_float32BesideInt4IsNotAMix(t *testing.T) {
	m := moeInt4Tiny(t)
	if got := m.w.quantLabel(); got != "int4" {
		t.Errorf("quantLabel() = %q, want int4", got)
	}
	g := bakeAndLoad(t, m, "moe.int4.giw")
	if g.w.bakedQuant != "int4" || g.Quant() != "int4" {
		t.Errorf("baked bundle: header %q, Quant() %q, want int4 for both", g.w.bakedQuant, g.Quant())
	}
	if err := g.CheckGiwQuantMatch("int4"); err != nil {
		t.Errorf("an explicit int4 against the int4 bundle was refused: %v", err)
	}
	if err := g.CheckGiwQuantMatch("int8int8"); err == nil {
		t.Error("an explicit int8int8 against the int4 bundle was accepted")
	}
}

// TestQuantLabel_oldInt4MixHeaderIsRederived: a bundle written under the rule that called float32 beside int4 a mix
// still says "int4mix" in its header. The reader checks that one value against the weights, so the file needs no rebuild.
func TestQuantLabel_oldInt4MixHeaderIsRederived(t *testing.T) {
	m := moeInt4Tiny(t)
	giwLabelForTest = "int4mix"
	g := bakeAndLoad(t, m, "moe.old.int4.giw")
	giwLabelForTest = ""
	if g.w.bakedQuant != "int4" || g.Quant() != "int4" {
		t.Errorf("old-header bundle: label %q, Quant() %q, want int4 re-derived from the weights", g.w.bakedQuant, g.Quant())
	}
	if err := g.CheckGiwQuantMatch("int4"); err != nil {
		t.Errorf("an explicit int4 against the old-header bundle was refused: %v", err)
	}
}

// TestQuantLabel_int8BesideInt4IsAMix is the control: int4 beside an int8 body weight is still int4mix, in memory and
// after a bake, and an explicit int4 against it is still refused.
func TestQuantLabel_int8BesideInt4IsAMix(t *testing.T) {
	m := moeInt4Tiny(t)
	l := &m.w.Layers[0]
	rows, cols := l.QProj.Rows(), l.QProj.Cols()
	f := make([]float32, rows*cols)
	for i := range f {
		f[i] = float32(i%7) * 0.01
	}
	l.QProj = quantizeWM(linalg.WrapF32(f, rows, cols), quantInt8I8)
	if got := m.w.quantLabel(); got != "int4mix" {
		t.Fatalf("quantLabel() = %q with an int8 projection beside int4, want int4mix", got)
	}
	g := bakeAndLoad(t, m, "moe.mix.giw")
	if g.Quant() != "int4mix" {
		t.Errorf("baked mix: Quant() %q, want int4mix", g.Quant())
	}
	if err := g.CheckGiwQuantMatch("int4"); err == nil {
		t.Error("an explicit int4 against a real int4mix bundle was accepted")
	}
}
