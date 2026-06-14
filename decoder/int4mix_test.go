package decoder

import "testing"

// TestInt4MixMode gates the per-tensor mixed-precision mode (idea #5): -quant int4mix
// keeps attention (q/k/v/o) at int8 — where the spike found the int4→int8 quality loss
// concentrated — and the FFN bulk (gate/up/down) at int4. Asserts the per-tensor kinds,
// that Quant() reports it distinctly (so the KV fingerprint won't collide with int4/int8),
// that it generates, and that a mixed model round-trips through the .giw serialize format.
func TestInt4MixMode(t *testing.T) {
	path := prequantGGUF(t)
	m, err := Load(path, Options{Quant: "int4mix"})
	if err != nil {
		t.Fatalf("load int4mix: %v", err)
	}
	defer m.Close()

	if m.Quant() != "int4mix" {
		t.Errorf("Quant() = %q, want int4mix (distinct fingerprint)", m.Quant())
	}

	l := &m.w.Layers[0]
	int8Tensor := func(name string, w interface {
		Int8() ([]int8, []float32, bool, bool)
	}) {
		if _, _, _, ok := w.Int8(); !ok {
			t.Errorf("%s should be int8 in mix mode", name)
		}
	}
	int4Tensor := func(name string, w interface {
		Int4() ([]byte, []float32, int, bool)
	}) {
		if _, _, _, ok := w.Int4(); !ok {
			t.Errorf("%s should be int4 in mix mode", name)
		}
	}
	int8Tensor("QProj", &l.QProj)
	int8Tensor("KProj", &l.KProj)
	int8Tensor("VProj", &l.VProj)
	int8Tensor("OProj", &l.OProj)
	int4Tensor("GateProj", &l.GateProj)
	int4Tensor("UpProj", &l.UpProj)
	int4Tensor("DownProj", &l.DownProj)

	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	tok := greedyFirst(t, m, prompt)
	if tok < 0 {
		t.Fatal("no token generated")
	}

	// A mixed model round-trips through the .giw format (per-weightMat kind), and the
	// greedy token is unchanged.
	blob, err := SerializeWeights(m.w, "int4mix-rt")
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	w2, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatalf("deserialize mixed model: %v", err)
	}
	if _, _, _, ok := w2.Layers[0].QProj.Int8(); !ok {
		t.Error("round-trip: QProj lost int8")
	}
	if _, _, _, ok := w2.Layers[0].DownProj.Int4(); !ok {
		t.Error("round-trip: DownProj lost int4")
	}
	m2, err := NewModel(w2, "cpu")
	if err != nil {
		t.Fatalf("new model: %v", err)
	}
	if got := greedyFirst(t, m2, prompt); got != tok {
		t.Fatalf("round-trip greedy token changed: %d → %d", tok, got)
	}
}
