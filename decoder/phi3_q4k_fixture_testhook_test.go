//go:build goinfer_testhooks

package decoder

import (
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestWritePhi3Q4KTinyGGUFForTest: the synthetic Phi-3 GGUF reads as what its callers take it for: a Phi-3 by its header, whose
// q4k load keeps every projection of both layers as Q4_K blocks and runs a forward over the 64-token vocab.
func TestWritePhi3Q4KTinyGGUFForTest(t *testing.T) {
	src := filepath.Join(t.TempDir(), "phi3.gguf")
	if err := WritePhi3Q4KTinyGGUFForTest(src); err != nil {
		t.Fatal(err)
	}
	if mt := PeekModelType(src); mt != "phi3" {
		t.Errorf("PeekModelType = %q, want phi3", mt)
	}
	m, err := Load(src, Options{Backend: "cpu", Quant: "q4k", DirectLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if n := len(m.w.Layers); n != 2 {
		t.Fatalf("%d layers, want 2 (gate S rebuilds a layer from the one before it)", n)
	}
	for i := range m.w.Layers {
		l := &m.w.Layers[i]
		for _, w := range []*linalg.WeightMat{&l.QProj, &l.KProj, &l.VProj, &l.OProj, &l.GateProj, &l.UpProj, &l.DownProj} {
			if _, ok := w.Q4K(); !ok {
				t.Errorf("layer %d: a %dx%d projection loads as %q, want Q4_K blocks", i, w.Rows(), w.Cols(), w.Kind())
			}
		}
	}
	if logits := d8bLogits(t, m); len(logits) != 8*64 {
		t.Errorf("8 forwards give %d logits, want 8 x the vocab's 64", len(logits))
	}
}
