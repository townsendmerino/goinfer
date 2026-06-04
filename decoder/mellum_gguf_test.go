package decoder

import (
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// TestMellumGGUF_runs is the end-to-end smoke test for Mellum2 from a bare GGUF:
// the mellum config + stacked-expert MoE + QK-norm un-permute + YaRN all resolve,
// and a forward step produces finite logits. Loads an ~8 GB GGUF (gitignored), so
// it skips when absent or under -short.
//
// Get the asset:
//
//	hf download CodeFault/Mellum2-12B-A2.5B-Instruct-GGUF \
//	  Mellum2-12B-A2.5B-Instruct-Q4_K_M.gguf --local-dir testdata/mellum-gguf
func TestMellumGGUF_runs(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads an ~8 GB Mellum2 GGUF")
	}
	path := "../testdata/mellum-gguf/Mellum2-12B-A2.5B-Instruct-Q4_K_M.gguf"
	if _, err := os.Stat(path); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Mellum2 GGUF at %s", path)
	}

	m, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	a := m.w.arch
	if a.Name != "mellum" {
		t.Fatalf("arch = %q, want mellum", a.Name)
	}
	if a.MoE == nil || a.MoE.NumExperts != 64 || a.MoE.TopK != 8 || a.MoE.IntermediateDim != 896 {
		t.Fatalf("MoE = %+v, want 64/8/896", a.MoE)
	}
	if !a.QKNorm {
		t.Fatal("QKNorm should be true for Mellum")
	}
	// The full-attention layers must carry the YaRN mscale; sliding layers 1.0.
	if !a.isGlobalLayer(3) || a.ropeMscale(3) <= 1.0 {
		t.Errorf("layer 3 should be global with YaRN mscale, got global=%v mscale=%v", a.isGlobalLayer(3), a.ropeMscale(3))
	}
	if a.isGlobalLayer(0) || a.ropeMscale(0) != 1.0 {
		t.Errorf("layer 0 should be sliding with mscale 1.0")
	}

	// One forward step on a few arbitrary token ids — the logits must be finite
	// and argmax in range (a wrong permute / expert slice would NaN or saturate).
	ids := []int{15, 9217, 327, 30}
	cache := m.NewCache(len(ids))
	for _, id := range ids[:len(ids)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			t.Fatalf("runLayers: %v", err)
		}
	}
	logits, err := m.forward(ids[len(ids)-1], cache)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if len(logits) != a.VocabSize {
		t.Fatalf("logits len %d, want vocab %d", len(logits), a.VocabSize)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit %d = %v (non-finite)", i, v)
		}
	}
	am := argmax(logits)
	if am < 0 || am >= len(logits) {
		t.Fatalf("argmax %d out of range", am)
	}
	t.Logf("mellum GGUF: %d layers, vocab %d, finite logits, argmax=%d", a.NumLayers, a.VocabSize, am)
}
