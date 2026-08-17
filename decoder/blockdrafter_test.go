package decoder

import (
	"testing"
)

// TestBlockDrafterWeights_matchesTrunk checks the interface reports what the drafter actually
// holds, on a real checkpoint.
//
// The compile-time assertions in blockdrafter.go prove both families SATISFY the interface; they
// prove nothing about the values. A DrafterGeometry that reported the wrong head count or handed
// back layer 0's weights for every index would satisfy the compiler and produce a resident
// drafter that runs, allocates plausibly, and drafts noise — exactly the failure this program has
// hit repeatedly (a wrong tensor that still has the right shape).
//
// So this cross-checks every field against the trunk's own state, and specifically requires that
// DIFFERENT LAYERS RETURN DIFFERENT WEIGHTS, which is the assertion an index bug fails.
func TestBlockDrafterWeights_matchesTrunk(t *testing.T) {
	dir := assetPath(t, "GOINFER_DFLASH_F32")
	d, err := LoadDFlashDrafter(dir)
	if err != nil {
		t.Fatalf("LoadDFlashDrafter: %v", err)
	}
	defer d.Close()

	var w BlockDrafterWeights = d
	g := w.DrafterGeometry()
	for _, c := range []struct {
		name     string
		got, exp int
	}{
		{"Layers", g.Layers, len(d.layers)},
		{"Hidden", g.Hidden, d.hidden},
		{"NumHeads", g.NumHeads, d.nHeads},
		{"NumKVHeads", g.NumKVHeads, d.nKV},
		{"HeadDim", g.HeadDim, d.headDim},
		{"Intermediate", g.Intermediate, d.inter},
		{"BlockSize", w.BlockSize(), d.blockSize},
	} {
		if c.got != c.exp {
			t.Errorf("%s = %d, want %d", c.name, c.got, c.exp)
		}
	}
	if g.NormEps != d.normEps {
		t.Errorf("NormEps = %v, want %v", g.NormEps, d.normEps)
	}
	if len(g.InvFreq) != len(d.invFreq) {
		t.Errorf("InvFreq len %d, want %d", len(g.InvFreq), len(d.invFreq))
	}

	// fc is the projection with no counterpart in a normal layer — the one most likely to be
	// mis-wired, and the one a backend cannot infer from geometry alone.
	if fc := w.DrafterFC(); fc.Rows() != d.hidden || fc.Cols() != d.fc.Cols() {
		t.Errorf("FC is %dx%d, want %dx%d", fc.Rows(), fc.Cols(), d.hidden, d.fc.Cols())
	}
	if got := len(w.DrafterHiddenNorm()); got != len(d.hiddenNorm) {
		t.Errorf("HiddenNorm len %d, want %d", got, len(d.hiddenNorm))
	}
	if got := len(w.DrafterFinalNorm()); got != len(d.finalNorm) {
		t.Errorf("FinalNorm len %d, want %d", got, len(d.finalNorm))
	}

	// Per-layer: shapes, and that the index is honoured. An accessor returning layer 0 for
	// every i has correct shapes everywhere and is catastrophically wrong.
	seen := map[*float32]int{}
	for i := 0; i < g.Layers; i++ {
		lw := w.DrafterLayer(i)
		if lw.Q.Rows() != d.nHeads*d.headDim {
			t.Errorf("layer %d Q rows %d, want %d", i, lw.Q.Rows(), d.nHeads*d.headDim)
		}
		if lw.K.Rows() != d.nKV*d.headDim {
			t.Errorf("layer %d K rows %d, want %d", i, lw.K.Rows(), d.nKV*d.headDim)
		}
		if lw.Gate.Rows() != d.inter {
			t.Errorf("layer %d Gate rows %d, want %d", i, lw.Gate.Rows(), d.inter)
		}
		if lw.Down.Cols() != d.inter {
			t.Errorf("layer %d Down cols %d, want %d", i, lw.Down.Cols(), d.inter)
		}
		if len(lw.InputNorm) != d.hidden {
			t.Errorf("layer %d InputNorm len %d, want %d", i, len(lw.InputNorm), d.hidden)
		}
		// identity check: each layer's norm slice must be a DISTINCT backing array
		if len(lw.InputNorm) > 0 {
			p := &lw.InputNorm[0]
			if prev, dup := seen[p]; dup {
				t.Fatalf("layer %d returns the SAME InputNorm storage as layer %d — "+
					"DrafterLayer ignores its index", i, prev)
			}
			seen[p] = i
		}
	}
	t.Logf("%d layers, hidden %d, heads %d/%d, headDim %d, inter %d, block %d, fc %dx%d — "+
		"interface agrees with the trunk on every field",
		g.Layers, g.Hidden, g.NumHeads, g.NumKVHeads, g.HeadDim, g.Intermediate,
		w.BlockSize(), w.DrafterFC().Rows(), w.DrafterFC().Cols())
}
