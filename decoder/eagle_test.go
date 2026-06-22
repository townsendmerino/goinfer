package decoder

import (
	"os"
	"path/filepath"
	"testing"
)

// TestLoadEagleHead gates the 05 head loader (inc 2): the AngelSlim Qwen3-1.7B EAGLE-3
// head (converted to f32 safetensors) loads with the confirmed shapes — fc fuses
// 3*hidden, the layer projects from 2*hidden, lm_head is the draft vocab, d2t maps
// back to the target vocab. Skips unless the converted head is present at
// ~/models/qwen3-1.7b-eagle3 (GINFER_EAGLE_DIR overrides).
func TestLoadEagleHead(t *testing.T) {
	dir := os.Getenv("GINFER_EAGLE_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "qwen3-1.7b-eagle3")
	}
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("no EAGLE head at %s (%v); convert AngelSlim/Qwen3-1.7B_eagle3", dir, err)
	}
	h, err := LoadEagleHead(dir)
	if err != nil {
		t.Fatalf("LoadEagleHead: %v", err)
	}
	defer h.Close()
	// Confirmed Qwen3-1.7B head geometry.
	if h.hidden != 2048 || h.draftVocab != 32000 || h.vocab != 151936 {
		t.Errorf("dims: hidden %d draftVocab %d vocab %d", h.hidden, h.draftVocab, h.vocab)
	}
	// fc fuses 3*hidden → hidden; q projects from 2*hidden.
	if r, c := h.fc.Rows(), h.fc.Cols(); r != h.hidden || c != 3*h.hidden {
		t.Errorf("fc shape [%d,%d], want [%d,%d]", r, c, h.hidden, 3*h.hidden)
	}
	if r, c := h.q.Rows(), h.q.Cols(); r != h.nHeads*h.headDim || c != 2*h.hidden {
		t.Errorf("q shape [%d,%d], want [%d,%d]", r, c, h.nHeads*h.headDim, 2*h.hidden)
	}
	if r, c := h.lmHead.Rows(), h.lmHead.Cols(); r != h.draftVocab || c != h.hidden {
		t.Errorf("lm_head shape [%d,%d], want [%d,%d]", r, c, h.draftVocab, h.hidden)
	}
	if len(h.d2t) != h.draftVocab {
		t.Errorf("d2t len %d, want %d", len(h.d2t), h.draftVocab)
	}
	if len(h.hiddenNorm) != h.hidden || len(h.finalNorm) != h.hidden {
		t.Errorf("norm lens: hidden %d final %d", len(h.hiddenNorm), len(h.finalNorm))
	}
	// d2t maps draft indices into the target vocab range.
	for i, off := range h.d2t {
		tid := i + int(off)
		if tid < 0 || tid >= h.vocab {
			t.Fatalf("d2t[%d]=%d → target id %d out of range [0,%d)", i, off, tid, h.vocab)
		}
	}
	t.Logf("EAGLE head loaded: hidden %d, heads %d/%d hd %d, inter %d, draftVocab %d, ropeTheta %.0f",
		h.hidden, h.nHeads, h.nKV, h.headDim, h.inter, h.draftVocab, h.ropeTheta)
}
