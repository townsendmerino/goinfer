package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// TestGemma3VL_textParity loads the tiny Gemma 3 VL checkpoint (scripts/
// pin_gemma3_vl_tiny.py) through goinfer's loader + forward and asserts the
// TEXT-ONLY path matches the HF golden. This is the multimodal P0 invariant: a VL
// checkpoint's text decoder (vision_tower / multi_modal_projector ignored) loads
// and runs exactly like a plain gemma3 — the prerequisite before any image seam.
func TestGemma3VL_textParity(t *testing.T) {
	const golden = "../testdata/gemma3_vl_tiny_text_golden.json"
	const ckpt = "../testdata/gemma3-vl-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_gemma3_vl_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_gemma3_vl_tiny.py", ckpt)
	}
	var g struct {
		PromptIDs       []int     `json:"prompt_ids"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
		NNew            int       `json:"n_new"`
		ContinuationIDs []int     `json:"continuation_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("gemma3-VL text parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.9999 {
		t.Errorf("last-logit cosine %.6f < 0.9999", cos)
	}

	// Greedy continuation must match id-for-id.
	cur := gotArg
	for i := 0; i < g.NNew; i++ {
		if cur != g.ContinuationIDs[i] {
			t.Fatalf("continuation[%d] = %d, want %d", i, cur, g.ContinuationIDs[i])
		}
		if logits, err = m.forward(cur, cache); err != nil {
			t.Fatalf("decode forward: %v", err)
		}
		cur = argmax(logits)
	}
}
