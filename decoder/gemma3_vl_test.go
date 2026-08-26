package decoder

import (
	"context"
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

// TestGemma3VL_imageParity is the P3 end-to-end gate: a multimodal prompt
// (text + an <image> placeholder run + text) with the projector's image embeddings
// interleaved at the placeholders under the bidirectional image-block mask must
// reproduce the HF Gemma3ForConditionalGeneration logits. It injects the golden's
// (separately gated) image_features so this isolates the decoder-side wiring —
// embed-by-vector + SetImageBlocks — from the vision tower.
func TestGemma3VL_imageParity(t *testing.T) {
	const golden = "../testdata/gemma3_vl_tiny_image_golden.json"
	const ckpt = "../testdata/gemma3-vl-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no image golden — run scripts/pin_gemma3_vl_image.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint — run scripts/pin_gemma3_vl_tiny.py")
	}
	var g struct {
		InputIDs        []int     `json:"input_ids"`
		ImageTokenStart int       `json:"image_token_start"`
		MMTokens        int       `json:"mm_tokens_per_image"`
		ImageFeatures   []float32 `json:"image_features"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	cache := m.NewCache(len(g.InputIDs))
	logits, err := m.prefillLogitsVL(context.Background(), g.InputIDs, g.ImageFeatures, g.ImageTokenStart, g.MMTokens, cache)
	if err != nil {
		t.Fatalf("prefillLogitsVL: %v", err)
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("gemma3-VL image parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("image-prompt argmax = %d, want %d", gotArg, g.Argmax)
	}
	// The image path runs through the batched attendBatchedHeads (f32 SIMD), whose
	// parity standard vs the HF f64 reference is argmax-exact + cosine ≥0.99 (the
	// TestForwardN_matchesSequential bar) — NOT the bit-exact 1.0 of the streaming
	// text path. A wiring bug (wrong interleave position or mask range) craters
	// cosine and flips argmax; this gates both.
	if cos < 0.99 {
		t.Errorf("image-prompt last-logit cosine %.6f < 0.99 (batched-attention floor)", cos)
	}
}
