package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// TestGemma4VL_textParity loads the tiny Gemma 4 VL checkpoint (scripts/
// pin_gemma4_vl_tiny.py) through goinfer's loader + forward and asserts the
// TEXT-ONLY path matches the HF golden. Same P0 invariant as Gemma 3's own
// TestGemma3VL_textParity: a VL checkpoint's text decoder (vision_tower/
// embed_vision ignored) loads and runs exactly like a plain gemma4.
//
// This fixture's num_kv_shared_layers=2 (of 4 layers) is not incidental — it
// is the exact shape that caught two real, pre-existing bugs in
// buildWeightsFromSafetensors's gemma4 branch this session (a missing
// cross-layer-KV-sharing skip, and a missing per-layer FFN-width discovery),
// found only by loading a real checkpoint of this shape for the first time.
// This test is what gives that fix CI-repeatable coverage.
func TestGemma4VL_textParity(t *testing.T) {
	const golden = "../testdata/gemma4_vl_tiny_text_golden.json"
	const ckpt = "../testdata/gemma4-vl-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_gemma4_vl_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_gemma4_vl_tiny.py", ckpt)
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
	if m.w.arch.gemma4 == nil {
		t.Fatalf("loaded model is not gemma4 (arch=%s)", m.w.arch.Name)
	}
	if got := m.w.arch.gemma4.SharedKVLayers; got != 2 {
		t.Fatalf("SharedKVLayers = %d, want 2 (fixture didn't load the shape this test exists to cover)", got)
	}

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("gemma4-VL text parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.9999 {
		t.Errorf("last-logit cosine %.6f < 0.9999", cos)
	}

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

// TestGemma4VL_imageParity is the end-to-end gate: a multimodal prompt (text +
// an image placeholder run + text) with the tower's projected features
// interleaved at the placeholders must reproduce the HF
// Gemma4ForConditionalGeneration logits. It injects the golden's (separately
// gated) image_features so this isolates the decoder-side wiring
// (prefillLogitsGemma4VL's sequential embed-by-vector walk) from the vision
// tower itself — same shape as Gemma 3's TestGemma3VL_imageParity.
func TestGemma4VL_imageParity(t *testing.T) {
	const golden = "../testdata/gemma4_vl_tiny_image_golden.json"
	const ckpt = "../testdata/gemma4-vl-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no image golden — run scripts/pin_gemma4_vl_image.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint — run scripts/pin_gemma4_vl_tiny.py")
	}
	var g struct {
		InputIDs        []int     `json:"input_ids"`
		ImageTokenStart int       `json:"image_token_start"`
		NImageTokens    int       `json:"n_image_tokens"`
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
	logits, err := m.prefillLogitsGemma4VL(context.Background(), g.InputIDs, g.ImageFeatures, g.ImageTokenStart, g.NImageTokens, cache)
	if err != nil {
		t.Fatalf("prefillLogitsGemma4VL: %v", err)
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("gemma4-VL image parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("image-prompt argmax = %d, want %d", gotArg, g.Argmax)
	}
	// Sequential (one-token-at-a-time) prefill through the exact same
	// runLayersGemma4/runLayersGemma4FromEmbed path plain decode uses — no
	// separate batched-attention kernel in play here, so this holds to the
	// same bit-tight bar as the text-only test, not the batched-attention
	// 0.99 floor Gemma 3's image gate uses.
	if cos < 0.9999 {
		t.Errorf("image-prompt last-logit cosine %.6f < 0.9999", cos)
	}
}
