package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// TestGenerateVL_streams is the multimodal P4 plumbing gate: GenerateVL prefills
// a multimodal prompt (text + interleaved image features under the bidirectional
// mask) and streams a greedy continuation. It reuses the P3 image golden, so the
// FIRST sampled token must equal the golden's argmax (greedy first token ==
// argmax of prefillLogitsVL's last-position logits, which TestGemma3VL_imageParity
// pins), confirming GenerateVL drives the proven prefill + a clean decode loop.
func TestGenerateVL_streams(t *testing.T) {
	const golden = "../testdata/gemma3_vl_tiny_image_golden.json"
	const ckpt = "../testdata/gemma3-vl-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no image golden — run scripts/pin_gemma3_vl_image.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skip("no checkpoint — run scripts/pin_gemma3_vl_tiny.py")
	}
	var g struct {
		InputIDs        []int     `json:"input_ids"`
		ImageTokenStart int       `json:"image_token_start"`
		MMTokens        int       `json:"mm_tokens_per_image"`
		ImageFeatures   []float32 `json:"image_features"`
		Argmax          int       `json:"argmax"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	const maxNew = 5
	stream, gen := m.GenerateVL(context.Background(), g.InputIDs, g.ImageFeatures, g.ImageTokenStart, g.MMTokens, maxNew, SamplingParams{Temperature: 0})
	var got []int
	for id := range stream {
		got = append(got, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("GenerateVL: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("GenerateVL streamed no tokens")
	}
	if len(got) > maxNew {
		t.Errorf("streamed %d tokens, want ≤ %d", len(got), maxNew)
	}
	// Greedy first token == the golden's pinned argmax (ties the stream to P3 parity).
	if got[0] != g.Argmax {
		t.Errorf("first token = %d, want golden argmax %d", got[0], g.Argmax)
	}
	t.Logf("GenerateVL streamed %d tokens (first=%d == argmax %d)", len(got), got[0], g.Argmax)
}
