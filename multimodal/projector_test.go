package multimodal

import (
	"encoding/json"
	"os"
	"testing"
)

// TestProjector_parity gates the Gemma 3 multimodal projector against the HF
// golden: feed the recorded vision last_hidden_state, compare the projected image
// embeddings to HF's (cosine ≈ 1.0). Isolates the projector from the encoder.
func TestProjector_parity(t *testing.T) {
	const ckpt = "../testdata/gemma3-vl-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no gemma3-vl-tiny checkpoint (%v); run scripts/pin_gemma3_vl_tiny.py", err)
	}
	raw, err := os.ReadFile("../testdata/gemma3_vl_tiny_image_golden.json")
	if err != nil {
		t.Skipf("no image golden (%v); run scripts/pin_gemma3_vl_image.py", err)
	}
	var g struct {
		VisionLHS     []float32 `json:"vision_last_hidden_state"`
		ImageFeatures []float32 `json:"image_features"`
		ImageShape    []int     `json:"image_features_shape"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if len(g.VisionLHS) == 0 {
		t.Skip("golden lacks vision_last_hidden_state — regenerate with the updated pin")
	}

	proj, err := LoadProjector(ckpt)
	if err != nil {
		t.Fatalf("LoadProjector: %v", err)
	}
	got, err := proj.Forward(g.VisionLHS)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if len(got) != len(g.ImageFeatures) {
		t.Fatalf("output len %d != golden %d (shape %v)", len(got), len(g.ImageFeatures), g.ImageShape)
	}
	cos, maxAbs := cosine(got, g.ImageFeatures)
	t.Logf("projector vs HF golden: cosine=%.8f, max abs diff=%.3e (shape %v, %d image tokens)", cos, maxAbs, g.ImageShape, proj.MMTokens())
	if cos < 0.9999 {
		t.Errorf("image_features cosine %.8f < 0.9999 — projector diverges from HF", cos)
	}
}
