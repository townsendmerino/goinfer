package multimodal

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/vision"
)

// TestQwenVisionEncoder_crosscheck runs the aikit v1.8.1 Qwen2.5-VL vision tower
// (LoadQwenVisionEncoder/Forward) on goinfer's OWN tiny checkpoint + pinned image
// golden (scripts/pin_qwen25vl_image.py) — a cross-repo consistency check that the
// encoder goinfer depends on produces what goinfer's decoder gate expects. Stage
// isolated: ForwardViT vs the ViT pre-merge hidden, Forward vs the merged features
// (the embeddings that replace <image> placeholders). aikit gates the same encoder
// against its own HF fixture; this confirms our fixtures agree.
func TestQwenVisionEncoder_crosscheck(t *testing.T) {
	const golden = "../testdata/qwen25vl_tiny_image_golden.json"
	const ckpt = "../testdata/qwen25vl-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_qwen25vl_image.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint — run scripts/pin_qwen25vl_tiny.py")
	}
	var g struct {
		PixelValues   []float32 `json:"pixel_values"`
		GridTHW       [][3]int  `json:"grid_thw"`
		VitHidden     []float32 `json:"vit_hidden"`
		ImageFeatures []float32 `json:"image_features"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	enc, err := vision.LoadQwenVisionEncoder(ckpt, false) // f32: bit-exact path
	if err != nil {
		t.Fatalf("LoadQwenVisionEncoder: %v", err)
	}

	// Stage 1: ViT pre-merge hidden.
	gotVit, err := enc.ForwardViT(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatalf("ForwardViT: %v", err)
	}
	if cos, maxAbs := cosine(gotVit, g.VitHidden); cos < 0.9999 {
		t.Errorf("ViT pre-merge cosine %.6f < 0.9999 (maxAbs %.4g, len got=%d want=%d)", cos, maxAbs, len(gotVit), len(g.VitHidden))
	}

	// Stage 2: merged image features (the placeholder-replacement embeddings).
	gotFeat, err := enc.Forward(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	cos, maxAbs := cosine(gotFeat, g.ImageFeatures)
	t.Logf("qwen ViT cross-check: merged cosine=%.6f maxAbs=%.4g (len got=%d want=%d)", cos, maxAbs, len(gotFeat), len(g.ImageFeatures))
	if cos < 0.9999 {
		t.Errorf("merged image_features cosine %.6f < 0.9999", cos)
	}
}
