package multimodal

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/vision"
)

// TestGemma4Preprocess_matchesTransformers is F1c of docs/multimodal.md ("Finishing this doc"): aikit's
// vision.Gemma4Preprocess (bicubic, the default since aikit v1.58.0) gives transformers' Gemma4ImageProcessor's pixels
// bit for bit, processor defaults, on four repo images none of which is at its target size: the same patch count,
// the same positions, and the same SHA-256 over the patches as uint8 (testdata/gemma4-preprocess-pixels.json,
// scripts/pin_gemma4_preprocess_pixels.py). The bilinear resampler, run beside it, must differ: the control that
// shows the hash sees a resampler change.
func TestGemma4Preprocess_matchesTransformers(t *testing.T) {
	raw, err := os.ReadFile("../testdata/gemma4-preprocess-pixels.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Transformers string `json:"transformers"`
		Images       []struct {
			Path      string   `json:"path"`
			NPatches  int      `json:"n_patches"`
			Positions [][2]int `json:"positions"`
			SHA256    string   `json:"sha256"`
		} `json:"images"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	hash := func(p []float32) string {
		u8 := make([]byte, len(p))
		for i, v := range p {
			u8[i] = byte(math.Round(float64(v) * 255))
		}
		s := sha256.Sum256(u8)
		return hex.EncodeToString(s[:])
	}
	for _, im := range g.Images {
		data, err := os.ReadFile("../" + im.Path)
		if err != nil {
			t.Fatal(err)
		}
		p, pos, err := vision.Gemma4Preprocess(data, 280)
		if err != nil {
			t.Fatal(err)
		}
		if len(pos) != im.NPatches || fmt.Sprint(pos) != fmt.Sprint(im.Positions) {
			t.Errorf("%s: %d patches (transformers %d) or positions differ", im.Path, len(pos), im.NPatches)
			continue
		}
		if h := hash(p); h != im.SHA256 {
			t.Errorf("%s: patch pixels differ from transformers %s (sha256 %s, want %s)", im.Path, g.Transformers, h[:12], im.SHA256[:12])
		}
		pb, _, err := vision.Gemma4PreprocessResize(data, 280, vision.Gemma4Bilinear)
		if err != nil {
			t.Fatal(err)
		}
		if hash(pb) == im.SHA256 {
			t.Errorf("%s: control: bilinear hashes equal to transformers' bicubic, so the hash cannot see the resampler", im.Path)
		}
	}
}
