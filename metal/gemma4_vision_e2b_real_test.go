//go:build darwin

package metal

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/vision"
)

// TestG4VMetal_gemma4E2B is F2a of docs/multimodal.md ("Finishing this doc"): Gemma 4 E2B's own vision tower
// (GOINFER_GEMMA4_E2B_VISION, default ~/models/gemma-4-E2B-unq, never the archive; finite clip bounds, no
// standardize) on Metal against aikit's CPU tower, every soft token at cosine >= 0.9999, on the four repo images the
// EmbeddingGemma 2 gates use, through aikit's own preprocessing (the reference bicubic). Times both, exploratory.
//
//	GOINFER_EG2_REAL=1 go test -count=1 -run '^TestG4VMetal_gemma4E2B$' -v ./metal/
func TestG4VMetal_gemma4E2B(t *testing.T) {
	if os.Getenv("GOINFER_EG2_REAL") != "1" {
		t.Skip("set GOINFER_EG2_REAL=1 (loads a real checkpoint)")
	}
	home, _ := os.UserHomeDir()
	dir := os.Getenv("GOINFER_GEMMA4_E2B_VISION")
	if dir == "" {
		dir = filepath.Join(home, "models", "gemma-4-E2B-unq")
	}
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", dir)
	}
	enc, err := vision.LoadGemma4Encoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	a, err := newG4VAccel(enc)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("E2B tower: hidden %d, %d layers, clipped %v, standardize %v, text width %d", enc.Cfg.HiddenSize, enc.Cfg.NumHiddenLayers, enc.Cfg.UseClippedLinears, enc.Cfg.Standardize, enc.TextHiddenSize)
	for _, img := range []string{"gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"} {
		data, err := os.ReadFile(filepath.Join("../testdata", img))
		if err != nil {
			t.Fatal(err)
		}
		p, pos, err := vision.Gemma4Preprocess(data, 280)
		if err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		cpu, err := enc.Forward(p, pos)
		tc := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		t0 = time.Now()
		h, err := a.Hidden(p, pos)
		if err != nil {
			t.Fatal(err)
		}
		got, err := enc.FinishHidden(h, pos)
		tm := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		w := g4vWorst(got, cpu, enc.TextHiddenSize)
		t.Logf("%-32s %4d patches: worst soft-token cosine %.9f; tower Metal %s, CPU %s (exploratory)", img, len(pos), w, tm.Round(time.Millisecond), tc.Round(time.Millisecond))
		if w < 0.9999 {
			t.Errorf("%s: worst soft-token cosine %.9f, under 0.9999", img, w)
		}
	}
}
