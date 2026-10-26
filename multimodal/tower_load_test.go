package multimodal

import (
	"math/rand/v2"
	"os"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// TestLoadGlmOcrTower_matchesTheEncoder: the wrapper the invoice example uses gives the encoder's own bits.
func TestLoadGlmOcrTower_matchesTheEncoder(t *testing.T) {
	const dir = "../testdata/glm-ocr-vision-tiny"
	tw, err := LoadGlmOcrTower(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer tw.Close()
	enc, err := vision.LoadGlmOcrVisionEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	c := enc.Cfg
	grid := [3]int{1, 6, 8}
	rng := rand.New(rand.NewPCG(1, 1))
	px := make([]float32, grid[0]*grid[1]*grid[2]*c.InChannels*c.TemporalPatchSize*c.PatchSize*c.PatchSize)
	for i := range px {
		px[i] = float32(rng.NormFloat64())
	}
	got, err := tw.Forward(px, grid)
	if err != nil {
		t.Fatal(err)
	}
	want, err := enc.Forward(px, [][3]int{grid})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 || !slices.Equal(got, want) {
		t.Fatalf("GlmOcrTower.Forward differs from the encoder's (%d against %d values)", len(got), len(want))
	}
}

// TestLoadSiglipTower_matchesTheEncoder: the wrapper the agent demo uses preprocesses and runs as the encoder does, and a
// resident request with no backend linked in is an error, not a CPU tower.
func TestLoadSiglipTower_matchesTheEncoder(t *testing.T) {
	const dir = "../testdata/siglip-tiny"
	img, err := os.ReadFile("../testdata/gemma3_preprocess_image.png")
	if err != nil {
		t.Fatal(err)
	}
	tw, err := LoadSiglipTower(dir, false, false)
	if err != nil {
		t.Fatal(err)
	}
	defer tw.Close()
	px, err := tw.Preprocess(img)
	if err != nil {
		t.Fatal(err)
	}
	pv, err := vision.Preprocess(img, vision.Gemma3())
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(px, pv.Data) {
		t.Fatal("SiglipTower.Preprocess differs from vision.Preprocess with Gemma3()")
	}
	enc, err := vision.LoadEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	in := make([]float32, enc.Cfg.NumChannels*enc.Cfg.ImageSize*enc.Cfg.ImageSize)
	rng := rand.New(rand.NewPCG(2, 1))
	for i := range in {
		in[i] = float32(rng.NormFloat64())
	}
	got, err := tw.Forward(in)
	if err != nil {
		t.Fatal(err)
	}
	want, err := enc.Forward(in)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 || !slices.Equal(got, want) {
		t.Fatal("SiglipTower.Forward differs from the encoder's")
	}
	if _, err := LoadSiglipTower(dir, false, true); err == nil {
		t.Fatal("a resident SigLIP tower loaded with no backend linked into the test binary")
	}
}
