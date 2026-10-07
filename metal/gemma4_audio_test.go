//go:build darwin

package metal

import (
	"compress/gzip"
	"encoding/json"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/audio"
)

// TestG4AMetal_matchesCPU is Phase A's gate AM1 on the tiny tower (docs/tasks/task-embeddinggemma2.md): the Metal
// blocks between aikit's subsampler and tail match aikit's CPU Forward to cosine 0.9999 on every soft token, and
// transformers' golden too, on the committed tiny random tower (testdata/gemma4-audio-tiny: finite asymmetric clip
// bounds binding in 15 of 20 projections, random norms and per_dim_scale, 33 soft tokens so the 12-key window binds).
func TestG4AMetal_matchesCPU(t *testing.T) {
	f, err := os.Open("../testdata/gemma4-audio-tiny/golden.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		TValid   int       `json:"t_valid"`
		Features []float32 `json:"features"`
		Embed    []float32 `json:"embed"`
	}
	if err := json.NewDecoder(zr).Decode(&g); err != nil {
		t.Fatal(err)
	}
	enc, err := audio.LoadGemma4AudioEncoder("../testdata/gemma4-audio-tiny")
	if err != nil {
		t.Fatal(err)
	}
	a, err := newG4AAccel(enc)
	if err != nil {
		t.Fatal(err)
	}
	cpu, err := enc.Forward(g.Features, g.TValid)
	if err != nil {
		t.Fatal(err)
	}
	h, n, err := enc.Subsample(g.Features, g.TValid)
	if err != nil {
		t.Fatal(err)
	}
	hb, err := a.Blocks(h, n)
	if err != nil {
		t.Fatal(err)
	}
	got, err := enc.FinishBlocks(hb, n)
	if err != nil {
		t.Fatal(err)
	}
	wc, wh := g4vWorst(got, cpu, enc.TextHiddenSize), g4vWorst(got, g.Embed, enc.TextHiddenSize)
	t.Logf("tiny audio tower, %d soft tokens: Metal blocks against the CPU tower worst soft-token cosine %.9f, against transformers %.9f", n, wc, wh)
	if wc < 0.9999 || wh < 0.9999 {
		t.Errorf("worst soft-token cosine %.9f (CPU) / %.9f (transformers), under 0.9999", wc, wh)
	}
}
