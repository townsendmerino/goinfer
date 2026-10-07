package embeddinggemma2

import (
	"fmt"
	"math"
	"testing"

	"compress/gzip"
	"encoding/json"
	"os"

	"github.com/townsendmerino/aikit/audio"
)

// TestTinyAudio_matchesTransformers is Phase A's gate A2t (docs/tasks/task-embeddinggemma2.md): aikit's gemma4_audio
// tower on the committed tiny random tower (testdata/gemma4-audio-tiny, scripts/pin_gemma4_audio_tiny.py: finite
// asymmetric clip bounds binding in 15 of 20 projections, random norm weights and per_dim_scale, a 1.3 s clip of 33
// soft tokens so the 12-key window binds) against transformers in float32 with sdpa: the log-mel within 1e-4 of the
// extractor's with its valid-frame count, and every stage (the subsampler, each block, the tower, the embedder) at
// cosine >= 0.9999 on every soft token.
func TestTinyAudio_matchesTransformers(t *testing.T) {
	var g struct {
		Samples  []float32   `json:"samples"`
		TValid   int         `json:"t_valid"`
		Features []float32   `json:"features"`
		NSoft    int         `json:"n_soft"`
		Sub      []float32   `json:"sub"`
		Blocks   [][]float32 `json:"blocks"`
		Tower    []float32   `json:"tower"`
		Embed    []float32   `json:"embed"`
	}
	f, err := os.Open("../testdata/gemma4-audio-tiny/golden.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.NewDecoder(zr).Decode(&g); err != nil {
		t.Fatal(err)
	}
	mel, T, err := audio.Gemma4Features(g.Samples)
	if err != nil {
		t.Fatal(err)
	}
	if T != g.TValid || len(mel) != len(g.Features) {
		t.Fatalf("log-mel: %d frames, the extractor %d", T, g.TValid)
	}
	md := 0.0
	for i := range mel {
		md = math.Max(md, math.Abs(float64(mel[i]-g.Features[i])))
	}
	if md > 1e-4 {
		t.Errorf("log-mel max|diff| %.3e, over 1e-4", md)
	}
	enc, err := audio.LoadGemma4AudioEncoder("../testdata/gemma4-audio-tiny")
	if err != nil {
		t.Fatal(err)
	}
	out, st, err := enc.ForwardStages(g.Features, g.TValid)
	if err != nil {
		t.Fatal(err)
	}
	if n := audio.Gemma4SoftTokens(T); n != g.NSoft {
		t.Fatalf("%d soft tokens, transformers %d", n, g.NSoft)
	}
	H := enc.Cfg.HiddenSize
	type stage struct {
		name      string
		got, want []float32
		d         int
	}
	stages := []stage{{"subsampler", st.Sub, g.Sub, H}}
	for L := range g.Blocks {
		stages = append(stages, stage{fmt.Sprintf("block %d", L), st.Blocks[L], g.Blocks[L], H})
	}
	stages = append(stages, stage{"tower", st.Tower, g.Tower, enc.Cfg.OutputProjDims}, stage{"embedder", out, g.Embed, enc.TextHiddenSize})
	line := ""
	for _, s := range stages {
		if len(s.got) != len(s.want) {
			t.Fatalf("%s: %d values, transformers %d", s.name, len(s.got), len(s.want))
		}
		w, m := cosRowsTiny(s.got, s.want, s.d)
		line += fmt.Sprintf(" %s %.9f/%.1e;", s.name, w, m)
		if w < 0.9999 {
			t.Errorf("%s: worst soft-token cosine %.9f, under 0.9999 (the first stage under the bar names the bug)", s.name, w)
			break
		}
	}
	t.Logf("log-mel %d frames max|diff| %.2e;%s", T, md, line)
}

func cosRowsTiny(got, want []float32, d int) (float64, float64) {
	worst, md := 1.0, 0.0
	for r := range len(want) / d {
		var dot, na, nb float64
		for j := range d {
			x, y := float64(got[r*d+j]), float64(want[r*d+j])
			dot, na, nb = dot+x*y, na+x*x, nb+y*y
			md = math.Max(md, math.Abs(x-y))
		}
		worst = math.Min(worst, dot/math.Sqrt(na*nb))
	}
	return worst, md
}
