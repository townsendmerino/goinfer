//go:build realckpt

package embeddinggemma2

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// cosRows is the worst per-row cosine of got against want, [n, d] each, and the max |diff|.
func cosRows(got, want []float32, d int) (float64, float64) {
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

// TestReal_audio is Phase A's gates A1-A4 (docs/tasks/task-embeddinggemma2.md, pre-registered before this ran) on the
// real google/embeddinggemma-2 (GOINFER_EG2_DIR, default ~/models/embeddinggemma-2) against sentence-transformers'
// references: testdata/embeddinggemma2-audio/golden.json and its WAVs (committed), and GOINFER_EG2_AUDIO_ARTIFACTS
// (default ~/goinfer-logs/embeddinggemma2-audio), HF's own log-mel and tower stages per clip
// (scripts/pin_embeddinggemma2_audio.py):
//
//	A1: aikit's log-mel has HF's valid-frame count and is within 1e-4 of HF's features on every clip.
//	A2: fed HF's features, every stage (the subsampler, each of the 12 blocks, the tower, the embedder) is at cosine
//	    >= 0.9999 on every soft token; the first stage under the bar is named.
//	A3: every case's ids equal the reference's.
//	A4: every case's end-to-end embedding (aikit's log-mel and tower, then the encoder) is at cosine >= 0.9999.
//
// CPU, float32.
func TestReal_audio(t *testing.T) {
	home, _ := os.UserHomeDir()
	dir := os.Getenv("GOINFER_EG2_DIR")
	if dir == "" {
		dir = filepath.Join(home, "models", "embeddinggemma-2")
	}
	art := os.Getenv("GOINFER_EG2_AUDIO_ARTIFACTS")
	if art == "" {
		art = filepath.Join(home, "goinfer-logs", "embeddinggemma2-audio")
	}
	for _, p := range []string{dir, art} {
		if strings.HasPrefix(p, "/Volumes/") || strings.HasPrefix(p, "/srv/models") {
			t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", p)
		}
	}
	raw, err := os.ReadFile("../testdata/embeddinggemma2-audio/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Clips []struct {
			Name   string
			TValid int `json:"t_valid"`
			NSoft  int `json:"n_soft"`
		} `json:"clips"`
		Items []struct {
			Clip, Prompt, Text string
			IDs                []int     `json:"ids"`
			NSoft              int       `json:"n_soft"`
			Embedding          []float64 `json:"embedding"`
		} `json:"items"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	e, err := LoadEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.EnableAudio(); err != nil {
		t.Fatal(err)
	}
	tw := e.AudioTower()
	H := e.Model().Config().Hidden
	feats := map[string][]float32{}
	for _, c := range g.Clips {
		wav, err := os.ReadFile(filepath.Join("../testdata/embeddinggemma2-audio", c.Name+".wav"))
		if err != nil {
			t.Fatal(err)
		}
		samples, err := DecodeWAV(wav)
		if err != nil {
			t.Fatal(err)
		}
		// A1
		mel, T, err := AudioLogMel(samples)
		if err != nil {
			t.Fatal(err)
		}
		hfMel := readF32(t, filepath.Join(art, c.Name+".features.f32"))
		md := 0.0
		if T == c.TValid && len(hfMel) == len(mel) {
			for i := range mel {
				md = math.Max(md, math.Abs(float64(mel[i]-hfMel[i])))
			}
		}
		t.Logf("%-5s %6d samples: A1 %d valid frames (HF %d), log-mel max|diff| %.3e", c.Name, len(samples), T, c.TValid, md)
		if T != c.TValid || len(hfMel) != len(mel) || md > 1e-4 {
			t.Errorf("%s: A1 %d frames (HF %d), max|diff| %.3e over 1e-4", c.Name, T, c.TValid, md)
		}
		// A2, from HF's features
		t0 := time.Now()
		out, st, err := tw.ForwardStages(hfMel, c.TValid)
		took := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		type stage struct {
			name string
			got  []float32
			file string
			d    int
		}
		stages := []stage{{"subsampler", st.Sub, "sub", tw.Cfg.HiddenSize}}
		for L, b := range st.Blocks {
			stages = append(stages, stage{fmt.Sprintf("block %d", L), b, fmt.Sprintf("block%d", L), tw.Cfg.HiddenSize})
		}
		stages = append(stages, stage{"tower", st.Tower, "tower", tw.Cfg.OutputProjDims}, stage{"embedder", out, "embed", H})
		first := ""
		line := ""
		for _, s := range stages {
			want := readF32(t, filepath.Join(art, c.Name+"."+s.file+".f32"))
			if len(want) != len(s.got) {
				t.Fatalf("%s %s: %d values, HF %d", c.Name, s.name, len(s.got), len(want))
			}
			w, m := cosRows(s.got, want, s.d)
			line += fmt.Sprintf(" %s %.6f/%.1e", s.file, w, m)
			if w < 0.9999 && first == "" {
				first = s.name
			}
		}
		t.Logf("%-5s A2 (worst row cosine/max|diff|):%s; tower %s (exploratory)", c.Name, line, took.Round(time.Millisecond))
		if first != "" {
			t.Errorf("%s: A2 first stage under 0.9999: %s", c.Name, first)
		}
		f, n, err := e.AudioFeatures(samples)
		if err != nil {
			t.Fatal(err)
		}
		if n != c.NSoft {
			t.Errorf("%s: %d soft tokens, HF %d", c.Name, n, c.NSoft)
		}
		feats[c.Name] = f
	}
	for i, it := range g.Items {
		in := AudioInput{Text: it.Text, Prompt: it.Prompt}
		n := len(feats[it.Clip]) / H
		ids, _, err := e.TokenizeAudio(in, n)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprint(ids) != fmt.Sprint(it.IDs) {
			t.Errorf("item %d (%s, prompt %q, text %q): A3 ids differ (%d against %d)", i, it.Clip, it.Prompt, it.Text, len(ids), len(it.IDs))
			continue
		}
		v, _, err := e.EmbedAudioFeatures(in, feats[it.Clip], n)
		if err != nil {
			t.Fatal(err)
		}
		c := cos64(v, it.Embedding)
		t.Logf("item %d %-5s prompt %-6q text %-5v: A4 cosine %.9f", i, it.Clip, it.Prompt, it.Text != "", c)
		if c < 0.9999 {
			t.Errorf("item %d: A4 end-to-end cosine %.9f, under 0.9999", i, c)
		}
	}
}
