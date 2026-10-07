//go:build darwin

package metal

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/embeddinggemma2"
)

// TestG4AMetal_realParity is Phase A's gates AM1 and AM2 (docs/tasks/task-embeddinggemma2.md) on the real
// google/embeddinggemma-2 (GOINFER_EG2_DIR, default ~/models/embeddinggemma-2, never the archive), with the encoder and
// the audio tower's blocks on Metal, on the committed clips (testdata/embeddinggemma2-audio):
//
//	AM1: the Metal tower's soft tokens against the CPU tower's, cosine >= 0.9999 on every one, all three clips.
//	AM2: end to end (aikit's log-mel, the Metal tower, the Metal encoder), every one of the golden's 9 cases at
//	     cosine >= 0.9999 with sentence-transformers.
//
// It also times the Metal and the CPU tower on each clip, alternating, in this process: exploratory, not a result.
//
//	GOINFER_EG2_REAL=1 go test -count=1 -run '^TestG4AMetal_realParity$' -v ./metal/
func TestG4AMetal_realParity(t *testing.T) {
	if os.Getenv("GOINFER_EG2_REAL") != "1" {
		t.Skip("set GOINFER_EG2_REAL=1 (loads the real checkpoint)")
	}
	home, _ := os.UserHomeDir()
	dir := os.Getenv("GOINFER_EG2_DIR")
	if dir == "" {
		dir = filepath.Join(home, "models", "embeddinggemma-2")
	}
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", dir)
	}
	raw, err := os.ReadFile("../testdata/embeddinggemma2-audio/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Clips []struct{ Name string } `json:"clips"`
		Items []struct {
			Clip, Prompt, Text string
			IDs                []int     `json:"ids"`
			Embedding          []float64 `json:"embedding"`
		} `json:"items"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	e, err := embeddinggemma2.LoadEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := e.UseAccelerator("metal"); err != nil {
		t.Fatal(err)
	}
	if err := e.EnableAudio(); err != nil {
		t.Fatal(err)
	}
	if e.AudioDevice() != "metal" {
		t.Fatalf("the audio tower runs on %q, want metal", e.AudioDevice())
	}
	H := e.Model().Config().Hidden
	feats := map[string][]float32{}
	for _, c := range g.Clips {
		wav, err := os.ReadFile(filepath.Join("../testdata/embeddinggemma2-audio", c.Name+".wav"))
		if err != nil {
			t.Fatal(err)
		}
		s, err := embeddinggemma2.DecodeWAV(wav)
		if err != nil {
			t.Fatal(err)
		}
		mel, T, err := embeddinggemma2.AudioLogMel(s)
		if err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		mf, n, err := e.AudioFeaturesFrom(mel, T)
		tm := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		t0 = time.Now()
		cf, _, err := e.AudioFeaturesCPU(mel, T)
		tc := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		w := g4vWorst(mf, cf, H)
		t.Logf("%-5s %3d soft tokens: AM1 worst soft-token cosine %.9f against the CPU tower; tower Metal %s, CPU %s (exploratory)", c.Name, n, w, tm.Round(time.Millisecond), tc.Round(time.Millisecond))
		if w < 0.9999 {
			t.Errorf("%s: AM1 worst soft-token cosine %.9f, under 0.9999", c.Name, w)
		}
		feats[c.Name] = mf
	}
	for i, it := range g.Items {
		in := embeddinggemma2.AudioInput{Text: it.Text, Prompt: it.Prompt}
		n := len(feats[it.Clip]) / H
		ids, _, err := e.TokenizeAudio(in, n)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprint(ids) != fmt.Sprint(it.IDs) {
			t.Errorf("item %d: ids differ from the reference", i)
			continue
		}
		v, _, err := e.EmbedAudioFeatures(in, feats[it.Clip], n)
		if err != nil {
			t.Fatal(err)
		}
		var dot, na, nb float64
		for j, r := range it.Embedding {
			dot += float64(v[j]) * r
			na += float64(v[j]) * float64(v[j])
			nb += r * r
		}
		c := dot / math.Sqrt(na*nb)
		t.Logf("item %d %-5s prompt %-6q text %-5v: AM2 cosine %.9f", i, it.Clip, it.Prompt, it.Text != "", c)
		if c < 0.9999 {
			t.Errorf("item %d: AM2 end-to-end cosine %.9f, under 0.9999", i, c)
		}
	}
}
