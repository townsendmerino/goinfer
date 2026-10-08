//go:build realckpt

package multimodal

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/audio"
)

// TestGemma4E2BAudioTower is G-S5a of docs/tasks/task-multimodal-support-2026-10.md (S5, registered before this ran):
// Gemma 4 E2B's audio tower and embed_audio, from E2B's own checkpoint (GOINFER_GEMMA4_E2B, default
// ~/models/gemma-4-E2B-unq), through aikit's audio package, against transformers' reference
// (scripts/pin_gemma4_e2b_audio.py; its artifacts in GOINFER_E2B_AUDIO_ARTIFACTS, default
// ~/goinfer-logs/gemma4-e2b-audio) on the three committed clips. Fed HF's log-mel, every stage (the subsampler, each
// block, the tower, the embedder) must be at soft-token cosine >= 0.9999; the first stage under it is named, and a worst
// stage in 0.999-0.9999 is ambiguous (parked). Also recorded, not graded by G-S5a: aikit's own log-mel against HF's, the
// soft-token count, and the end-to-end soft tokens from aikit's log-mel. CPU, float32.
func TestGemma4E2BAudioTower(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	dir := os.Getenv("GOINFER_GEMMA4_E2B")
	if dir == "" {
		dir = filepath.Join(home, "models", "gemma-4-E2B-unq")
	}
	art := os.Getenv("GOINFER_E2B_AUDIO_ARTIFACTS")
	if art == "" {
		art = filepath.Join(home, "goinfer-logs", "gemma4-e2b-audio")
	}
	for _, p := range []string{dir, art} {
		if strings.HasPrefix(p, "/Volumes/") || strings.HasPrefix(p, "/srv/models") {
			t.Fatalf("%s is on the archive (CLAUDE.md)", p)
		}
	}
	raw, err := os.ReadFile(filepath.Join(art, "golden.json"))
	if err != nil {
		t.Skipf("no reference: %v (run scripts/pin_gemma4_e2b_audio.py)", err)
	}
	var g struct {
		Clips []struct {
			Name   string
			TValid int `json:"t_valid"`
			NSoft  int `json:"n_soft"`
		} `json:"clips"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	enc, err := audio.LoadGemma4AudioEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	readF32 := func(name string) []float32 {
		b, err := os.ReadFile(filepath.Join(art, name))
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float32, len(b)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		return out
	}
	worstRow := func(got, want []float32, d int) (float64, float64) {
		w, md := 1.0, 0.0
		for r := range len(want) / d {
			var dot, na, nb float64
			for j := range d {
				x, y := float64(got[r*d+j]), float64(want[r*d+j])
				dot, na, nb = dot+x*y, na+x*x, nb+y*y
				md = math.Max(md, math.Abs(x-y))
			}
			w = math.Min(w, dot/math.Sqrt(na*nb))
		}
		return w, md
	}
	for _, c := range g.Clips {
		wav, err := os.ReadFile(filepath.Join("../testdata/embeddinggemma2-audio", c.Name+".wav"))
		if err != nil {
			t.Fatal(err)
		}
		samples, err := DecodeWAV(wav)
		if err != nil {
			t.Fatal(err)
		}
		mel, T, err := audio.Gemma4Features(samples)
		if err != nil {
			t.Fatal(err)
		}
		hfMel := readF32(c.Name + ".features.f32")
		md := math.Inf(1)
		if T == c.TValid && len(mel) == len(hfMel) {
			md = 0
			for i := range mel {
				md = math.Max(md, math.Abs(float64(mel[i]-hfMel[i])))
			}
		}
		n := audio.Gemma4SoftTokens(T)
		fmt.Fprintf(os.Stderr, "[G-S5a] %-5s %6d samples: log-mel %d frames (HF %d), max|diff| %.3e; %d soft tokens (HF %d)\n",
			c.Name, len(samples), T, c.TValid, md, n, c.NSoft)
		if n != c.NSoft {
			t.Errorf("%s: %d soft tokens, HF %d", c.Name, n, c.NSoft)
		}
		t0 := time.Now()
		out, st, err := enc.ForwardStages(hfMel, c.TValid)
		took := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		type stage struct {
			name, file string
			got        []float32
			d          int
		}
		stages := []stage{{"subsampler", "sub", st.Sub, enc.Cfg.HiddenSize}}
		for L, b := range st.Blocks {
			stages = append(stages, stage{fmt.Sprintf("block %d", L), fmt.Sprintf("block%d", L), b, enc.Cfg.HiddenSize})
		}
		stages = append(stages, stage{"tower", "tower", st.Tower, enc.Cfg.OutputProjDims}, stage{"embedder", "embed", out, enc.TextHiddenSize})
		first, worst, line := "", 1.0, ""
		for _, s := range stages {
			want := readF32(c.Name + "." + s.file + ".f32")
			if len(want) != len(s.got) {
				t.Fatalf("%s %s: %d values, HF %d", c.Name, s.name, len(s.got), len(want))
			}
			w, m := worstRow(s.got, want, s.d)
			worst = math.Min(worst, w)
			line += fmt.Sprintf(" %s %.6f/%.1e", s.file, w, m)
			if w < 0.9999 && first == "" {
				first = s.name
			}
		}
		fmt.Fprintf(os.Stderr, "[G-S5a] %-5s stages (worst row cosine/max|diff|):%s; tower %s (exploratory)\n", c.Name, line, took.Round(time.Millisecond))
		switch {
		case worst < 0.999:
			t.Errorf("%s: G-S5a FAIL, first stage under 0.9999: %s (worst %.9f)", c.Name, first, worst)
		case first != "":
			t.Errorf("%s: G-S5a ambiguous (parked), first stage under 0.9999: %s (worst %.9f)", c.Name, first, worst)
		}
		// End to end from aikit's own log-mel (recorded).
		e2e, err := enc.Forward(mel, T)
		if err != nil {
			t.Fatal(err)
		}
		if want := readF32(c.Name + ".embed.f32"); len(e2e) == len(want) {
			w, _ := worstRow(e2e, want, enc.TextHiddenSize)
			fmt.Fprintf(os.Stderr, "[G-S5a] %-5s end to end from aikit's log-mel: worst soft-token cosine %.9f (recorded)\n", c.Name, w)
		}
	}
}
