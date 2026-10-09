package multimodal

import (
	"encoding/json"
	"math"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/audio"
)

// G-S14d1 of docs/tasks/task-multimodal-support-2026-10.md (registered before this code): aikit's Whisper encoder against transformers' WhisperEncoder on a tiny random-weight Whisper
// (scripts/pin_whisper_tiny.py): per-frame cosine >= 0.99999 AND max |diff| <= 2e-05, and every planted defect red against both.

const (
	whisperEncCosBar = 0.99999
	// whisperEncAbsBar: registered as 2e-05 (the bar G-S14b2 used, on 104-position attention windows) and amended to 5e-04 before any real-checkpoint reading, with the measurement: on this
	// 1500-position input transformers' OWN float32 differs from a float64 evaluation of the same weights by 1.9e-05 (6.5 s clip) and 1.16e-04 (25 s clip), and the port differs from the
	// pinned reference by 2.2e-05 and 1.29e-04: the same size, because float32 accumulation over 1500 keys is where the error is. 5e-04 is about 4x that noise and 60x below the weakest planted defect.
	whisperEncAbsBar = 5e-4
)

func loadTinyWhisperEncoder(t *testing.T) (*audio.WhisperEncoder, map[string][]byte, []string) {
	t.Helper()
	dir := filepath.Join("..", "testdata", "whisper-tiny-rand")
	files := readZip(t, filepath.Join(dir, "golden.zip"))
	var meta struct{ Cases []struct{ Name string } }
	if err := json.Unmarshal(files["meta.json"], &meta); err != nil {
		t.Fatal(err)
	}
	enc, err := audio.LoadWhisperEncoder(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	var names []string
	for _, c := range meta.Cases {
		names = append(names, c.Name)
	}
	return enc, files, names
}

func TestWhisperEncoder_tinyMatchesTransformers(t *testing.T) {
	enc, files, names := loadTinyWhisperEncoder(t)
	if len(names) != 2 {
		t.Fatalf("%d cases, want 2", len(names))
	}
	for _, name := range names {
		got, err := enc.Forward(f32s(files[name+".feats.f32"]))
		if err != nil {
			t.Fatal(err)
		}
		want := f32s(files[name+".enc.f32"])
		cos, ab := cosineRows(got, want, audio.WhisperFrames/2)
		t.Logf("%-8s worst per-frame cosine %.9f, max |diff| %.2e", name, cos, ab)
		if cos < whisperEncCosBar || ab > whisperEncAbsBar {
			t.Errorf("%s: cosine %.9f (bar %.5f), max |diff| %.2e (bar %.0e)", name, cos, whisperEncCosBar, ab, whisperEncAbsBar)
		}
		// the Go front end instead of the reference's features: recorded
		f, _, err := audio.WhisperFeatures(f32s(files[name+".samples.f32"]), 80)
		if err != nil {
			t.Fatal(err)
		}
		got2, err := enc.Forward(f)
		if err != nil {
			t.Fatal(err)
		}
		c2, a2 := cosineRows(got2, want, audio.WhisperFrames/2)
		t.Logf("%-8s with the Go front end: worst per-frame cosine %.9f, max |diff| %.2e", name, c2, a2)
	}
}

func TestWhisperEncoder_tinyPlantedDefectsAreRed(t *testing.T) {
	enc, files, names := loadTinyWhisperEncoder(t)
	defer enc.SetDefectForTest(0)
	for d := 1; d <= audio.WhisperEncDefectCountForTest; d++ {
		enc.SetDefectForTest(d)
		worstCos, worstAbs := 1.0, 0.0
		for _, name := range names {
			got, err := enc.Forward(f32s(files[name+".feats.f32"]))
			if err != nil {
				t.Fatal(err)
			}
			cos, ab := cosineRows(got, f32s(files[name+".enc.f32"]), audio.WhisperFrames/2)
			worstCos, worstAbs = math.Min(worstCos, cos), math.Max(worstAbs, ab)
		}
		red := worstCos < whisperEncCosBar || worstAbs > whisperEncAbsBar
		t.Logf("planted defect %d: worst cosine %.8f, max |diff| %.2e, red=%v", d, worstCos, worstAbs, red)
		if !red {
			t.Errorf("planted defect %d stayed green (cosine %.8f, max |diff| %.2e)", d, worstCos, worstAbs)
		}
	}
}

func TestWhisperEncoder_refusesOtherLengths(t *testing.T) {
	enc, _, _ := loadTinyWhisperEncoder(t)
	if _, err := enc.Forward(make([]float32, 80*2999)); err == nil {
		t.Error("an input of 2999 frames was accepted: the encoder is built for exactly 30 s")
	}
}
