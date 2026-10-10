package multimodal

import (
	"archive/zip"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"testing"

	"github.com/townsendmerino/aikit/audio"
)

// G-S14a (docs/tasks/task-multimodal-support-2026-10.md): aikit's Whisper front end against transformers 5.15.0's WhisperFeatureExtractor, both of its
// paths. Bars per case: the shape, the largest absolute difference <= 5e-05 against each reference path, the mean <= 1e-06, and the valid-frame count equal to the extractor's mask. The
// reference paths differ from each other by up to 3.1e-05 on these inputs (the golden's meta.json), so 5e-05 is about 1.6 times their own spread. Six planted defects must each put some
// case over the bar. The golden is scripts/pin_whisper_features.py's.

const whisperMaxBar, whisperMeanBar = 5e-5, 1e-6

type whisperGolden struct {
	files map[string][]byte
	meta  struct {
		Transformers string
		Cases        []struct {
			Name          string
			Mels, Samples int
			NPvsTorchMax  float64 `json:"np_vs_torch_max"`
		}
		Counts []struct {
			Samples     int
			ValidFrames int `json:"valid_frames"`
		}
	}
}

func loadWhisperGolden(t *testing.T) *whisperGolden {
	t.Helper()
	zr, err := zip.OpenReader("../testdata/whisper_features_golden.zip")
	if err != nil {
		t.Skipf("no golden (%v): run scripts/pin_whisper_features.py", err)
	}
	t.Cleanup(func() { _ = zr.Close() })
	g := &whisperGolden{files: map[string][]byte{}}
	for _, f := range zr.File {
		rc, err := f.Open()
		if err != nil {
			t.Fatal(err)
		}
		b, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			t.Fatal(err)
		}
		g.files[f.Name] = b
	}
	if err := json.Unmarshal(g.files["meta.json"], &g.meta); err != nil {
		t.Fatal(err)
	}
	return g
}

func f32s(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func diffStats(a, b []float32) (maxAbs, mean float64) {
	for i := range a {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		maxAbs = math.Max(maxAbs, d)
		mean += d
	}
	return maxAbs, mean / float64(len(a))
}

func TestWhisperFeatures_matchTransformers(t *testing.T) {
	g := loadWhisperGolden(t)
	if len(g.meta.Cases) != 7 {
		t.Fatalf("golden has %d cases, want 7", len(g.meta.Cases))
	}
	compared := 0
	for _, c := range g.meta.Cases {
		x := f32s(g.files[c.Name+".in.f32"])
		got, _, err := audio.WhisperFeatures(x, c.Mels)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		for _, ref := range []string{"torch", "np"} {
			b, ok := g.files[c.Name+"."+ref+".f32"]
			if !ok {
				continue
			}
			want := f32s(b)
			if len(got) != len(want) {
				t.Fatalf("%s: %d values, the %s path has %d", c.Name, len(got), ref, len(want))
			}
			mx, mean := diffStats(got, want)
			compared++
			t.Logf("%-10s %3d mels, %6d samples against the %-5s path: max |diff| %.3e, mean %.3e (the reference paths differ by %.3e)", c.Name, c.Mels, c.Samples, ref, mx, mean, c.NPvsTorchMax)
			if mx > whisperMaxBar || mean > whisperMeanBar {
				t.Errorf("%s against the %s path: max %.3e (bar %.0e), mean %.3e (bar %.0e)", c.Name, ref, mx, whisperMaxBar, mean, whisperMeanBar)
			}
		}
	}
	if compared != 7+3 {
		t.Errorf("%d comparisons ran, want 10 (7 against the torch path, 3 against the NumPy path)", compared)
	}
	for _, c := range g.meta.Counts {
		if got := audio.WhisperValidFrames(c.Samples); got != c.ValidFrames {
			t.Errorf("%d samples: %d valid frames, the extractor's mask says %d", c.Samples, got, c.ValidFrames)
		}
	}
}

// Each planted defect must put some case over the bar, against the NumPy path (the closer reference); the largest excess is logged so the margin is on record.
func TestWhisperFeatures_plantedDefectsAreRed(t *testing.T) {
	g := loadWhisperGolden(t)
	for d := 1; d <= audio.WhisperDefectCountForTest; d++ {
		var worst float64
		for _, name := range []string{"libri", "libri80", "synth10"} {
			mels := 128
			if name == "libri80" {
				mels = 80
			}
			got, _, err := audio.WhisperFeaturesDefectForTest(f32s(g.files[name+".in.f32"]), mels, d)
			if err != nil {
				t.Fatal(err)
			}
			mx, _ := diffStats(got, f32s(g.files[name+".np.f32"]))
			worst = math.Max(worst, mx)
		}
		t.Logf("planted defect %d: worst max |diff| %.3e against the bar %.0e", d, worst, whisperMaxBar)
		if worst <= whisperMaxBar {
			t.Errorf("planted defect %d stayed under the bar (%.3e): the bar is too loose or the seam is dead", d, worst)
		}
	}
}
