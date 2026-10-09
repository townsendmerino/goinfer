package multimodal

import (
	"archive/zip"
	"encoding/json"
	"io"
	"math"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/audio"
)

// G-S14b1 and G-S14b2 of docs/tasks/task-multimodal-support-2026-10.md (registered before this code): aikit's Qwen3-ASR front end against transformers 5.15.0's
// Qwen3ASRFeatureExtractor, and its audio encoder and projector against the transformers modules on a tiny random-weight checkpoint in the real layout. Goldens: scripts/pin_qwen3asr_features.py,
// scripts/pin_qwen3asr_tiny.py.

const (
	qwenFeatMaxBar, qwenFeatMeanBar = 1e-4, 1e-6
	qwenEncCosBar                   = 0.99999
	// qwenEncAbsBar is a second criterion added to the registered cosine one (amendment A1 of G-S14b, 2026-10-08): two GELUs that differ by 1e-3 leave the projected cosine at 0.9999998, above
	// the cosine bar, so cosine alone cannot see the tanh-for-erf defect. The projected outputs have rms 1.3-1.9 and the port matches the reference to ~6e-06.
	qwenEncAbsBar = 2e-5
)

func readZip(t *testing.T, path string) map[string][]byte {
	t.Helper()
	zr, err := zip.OpenReader(path)
	if err != nil {
		t.Skipf("no golden (%v)", err)
	}
	defer zr.Close()
	files := map[string][]byte{}
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
		files[f.Name] = b
	}
	return files
}

type qwenFeatCase struct {
	Name              string
	Samples, T, Valid int
	Stored            bool
}

func qwenFeatCases(t *testing.T) (map[string][]byte, []qwenFeatCase, map[string][]float32) {
	files := readZip(t, "../testdata/qwen3asr_features_golden.zip")
	var meta struct{ Cases []qwenFeatCase }
	if err := json.Unmarshal(files["meta.json"], &meta); err != nil {
		t.Fatal(err)
	}
	inputs := map[string][]float32{}
	for _, c := range meta.Cases {
		if c.Stored {
			inputs[c.Name] = f32s(files[c.Name+".in.f32"])
		}
	}
	// the one recipe: 30.5 s, the LibriSpeech clip tiled, past the 30 s a Whisper extractor stops at
	libri := inputs["libri"]
	long := make([]float32, 488000)
	for i := range long {
		long[i] = libri[i%len(libri)]
	}
	inputs["long30s5"] = long
	return files, meta.Cases, inputs
}

func TestQwenASRFeatures_matchTransformers(t *testing.T) {
	files, cases, inputs := qwenFeatCases(t)
	if len(cases) != 7 {
		t.Fatalf("golden has %d cases, want 7", len(cases))
	}
	for _, c := range cases {
		x := inputs[c.Name]
		if len(x) != c.Samples {
			t.Fatalf("%s: %d input samples, golden says %d", c.Name, len(x), c.Samples)
		}
		got, err := audio.QwenASRFeatures(x)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		want, wmask := f32s(files[c.Name+".feats.f32"]), files[c.Name+".mask.u8"]
		if got.T != c.T || got.Valid != c.Valid || len(got.Data) != len(want) {
			t.Errorf("%s: T %d valid %d (%d values), the extractor has T %d valid %d (%d values)", c.Name, got.T, got.Valid, len(got.Data), c.T, c.Valid, len(want))
			continue
		}
		for i := range wmask {
			if got.Mask[i] != wmask[i] {
				t.Errorf("%s: mask differs at frame %d", c.Name, i)
				break
			}
		}
		mx, mean := diffStats(got.Data, want)
		t.Logf("%-10s %6d samples: T %4d valid %4d, max |diff| %.3e, mean %.3e", c.Name, c.Samples, got.T, got.Valid, mx, mean)
		if mx > qwenFeatMaxBar || mean > qwenFeatMeanBar {
			t.Errorf("%s: max %.3e (bar %.0e), mean %.3e (bar %.0e)", c.Name, mx, qwenFeatMaxBar, mean, qwenFeatMeanBar)
		}
	}
}

func TestQwenASRFeatures_plantedDefectsAreRed(t *testing.T) {
	files, cases, inputs := qwenFeatCases(t)
	for d := 1; d <= audio.QwenASRDefectCountForTest; d++ {
		red, how := false, ""
		for _, c := range cases {
			got, err := audio.QwenASRFeaturesDefectForTest(inputs[c.Name], d)
			if err != nil {
				continue
			}
			want, wmask := f32s(files[c.Name+".feats.f32"]), files[c.Name+".mask.u8"]
			switch {
			case got.T != c.T || got.Valid != c.Valid:
				red, how = true, c.Name+": shape or valid count"
			default:
				for i := range wmask {
					if got.Mask[i] != wmask[i] {
						red, how = true, c.Name+": mask"
						break
					}
				}
				if mx, _ := diffStats(got.Data, want); mx > qwenFeatMaxBar {
					red, how = true, c.Name+": values"
				}
			}
			if red {
				break
			}
		}
		t.Logf("planted defect %d: red=%v (%s)", d, red, how)
		if !red {
			t.Errorf("planted defect %d stayed green on every case", d)
		}
	}
}

func cosineRows(a, b []float32, rows int) (minCos, maxAbs float64) {
	d := len(a) / rows
	minCos = 1
	for r := range rows {
		var dot, na, nb float64
		for j := range d {
			x, y := float64(a[r*d+j]), float64(b[r*d+j])
			dot += x * y
			na += x * x
			nb += y * y
			maxAbs = math.Max(maxAbs, math.Abs(x-y))
		}
		minCos = math.Min(minCos, dot/math.Sqrt(na*nb))
	}
	return minCos, maxAbs
}

type qwenEncCase struct {
	Name         string
	T, Valid     int
	Tokens       int
	DModel, OutD int `json:"-"`
}

func loadTinyEncoder(t *testing.T) (*audio.QwenASREncoder, map[string][]byte, []qwenEncCase) {
	t.Helper()
	dir := filepath.Join("..", "testdata", "qwen3asr-tiny")
	files := readZip(t, filepath.Join(dir, "golden.zip"))
	var meta struct {
		Cases []struct {
			Name             string
			T, Valid, Tokens int
			DModel           int `json:"d_model"`
			OutDim           int `json:"out_dim"`
		}
	}
	if err := json.Unmarshal(files["meta.json"], &meta); err != nil {
		t.Fatal(err)
	}
	enc, err := audio.LoadQwenASREncoder(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	var cs []qwenEncCase
	for _, c := range meta.Cases {
		cs = append(cs, qwenEncCase{Name: c.Name, T: c.T, Valid: c.Valid, Tokens: c.Tokens, DModel: c.DModel, OutD: c.OutDim})
	}
	return enc, files, cs
}

func TestQwenASREncoder_tinyMatchesTransformers(t *testing.T) {
	enc, files, cases := loadTinyEncoder(t)
	if len(cases) != 3 {
		t.Fatalf("%d cases, want 3", len(cases))
	}
	for _, c := range cases {
		feats := f32s(files[c.Name+".feats.f32"])
		if audio.QwenASRTokenCount(c.Valid) != c.Tokens {
			t.Errorf("%s: token count %d for %d valid frames, the reference has %d", c.Name, audio.QwenASRTokenCount(c.Valid), c.Valid, c.Tokens)
		}
		tower, n, err := enc.Tower(feats, c.T, c.Valid)
		if err != nil || n != c.Tokens {
			t.Fatalf("%s: tower tokens %d (want %d), err %v", c.Name, n, c.Tokens, err)
		}
		out, n2, err := enc.Forward(feats, c.T, c.Valid)
		if err != nil || n2 != c.Tokens {
			t.Fatalf("%s: forward tokens %d, err %v", c.Name, n2, err)
		}
		cosT, absT := cosineRows(tower, f32s(files[c.Name+".tower.f32"]), c.Tokens)
		cosO, absO := cosineRows(out, f32s(files[c.Name+".out.f32"]), c.Tokens)
		t.Logf("%-8s %2d tokens: tower worst cosine %.8f (max |diff| %.2e), projected %.8f (%.2e)", c.Name, c.Tokens, cosT, absT, cosO, absO)
		if cosT < qwenEncCosBar || cosO < qwenEncCosBar || absT > qwenEncAbsBar || absO > qwenEncAbsBar {
			t.Errorf("%s: worst per-token cosine %.8f (tower) / %.8f (projected), bar %.5f; max |diff| %.2e / %.2e, bar %.0e", c.Name, cosT, cosO, qwenEncCosBar, absT, absO, qwenEncAbsBar)
		}
	}
}

func TestQwenASREncoder_tinyPlantedDefectsAreRed(t *testing.T) {
	enc, files, cases := loadTinyEncoder(t)
	defer enc.SetDefectForTest(0)
	for d := 1; d <= audio.QwenASREncDefectCountForTest; d++ {
		enc.SetDefectForTest(d)
		worstCos, worstAbs := 1.0, 0.0
		how := ""
		for _, c := range cases {
			out, n, err := enc.Forward(f32s(files[c.Name+".feats.f32"]), c.T, c.Valid)
			if err != nil || n != c.Tokens {
				worstCos, how = -1, c.Name+": token count or error"
				break
			}
			cos, ab := cosineRows(out, f32s(files[c.Name+".out.f32"]), c.Tokens)
			worstCos, worstAbs = math.Min(worstCos, cos), math.Max(worstAbs, ab)
		}
		red := worstCos < qwenEncCosBar || worstAbs > qwenEncAbsBar
		t.Logf("planted defect %d: worst projected cosine %.8f, max |diff| %.2e, red=%v %s", d, worstCos, worstAbs, red, how)
		if !red {
			t.Errorf("planted defect %d left the projected cosine at %.8f and the max difference at %.2e: the gate cannot see it", d, worstCos, worstAbs)
		}
	}
}
