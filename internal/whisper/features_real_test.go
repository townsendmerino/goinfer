//go:build realckpt

package whisper

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// G-S14g1 of docs/tasks/task-multimodal-support-2026-10.md: LongestFeatures against transformers 5.15.0's WhisperFeatureExtractor(truncation=False, padding="longest") on three clips
// (scripts/pin_whisper_longest.py). Bars: the same shape, max |diff| <= 1e-4, mean |diff| <= 1e-6. Each planted defect must break the bar.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_WHISPER_LONGREF=<pin output dir> go test -tags realckpt ./internal/whisper/ -run TestLongestFeatures -v
func TestLongestFeatures_matchesExtractor(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	ref := os.Getenv("GOINFER_WHISPER_LONGREF")
	if ref == "" {
		t.Skip("set GOINFER_WHISPER_LONGREF to scripts/pin_whisper_longest.py's output directory")
	}
	rd := func(name string) []float32 {
		b, err := os.ReadFile(filepath.Join(ref, name))
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float32, len(b)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		return out
	}
	var meta struct {
		Mels  int
		Cases map[string]struct{ Samples, Frames, Mask int }
	}
	b, err := os.ReadFile(filepath.Join(ref, "meta.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &meta); err != nil {
		t.Fatal(err)
	}
	check := func(defect int, name string) (maxd, meand float64, shapeOK bool) {
		shapeOK = true
		var sum float64
		var cnt int
		for c, info := range meta.Cases {
			x, want := rd(c+".in.f32"), rd(c+".feats.f32")
			got, T, err := longestFeatures(x, meta.Mels, defect)
			if err != nil {
				t.Fatal(err)
			}
			if T != info.Frames || info.Mask != info.Frames || len(got) != len(want) {
				shapeOK = false
				continue
			}
			for i := range got {
				d := math.Abs(float64(got[i] - want[i]))
				maxd = math.Max(maxd, d)
				sum += d
				cnt++
			}
		}
		if cnt > 0 {
			meand = sum / float64(cnt)
		}
		t.Logf("%-28s shape ok %v, max |diff| %.3e, mean |diff| %.3e", name, shapeOK, maxd, meand)
		return
	}
	maxd, meand, ok := check(featNone, "LongestFeatures")
	if !ok || maxd > 1e-4 || meand > 1e-6 {
		t.Errorf("LongestFeatures differs from the extractor: shape %v, max %.3e, mean %.3e", ok, maxd, meand)
	}
	for name, d := range map[string]int{"padded to 30 s": featPadTo30s, "padded to a multiple of 30 s": featPadToMultiple, "last frame kept": featKeepLast, "no clamp": featNoClamp, "clamp maximum per window": featWindowMax} {
		m, mn, shape := check(d, "defect: "+name)
		if shape && m <= 1e-4 && mn <= 1e-6 {
			t.Errorf("defect %q did not break the bar", name)
		}
	}
}
