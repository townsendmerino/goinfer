//go:build darwin

package metal

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// G-S3a of docs/tasks/task-multimodal-support-2026-10.md (S3): aikit's Metal SigLIP tower (gpu/visionmetal, imported by
// this package) and Metal Qwen2.5-VL tower (gpu/qwenmetal) against aikit's CPU tower at the same quantization — SigLIP
// int8 both (the device tower needs int8), Qwen2.5-VL f32 both — every output token at cosine >= 0.9999; 0.999-0.9999 is
// ambiguous (parked). The tiny towers always; the real ones (~/models/gemma-3-4b-it, ~/models/qwen25vl-3b-instruct, never
// the archive) under GOINFER_HEAVY_TESTS=1, on the four images F2a uses. Times are exploratory.

func s3Grade(t *testing.T, what string, got, want []float32, width int) {
	t.Helper()
	w := gvWorst(t, got, want, width)
	fmt.Fprintf(os.Stderr, "[S3 G-S3a] %s: worst token cosine %.9f\n", what, w)
	switch {
	case w < 0.999:
		t.Errorf("%s: worst token cosine %.9f under 0.999 (G-S3a FAIL)", what, w)
	case w < 0.9999:
		t.Errorf("%s: worst token cosine %.9f in 0.999-0.9999 (G-S3a ambiguous, parked)", what, w)
	}
}

func s3Real(t *testing.T, name string) string {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1 (loads the real tower)")
	}
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", name)
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is on the archive (CLAUDE.md)", dir)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no %s: %v", dir, err)
	}
	return dir
}

var s3Images = []string{"gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"}

// sigLIPPair loads the tower twice, int8, and attaches the Metal resident to the second.
func sigLIPPair(t *testing.T, dir string) (cpu, dev *vision.Encoder) {
	t.Helper()
	var err error
	if cpu, err = vision.LoadEncoder(dir, true); err != nil {
		t.Fatal(err)
	}
	if dev, err = vision.LoadEncoder(dir, true); err != nil {
		t.Fatal(err)
	}
	if err := dev.EnableResident(); err != nil {
		t.Fatalf("the Metal SigLIP tower did not attach: %v", err)
	}
	t.Cleanup(dev.Close)
	return cpu, dev
}

func qwenVLPair(t *testing.T, dir string) (cpu, dev *vision.QwenVisionEncoder) {
	t.Helper()
	var err error
	if cpu, err = vision.LoadQwenVisionEncoder(dir, false); err != nil {
		t.Fatal(err)
	}
	if dev, err = vision.LoadQwenVisionEncoder(dir, false); err != nil {
		t.Fatal(err)
	}
	if err := dev.EnableResident(); err != nil || !dev.ResidentEnabled() {
		t.Fatalf("the Metal Qwen2.5-VL tower did not attach: %v", err)
	}
	t.Cleanup(dev.Close)
	return cpu, dev
}

func TestS3Towers_tiny(t *testing.T) {
	t.Run("siglip", func(t *testing.T) {
		cpu, dev := sigLIPPair(t, "../testdata/siglip-tiny")
		c := cpu.Cfg
		rng := rand.New(rand.NewSource(9))
		px := make([]float32, c.NumChannels*c.ImageSize*c.ImageSize)
		for i := range px {
			px[i] = float32(rng.NormFloat64())
		}
		want, err := cpu.Forward(px)
		if err != nil {
			t.Fatal(err)
		}
		got, err := dev.Forward(px)
		if err != nil {
			t.Fatal(err)
		}
		s3Grade(t, "siglip-tiny", got, want, c.HiddenSize)
	})
	t.Run("qwen2.5-vl", func(t *testing.T) {
		cpu, dev := qwenVLPair(t, "../testdata/qwen25vl-tiny")
		c := cpu.Cfg
		pd := c.InChans * c.TemporalPatchSize * c.PatchSize * c.PatchSize
		rng := rand.New(rand.NewSource(10))
		for _, g := range [][3]int{{1, 8, 8}, {1, 12, 16}} {
			px := make([]float32, g[0]*g[1]*g[2]*pd)
			for i := range px {
				px[i] = float32(rng.NormFloat64())
			}
			want, err := cpu.Forward(px, [][3]int{g})
			if err != nil {
				t.Fatal(err)
			}
			got, err := dev.Forward(px, [][3]int{g})
			if err != nil {
				t.Fatal(err)
			}
			s3Grade(t, fmt.Sprintf("qwen25vl-tiny %v", g), got, want, c.OutHiddenSize)
		}
	})
}

func TestS3Towers_real(t *testing.T) {
	t.Run("siglip", func(t *testing.T) {
		cpu, dev := sigLIPPair(t, s3Real(t, "gemma-3-4b-it"))
		for _, img := range s3Images {
			data, err := os.ReadFile(filepath.Join("../testdata", img))
			if err != nil {
				t.Fatal(err)
			}
			pv, err := vision.Preprocess(data, vision.Gemma3())
			if err != nil {
				t.Fatal(err)
			}
			t0 := time.Now()
			want, err := cpu.Forward(pv.Data)
			tc := time.Since(t0)
			if err != nil {
				t.Fatal(err)
			}
			t0 = time.Now()
			got, err := dev.Forward(pv.Data)
			tm := time.Since(t0)
			if err != nil {
				t.Fatal(err)
			}
			fmt.Fprintf(os.Stderr, "[S3 real siglip] %-30s tower Metal %s, CPU %s (exploratory)\n", img, tm.Round(time.Millisecond), tc.Round(time.Millisecond))
			s3Grade(t, "gemma-3-4b-it "+img, got, want, cpu.Cfg.HiddenSize)
		}
	})
	t.Run("qwen2.5-vl", func(t *testing.T) {
		dir := s3Real(t, "qwen25vl-3b-instruct")
		cpu, dev := qwenVLPair(t, dir)
		pp, err := multimodal.LoadQwenPreprocessConfig(dir)
		if err != nil {
			t.Fatal(err)
		}
		for _, img := range s3Images {
			data, err := os.ReadFile(filepath.Join("../testdata", img))
			if err != nil {
				t.Fatal(err)
			}
			px, grid, err := multimodal.QwenPreprocess(data, pp)
			if err != nil {
				t.Fatal(err)
			}
			g := [][3]int{grid}
			t0 := time.Now()
			want, err := cpu.Forward(px, g)
			tc := time.Since(t0)
			if err != nil {
				t.Fatal(err)
			}
			t0 = time.Now()
			got, err := dev.Forward(px, g)
			tm := time.Since(t0)
			if err != nil {
				t.Fatal(err)
			}
			fmt.Fprintf(os.Stderr, "[S3 real qwen2.5-vl] %-30s grid %v: tower Metal %s, CPU %s (exploratory)\n", img, grid, tm.Round(time.Millisecond), tc.Round(time.Millisecond))
			s3Grade(t, "qwen25vl-3b "+img, got, want, cpu.Cfg.OutHiddenSize)
		}
	})
}
