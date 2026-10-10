//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// G-S4q of docs/tasks/task-multimodal-support-2026-10.md (S4), the CUDA twin of metal/s3_towers_test.go (G-S3a). It grades
// whatever EnableResident attaches, which is goinfer's own Qwen2.5-VL tower (qwen25_vision.go; aikit's gpu/qwencuda is no
// longer imported), against aikit's CPU tower, f32 on both sides, every output token at cosine >= 0.9999; 0.999-0.9999 is
// ambiguous (parked). aikit's CUDA SigLIP tower (gpu/visioncuda) is read in cuda/s4siglip, a package of its own, because
// inside package cuda goinfer's own SigLIP tower (cuda/vision_register.go) registers last and wins the global hook; that
// read is information for the aikit defect report, not a gate. The tiny towers always; the real ones (~/models, never the
// archive) under GOINFER_HEAVY_TESTS=1, on the four images F2a uses. Times are exploratory.

// s4Worst is the lowest per-token cosine between got and want, both [tokens x width] row-major (the grading helper of metal/grid_vision_test.go gvWorst).
func s4Worst(t *testing.T, got, want []float32, width int) float64 {
	t.Helper()
	if len(got) != len(want) || len(got)%width != 0 {
		t.Fatalf("%d values against %d (width %d)", len(got), len(want), width)
	}
	worst := 1.0
	for r := 0; r < len(got)/width; r++ {
		var dot, na, nb float64
		for j := range width {
			a, b := float64(got[r*width+j]), float64(want[r*width+j])
			dot, na, nb = dot+a*b, na+a*a, nb+b*b
		}
		c := dot / math.Sqrt(na*nb)
		if math.IsNaN(c) {
			c = -1
		}
		worst = math.Min(worst, c)
	}
	return worst
}

func s4Grade(t *testing.T, what string, got, want []float32, width int, gate bool) {
	t.Helper()
	w := s4Worst(t, got, want, width)
	fmt.Fprintf(os.Stderr, "[S4 G-S4q] %s: worst token cosine %.9f\n", what, w)
	if !gate {
		return
	}
	switch {
	case w < 0.999:
		t.Errorf("%s: worst token cosine %.9f under 0.999 (G-S4q FAIL)", what, w)
	case w < 0.9999:
		t.Errorf("%s: worst token cosine %.9f in 0.999-0.9999 (G-S4q ambiguous, parked)", what, w)
	}
}

func s4Real(t *testing.T, name string) string {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1 (loads the real tower)")
	}
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", name)
	if strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is on the archive (CLAUDE.md)", dir)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no %s: %v", dir, err)
	}
	return dir
}

var s4Images = []string{"gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"}

func s4QwenPair(t *testing.T, dir string) (cpu, dev *vision.QwenVisionEncoder) {
	t.Helper()
	newTestTower(t, 64) // skips without a CUDA device (CI's cuda job has none); with one, a failed attach below is a real failure
	var err error
	if cpu, err = vision.LoadQwenVisionEncoder(dir, false); err != nil {
		t.Fatal(err)
	}
	if dev, err = vision.LoadQwenVisionEncoder(dir, false); err != nil {
		t.Fatal(err)
	}
	if err := dev.EnableResident(); err != nil || !dev.ResidentEnabled() {
		t.Fatalf("the CUDA Qwen2.5-VL tower did not attach: %v", err)
	}
	t.Cleanup(dev.Close)
	return cpu, dev
}

func TestS4Towers_tiny(t *testing.T) {
	t.Run("qwen2.5-vl", func(t *testing.T) {
		cpu, dev := s4QwenPair(t, "../testdata/qwen25vl-tiny")
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
			s4Grade(t, fmt.Sprintf("qwen25vl-tiny %v", g), got, want, c.OutHiddenSize, true)
		}
	})
}

func TestS4Towers_real(t *testing.T) {
	t.Run("qwen2.5-vl", func(t *testing.T) {
		dir := s4Real(t, "qwen25vl-3b-instruct")
		cpu, dev := s4QwenPair(t, dir)
		pp, err := multimodal.LoadQwenPreprocessConfig(dir)
		if err != nil {
			t.Fatal(err)
		}
		for _, img := range s4Images {
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
			fmt.Fprintf(os.Stderr, "[S4 real qwen2.5-vl] %-30s grid %v: tower CUDA %s, CPU %s (exploratory)\n", img, grid, tm.Round(time.Millisecond), tc.Round(time.Millisecond))
			s4Grade(t, "qwen25vl-3b "+img, got, want, cpu.Cfg.OutHiddenSize, true)
		}
	})
}
