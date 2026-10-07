//go:build cuda && goinfer_testhooks

// Package s4siglip reads aikit's CUDA SigLIP tower (gpu/visioncuda) at real size, alone. It is its own package because both aikit's visioncuda and goinfer's
// own cuda.VisionEncoder (cuda/vision_register.go) register through the one global vision.RegisterResident hook and the LAST registration wins: inside package
// cuda the goinfer tower wins, so a test there measures goinfer's tower, not aikit's (found 2026-10-07; the attached type was *cuda.VisionEncoder).
// This package imports nothing from goinfer's cuda, so only visioncuda registers. Information for the aikit defect report, not a gate (G-S4q).
package s4siglip

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	_ "github.com/townsendmerino/aikit/gpu/visioncuda"
	"github.com/townsendmerino/aikit/vision"
)

func worst(t *testing.T, got, want []float32, width int) (w float64, below int) {
	t.Helper()
	if len(got) != len(want) || len(got)%width != 0 {
		t.Fatalf("%d values against %d (width %d)", len(got), len(want), width)
	}
	w = 1.0
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
		w = math.Min(w, c)
		if c < 0.9999 {
			below++
		}
	}
	return w, below
}

var images = []string{"gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"}

func TestVisionCUDA_real(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1 (loads the real tower)")
	}
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "gemma-3-4b-it")
	if strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is on the archive (CLAUDE.md)", dir)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no %s: %v", dir, err)
	}
	cpu, err := vision.LoadEncoder(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	dev, err := vision.LoadEncoder(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	if err := dev.EnableResident(); err != nil {
		t.Fatalf("aikit's CUDA SigLIP tower did not attach: %v", err)
	}
	defer dev.Close()
	typ := reflect.ValueOf(dev).Elem().FieldByName("resident").Elem().Type().String()
	fmt.Fprintf(os.Stderr, "[S4 visioncuda] resident encoder attached: %s\n", typ)
	if !strings.Contains(typ, "visioncuda") {
		t.Fatalf("the attached tower is %s, not aikit's visioncuda", typ)
	}
	for _, img := range images {
		data, err := os.ReadFile(filepath.Join("../../testdata", img))
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
		w, below := worst(t, got, want, cpu.Cfg.HiddenSize)
		fmt.Fprintf(os.Stderr, "[S4 visioncuda] %-30s tower CUDA %s, CPU %s (exploratory); worst token cosine %.9f; tokens under 0.9999: %d of %d\n",
			img, tm.Round(time.Millisecond), tc.Round(time.Millisecond), w, below, len(want)/cpu.Cfg.HiddenSize)
	}
}
