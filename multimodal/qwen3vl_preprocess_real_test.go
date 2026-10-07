package multimodal

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestQwen3VLPreprocess_againstHF is S10's owed preprocessing record (docs/tasks/task-multimodal-support-2026-10.md):
// goinfer's QwenPreprocess with Qwen3-VL-2B's own preprocessor config (uncapped, so the grids match the processor's)
// against transformers' Qwen2VLImageProcessor pixel values for the four F2a images (written by
// scripts/pin_qwen3vl_tower_real.py on nobara; GOINFER_QWEN3VL_TOWER_ARTIFACTS, default ~/goinfer-logs/qwen3vl-tower).
// It reports the grid and the pixel differences; it fails only on a grid mismatch, which would make the image a
// different number of tokens. Real images are resized, which the bit-exact grid-aligned gate
// (TestQwen3Preprocess_bitExact) never exercises.
func TestQwen3VLPreprocess_againstHF(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	art := os.Getenv("GOINFER_QWEN3VL_TOWER_ARTIFACTS")
	if art == "" {
		art = filepath.Join(home, "goinfer-logs", "qwen3vl-tower")
	}
	raw, err := os.ReadFile(filepath.Join(art, "golden.json"))
	if err != nil {
		t.Skipf("no artifacts: %v", err)
	}
	var g struct {
		Images []struct {
			Image string `json:"image"`
			Name  string `json:"name"`
			Grid  [3]int `json:"grid"`
		} `json:"images"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	pp, err := LoadQwen3PreprocessConfig(filepath.Join(home, "models", "qwen3-vl-2b-instruct"))
	if err != nil {
		t.Skipf("no checkpoint config: %v", err)
	}
	for _, im := range g.Images {
		data, err := os.ReadFile(filepath.Join("../testdata", im.Image))
		if err != nil {
			t.Fatal(err)
		}
		px, grid, err := QwenPreprocess(data, pp)
		if err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(filepath.Join(art, im.Name+".pixels.f32"))
		if err != nil {
			t.Fatal(err)
		}
		hf := make([]float32, len(b)/4)
		for i := range hf {
			hf[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		if grid != im.Grid || len(px) != len(hf) {
			t.Errorf("%s: grid %v (%d values), HF %v (%d values)", im.Image, grid, len(px), im.Grid, len(hf))
			continue
		}
		pd := len(px) / (grid[0] * grid[1] * grid[2])
		var maxd, sumd float64
		worst := 1.0
		for r := range len(px) / pd {
			var dot, na, nb float64
			for j := range pd {
				x, y := float64(px[r*pd+j]), float64(hf[r*pd+j])
				d := math.Abs(x - y)
				maxd, sumd = math.Max(maxd, d), sumd+d
				dot, na, nb = dot+x*y, na+x*x, nb+y*y
			}
			if na > 0 && nb > 0 {
				worst = math.Min(worst, dot/math.Sqrt(na*nb))
			}
		}
		fmt.Fprintf(os.Stderr, "[S10 preprocess] %-30s grid %v: max|diff| %.3g, mean|diff| %.3g, worst patch cosine %.6f (pixel values in [-1, 1])\n",
			im.Image, grid, maxd, sumd/float64(len(px)), worst)
	}
}
