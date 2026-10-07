//go:build darwin

package metal

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// G-S3b's Gemma 3 divergence, phase 1 (docs/tasks/task-multimodal-support-2026-10.md, S3): the projected image features
// serve splices into the prompt for testdata/glm_ocr/table.png, from aikit's CPU SigLIP tower and from the Metal one
// (vl_towers.go), both float32, written to GOINFER_G3_FEATS as {"cpu": [...], "metal": [...]} for the decoder-side test
// (decoder/gemma3_tower_sensitivity_real_test.go), which needs the 4B and so may run on another machine. Also logs how
// far apart the two are per soft token, the size the decoder test's random-perturbation control matches.
func TestGemma3TowerDump(t *testing.T) {
	out := os.Getenv("GOINFER_G3_FEATS")
	if out == "" {
		t.Skip("set GOINFER_G3_FEATS to the output file")
	}
	dir := s3Real(t, "gemma-3-4b-it")
	enc, err := vision.LoadEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	r, err := newSiglipVResident(enc)
	if err != nil {
		t.Fatal(err)
	}
	proj, err := multimodal.LoadProjector(dir)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join("../testdata", "glm_ocr/table.png"))
	if err != nil {
		t.Fatal(err)
	}
	pv, err := vision.Preprocess(data, vision.Gemma3())
	if err != nil {
		t.Fatal(err)
	}
	hc, err := enc.Forward(pv.Data) // no resident attached: the CPU tower
	if err != nil {
		t.Fatal(err)
	}
	patches, err := enc.GridPatches(pv.Data)
	if err != nil {
		t.Fatal(err)
	}
	hm, err := r.ForwardPatches(patches)
	if err != nil {
		t.Fatal(err)
	}
	fc, err := proj.Forward(hc)
	if err != nil {
		t.Fatal(err)
	}
	fm, err := proj.Forward(hm)
	if err != nil {
		t.Fatal(err)
	}
	n := proj.MMTokens()
	d := len(fc) / n
	worst, maxRel := 1.0, 0.0
	for i := range n {
		var dot, na, nb, dd float64
		for j := range d {
			a, b := float64(fc[i*d+j]), float64(fm[i*d+j])
			dot, na, nb, dd = dot+a*b, na+a*a, nb+b*b, dd+(a-b)*(a-b)
		}
		worst = math.Min(worst, dot/math.Sqrt(na*nb))
		maxRel = math.Max(maxRel, math.Sqrt(dd/na))
	}
	fmt.Fprintf(os.Stderr, "[g3 dump] %d soft tokens x %d: worst cosine %.9f, worst relative L2 %.3g (Metal tower against CPU)\n", n, d, worst, maxRel)
	b, err := json.Marshal(map[string][]float32{"cpu": fc, "metal": fm})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(out, b, 0o644); err != nil {
		t.Fatal(err)
	}
}
