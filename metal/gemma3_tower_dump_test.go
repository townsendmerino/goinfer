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
// serve splices into the prompt for one image (GOINFER_G3_IMAGE under testdata/, default glm_ocr/table.png), from aikit's
// CPU SigLIP tower, the Metal f16 tower and the Metal int8 tower (vl_towers.go), written to GOINFER_G3_FEATS as
// {"cpu": [...], "metal": [...], "metal_int8": [...]} for the decoder-side test
// (decoder/gemma3_tower_sensitivity_real_test.go), which needs the 4B and so may run on another machine. Also logs how
// far each Metal tower is from the CPU one per soft token, the size the decoder test's noise control matches.
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
	img := os.Getenv("GOINFER_G3_IMAGE")
	if img == "" {
		img = "glm_ocr/table.png"
	}
	data, err := os.ReadFile(filepath.Join("../testdata", img))
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
	// The int8 Metal tower (tower_gemm_w8) from the same encoder, for G-S18g's follow-up.
	siglipForceInt8 = true
	r8, err := newSiglipVResident(enc)
	siglipForceInt8 = false
	if err != nil {
		t.Fatal(err)
	}
	if !r8.a.int8W {
		t.Fatal("the int8 tower was not built")
	}
	h8, err := r8.ForwardPatches(patches)
	if err != nil {
		t.Fatal(err)
	}
	f8, err := proj.Forward(h8)
	if err != nil {
		t.Fatal(err)
	}
	n := proj.MMTokens()
	d := len(fc) / n
	for _, arm := range []struct {
		name string
		f    []float32
	}{{"Metal f16 tower", fm}, {"Metal int8 tower", f8}} {
		worst, maxRel, sumRel := 1.0, 0.0, 0.0
		for i := range n {
			var dot, na, nb, dd float64
			for j := range d {
				a, b := float64(fc[i*d+j]), float64(arm.f[i*d+j])
				dot, na, nb, dd = dot+a*b, na+a*a, nb+b*b, dd+(a-b)*(a-b)
			}
			worst = math.Min(worst, dot/math.Sqrt(na*nb))
			maxRel = math.Max(maxRel, math.Sqrt(dd/na))
			sumRel += math.Sqrt(dd / na)
		}
		fmt.Fprintf(os.Stderr, "[g3 dump] %s, %d soft tokens x %d: worst cosine %.9f, relative L2 mean %.3g worst %.3g (%s against CPU)\n", img, n, d, worst, sumRel/float64(n), maxRel, arm.name)
	}
	b, err := json.Marshal(map[string][]float32{"cpu": fc, "metal": fm, "metal_int8": f8})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(out, b, 0o644); err != nil {
		t.Fatal(err)
	}
}
