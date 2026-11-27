//go:build gpu

package towers

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/kernels"
	"github.com/townsendmerino/goinfer/internal/vision"
)

// TestVisionEncoder_parity: the resident GPU SigLIP forward vs the CPU int8
// (W8A8) encoder on the tiny checkpoint. Both W8A8, so they should be close.
func TestVisionEncoder_parity(t *testing.T) {
	c, err := kernels.New()
	if err != nil {
		t.Skipf("no gpu: %v", err)
	}
	defer c.Close()
	const ckpt = "../../../../testdata/siglip-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skip("no siglip-tiny")
	}
	raw, err := os.ReadFile("../../../../testdata/siglip_vision_golden.json")
	if err != nil {
		t.Skip("no golden")
	}
	var g struct {
		PixelValues     []float32 `json:"pixel_values"`
		LastHiddenState []float32 `json:"last_hidden_state"`
	}
	json.Unmarshal(raw, &g)

	enc, err := vision.LoadEncoder(ckpt, true)
	if err != nil {
		t.Fatal(err)
	}
	cpuOut, err := enc.Forward(g.PixelValues)
	if err != nil {
		t.Fatal(err)
	}
	w, err := enc.GPUWeights()
	if err != nil {
		t.Fatal(err)
	}
	ve, err := NewVisionEncoder(c, w)
	if err != nil {
		t.Fatal(err)
	}
	defer ve.Close()
	patches, err := enc.GridPatches(g.PixelValues)
	if err != nil {
		t.Fatal(err)
	}
	gpuOut, err := ve.ForwardPatches(patches)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("resident GPU vs CPU-W8A8 cosine=%.6f | GPU vs HF golden cosine=%.6f", cosG(gpuOut, cpuOut), cosG(gpuOut, g.LastHiddenState))
	if v := cosG(gpuOut, cpuOut); v < 0.99 {
		t.Errorf("resident GPU vs CPU cosine %.6f < 0.99", v)
	}
}

func cosG(a, b []float32) float64 {
	var d, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		d += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return d / (math.Sqrt(na) * math.Sqrt(nb))
}

// TestVisionEnableResident_parity exercises the full serve seam: EnableResident
// (gpu init → vision.RegisterResident → residentFactory) then Encoder.Forward
// delegating to the device. Compares vs a separate CPU encoder.
func TestVisionEnableResident_parity(t *testing.T) {
	if _, err := kernels.New(); err != nil {
		t.Skipf("no gpu: %v", err)
	}
	const ckpt = "../../../../testdata/siglip-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skip("no siglip-tiny")
	}
	raw, err := os.ReadFile("../../../../testdata/siglip_vision_golden.json")
	if err != nil {
		t.Skip("no golden")
	}
	var g struct {
		PixelValues []float32 `json:"pixel_values"`
	}
	json.Unmarshal(raw, &g)

	cpu, err := vision.LoadEncoder(ckpt, true)
	if err != nil {
		t.Fatal(err)
	}
	want, err := cpu.Forward(g.PixelValues)
	if err != nil {
		t.Fatal(err)
	}

	gpuEnc, err := vision.LoadEncoder(ckpt, true)
	if err != nil {
		t.Fatal(err)
	}
	if err := gpuEnc.EnableResident(); err != nil {
		t.Fatal(err)
	}
	defer gpuEnc.Close()
	got, err := gpuEnc.Forward(g.PixelValues) // delegates to the device
	if err != nil {
		t.Fatal(err)
	}
	c := cosG(got, want)
	t.Logf("EnableResident Forward vs CPU cosine=%.6f", c)
	if c < 0.99 {
		t.Errorf("resident-delegated Forward cosine %.6f < 0.99", c)
	}
}
