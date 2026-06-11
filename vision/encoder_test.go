package vision

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestSiglipEncoder_parity gates the pure-Go SigLIP forward against the HF
// SiglipVisionModel golden (scripts/pin_siglip_vision.py): same tiny checkpoint,
// same pixel_values → last_hidden_state cosine ≈ 1.0. Asset-gated on
// testdata/siglip-tiny (committed) + the golden.
func TestSiglipEncoder_parity(t *testing.T) {
	const ckpt = "../testdata/siglip-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no siglip-tiny checkpoint (%v); run scripts/pin_siglip_vision.py", err)
	}
	raw, err := os.ReadFile("../testdata/siglip_vision_golden.json")
	if err != nil {
		t.Skipf("no golden (%v)", err)
	}
	var g struct {
		PixelValues     []float32 `json:"pixel_values"`
		LastHiddenState []float32 `json:"last_hidden_state"`
		LastHiddenShape []int     `json:"last_hidden_state_shape"`
		NumPatches      int       `json:"num_patches"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	enc, err := LoadEncoder(ckpt, false) // f32: bit-exact path
	if err != nil {
		t.Fatalf("LoadEncoder: %v", err)
	}
	if enc.numPatches != g.NumPatches {
		t.Fatalf("numPatches %d != golden %d", enc.numPatches, g.NumPatches)
	}

	got, err := enc.Forward(g.PixelValues)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if len(got) != len(g.LastHiddenState) {
		t.Fatalf("output len %d != golden %d (shape %v)", len(got), len(g.LastHiddenState), g.LastHiddenShape)
	}

	cos, maxAbs := cosine(got, g.LastHiddenState)
	t.Logf("SigLIP encoder vs HF golden (f32): cosine=%.8f, max abs diff=%.3e (shape %v)", cos, maxAbs, g.LastHiddenShape)
	if cos < 0.9999 {
		t.Errorf("last_hidden_state cosine %.8f < 0.9999 — SigLIP forward diverges from HF", cos)
	}

	// int8 (W8A8) encoder: the projections/FFN run as integer matmuls. Lossier
	// than f32 but must stay close to the HF golden (the W8A8 decode tolerance).
	encQ, err := LoadEncoder(ckpt, true)
	if err != nil {
		t.Fatalf("LoadEncoder int8: %v", err)
	}
	gotQ, err := encQ.Forward(g.PixelValues)
	if err != nil {
		t.Fatalf("Forward int8: %v", err)
	}
	cosQ, maxAbsQ := cosine(gotQ, g.LastHiddenState)
	t.Logf("SigLIP encoder vs HF golden (int8 W8A8): cosine=%.8f, max abs diff=%.3e", cosQ, maxAbsQ)
	if cosQ < 0.99 {
		t.Errorf("int8 last_hidden_state cosine %.8f < 0.99", cosQ)
	}
}

func cosine(a, b []float32) (cos, maxAbs float64) {
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
		if d := math.Abs(x - y); d > maxAbs {
			maxAbs = d
		}
	}
	if na == 0 || nb == 0 {
		return 0, maxAbs
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb)), maxAbs
}
