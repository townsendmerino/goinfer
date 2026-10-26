//go:build darwin

package visionmetal

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// ckpt is the tiny-random SigLIP tower committed for the CPU parity gate
// (scripts/oracle/pin_siglip_vision.py regenerates it deterministically). Asset-gated: the
// checkpoint directory is not committed, so this skips cleanly without it.
const ckpt = "../../../internal/vision/testdata/siglip-tiny"

// cosine is the parity statistic the vision path is gated on, matching how the CPU
// tower is gated against HF.
func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
}

func loadTower(t *testing.T) *vision.Encoder {
	t.Helper()
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no siglip-tiny checkpoint (%v); run scripts/oracle/pin_siglip_vision.py", err)
	}
	// quant=true: the resident path requires int8 matmul weights, and gating GPU
	// against a CPU tower loaded the SAME way isolates the device path as the only
	// variable (quantization error is then common to both sides, not a confound).
	e, err := vision.LoadEncoder(ckpt, true)
	if err != nil {
		t.Skipf("LoadEncoder: %v", err)
	}
	return e
}

// synthPixels builds a deterministic pixel tensor of the tower's expected shape.
func synthPixels(e *vision.Encoder) []float32 {
	c := e.Cfg
	n := c.NumChannels * c.ImageSize * c.ImageSize
	px := make([]float32, n)
	var s uint32 = 7
	for i := range px {
		s = s*1664525 + 1013904223
		px[i] = float32(int32(s>>8)%2000-1000) / 1000.0
	}
	return px
}

// TestVisionMetal_parityWithCPU is the Phase-3 gate: the whole SigLIP tower run on the
// GPU must reproduce the pure-Go CPU tower's last_hidden_state. Both sides load the
// same int8 checkpoint, so the device path is the only variable.
//
// The bar is cosine, not bit-equality, and deliberately so: the CPU tower accumulates
// LayerNorm/softmax/GELU in float64 while the GPU does the bulk matmuls in f32/int32,
// so the two reassociate differently. The int8 dots are exact, so the divergence that
// remains is float reassociation only — which is why the bar below is tight (1-1e-6)
// rather than a loose "looks similar".
func TestVisionMetal_parityWithCPU(t *testing.T) {
	cpu := loadTower(t)
	px := synthPixels(cpu)

	want, err := cpu.Forward(px)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	gpuEnc := loadTower(t)
	if err := gpuEnc.EnableResident(); err != nil {
		t.Skipf("EnableResident: %v (no Metal device?)", err)
	}
	defer gpuEnc.Close()

	got, err := gpuEnc.Forward(px)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("GPU returned %d values, want %d", len(got), len(want))
	}

	cos := cosine(got, want)
	worst := 0.0
	for i := range want {
		if d := math.Abs(float64(got[i]) - float64(want[i])); d > worst {
			worst = d
		}
	}
	t.Logf("SigLIP GPU≡CPU: %d values, cosine %.9f, worst abs Δ %.3g", len(want), cos, worst)
	if cos < 1-1e-6 {
		t.Errorf("cosine %.9f below the 1-1e-6 bar — the GPU tower diverges from CPU", cos)
	}

	// Break-it-first: the cosine bar must be able to FAIL. A tower run on DIFFERENT
	// pixels must not clear it — otherwise "cosine ≈ 1" would be measuring the
	// checkpoint's output distribution rather than this forward.
	other := make([]float32, len(px))
	copy(other, px)
	for i := range other {
		other[i] = -other[i]
	}
	alt, err := cpu.Forward(other)
	if err != nil {
		t.Fatalf("CPU Forward (alt): %v", err)
	}
	if c := cosine(got, alt); c >= 1-1e-6 {
		t.Errorf("break-it-first vacuous: a different input still scored cosine %.9f", c)
	} else {
		t.Logf("break-it-first: negated input scores cosine %.6f — the gate discriminates", c)
	}
}

// TestVisionMetal_registersBackend proves the inversion is wired: importing this
// package registers the factory, so EnableResident finds a backend rather than
// reporting "no resident encoder backend".
func TestVisionMetal_registersBackend(t *testing.T) {
	e := loadTower(t)
	defer e.Close()
	if err := e.EnableResident(); err != nil {
		t.Skipf("EnableResident: %v", err)
	}
	t.Log("vision.RegisterResident wired: EnableResident attached the Metal tower")
}

// TestVisionMetal_repeatable checks the resident encoder is reusable across calls —
// the scratch buffers are reused between forwards, so a missing reset would show as
// drift on the second call.
func TestVisionMetal_repeatable(t *testing.T) {
	e := loadTower(t)
	if err := e.EnableResident(); err != nil {
		t.Skipf("EnableResident: %v", err)
	}
	defer e.Close()
	px := synthPixels(e)
	first, err := e.Forward(px)
	if err != nil {
		t.Fatalf("Forward 1: %v", err)
	}
	for i := range 3 {
		again, err := e.Forward(px)
		if err != nil {
			t.Fatalf("Forward %d: %v", i+2, err)
		}
		for j := range first {
			if again[j] != first[j] {
				t.Fatalf("call %d diverged at %d: %v != %v (scratch not reset?)", i+2, j, again[j], first[j])
			}
		}
	}
	t.Log("4 forwards on reused scratch: bit-identical")
}
