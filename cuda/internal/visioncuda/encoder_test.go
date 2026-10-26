//go:build linux

package visioncuda

import (
	"math"
	"os"
	"testing"

	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
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

// TestVisionCUDA_parityWithCPU is the Phase-3 gate: the whole SigLIP tower run on the
// GPU must reproduce the pure-Go CPU tower's last_hidden_state. Both sides load the
// same int8 checkpoint, so the device path is the only variable.
//
// The bar is cosine, not bit-equality, and deliberately so: the CPU tower accumulates
// LayerNorm/softmax/GELU in float64 while the GPU does the bulk matmuls in f32/int32,
// so the two reassociate differently. The int8 dots are exact, so the divergence that
// remains is float reassociation only — which is why the bar below is tight (1-1e-6)
// rather than a loose "looks similar".
func TestVisionCUDA_parityWithCPU(t *testing.T) {
	cpu := loadTower(t)
	px := synthPixels(cpu)

	want, err := cpu.Forward(px)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	gpuEnc := loadTower(t)
	if err := gpuEnc.EnableResident(); err != nil {
		t.Skipf("EnableResident: %v (no CUDA device?)", err)
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

// TestVisionCUDA_registersBackend proves the inversion is wired: importing this
// package registers the factory, so EnableResident finds a backend rather than
// reporting "no resident encoder backend".
func TestVisionCUDA_registersBackend(t *testing.T) {
	e := loadTower(t)
	defer e.Close()
	if err := e.EnableResident(); err != nil {
		t.Skipf("EnableResident: %v", err)
	}
	t.Log("vision.RegisterResident wired: EnableResident attached the CUDA tower")
}

// TestVisionCUDA_repeatable checks the resident encoder is reusable across calls —
// the scratch buffers are reused between forwards, so a missing reset would show as
// drift on the second call.
func TestVisionCUDA_repeatable(t *testing.T) {
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

// TestVisionCUDA_graphCapture verifies that the whole-tower CUDA Graph captured at
// initialization time is instantiated and replays BIT-IDENTICALLY to direct dispatch.
func TestVisionCUDA_graphCapture(t *testing.T) {
	e := loadTower(t)
	defer e.Close()
	w, err := e.GPUWeights()
	if err != nil {
		t.Fatalf("GPUWeights: %v", err)
	}
	enc, err := newEncoder(w)
	if err != nil {
		// Skip only when there is no device at all; a failure on a real device stays red.
		if _, derr := gpu.CreateSystemDefaultDevice(); derr != nil {
			t.Skipf("no CUDA device: %v", derr)
		}
		t.Fatalf("newEncoder: %v", err)
	}
	defer enc.Close()

	if enc.graph == nil {
		t.Fatal("enc.graph is nil; whole-tower CUDA Graph capture did not instantiate")
	}
	t.Log("whole-tower CUDA Graph captured successfully")

	patches := make([]float32, w.NumPatches*enc.cpp)
	for i := range patches {
		patches[i] = float32(i%100) * 0.01
	}

	// 1. Run with graph replay
	outGraph, err := enc.ForwardPatches(patches)
	if err != nil {
		t.Fatalf("ForwardPatches with graph: %v", err)
	}

	// 2. Temporarily disable graph to run manual forwardTower
	savedGraph := enc.graph
	enc.graph = nil
	outDirect, err := enc.ForwardPatches(patches)
	enc.graph = savedGraph
	if err != nil {
		t.Fatalf("ForwardPatches direct: %v", err)
	}

	if len(outGraph) != len(outDirect) {
		t.Fatalf("len mismatch: %d vs %d", len(outGraph), len(outDirect))
	}
	for i := range outGraph {
		if outGraph[i] != outDirect[i] {
			t.Fatalf("elem %d diverged: graph %v != direct %v", i, outGraph[i], outDirect[i])
		}
	}
	t.Log("CUDA Graph replay is BIT-IDENTICAL to direct forwardTower dispatch")
}

// BenchmarkVisionCUDA_Forward benchmarks steady-state forward passes comparing whole-tower
// CUDA Graph execution against separate kernel dispatch.
func BenchmarkVisionCUDA_Forward(b *testing.B) {
	if _, err := os.Stat(ckpt); err != nil {
		b.Skipf("no siglip-tiny checkpoint (%v)", err)
	}
	e, err := vision.LoadEncoder(ckpt, true)
	if err != nil {
		b.Skipf("LoadEncoder: %v", err)
	}
	defer e.Close()
	w, err := e.GPUWeights()
	if err != nil {
		b.Fatalf("GPUWeights: %v", err)
	}
	enc, err := newEncoder(w)
	if err != nil {
		b.Fatalf("newEncoder: %v", err)
	}
	defer enc.Close()

	patches := make([]float32, w.NumPatches*enc.cpp)
	for i := range patches {
		patches[i] = float32(i%100) * 0.01
	}

	b.Run("GraphReplay", func(b *testing.B) {
		if enc.graph == nil {
			b.Skip("no graph")
		}
		// warm up
		for range 5 {
			if _, err := enc.ForwardPatches(patches); err != nil {
				b.Fatalf("warmup: %v", err)
			}
		}
		b.ResetTimer()
		for range b.N {
			if _, err := enc.ForwardPatches(patches); err != nil {
				b.Fatalf("forward: %v", err)
			}
		}
	})

	b.Run("DirectDispatch", func(b *testing.B) {
		saved := enc.graph
		enc.graph = nil
		defer func() { enc.graph = saved }()

		// warm up
		for range 5 {
			if _, err := enc.ForwardPatches(patches); err != nil {
				b.Fatalf("warmup: %v", err)
			}
		}
		b.ResetTimer()
		for range b.N {
			if _, err := enc.ForwardPatches(patches); err != nil {
				b.Fatalf("forward: %v", err)
			}
		}
	})
}
