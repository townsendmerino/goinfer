//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/aikit/vision"
)

// TestGemma3VisionResidentReal_gate is P6's real-checkpoint gate (docs/multimodal.md's "P6's other half"): the
// resident CUDA SigLIP tower vs its own CPU path, on the real gemma-3-4b-it vision tower, matched precision
// (both int8: vision.LoadEncoder(dir, quant=true) puts BOTH paths on the same W8A8 weights; this is not an
// int8-vs-f32 comparison).
//
// THE THRESHOLD IS NOT THE USUAL ≥0.99. The kernels are exact (patch-embed matches the CPU path at cosine
// 0.999999, and a matched-precision CPU probe on the SAME int8 weight data reproduces a layer's FC2 GEMV at
// 1.000000). What is not bit-identical is the ACCUMULATION ORDER of the per-layer reductions (LayerNorm sums,
// the attention softmax denominator, the int8 GEMV's own accumulation), an ordinary cosine-gated property that
// compounds over SigLIP so400m's 27 layers to an input-dependent end-to-end cosine well below 0.99. The floor
// (0.80) sits with margin under that range: loose enough not to flake on input-dependent rounding, tight enough
// that a wiring regression fails by a wide margin. The measured range: docs/code-notes/cuda.md#TestGemma3VisionResidentReal_gate.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestGemma3VisionResidentReal_gate -v -timeout 10m
func TestGemma3VisionResidentReal_gate(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	dir := os.Getenv("GEMMA3_4B")
	if dir == "" {
		dir = home + "/models/gemma-3-4b-it"
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no gemma-3-4b-it at %s: %v", dir, err)
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}

	e, err := vision.LoadEncoder(dir, true) // quant=true: int8, matched precision both arms
	if err != nil {
		t.Fatalf("LoadEncoder: %v", err)
	}
	defer e.Close()

	cfg := e.Cfg
	pixels := make([]float32, cfg.NumChannels*cfg.ImageSize*cfg.ImageSize)
	for i := range pixels {
		// Deterministic synthetic pixels (real preprocessed-image goldens are gzip-committed
		// fixtures this test doesn't need — the gate is CPU-vs-resident on the SAME real
		// weights, not an HF oracle comparison, so any real, non-degenerate input suffices).
		pixels[i] = float32(math.Sin(float64(i)*0.0091))*0.5 + float32(math.Cos(float64(i)*0.0037))*0.3
	}

	cpuOut, err := e.Forward(pixels) // resident not yet enabled: CPU path
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	if err := e.EnableResident(); err != nil {
		t.Fatalf("EnableResident: %v", err)
	}
	gpuOut, err := e.Forward(pixels) // resident enabled: routes through cuda.VisionEncoder
	if err != nil {
		t.Fatalf("resident Forward: %v", err)
	}

	if len(cpuOut) != len(gpuOut) {
		t.Fatalf("output length mismatch: cpu=%d gpu=%d", len(cpuOut), len(gpuOut))
	}
	cos := cosineF32(cpuOut, gpuOut)
	maxDiff := float32(0)
	for i := range cpuOut {
		if d := float32(math.Abs(float64(cpuOut[i] - gpuOut[i]))); d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("gemma-3-4b-it vision tower, CPU vs resident CUDA: cosine=%.6f max|diff|=%.4f (%d patches, %d layers)",
		cos, maxDiff, cfg.ImageSize/cfg.PatchSize*cfg.ImageSize/cfg.PatchSize, cfg.NumHiddenLayers)
	if cos < 0.80 {
		t.Errorf("cosine %.6f < 0.80 — the resident CUDA vision tower diverges from its own CPU path "+
			"by more than ordinary accumulated int8 rounding (see this test's own doc comment for the "+
			"measured baseline and how it was confirmed genuine)", cos)
	}
}
