//go:build gpu

package gpu_test

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/gpu"
)

// TestResidentQKNorm_parity is the gate for Lever C increment 1: per-head QK-norm in the
// resident DecodeRunner (gpu/qknorm.go), which makes Qwen3 (and any dense QK-norm arch)
// GPU-resident — previously decodeRunnerEligible excluded QKNorm outright. Two checks:
//
//  1. the model goes RESIDENT (the eligibility unlock actually engaged); and
//  2. the resident greedy next token matches the HF f32 oracle's argmax — proving the
//     GPU per-head RMSNorm of q/k before RoPE is correct (a wrong/absent QK-norm would
//     shift the logits and miss the argmax). Single forward, so no quant-accumulation
//     drift; the prompt prefill still applies QK-norm at every position.
//
// Real Qwen3-1.7B (Q8_0 GGUF → W8A8 resident). The CPU forward already gates this golden
// (decoder.TestQwen3_forwardParity); here it's the resident path.
//
//	go test -tags gpu ./gpu/ -run TestResidentQKNorm_parity -v -timeout 20m
func TestResidentQKNorm_parity(t *testing.T) {
	ctx, err := gpu.New()
	if err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	ctx.Close()

	home, _ := os.UserHomeDir()
	gguf := os.Getenv("GOINFER_QWEN3_GGUF")
	if gguf == "" {
		gguf = filepath.Join("..", "testdata", "qwen3-gguf", "Qwen3-1.7B-Q8_0.gguf")
		if _, err := os.Stat(gguf); errors.Is(err, fs.ErrNotExist) {
			gguf = filepath.Join(home, "models", "qwen3-gguf", "Qwen3-1.7B-Q8_0.gguf")
		}
	}
	if _, err := os.Stat(gguf); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Qwen3 GGUF at %s", gguf)
	}
	raw, err := os.ReadFile("../testdata/qwen3_forward_golden.json")
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no qwen3_forward_golden.json")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	m, err := decoder.Load(gguf, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load webgpu: %v", err)
	}
	defer m.Close()
	if !m.ResidentActive() {
		t.Fatal("Qwen3 not GPU-resident — QK-norm eligibility did not engage (Lever C1 regressed)")
	}

	out, gen := m.Generate(context.Background(), g.IDs, 1, decoder.SamplingParams{Temperature: 0})
	got, ok := <-out
	for range out {
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	if !ok {
		t.Fatal("no token generated")
	}
	t.Logf("resident QK-norm (Qwen3-1.7B W8A8): greedy token=%d, golden argmax=%d", got, g.Argmax)
	if got != g.Argmax {
		t.Errorf("resident QK-norm greedy token = %d, want %d (golden) — QK-norm path wrong", got, g.Argmax)
	}
}
