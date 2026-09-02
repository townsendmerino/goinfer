//go:build gpu

package gpu_test

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/gpu"
)

const (
	gemmaForwardGoldenPath = "../testdata/gemma_forward_golden.json"
	gemmaModelDir          = "../testdata/gemma-3-270m"
)

// forwardGolden mirrors the (only) fields this test needs from the decoder's
// pinned oracle. The full struct lives in the decoder package's tests.
type forwardGolden struct {
	IDs    []int `json:"ids"`
	Argmax int   `json:"argmax"`
}

// TestWebGPU_forwardParity runs a full forward on the WebGPU backend and checks
// the greedy next token matches the f32 oracle — end-to-end proof that resident
// weights (every projection + the tied LM head) produce correct logits on real
// hardware. It blank-relies on this package's init to register "webgpu" with
// the decoder, then drives it through the decoder's exported Generate API so
// the decoder module never imports webgpu. Run with `go test -tags gpu ./...`;
// skips with no adapter or no checkpoint.
func TestWebGPU_forwardParity(t *testing.T) {
	// Skip cleanly on a headless box (no adapter) before touching the model.
	ctx, err := gpu.New()
	if err != nil {
		// A device that worked earlier in this process and is now gone means the test
		// binary exhausted it — skipping there is how this gate silently stopped gating.
		if gpu.GPUEverAvailable() {
			t.Fatalf("GPU exhausted by this test binary (not a missing GPU) — this gate must not skip: %v", err)
		}
		t.Skipf("no WebGPU adapter: %v", err)
	}
	ctx.Close()

	raw, err := os.ReadFile(gemmaForwardGoldenPath)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no forward golden at %s", gemmaForwardGoldenPath)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(gemmaModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s", gemmaModelDir)
	}

	m, err := decoder.Load(gemmaModelDir, decoder.Options{Backend: "webgpu"})
	if err != nil {
		t.Fatalf("Load webgpu: %v", err)
	}

	// Greedy (Temperature 0) → the first emitted token is argmax(logits) after
	// prefilling the whole prompt — exactly the oracle's pinned argmax.
	out, gen := m.Generate(context.Background(), g.IDs, 1, decoder.SamplingParams{Temperature: 0})
	got, ok := <-out
	for range out { // drain in case maxTokens emits more before close
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	if !ok {
		t.Fatal("no token generated")
	}
	if got != g.Argmax {
		t.Errorf("GPU greedy token = %d, want %d", got, g.Argmax)
	}
}
