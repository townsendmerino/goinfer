//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/aikit/vision"
)

// TestVisionEncoder_forwardPatchesReleasesDeviceScratch is M-18's own gate
// (docs/audit-2026-09-10.md): ForwardPatches allocated 19 per-call scratch device buffers plus a
// fresh command queue, and released none of them — every call leaked. Mirrors
// TestGemma3VisionResidentReal_gate's real-checkpoint setup, but goes straight to
// cuda.NewVisionEncoder (bypassing the vision.Encoder wrapper) so this can read r.dev.Context()'s
// own MemInfo directly, and calls ForwardPatches repeatedly rather than once — a single call
// can't distinguish "leaks every time" from "never releases the very first allocation", both of
// which would pass a correctness-only (cosine) gate but not this one.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestVisionEncoder_forwardPatchesReleasesDeviceScratch -v -timeout 10m
func TestVisionEncoder_forwardPatchesReleasesDeviceScratch(t *testing.T) {
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

	e, err := vision.LoadEncoder(dir, true) // quant=true: int8 weights, required for GPUWeights
	if err != nil {
		t.Fatalf("LoadEncoder: %v", err)
	}
	defer e.Close()
	w, err := e.GPUWeights()
	if err != nil {
		t.Fatalf("GPUWeights: %v", err)
	}

	ve, err := NewVisionEncoder(w)
	if err != nil {
		t.Fatalf("NewVisionEncoder: %v", err)
	}
	defer ve.Close()

	patches := make([]float32, ve.numPatches*ve.cpp)
	for i := range patches {
		// Same deterministic synthetic pattern as TestGemma3VisionResidentReal_gate; this test
		// only cares about device memory, not output correctness.
		patches[i] = float32(math.Sin(float64(i)*0.0091))*0.5 + float32(math.Cos(float64(i)*0.0037))*0.3
	}

	// Warm-up call: absorbs any one-time driver/context allocation that isn't part of the
	// per-call scratch this test is measuring (the FIRST cuInit-adjacent allocation on a device
	// can look different from steady state on some drivers).
	if _, err := ve.ForwardPatches(patches); err != nil {
		t.Fatalf("warmup ForwardPatches: %v", err)
	}
	free0, _, err := ve.dev.Context().MemInfo()
	if err != nil {
		t.Skipf("MemInfo: %v", err)
	}

	const n = 10
	for i := 0; i < n; i++ {
		if _, err := ve.ForwardPatches(patches); err != nil {
			t.Fatalf("ForwardPatches call %d: %v", i, err)
		}
	}
	free1, _, err := ve.dev.Context().MemInfo()
	if err != nil {
		t.Skipf("MemInfo: %v", err)
	}

	drop := int64(free0) - int64(free1)
	t.Logf("free VRAM before %d more calls: %d bytes, after: %d bytes (delta %d bytes)", n, free0, free1, drop)
	// Each call's scratch is comfortably tens of MB (M=numPatches rows across ~13 distinct
	// buffer shapes at this checkpoint's hidden/inter dims) — if leaking, n=10 calls would drop
	// free memory by roughly 10x that. 16 MiB of slack absorbs ordinary allocator
	// fragmentation/rounding, which is ordinary and not the failure mode this test is for.
	const slack = 16 << 20
	if drop > slack {
		t.Errorf("free VRAM dropped by %d bytes over %d ForwardPatches calls (> %d byte slack) — "+
			"scratch buffers are not being released between calls", drop, n, slack)
	}
}
