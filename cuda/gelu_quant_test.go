//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/linalg"
)

// TestGeluQuantBatched_matchesCPUReference pins gelu_quant_batched against linalg.GELUTanhInto
// (aikit's own reference — never reimplemented here), at SigLIP so400m's real intermediate
// width (4304) so the kernel's real N/4 packing path is exercised, not just a round number.
func TestGeluQuantBatched_matchesCPUReference(t *testing.T) {
	dev := requireBareCUDADevice(t)
	mod, err := dev.CompileLibrary(geluQuantPTX)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := dev.NewComputePipeline(mod, "gelu_quant_batched")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const M, N = 5, 4304 // SigLIP so400m's real FC1 output width
	x := make([]float32, M*N)
	for i := range x {
		// A wide range including negatives (GELU_tanh(x) -> 0 for x << 0) and large positives
		// (GELU_tanh(x) -> x for x >> 0), matching what a real FC1 output distribution covers.
		x[i] = float32(math.Sin(float64(i)*0.023)) * 15.0
	}
	ref := append([]float32(nil), x...)
	linalg.GELUTanhInto(ref, ref)

	q := dev.NewCommandQueue()
	xb := gpu.NewBufferLenOf[float32](dev, len(x))
	if err := gpu.Upload(xb, x); err != nil {
		t.Fatalf("upload x: %v", err)
	}
	qOut := gpu.NewBufferLenOf[int32](dev, M*N/4)
	sOut := gpu.NewBufferLenOf[float32](dev, M)
	if err := q.Launch(pipe, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((256 + N) * 4)},
		Arg(xb), gpu.ArgValue(int32(N)), Arg(qOut), Arg(sOut)); err != nil {
		t.Fatalf("launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("sync: %v", err)
	}
	packed := make([]int32, M*N/4)
	scale := make([]float32, M)
	if err := gpu.Download(qOut, packed); err != nil {
		t.Fatalf("download q: %v", err)
	}
	if err := gpu.Download(sOut, scale); err != nil {
		t.Fatalf("download s: %v", err)
	}
	got := dequantRows(packed, scale, M, N)

	cos := cosineF32(ref, got)
	t.Logf("gelu_quant_batched vs linalg.GELUTanhInto: cosine=%.6f", cos)
	if cos < 0.999 {
		t.Errorf("cosine %.6f < 0.999 — gelu_quant_batched diverges from the CPU reference", cos)
	}
}
