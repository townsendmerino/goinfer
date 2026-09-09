//go:build gpu && goinfer_testhooks

package gpu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestWebGPUBackend_MatmulW4A8_matchesCPU is G6's staged-int4 gate (docs/task-gpu-paths-2026-09.md):
// webgpuBackend.MatmulW4A8, the actual decoder.QuantBackend4 entry point matmulInto/matmul now
// call, against the CPU reference (linalg.MatmulBTW4A8Into) — the same comparison
// TestWebGPUBackend_matchesCPU does for the f32 path, but nothing previously did this directly
// for MatmulW8A8 either (worth noting: this establishes the pattern, not just mirrors it).
//
// K=517 is deliberately NOT a multiple of 32 (w4a8GroupSize): it exercises both the fallback
// unpack-and-repack upload path (K%32==0 is the fast path) and GEMVRunner's zero-tail-padding
// invariant (aBuf is sized to kPad() bytes, and CreateBuffer zero-inits the untouched tail —
// see GEMVRunner.Run's own comment).
func TestWebGPUBackend_MatmulW4A8_matchesCPU(t *testing.T) {
	be, err := newWebGPUBackend("decoder")
	if err != nil {
		t.Skipf("no webgpu device: %v", err)
	}
	defer be.Close()

	const N, K = 12, 517
	const group = w4a8GroupSize
	rng := rand.New(rand.NewSource(7))

	// Packed on-disk layout: 2 nibbles/byte, low nibble first — decoder's native int4 format,
	// matching uploadProj's own unpack loop (b&0x0F for even k, b>>4 for odd k).
	rowBytes := (K + 1) / 2
	bQ4 := make([]byte, N*rowBytes)
	for i := range bQ4 {
		bQ4[i] = byte(rng.Intn(256))
	}
	nGroups := (K + group - 1) / group
	scales := make([]float32, N*nGroups)
	for i := range scales {
		scales[i] = 0.01 + rng.Float32()*0.05
	}
	a := make([]float32, K)
	for i := range a {
		a[i] = rng.Float32()*2 - 1
	}

	dstGPU := make([]float32, N)
	if !be.MatmulW4A8(a, bQ4, scales, group, dstGPU, 1, K, N) {
		t.Fatal("MatmulW4A8 declined — expected it to run on a real device")
	}

	dstCPU := make([]float32, N)
	var ws linalg.Workspace
	linalg.MatmulBTW4A8Into(&ws, a, bQ4, scales, dstCPU, 1, K, N, group)

	var dot, na, nb, maxAbs float64
	for i := range dstGPU {
		g, c := float64(dstGPU[i]), float64(dstCPU[i])
		if d := math.Abs(g - c); d > maxAbs {
			maxAbs = d
		}
		dot += g * c
		na += g * g
		nb += c * c
	}
	cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
	t.Logf("MatmulW4A8 vs CPU: cosine=%.6f maxAbs=%.4e", cos, maxAbs)
	if cos < 0.999 {
		t.Errorf("MatmulW4A8 diverges from CPU: cosine=%.6f maxAbs=%.4e", cos, maxAbs)
	}
}

// TestWebGPUBackend_MatmulW4A8_declinesPrefill pins the documented M>1 decline: this backend has
// no int4 tiled/prefill kernel, so MatmulW4A8 must return false (not silently run M=1 math on a
// multi-row activation) and let the caller fall back to the CPU kernel.
func TestWebGPUBackend_MatmulW4A8_declinesPrefill(t *testing.T) {
	be, err := newWebGPUBackend("decoder")
	if err != nil {
		t.Skipf("no webgpu device: %v", err)
	}
	defer be.Close()

	const N, K, M = 4, 64, 3
	bQ4 := make([]byte, N*K/2)
	scales := make([]float32, N)
	a := make([]float32, M*K)
	dst := make([]float32, M*N)
	if be.MatmulW4A8(a, bQ4, scales, w4a8GroupSize, dst, M, K, N) {
		t.Error("MatmulW4A8 must decline for M>1 (no int4 prefill kernel on this backend)")
	}
}
