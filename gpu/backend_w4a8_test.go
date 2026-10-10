//go:build gpu && goinfer_testhooks

package gpu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestWebGPUBackend_MatmulW4A8_matchesCPU is the staged-int4 gate
// (docs/tasks/task-gpu-paths-2026-09.md, G6): webgpuBackend.MatmulW4A8, the decoder.QuantBackend4
// entry point matmulInto/matmul call, against the CPU reference linalg.MatmulBTW4A8Into.
//
// K=517 is deliberately not a multiple of 32 (w4a8GroupSize): it takes the fallback
// unpack-and-repack upload and exercises GEMVRunner's zero-tail-padding invariant (aBuf is sized
// to kPad() bytes and CreateBuffer zero-inits the tail; see GEMVRunner.Run). K=512 is the common
// 32-aligned dim, residentW4A8For's UploadW4A8Packed straight-upload path that real checkpoints
// take, so this gate covers both upload paths itself.
func TestWebGPUBackend_MatmulW4A8_matchesCPU(t *testing.T) {
	for name, K := range map[string]int{
		"K=517 (not a multiple of 32 -- fallback unpack-and-repack path)": 517,
		"K=512 (multiple of 32 -- production fast path)":                  512,
	} {
		t.Run(name, func(t *testing.T) {
			be, err := newWebGPUBackend("decoder")
			if err != nil {
				t.Skipf("no webgpu device: %v", err)
			}
			defer be.Close()

			const N = 12
			const group = w4a8GroupSize
			rng := rand.New(rand.NewSource(7))

			// Packed on-disk layout: 2 nibbles/byte, low nibble first — decoder's native int4
			// format, matching uploadProj's own unpack loop (b&0x0F for even k, b>>4 for odd k).
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
			if !be.MatmulW4A8(a, bQ4, linalg.F32ToF16Scales(scales), group, dstGPU, 1, K, N) {
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
		})
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
	if be.MatmulW4A8(a, bQ4, linalg.F32ToF16Scales(scales), w4a8GroupSize, dst, M, K, N) {
		t.Error("MatmulW4A8 must decline for M>1 (no int4 prefill kernel on this backend)")
	}
}
