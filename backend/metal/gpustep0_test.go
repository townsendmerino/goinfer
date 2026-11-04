//go:build darwin

package metal

import (
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGPUStep0 — headroom decision-tree Step 0 (cgo-free, no Instruments). Separates the
// per-token wall time into GPU-busy (GPUEndTime-GPUStartTime) vs host bubble (wall - GPU-busy),
// using the MTLCommandBuffer timestamps. Answers: (a) is decode GPU-bound (bubble small) or is
// there a fat token-boundary bubble to close with encode-ahead? (b) what effective GB/s does
// the GPU-only window imply, vs the ~150-170 GPU-reachable ceiling on M1 Pro.
func TestGPUStep0(t *testing.T) {
	requireHeavyModel(t)
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not present: %v", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}

	// Weight traffic per token (int4 nibbles + f32 group scales), Fable's corrected budget.
	H, nL, nH, nKV, hd, I, V := m.Dims()
	kvDim := nKV * hd
	qkvRows := nH*hd + 2*kvDim
	w4 := func(N, K int) float64 { return float64(N)*float64(K)/2 + float64(N)*float64(K)/32*2 } // nibbles + f16 scales (L1)
	perLayer := w4(qkvRows, H) + w4(2*I, H) + w4(H, nH*hd) + w4(H, I)
	bytesTok := float64(nL)*perLayer + w4(V, H)

	const warm, iters = 8, 60
	tok, pos := 12095, 30
	for i := range warm {
		r.Forward(tok, pos+i)
	}
	base := pos + warm
	// Track wall/GPU/kernel from the SAME iteration (the min-wall one) so the bubble is a real
	// per-token decomposition, plus the min GPU-busy seen (steady-state GPU floor).
	bestWall := time.Hour
	var gpuAtBest, kernAtBest float64
	minGPU := 1e9
	for i := range iters {
		t0 := time.Now()
		r.Forward(tok, base+i)
		wall := time.Since(t0)
		gpuBusy, kernTotal := r.LastGPUTimes()
		if wall < bestWall {
			bestWall, gpuAtBest, kernAtBest = wall, gpuBusy, kernTotal
		}
		if gpuBusy < minGPU {
			minGPU = gpuBusy
		}
	}
	wallMs := bestWall.Seconds() * 1000
	gpuMs := gpuAtBest * 1000
	bubbleMs := wallMs - gpuMs
	_ = minGPU
	t.Logf("per-token (best-of-%d warm):", iters)
	t.Logf("  wall        = %.2f ms  (%.1f tok/s)", wallMs, 1000/wallMs)
	t.Logf("  GPU-busy    = %.2f ms  (GPUEnd-GPUStart)", gpuMs)
	t.Logf("  kernel win  = %.2f ms  (kernelEnd-kernelStart, incl scheduling)", kernAtBest*1000)
	t.Logf("  host bubble = %.2f ms  (wall - GPU-busy)  => %s",
		bubbleMs, bubbleVerdict(bubbleMs))
	t.Logf("weight traffic = %.1f MB/token (nibbles + f16 scales)", bytesTok/1e6)
	t.Logf("  effective BW over GPU-busy = %.1f GB/s  (min-GPU %.1f GB/s; M1 Pro GPU-reachable ~150-170)",
		bytesTok/1e9/gpuAtBest, bytesTok/1e9/minGPU)
	t.Logf("  effective BW over wall     = %.1f GB/s", bytesTok/1e9/bestWall.Seconds())
}

func bubbleVerdict(ms float64) string {
	switch {
	case ms > 1.0:
		return "FAT bubble -> encode-ahead (L2) first"
	case ms > 0.4:
		return "moderate bubble -> encode-ahead worth ~+2-4 tok/s"
	default:
		return "small -> GPU-bound, value is in-kernel (L1/L4/L5)"
	}
}
