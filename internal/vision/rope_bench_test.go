package vision

import (
	"math/rand"
	"testing"
)

// BenchmarkApplyRotaryVision reports ns/elem at a realistic head_dim (128,
// the Qwen2-VL vision tower's shape per attentionInto's caller). Reflects
// whichever build (scalar or GOEXPERIMENT=simd) it's run under, since
// applyRotaryVision is the same call either way.
func BenchmarkApplyRotaryVision(b *testing.B) {
	const headDim = 128
	rng := rand.New(rand.NewSource(1))
	vec := make([]float32, headDim)
	cos := make([]float32, headDim)
	sin := make([]float32, headDim)
	for i := range vec {
		vec[i] = float32(rng.NormFloat64())
		cos[i] = float32(rng.Float64())
		sin[i] = float32(rng.Float64())
	}
	for b.Loop() {
		applyRotaryVision(vec, cos, sin)
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*headDim), "ns/elem")
}
