//go:build goexperiment.simd

package vision

import (
	"math"
	"math/rand"
	"testing"
)

// applyRotaryVisionScalar is rope_scalar.go's body, reproduced here since
// that file is excluded from this build — the reference this test checks
// applyRotaryVision (the SIMD build) against.
func applyRotaryVisionScalar(vec, cos, sin []float32) {
	half := len(vec) / 2
	for d := range half {
		x, y := vec[d], vec[d+half]
		vec[d] = x*cos[d] - y*sin[d]
		vec[d+half] = y*cos[d+half] + x*sin[d+half]
	}
}

// TestApplyRotaryVision_matchesScalar checks the SIMD rotation against the
// scalar reference at every head_dim that plausibly occurs (small ones to
// exercise every tail remainder, plus real head_dim values) and confirms the
// actual vector-vs-scalar gap — pure Mul/Sub/Add carries no MulAdd/FMA
// contraction, so this is not assumed bit-identical, just measured.
func TestApplyRotaryVision_matchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	var maxDiff float32
	for _, headDim := range []int{2, 4, 6, 8, 16, 30, 32, 64, 80, 128} {
		for trial := range 20 {
			vecA := make([]float32, headDim)
			cos := make([]float32, headDim)
			sin := make([]float32, headDim)
			for i := range vecA {
				vecA[i] = float32(rng.NormFloat64())
				theta := rng.Float64() * math.Pi
				cos[i] = float32(math.Cos(theta))
				sin[i] = float32(math.Sin(theta))
			}
			vecB := append([]float32(nil), vecA...)

			applyRotaryVisionScalar(vecA, cos, sin)
			applyRotaryVision(vecB, cos, sin)

			for i := range vecA {
				d := float32(math.Abs(float64(vecA[i] - vecB[i])))
				if d > maxDiff {
					maxDiff = d
				}
				if d > 1e-5 {
					t.Fatalf("headDim=%d trial=%d i=%d: scalar=%v simd=%v diff=%v",
						headDim, trial, i, vecA[i], vecB[i], d)
				}
			}
		}
	}
	t.Logf("max abs diff vs scalar: %v", maxDiff)
}
