//go:build goexperiment.simd

package vision

import "simd"

// applyRotaryVision (SIMD build) — see rope_scalar.go for the algorithm and
// the default-build body. Vectorizes the elementwise rotation over the
// `half`-length axis: x = vec[:half], y = vec[half:] read fully before
// either half is overwritten, matching the scalar body's own ordering
// guarantee. Pure Mul/Sub/Add (no MulAdd) — unlike the exp/softmax landing
// this does NOT rely on FMA contraction, so TestApplyRotaryVision_matchesScalar
// measures the actual vector-vs-scalar gap rather than assuming a bound.
func applyRotaryVision(vec, cos, sin []float32) {
	half := len(vec) / 2
	x := vec[:half]
	y := vec[half:]
	c1, c2 := cos[:half], cos[half:]
	s1, s2 := sin[:half], sin[half:]

	var probe simd.Float32s
	L := probe.Len()
	i := 0
	for ; i+L <= half; i += L {
		xv := simd.LoadFloat32s(x[i : i+L])
		yv := simd.LoadFloat32s(y[i : i+L])
		c1v := simd.LoadFloat32s(c1[i : i+L])
		c2v := simd.LoadFloat32s(c2[i : i+L])
		s1v := simd.LoadFloat32s(s1[i : i+L])
		s2v := simd.LoadFloat32s(s2[i : i+L])

		newX := xv.Mul(c1v).Sub(yv.Mul(s1v))
		newY := yv.Mul(c2v).Add(xv.Mul(s2v))

		newX.Store(x[i : i+L])
		newY.Store(y[i : i+L])
	}
	for ; i < half; i++ { // tail
		xi, yi := x[i], y[i]
		x[i] = xi*c1[i] - yi*s1[i]
		y[i] = yi*c2[i] + xi*s2[i]
	}
}
