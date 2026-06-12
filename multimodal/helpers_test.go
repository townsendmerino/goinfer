package multimodal

import "math"

// cosine returns the cosine similarity and max abs elementwise difference of two
// equal-length vectors — the parity metric for TestProjector_parity. (Moved here
// with the projector; the encoder half's copy now lives in aikit/vision.)
func cosine(a, b []float32) (cos, maxAbs float64) {
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
		if d := math.Abs(x - y); d > maxAbs {
			maxAbs = d
		}
	}
	if na == 0 || nb == 0 {
		return 0, maxAbs
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb)), maxAbs
}
