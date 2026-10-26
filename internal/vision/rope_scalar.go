//go:build !goexperiment.simd

package vision

// applyRotaryVision (default build) — byte-for-byte the body this function had
// before rope_simd.go existed. In-place pairwise: each (vec[d], vec[d+half])
// rotates into itself, so no per-call scratch is needed — this was ~8M tiny
// allocs on a realistic image (~8k patches x 16 heads x 32 blocks). Reads x,y
// before overwriting either, and is bit-identical to the tmp version
// (a+(-b)*s == a-b*s in IEEE).
func applyRotaryVision(vec, cos, sin []float32) {
	half := len(vec) / 2
	if half == 0 {
		return
	}
	_ = vec[2*half-1]
	_ = cos[2*half-1]
	_ = sin[2*half-1]

	d := 0
	for ; d+3 < half; d += 4 {
		x0, y0 := vec[d+0], vec[d+0+half]
		x1, y1 := vec[d+1], vec[d+1+half]
		x2, y2 := vec[d+2], vec[d+2+half]
		x3, y3 := vec[d+3], vec[d+3+half]

		vec[d+0] = x0*cos[d+0] - y0*sin[d+0]
		vec[d+0+half] = y0*cos[d+0+half] + x0*sin[d+0+half]
		vec[d+1] = x1*cos[d+1] - y1*sin[d+1]
		vec[d+1+half] = y1*cos[d+1+half] + x1*sin[d+1+half]
		vec[d+2] = x2*cos[d+2] - y2*sin[d+2]
		vec[d+2+half] = y2*cos[d+2+half] + x2*sin[d+2+half]
		vec[d+3] = x3*cos[d+3] - y3*sin[d+3]
		vec[d+3+half] = y3*cos[d+3+half] + x3*sin[d+3+half]
	}
	for ; d < half; d++ {
		x, y := vec[d], vec[d+half]
		vec[d] = x*cos[d] - y*sin[d]
		vec[d+half] = y*cos[d+half] + x*sin[d+half]
	}
}
