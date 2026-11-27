package cpuengine

import "github.com/townsendmerino/aikit/linalg"

// LoRADelta holds one projection's low-rank adapter matrices A [R, In] and B [Out, R] plus scaling factor.
type LoRADelta struct {
	A     []float32
	B     []float32
	R     int
	In    int
	Out   int
	Scale float64
}

// LoRALayerDelta holds one transformer layer's per-projection compute-time LoRA deltas.
type LoRALayerDelta struct {
	Q, K, V, O, Gate, Up, Down *LoRADelta
}

// ApplyLoRA adds the low-rank delta Scale*B(A*x) into y in place.
func ApplyLoRA(d *LoRADelta, x, y []float32, scr *DecodeScratch) {
	if d == nil {
		return
	}
	t := scr.loraBuf(d.R)
	linalg.MatmulBT(x, d.A, t, 1, d.In, d.R)
	sc := float32(d.Scale)
	for o := 0; o < d.Out; o++ {
		brow := d.B[o*d.R : o*d.R+d.R]
		var s float32
		for k := 0; k < d.R; k++ {
			s += brow[k] * t[k]
		}
		y[o] += sc * s
	}
}
