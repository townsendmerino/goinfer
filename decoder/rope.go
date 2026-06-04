package decoder

import "math"

// applyRoPE rotates a single position's projected heads in place. vec is
// [heads, headDim] flattened row-major (a query or key vector for ONE token at
// absolute position pos). invFreq is the precomputed inverse-frequency table
// (arch.ropeInvFreq, with base, scaling, and rotary dim already resolved), so
// the rotated span is rotaryDim = 2*len(invFreq).
//
// NeoX / HF convention (rotate_half, non-interleaved), matching encoder/rope.go
// and HF's apply_rotary_pos_emb:
//
//	x1 = x[:half]; x2 = x[half:rot]
//	out[:half]    = x1*cos - x2*sin
//	out[half:rot] = x2*cos + x1*sin
//
// where θ_d = pos * invFreq[d] for d ∈ [0, rotaryDim/2). When rotaryDim <
// headDim (Phi's partial_rotary_factor), the trailing headDim-rotaryDim dims
// pass through unrotated.
//
// scale is YaRN's attention_factor (mscale), folded into cos/sin so the rotated
// q/k are scaled by it — matching HF, which multiplies cos/sin by
// attention_scaling. Pass 1.0 for non-YaRN families (no scaling).
func applyRoPE(vec []float32, heads, headDim, pos int, invFreq []float64, scale float64) {
	half := len(invFreq) // == rotaryDim/2
	posF := float64(pos)
	for d := range half {
		theta := posF * invFreq[d]
		c := math.Cos(theta) * scale
		s := math.Sin(theta) * scale
		for h := range heads {
			off := h * headDim
			x1 := float64(vec[off+d])
			x2 := float64(vec[off+half+d])
			vec[off+d] = float32(x1*c - x2*s)
			vec[off+half+d] = float32(x2*c + x1*s)
		}
	}
}
