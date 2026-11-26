package archparts

import (
	"math"
)

// Gemma4InvFreq builds the RoPE inverse-frequency table. For full rotary,
// rotaryDim == headDim and the result is headDim/2 entries. For proportional
// (partial) rotary, rotaryDim < headDim: the first rotaryDim/2 entries are the
// real frequencies (spaced by 1/headDim, NOT 1/rotaryDim) and the rest are zero,
// so rope.Apply (which rotates len(invFreq) pairs split at headDim/2) leaves the
// trailing dims unrotated — matching HF's "proportional" RoPE.
func Gemma4InvFreq(headDim, rotaryDim int, base float64) []float64 {
	half := headDim / 2
	inv := make([]float64, half)
	angles := rotaryDim / 2
	for i := range angles {
		inv[i] = 1.0 / math.Pow(base, float64(2*i)/float64(headDim))
	}
	return inv // entries [angles:half] stay 0 → no rotation there
}
