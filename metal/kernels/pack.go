package kernels

import (
	"math"

	"github.com/townsendmerino/goinfer/internal/f16"
)

// PackW4A8Row packs one f32 weight row into goinfer's W4A8 layout: group = 32, per-group
// scale = maxabs/7 (f32), nibble = q+8 ∈ [1,15] (8=zero); element k → u32 word k/8, nibble
// at bit 4·(k%8). Returns the packed words (K/8) and per-group scales (K/32). Validated
// bit-exact by TestLayerB_gemvW4A8Parity.
func PackW4A8Row(row []float32) (words []uint32, scales []float32) {
	words, scales = make([]uint32, len(row)/8), make([]float32, len(row)/32)
	PackW4A8RowInto(row, words, scales)
	return
}

// PackW4A8RowInto is PackW4A8Row into caller-owned words (len(row)/8, overwritten) and scales (len(row)/32), so a caller
// packing many rows allocates nothing per row.
func PackW4A8RowInto(row []float32, words []uint32, scales []float32) {
	K := len(row)
	clear(words[:K/8])
	for g := 0; g < K/32; g++ {
		mx := float32(0)
		for k := g * 32; k < g*32+32; k++ {
			if a := float32(math.Abs(float64(row[k]))); a > mx {
				mx = a
			}
		}
		sc := mx / 7
		scales[g] = sc
		inv := float32(0)
		if sc != 0 {
			inv = 1 / sc
		}
		for k := g * 32; k < g*32+32; k++ {
			q := max(min(int(math.Round(float64(row[k]*inv))), 7), -7)
			words[k/8] |= uint32(q+8) << (4 * (uint(k) % 8))
		}
	}
}

// QuantG32 is tower_gemm_w8's weight form (S18): each row of w [rows, cols] in groups of 32 along cols, each group
// scaled by its largest magnitude over 127 and rounded to int8. The scales are [rows, ceil(cols/32)]. The kernel's
// weight is float(q)·scale, which dequantG32 reproduces for the CPU reference.
func QuantG32(w []float32, rows, cols int) ([]int8, []float32) {
	nG := (cols + 31) / 32
	q, sc := make([]int8, len(w)), make([]float32, rows*nG)
	for r := range rows {
		for g := range nG {
			lo, hi := r*cols+g*32, r*cols+min(g*32+32, cols)
			var mx float32
			for _, v := range w[lo:hi] {
				mx = max(mx, float32(math.Abs(float64(v))))
			}
			if mx == 0 {
				continue
			}
			s := mx / 127
			sc[r*nG+g] = s
			for i := lo; i < hi; i++ {
				q[i] = int8(max(-127, min(127, math.Round(float64(w[i]/s)))))
			}
		}
	}
	return q, sc
}

// DequantG32 is the weight tower_gemm_w8 computes with: float(q)·scale, per group of 32.
func DequantG32(q []int8, sc []float32, rows, cols int) []float32 {
	nG := (cols + 31) / 32
	w := make([]float32, len(q))
	for i := range w {
		r, c := i/cols, i%cols
		w[i] = float32(q[i]) * sc[r*nG+c/32]
	}
	return w
}

// F32ToF16 is f16.Bits, the conversion the W4A8 kernels read their group scales in.
func F32ToF16(f float32) uint16 { return f16.Bits(f) }
