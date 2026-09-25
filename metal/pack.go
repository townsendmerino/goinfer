//go:build darwin

package metal

import (
	"math"

	"github.com/townsendmerino/goinfer/decoder"
)

// packW4A8Row packs one f32 weight row into goinfer's W4A8 layout: group = 32, per-group
// scale = maxabs/7 (f32), nibble = q+8 ∈ [1,15] (8=zero); element k → u32 word k/8, nibble
// at bit 4·(k%8). Returns the packed words (K/8) and per-group scales (K/32). Validated
// bit-exact by TestLayerB_gemvW4A8Parity.
func packW4A8Row(row []float32) (words []uint32, scales []float32) {
	K := len(row)
	words = make([]uint32, K/8)
	scales = make([]float32, K/32)
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
	return
}

// dequantInt8ToF32 reconstructs a row-major int8 weight matrix (q8[N*K], per-row scales)
// to f32 — used to re-pack int8-loaded weights into W4A8 for the int4 GEMV. K padded to a
// multiple of 32 (W4A8 group) with zeros.
func dequantInt8ToF32Row(q8 []int8, scale float32, K, Kpad int) []float32 {
	f := make([]float32, Kpad)
	for k := range K {
		f[k] = float32(q8[k]) * scale
	}
	return f
}

// f32ToF16 converts to IEEE-754 half. W4A8 group scales are small positive finite values, so the
// subnormal/overflow edges are for completeness. f16's ~2^-11 relative precision is far below the 4-bit
// weight-quant noise, so this is parity-neutral (gated). It is decoder.F16Bits — ONE definition, because a
// metal-target .giw (weights format v14) stores scales already converted, and the kernels must see the same
// bits whether a scale came from the file or was converted here.
func f32ToF16(f float32) uint16 { return decoder.F16Bits(f) }
