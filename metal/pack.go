//go:build darwin

package metal

import "math"

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
			q := min(int(math.Round(float64(row[k]*inv))), 7)
			if q < -7 {
				q = -7
			}
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

// f32ToF16 converts to IEEE-754 half (round-to-nearest). W4A8 group scales are small positive
// finite values, so the subnormal/overflow edges are for completeness. f16's ~2^-11 relative
// precision is far below the 4-bit weight-quant noise, so this is parity-neutral (gated).
func f32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	e := int32((b>>23)&0xFF) - 112 // f16 biased exp = f32biasedexp - 127 + 15
	m := b & 0x7FFFFF
	switch {
	case (b>>23)&0xFF == 0xFF: // inf / nan
		if m != 0 {
			return sign | 0x7E00
		}
		return sign | 0x7C00
	case e >= 0x1F: // overflow -> inf
		return sign | 0x7C00
	case e <= 0: // subnormal / underflow
		if e < -10 {
			return sign
		}
		m |= 0x800000
		sh := uint32(14 - e)
		return sign | uint16((m+(1<<(sh-1)))>>sh)
	default:
		half := sign | uint16(e<<10) | uint16(m>>13)
		if m&0x1000 != 0 { // round up; a mantissa carry propagates into the exp field
			half++
		}
		return half
	}
}
