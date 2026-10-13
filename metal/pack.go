//go:build darwin

package metal

import (
	"github.com/townsendmerino/goinfer/internal/f16"
	"github.com/townsendmerino/goinfer/metal/kernels"
)

func packW4A8Row(row []float32) (words []uint32, scales []float32) { return kernels.PackW4A8Row(row) }

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
// .giw stores scales as binary16, and the kernels must see the same bits whether a scale came from the file or
// was converted here.
func f32ToF16(f float32) uint16 { return f16.Bits(f) }
