package decoder

import (
	"fmt"
	"math"
	"strings"

	"github.com/townsendmerino/aikit/embed"
)

// NVFP4 (NVIDIA FP4 / ModelOpt): 16-element micro-blocks of 4-bit E2M1 weights
// paired with an FP8 E4M3 (e4m3fn) per-block scale and an optional per-tensor
// F32 global scale (weight_scale_2).
//
// In NVIDIA ModelOpt / compressed-tensors safetensors checkpoints:
//   - <prefix>.weight          U8      [out, in/2]   — two E2M1 nibbles per byte (low = even col 2j, high = odd col 2j+1)
//   - <prefix>.weight_scale    F8_E4M3 [out, in/16]  — one E4M3fn scale per 16-element micro-block along in
//   - <prefix>.weight_scale_2  F32     [] or [1]     — optional per-tensor second-level scale (1.0 when absent)
//   - <prefix>.input_scale     F32     [] or [1]     — activation scale, unused for weight reconstruction
//
// Reconstruction for row r, micro-block g = c / 16, byte j = (c % 16) / 2:
//
//	w[r, 2k]   = nvfp4E2M1Table[b & 0x0F] * (fp8E4M3Table[weight_scale[r, g]] * weight_scale_2)
//	w[r, 2k+1] = nvfp4E2M1Table[b >> 4]   * (fp8E4M3Table[weight_scale[r, g]] * weight_scale_2)
const (
	nvfp4BlockElems  = 16
	nvfp4PackedBytes = 8
)

// nvfp4E2M1ToF32 decodes a 4-bit FP4 E2M1 nibble (bits [3:0]: 1 sign, 2 exponent with bias 1, 1 mantissa).
// There are no infinities or NaNs:
//
//	0b0000 -> +0.0   0b1000 -> -0.0
//	0b0001 -> +0.5   0b1001 -> -0.5  (subnormal: 2^(1-1) * (0 + 1/2) = 0.5)
//	0b0010 -> +1.0   0b1010 -> -1.0
//	0b0011 -> +1.5   0b1011 -> -1.5
//	0b0100 -> +2.0   0b1100 -> -2.0
//	0b0101 -> +3.0   0b1101 -> -3.0
//	0b0110 -> +4.0   0b1110 -> -4.0
//	0b0111 -> +6.0   0b1111 -> -6.0
func nvfp4E2M1ToF32(nibble byte) float32 {
	sign := float64(1)
	if nibble&0x08 != 0 {
		sign = -1
	}
	exp := int((nibble >> 1) & 0x03)
	mant := int(nibble & 0x01)
	if exp == 0 {
		return float32(sign * (float64(mant) * 0.5))
	}
	return float32(sign * math.Ldexp(1+float64(mant)*0.5, exp-1))
}

// nvfp4E2M1Table is the 16-entry E2M1 lookup table used in the hot dequant loops.
var nvfp4E2M1Table = func() [16]float32 {
	var t [16]float32
	for i := range t {
		t[i] = nvfp4E2M1ToF32(byte(i))
	}
	return t
}()

// nvfp4DequantRow dequantizes one row of `in` NVFP4 elements (`len(packed) == in/2`,
// `len(scales) == in/16`) into `dst[:in]`.
func nvfp4DequantRow(packed []byte, scales []float32, globalScale float32, in int, dst []float32) error {
	if in <= 0 || in%nvfp4BlockElems != 0 {
		return fmt.Errorf("decoder(nvfp4): row width %d is not a positive multiple of %d", in, nvfp4BlockElems)
	}
	nBlocks := in / nvfp4BlockElems
	if len(packed) != nBlocks*nvfp4PackedBytes {
		return fmt.Errorf("decoder(nvfp4): %d packed bytes for %d blocks (want %d)", len(packed), nBlocks, nBlocks*nvfp4PackedBytes)
	}
	if len(scales) != nBlocks {
		return fmt.Errorf("decoder(nvfp4): %d scales for %d blocks", len(scales), nBlocks)
	}
	if len(dst) < in {
		return fmt.Errorf("decoder(nvfp4): dst len %d < %d", len(dst), in)
	}
	for g := range nBlocks {
		s := scales[g] * globalScale
		qs := packed[g*nvfp4PackedBytes : (g+1)*nvfp4PackedBytes]
		out := dst[g*nvfp4BlockElems : (g+1)*nvfp4BlockElems]
		for j := range nvfp4PackedBytes {
			b := qs[j]
			out[2*j] = nvfp4E2M1Table[b&0x0F] * s
			out[2*j+1] = nvfp4E2M1Table[b>>4] * s
		}
	}
	return nil
}

// nvfp4ScaleTensorF32 reads an NVFP4 block-scale tensor (`F8_E4M3` or `F32`/`BF16`/`F16`)
// and widens it to `[]float32`, validating its shape against `want`.
func nvfp4ScaleTensorF32(st *embed.SafetensorsFile, name string, want ...int) ([]float32, error) {
	t, err := st.Tensor(name)
	if err != nil {
		return nil, err
	}
	if len(want) > 0 {
		if len(t.Shape) != len(want) {
			return nil, fmt.Errorf("safetensors: tensor %q shape %v != want %v", name, t.Shape, want)
		}
		for i := range want {
			if t.Shape[i] != want[i] {
				return nil, fmt.Errorf("safetensors: tensor %q shape %v != want %v", name, t.Shape, want)
			}
		}
	}
	if t.DType == "F8_E4M3" || t.DType == "U8" {
		raw, err := t.Uint8s()
		if err != nil {
			return nil, err
		}
		out := make([]float32, len(raw))
		for i, b := range raw {
			out[i] = fp8E4M3Table[b]
		}
		return out, nil
	}
	return st.TensorF32(name, want...)
}

// nvfp4GlobalScale reads the optional `<prefix>.weight_scale_2` per-tensor F32 scale,
// returning 1.0 when the tensor is absent.
func nvfp4GlobalScale(st *embed.SafetensorsFile, prefix string) (float32, error) {
	name := prefix + ".weight_scale_2"
	t, err := st.Tensor(name)
	if err != nil {
		return 1.0, nil
	}
	if len(t.Shape) > 1 || (len(t.Shape) == 1 && t.Shape[0] != 1) {
		return 0, fmt.Errorf("nvfp4 %q: shape %v, want scalar [] or [1]", name, t.Shape)
	}
	vals, err := st.TensorF32(name)
	if err != nil {
		return 0, fmt.Errorf("nvfp4 %q: %w", name, err)
	}
	if len(vals) != 1 {
		return 0, fmt.Errorf("nvfp4 %q: got %d values, want 1", name, len(vals))
	}
	return vals[0], nil
}

// nvfp4Reconstruct reads one NVFP4 linear projection from a safetensors checkpoint
// and returns it as a row-major `[out, in]` `[]float32`. If the named tensor is stored
// unquantized (`F32`/`BF16`/`F16`, e.g. a layer listed under `ignore` in a ModelOpt
// config), it delegates to `st.TensorF32(name, out, in)`.
func nvfp4Reconstruct(st *embed.SafetensorsFile, name string, in, out int) ([]float32, error) {
	wT, err := st.Tensor(name)
	if err != nil {
		return nil, fmt.Errorf("nvfp4 %q: %w", name, err)
	}
	if wT.DType == "F32" || wT.DType == "BF16" || wT.DType == "F16" {
		return st.TensorF32(name, out, in)
	}
	if wT.DType != "U8" {
		return nil, fmt.Errorf("nvfp4 %q: dtype %s, want U8 (or F32/BF16/F16 for unquantized layers)", name, wT.DType)
	}
	if in <= 0 || out <= 0 || in%nvfp4BlockElems != 0 {
		return nil, fmt.Errorf("nvfp4 %q: out=%d in=%d; in must be a positive multiple of %d", name, out, in, nvfp4BlockElems)
	}
	packedCols := in / 2
	if len(wT.Shape) != 2 || wT.Shape[0] != out || wT.Shape[1] != packedCols {
		return nil, fmt.Errorf("nvfp4 %q: shape %v, want [%d,%d] for logical [%d,%d]", name, wT.Shape, out, packedCols, out, in)
	}
	raw, err := wT.Uint8s()
	if err != nil {
		return nil, fmt.Errorf("nvfp4 %q payload: %w", name, err)
	}
	prefix := strings.TrimSuffix(name, ".weight")
	scaleCols := in / nvfp4BlockElems
	scales, err := nvfp4ScaleTensorF32(st, prefix+".weight_scale", out, scaleCols)
	if err != nil {
		return nil, fmt.Errorf("nvfp4 %q scales: %w", prefix+".weight_scale", err)
	}
	globalScale, err := nvfp4GlobalScale(st, prefix)
	if err != nil {
		return nil, err
	}
	dst := make([]float32, out*in)
	for r := range out {
		pRow := raw[r*packedCols : (r+1)*packedCols]
		sRow := scales[r*scaleCols : (r+1)*scaleCols]
		if err := nvfp4DequantRow(pRow, sRow, globalScale, in, dst[r*in:(r+1)*in]); err != nil {
			return nil, fmt.Errorf("nvfp4 %q row %d: %w", name, r, err)
		}
	}
	return dst, nil
}
