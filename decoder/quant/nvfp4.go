package quant

import (
	"fmt"
	"math"
	"strings"

	"github.com/townsendmerino/aikit/embed"
)

const (
	NVFP4BlockElems  = 16
	NVFP4PackedBytes = 8
)

// NVFP4E2M1ToF32 decodes a 4-bit FP4 E2M1 nibble (bits [3:0]: 1 sign, 2 exponent
// with bias 1, 1 mantissa) into float32.
func NVFP4E2M1ToF32(nibble byte) float32 {
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

// NVFP4E2M1Table is the 16-entry E2M1 lookup table used in the hot dequant loops.
var NVFP4E2M1Table = func() [16]float32 {
	var t [16]float32
	for i := range t {
		t[i] = NVFP4E2M1ToF32(byte(i))
	}
	return t
}()

// NVFP4DequantRow dequantizes one row of in NVFP4 elements (len(packed) == in/2,
// len(scales) == in/16) into dst[:in].
func NVFP4DequantRow(packed []byte, scales []float32, globalScale float32, in int, dst []float32) error {
	if in <= 0 || in%NVFP4BlockElems != 0 {
		return fmt.Errorf("decoder(nvfp4): row width %d is not a positive multiple of %d", in, NVFP4BlockElems)
	}
	nBlocks := in / NVFP4BlockElems
	if len(packed) != nBlocks*NVFP4PackedBytes {
		return fmt.Errorf("decoder(nvfp4): %d packed bytes for %d blocks (want %d)", len(packed), nBlocks, nBlocks*NVFP4PackedBytes)
	}
	if len(scales) != nBlocks {
		return fmt.Errorf("decoder(nvfp4): %d scales for %d blocks", len(scales), nBlocks)
	}
	if len(dst) < in {
		return fmt.Errorf("decoder(nvfp4): dst len %d < %d", len(dst), in)
	}
	for g := range nBlocks {
		s := scales[g] * globalScale
		qs := packed[g*NVFP4PackedBytes : (g+1)*NVFP4PackedBytes]
		out := dst[g*NVFP4BlockElems : (g+1)*NVFP4BlockElems]
		for j := range NVFP4PackedBytes {
			b := qs[j]
			out[2*j] = NVFP4E2M1Table[b&0x0F] * s
			out[2*j+1] = NVFP4E2M1Table[b>>4] * s
		}
	}
	return nil
}

// NVFP4ScaleTensorF32 reads an NVFP4 block-scale tensor (F8_E4M3 or F32/BF16/F16)
// and widens it to []float32, validating its shape against want.
func NVFP4ScaleTensorF32(st *embed.SafetensorsFile, name string, want ...int) ([]float32, error) {
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
			out[i] = FP8E4M3Table[b]
		}
		return out, nil
	}
	return st.TensorF32(name, want...)
}

// NVFP4GlobalScale reads the optional <prefix>.weight_scale_2 per-tensor F32 scale,
// returning 1.0 when the tensor is absent.
func NVFP4GlobalScale(st *embed.SafetensorsFile, prefix string) (float32, error) {
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

// NVFP4RowReader2D validates a 2-D NVFP4 quantized tensor (<prefix>.weight U8 [out, in/2]
// + <prefix>.weight_scale + optional <prefix>.weight_scale_2) and returns a closure
// that dequantizes row r (0 <= r < out) into dst[:in] without materializing the full
// [out, in] matrix at once.
func NVFP4RowReader2D(st *embed.SafetensorsFile, name string, out, in int) (func(r int, dst []float32) error, error) {
	wT, err := st.Tensor(name)
	if err != nil {
		return nil, fmt.Errorf("nvfp4 %q: %w", name, err)
	}
	if wT.DType != "U8" {
		return nil, fmt.Errorf("nvfp4 %q: dtype %s, want U8 (or F32/BF16/F16 for unquantized layers)", name, wT.DType)
	}
	if in <= 0 || out <= 0 || in%NVFP4BlockElems != 0 {
		return nil, fmt.Errorf("nvfp4 %q: out=%d in=%d; in must be a positive multiple of %d", name, out, in, NVFP4BlockElems)
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
	scaleCols := in / NVFP4BlockElems
	scales, err := NVFP4ScaleTensorF32(st, prefix+".weight_scale", out, scaleCols)
	if err != nil {
		return nil, fmt.Errorf("nvfp4 %q scales: %w", prefix+".weight_scale", err)
	}
	globalScale, err := NVFP4GlobalScale(st, prefix)
	if err != nil {
		return nil, err
	}
	return func(r int, dst []float32) error {
		pRow := raw[r*packedCols : (r+1)*packedCols]
		sRow := scales[r*scaleCols : (r+1)*scaleCols]
		return NVFP4DequantRow(pRow, sRow, globalScale, in, dst)
	}, nil
}

// NVFP4Reconstruct reads one NVFP4 linear projection from a safetensors checkpoint
// and returns it as a row-major [out, in] []float32. Unquantized tensors (F32/BF16/F16)
// fall back to st.TensorF32(name, out, in).
func NVFP4Reconstruct(st *embed.SafetensorsFile, name string, in, out int) ([]float32, error) {
	wT, err := st.Tensor(name)
	if err != nil {
		return nil, fmt.Errorf("nvfp4 %q: %w", name, err)
	}
	if wT.DType == "F32" || wT.DType == "BF16" || wT.DType == "F16" {
		return st.TensorF32(name, out, in)
	}
	rowInto, err := NVFP4RowReader2D(st, name, out, in)
	if err != nil {
		return nil, err
	}
	dst := make([]float32, out*in)
	for r := range out {
		if err := rowInto(r, dst[r*in:(r+1)*in]); err != nil {
			return nil, fmt.Errorf("nvfp4 %q row %d: %w", name, r, err)
		}
	}
	return dst, nil
}

// NVFP4Reconstructor implements Reconstructor and RowStreamReconstructor for NVIDIA NVFP4 tensors.
type NVFP4Reconstructor struct{}

func (NVFP4Reconstructor) Method() string { return "nvfp4" }

func (NVFP4Reconstructor) ReconstructMat2D(st *embed.SafetensorsFile, name string, out, in int) ([]float32, error) {
	return NVFP4Reconstruct(st, name, in, out)
}

func (NVFP4Reconstructor) RowReader2D(st *embed.SafetensorsFile, name string, out, in int) (func(r int, dst []float32) error, error) {
	return NVFP4RowReader2D(st, name, out, in)
}
