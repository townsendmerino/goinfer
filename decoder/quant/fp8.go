package quant

import (
	"fmt"
	"math"

	"github.com/townsendmerino/aikit/embed"
)

const (
	fp8E4M3Bias     = 7
	fp8E4M3MantBits = 3
)

// FP8E4M3ToF32 decodes one OCP e4m3fn byte into float32. NaN (0x7F/0xFF) maps to NaN;
// there are no infinities (max finite is 448 at 0x7E).
func FP8E4M3ToF32(b byte) float32 {
	sign := float64(1)
	if b&0x80 != 0 {
		sign = -1
	}
	exp := int((b >> fp8E4M3MantBits) & 0x0F)
	mant := int(b & 0x07)
	switch {
	case exp == 0x0F && mant == 0x07:
		return float32(math.NaN())
	case exp == 0:
		return float32(sign * math.Ldexp(float64(mant)/8, 1-fp8E4M3Bias))
	default:
		return float32(sign * math.Ldexp(1+float64(mant)/8, exp-fp8E4M3Bias))
	}
}

// FP8E4M3Table is the 256-entry lookup table for FP8 E4M3 (e4m3fn) bytes.
var FP8E4M3Table = func() [256]float32 {
	var t [256]float32
	for i := range t {
		t[i] = FP8E4M3ToF32(byte(i))
	}
	return t
}()

// FP8DequantBlocked expands a block-quantized fp8 weight matrix [rows, cols] into float32
// using the [ceil(rows/blockR), ceil(cols/blockC)] scale grid.
func FP8DequantBlocked(w []byte, scales []float32, rows, cols, blockR, blockC int) ([]float32, error) {
	if blockR <= 0 || blockC <= 0 {
		return nil, fmt.Errorf("decoder(fp8): block size %dx%d must be positive", blockR, blockC)
	}
	if len(w) != rows*cols {
		return nil, fmt.Errorf("decoder(fp8): %d payload bytes for a [%d,%d] tensor (want %d)", len(w), rows, cols, rows*cols)
	}
	sR, sC := (rows+blockR-1)/blockR, (cols+blockC-1)/blockC
	if len(scales) != sR*sC {
		return nil, fmt.Errorf("decoder(fp8): %d scales for a [%d,%d] tensor at %dx%d blocks (want %d = %dx%d)",
			len(scales), rows, cols, blockR, blockC, sR*sC, sR, sC)
	}
	out := make([]float32, rows*cols)
	for r := range rows {
		srow := (r / blockR) * sC
		wrow, orow := r*cols, r*cols
		for c := range cols {
			out[orow+c] = FP8E4M3Table[w[wrow+c]] * scales[srow+c/blockC]
		}
	}
	return out, nil
}

// FP8Reconstruct reads one block-quantized fp8 linear from a safetensors checkpoint
// and returns it as a [out, in] row-major float32 matrix.
func FP8Reconstruct(st *embed.SafetensorsFile, name string, in, out, blockR, blockC int) ([]float32, error) {
	wT, err := st.Tensor(name)
	if err != nil {
		return nil, fmt.Errorf("fp8 %q: %w", name, err)
	}
	if wT.DType != "F8_E4M3" {
		return nil, fmt.Errorf("fp8 %q: dtype %s, want F8_E4M3", name, wT.DType)
	}
	raw, err := wT.Uint8s()
	if err != nil {
		return nil, fmt.Errorf("fp8 %q payload: %w", name, err)
	}
	if len(wT.Shape) != 2 || wT.Shape[0] != out || wT.Shape[1] != in {
		return nil, fmt.Errorf("fp8 %q: shape %v, arch expects [%d,%d] — a transposed tensor has "+
			"the same element count and would load silently", name, wT.Shape, out, in)
	}
	if len(raw) != out*in {
		return nil, fmt.Errorf("fp8 %q: %d payload bytes for [%d,%d]", name, len(raw), out, in)
	}
	sName := name + "_scale_inv"
	sR, sC := (out+blockR-1)/blockR, (in+blockC-1)/blockC
	scales, err := st.TensorF32(sName, sR, sC)
	if err != nil {
		return nil, fmt.Errorf("fp8 %q scales: %w", sName, err)
	}
	return FP8DequantBlocked(raw, scales, out, in, blockR, blockC)
}

// FP8Reconstructor implements Reconstructor for block-quantized FP8 E4M3 tensors.
type FP8Reconstructor struct {
	BlockR, BlockC int
}

func (FP8Reconstructor) Method() string { return "fp8" }

func (r FP8Reconstructor) ReconstructMat2D(st *embed.SafetensorsFile, name string, out, in int) ([]float32, error) {
	return FP8Reconstruct(st, name, in, out, r.BlockR, r.BlockC)
}
