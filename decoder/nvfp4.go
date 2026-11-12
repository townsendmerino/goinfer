package decoder

import (
	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/goinfer/decoder/quant"
)

// NVFP4 (NVIDIA FP4 / ModelOpt): 16-element micro-blocks of 4-bit E2M1 weights
// paired with an FP8 E4M3 (e4m3fn) per-block scale and an optional per-tensor
// F32 global scale (weight_scale_2). The implementation lives in package
// decoder/quant; these package-private wrappers forward to it.
const (
	nvfp4BlockElems  = quant.NVFP4BlockElems
	nvfp4PackedBytes = quant.NVFP4PackedBytes
)

func nvfp4E2M1ToF32(nibble byte) float32 {
	return quant.NVFP4E2M1ToF32(nibble)
}

var nvfp4E2M1Table = quant.NVFP4E2M1Table

func nvfp4DequantRow(packed []byte, scales []float32, globalScale float32, in int, dst []float32) error {
	return quant.NVFP4DequantRow(packed, scales, globalScale, in, dst)
}

func nvfp4ScaleTensorF32(st *embed.SafetensorsFile, name string, want ...int) ([]float32, error) {
	return quant.NVFP4ScaleTensorF32(st, name, want...)
}

func nvfp4GlobalScale(st *embed.SafetensorsFile, prefix string) (float32, error) {
	return quant.NVFP4GlobalScale(st, prefix)
}

func nvfp4Reconstruct(st *embed.SafetensorsFile, name string, in, out int) ([]float32, error) {
	return quant.NVFP4Reconstruct(st, name, in, out)
}
