package decoder

import (
	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/goinfer/decoder/quant"
)

// AWQ (safetensors-resident int4) — the AutoAWQ "GEMM" packing.
// The implementation lives in package decoder/quant; this package-private wrapper forwards to it.

// awqReconstruct dequantizes one AWQ linear (named base) to a [out, in]
// row-major f32 matrix.
func awqReconstruct(st *embed.SafetensorsFile, base string, in, out int) ([]float32, error) {
	return quant.AWQReconstruct(st, base, in, out)
}
