package decoder

import "github.com/townsendmerino/goinfer/decoder/quant"

// FP8 E4M3 (OCP "e4m3fn": finite, no infinities), the weight dtype of DeepSeek V3/V4 and Qwen3-FP8 safetensors
// checkpoints, where it appears as dtype "F8_E4M3". The implementation lives in package decoder/quant; these
// package-private wrappers forward to it.

// fp8E4M3ToF32 decodes one e4m3fn byte. NaN maps to NaN; there are no infinities.
func fp8E4M3ToF32(b byte) float32 {
	return quant.FP8E4M3ToF32(b)
}

// fp8E4M3Table is the decode as a 256-entry lookup, built once.
var fp8E4M3Table = quant.FP8E4M3Table

// fp8DequantBlocked expands a block-quantized fp8 weight matrix into f32.
func fp8DequantBlocked(w []byte, scales []float32, rows, cols, blockR, blockC int) ([]float32, error) {
	return quant.FP8DequantBlocked(w, scales, rows, cols, blockR, blockC)
}
