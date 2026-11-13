package decoder

import (
	"encoding/json"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/goinfer/decoder/quant"
)

// GPTQ (safetensors-resident int4) and checkpoint quantization configuration.
// The implementation lives in package decoder/quant; these package-private aliases
// and wrappers forward to it.

// quantConfig is the resolved quantization_config for a pre-quantized safetensors checkpoint.
type quantConfig = quant.Config

// parseCheckpointQuantConfig resolves a safetensors checkpoint's quantization
// metadata from either `quantization_config` (HF / GPTQ / AWQ / FP8 / NVFP4 / MLX)
// or top-level `quantization` (MLX).
func parseCheckpointQuantConfig(cfg *Config) (*quantConfig, error) {
	return quant.ParseCheckpointConfig(cfg.QuantizationConfig, cfg.Quantization)
}

// parseQuantConfig reads config.json's quantization_config (or MLX's quantization).
// Returns nil for a full-precision checkpoint (absent/null).
func parseQuantConfig(raw json.RawMessage) (*quantConfig, error) {
	return quant.ParseConfig(raw)
}

// gptqReconstruct dequantizes one GPTQ linear (named base, e.g.
// "model.layers.0.self_attn.q_proj") to a [out, in] row-major f32 matrix.
func gptqReconstruct(st *embed.SafetensorsFile, base string, in, out int) ([]float32, error) {
	return quant.GPTQReconstruct(st, base, in, out)
}
