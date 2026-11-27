package wbuild

import (
	"github.com/townsendmerino/goinfer/decoder/config"
	"github.com/townsendmerino/goinfer/decoder/quant"
)

// GPTQ (safetensors-resident int4) and checkpoint quantization configuration.
// The implementation lives in package decoder/quant; these package-private aliases
// and wrappers forward to it.

// quantConfig is the resolved quantization_config for a pre-quantized safetensors checkpoint.

// ParseCheckpointQuantConfig resolves a safetensors checkpoint's quantization
// metadata from either `quantization_config` (HF / GPTQ / AWQ / FP8 / NVFP4 / MLX)
// or top-level `quantization` (MLX).
func ParseCheckpointQuantConfig(cfg *config.Config) (*quant.Config, error) {
	return quant.ParseCheckpointConfig(cfg.QuantizationConfig, cfg.Quantization)
}
