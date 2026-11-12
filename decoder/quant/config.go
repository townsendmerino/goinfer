// Package quant parses pre-quantized safetensors checkpoint configurations and
// reconstructs quantized weight tensors (FP8 E4M3, GPTQ, AWQ, NVIDIA NVFP4, and
// MLX affine/nvfp4/mxfp4) to row-major float32.
//
// This package is a Tier 3 leaf subpackage of decoder: it depends only on the Go
// standard library and aikit/embed, and never imports package decoder.
package quant

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/townsendmerino/aikit/embed"
)

// Reconstructor dequantizes a named 2-D [out, in] weight tensor from a safetensors
// checkpoint into row-major float32.
type Reconstructor interface {
	// Method returns the canonical quantization method name ("fp8", "gptq", "awq", "nvfp4", "mlx").
	Method() string
	// ReconstructMat2D reads and dequantizes name from st into a freshly allocated
	// [out, in] row-major float32 slice.
	ReconstructMat2D(st *embed.SafetensorsFile, name string, out, in int) ([]float32, error)
}

// RowStreamReconstructor is an optional extension of Reconstructor for formats
// that can dequantize individual rows r in [0, out) into dst[:in] without
// materializing the full [out, in] matrix at once (used when streaming large
// embedding tables during transcode).
type RowStreamReconstructor interface {
	Reconstructor
	RowReader2D(st *embed.SafetensorsFile, name string, out, in int) (func(r int, dst []float32) error, error)
}

// Config is the resolved quantization_config (or MLX top-level quantization) for
// a pre-quantized safetensors checkpoint.
type Config struct {
	Method    string // "gptq" | "awq" | "fp8" | "nvfp4" | "mlx"
	Bits      int
	GroupSize int
	DescAct   bool   // gptq act-order
	Sym       bool   // gptq symmetric
	Mode      string // mlx mode: "affine" | "nvfp4" | "mxfp4"
	// fp8 only: the 2-D weight block a single scale covers (config.json's
	// weight_block_size, [128,128] for DeepSeek V3/V4 and Qwen3-FP8).
	BlockR, BlockC int
}

// Reconstructor returns the Reconstructor for c, or nil when c is nil.
func (c *Config) Reconstructor() Reconstructor {
	if c == nil {
		return nil
	}
	switch c.Method {
	case "fp8":
		return FP8Reconstructor{BlockR: c.BlockR, BlockC: c.BlockC}
	case "nvfp4":
		return NVFP4Reconstructor{}
	case "mlx":
		return MLXReconstructor{}
	case "awq":
		return AWQReconstructor{}
	default:
		return GPTQReconstructor{}
	}
}

// TensorFormat classifies how an individual safetensors .weight tensor is packed on disk.
type TensorFormat uint8

const (
	FormatNone TensorFormat = iota
	FormatMLX
	FormatNVFP4
)

// ClassifyTensor inspects name and its companion scale tensor in st to classify
// whether the tensor is MLX (U32 + .scales), NVIDIA ModelOpt NVFP4 (U8 + .weight_scale),
// or a plain floating-point tensor (FormatNone).
func ClassifyTensor(st *embed.SafetensorsFile, name string) TensorFormat {
	prefix, ok := strings.CutSuffix(name, ".weight")
	if !ok {
		return FormatNone
	}
	wT, err := st.Tensor(name)
	if err != nil {
		return FormatNone
	}
	switch wT.DType {
	case "U32":
		if _, serr := st.Tensor(prefix + ".scales"); serr == nil {
			return FormatMLX
		}
	case "U8":
		if _, serr := st.Tensor(prefix + ".weight_scale"); serr == nil {
			return FormatNVFP4
		}
	}
	return FormatNone
}

// ForTensor returns the Reconstructor for name in st: the checkpoint-level
// Reconstructor when qc is non-nil, else a per-tensor Reconstructor when
// ClassifyTensor detects MLX or NVFP4 packing, else nil for unquantized tensors.
func ForTensor(st *embed.SafetensorsFile, qc *Config, name string) Reconstructor {
	if qc != nil {
		return qc.Reconstructor()
	}
	switch ClassifyTensor(st, name) {
	case FormatMLX:
		return MLXReconstructor{}
	case FormatNVFP4:
		return NVFP4Reconstructor{}
	default:
		return nil
	}
}

// ReconstructMat2D reads a 2-D [out, in] weight matrix from st, dispatching
// through ForTensor(st, qc, name) when quantized or st.TensorF32(name, out, in)
// when unquantized. fresh reports whether the returned slice is freshly allocated
// (true for every reconstructed tensor; false when st.TensorF32 may alias an
// on-disk F32 mapping).
func ReconstructMat2D(st *embed.SafetensorsFile, qc *Config, name string, out, in int) (data []float32, fresh bool, err error) {
	if r := ForTensor(st, qc, name); r != nil {
		data, err = r.ReconstructMat2D(st, name, out, in)
		return data, true, err
	}
	data, err = st.TensorF32(name, out, in)
	return data, false, err
}

// ParseCheckpointConfig resolves a safetensors checkpoint's quantization metadata
// from either quantization_config (HF / GPTQ / AWQ / FP8 / NVFP4 / MLX) or
// top-level quantization (MLX).
func ParseCheckpointConfig(quantizationConfig, quantization json.RawMessage) (*Config, error) {
	if len(quantizationConfig) > 0 && string(quantizationConfig) != "null" {
		return ParseConfig(quantizationConfig)
	}
	if len(quantization) > 0 && string(quantization) != "null" {
		return ParseConfig(quantization)
	}
	return nil, nil
}

// ParseConfig reads config.json's quantization_config (or MLX's quantization).
// Returns nil for a full-precision checkpoint (absent/null).
func ParseConfig(raw json.RawMessage) (*Config, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var obj struct {
		QuantMethod string `json:"quant_method"`
		QuantAlgo   string `json:"quant_algo"` // NVIDIA ModelOpt: "NVFP4"
		Bits        int    `json:"bits"`
		GroupSize   int    `json:"group_size"`
		Mode        string `json:"mode"` // MLX: "affine" | "nvfp4" | "mxfp4"
		DescAct     bool   `json:"desc_act"`
		Sym         *bool  `json:"sym"`
		// checkpoint_format distinguishes GPTQModel's v2 export (zero-points without the v1 "+1" bias) from v1; ParseConfig
		// refuses v2.
		CheckpointFormat string `json:"checkpoint_format"`
		Fmt              string `json:"fmt"`               // fp8: "e4m3" | "e5m2"
		WeightBlock      []int  `json:"weight_block_size"` // fp8: [blockR, blockC]
		ConfigGroups     map[string]struct {
			Weights struct {
				NumBits   int    `json:"num_bits"`
				Type      string `json:"type"`
				GroupSize int    `json:"group_size"`
			} `json:"weights"`
		} `json:"config_groups"`
	}
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, fmt.Errorf("quantization_config: %w", err)
	}
	// fp8 returns EARLY: it shares none of the gptq/awq shape below.
	if obj.QuantMethod == "fp8" {
		if obj.Fmt != "" && obj.Fmt != "e4m3" {
			return nil, fmt.Errorf("quantization_config(fp8): fmt %q unsupported (have: e4m3)", obj.Fmt)
		}
		if len(obj.WeightBlock) != 2 || obj.WeightBlock[0] <= 0 || obj.WeightBlock[1] <= 0 {
			return nil, fmt.Errorf("quantization_config(fp8): weight_block_size %v unsupported "+
				"(need a 2-element block, e.g. [128,128]); per-tensor/per-channel fp8 is a different layout", obj.WeightBlock)
		}
		return &Config{Method: "fp8", BlockR: obj.WeightBlock[0], BlockC: obj.WeightBlock[1]}, nil
	}
	// NVIDIA ModelOpt / compressed-tensors NVFP4.
	if obj.QuantMethod == "nvfp4" || obj.QuantMethod == "modelopt" {
		if obj.QuantMethod == "modelopt" && obj.QuantAlgo != "" && obj.QuantAlgo != "NVFP4" && obj.QuantAlgo != "nvfp4" {
			return nil, fmt.Errorf("quantization_config(modelopt): quant_algo %q unsupported (have: NVFP4)", obj.QuantAlgo)
		}
		for gName, cg := range obj.ConfigGroups {
			if cg.Weights.NumBits != 0 && cg.Weights.NumBits != 4 {
				return nil, fmt.Errorf("quantization_config(%s): group %q num_bits=%d unsupported (have: 4)", obj.QuantMethod, gName, cg.Weights.NumBits)
			}
			if cg.Weights.GroupSize != 0 && cg.Weights.GroupSize != 16 {
				return nil, fmt.Errorf("quantization_config(%s): group %q group_size=%d unsupported (have: 16)", obj.QuantMethod, gName, cg.Weights.GroupSize)
			}
		}
		return &Config{Method: "nvfp4", Bits: 4, GroupSize: 16}, nil
	}
	// MLX quantized safetensors (`quantization` or `quantization_config` with no quant_method or quant_method="mlx").
	if obj.QuantMethod == "" || obj.QuantMethod == "mlx" {
		mode := obj.Mode
		if mode == "" {
			mode = "affine"
		}
		bits := obj.Bits
		gs := obj.GroupSize
		switch mode {
		case "affine":
			if bits != 4 && bits != 8 {
				return nil, fmt.Errorf("quantization_config(mlx): %d-bit unsupported for mode %q (have: 4, 8)", bits, mode)
			}
			if gs != 32 && gs != 64 && gs != 128 {
				return nil, fmt.Errorf("quantization_config(mlx): group_size %d unsupported for mode %q (have: 32, 64, 128)", gs, mode)
			}
		case "nvfp4":
			if bits == 0 {
				bits = 4
			}
			if gs == 0 {
				gs = 16
			}
			if bits != 4 || gs != 16 {
				return nil, fmt.Errorf("quantization_config(mlx): mode=nvfp4 requires bits=4 group_size=16, got bits=%d group_size=%d", bits, gs)
			}
		case "mxfp4":
			if bits == 0 {
				bits = 4
			}
			if gs == 0 {
				gs = 32
			}
			if bits != 4 || gs != 32 {
				return nil, fmt.Errorf("quantization_config(mlx): mode=mxfp4 requires bits=4 group_size=32, got bits=%d group_size=%d", bits, gs)
			}
		default:
			return nil, fmt.Errorf("quantization_config(mlx): mode %q unsupported (have: affine, nvfp4, mxfp4)", mode)
		}
		return &Config{Method: "mlx", Bits: bits, GroupSize: gs, Mode: mode}, nil
	}
	if obj.QuantMethod != "gptq" && obj.QuantMethod != "awq" {
		return nil, fmt.Errorf("quantization_config: method %q unsupported (have: gptq, awq, fp8, nvfp4, mlx)", obj.QuantMethod)
	}
	if obj.Bits != 4 {
		return nil, fmt.Errorf("quantization_config(%s): %d-bit unsupported (have: 4-bit)", obj.QuantMethod, obj.Bits)
	}
	if obj.GroupSize <= 0 {
		return nil, fmt.Errorf("quantization_config(%s): group_size %d unsupported (need a positive group)", obj.QuantMethod, obj.GroupSize)
	}
	if obj.QuantMethod == "gptq" && obj.CheckpointFormat != "" && obj.CheckpointFormat != "gptq" {
		return nil, fmt.Errorf("quantization_config(gptq): checkpoint_format %q unsupported "+
			"(have: gptq / v1). This loader adds the v1 +1 zero-point bias unconditionally, so a "+
			"%s checkpoint would load one scale step low on EVERY weight without erroring",
			obj.CheckpointFormat, obj.CheckpointFormat)
	}
	sym := true
	if obj.Sym != nil {
		sym = *obj.Sym
	}
	return &Config{Method: obj.QuantMethod, Bits: obj.Bits, GroupSize: obj.GroupSize, DescAct: obj.DescAct, Sym: sym}, nil
}
