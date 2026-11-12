package decoder

import (
	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder/quant"
)

// MLX quantized safetensors checkpoints (mlx-lm).
// The tensor reconstruction implementation lives in package decoder/quant;
// these package-private wrappers forward to it, while loadSwitchMLPExperts
// integrates the reconstructed 3-D expert slices with decoder's LayerWeights.

func mlxDequantAffineRow(words []uint32, scales, biases []float32, in, bits, groupSize int, dst []float32) error {
	return quant.MLXDequantAffineRow(words, scales, biases, in, bits, groupSize, dst)
}

func mlxDequantNVFP4Row(words []uint32, scaleBytes []byte, in int, dst []float32) error {
	return quant.MLXDequantNVFP4Row(words, scaleBytes, in, dst)
}

func mlxDequantMXFP4Row(words []uint32, scaleBytes []byte, in int, dst []float32) error {
	return quant.MLXDequantMXFP4Row(words, scaleBytes, in, dst)
}

func reconstructSafetensorsMat2D(st *embed.SafetensorsFile, qc *quantConfig, name string, out, in int) (data []float32, fresh bool, err error) {
	return quant.ReconstructMat2D(st, qc, name, out, in)
}

// loadSwitchMLPExperts loads MLX's 3-D stacked MoE experts (`mlp.switch_mlp.{gate_proj,up_proj,down_proj}`)
// one expert at a time through `forExperts`.
func loadSwitchMLPExperts(st *embed.SafetensorsFile, switchPrefix string, nExpert, inter, hidden int, qMode quantMode, needCanonical, skipRow4 bool, sink *giwWriter) ([]expertWeights, error) {
	gateSlice, err := quant.MLXExpertSlice3D(st, switchPrefix+".gate_proj", nExpert, inter, hidden)
	if err != nil {
		return nil, err
	}
	upSlice, err := quant.MLXExpertSlice3D(st, switchPrefix+".up_proj", nExpert, inter, hidden)
	if err != nil {
		return nil, err
	}
	downSlice, err := quant.MLXExpertSlice3D(st, switchPrefix+".down_proj", nExpert, hidden, inter)
	if err != nil {
		return nil, err
	}
	q := func(data []float32, rows, cols int) linalg.WeightMat {
		return quantizeLayerWM(linalg.WrapF32(data, rows, cols), qMode, needCanonical, skipRow4)
	}
	experts := make([]expertWeights, nExpert)
	err = forExperts(nExpert, sink, func(e int) error {
		gData, err := gateSlice(e)
		if err != nil {
			return err
		}
		uData, err := upSlice(e)
		if err != nil {
			return err
		}
		dData, err := downSlice(e)
		if err != nil {
			return err
		}
		experts[e] = expertWeights{
			Gate: q(gData, inter, hidden),
			Up:   q(uData, inter, hidden),
			Down: q(dData, hidden, inter),
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return experts, nil
}
