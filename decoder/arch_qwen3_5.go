package decoder

import (
	"fmt"
	"math"
)

// qwen35DenseArchitecture expresses Qwen3.8 (model_type qwen3_5): the same Gated-DeltaNet / softmax 3:1 hybrid as qwen3_5_moe and
// qwen3_next, with a plain SwiGLU where they have a router. A dense sibling, not a new family: the DeltaNet math, the double-width gated
// q_proj, the per-head QK-norm, the partial RoPE and the hybrid cache are all shared. Checked against the released checkpoint and
// transformers' models/qwen3_5, not inherited by resemblance:
//
//   - head_dim is 256 with hidden 5120 and 24 heads, so nH·hd = 6144 ≠ hidden. Never derive head_dim from hidden for this family.
//   - RMSAddOne is TRUE: Qwen3_5RMSNorm is `output * (1.0 + weight)` with weight zero-init, as its MoE siblings.
//   - The DeltaNet projections ship as FOUR separate tensors (in_proj_qkv / in_proj_z / in_proj_a / in_proj_b), qwen3_5_moe's packing, not
//     qwen3_next's fused in_proj_qkvz/in_proj_ba, so FusedDeltaNetProj stays false.
//   - The attention output gate is torch.sigmoid(gate) whatever the config's output_gate_type "swish" says: the modeling code does not
//     consult it.
//   - m-RoPE: the text config carries mrope_section with mrope_interleaved true. For text it reduces exactly to standard partial RoPE
//     (position_ids are expanded to three identical components, so apply_interleaved_mrope overwrites with identical values). ropeAt
//     consults MRopeSection only when a cache carries mropePos, which only GenerateQwenVL sets; parseMRopeFlat sets it for the image path,
//     and TestQwen35_textIdentityHashes pins the text path byte-identical.
//
// rope fields are NESTED under rope_parameters on this release, which parseRopeFlat reads.
func qwen35DenseArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateQwen35Dense(); err != nil {
		return nil, nil, err
	}
	spec, partialRotary, err := parseRopeFlat(cfg.RopeParameters)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(qwen3_5): %w", err)
	}
	rotaryDim := 0
	if partialRotary > 0 && partialRotary < 1 {
		rotaryDim = int(partialRotary * float64(cfg.HeadDim))
	}
	arch := &Architecture{
		Name:             "qwen3_5",
		HiddenDim:        cfg.HiddenDim,
		NumLayers:        cfg.NumLayers,
		NumHeads:         cfg.NumHeads,
		NumKVHeads:       cfg.NumKVHeads,
		HeadDim:          cfg.HeadDim,
		IntermediateDim:  cfg.IntermediateDim, // the DENSE SwiGLU width — load-bearing here, unlike the MoE sibling
		VocabSize:        cfg.VocabSize,
		Norm:             NormRMS,
		RMSAddOne:        true,
		NormEps:          cfg.RMSNormEps,
		NormPlacement:    NormPre2,
		Act:              ActSiLU,
		MoE:              nil, // the one structural difference; the program's FFN branches on it
		QKNorm:           true,
		AttnScale:        math.Pow(float64(cfg.HeadDim), -0.5),
		layerIsGlobal:    cfg.IsGlobalLayer,
		layerIsLinear:    cfg.IsLinearLayer,
		RoPELocalBase:    spec.base,
		RoPEGlobalBase:   spec.base,
		ropeScaling:      spec.scaling,
		ropeScalingLocal: spec.scaling,
		RotaryDim:        rotaryDim,
		EmbedScale:       0,
		TiedLMHead:       false,
		qwen35: &qwen35Params{
			ConvKernel:    cfg.LinearConvKernelDim,
			KeyHeadDim:    cfg.LinearKeyHeadDim,
			ValueHeadDim:  cfg.LinearValueHeadDim,
			NumKeyHeads:   cfg.LinearNumKeyHeads,
			NumValueHeads: cfg.LinearNumValueHeads,
			AttnGate:      true, // qwen3.5's own double-width [query‖gate] q_proj — the dense sibling shares qwen3_5_moe's attention layer
		},
	}
	arch.MRopeSection, arch.MRopeInterleaved = parseMRopeFlat(cfg.RopeParameters)
	return arch, &qwen35DenseTensorSchema, nil
}
