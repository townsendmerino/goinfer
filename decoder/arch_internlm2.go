package decoder

import (
	"fmt"
	"math"
)

// internlm2Architecture expresses InternLM2 (model_type internlm2). The DESCRIPTOR is llama's
// — same norms, same SwiGLU, same single-base RoPE, no biases, no QK-norm — because the math
// is llama's. What differs is entirely on the loader side (renamed tensors, a grouped fused
// wqkv), which is why this returns llama's shape with its OWN tensor schema rather than
// aliasing llamaArchitecture outright: an alias would send the loader looking for
// self_attn.q_proj in a checkpoint that has never heard of it.
//
// rope_scaling is "dynamic" on the released checkpoints, which parseRopeScaling accepts as no
// scaling (exact within the trained window; see its comment for the boundary caveat).
func internlm2Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := backfillFlatRope(cfg, "internlm2"); err != nil {
		return nil, nil, err
	}
	if err := cfg.validateLlama(); err != nil {
		return nil, nil, fmt.Errorf("decoder(internlm2): %w", err)
	}
	scaling, err := parseRopeScaling(cfg.RopeScaling)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(internlm2): %w", err)
	}
	hd := cfg.headDim()
	return &Architecture{
		Name:            "internlm2",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		AttnScale:       math.Pow(float64(hd), -0.5),
		RoPELocalBase:   cfg.RoPEGlobalBase,
		RoPEGlobalBase:  cfg.RoPEGlobalBase,
		ropeScaling:     scaling,
		TiedLMHead:      false,
	}, &internlm2TensorSchema, nil
}
