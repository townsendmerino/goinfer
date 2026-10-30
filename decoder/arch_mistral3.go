package decoder

import (
	"encoding/json"
	"fmt"
	"math"
)

// ministral3Architecture expresses Ministral 3 (model_type "mistral3", the OUTER Mistral3ForConditionalGeneration wrapper's type: loadConfig's
// text_config flattening re-applies it last over the nested text_config's own "ministral3", so it is the registry key a real checkpoint
// resolves under): Mistral's GQA skeleton (same tensor names) with two real deltas, checked against the released config:
//
//  1. sliding_window is null on the release, which mistralArchitecture already treats as full attention: no new code.
//  2. rope_parameters is rope_type "yarn" plus a load-bearing llama_4_scaling_beta: get_llama_4_attn_scale multiplies the QUERY by
//     1 + beta·ln(1 + floor(pos/original_max_position_embeddings)) AFTER RoPE, on every layer. That is the attnTemp/floorScale formula
//     llama4Architecture implements, but Llama4 applies it instead of RoPE on NoPE layers only, never combined with RoPE. It is the generic
//     Architecture.AttnTempBeta/AttnTempOrigMaxPos pair, wired into the generic attention of both program walkers; every other family leaves
//     both at the zero-value no-op.
//
// mscale/mscale_all_dim (both 1.0 on the release) are DeepSeek's spelling of the YaRN attention_factor, not the generic `attention_factor`
// key parseRopeScaling reads. Left unhandled, its default (0.1·ln(16)+1 ≈ 1.277) would silently override the correct 1.0, so it is
// overridden the way deepseekArchitecture does.
func ministral3Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	// backfillFlatRope BEFORE validateLlama: the released checkpoints carry ONLY a nested
	// rope_parameters (no flat rope_theta), and validateLlama itself requires RoPEGlobalBase > 0
	// — the same ordering mistralArchitecture uses, for the same reason.
	if err := backfillFlatRope(cfg, "mistral3"); err != nil {
		return nil, nil, err
	}
	if err := cfg.validateLlama(); err != nil {
		return nil, nil, err
	}
	base, scaling, err := ropeBaseFlatOrNested(cfg, "mistral3")
	if err != nil {
		return nil, nil, err
	}
	if base <= 0 {
		return nil, nil, fmt.Errorf("decoder(mistral3): rope_theta must be >0, got %v", base)
	}
	if scaling != nil && scaling.kind == ropeScaleYarn {
		var y struct {
			Factor       float64  `json:"factor"`
			Mscale       *float64 `json:"mscale"`
			MscaleAllDim *float64 `json:"mscale_all_dim"`
		}
		_ = json.Unmarshal(cfg.RopeParameters, &y)
		if y.Mscale != nil || y.MscaleAllDim != nil {
			m, mAll := 1.0, 1.0
			if y.Mscale != nil {
				m = *y.Mscale
			}
			if y.MscaleAllDim != nil {
				mAll = *y.MscaleAllDim
			}
			scaling.mscale = yarnGetMscale(y.Factor, m) / yarnGetMscale(y.Factor, mAll)
		}
	}
	var attnTemp struct {
		Beta       float64 `json:"llama_4_scaling_beta"`
		OrigMaxPos float64 `json:"original_max_position_embeddings"`
	}
	_ = json.Unmarshal(cfg.RopeParameters, &attnTemp)

	hd := cfg.headDim()
	var layerIsGlobal func(int) bool
	if cfg.SlidingWindow > 0 {
		layerIsGlobal = func(int) bool { return false }
	}
	return &Architecture{
		Name:               "mistral3",
		HiddenDim:          cfg.HiddenDim,
		NumLayers:          cfg.NumLayers,
		NumHeads:           cfg.NumHeads,
		NumKVHeads:         cfg.NumKVHeads,
		HeadDim:            hd,
		IntermediateDim:    cfg.IntermediateDim,
		VocabSize:          cfg.VocabSize,
		Norm:               NormRMS,
		RMSAddOne:          false,
		NormEps:            cfg.RMSNormEps,
		NormPlacement:      NormPre2,
		Act:                ActSiLU,
		QKNorm:             false,
		AttnScale:          math.Pow(float64(hd), -0.5),
		AttnTempBeta:       attnTemp.Beta,
		AttnTempOrigMaxPos: attnTemp.OrigMaxPos,
		SlidingWindow:      cfg.SlidingWindow, // 0 ⇒ full attention — confirmed null on the real release
		layerIsGlobal:      layerIsGlobal,
		RoPELocalBase:      base,
		RoPEGlobalBase:     base,
		RotaryDim:          cfg.rotaryDim(),
		ropeScaling:        scaling,
		EmbedScale:         0,
		TiedLMHead:         false, // finalized from lm_head.weight presence at load
	}, &llamaTensorSchema, nil
}
