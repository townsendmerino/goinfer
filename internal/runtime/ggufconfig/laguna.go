package ggufconfig

import (
	"encoding/json"
	"fmt"

	"github.com/townsendmerino/aikit/embed"
)

// ggufLagunaConfig builds a Laguna Config from the laguna.* metadata. llama.cpp has first-class laguna support
// (general.architecture == "laguna"), and the GGUF carries the family's two awkward parts more cleanly than the
// safetensors config does: the per-layer query head counts arrive as an array (laguna.attention.head_count), and the two
// layer types' rotary widths as separate rope.dimension_count / rope.dimension_count_swa scalars.
//
// Three things the GGUF does not say, each handled explicitly:
//
//  1. Which layers are full. There is no layer_types and no sliding-window pattern key, so it is derived. The head-count
//     array encodes it (full-attention and sliding layers have different query head counts) and layer 0 is full in every
//     released config, so layers matching heads[0] are the full ones. That is an inference, so it is validated: at most
//     two distinct counts, and the derived split is reported to the caller through LayerTypes for the gate to assert.
//
//  2. The gate's granularity. There is no gating key at all, matching the safetensors path, where the declared value is
//     unreliable anyway (XS.2 declares `gating: true` and ships a per-head tensor). The loader reads it from
//     blk.0.attn_gate.weight's shape, which is the authority in both formats.
//
//  3. YaRN's attention_factor. llama.cpp writes rope.scaling.yarn_attn_factor = 1.0 as its "unset" sentinel and computes
//     the mscale itself. Passing 1.0 through would replace YaRN's mscale with a no-op: goinfer's attention_factor is a
//     *float64 whose nil means "compute get_mscale(factor) = 0.1*ln(factor)+1", which is what the safetensors config
//     states. So the field is omitted at the sentinel and passed through otherwise.
func ggufLagunaConfig(g *embed.GGUFFile) (*Config, error) {
	u := ggufUintReader(g, "laguna.")
	gf := ggufFloatReader(g, "laguna.")
	headDim := u("attention.key_length")
	numLayers := u("block_count")
	if headDim <= 0 || numLayers <= 0 {
		return nil, fmt.Errorf("decoder(gguf-laguna): missing attention.key_length / block_count")
	}
	// Bound block_count before anything allocates from it: laguna goes on to make([]string, numLayers), and a GGUF declaring
	// laguna.block_count = 2^36 asks for 1 TiB, under Go's maxAlloc, so it is a fatal "out of memory" rather than a panic
	// and the loader's own recover() cannot catch it. Every builder that allocates from block_count calls ggufLayerCount.
	if _, err := ggufLayerCount(int(numLayers)); err != nil {
		return nil, err
	}
	// Sigmoid routing is not optional in this family; llama.cpp encodes it as
	// expert_gating_func 2 (softmax is 1). Fail loudly rather than route by softmax.
	if gfn := u("expert_gating_func"); gfn != 0 && gfn != 2 {
		return nil, fmt.Errorf("decoder(gguf-laguna): expert_gating_func=%d unsupported (2 = sigmoid)", gfn)
	}
	// expert_weights_norm is a GGUF bool, but writers have spelled it as an integer
	// too. Read both with plain assertions rather than a type switch — the decoder
	// dispatch census (TestDispatchCensus) exists to stop type switches on a VALUE'S
	// ENCODING creeping into this package, and that is exactly what this would be.
	normTopK := true
	if b, ok := g.Metadata["laguna.expert_weights_norm"].(bool); ok {
		normTopK = b
	} else if n, ok := g.Uint("laguna.expert_weights_norm"); ok {
		normTopK = n != 0
	}
	scale := gf("expert_weights_scale")
	if scale == 0 {
		scale = 1
	}
	cfg := &Config{
		ModelType:                    "laguna",
		MaxPositions:                 u("context_length"),
		HiddenDim:                    u("embedding_length"),
		NumLayers:                    numLayers,
		NumHeads:                     u("attention.head_count"), // scalar fallback; the array overrides below
		NumKVHeads:                   u("attention.head_count_kv"),
		HeadDim:                      headDim,
		IntermediateDim:              u("feed_forward_length"),
		MoeIntermediateSize:          u("expert_feed_forward_length"),
		SharedExpertIntermediateSize: u("expert_shared_feed_forward_length"),
		NumExperts:                   u("expert_count"),
		NumExpertsPerTok:             u("expert_used_count"),
		SlidingWindow:                u("attention.sliding_window"),
		MoeRoutedScalingFactor:       scale,
		NormTopKProb:                 &normTopK,
		HiddenAct:                    "silu",
		VocabSize:                    ggufVocabSize(g),
		RMSNormEps:                   gf("attention.layer_norm_rms_epsilon"),
	}
	// Per-layer QUERY heads. Absent (M.1, all-full uniform) ⇒ leave nil and the
	// scalar head_count applies to every layer.
	heads := ggufIntArray(g.Metadata["laguna.attention.head_count"])
	if len(heads) > 0 {
		if len(heads) != numLayers {
			return nil, fmt.Errorf("decoder(gguf-laguna): attention.head_count has %d entries, want %d", len(heads), numLayers)
		}
		cfg.NumAttentionHeadsPerLayer = heads
		cfg.NumHeads = heads[0]
	}
	// Derive layer_types (see 1 above). Only when the model actually has a window.
	if cfg.SlidingWindow > 0 && len(heads) > 0 {
		distinct := map[int]bool{}
		for _, h := range heads {
			distinct[h] = true
		}
		if len(distinct) > 2 {
			return nil, fmt.Errorf("decoder(gguf-laguna): attention.head_count has %d distinct values; "+
				"cannot derive full/sliding layer types from it", len(distinct))
		}
		cfg.LayerTypes = make([]string, numLayers)
		for i, h := range heads {
			if h == heads[0] {
				cfg.LayerTypes[i] = "full_attention"
			} else {
				cfg.LayerTypes[i] = "sliding_attention"
			}
		}
	}
	// leading_dense_block_count → the mlp_layer_types spelling LagunaFirstKDense reads,
	// so the GGUF and safetensors paths converge on one code path.
	if k := u("leading_dense_block_count"); k > 0 {
		cfg.MlpLayerTypes = make([]string, numLayers)
		for i := range cfg.MlpLayerTypes {
			if i < k {
				cfg.MlpLayerTypes[i] = "dense"
			} else {
				cfg.MlpLayerTypes[i] = "sparse"
			}
		}
	}
	// rope_parameters, keyed by layer type exactly as the safetensors config is.
	base := gf("rope.freq_base")
	baseSwa := gf("rope.freq_base_swa")
	if baseSwa == 0 {
		baseSwa = base
	}
	partial := func(k string) float64 {
		rot := u(k)
		if rot <= 0 || headDim <= 0 {
			return 1
		}
		return float64(rot) / float64(headDim)
	}
	attnFactor := ""
	if af := gf("rope.scaling.yarn_attn_factor"); af != 0 && af != 1 {
		attnFactor = fmt.Sprintf(`,"attention_factor":%g`, af)
	}
	full := fmt.Sprintf(`{"rope_type":"yarn","rope_theta":%g,"factor":%g,"original_max_position_embeddings":%g,`+
		`"beta_fast":%g,"beta_slow":%g,"partial_rotary_factor":%g%s}`,
		base, gf("rope.scaling.factor"), gf("rope.scaling.original_context_length"),
		gf("rope.scaling.yarn_beta_fast"), gf("rope.scaling.yarn_beta_slow"),
		partial("rope.dimension_count"), attnFactor)
	if gf("rope.scaling.factor") == 0 {
		full = fmt.Sprintf(`{"rope_type":"default","rope_theta":%g,"partial_rotary_factor":%g}`,
			base, partial("rope.dimension_count"))
	}
	if cfg.SlidingWindow > 0 {
		cfg.RopeParameters = json.RawMessage(fmt.Sprintf(
			`{"full_attention":%s,"sliding_attention":{"rope_type":"default","rope_theta":%g,"partial_rotary_factor":%g}}`,
			full, baseSwa, partial("rope.dimension_count_swa")))
	} else {
		cfg.RopeParameters = json.RawMessage(fmt.Sprintf(`{"full_attention":%s}`, full))
	}
	ggufEOS(g, cfg)
	return cfg, nil
}
