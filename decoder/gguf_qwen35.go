package decoder

import (
	"encoding/json"
	"fmt"

	"github.com/townsendmerino/aikit/embed"
)

// GGUF loading for the qwen3_5_moe (Qwen3.6-35B-A3B) architecture
// (general.architecture = "qwen35moe"). This is a TRANSFORM REVERSER, not a
// name-map: llama.cpp's converter bakes several layout changes into the GGUF that
// must be undone here so the existing safetensors-shaped forward runs unchanged —
// the V-head tiling (reorderVHeads/untileVHeads below), the −exp(A_log) bake
// (negExpAFromLog), and the (1+w) norm bake on the standard norms (the vnorm
// closure in buildWeightsFromGGUF; ssm_norm is exempt). The tensor loading itself
// lives in buildWeightsFromGGUF's qwen35 branch; this file holds the config map
// and the reorder primitive.

// ggufQwen35Config synthesizes a qwen3_5_moe Config from the qwen35moe.* GGUF
// metadata. It sets ModelType "qwen3_5_moe" so resolveArchitecture reuses the
// exact same adapter + forward as the safetensors path. Two GGUF-specific
// recoveries: NumLayers excludes the trailing NextN/MTP block
// (block_count − nextn_predict_layers), and layer_types is rebuilt from
// full_attention_interval (every interval-th layer, 1-indexed, is full attention;
// the rest are Gated DeltaNet linear-attention).
func ggufQwen35Config(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("qwen35moe." + k)
		return int(v)
	}
	blocks := u("block_count")
	numLayers := blocks - u("nextn_predict_layers") // drop the NextN/MTP block(s)
	// Bound before the per-layer append loop below: a hostile block_count would grow
	// LayerTypes unboundedly (a fatal OOM recover can't catch) before validateGGUFDims
	// runs (M16).
	if numLayers <= 0 || numLayers > maxGGUFLayers {
		return nil, fmt.Errorf("decoder(gguf-qwen35): bad block_count=%d (nextn=%d)", blocks, u("nextn_predict_layers"))
	}
	headDim := u("attention.key_length")
	numV := u("ssm.time_step_rank") // linear_num_value_heads
	valueHeadDim := 0
	if numV > 0 {
		valueHeadDim = u("ssm.inner_size") / numV // inner_size = value_head_dim · num_value_heads
	}
	interval := u("full_attention_interval")
	if interval <= 0 {
		interval = 4
	}
	cfg := &Config{
		ModelType:                    "qwen3_5_moe", // reuse the safetensors adapter + forward
		HiddenDim:                    u("embedding_length"),
		NumLayers:                    numLayers,
		NumHeads:                     u("attention.head_count"),
		NumKVHeads:                   u("attention.head_count_kv"),
		HeadDim:                      headDim,
		MoeIntermediateSize:          u("expert_feed_forward_length"),
		SharedExpertIntermediateSize: u("expert_shared_feed_forward_length"),
		NumExperts:                   u("expert_count"),
		NumExpertsPerTok:             u("expert_used_count"),
		LinearConvKernelDim:          u("ssm.conv_kernel"),
		LinearKeyHeadDim:             u("ssm.state_size"),  // per-head key/query dim
		LinearValueHeadDim:           valueHeadDim,         // derived from inner_size
		LinearNumKeyHeads:            u("ssm.group_count"), // linear_num_key_heads
		LinearNumValueHeads:          numV,
		HiddenAct:                    "silu",
		VocabSize:                    ggufVocabSize(g),
	}
	if eps, ok := g.Float("qwen35moe.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	for i := range numLayers {
		if (i+1)%interval == 0 {
			cfg.LayerTypes = append(cfg.LayerTypes, "full_attention")
		} else {
			cfg.LayerTypes = append(cfg.LayerTypes, "linear_attention")
		}
	}
	// rope_parameters: the text decoder uses standard partial-rotary RoPE (the
	// mrope sections are image/video-only and unused here). Synthesize the flat
	// form parseRopeFlat expects from the GGUF rope metadata.
	base := 1e7
	if b, ok := g.Float("qwen35moe.rope.freq_base"); ok {
		base = b
	}
	partial := 0.25
	if dc := u("rope.dimension_count"); dc > 0 && headDim > 0 {
		partial = float64(dc) / float64(headDim)
	}
	cfg.RopeParameters = json.RawMessage(fmt.Sprintf(
		`{"rope_type":"default","rope_theta":%g,"partial_rotary_factor":%g}`, base, partial))
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufQwen35DenseConfig reads the DENSE hybrid (llama.cpp arch "qwen35" — Qwen3.8), the sibling of
// "qwen35moe" above. Same Gated-DeltaNet/softmax 3:1 interleave, same SSM geometry keys; the only
// structural difference is a plain SwiGLU where the MoE sibling has a router, so the expert keys are
// replaced by feed_forward_length. It resolves to the qwen3_5 adapter, which is the dense descriptor
// the safetensors bring-up added.
//
// WHY IT IS WORTH HAVING at all, given the safetensors loader already works: the 27.8B ships as
// 55.6 GB of bf16, which goinfer re-quantizes on EVERY load (68 s measured), against 16.5 GB of
// pre-quantized GGUF that mmaps in about a second. Same model, 3.4x less disk and a load that stops
// dominating short runs.
func ggufQwen35DenseConfig(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("qwen35." + k)
		return int(v)
	}
	blocks := u("block_count")
	// Same MTP tail as the MoE sibling: block_count counts the NextN/multi-token-prediction block
	// this decoder does not run, so subtract it or every per-layer loop walks off the end.
	numLayers := blocks - u("nextn_predict_layers")
	if numLayers <= 0 || numLayers > maxGGUFLayers {
		return nil, fmt.Errorf("decoder(gguf-qwen35-dense): bad block_count=%d (nextn=%d)", blocks, u("nextn_predict_layers"))
	}
	headDim := u("attention.key_length")
	numV := u("ssm.time_step_rank") // linear_num_value_heads
	valueHeadDim := 0
	if numV > 0 {
		valueHeadDim = u("ssm.inner_size") / numV
	}
	interval := u("full_attention_interval")
	if interval <= 0 {
		interval = 4
	}
	cfg := &Config{
		ModelType:           "qwen3_5", // the DENSE adapter (qwen35DenseArchitecture)
		HiddenDim:           u("embedding_length"),
		NumLayers:           numLayers,
		NumHeads:            u("attention.head_count"),
		NumKVHeads:          u("attention.head_count_kv"),
		HeadDim:             headDim,
		IntermediateDim:     u("feed_forward_length"), // dense SwiGLU width; load-bearing for this adapter
		LinearConvKernelDim: u("ssm.conv_kernel"),
		LinearKeyHeadDim:    u("ssm.state_size"),
		LinearValueHeadDim:  valueHeadDim,
		LinearNumKeyHeads:   u("ssm.group_count"),
		LinearNumValueHeads: numV,
		HiddenAct:           "silu",
		VocabSize:           ggufVocabSize(g),
	}
	if eps, ok := g.Float("qwen35.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	// layer_types is COMPUTED here, exactly as in qwen3_next: llama.cpp writes
	// full_attention_interval rather than a per-layer list.
	for i := range numLayers {
		if (i+1)%interval == 0 {
			cfg.LayerTypes = append(cfg.LayerTypes, "full_attention")
		} else {
			cfg.LayerTypes = append(cfg.LayerTypes, "linear_attention")
		}
	}
	base := 1e7
	if b, ok := g.Float("qwen35.rope.freq_base"); ok {
		base = b
	}
	partial := 0.25
	if dc := u("rope.dimension_count"); dc > 0 && headDim > 0 {
		partial = float64(dc) / float64(headDim)
	}
	// rope.dimension_sections ([11,11,10,0]) is the m-RoPE split and is deliberately IGNORED: for
	// TEXT the three position components are identical, so interleaved m-RoPE reduces exactly to
	// standard partial RoPE (verified against modeling_qwen3_5.py during the safetensors bring-up).
	cfg.RopeParameters = json.RawMessage(fmt.Sprintf(
		`{"rope_type":"default","rope_theta":%g,"partial_rotary_factor":%g}`, base, partial))
	ggufEOS(g, cfg)
	return cfg, nil
}

// reorderVHeads swaps the (outer, inner) factorization of the reordered axis of a
// [lead, outer*inner*hd, trail] row-major buffer: the element at (outer=o,
// inner=i, ...) moves to (outer=i, inner=o, ...). hd is the contiguous head block
// that rides along untouched; trail is the product of all dims after the reordered
// axis (e.g. the "in"/columns for a row reorder, or 1 for a column reorder); lead
// is the product of all dims before it (1 for a row reorder, the rows for a column
// reorder).
//
// This mirrors llama.cpp's _reorder_v_heads (gguf-py conversion/qwen.py): for
// linear-attn (Gated DeltaNet) layers where num_v_heads > num_k_heads, the
// converter takes HF's V heads stored GROUPED by K head and writes them TILED for
// ggml's broadcast. The forward (HF→GGUF) is reorderVHeads(x, lead, numKHeads,
// numVPerK, hd, trail); the loader UN-tiles (GGUF→HF) with the two factors
// swapped — see untileVHeads. reorderVHeads is its own inverse under that swap.
func reorderVHeads(src []float32, lead, outer, inner, hd, trail int) []float32 {
	mid := outer * inner * hd
	dst := make([]float32, len(src))
	for l := range lead {
		base := l * mid
		for o := range outer {
			for i := range inner {
				srcRow := base + (o*inner+i)*hd // [outer, inner, hd]
				dstRow := base + (i*outer+o)*hd // [inner, outer, hd]
				for c := range hd {
					s := (srcRow + c) * trail
					d := (dstRow + c) * trail
					copy(dst[d:d+trail], src[s:s+trail])
				}
			}
		}
	}
	return dst
}

// untileVHeads undoes the GGUF converter's V-head tiling: it takes a GGUF tensor
// block whose V heads are in tiled order and returns it in HF's grouped-by-K
// order. numVPerK = numVHeads / numKHeads. It is reorderVHeads with the converter's
// (numKHeads, numVPerK) factors swapped.
func untileVHeads(src []float32, lead, numKHeads, numVPerK, hd, trail int) []float32 {
	return reorderVHeads(src, lead, numVPerK, numKHeads, hd, trail)
}
