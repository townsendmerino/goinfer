package decoder

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// GGUF loading — read a quantized llama.cpp checkpoint and
// run it through the generic forward. The GGUF file carries both the
// architecture config (metadata) and the weights (dequantized from the mmap,
// then optionally re-quantized to resident int8/int4 per the quant mode), so no
// separate config.json/safetensors is needed. Layout quirks vs the HF safetensors
// path are normalized at load: tensors use llama.cpp's blk.N.* names; the q/k
// projections of NORM-rope archs (llama/mellum) are in llama.cpp's interleaved-
// RoPE permutation, inverted here to this package's rotate_half order (NEOX-rope
// archs — qwen/gemma — are left as-is, see ggufQKPermuted); and Gemma's (1+w)
// norm offset, baked into the stored weights by llama.cpp, is subtracted back out.
//
// Architectures: llama, qwen2, qwen3, gemma3, mellum. Quant types: F32/F16,
// Q8_0/Q4_0/Q5_0, the K-quants Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, and IQ4_NL/IQ4_XS.

// ggufConfig synthesizes a Config from GGUF metadata, dispatching on
// general.architecture. The resolved Config feeds resolveArchitecture, so a
// GGUF reuses the same per-family adapter as the safetensors path.
func ggufConfig(g *embed.GGUFFile) (cfg *Config, err error) {
	arch, _ := g.Str("general.architecture")
	// Hostile metadata can drive a family builder into an integer divide-by-zero
	// (e.g. a missing head_count ⇒ hidden/0) or a similar panic before
	// validateGGUFDims ever runs. Convert any such panic into a typed error so the
	// loader honors its never-panic contract for every architecture, current and
	// future (M16). Unbounded per-layer allocations are guarded explicitly by
	// ggufLayerCount below — a huge makeslice is a fatal OOM that recover can't catch.
	defer func() {
		if r := recover(); r != nil {
			cfg, err = nil, fmt.Errorf("decoder(gguf): malformed metadata for architecture %q: %v", arch, r)
		}
	}()
	switch arch {
	case "llama":
		return ggufLlamaConfig(g)
	case "qwen2":
		return ggufQwen2Config(g)
	case "qwen3":
		return ggufQwen3Config(g)
	case "gemma3", "gemma3_text":
		return ggufGemmaConfig(g)
	case "gemma4":
		return ggufGemma4Config(g)
	case "mellum":
		return ggufMellumConfig(g)
	case "qwen35moe":
		return ggufQwen35Config(g)
	case "laguna":
		return ggufLagunaConfig(g)
	case "glm4moe":
		return ggufGlm4MoeConfig(g)
	case "granitehybrid":
		return ggufGraniteConfig(g)
	case "nemotron_h":
		return ggufNemotronConfig(g)
	case "nemotron_h_moe": // Nemotron 3 Nano — same HF model_type "nemotron_h", llama.cpp gives it its own GGUF arch string
		return ggufNemotronConfig(g)
	case "deepseek2":
		return ggufDeepseekConfig(g)
	case "phi3":
		return ggufPhi3Config(g)
	case "llama4":
		return ggufLlama4Config(g)
	case "gpt-oss":
		return ggufGptOssConfig(g)
	default:
		return nil, fmt.Errorf("decoder(gguf): architecture %q unsupported (have: llama, qwen2, qwen3, gemma3, gemma4 [wip], mellum, qwen35moe, glm4moe, laguna, granitehybrid, nemotron_h, nemotron_h_moe, deepseek2, phi3, gpt-oss)", arch)
	}
}

// ggufVocabSize reads the vocab size from the embedding tensor's dims (no
// dequant) or, failing that, the tokens metadata array length.
func ggufVocabSize(g *embed.GGUFFile) int {
	if dims, ok := g.Dims("token_embd.weight"); ok && len(dims) == 2 {
		return dims[1] // [in=hidden, out=vocab]
	}
	if toks, ok := g.Metadata["tokenizer.ggml.tokens"].([]any); ok {
		return len(toks)
	}
	return 0
}

// ggufEOS sets cfg.EOSTokenID from the GGUF tokenizer metadata (resolveEOSIDs
// falls back to this when there's no generation_config.json next to the .gguf).
func ggufEOS(g *embed.GGUFFile, cfg *Config) {
	if eos, ok := g.Uint("tokenizer.ggml.eos_token_id"); ok {
		cfg.EOSTokenID = fmt.Appendf(nil, "%d", eos)
	}
}

func ggufLlamaConfig(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint(k)
		return int(v)
	}
	cfg := &Config{
		ModelType:       "llama",
		HiddenDim:       u("llama.embedding_length"),
		NumLayers:       u("llama.block_count"),
		NumHeads:        u("llama.attention.head_count"),
		NumKVHeads:      u("llama.attention.head_count_kv"),
		IntermediateDim: u("llama.feed_forward_length"),
		HiddenAct:       "silu",
		VocabSize:       ggufVocabSize(g),
	}
	if eps, ok := g.Float("llama.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	if base, ok := g.Float("llama.rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	} else {
		cfg.RoPEGlobalBase = 10000 // llama.cpp default
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufQwen2Config builds a Qwen2/Qwen2.5 Config from the qwen2.* metadata. It is
// the llama config plus the qwen2 model type — the only architectural difference
// is the additive q/k/v projection bias, carried by the qwen2 descriptor
// (QKVBias) and loaded by buildWeightsFromGGUF. RoPE base defaults to llama.cpp's
// qwen2 default (1e6) when absent.
func ggufQwen2Config(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("qwen2." + k)
		return int(v)
	}
	cfg := &Config{
		ModelType:       "qwen2",
		HiddenDim:       u("embedding_length"),
		NumLayers:       u("block_count"),
		NumHeads:        u("attention.head_count"),
		NumKVHeads:      u("attention.head_count_kv"),
		IntermediateDim: u("feed_forward_length"),
		HiddenAct:       "silu",
		VocabSize:       ggufVocabSize(g),
	}
	if eps, ok := g.Float("qwen2.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	if base, ok := g.Float("qwen2.rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	} else {
		cfg.RoPEGlobalBase = 1000000 // llama.cpp qwen2 default
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufQwen3Config builds a Qwen3 dense Config from the qwen3.* metadata. Versus
// qwen2 it drops the q/k/v bias and adds QK-norm (per-head RMSNorm over head_dim
// before RoPE) — both carried by the qwen3 descriptor, so the GGUF loader only
// needs the metadata mapping (head_dim is explicit via attention.key_length, not
// derived). NEOX rope (no q/k permute, via ggufQKPermuted). RoPE base defaults to
// llama.cpp's qwen3 default (1e6) when absent.
func ggufQwen3Config(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("qwen3." + k)
		return int(v)
	}
	cfg := &Config{
		ModelType:       "qwen3",
		HiddenDim:       u("embedding_length"),
		NumLayers:       u("block_count"),
		NumHeads:        u("attention.head_count"),
		NumKVHeads:      u("attention.head_count_kv"),
		HeadDim:         u("attention.key_length"),
		IntermediateDim: u("feed_forward_length"),
		HiddenAct:       "silu",
		VocabSize:       ggufVocabSize(g),
	}
	if eps, ok := g.Float("qwen3.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	if base, ok := g.Float("qwen3.rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	} else {
		cfg.RoPEGlobalBase = 1000000 // llama.cpp qwen3 default
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufGemmaConfig builds a Gemma 3 (text) Config from the gemma3.* metadata. The
// gemma3 descriptor already supplies the architecture knobs (sandwich norms,
// GeGLU, QK-norm, embed scale √hidden, dual RoPE bases, tied head); this maps the
// dims + the few values the GGUF doesn't carry explicitly. NEOX rope (no q/k
// permute). The GGUF omits the local RoPE base, the sliding/global pattern, and
// query_pre_attn_scalar, so they default to gemma3's fixed values (10000, a 5:1
// pattern, and head_dim). Gemma 3 has no logit/attn soft-capping (that's Gemma 2),
// so those stay zero — ValidateAssumptions rejects a Gemma 2 GGUF here.
func ggufGemmaConfig(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("gemma3." + k)
		return int(v)
	}
	cfg := &Config{
		ModelType:        "gemma3",
		HiddenDim:        u("embedding_length"),
		NumLayers:        u("block_count"),
		NumHeads:         u("attention.head_count"),
		NumKVHeads:       u("attention.head_count_kv"),
		HeadDim:          u("attention.key_length"),
		IntermediateDim:  u("feed_forward_length"),
		SlidingWindow:    u("attention.sliding_window"),
		HiddenActivation: "gelu_pytorch_tanh",
		VocabSize:        ggufVocabSize(g),
	}
	if eps, ok := g.Float("gemma3.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	if base, ok := g.Float("gemma3.rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	} else {
		cfg.RoPEGlobalBase = 1000000 // gemma3 rope_theta (global layers)
	}
	cfg.RoPELocalBase = 10000 // gemma3 rope_local_base_freq (sliding layers); not in GGUF metadata
	// Sliding/global pattern: gemma3 is 5 local : 1 global. The GGUF carries no
	// layer_types, so IsGlobalLayer falls back to this period ((i+1)%6==0 global).
	if p := u("attention.sliding_window_pattern"); p > 0 {
		cfg.SlidingWindowPattern = p
	} else {
		cfg.SlidingWindowPattern = 6
	}
	// query_pre_attn_scalar == head_dim for gemma3 (the attention scale is its
	// inverse sqrt); the GGUF doesn't store it.
	cfg.QueryPreAttnScalar = float64(cfg.HeadDim)
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufIntArray / ggufBoolArray coerce a GGUF array metadatum (parsed as []any)
// into a typed slice — for gemma4's per-layer feed_forward_length and the
// sliding/global bool pattern.
func ggufIntArray(v any) []int {
	a, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]int, len(a))
	for i, e := range a {
		switch n := e.(type) {
		case int:
			out[i] = n
		case int32:
			out[i] = int(n)
		case int64:
			out[i] = int(n)
		case uint32:
			out[i] = int(n)
		case uint64:
			out[i] = int(n)
		case float64:
			out[i] = int(n)
		}
	}
	return out
}

func ggufBoolArray(v any) []bool {
	a, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]bool, len(a))
	for i, e := range a {
		if b, ok := e.(bool); ok {
			out[i] = b
		}
	}
	return out
}

// ggufGemma4Config builds a Gemma 4 (E-model) Config from the gemma4.* metadata:
// the variable per-layer FFN width + head_dim, the cross-layer KV-sharing count,
// the PLE (per-layer-embedding) dims, the explicit sliding/global bool pattern,
// dual-base RoPE, and the re-added final-logit softcap (30). The proportional
// (partial-rotary) factor for the global layers isn't in the GGUF metadata, so
// it's set from the known Gemma 4 value (0.25). Defaults mirror ggufGemmaConfig.
func ggufGemma4Config(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("gemma4." + k)
		return int(v)
	}
	cfg := &Config{
		ModelType:               "gemma4",
		HiddenDim:               u("embedding_length"),
		NumLayers:               u("block_count"),
		NumHeads:                u("attention.head_count"),
		NumKVHeads:              u("attention.head_count_kv"),
		HeadDim:                 u("attention.key_length_swa"), // sliding/local head_dim (256)
		GlobalHeadDim:           u("attention.key_length"),     // global/full head_dim (512)
		SlidingWindow:           u("attention.sliding_window"),
		SharedKVLayers:          u("attention.shared_kv_layers"),
		HiddenSizePerLayerInput: u("embedding_length_per_layer_input"),
		HiddenActivation:        "gelu_pytorch_tanh",
		VocabSize:               ggufVocabSize(g),
		PartialRotaryFactor:     0.25, // proportional RoPE on the global layers; not in GGUF metadata
	}
	cfg.VocabSizePerLayerInput = cfg.VocabSize // PLE table shares the main vocab
	// Variable per-layer FFN width (e.g. 6144 then 12288); fall back to a scalar.
	if a, ok := g.Metadata["gemma4.feed_forward_length"]; ok {
		cfg.FFNPerLayer = ggufIntArray(a)
	}
	if len(cfg.FFNPerLayer) > 0 {
		cfg.IntermediateDim = cfg.FFNPerLayer[0]
	} else {
		cfg.IntermediateDim = u("feed_forward_length")
	}
	// Explicit per-layer attention type (true = sliding, false = full/global).
	if a, ok := g.Metadata["gemma4.attention.sliding_window_pattern"]; ok {
		for _, sliding := range ggufBoolArray(a) {
			if sliding {
				cfg.LayerTypes = append(cfg.LayerTypes, "sliding_attention")
			} else {
				cfg.LayerTypes = append(cfg.LayerTypes, "full_attention")
			}
		}
	}
	// head_count_kv may be a per-layer array (12B: 8 on sliding, 1 on global) or a
	// scalar (E2B/E4B: 1). For the array, pull the sliding + global values; the
	// scalar literal above already covered the uniform case.
	if arr := ggufIntArray(g.Metadata["gemma4.attention.head_count_kv"]); len(arr) > 0 {
		for i, lt := range cfg.LayerTypes {
			if i >= len(arr) {
				break
			}
			if lt == "full_attention" {
				cfg.NumGlobalKVHeads = arr[i]
			} else {
				cfg.NumKVHeads = arr[i]
			}
		}
	}
	if eps, ok := g.Float("gemma4.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	cfg.RoPEGlobalBase = 1000000
	if base, ok := g.Float("gemma4.rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	}
	cfg.RoPELocalBase = 10000
	if base, ok := g.Float("gemma4.rope.freq_base_swa"); ok {
		cfg.RoPELocalBase = base
	}
	if sc, ok := g.Float("gemma4.final_logit_softcapping"); ok {
		cfg.FinalLogitSoftcap = sc
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufMellumConfig builds a Mellum2 Config from the mellum.* metadata: dims,
// the MoE counts (incl. the narrower expert_feed_forward_length), the sliding/
// full layer pattern, and the YaRN rope scaling — synthesized into the same
// shapes the mellum adapter consumes (LayerTypes + a rope_parameters JSON), so
// resolveArchitecture runs the identical descriptor build as the safetensors path.
func ggufMellumConfig(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("mellum." + k)
		return int(v)
	}
	gf := func(k string) float64 { // float-or-int metadata
		if v, ok := g.Float("mellum." + k); ok {
			return v
		}
		v, _ := g.Uint("mellum." + k)
		return float64(v)
	}
	normTopK := true
	cfg := &Config{
		ModelType:           "mellum",
		HiddenDim:           u("embedding_length"),
		NumLayers:           u("block_count"),
		NumHeads:            u("attention.head_count"),
		NumKVHeads:          u("attention.head_count_kv"),
		HeadDim:             u("attention.key_length"),
		IntermediateDim:     u("feed_forward_length"),        // dense (vestigial)
		MoeIntermediateSize: u("expert_feed_forward_length"), // expert FFN width
		NumExperts:          u("expert_count"),
		NumExpertsPerTok:    u("expert_used_count"),
		SlidingWindow:       u("attention.sliding_window"),
		NormTopKProb:        &normTopK,
		HiddenAct:           "silu",
		VocabSize:           ggufVocabSize(g),
		RMSNormEps:          gf("attention.layer_norm_rms_epsilon"),
	}
	// Per-layer attention type: sliding_window_pattern[i] true ⇒ sliding (local),
	// false ⇒ full (global).
	pat, ok := g.Metadata["mellum.attention.sliding_window_pattern"].([]any)
	if !ok || len(pat) != cfg.NumLayers {
		return nil, fmt.Errorf("decoder(gguf-mellum): sliding_window_pattern missing or wrong length")
	}
	for _, p := range pat {
		if b, _ := p.(bool); b {
			cfg.LayerTypes = append(cfg.LayerTypes, "sliding_attention")
		} else {
			cfg.LayerTypes = append(cfg.LayerTypes, "full_attention")
		}
	}
	// Synthesize rope_parameters: YaRN on the full layers (freq_base + scaling),
	// plain RoPE on the sliding layers (freq_base_swa). llama.cpp truncates the
	// attention_factor to f32; the mscale tolerance absorbs that.
	base := gf("rope.freq_base")
	baseSwa := gf("rope.freq_base_swa")
	if baseSwa == 0 {
		baseSwa = base
	}
	cfg.RopeParameters = json.RawMessage(fmt.Sprintf(
		`{"full_attention":{"rope_type":"yarn","rope_theta":%g,"factor":%g,"original_max_position_embeddings":%g,"beta_fast":%g,"beta_slow":%g,"attention_factor":%g},`+
			`"sliding_attention":{"rope_type":"default","rope_theta":%g}}`,
		base, gf("rope.scaling.factor"), gf("rope.scaling.original_context_length"),
		gf("rope.scaling.yarn_beta_fast"), gf("rope.scaling.yarn_beta_slow"),
		gf("rope.scaling.yarn_attn_factor"), baseSwa))
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufLagunaConfig builds a Laguna Config from the laguna.* metadata. llama.cpp
// has FIRST-CLASS laguna support (general.architecture == "laguna"), and the GGUF
// carries the family's two awkward parts more cleanly than the safetensors config
// does: the per-layer QUERY head counts arrive as an ARRAY
// (laguna.attention.head_count), and the two layer types' rotary widths as separate
// rope.dimension_count / rope.dimension_count_swa scalars.
//
// THREE THINGS THE GGUF DOES NOT SAY, each handled explicitly:
//
//  1. WHICH LAYERS ARE FULL. There is no layer_types and no sliding-window PATTERN
//     key, so it must be derived. The head-count array encodes it (48 on
//     full_attention, 64 on sliding on the XS line) and layer 0 is full in every
//     released config, so layers matching heads[0] are the full ones. That is an
//     INFERENCE, so it is validated: at most two distinct counts, and the derived
//     split is reported to the caller through LayerTypes for the gate to assert.
//
//  2. THE GATE'S GRANULARITY. There is no gating key at all — matching the
//     safetensors path, where the declared value is unreliable anyway (XS.2 says
//     `gating: true` and ships a per-HEAD tensor). The loader reads it from
//     blk.0.attn_gate.weight's shape, which is the authority in both formats.
//
//  3. YARN'S attention_factor. llama.cpp writes rope.scaling.yarn_attn_factor = 1.0
//     as its "unset" sentinel and computes the mscale itself. Passing 1.0 through
//     would REPLACE YaRN's mscale with a no-op: goinfer's attention_factor is a
//     *float64 whose nil means "compute get_mscale(factor) = 0.1·ln(factor)+1", which
//     for factor 32 is 1.3465735902799727 — exactly what the safetensors config
//     states. So the field is OMITTED at the sentinel and passed through otherwise.
func ggufLagunaConfig(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("laguna." + k)
		return int(v)
	}
	gf := func(k string) float64 {
		if v, ok := g.Float("laguna." + k); ok {
			return v
		}
		v, _ := g.Uint("laguna." + k)
		return float64(v)
	}
	headDim := u("attention.key_length")
	numLayers := u("block_count")
	if headDim <= 0 || numLayers <= 0 {
		return nil, fmt.Errorf("decoder(gguf-laguna): missing attention.key_length / block_count")
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
	// leading_dense_block_count → the mlp_layer_types spelling lagunaFirstKDense reads,
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

// ggufGlm4MoeConfig builds a GLM-4.5/4.6 Config from the glm4moe.* metadata: dims,
// the DeepSeek-style MoE counts (expert_count / expert_used_count /
// expert_shared_count / expert_feed_forward_length), the leading_dense_block_count
// dense prefix, routed_scaling_factor (expert_weights_scale), and a synthesized
// rope_parameters JSON (partial_rotary_factor = rope.dimension_count / head_dim) so
// glm4moeArchitecture runs the identical descriptor build as the safetensors path.
func ggufGlm4MoeConfig(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("glm4moe." + k)
		return int(v)
	}
	gf := func(k string) float64 {
		if v, ok := g.Float("glm4moe." + k); ok {
			return v
		}
		v, _ := g.Uint("glm4moe." + k)
		return float64(v)
	}
	normTopK := true
	if b, ok := g.Metadata["glm4moe.expert_weights_norm"].(bool); ok {
		normTopK = b
	}
	headDim := u("attention.key_length")
	scale := gf("expert_weights_scale")
	if scale == 0 {
		scale = 1
	}
	// block_count includes the trailing NextN/MTP block(s) goinfer drops.
	numLayers := u("block_count") - u("nextn_predict_layers")
	// GLM-4.5/4.6 carry q/k/v bias but NO QK-norm; the tiny synthetic is the
	// reverse. Neither has a metadata flag, so detect both from blk.0's tensors.
	_, attnBias := g.Dims("blk.0.attn_q.bias")
	_, qkNorm := g.Dims("blk.0.attn_q_norm.weight")
	cfg := &Config{
		ModelType:           "glm4_moe",
		HiddenDim:           u("embedding_length"),
		NumLayers:           numLayers,
		AttentionBias:       attnBias,
		UseQKNorm:           qkNorm,
		NumHeads:            u("attention.head_count"),
		NumKVHeads:          u("attention.head_count_kv"),
		HeadDim:             headDim,
		IntermediateDim:     u("feed_forward_length"),        // dense prefix width
		MoeIntermediateSize: u("expert_feed_forward_length"), // expert FFN width
		NRoutedExperts:      u("expert_count"),
		NumExpertsPerTok:    u("expert_used_count"),
		NSharedExperts:      u("expert_shared_count"),
		FirstKDenseReplace:  u("leading_dense_block_count"),
		RoutedScalingFactor: scale,
		NormTopKProb:        &normTopK,
		HiddenAct:           "silu",
		VocabSize:           ggufVocabSize(g),
		RMSNormEps:          gf("attention.layer_norm_rms_epsilon"),
	}
	rot := u("rope.dimension_count")
	partial := 0.0
	if headDim > 0 {
		partial = float64(rot) / float64(headDim)
	}
	cfg.RopeParameters = json.RawMessage(fmt.Sprintf(
		`{"rope_type":"default","rope_theta":%g,"partial_rotary_factor":%g}`,
		gf("rope.freq_base"), partial))
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufGptOssConfig builds a gpt-oss Config from the gpt-oss.* metadata. The SwiGLU
// alpha/limit are gpt-oss constants (not in the GGUF), set by gptOssArchitecture.
// YaRN is synthesized from the rope.scaling.* keys with truncate=false (gpt-oss's
// setting — floor/ceiling the correction range would shift inv_freq by ~2e-4).
func ggufGptOssConfig(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("gpt-oss." + k)
		return int(v)
	}
	gf := func(k string) float64 {
		if v, ok := g.Float("gpt-oss." + k); ok {
			return v
		}
		v, _ := g.Uint("gpt-oss." + k)
		return float64(v)
	}
	cfg := &Config{
		ModelType:           "gpt_oss",
		HiddenDim:           u("embedding_length"),
		NumLayers:           u("block_count"),
		NumHeads:            u("attention.head_count"),
		NumKVHeads:          u("attention.head_count_kv"),
		HeadDim:             u("attention.key_length"),
		MoeIntermediateSize: u("expert_feed_forward_length"),
		NumLocalExperts:     u("expert_count"),
		NumExpertsPerTok:    u("expert_used_count"),
		SlidingWindow:       u("attention.sliding_window"),
		HiddenAct:           "silu",
		VocabSize:           ggufVocabSize(g),
		RMSNormEps:          gf("attention.layer_norm_rms_epsilon"),
		RoPEGlobalBase:      gf("rope.freq_base"),
	}
	factor := gf("rope.scaling.factor")
	if factor == 0 {
		factor = 1
	}
	origCtx := gf("rope.scaling.original_context_length")
	betaFast := gf("rope.scaling.yarn_beta_fast")
	betaSlow := gf("rope.scaling.yarn_beta_slow")
	if scaleType, _ := g.Str("gpt-oss.rope.scaling.type"); scaleType == "yarn" {
		cfg.RopeScaling = json.RawMessage(fmt.Sprintf(
			`{"rope_type":"yarn","factor":%g,"beta_fast":%g,"beta_slow":%g,"original_max_position_embeddings":%g,"truncate":false}`,
			factor, betaFast, betaSlow, origCtx))
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufGraniteConfig builds a Granite-4.0-H Config from the granitehybrid.* metadata.
// Per-layer kind comes from the attention.head_count_kv array (>0 ⇒ attention layer,
// 0 ⇒ mamba). The Mamba-2 geometry is the ssm.* keys (time_step_rank IS the head
// count; head_dim = inner_size / time_step_rank). The four Granite scalars are
// embedding_scale / attention.scale / residual_scale / logit_scale. rope_parameters
// is synthesized so graniteArchitecture runs the same descriptor build as safetensors.
func ggufGraniteConfig(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("granitehybrid." + k)
		return int(v)
	}
	gf := func(k string) float64 {
		if v, ok := g.Float("granitehybrid." + k); ok {
			return v
		}
		v, _ := g.Uint("granitehybrid." + k)
		return float64(v)
	}
	nLayers, err := ggufLayerCount(u("block_count"))
	if err != nil {
		return nil, err
	}
	// Per-layer kind + the (uniform) attention KV-head count from the head_count_kv
	// array: nonzero entries are the attention layers.
	kvArr, _ := g.Metadata["granitehybrid.attention.head_count_kv"].([]any)
	layerTypes := make([]string, nLayers)
	kvHeads := 0
	for i := range nLayers {
		kv := 0
		if i < len(kvArr) {
			switch v := kvArr[i].(type) {
			case uint32:
				kv = int(v)
			case int32:
				kv = int(v)
			case uint64:
				kv = int(v)
			case int64:
				kv = int(v)
			}
		}
		if kv > 0 {
			layerTypes[i] = "attention"
			kvHeads = kv
		} else {
			layerTypes[i] = "mamba"
		}
	}
	nHeadsMamba := u("ssm.time_step_rank")
	innerSize := u("ssm.inner_size")
	cfg := &Config{
		ModelType:              "granitemoehybrid",
		HiddenDim:              u("embedding_length"),
		NumLayers:              nLayers,
		NumHeads:               u("attention.head_count"),
		NumKVHeads:             kvHeads,
		IntermediateDim:        u("feed_forward_length"),               // expert FFN width
		SharedIntermediateSize: u("expert_shared_feed_forward_length"), // shared expert width
		NumLocalExperts:        u("expert_count"),
		NumExpertsPerTok:       u("expert_used_count"),
		MambaNHeads:            nHeadsMamba,
		MambaDHead:             innerSize / nHeadsMamba,
		MambaDState:            u("ssm.state_size"),
		MambaDConv:             u("ssm.conv_kernel"),
		MambaNGroups:           u("ssm.group_count"),
		EmbeddingMultiplier:    gf("embedding_scale"),
		AttentionMultiplier:    gf("attention.scale"),
		ResidualMultiplier:     gf("residual_scale"),
		LogitsScaling:          gf("logit_scale"),
		LayerTypes:             layerTypes,
		HiddenAct:              "silu",
		VocabSize:              ggufVocabSize(g),
		RMSNormEps:             gf("attention.layer_norm_rms_epsilon"),
	}
	cfg.RopeParameters = json.RawMessage(fmt.Sprintf(
		`{"rope_type":"default","rope_theta":%g}`, gf("rope.freq_base")))
	// NoPE. The released granite-4.0-h models set position_embedding_type "nope" and the
	// converter carries that across as rope.scaling.finetuned — the rope.dimension_count /
	// rope.freq_base keys are written regardless and are vestigial here (this file has
	// dimension_count 128 on a model HF ropes not at all). Only an explicitly present key
	// flips the behaviour; absent leaves the roped path, so an older GGUF cannot silently
	// lose its RoPE. Verified against the bf16 oracle: roped ⇒ cosine 0.9936 + a wrong
	// continuation, NoPE ⇒ 0.9995 + exact.
	if finetuned, ok := g.Metadata["granitehybrid.rope.scaling.finetuned"].(bool); ok && !finetuned {
		cfg.PositionEmbeddingType = "nope"
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufNemotronConfig builds a Nemotron-H Config from the nemotron_h.* (or, for
// Nemotron 3 Nano's MoE variant, nemotron_h_moe.*) metadata. Per-layer kind comes
// from two parallel arrays: attention.head_count_kv (>0 ⇒ attention) and
// feed_forward_length (>0 ⇒ mlp for plain nemotron_h, moe for nemotron_h_moe — that
// architecture string never carries a plain dense-mlp layer, confirmed against a
// real checkpoint's metadata, not assumed); the rest are mamba. head_dim is
// attention.key_length (NOT embedding/heads). Attention is NoPE (the rope.* keys are
// vestigial). ModelType is normalized to "nemotron_h" either way — llama.cpp's GGUF
// arch string differs from HF's model_type (which stays "nemotron_h" for BOTH the
// dense and MoE checkpoints), but goinfer's own registry dispatches on the latter.
func ggufNemotronConfig(g *embed.GGUFFile) (*Config, error) {
	archStr, _ := g.Str("general.architecture") // "nemotron_h" or "nemotron_h_moe"
	isMoE := archStr == "nemotron_h_moe"
	u := func(k string) int {
		v, _ := g.Uint(archStr + "." + k)
		return int(v)
	}
	gf := func(k string) float64 {
		if v, ok := g.Float(archStr + "." + k); ok {
			return v
		}
		v, _ := g.Uint(archStr + "." + k)
		return float64(v)
	}
	intAt := func(arr []any, i int) int { // tolerant int read from a metadata array
		if i >= len(arr) {
			return 0
		}
		switch v := arr[i].(type) {
		case uint32:
			return int(v)
		case int32:
			return int(v)
		case uint64:
			return int(v)
		case int64:
			return int(v)
		}
		return 0
	}
	nLayers, err := ggufLayerCount(u("block_count"))
	if err != nil {
		return nil, err
	}
	kvArr, _ := g.Metadata[archStr+".attention.head_count_kv"].([]any)
	ffArr, _ := g.Metadata[archStr+".feed_forward_length"].([]any)
	types := make([]string, nLayers)
	kvHeads, ffLen := 0, 0
	ffnKind := "mlp"
	if isMoE {
		ffnKind = "moe"
	}
	for i := range nLayers {
		switch {
		case intAt(kvArr, i) > 0:
			types[i] = "attention"
			kvHeads = intAt(kvArr, i)
		case intAt(ffArr, i) > 0:
			types[i] = ffnKind
			ffLen = intAt(ffArr, i)
		default:
			types[i] = "mamba"
		}
	}
	nHeadsMamba := u("ssm.time_step_rank")
	inner := u("ssm.inner_size")
	eps := gf("attention.layer_norm_rms_epsilon")
	if eps == 0 {
		eps = gf("attention.layer_norm_epsilon")
	}
	cfg := &Config{
		ModelType:        "nemotron_h",
		HiddenDim:        u("embedding_length"),
		NumLayers:        nLayers,
		NumHeads:         u("attention.head_count"),
		NumKVHeads:       kvHeads,
		HeadDim:          u("attention.key_length"), // explicit — heads*head_dim != hidden
		IntermediateDim:  ffLen,
		MambaNumHeads:    nHeadsMamba,
		MambaHeadDim:     inner / nHeadsMamba,
		SSMStateSize:     u("ssm.state_size"),
		NGroups:          u("ssm.group_count"),
		ConvKernel:       u("ssm.conv_kernel"),
		LayersBlockType:  types,
		LayerNormEpsilon: eps,
		HiddenAct:        "silu",
		VocabSize:        ggufVocabSize(g),
	}
	if isMoE {
		// Nemotron 3 Nano's MoE fields — key names verified against a real GGUF file's
		// metadata (bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF), not assumed from the
		// safetensors config's field names or llama.cpp's convert script alone:
		// expert_count/expert_used_count/expert_feed_forward_length/
		// expert_shared_feed_forward_length/expert_shared_count/expert_weights_norm/
		// expert_weights_scale/expert_group_count/expert_group_used_count — all present,
		// including expert_group_used_count (the topk_group equivalent), which is easy to
		// assume absent since the safetensors config's own topk_group has no direct GGUF
		// key of the same name.
		normTopK := true
		if b, ok := g.Metadata[archStr+".expert_weights_norm"].(bool); ok {
			normTopK = b
		}
		cfg.NRoutedExperts = u("expert_count")
		cfg.NumExpertsPerTok = u("expert_used_count")
		cfg.MoeIntermediateSize = u("expert_feed_forward_length")
		cfg.MoeSharedExpertIntermediateSize = u("expert_shared_feed_forward_length")
		cfg.NSharedExperts = u("expert_shared_count")
		cfg.NormTopKProb = &normTopK
		cfg.RoutedScalingFactor = gf("expert_weights_scale")
		cfg.NGroup = u("expert_group_count")
		cfg.TopkGroup = u("expert_group_used_count")
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufDeepseekConfig builds a DeepSeek-V2/V3 (MLA) Config from the deepseek2.* metadata
// (llama.cpp maps BOTH V2 and V3 to arch "deepseek2"; the routing flavor is
// expert_gating_func: 1/absent = softmax = V2, 2 = sigmoid noaux_tc = V3). The MLA head
// split comes from attention.key_length (= qk_head_dim = nope+rope) minus
// rope.dimension_count (= qk_rope), with value_length the (different) v_head_dim and
// kv_lora_rank the cached-latent width; q_lora_rank is present only on the q-LoRA models.
// The yarn block is synthesized into rope_scaling so deepseekArchitecture runs the same
// descriptor build as safetensors (yarn_log_multiplier = 0.1·mscale; DeepSeek's
// mscale == mscale_all_dim ⇒ the cos/sin attention_factor ratio is 1.0).
func ggufDeepseekConfig(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("deepseek2." + k)
		return int(v)
	}
	gf := func(k string) float64 {
		if v, ok := g.Float("deepseek2." + k); ok {
			return v
		}
		v, _ := g.Uint("deepseek2." + k)
		return float64(v)
	}
	keyLen := u("attention.key_length")  // qk_nope_head_dim + qk_rope_head_dim
	ropeDim := u("rope.dimension_count") // qk_rope_head_dim
	sigmoid := u("expert_gating_func") == 2
	modelType := "deepseek_v2"
	scoring := "softmax"
	if sigmoid {
		modelType, scoring = "deepseek_v3", "sigmoid"
	}
	// expert_weights_norm: absent ⇒ V2 false / V3 true (matches the released configs).
	normTopK := sigmoid
	if b, ok := g.Metadata["deepseek2.expert_weights_norm"].(bool); ok {
		normTopK = b
	}
	scale := gf("expert_weights_scale")
	if scale == 0 {
		scale = 1
	}
	cfg := &Config{
		ModelType:           modelType,
		ScoringFunc:         scoring,
		HiddenDim:           u("embedding_length"),
		NumLayers:           u("block_count"),
		NumHeads:            u("attention.head_count"),
		NumKVHeads:          u("attention.head_count_kv"),
		QLoRARank:           u("attention.q_lora_rank"), // absent ⇒ 0 (direct q_proj)
		KVLoRARank:          u("attention.kv_lora_rank"),
		QKRopeHeadDim:       ropeDim,
		QKNopeHeadDim:       keyLen - ropeDim,
		VHeadDim:            u("attention.value_length"),
		IntermediateDim:     u("feed_forward_length"),        // dense prefix width
		MoeIntermediateSize: u("expert_feed_forward_length"), // expert FFN width
		NRoutedExperts:      u("expert_count"),
		NumExpertsPerTok:    u("expert_used_count"),
		NSharedExperts:      u("expert_shared_count"),
		FirstKDenseReplace:  u("leading_dense_block_count"),
		RoutedScalingFactor: scale,
		NGroup:              u("expert_group_count"),      // absent ⇒ 0 (no group limiting)
		TopkGroup:           u("expert_group_used_count"), // absent ⇒ 0
		NormTopKProb:        &normTopK,
		HiddenAct:           "silu",
		VocabSize:           ggufVocabSize(g),
		RMSNormEps:          gf("attention.layer_norm_rms_epsilon"),
		RoPEGlobalBase:      gf("rope.freq_base"),
	}
	// YaRN (V2-Lite, V3): synthesize the rope_scaling object. llama.cpp stores
	// yarn_log_multiplier = 0.1·mscale; DeepSeek sets mscale == mscale_all_dim, so emit
	// both (deepseekArchitecture's ratio collapses to 1.0). The inv-freq ramp uses
	// factor + original_context_length + the default beta_fast/beta_slow.
	if t, _ := g.Str("deepseek2.rope.scaling.type"); t == "yarn" {
		mscale := gf("rope.scaling.yarn_log_multiplier") * 10 // 0.1·mscale ⇒ mscale
		cfg.RopeScaling = json.RawMessage(fmt.Sprintf(
			`{"type":"yarn","factor":%g,"original_max_position_embeddings":%g,"mscale":%g,"mscale_all_dim":%g,"beta_fast":32,"beta_slow":1}`,
			gf("rope.scaling.factor"), gf("rope.scaling.original_context_length"), mscale, mscale))
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufPhi3Config builds a Phi-3/Phi-4 (phi3) Config from the phi3.* metadata. The fused
// attn_qkv + ffn_up (gate‖up) tensors are split by buildWeightsFromGGUF's phi3 branch.
// rope.freq_base is omitted when default (10000); rope.dimension_count < head_dim ⇒ partial
// rotary. LongRoPE (128k variants) is deferred — those store a rope.scaling block that this
// minimal reader doesn't translate (and phi3Architecture would reject).
func ggufPhi3Config(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("phi3." + k)
		return int(v)
	}
	gf := func(k string) float64 {
		if v, ok := g.Float("phi3." + k); ok {
			return v
		}
		v, _ := g.Uint("phi3." + k)
		return float64(v)
	}
	hidden, heads := u("embedding_length"), u("attention.head_count")
	headDim := hidden / heads
	base := gf("rope.freq_base")
	if base == 0 {
		base = 10000 // llama.cpp omits the key at the default
	}
	cfg := &Config{
		ModelType:       "phi3",
		HiddenDim:       hidden,
		NumLayers:       u("block_count"),
		NumHeads:        heads,
		NumKVHeads:      u("attention.head_count_kv"),
		IntermediateDim: u("feed_forward_length"),
		VocabSize:       ggufVocabSize(g),
		RMSNormEps:      gf("attention.layer_norm_rms_epsilon"),
		RoPEGlobalBase:  base,
		MaxPositions:    u("context_length"),
		HiddenAct:       "silu",
	}
	if rd := u("rope.dimension_count"); rd > 0 && rd < headDim {
		cfg.PartialRotaryFactor = float64(rd) / float64(headDim)
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufLlama4Config builds a Llama 4 text-decoder Config from the llama4.* metadata. The
// GGUF carries the dims/expert counts but OMITS the iRoPE scalars (no_rope_layers,
// use_qk_norm, attn_temperature, floor_scale, attn_scale) — llama.cpp applies them by arch
// convention — so they are injected here: NoPE every 4th layer (the interval-4 pattern),
// QK-norm + attention-temperature on (8192, 0.1). interleave_moe_layer_step (in the GGUF)
// drives the dense/MoE split (Scout=1 ⇒ all MoE). The llama3 long-context rope scaling is
// NOT applied (negligible below original_max_position_embeddings; the GGUF stores it as a
// precomputed rope_freqs.weight goinfer doesn't consume) — fine for short-prompt gates.
func ggufLlama4Config(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("llama4." + k)
		return int(v)
	}
	gf := func(k string) float64 {
		if v, ok := g.Float("llama4." + k); ok {
			return v
		}
		v, _ := g.Uint("llama4." + k)
		return float64(v)
	}
	nLayers, err := ggufLayerCount(u("block_count"))
	if err != nil {
		return nil, err
	}
	cfg := &Config{
		ModelType:              "llama4_text",
		HiddenDim:              u("embedding_length"),
		NumLayers:              nLayers,
		NumHeads:               u("attention.head_count"),
		NumKVHeads:             u("attention.head_count_kv"),
		HeadDim:                u("attention.key_length"),
		IntermediateDim:        u("expert_feed_forward_length"), // routed/shared expert width (intermediate_size)
		IntermediateSizeMLP:    u("feed_forward_length"),        // dense-layer width (intermediate_size_mlp)
		NumLocalExperts:        u("expert_count"),
		NumExpertsPerTok:       u("expert_used_count"),
		InterleaveMoeLayerStep: u("interleave_moe_layer_step"),
		VocabSize:              ggufVocabSize(g),
		RMSNormEps:             gf("attention.layer_norm_rms_epsilon"),
		RoPEGlobalBase:         gf("rope.freq_base"),
		HiddenAct:              "silu",
		// Injected iRoPE scalars (absent from the GGUF; llama.cpp arch defaults).
		UseQKNorm:             true,
		AttnTemperatureTuning: true,
		FloorScale:            8192,
		AttnScaleL4:           0.1,
	}
	// no_rope_layers: NoPE on every 4th layer ((i+1)%4==0), else RoPE — the released
	// Scout/Maverick pattern.
	cfg.NoRopeLayers = make([]int, nLayers)
	for i := range cfg.NoRopeLayers {
		if (i+1)%4 == 0 {
			cfg.NoRopeLayers[i] = 0
		} else {
			cfg.NoRopeLayers[i] = 1
		}
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

// loadDeepseekAttnGGUF loads one DeepSeek MLA layer's attention tensors as f32 (the
// parity-first forward uses plain matvec, like the qwen35/granite mixer weights), from
// llama.cpp's deepseek2 names: attn_q_a/attn_q_a_norm/attn_q_b (q-LoRA) or attn_q
// (direct), attn_kv_a_mqa, attn_kv_a_norm, attn_kv_b, attn_output. GGUF stores each matrix
// as dims [in, out] with the data row-major [out, in] — exactly the layout mlaWeights
// expects (and what g.Tensor dequantizes to).
func loadDeepseekAttnGGUF(g *embed.GGUFFile, p string, l *LayerWeights, arch *Architecture) error {
	pa := arch.mla
	qkHeadDim := pa.qkHeadDim()
	f32 := func(name string, out, in int) ([]float32, error) {
		dims, data, err := g.Tensor(name)
		if err != nil {
			return nil, err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return nil, fmt.Errorf("decoder(gguf-deepseek): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		return data, nil
	}
	flat := func(name string, n int) ([]float32, error) {
		dims, data, err := g.Tensor(name)
		if err != nil {
			return nil, err
		}
		if len(data) != n {
			return nil, fmt.Errorf("decoder(gguf-deepseek): %q has %d elems, want %d (dims %v)", name, len(data), n, dims)
		}
		return data, nil
	}
	w := &mlaWeights{}
	var err error
	if pa.QLoRARank > 0 { // q-LoRA bottleneck (V3 671B); V2-Lite/Moonlight are direct-q
		if w.qAProj, err = f32(p+"attn_q_a.weight", pa.QLoRARank, arch.HiddenDim); err != nil {
			return err
		}
		if w.qALayernorm, err = flat(p+"attn_q_a_norm.weight", pa.QLoRARank); err != nil {
			return err
		}
		if w.qBProj, err = f32(p+"attn_q_b.weight", arch.NumHeads*qkHeadDim, pa.QLoRARank); err != nil {
			return err
		}
	} else {
		if w.qProj, err = f32(p+"attn_q.weight", arch.NumHeads*qkHeadDim, arch.HiddenDim); err != nil {
			return err
		}
	}
	if w.kvAProj, err = f32(p+"attn_kv_a_mqa.weight", pa.KVLoRARank+pa.QKRopeHeadDim, arch.HiddenDim); err != nil {
		return err
	}
	if w.kvALayernorm, err = flat(p+"attn_kv_a_norm.weight", pa.KVLoRARank); err != nil {
		return err
	}
	if w.kvBProj, err = f32(p+"attn_kv_b.weight", arch.NumHeads*(pa.QKNopeHeadDim+pa.VHeadDim), pa.KVLoRARank); err != nil {
		return err
	}
	if w.oProj, err = f32(p+"attn_output.weight", arch.HiddenDim, arch.NumHeads*pa.VHeadDim); err != nil {
		return err
	}
	l.mla = w
	return nil
}

// loadGGUFWeights parses a .gguf file and builds the weight bundle, mapping
// llama.cpp tensor names to the descriptor and un-permuting q/k.
func loadGGUFWeights(path string, quant quantMode, embedInt4 bool) (*Weights, error) {
	// mmap, not heap-read: the raw quantized bytes stay in reclaimable page
	// cache while we dequantize tensor-by-tensor. The weights end up as fresh
	// (f32 or int8) copies, so the mapping is unneeded once the build returns.
	g, err := embed.OpenGGUFMmap(path)
	if err != nil {
		return nil, err
	}
	defer g.Close()
	return buildGGUFWeights(g, quant, embedInt4)
}

// buildGGUFWeights builds the weight bundle from an already-open GGUF — whether
// memory-mapped (loadGGUFWeights) or backed by an in-memory slice
// (LoadGGUFBytes). It resolves the config + architecture from the GGUF's own
// metadata, so both entry points produce an identical model.
func buildGGUFWeights(g *embed.GGUFFile, quant quantMode, embedInt4 bool) (*Weights, error) {
	cfg, err := ggufConfig(g)
	if err != nil {
		return nil, err
	}
	if err := validateGGUFDims(cfg); err != nil {
		return nil, err
	}
	arch, _, err := resolveArchitecture(cfg) // llama descriptor + finalizeRoPE
	if err != nil {
		return nil, err
	}
	return buildWeightsFromGGUF(cfg, arch, g, quant, embedInt4, nil, "")
}

// StreamTranscodeGGUF transcodes the GGUF at path into a .giw weights body written
// to out (the GINFW serialization + trailing CRC), loading ONE layer at a time so
// peak RAM is ~one layer rather than the whole model — the streaming analogue of
// Load + SerializeWeightsTo, for models too large to hold resident (the build-time
// hump that otherwise blocks running a >RAM MoE). Generic-loader families stream;
// the qwen35/gemma4 dedicated loaders fall back to a resident build + serialize
// (those models fit). Returns the bytes written. Typically invoked inside
// giw.WriteStream as the weights half of a .giw bundle.
func StreamTranscodeGGUF(ctx context.Context, path string, out io.Writer, quant string, embedInt4 bool, id string) (int64, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	// Cancellation (M-21): a transcode of a >RAM model can run for minutes writing tens
	// of GB. Wrapping the sink makes every per-layer write observe ctx, so a cancelled
	// context aborts at the next layer boundary (the granularity the streaming loop
	// writes at) rather than only after the whole pass. Covers both the streaming path
	// and the qwen35 resident-then-serialize path, which also writes through out.
	out = &ctxWriter{ctx: ctx, w: out}
	q, err := parseQuant(quant)
	if err != nil {
		return 0, err
	}
	g, err := embed.OpenGGUFMmap(path)
	if err != nil {
		return 0, err
	}
	defer g.Close()
	cfg, err := ggufConfig(g)
	if err != nil {
		return 0, err
	}
	if err := validateGGUFDims(cfg); err != nil {
		return 0, err
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		return 0, err
	}
	// Refuse families whose per-layer state the .giw writer can't express (MLA / Mamba-2 / Gemma-4
	// PLE / Llama-4) BEFORE the load — otherwise the granite/nemotron/llama4 branches stream a
	// header declaring N layers followed by zero layers, or gemma4 loads fully then drops its PLE
	// stack, producing a CRC-valid bundle that nil-derefs at the first forward (C2).
	if serr := canSerialize(arch); serr != nil {
		return 0, serr
	}
	// Dedicated-loader families (qwen35) can't stream; they fit resident.
	if arch.qwen35 != nil {
		w, berr := buildWeightsFromGGUF(cfg, arch, g, q, embedInt4, nil, "")
		if berr != nil {
			return 0, berr
		}
		return SerializeWeightsTo(out, w, id)
	}
	wr := &giwWriter{sink: out}
	if _, err := buildWeightsFromGGUF(cfg, arch, g, q, embedInt4, wr, id); err != nil {
		return wr.n, err
	}
	var crc [4]byte
	binary.LittleEndian.PutUint32(crc[:], wr.crc) // running CRC over the streamed body
	if _, err := out.Write(crc[:]); err != nil {
		return wr.n, err
	}
	return wr.n + 4, nil
}

// ctxWriter aborts an in-flight streamed transcode: it returns ctx.Err() before each
// underlying Write once the context is cancelled, so StreamTranscodeGGUF stops writing
// (and the caller removes the partial .giw) instead of running the whole multi-GB pass
// to completion after a shutdown/disconnect (audit M-21).
type ctxWriter struct {
	ctx context.Context
	w   io.Writer
}

func (c *ctxWriter) Write(p []byte) (int, error) {
	if err := c.ctx.Err(); err != nil {
		return 0, err
	}
	return c.w.Write(p)
}

// Generous sanity ceilings for the metadata-derived core dims — orders of
// magnitude above any real checkpoint (largest open models: ~120 layers, hidden
// ~16K, vocab ~256K), but low enough that an untrusted GGUF can't drive a
// pathological allocation. They exist to turn a hostile header into a typed
// error instead of a panic/OOM, not to police legitimate models.
const (
	maxGGUFLayers    = 4096
	maxGGUFHidden    = 1 << 20
	maxGGUFHeads     = 1 << 16
	maxGGUFVocabSize = 1 << 24
)

// validateGGUFDims rejects core dimensions that are out of range before they
// reach an allocation. GGUF reads counts as uint64 and narrows to int, so a
// hostile metadata value (e.g. block_count = 1<<63) silently becomes negative
// and would panic make([]LayerWeights, n) ("makeslice: len out of range"); a
// merely-huge positive value would OOM. Both are caught here. Only the dims
// every family sets from metadata are checked (HeadDim is derived per family;
// IntermediateDim is vestigial/zero for some MoE checkpoints).
// ggufLayerCount bounds a block_count before any per-layer slice is allocated or
// iterated. validateGGUFDims catches an out-of-range count too, but only after the
// family builder has already run — a hostile count would makeslice a multi-TB array
// (a fatal OOM, unrecoverable) or, wrapped negative, panic, before that. Call this
// right after reading block_count in any builder that allocates from it (M16).
func ggufLayerCount(n int) (int, error) {
	if n <= 0 || n > maxGGUFLayers {
		return 0, fmt.Errorf("decoder(gguf): block_count %d out of range (1..%d)", n, maxGGUFLayers)
	}
	return n, nil
}

func validateGGUFDims(cfg *Config) error {
	switch {
	case cfg.NumLayers <= 0 || cfg.NumLayers > maxGGUFLayers:
		return fmt.Errorf("decoder(gguf): block_count %d out of range (1..%d)", cfg.NumLayers, maxGGUFLayers)
	case cfg.HiddenDim <= 0 || cfg.HiddenDim > maxGGUFHidden:
		return fmt.Errorf("decoder(gguf): embedding_length %d out of range (1..%d)", cfg.HiddenDim, maxGGUFHidden)
	case cfg.NumHeads <= 0 || cfg.NumHeads > maxGGUFHeads:
		return fmt.Errorf("decoder(gguf): head_count %d out of range (1..%d)", cfg.NumHeads, maxGGUFHeads)
	case cfg.NumKVHeads <= 0 || cfg.NumKVHeads > maxGGUFHeads:
		return fmt.Errorf("decoder(gguf): head_count_kv %d out of range (1..%d)", cfg.NumKVHeads, maxGGUFHeads)
	case cfg.VocabSize <= 0 || cfg.VocabSize > maxGGUFVocabSize:
		return fmt.Errorf("decoder(gguf): vocab_size %d out of range (1..%d)", cfg.VocabSize, maxGGUFVocabSize)
	}
	return nil
}

// LoadGGUFBytes loads a GGUF model from an in-memory slice (e.g. an inflated
// //go:embed asset) — same result as Load on the equivalent .gguf file, but
// nothing touches the filesystem (no temp file, no writable disk needed). raw
// is parsed in place and only needs to live until this returns; the built
// weights are fresh copies.
//
// EOS ids come from the GGUF's own tokenizer metadata (there is no directory to
// read a generation_config.json from), matching the file path's resolution for
// a bare .gguf.
func LoadGGUFBytes(raw []byte, opts Options) (*Model, error) {
	be, beErr := NewBackend(opts.Backend)
	quant, err := parseQuant(opts.Quant)
	if err != nil {
		closeBackend(be)
		return nil, err
	}
	g, err := embed.OpenGGUFBytes(raw)
	if err != nil {
		closeBackend(be)
		return nil, err
	}
	w, err := buildGGUFWeights(g, quant, opts.EmbedInt4)
	g.Close()
	if err != nil {
		closeBackend(be)
		return nil, err
	}
	if beErr != nil {
		fmt.Fprintln(os.Stderr, beErr) // webgpu requested but fell back — not fatal (N-03: stderr, not stdout — never contaminate a piped token stream)
	}
	return (&Model{w: w, be: be, quant: opts.Quant, eosIDs: w.Cfg.EOSIDs(), kvF16: opts.KVPrecision == "f16", kvPrecI8: opts.KVPrecision == "i8", kvI8: opts.KVQuant == "i8", resCtxReq: opts.ResidentContext, moeCache: opts.MoECacheExperts, moeSlots: opts.MoECacheSlots}).withResidency(), nil
}

// buildWeightsFromGGUF dequantizes the GGUF tensors into the weight bundle.
// When quant is set, each matmul tensor is re-quantized (per-row int8 or
// group-wise int4) right after it is dequantized (and un-permuted) and its f32
// is freed — so a Q4/Q8 GGUF lands resident as int8/int4 (~¼ / ~⅛ f32) without
// ever materializing the whole model in f32 (see loadWeights). The GGUF's own
// quant is lossy and so is the re-quant, but it captures nearly all of what a
// Q4_K_M file carries.
// When sink is non-nil, the weights are STREAMED to it (a .giw body) instead of
// retained: the header + globals are written, then each layer is loaded, written,
// and freed in turn, so peak RAM is ~one layer rather than the whole model — this is
// what lets a model larger than RAM be transcoded. The returned *Weights then holds
// no layer tensors (they were freed); the caller writes the trailing CRC. Streaming
// is supported for the generic per-layer loader (llama/qwen2/qwen3/mellum/glm4_moe);
// the qwen35/gemma4 dedicated paths reject it (those models fit resident).
func buildWeightsFromGGUF(cfg *Config, arch *Architecture, g *embed.GGUFFile, quant quantMode, embedInt4 bool, sink *giwWriter, id string) (*Weights, error) {
	hidden, hd := arch.HiddenDim, arch.HeadDim
	w := &Weights{Cfg: *cfg, arch: arch, Layers: make([]LayerWeights, arch.NumLayers)}

	// streamMat builds a [out, in] linalg.WeightMat by streaming the tensor's rows through
	// the GGUF dequantizer and quantizing each row directly into the resident
	// arrays — no whole-tensor f32 intermediate (see streamQuantized). rowSrc maps
	// a destination row index to its source element offset in the tensor (identity
	// for a plain load; a permutation for the RoPE-permuted q/k projections).
	streamMat := func(name string, out, in int, mode quantMode, rowSrc func(r int) int) (linalg.WeightMat, error) {
		dims, into, err := g.RowDequantizer(name)
		if err != nil {
			return linalg.WeightMat{}, err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return linalg.WeightMat{}, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		return streamQuantized(out, in, mode, func(r int, dst []float32) error {
			return into(rowSrc(r), dst)
		})
	}
	// mat loads a tensor as a [out, in] linalg.WeightMat, shape-checked, quantized when
	// requested.
	mat := func(name string, out, in int) (linalg.WeightMat, error) {
		return streamMat(name, out, in, matmulQuant(quant, name), func(r int) int { return r * in })
	}
	// embMat loads the embedding / LM head, which is logit-critical — quantize
	// it with the embedding policy (int8 even in int4 mode), not the projection
	// mode.
	embMat := func(name string, out, in int) (linalg.WeightMat, error) {
		return streamMat(name, out, in, quant.embeddingWith(embedInt4), func(r int) int { return r * in })
	}
	// vec loads a 1-D tensor (norm or bias).
	vec := func(name string, n int) ([]float32, error) {
		dims, data, err := g.Tensor(name)
		if err != nil {
			return nil, err
		}
		if len(dims) != 1 || dims[0] != n {
			return nil, fmt.Errorf("decoder(gguf): %q dims %v, want [%d]", name, dims, n)
		}
		return data, nil
	}
	// vnorm loads a norm weight. For the Gemma family (RMSAddOne), llama.cpp's GGUF
	// conversion bakes the (1+w) offset into every stored *norm.weight, so this
	// package's RMSAddOne forward would apply it twice — subtract the 1 back out at
	// load to restore the HF convention the shared descriptor expects. No-op for the
	// other architectures (plain RMSNorm), so all norm loads route through here.
	subOneNorm := arch.RMSAddOne
	vnorm := func(name string, n int) ([]float32, error) {
		v, err := vec(name, n)
		if err != nil {
			return nil, err
		}
		if subOneNorm {
			for i := range v {
				v[i] -= 1
			}
		}
		return v, nil
	}
	// permMat loads a q/k projection and inverts llama.cpp's RoPE row permutation.
	// Because the permutation is a pure reorder of whole rows (output features), it
	// commutes with per-row quantization: rather than permute an f32 buffer then
	// quantize, we dequant the rows straight into HF order — destination row hfRow
	// pulls from GGUF source row (h*hd + 2*j + s) — and quantize in place. See
	// ggufInvPermute for the index derivation.
	permMat := func(name string, out, in, nHead int) (linalg.WeightMat, error) {
		// nHead is a validated head count (>0), but a hostile head_dim can still make
		// out/nHead == 1 (half == 0 ⇒ divide-by-zero in the row map below) or odd (the
		// RoPE row permutation is only defined for an even head_dim) — M16.
		if nHead <= 0 || out%nHead != 0 || (out/nHead)%2 != 0 {
			return linalg.WeightMat{}, fmt.Errorf("decoder(gguf): %q needs an even head_dim (out=%d, heads=%d)", name, out, nHead)
		}
		hd := out / nHead
		half := hd / 2
		return streamMat(name, out, in, matmulQuant(quant, name), func(hfRow int) int {
			h, rem := hfRow/hd, hfRow%hd
			ggufRow := h*hd + 2*(rem%half) + rem/half
			return ggufRow * in
		})
	}
	// stackedExperts loads a GGUF MoE expert tensor — a 3-D [in, out, nExpert]
	// (fastest-first) blob where each expert occupies a contiguous [out, in]
	// row-major slice — and returns one quantized linalg.WeightMat per expert.
	stackedExperts := func(name string, out, in, nExpert int) ([]linalg.WeightMat, error) {
		dims, into, err := g.RowDequantizer(name)
		if err != nil {
			return nil, err
		}
		if len(dims) != 3 || dims[0] != in || dims[1] != out || dims[2] != nExpert {
			return nil, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d, experts=%d]", name, dims, in, out, nExpert)
		}
		// Each expert occupies a contiguous [out, in] row-major slice; stream its
		// rows directly into a per-expert quantized linalg.WeightMat (no whole-tensor f32).
		res := make([]linalg.WeightMat, nExpert)
		for e := range nExpert {
			m, err := streamQuantized(out, in, matmulQuant(quant, name), func(r int, dst []float32) error {
				return into((e*out+r)*in, dst)
			})
			if err != nil {
				return nil, err
			}
			res[e] = m
		}
		return res, nil
	}
	// fusedSplit streams the output-row range [rowStart, rowStart+rows) of a fused
	// [outTotal, in] tensor into its own quantized linalg.WeightMat — Phi-3's fused
	// attn_qkv (q‖k‖v) and ffn_up (gate‖up), split without a whole-tensor f32 buffer.
	fusedSplit := func(name string, outTotal, in, rowStart, rows int) (linalg.WeightMat, error) {
		dims, into, err := g.RowDequantizer(name)
		if err != nil {
			return linalg.WeightMat{}, err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != outTotal {
			return linalg.WeightMat{}, fmt.Errorf("decoder(gguf-phi3): %q dims %v, want [in=%d, out=%d]", name, dims, in, outTotal)
		}
		return streamQuantized(rows, in, matmulQuant(quant, name), func(r int, dst []float32) error {
			return into((rowStart+r)*in, dst)
		})
	}

	var err error
	if w.Embed, err = embMat("token_embd.weight", cfg.VocabSize, hidden); err != nil {
		return nil, err
	}
	if w.FinalNorm, err = vnorm("output_norm.weight", hidden); err != nil {
		return nil, err
	}
	// Separate output head when present; else tied to the embedding.
	arch.TiedLMHead = true
	if g.Has("output.weight") {
		if w.LMHead, err = embMat("output.weight", cfg.VocabSize, hidden); err != nil {
			return nil, err
		}
		arch.TiedLMHead = false
	}

	// Streaming transcode: emit the header + globals now; each layer is written and
	// freed as it loads (below), so the whole model never sits resident.
	if sink != nil {
		if arch.qwen35 != nil || arch.gemma4 != nil {
			return nil, fmt.Errorf("decoder(gguf): streaming transcode unsupported for %s (load resident + prequant instead)", arch.Name)
		}
		if err := sink.writeHeadGlobals(w, id); err != nil {
			return nil, err
		}
	}

	// qwen3_5_moe (qwen35moe): the Gated DeltaNet / gated-softmax hybrid. The GGUF
	// converter bakes layout transforms the HF safetensors path never sees — V
	// heads tiled (num_v>num_k), A_log stored as −exp(A_log), the standard norms
	// (1+w)'d (ssm_norm exempt) — all reversed here so the shared forward runs
	// unchanged. Attention stays f32 (parity-first, like the safetensors path);
	// experts/embeddings quantize. Only blk.{0..NumLayers-1} load — the trailing
	// NextN/MTP block is excluded by ggufQwen35Config's NumLayers.
	if q := arch.qwen35; q != nil {
		numK, numV := q.NumKeyHeads, q.NumValueHeads
		vpk := numV / numK
		hkd, hvd := q.KeyHeadDim, q.ValueHeadDim
		keyDim, valueDim := hkd*numK, hvd*numV
		convDim := 2*keyDim + valueDim
		// f32mat dequantizes a [out, in] tensor to a row-major f32 slice. Some of these need a
		// layout transform (untileVHeads) before they can be used, so the f32 form is still the
		// working intermediate; wmQ below wraps the result and applies Options.Quant.
		f32mat := func(name string, out, in int) ([]float32, error) {
			dims, data, derr := g.Tensor(name)
			if derr != nil {
				return nil, derr
			}
			if len(dims) != 2 || dims[0] != in || dims[1] != out {
				return nil, fmt.Errorf("decoder(gguf-qwen35): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
			}
			return data, nil
		}
		// wmQ is f32mat + Options.Quant, for the projections that dominate decode bandwidth. Until
		// 2026-08-19 this path (like its safetensors twin) kept them f32 regardless of the requested
		// quant — "parity-first" from the bring-up — which on a 27.8B Qwen3.8 meant ~29 GB of f32
		// weights streamed per token while the FFN was int4. Transform-then-quantize: the untile
		// below has to see f32.
		wmQ := func(name string, out, in int) (linalg.WeightMat, error) {
			f, e := f32mat(name, out, in)
			if e != nil {
				return linalg.WeightMat{}, e
			}
			return quantizeWM(linalg.WrapF32(f, out, in), quant), nil
		}
		wrapQ := func(f []float32, out, in int) linalg.WeightMat {
			return quantizeWM(linalg.WrapF32(f, out, in), quant)
		}
		loadQ35 := func(i int) error {
			l := &w.Layers[i]
			p := fmt.Sprintf("blk.%d.", i)
			var e error
			// attn_norm = input_layernorm (pre-attn); post_attention_norm =
			// post_attention_layernorm (pre-MLP, Pre2). Both (1+w)'d → vnorm subtracts.
			if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
				return e
			}
			if l.PreMLPNorm, e = vnorm(p+"post_attention_norm.weight", hidden); e != nil {
				return e
			}
			if arch.isLinearLayer(i) {
				d := &deltaNetWeights{}
				// in_proj_qkv (attn_qkv): only the V output rows are tiled; q/k untouched.
				qkv, err := f32mat(p+"attn_qkv.weight", convDim, hidden)
				if err != nil {
					return err
				}
				vOff := 2 * keyDim * hidden
				copy(qkv[vOff:], untileVHeads(qkv[vOff:], 1, numK, vpk, hvd, hidden))
				d.inProjQKV = wrapQ(qkv, convDim, hidden)
				// in_proj_z (attn_gate): all rows tiled.
				z, err := f32mat(p+"attn_gate.weight", valueDim, hidden)
				if err != nil {
					return err
				}
				d.inProjZ = wrapQ(untileVHeads(z, 1, numK, vpk, hvd, hidden), valueDim, hidden)
				// in_proj_a / in_proj_b (ssm_alpha/ssm_beta): one row per value head (hd=1).
				a, err := f32mat(p+"ssm_alpha.weight", numV, hidden)
				if err != nil {
					return err
				}
				d.inProjA = untileVHeads(a, 1, numK, vpk, 1, hidden)
				b, err := f32mat(p+"ssm_beta.weight", numV, hidden)
				if err != nil {
					return err
				}
				d.inProjB = untileVHeads(b, 1, numK, vpk, 1, hidden)
				// conv1d (ssm_conv1d): only the V channel block tiled; trail = kernel width.
				cw, err := f32mat(p+"ssm_conv1d.weight", convDim, q.ConvKernel)
				if err != nil {
					return err
				}
				vOffC := 2 * keyDim * q.ConvKernel
				copy(cw[vOffC:], untileVHeads(cw[vOffC:], 1, numK, vpk, hvd, q.ConvKernel))
				d.convW = cw
				// ssm_a is already −exp(A_log); reorder by value head (1-D).
				av, err := vec(p+"ssm_a", numV)
				if err != nil {
					return err
				}
				d.negExpA = untileVHeads(av, 1, numK, vpk, 1, 1)
				// dt_bias (ssm_dt.bias): reorder by value head (1-D).
				dt, err := vec(p+"ssm_dt.bias", numV)
				if err != nil {
					return err
				}
				d.dtBias = untileVHeads(dt, 1, numK, vpk, 1, 1)
				// ssm_norm: per-head_v_dim gated-RMSNorm weight — NOT tiled and NOT
				// (1+w)'d by the converter, so load raw (vec, not vnorm).
				if d.normW, err = vec(p+"ssm_norm.weight", hvd); err != nil {
					return err
				}
				// out_proj (ssm_out): the value/input columns are tiled.
				op, err := f32mat(p+"ssm_out.weight", hidden, valueDim)
				if err != nil {
					return err
				}
				d.outProj = wrapQ(untileVHeads(op, hidden, numK, vpk, hvd, 1), hidden, valueDim)
				l.delta = d
			} else {
				// Gated softmax layer: double-width q_proj (query ‖ gate per head, kept
				// fused in attn_q), per-head QK-norm. NEOX rope ⇒ no q/k permute.
				a := &qwenAttnWeights{}
				shd := arch.HeadDim
				kvDim := arch.NumKVHeads * shd
				if a.qProj, e = wmQ(p+"attn_q.weight", arch.NumHeads*shd*2, hidden); e != nil {
					return e
				}
				if a.kProj, e = wmQ(p+"attn_k.weight", kvDim, hidden); e != nil {
					return e
				}
				if a.vProj, e = wmQ(p+"attn_v.weight", kvDim, hidden); e != nil {
					return e
				}
				if a.oProj, e = wmQ(p+"attn_output.weight", hidden, arch.NumHeads*shd); e != nil {
					return e
				}
				if a.qNorm, e = vnorm(p+"attn_q_norm.weight", shd); e != nil {
					return e
				}
				if a.kNorm, e = vnorm(p+"attn_k_norm.weight", shd); e != nil {
					return e
				}
				l.qattn = a
			}
			// MoE FFN (every layer): router f32 (matching safetensors), 256 stacked
			// experts quantized, shared expert quantized + f32 sigmoid gate.
			if l.Router, e = streamMat(p+"ffn_gate_inp.weight", arch.MoE.NumExperts, hidden, quantNone, func(r int) int { return r * hidden }); e != nil {
				return e
			}
			expInter := arch.MoE.IntermediateDim
			gate, gerr := stackedExperts(p+"ffn_gate_exps.weight", expInter, hidden, arch.MoE.NumExperts)
			up, uerr := stackedExperts(p+"ffn_up_exps.weight", expInter, hidden, arch.MoE.NumExperts)
			down, derr := stackedExperts(p+"ffn_down_exps.weight", hidden, expInter, arch.MoE.NumExperts)
			if gerr != nil || uerr != nil || derr != nil {
				return fmt.Errorf("decoder(gguf-qwen35): experts layer %d: %v / %v / %v", i, gerr, uerr, derr)
			}
			l.Experts = make([]expertWeights, arch.MoE.NumExperts)
			for ei := range arch.MoE.NumExperts {
				l.Experts[ei] = expertWeights{Gate: gate[ei], Up: up[ei], Down: down[ei]}
			}
			sInter := arch.MoE.SharedIntermediateDim
			if l.SharedExpert.Gate, e = mat(p+"ffn_gate_shexp.weight", sInter, hidden); e != nil {
				return e
			}
			if l.SharedExpert.Up, e = mat(p+"ffn_up_shexp.weight", sInter, hidden); e != nil {
				return e
			}
			if l.SharedExpert.Down, e = mat(p+"ffn_down_shexp.weight", hidden, sInter); e != nil {
				return e
			}
			// shared_expert_gate (ffn_gate_inp_shexp): 1-D [hidden] → [1, hidden] f32.
			sg, err := vec(p+"ffn_gate_inp_shexp.weight", hidden)
			if err != nil {
				return err
			}
			l.SharedGate = linalg.WrapF32(sg, 1, hidden)
			return nil
		}
		if err := parallelLayers(arch.NumLayers, loadQ35); err != nil {
			return nil, err
		}
		return w, nil
	}

	// gpt-oss: sparse MoE on every layer with per-head attention sinks + per-expert
	// biases + a router-logit bias. q/k/v/o all carry biases; no QK-norm. Attention/
	// embeddings/router quantize per policy; the (F32) sinks + biases load raw. The
	// expert weights are MXFP4 in the real checkpoint — stackedExperts routes them
	// through aikit's RowDequantizer, so this loads once aikit dequants ggml type 39
	// (until then a Q8_0/F32 tiny model exercises the same path).
	if arch.gptoss != nil {
		nH, nKV := arch.NumHeads, arch.NumKVHeads
		qDim, kvDim := nH*hd, nKV*hd
		nExp := arch.MoE.NumExperts
		expInter := arch.MoE.IntermediateDim
		// stackedExpertBias slices a 2-D [out, nExpert] (fastest-first) f32 bias
		// tensor into one [out] slice per expert (expert e occupies [e*out, e*out+out)).
		stackedExpertBias := func(name string, out int) ([][]float32, error) {
			dims, data, derr := g.Tensor(name)
			if derr != nil {
				return nil, derr
			}
			if len(dims) != 2 || dims[0] != out || dims[1] != nExp {
				return nil, fmt.Errorf("decoder(gguf-gptoss): %q dims %v, want [out=%d, experts=%d]", name, dims, out, nExp)
			}
			res := make([][]float32, nExp)
			for e := range nExp {
				res[e] = append([]float32(nil), data[e*out:(e+1)*out]...)
			}
			return res, nil
		}
		loadGptOss := func(i int) error {
			l := &w.Layers[i]
			p := fmt.Sprintf("blk.%d.", i)
			var e error
			if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
				return e
			}
			if l.PreMLPNorm, e = vnorm(p+"post_attention_norm.weight", hidden); e != nil {
				return e
			}
			// Attention: q/k/v/o projections + biases. gpt-oss GGUF stores q/k in HF
			// order (no undo_permute), so a plain mat load — no permMat.
			if l.QProj, e = mat(p+"attn_q.weight", qDim, hidden); e != nil {
				return e
			}
			if l.KProj, e = mat(p+"attn_k.weight", kvDim, hidden); e != nil {
				return e
			}
			if l.VProj, e = mat(p+"attn_v.weight", kvDim, hidden); e != nil {
				return e
			}
			if l.OProj, e = mat(p+"attn_output.weight", hidden, qDim); e != nil {
				return e
			}
			if l.QBias, e = vec(p+"attn_q.bias", qDim); e != nil {
				return e
			}
			if l.KBias, e = vec(p+"attn_k.bias", kvDim); e != nil {
				return e
			}
			if l.VBias, e = vec(p+"attn_v.bias", kvDim); e != nil {
				return e
			}
			if l.OBias, e = vec(p+"attn_output.bias", hidden); e != nil {
				return e
			}
			if l.AttnSinks, e = vec(p+"attn_sinks.weight", nH); e != nil {
				return e
			}
			// MoE: router (+ logit bias) + stacked routed experts (+ per-expert biases). Router
			// stays f32 regardless of the ambient quant mode (matching gptoss_safetensors.go and
			// qwen35's streamMat(..., quantNone, ...) below) — top-k selection is discrete, so
			// quantizing it flips which experts win rather than adding rounding noise. Plain mat()
			// here was a latent bug: it had never been exercised at a non-f32 quant until Metal
			// residency's f32Mat(router) panic caught it.
			if l.Router, e = streamMat(p+"ffn_gate_inp.weight", nExp, hidden, quantNone, func(r int) int { return r * hidden }); e != nil {
				return e
			}
			if l.RouterBias, e = vec(p+"ffn_gate_inp.bias", nExp); e != nil {
				return e
			}
			gate, ge := stackedExperts(p+"ffn_gate_exps.weight", expInter, hidden, nExp)
			up, ue := stackedExperts(p+"ffn_up_exps.weight", expInter, hidden, nExp)
			down, de := stackedExperts(p+"ffn_down_exps.weight", hidden, expInter, nExp)
			if ge != nil || ue != nil || de != nil {
				return fmt.Errorf("decoder(gguf-gptoss): experts layer %d: %v / %v / %v", i, ge, ue, de)
			}
			gb, gbe := stackedExpertBias(p+"ffn_gate_exps.bias", expInter)
			ub, ube := stackedExpertBias(p+"ffn_up_exps.bias", expInter)
			db, dbe := stackedExpertBias(p+"ffn_down_exps.bias", hidden)
			if gbe != nil || ube != nil || dbe != nil {
				return fmt.Errorf("decoder(gguf-gptoss): expert biases layer %d: %v / %v / %v", i, gbe, ube, dbe)
			}
			l.Experts = make([]expertWeights, nExp)
			for ei := range nExp {
				l.Experts[ei] = expertWeights{
					Gate: gate[ei], Up: up[ei], Down: down[ei],
					GateBias: gb[ei], UpBias: ub[ei], DownBias: db[ei],
				}
			}
			return nil
		}
		if err := parallelLayers(arch.NumLayers, loadGptOss); err != nil {
			return nil, err
		}
		return w, nil
	}

	// Granite-4.0-H (granitehybrid): per-layer Mamba-2 mixer or GQA attention, MoE on
	// every layer. The Mamba-2 ssm_* tensors use llama.cpp's conventions (shared with
	// qwen35): ssm_a stores −exp(A_log) directly (reversed to A_log via log(−a) so the
	// shared mamba2Step works), conv1d is [convDim, K], ssm_norm raw. Mixer stays f32
	// (parity-first); experts/attention/embeddings quantize. NEOX rope ⇒ no q/k permute.
	// Laguna (poolside). llama.cpp names its tensors the way every other MoE family
	// here is named, so the only genuinely new one is blk.N.attn_gate.weight — the
	// softplus output gate. Two shape details drive the rest:
	//
	//   * qDim is PER LAYER (arch.headsAt(i)): full-attention layers project 48 heads
	//     and sliding layers 64 on the XS line, so attn_q/attn_output differ by layer.
	//   * the gate's granularity is read from attn_gate's ROW COUNT, exactly as the
	//     safetensors loader does, because neither format carries a trustworthy
	//     declaration of it (XS.2's config says per-element and ships per-head).
	//
	// Experts are FUSED+STACKED per projection (ffn_*_exps), the shared expert is
	// *_shexp, and the router bias is exp_probs_b — all shapes goinfer already reads
	// for GLM/DeepSeek/Granite. Layers below FirstKDense are plain dense FFNs.
	if arch.laguna != nil {
		kvDim := arch.NumKVHeads * arch.HeadDim
		// exp_probs_b is 1-D; the shared `mat` helper wants 2-D, so read it as a flat
		// vector the same way the other MoE GGUF loaders do.
		flat := func(name string, n int) ([]float32, error) { // 1-D or [1,n]/[n,1] → n floats
			_, data, derr := g.Tensor(name)
			if derr != nil {
				return nil, derr
			}
			if len(data) != n {
				return nil, fmt.Errorf("decoder(gguf-laguna): %q has %d elems, want %d", name, len(data), n)
			}
			return data, nil
		}
		loadLaguna := func(i int) error {
			l := &w.Layers[i]
			p := fmt.Sprintf("blk.%d.", i)
			lqDim := arch.headsAt(i) * arch.HeadDim
			var e error
			if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
				return e
			}
			if l.PreMLPNorm, e = vnorm(p+"ffn_norm.weight", hidden); e != nil {
				return e
			}
			if l.QProj, e = mat(p+"attn_q.weight", lqDim, hidden); e != nil {
				return e
			}
			if l.KProj, e = mat(p+"attn_k.weight", kvDim, hidden); e != nil {
				return e
			}
			if l.VProj, e = mat(p+"attn_v.weight", kvDim, hidden); e != nil {
				return e
			}
			if l.OProj, e = mat(p+"attn_output.weight", hidden, lqDim); e != nil {
				return e
			}
			if l.QNorm, e = vnorm(p+"attn_q_norm.weight", arch.HeadDim); e != nil {
				return e
			}
			if l.KNorm, e = vnorm(p+"attn_k_norm.weight", arch.HeadDim); e != nil {
				return e
			}
			// The gate: per-head (headsAt rows) or per-element (headsAt*headDim). Try the
			// per-head shape first and fall back, so both are accepted and anything else
			// is a hard error rather than a silently mis-shaped gate.
			if l.GProj, e = mat(p+"attn_gate.weight", arch.headsAt(i), hidden); e != nil {
				if l.GProj, e = mat(p+"attn_gate.weight", lqDim, hidden); e != nil {
					return fmt.Errorf("decoder(gguf-laguna): layer %d attn_gate is neither per-head [%d,%d] nor per-element [%d,%d]: %w",
						i, arch.headsAt(i), hidden, lqDim, hidden, e)
				}
			}
			if i < arch.FirstKDense {
				// Dense prefix layer: a plain SwiGLU at the model's feed_forward_length.
				if l.GateProj, e = mat(p+"ffn_gate.weight", arch.IntermediateDim, hidden); e != nil {
					return e
				}
				if l.UpProj, e = mat(p+"ffn_up.weight", arch.IntermediateDim, hidden); e != nil {
					return e
				}
				if l.DownProj, e = mat(p+"ffn_down.weight", hidden, arch.IntermediateDim); e != nil {
					return e
				}
				return nil
			}
			moe := arch.MoE
			if l.Router, e = mat(p+"ffn_gate_inp.weight", moe.NumExperts, hidden); e != nil {
				return e
			}
			// exp_probs_b is llama.cpp's name for e_score_correction_bias.
			if l.RouterBias, e = flat(p+"exp_probs_b.bias", moe.NumExperts); e != nil {
				return e
			}
			expInter := moe.IntermediateDim
			gate, ge := stackedExperts(p+"ffn_gate_exps.weight", expInter, hidden, moe.NumExperts)
			up, ue := stackedExperts(p+"ffn_up_exps.weight", expInter, hidden, moe.NumExperts)
			down, de := stackedExperts(p+"ffn_down_exps.weight", hidden, expInter, moe.NumExperts)
			if ge != nil || ue != nil || de != nil {
				return fmt.Errorf("decoder(gguf-laguna): experts layer %d: %v / %v / %v", i, ge, ue, de)
			}
			l.Experts = make([]expertWeights, moe.NumExperts)
			for ei := range moe.NumExperts {
				l.Experts[ei] = expertWeights{Gate: gate[ei], Up: up[ei], Down: down[ei]}
			}
			if sInter := moe.SharedIntermediateDim; sInter > 0 {
				if l.SharedExpert.Gate, e = mat(p+"ffn_gate_shexp.weight", sInter, hidden); e != nil {
					return e
				}
				if l.SharedExpert.Up, e = mat(p+"ffn_up_shexp.weight", sInter, hidden); e != nil {
					return e
				}
				if l.SharedExpert.Down, e = mat(p+"ffn_down_shexp.weight", hidden, sInter); e != nil {
					return e
				}
			}
			return nil
		}
		if err := parallelLayers(arch.NumLayers, loadLaguna); err != nil {
			return nil, err
		}
		return w, nil
	}

	if gp := arch.granite; gp != nil {
		dInner := gp.NHeads * gp.HeadDim
		convDim := dInner + 2*gp.NGroups*gp.DState
		projDim := 2*dInner + 2*gp.NGroups*gp.DState + gp.NHeads
		hd := arch.HeadDim
		qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
		f32mat := func(name string, out, in int) ([]float32, error) {
			dims, data, derr := g.Tensor(name)
			if derr != nil {
				return nil, derr
			}
			if len(dims) != 2 || dims[0] != in || dims[1] != out {
				return nil, fmt.Errorf("decoder(gguf-granite): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
			}
			return data, nil
		}
		flat := func(name string, n int) ([]float32, error) { // 1-D or [1,n]/[n,1] tensor → n floats
			_, data, derr := g.Tensor(name)
			if derr != nil {
				return nil, derr
			}
			if len(data) != n {
				return nil, fmt.Errorf("decoder(gguf-granite): %q has %d elems, want %d", name, len(data), n)
			}
			return data, nil
		}
		loadGranite := func(i int) error {
			l := &w.Layers[i]
			p := fmt.Sprintf("blk.%d.", i)
			var e error
			if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
				return e
			}
			if l.PreMLPNorm, e = vnorm(p+"ffn_norm.weight", hidden); e != nil {
				return e
			}
			if arch.isMambaLayer(i) {
				mw := &mamba2Weights{}
				if mw.inProj, e = f32mat(p+"ssm_in.weight", projDim, hidden); e != nil {
					return e
				}
				if mw.convW, e = f32mat(p+"ssm_conv1d.weight", convDim, gp.DConv); e != nil {
					return e
				}
				if mw.convB, e = flat(p+"ssm_conv1d.bias", convDim); e != nil {
					return e
				}
				if mw.d, e = flat(p+"ssm_d", gp.NHeads); e != nil {
					return e
				}
				if mw.dtBias, e = flat(p+"ssm_dt.bias", gp.NHeads); e != nil {
					return e
				}
				if mw.normW, e = flat(p+"ssm_norm.weight", dInner); e != nil {
					return e
				}
				if mw.outProj, e = f32mat(p+"ssm_out.weight", hidden, dInner); e != nil {
					return e
				}
				// ssm_a is −exp(A_log); reverse to A_log so mamba2Step's −exp(aLog) recovers it.
				a, aerr := flat(p+"ssm_a", gp.NHeads)
				if aerr != nil {
					return aerr
				}
				mw.aLog = make([]float32, gp.NHeads)
				for h := range a {
					mw.aLog[h] = float32(math.Log(-float64(a[h])))
				}
				l.mamba = mw
			} else {
				// llama.cpp converts GraniteHybrid attention with undo_permute=True, so
				// q/k are RoPE-permuted in the GGUF (like Llama) and must be un-permuted.
				if l.QProj, e = permMat(p+"attn_q.weight", qDim, hidden, arch.NumHeads); e != nil {
					return e
				}
				if l.KProj, e = permMat(p+"attn_k.weight", kvDim, hidden, arch.NumKVHeads); e != nil {
					return e
				}
				if l.VProj, e = mat(p+"attn_v.weight", kvDim, hidden); e != nil {
					return e
				}
				if l.OProj, e = mat(p+"attn_output.weight", hidden, qDim); e != nil {
					return e
				}
			}
			// MoE on every layer: router + stacked routed experts + ungated shared.
			expInter := arch.MoE.IntermediateDim
			if l.Router, e = mat(p+"ffn_gate_inp.weight", arch.MoE.NumExperts, hidden); e != nil {
				return e
			}
			gate, ge := stackedExperts(p+"ffn_gate_exps.weight", expInter, hidden, arch.MoE.NumExperts)
			up, ue := stackedExperts(p+"ffn_up_exps.weight", expInter, hidden, arch.MoE.NumExperts)
			down, de := stackedExperts(p+"ffn_down_exps.weight", hidden, expInter, arch.MoE.NumExperts)
			if ge != nil || ue != nil || de != nil {
				return fmt.Errorf("decoder(gguf-granite): experts layer %d: %v / %v / %v", i, ge, ue, de)
			}
			l.Experts = make([]expertWeights, arch.MoE.NumExperts)
			for ei := range arch.MoE.NumExperts {
				l.Experts[ei] = expertWeights{Gate: gate[ei], Up: up[ei], Down: down[ei]}
			}
			sInter := arch.MoE.SharedIntermediateDim
			if l.SharedExpert.Gate, e = mat(p+"ffn_gate_shexp.weight", sInter, hidden); e != nil {
				return e
			}
			if l.SharedExpert.Up, e = mat(p+"ffn_up_shexp.weight", sInter, hidden); e != nil {
				return e
			}
			if l.SharedExpert.Down, e = mat(p+"ffn_down_shexp.weight", hidden, sInter); e != nil {
				return e
			}
			return nil
		}
		if err := parallelLayers(arch.NumLayers, loadGranite); err != nil {
			return nil, err
		}
		return w, nil
	}

	// Nemotron-H (nemotron_h): single-op-per-block hybrid. Each layer is one of
	// mamba / attention (NoPE — no q/k permute) / non-gated mlp, under a single
	// attn_norm. The ssm_* tensors reuse llama.cpp's conventions (ssm_a = −exp(A_log),
	// conv [convDim,K], grouped ssm_norm). Mixer f32; attention/mlp/embeddings quantize.
	if npar := arch.nemotron; npar != nil {
		dInner := npar.NHeads * npar.HeadDim
		convDim := dInner + 2*npar.NGroups*npar.DState
		projDim := 2*dInner + 2*npar.NGroups*npar.DState + npar.NHeads
		hd := arch.HeadDim
		qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
		inter := arch.IntermediateDim
		f32mat := func(name string, out, in int) ([]float32, error) {
			dims, data, derr := g.Tensor(name)
			if derr != nil {
				return nil, derr
			}
			if len(dims) != 2 || dims[0] != in || dims[1] != out {
				return nil, fmt.Errorf("decoder(gguf-nemotron): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
			}
			return data, nil
		}
		flat := func(name string, n int) ([]float32, error) {
			_, data, derr := g.Tensor(name)
			if derr != nil {
				return nil, derr
			}
			if len(data) != n {
				return nil, fmt.Errorf("decoder(gguf-nemotron): %q has %d elems, want %d", name, len(data), n)
			}
			return data, nil
		}
		loadNemo := func(i int) error {
			l := &w.Layers[i]
			p := fmt.Sprintf("blk.%d.", i)
			var e error
			if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
				return e
			}
			switch npar.blockKind[i] {
			case nemoMamba:
				mw := &mamba2Weights{}
				if mw.inProj, e = f32mat(p+"ssm_in.weight", projDim, hidden); e != nil {
					return e
				}
				if mw.convW, e = f32mat(p+"ssm_conv1d.weight", convDim, npar.DConv); e != nil {
					return e
				}
				if mw.convB, e = flat(p+"ssm_conv1d.bias", convDim); e != nil {
					return e
				}
				if mw.d, e = flat(p+"ssm_d", npar.NHeads); e != nil {
					return e
				}
				if mw.dtBias, e = flat(p+"ssm_dt.bias", npar.NHeads); e != nil {
					return e
				}
				if mw.normW, e = flat(p+"ssm_norm.weight", dInner); e != nil {
					return e
				}
				if mw.outProj, e = f32mat(p+"ssm_out.weight", hidden, dInner); e != nil {
					return e
				}
				a, aerr := flat(p+"ssm_a", npar.NHeads)
				if aerr != nil {
					return aerr
				}
				mw.aLog = make([]float32, npar.NHeads)
				for h := range a {
					mw.aLog[h] = float32(math.Log(-float64(a[h])))
				}
				l.mamba = mw
			case nemoAttn:
				// NoPE → no q/k permute.
				if l.QProj, e = mat(p+"attn_q.weight", qDim, hidden); e != nil {
					return e
				}
				if l.KProj, e = mat(p+"attn_k.weight", kvDim, hidden); e != nil {
					return e
				}
				if l.VProj, e = mat(p+"attn_v.weight", kvDim, hidden); e != nil {
					return e
				}
				if l.OProj, e = mat(p+"attn_output.weight", hidden, qDim); e != nil {
					return e
				}
			case nemoMLP:
				if l.UpProj, e = mat(p+"ffn_up.weight", inter, hidden); e != nil {
					return e
				}
				if l.DownProj, e = mat(p+"ffn_down.weight", hidden, inter); e != nil {
					return e
				}
			case nemoMoE:
				// Nemotron 3 Nano's MoE FFN. Tensor names verified against a real GGUF
				// file's tensor list (bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF,
				// fetched directly and parsed with this package's own GGUF reader — not
				// assumed from llama.cpp's conversion-script source mapping, which names
				// the SOURCE safetensors tensor, not the output GGUF tensor). Experts are
				// FUSED per projection (one 3-D [in,out,nExpert] tensor each), unlike the
				// safetensors path's one-tensor-per-expert layout — stackedExperts is the
				// existing helper other GGUF MoE families already use for this shape.
				// exp_probs_b.bias is llama.cpp's name for e_score_correction_bias.
				moe := arch.MoE
				if l.Router, e = mat(p+"ffn_gate_inp.weight", moe.NumExperts, hidden); e != nil {
					return e
				}
				if l.RouterBias, e = flat(p+"exp_probs_b.bias", moe.NumExperts); e != nil {
					return e
				}
				upExp, uerr := stackedExperts(p+"ffn_up_exps.weight", moe.IntermediateDim, hidden, moe.NumExperts)
				if uerr != nil {
					return uerr
				}
				downExp, derr := stackedExperts(p+"ffn_down_exps.weight", hidden, moe.IntermediateDim, moe.NumExperts)
				if derr != nil {
					return derr
				}
				l.Experts = make([]expertWeights, moe.NumExperts)
				for ei := range l.Experts {
					l.Experts[ei].Up, l.Experts[ei].Down = upExp[ei], downExp[ei]
				}
				if moe.SharedIntermediateDim > 0 {
					if l.SharedExpert.Up, e = mat(p+"ffn_up_shexp.weight", moe.SharedIntermediateDim, hidden); e != nil {
						return e
					}
					if l.SharedExpert.Down, e = mat(p+"ffn_down_shexp.weight", hidden, moe.SharedIntermediateDim); e != nil {
						return e
					}
				}
			}
			return nil
		}
		if err := parallelLayers(arch.NumLayers, loadNemo); err != nil {
			return nil, err
		}
		return w, nil
	}

	// Gemma 4 (E-models): a dedicated layer loader. Per-layer head_dim/FFN, the
	// global/sliding split, the cross-layer KV-shared tail (layers ≥ first-shared
	// carry no k/v), and the PLE branch (inp_gate/proj/post_norm + layer scalar).
	// The model-level PLE inputs (per_layer_*) load here. NEOX rope ⇒ no q/k
	// permute, so plain mat() throughout.
	if g4 := arch.gemma4; g4 != nil {
		if g4.HiddenSizePerLayerInput > 0 {
			pleTotal := arch.NumLayers * g4.HiddenSizePerLayerInput
			if w.PerLayerTokenEmbed, err = embMat("per_layer_token_embd.weight", cfg.VocabSize, pleTotal); err != nil {
				return nil, err
			}
			if w.PerLayerModelProj, err = mat("per_layer_model_proj.weight", pleTotal, hidden); err != nil {
				return nil, err
			}
			if w.PerLayerProjNorm, err = vnorm("per_layer_proj_norm.weight", g4.HiddenSizePerLayerInput); err != nil {
				return nil, err
			}
		}
		firstShared := arch.NumLayers - g4.SharedKVLayers
		loadG4 := func(i int) error {
			l := &w.Layers[i]
			p := fmt.Sprintf("blk.%d.", i)
			hdL := arch.headDimAt(i)
			qDimL, kvDimL, ffn := arch.NumHeads*hdL, arch.kvHeadsAt(i)*hdL, arch.ffnAt(i)
			var e error
			if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
				return e
			}
			if l.PostAttnNorm, e = vnorm(p+"post_attention_norm.weight", hidden); e != nil {
				return e
			}
			if l.PreMLPNorm, e = vnorm(p+"ffn_norm.weight", hidden); e != nil {
				return e
			}
			if l.PostMLPNorm, e = vnorm(p+"post_ffw_norm.weight", hidden); e != nil {
				return e
			}
			if l.QProj, e = mat(p+"attn_q.weight", qDimL, hidden); e != nil {
				return e
			}
			if l.QNorm, e = vnorm(p+"attn_q_norm.weight", hdL); e != nil {
				return e
			}
			l.KVShared = i >= firstShared
			if !l.KVShared {
				if l.KProj, e = mat(p+"attn_k.weight", kvDimL, hidden); e != nil {
					return e
				}
				if l.KNorm, e = vnorm(p+"attn_k_norm.weight", hdL); e != nil {
					return e
				}
				// attention_k_eq_v (12B global layers): no v_proj — V reuses K's
				// projection. Detect by the tensor's absence (robust to whether the
				// converter omits or duplicates it).
				if g.Has(p + "attn_v.weight") {
					if l.VProj, e = mat(p+"attn_v.weight", kvDimL, hidden); e != nil {
						return e
					}
				} else {
					l.VFromK = true
				}
			}
			if l.OProj, e = mat(p+"attn_output.weight", hidden, qDimL); e != nil {
				return e
			}
			if l.GateProj, e = mat(p+"ffn_gate.weight", ffn, hidden); e != nil {
				return e
			}
			if l.UpProj, e = mat(p+"ffn_up.weight", ffn, hidden); e != nil {
				return e
			}
			if l.DownProj, e = mat(p+"ffn_down.weight", hidden, ffn); e != nil {
				return e
			}
			if g4.HiddenSizePerLayerInput > 0 {
				if l.PLEGate, e = mat(p+"inp_gate.weight", g4.HiddenSizePerLayerInput, hidden); e != nil {
					return e
				}
				if l.PLEProj, e = mat(p+"proj.weight", hidden, g4.HiddenSizePerLayerInput); e != nil {
					return e
				}
				if l.PostPLENorm, e = vnorm(p+"post_norm.weight", hidden); e != nil {
					return e
				}
			}
			l.LayerScalar = 1
			if sc, se := vec(p+"layer_output_scale.weight", 1); se == nil {
				l.LayerScalar = sc[0]
			}
			return nil
		}
		if err = parallelLayers(arch.NumLayers, loadG4); err != nil {
			return nil, err
		}
		// K=V (attention_k_eq_v) on the 12B global layers: V reuses K's projection
		// (v_norm(k_proj) — see loadG4/runLayersGemma4). Parity-gated against the HF
		// bf16 oracle by TestGemma4_12B_logitParity (argmax exact, cosine 0.990).
		return w, nil
	}

	if lp := arch.llama4; lp != nil {
		// Llama 4: separate q/k/v/o (no q/k-norm tensors — the L2 QK-norm is
		// parameter-free, applied in the forward), per-layer dense/MoE. MoE = router
		// (NO exp_probs_b — top-1 sigmoid, no bias) + stacked per-expert SwiGLU
		// (ffn_{gate,up,down}_exps, the standard [in,out,nE] layout) + an ungated shared
		// expert. NEOX/interleaved rope ⇒ no q/k permute.
		qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
		expInter, denseInter, nE := arch.MoE.IntermediateDim, arch.IntermediateDim, arch.MoE.NumExperts
		loadL4 := func(i int) error {
			l := &w.Layers[i]
			p := fmt.Sprintf("blk.%d.", i)
			var e error
			if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
				return e
			}
			if l.QProj, e = mat(p+"attn_q.weight", qDim, hidden); e != nil {
				return e
			}
			if l.KProj, e = mat(p+"attn_k.weight", kvDim, hidden); e != nil {
				return e
			}
			if l.VProj, e = mat(p+"attn_v.weight", kvDim, hidden); e != nil {
				return e
			}
			if l.OProj, e = mat(p+"attn_output.weight", hidden, qDim); e != nil {
				return e
			}
			if l.PreMLPNorm, e = vnorm(p+"ffn_norm.weight", hidden); e != nil {
				return e
			}
			if !lp.isMoE[i] {
				if l.GateProj, e = mat(p+"ffn_gate.weight", denseInter, hidden); e != nil {
					return e
				}
				if l.UpProj, e = mat(p+"ffn_up.weight", denseInter, hidden); e != nil {
					return e
				}
				if l.DownProj, e = mat(p+"ffn_down.weight", hidden, denseInter); e != nil {
					return e
				}
				return nil
			}
			if l.Router, e = mat(p+"ffn_gate_inp.weight", nE, hidden); e != nil {
				return e
			}
			gate, ge := stackedExperts(p+"ffn_gate_exps.weight", expInter, hidden, nE)
			up, ue := stackedExperts(p+"ffn_up_exps.weight", expInter, hidden, nE)
			down, de := stackedExperts(p+"ffn_down_exps.weight", hidden, expInter, nE)
			if ge != nil || ue != nil || de != nil {
				return fmt.Errorf("decoder(gguf-llama4): experts layer %d: %v / %v / %v", i, ge, ue, de)
			}
			l.Experts = make([]expertWeights, nE)
			for x := range l.Experts {
				l.Experts[x] = expertWeights{Gate: gate[x], Up: up[x], Down: down[x]}
			}
			if l.SharedExpert.Gate, e = mat(p+"ffn_gate_shexp.weight", expInter, hidden); e != nil {
				return e
			}
			if l.SharedExpert.Up, e = mat(p+"ffn_up_shexp.weight", expInter, hidden); e != nil {
				return e
			}
			if l.SharedExpert.Down, e = mat(p+"ffn_down_shexp.weight", hidden, expInter); e != nil {
				return e
			}
			return nil
		}
		if err = parallelLayers(arch.NumLayers, loadL4); err != nil {
			return nil, err
		}
		return w, nil
	}

	qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
	// llama.cpp permutes the q/k weights (and their per-row biases / head-dim
	// norms) only for the NORM rope type — llama and its derivatives (incl.
	// mellum). The NEOX rope type (qwen2/qwen3/gemma) leaves q/k in HF order, so
	// no un-permutation is needed there.
	permuteQK := ggufQKPermuted(arch.Name)
	loadQK := func(name string, out, nHead int) (linalg.WeightMat, error) {
		if permuteQK {
			return permMat(name, out, hidden, nHead)
		}
		return mat(name, out, hidden)
	}
	// Load the layers in parallel: each is independent (its own linalg.WeightMat slots
	// over the read-only mmap), and the per-tensor dequant + re-quant is the load's
	// cost — fanning it out across cores turns a 12B GGUF's ~2 min load into seconds.
	loadLayer := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("blk.%d.", i)
		var err error
		if l.PreAttnNorm, err = vnorm(p+"attn_norm.weight", hidden); err != nil {
			return err
		}
		if arch.mla != nil {
			// DeepSeek MLA: the latent-attention tensor set (q-LoRA / kv down+up + the
			// two latent norms), loaded f32 — not the standard q/k/v/o. The FFN block
			// below is shared with the other DeepSeek-style MoE families.
			if err = loadDeepseekAttnGGUF(g, p, l, arch); err != nil {
				return err
			}
		} else if arch.Name == "phi3" {
			// Phi-3/Phi-4: fused attn_qkv [qDim+2*kvDim, hidden] → Q ‖ K ‖ V by output
			// rows (NEOX rope ⇒ no permute). o_proj is unfused.
			if l.QProj, err = fusedSplit(p+"attn_qkv.weight", qDim+2*kvDim, hidden, 0, qDim); err != nil {
				return err
			}
			if l.KProj, err = fusedSplit(p+"attn_qkv.weight", qDim+2*kvDim, hidden, qDim, kvDim); err != nil {
				return err
			}
			if l.VProj, err = fusedSplit(p+"attn_qkv.weight", qDim+2*kvDim, hidden, qDim+kvDim, kvDim); err != nil {
				return err
			}
			if l.OProj, err = mat(p+"attn_output.weight", hidden, qDim); err != nil {
				return err
			}
		} else {
			if l.QProj, err = loadQK(p+"attn_q.weight", qDim, arch.NumHeads); err != nil {
				return err
			}
			if l.KProj, err = loadQK(p+"attn_k.weight", kvDim, arch.NumKVHeads); err != nil {
				return err
			}
			if l.VProj, err = mat(p+"attn_v.weight", kvDim, hidden); err != nil {
				return err
			}
			if l.OProj, err = mat(p+"attn_output.weight", hidden, qDim); err != nil {
				return err
			}
			// q/k/v projection bias (Qwen2): the q and k biases are per-output-row, so
			// llama.cpp's RoPE row permutation applies to them exactly as to the q/k
			// weight rows (ggufInvPermute with in=1); the v bias is not permuted.
			if arch.QKVBias {
				qb, qe := vec(p+"attn_q.bias", qDim)
				kb, ke := vec(p+"attn_k.bias", kvDim)
				vb, ve := vec(p+"attn_v.bias", kvDim)
				if qe != nil || ke != nil || ve != nil {
					return fmt.Errorf("decoder(gguf): qkv bias layer %d: %v / %v / %v", i, qe, ke, ve)
				}
				if permuteQK {
					qb = ggufInvPermute(qb, qDim, 1, arch.NumHeads)
					kb = ggufInvPermute(kb, kvDim, 1, arch.NumKVHeads)
				}
				l.QBias, l.KBias, l.VBias = qb, kb, vb
			}
			// QK-norm (Mellum, Qwen3): per-head RMSNorm over head_dim, before RoPE.
			// llama.cpp permutes the q/k weights for its RoPE, so the matching
			// per-head-dim norm weights are un-permuted the same way.
			if arch.QKNorm {
				qn, kerr := vnorm(p+"attn_q_norm.weight", hd)
				if kerr != nil {
					return kerr
				}
				kn, kerr := vnorm(p+"attn_k_norm.weight", hd)
				if kerr != nil {
					return kerr
				}
				if permuteQK {
					qn, kn = ggufInvPermuteVec(qn), ggufInvPermuteVec(kn)
				}
				l.QNorm, l.KNorm = qn, kn
			}
		}
		// Pre-MLP norm. GLM4_MOE has no ffn_norm — llama.cpp stores its (positionally
		// pre-MLP, Pre2) post_attention_layernorm under the ATTN_POST_NORM name; every
		// other family here uses ffn_norm.
		preMLPNorm := "ffn_norm.weight"
		if arch.Name == "glm4_moe" {
			preMLPNorm = "post_attention_norm.weight"
		}
		if l.PreMLPNorm, err = vnorm(p+preMLPNorm, hidden); err != nil {
			return err
		}
		// Sandwich norms (Gemma 3): a post-attention and a post-FFN RMSNorm in
		// addition to the pre-norms. GGUF names them post_attention_norm /
		// post_ffw_norm.
		if arch.NormPlacement == NormSandwich4 {
			if l.PostAttnNorm, err = vnorm(p+"post_attention_norm.weight", hidden); err != nil {
				return err
			}
			if l.PostMLPNorm, err = vnorm(p+"post_ffw_norm.weight", hidden); err != nil {
				return err
			}
		}
		if arch.MoE != nil && i >= arch.FirstKDense {
			// Sparse MoE: router + stacked per-expert SwiGLU at the narrower
			// moe_intermediate_size. Mellum stops there; GLM/DeepSeek add an
			// e_score_correction_bias (exp_probs_b) and an always-on shared expert
			// (ffn_*_shexp, ungated). GLM's first_k_dense_replace prefix (i < FirstKDense)
			// has no router and falls through to the dense FFN below.
			expInter := arch.MoE.IntermediateDim
			if l.Router, err = mat(p+"ffn_gate_inp.weight", arch.MoE.NumExperts, hidden); err != nil {
				return err
			}
			if arch.MoE.RouterSigmoid { // DeepSeek/GLM e_score_correction_bias (a bias term)
				if l.RouterBias, err = vec(p+"exp_probs_b.bias", arch.MoE.NumExperts); err != nil {
					return err
				}
			}
			gate, gerr := stackedExperts(p+"ffn_gate_exps.weight", expInter, hidden, arch.MoE.NumExperts)
			up, uerr := stackedExperts(p+"ffn_up_exps.weight", expInter, hidden, arch.MoE.NumExperts)
			down, derr := stackedExperts(p+"ffn_down_exps.weight", hidden, expInter, arch.MoE.NumExperts)
			if gerr != nil || uerr != nil || derr != nil {
				return fmt.Errorf("decoder(gguf): experts layer %d: %v / %v / %v", i, gerr, uerr, derr)
			}
			l.Experts = make([]expertWeights, arch.MoE.NumExperts)
			for e := range arch.MoE.NumExperts {
				l.Experts[e] = expertWeights{Gate: gate[e], Up: up[e], Down: down[e]}
			}
			// Shared always-on expert (GLM: ffn_*_shexp). Ungated for GLM/DeepSeek;
			// Qwen2-MoE-style families would also carry a sigmoid gate (not in GGUF here).
			if arch.MoE.SharedIntermediateDim > 0 {
				sInter := arch.MoE.SharedIntermediateDim
				if l.SharedExpert.Gate, err = mat(p+"ffn_gate_shexp.weight", sInter, hidden); err != nil {
					return err
				}
				if l.SharedExpert.Up, err = mat(p+"ffn_up_shexp.weight", sInter, hidden); err != nil {
					return err
				}
				if l.SharedExpert.Down, err = mat(p+"ffn_down_shexp.weight", hidden, sInter); err != nil {
					return err
				}
				if !arch.MoE.SharedUngated {
					sg, serr := vec(p+"ffn_gate_inp_shexp.weight", hidden)
					if serr != nil {
						return serr
					}
					l.SharedGate = linalg.WrapF32(sg, 1, hidden)
				}
			}
			return nil
		}
		if arch.Name == "phi3" {
			// Phi-3/Phi-4: fused ffn_up [2*inter, hidden] is gate‖up (no separate
			// ffn_gate); ffn_down is unfused.
			inter := arch.IntermediateDim
			if l.GateProj, err = fusedSplit(p+"ffn_up.weight", 2*inter, hidden, 0, inter); err != nil {
				return err
			}
			if l.UpProj, err = fusedSplit(p+"ffn_up.weight", 2*inter, hidden, inter, inter); err != nil {
				return err
			}
			if l.DownProj, err = mat(p+"ffn_down.weight", hidden, inter); err != nil {
				return err
			}
			return nil
		}
		if l.GateProj, err = mat(p+"ffn_gate.weight", arch.IntermediateDim, hidden); err != nil {
			return err
		}
		if l.UpProj, err = mat(p+"ffn_up.weight", arch.IntermediateDim, hidden); err != nil {
			return err
		}
		if l.DownProj, err = mat(p+"ffn_down.weight", hidden, arch.IntermediateDim); err != nil {
			return err
		}
		return nil
	}
	if sink != nil {
		// Sequential load → write → free: peak RAM is one layer, not the model.
		for i := 0; i < arch.NumLayers; i++ {
			if err := loadLayer(i); err != nil {
				return nil, err
			}
			sink.layer(&w.Layers[i])
			if sink.err != nil {
				return nil, sink.err
			}
			w.Layers[i] = LayerWeights{} // release before the next layer
		}
		return w, nil
	}
	if err := parallelLayers(arch.NumLayers, loadLayer); err != nil {
		return nil, err
	}
	return w, nil
}

// ggufQKPermuted reports whether llama.cpp's GGUF conversion permuted this
// architecture's q/k weights (and their per-row biases and head-dim norms). It
// permutes only for the NORM rope type — llama and its derivatives, including
// mellum; the NEOX rope type (qwen2/qwen3/gemma and the other modern families)
// leaves q/k in HF rotate_half order, so no un-permutation is needed. Unknown
// archs default to NEOX (no permute), the common modern case.
func ggufQKPermuted(archName string) bool {
	switch archName {
	case "llama", "mellum":
		return true
	default:
		return false
	}
}

// ggufInvPermuteVec inverts llama.cpp's q/k RoPE permutation for a single
// head_dim-wide vector (a QK-norm weight, shared across heads): HF position
// (s*half + j) comes from GGUF position (2*j + s). Mirrors ggufInvPermute with
// nHead=1 and no input dimension.
func ggufInvPermuteVec(v []float32) []float32 {
	hd := len(v)
	half := hd / 2
	res := make([]float32, hd)
	for s := range 2 {
		for j := range half {
			res[s*half+j] = v[2*j+s]
		}
	}
	return res
}

// ggufInvPermute inverts llama.cpp's q/k weight permutation. llama.cpp stores
// the projection rows in interleaved-pair order for its RoPE; this package uses
// HF's rotate_half (first-half / second-half) order. For head h, the HF row
// (s*half + j) — s∈{0,1} selecting the half, j the position within it — comes
// from the GGUF row (2*j + s). w is [out, in] row-major; out = nHead*headDim.
func ggufInvPermute(w []float32, out, in, nHead int) []float32 {
	hd := out / nHead
	half := hd / 2
	res := make([]float32, len(w))
	for h := range nHead {
		for s := range 2 {
			for j := range half {
				hfRow := h*hd + s*half + j
				ggufRow := h*hd + 2*j + s
				copy(res[hfRow*in:hfRow*in+in], w[ggufRow*in:ggufRow*in+in])
			}
		}
	}
	return res
}
