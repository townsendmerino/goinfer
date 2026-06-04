package decoder

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// archAdapter resolves a parsed config.json into the family-agnostic
// Architecture descriptor + the family's tensor-name schema, both consumed by
// the generic forward pass / loader.
type archAdapter func(*Config) (*Architecture, *tensorSchema, error)

// registry maps config.json model_type → its adapter. Adding a family
// (multi-model-plan G2+) is a new entry here plus its tensor schema — the
// forward pass itself doesn't change.
var registry = map[string]archAdapter{
	"gemma3":      gemma3Architecture,
	"gemma3_text": gemma3Architecture,   // the 270M/1B text checkpoints
	"qwen3":       qwen3Architecture,    // Qwen3 dense (0.6B/1.7B/4B/8B/…)
	"qwen2":       qwen2Architecture,    // Qwen2/Qwen2.5 dense (llama + q/k/v bias)
	"qwen2_moe":   qwen2MoeArchitecture, // Qwen-MoE/Qwen2-MoE (qwen2 + sparse MoE + shared expert)
	"llama":       llamaArchitecture,    // Llama-2/3 dense (single-base RoPE, no QK-norm)
	"mistral":     mistralArchitecture,  // Llama + all-layer sliding-window attention
	"gpt2":        gpt2Architecture,     // GPT-2: LayerNorm, learned pos, non-gated GELU MLP, fused QKV
	"mixtral":     mixtralArchitecture,  // Llama + sparse MoE FFN (router + top-k experts)
	"mellum":      mellumArchitecture,   // JetBrains Mellum2: MoE + sliding/full interleave + YaRN
}

// resolveArchitecture picks the adapter for cfg.ModelType and builds the
// descriptor + schema. An unknown model_type is a loud error, not a silent
// wrong load.
func resolveArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	adapter, ok := registry[cfg.ModelType]
	if !ok {
		return nil, nil, fmt.Errorf("decoder: unsupported model_type %q (have: %s)", cfg.ModelType, knownModelTypes())
	}
	arch, schema, err := adapter(cfg)
	if err != nil {
		return nil, nil, err
	}
	arch.finalizeRoPE() // precompute inv-freq tables (base + scaling + rotary dim)
	return arch, schema, nil
}

func knownModelTypes() string {
	keys := make([]string, 0, len(registry))
	for k := range registry {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return strings.Join(keys, ", ")
}

// gemma3Architecture expresses Gemma 3 as a descriptor: RMSNorm with (1+w),
// the 4-norm sandwich, GeGLU, QK-norm, the query_pre_attn_scalar attention
// scale, dual-base RoPE (local/global per layer), √hidden embedding scale, and
// a tied LM head with no soft-capping. ValidateAssumptions pins the bits this
// forward pass can't vary (Gemma-2 soft-capping, unsupported activation).
func gemma3Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.ValidateAssumptions(); err != nil {
		return nil, nil, err
	}
	return &Architecture{
		Name:              "gemma3",
		HiddenDim:         cfg.HiddenDim,
		NumLayers:         cfg.NumLayers,
		NumHeads:          cfg.NumHeads,
		NumKVHeads:        cfg.NumKVHeads,
		HeadDim:           cfg.HeadDim,
		IntermediateDim:   cfg.IntermediateDim,
		VocabSize:         cfg.VocabSize,
		Norm:              NormRMS,
		RMSAddOne:         true,
		NormEps:           cfg.RMSNormEps,
		NormPlacement:     NormSandwich4,
		Act:               ActGeluTanh,
		QKNorm:            true,
		AttnScale:         math.Pow(cfg.QueryPreAttnScalar, -0.5),
		SlidingWindow:     cfg.SlidingWindow,
		layerIsGlobal:     cfg.IsGlobalLayer,
		RoPELocalBase:     cfg.RoPELocalBase,
		RoPEGlobalBase:    cfg.RoPEGlobalBase,
		EmbedScale:        math.Sqrt(float64(cfg.HiddenDim)),
		TiedLMHead:        true,
		FinalLogitSoftcap: cfg.FinalLogitSoftcap, // 0 (ValidateAssumptions rejects nonzero)
		AttnLogitSoftcap:  cfg.AttnLogitSoftcap,
	}, &gemma3TensorSchema, nil
}

// qwen3Architecture expresses Qwen3 dense (multi-model-plan G2): RMSNorm
// without the (1+w) offset, the Pre2 norm placement (no post-sublayer norms),
// SwiGLU, QK-norm (Qwen3 keeps it), 1/√head_dim attention scale, single-base
// RoPE, no embedding scale, and a separate lm_head (untied). No QKV bias
// (Qwen3 dropped Qwen2's). The tensor schema is qwen3TensorSchema.
func qwen3Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateQwen3(); err != nil {
		return nil, nil, err
	}
	return &Architecture{
		Name:            "qwen3",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         cfg.HeadDim,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		QKNorm:          true,
		AttnScale:       math.Pow(float64(cfg.HeadDim), -0.5),
		SlidingWindow:   0,                  // Qwen3 dense: full attention
		layerIsGlobal:   nil,                // all-global
		RoPELocalBase:   cfg.RoPEGlobalBase, // single base (rope_theta)
		RoPEGlobalBase:  cfg.RoPEGlobalBase,
		EmbedScale:      0,     // none
		TiedLMHead:      false, // finalized from lm_head.weight presence at load
	}, &qwen3TensorSchema, nil
}

// llamaArchitecture expresses Llama-2/3 dense: like Qwen3 (RMSNorm no-offset,
// Pre2 placement, SwiGLU, 1/√head_dim scale, single-base RoPE, no embed scale)
// but WITHOUT QK-norm — Llama's attention applies RoPE to raw q/k. head_dim is
// derived (headDim()) since many Llama configs omit it. The LM head is tied on
// the small text models (1B/3B) and untied on 8B+, finalized from
// lm_head.weight presence at load. validateLlama rejects scaled RoPE (G4) and
// attention bias (a later add), so reaching here implies a plain checkpoint.
func llamaArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateLlama(); err != nil {
		return nil, nil, err
	}
	scaling, err := parseRopeScaling(cfg.RopeScaling) // G4: linear + llama3; unsupported types error here
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(llama): %w", err)
	}
	hd := cfg.headDim()
	return &Architecture{
		Name:            "llama",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		QKNorm:          false, // the one knob that differs from Qwen3
		AttnScale:       math.Pow(float64(hd), -0.5),
		SlidingWindow:   0,                  // Llama dense: full attention
		layerIsGlobal:   nil,                // all-global
		RoPELocalBase:   cfg.RoPEGlobalBase, // single base (rope_theta)
		RoPEGlobalBase:  cfg.RoPEGlobalBase,
		RotaryDim:       cfg.rotaryDim(), // 0 = full head_dim (Llama); partial for Phi
		ropeScaling:     scaling,         // llama3 (3.1+/3.2) / linear; nil = none
		EmbedScale:      0,               // none
		TiedLMHead:      false,           // finalized from lm_head.weight presence at load
	}, &llamaTensorSchema, nil
}

// mistralArchitecture expresses Mistral dense: the llama descriptor (RMS
// no-offset, Pre2, SwiGLU, single-base RoPE, no QK-norm, no bias, derived
// head_dim) with sliding-window attention on EVERY layer (Gemma's window
// machinery, but all-local rather than 5:1). A checkpoint with sliding_window
// null/0 (Mistral-v0.2+) falls back to full attention. The tensor schema is
// llamaTensorSchema (Mistral and Llama share tensor names).
func mistralArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateLlama(); err != nil {
		return nil, nil, err
	}
	scaling, err := parseRopeScaling(cfg.RopeScaling)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(mistral): %w", err)
	}
	hd := cfg.headDim()
	// All layers local when a window is set; else all global (full attention).
	var layerIsGlobal func(int) bool
	if cfg.SlidingWindow > 0 {
		layerIsGlobal = func(int) bool { return false }
	}
	return &Architecture{
		Name:            "mistral",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		QKNorm:          false,
		AttnScale:       math.Pow(float64(hd), -0.5),
		SlidingWindow:   cfg.SlidingWindow, // 0 ⇒ full attention
		layerIsGlobal:   layerIsGlobal,     // all-local when windowed
		RoPELocalBase:   cfg.RoPEGlobalBase,
		RoPEGlobalBase:  cfg.RoPEGlobalBase,
		RotaryDim:       cfg.rotaryDim(),
		ropeScaling:     scaling,
		EmbedScale:      0,
		TiedLMHead:      false, // finalized from lm_head.weight presence at load
	}, &llamaTensorSchema, nil
}

// gpt2Architecture expresses GPT-2 (multi-model-plan G5): the GPT-2/NeoX class
// that breaks the Llama mold on several axes — LayerNorm (mean-centered, with
// bias) instead of RMSNorm, learned absolute position embeddings instead of
// RoPE, a non-gated GELU MLP (up→gelu→down) instead of a gated one, fused q/k/v
// with bias, an attention output bias, and tied embeddings. The Conv1D weight
// layout + fused projections need a dedicated loader (buildGPT2Weights), so
// this returns the gpt2TensorSchema as a marker; the schema's field names are
// unused.
func gpt2Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateGPT2(); err != nil {
		return nil, nil, err
	}
	hd := cfg.NEmbd / cfg.NHead
	inter := cfg.NInner
	if inter == 0 {
		inter = 4 * cfg.NEmbd // GPT-2 default feed-forward width
	}
	// Canonicalize the standard Config dims from GPT-2's n_* keys so Model.Config()
	// (used by the demos for the load banner) reports them uniformly. cfg is a
	// freshly-parsed, per-Load pointer, so this mutation is local.
	cfg.HiddenDim, cfg.NumLayers, cfg.NumHeads = cfg.NEmbd, cfg.NLayer, cfg.NHead
	cfg.NumKVHeads, cfg.HeadDim, cfg.IntermediateDim = cfg.NHead, hd, inter
	cfg.MaxPositions = cfg.NPositions
	return &Architecture{
		Name:            "gpt2",
		HiddenDim:       cfg.NEmbd,
		NumLayers:       cfg.NLayer,
		NumHeads:        cfg.NHead,
		NumKVHeads:      cfg.NHead, // no GQA
		HeadDim:         hd,
		IntermediateDim: inter,
		VocabSize:       cfg.VocabSize,
		MaxPositions:    cfg.NPositions,
		Norm:            NormLayer,
		NormEps:         cfg.LayerNormEpsilon,
		NormPlacement:   NormPre2, // ln_1 pre-attn, ln_2 pre-MLP
		Act:             ActGeluTanh,
		NonGatedMLP:     true,
		QKVBias:         true,
		OutBias:         true,
		QKNorm:          false,
		LearnedPosEmbed: true,
		AttnScale:       math.Pow(float64(hd), -0.5),
		EmbedScale:      0,
		TiedLMHead:      true, // GPT-2 ties wte as the LM head
	}, &gpt2TensorSchema, nil
}

// mixtralArchitecture expresses Mixtral (multi-model-plan G6): the llama
// descriptor (RMS no-offset, Pre2, SwiGLU experts, single-base RoPE, no QK-norm,
// no bias, untied head) with the dense FFN replaced by a sparse mixture of
// experts — a router picks top-k of NumExperts experts per token. Recent HF
// Mixtral uses full attention (the config's sliding_window is vestigial), so
// this does too. The tensor schema is mixtralTensorSchema.
func mixtralArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateMixtral(); err != nil {
		return nil, nil, err
	}
	scaling, err := parseRopeScaling(cfg.RopeScaling)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(mixtral): %w", err)
	}
	hd := cfg.headDim()
	normTopK := true // HF MixtralConfig default
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	return &Architecture{
		Name:            "mixtral",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		MoE: &MoEConfig{
			NumExperts:      cfg.NumLocalExperts,
			TopK:            cfg.NumExpertsPerTok,
			NormTopKProb:    normTopK,
			IntermediateDim: cfg.IntermediateDim, // Mixtral experts use the dense width
		},
		QKNorm:         false,
		AttnScale:      math.Pow(float64(hd), -0.5),
		SlidingWindow:  0, // full attention (HF Mixtral ignores config sliding_window)
		layerIsGlobal:  nil,
		RoPELocalBase:  cfg.RoPEGlobalBase,
		RoPEGlobalBase: cfg.RoPEGlobalBase,
		RotaryDim:      cfg.rotaryDim(),
		ropeScaling:    scaling,
		EmbedScale:     0,
		TiedLMHead:     false, // finalized from lm_head.weight presence at load
	}, &mixtralTensorSchema, nil
}

// mellumArchitecture expresses JetBrains Mellum2 (a 12B-A2.5B code model): the
// llama skeleton (RMS no-offset, Pre2, SwiGLU, derived head_dim, no QK-norm, no
// bias, untied head) combining two axes we already had separately — a sparse MoE
// FFN on EVERY layer (64 experts, top-8, the narrower moe_intermediate_size) and
// a 3:1 sliding/full attention interleave (layer_types) — plus the one new piece:
// per-attention-type RoPE from rope_parameters, with YaRN (and its attention_factor
// mscale) on the full layers and plain RoPE on the sliding layers, both at theta
// 500000.
func mellumArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateMellum(); err != nil {
		return nil, nil, err
	}
	full, sliding, err := parseRopeParameters(cfg.RopeParameters)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(mellum): %w", err)
	}
	if full == nil || sliding == nil {
		return nil, nil, fmt.Errorf("decoder(mellum): rope_parameters needs full_attention + sliding_attention")
	}
	hd := cfg.headDim()
	normTopK := true
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	return &Architecture{
		Name:            "mellum",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim, // dense width (vestigial; experts use the MoE width)
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		MoE: &MoEConfig{
			NumExperts:      cfg.NumExperts,
			TopK:            cfg.NumExpertsPerTok,
			NormTopKProb:    normTopK,
			IntermediateDim: cfg.MoeIntermediateSize,
		},
		QKNorm:           true, // Mellum has q_norm/k_norm per head (like Qwen3)
		AttnScale:        math.Pow(float64(hd), -0.5),
		SlidingWindow:    cfg.SlidingWindow,
		layerIsGlobal:    cfg.IsGlobalLayer, // from layer_types (3:1 sliding/full)
		RoPEGlobalBase:   full.base,         // full_attention layers
		RoPELocalBase:    sliding.base,      // sliding_attention layers (same theta, plain)
		ropeScaling:      full.scaling,      // YaRN on full layers
		ropeScalingLocal: sliding.scaling,   // nil (plain) on sliding layers
		RotaryDim:        cfg.rotaryDim(),
		EmbedScale:       0,
		TiedLMHead:       false, // finalized from lm_head.weight presence at load
	}, &mellumTensorSchema, nil
}

// qwen2Architecture expresses Qwen2/Qwen2.5 dense: identical to the llama
// descriptor (RMS no-offset, Pre2, SwiGLU, single-base RoPE, no QK-norm, derived
// head_dim, tied-or-untied head) plus QKVBias — Qwen2 carries an additive bias
// on the q/k/v projections (o_proj stays biasless). The tensor schema is
// qwen2TensorSchema.
func qwen2Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateQwen2(); err != nil {
		return nil, nil, err
	}
	scaling, err := parseRopeScaling(cfg.RopeScaling) // linear + llama3 (Qwen2.5 leaves it null)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(qwen2): %w", err)
	}
	hd := cfg.headDim()
	return &Architecture{
		Name:            "qwen2",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		QKVBias:         true, // the one knob that differs from llama
		QKNorm:          false,
		AttnScale:       math.Pow(float64(hd), -0.5),
		SlidingWindow:   0,   // validateQwen2 rejects use_sliding_window=true
		layerIsGlobal:   nil, // all-global
		RoPELocalBase:   cfg.RoPEGlobalBase,
		RoPEGlobalBase:  cfg.RoPEGlobalBase,
		RotaryDim:       cfg.rotaryDim(),
		ropeScaling:     scaling,
		EmbedScale:      0,
		TiedLMHead:      false, // finalized from lm_head.weight presence at load
	}, &qwen2TensorSchema, nil
}

// qwen2MoeArchitecture expresses Qwen-MoE / Qwen2-MoE (Qwen1.5-MoE-A2.7B,
// Qwen2-57B-A14B): qwen2's attention (q/k/v bias, no QK-norm, derived head_dim,
// single-base RoPE) with the FFN replaced on every layer by a sparse MoE PLUS an
// always-on shared expert. The router picks top-k of num_experts experts at
// moe_intermediate_size; the shared expert is a gated MLP at
// shared_expert_intermediate_size scaled by sigmoid(shared_gate·h). The tensor
// schema is qwen2MoeTensorSchema.
func qwen2MoeArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateQwen2Moe(); err != nil {
		return nil, nil, err
	}
	scaling, err := parseRopeScaling(cfg.RopeScaling)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(qwen2_moe): %w", err)
	}
	hd := cfg.headDim()
	normTopK := false // HF Qwen2MoeConfig default (norm_topk_prob false)
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	return &Architecture{
		Name:            "qwen2_moe",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		QKVBias:         true, // qwen2-style q/k/v bias
		QKNorm:          false,
		MoE: &MoEConfig{
			NumExperts:            cfg.NumExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          normTopK,
			IntermediateDim:       cfg.MoeIntermediateSize,
			SharedIntermediateDim: cfg.SharedExpertIntermediateSize,
		},
		AttnScale:      math.Pow(float64(hd), -0.5),
		SlidingWindow:  0, // full attention
		layerIsGlobal:  nil,
		RoPELocalBase:  cfg.RoPEGlobalBase,
		RoPEGlobalBase: cfg.RoPEGlobalBase,
		RotaryDim:      cfg.rotaryDim(),
		ropeScaling:    scaling,
		EmbedScale:     0,
		TiedLMHead:     false, // finalized from lm_head.weight presence at load
	}, &qwen2MoeTensorSchema, nil
}
