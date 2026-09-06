package decoder

import (
	"encoding/json"
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
// is a new entry here plus its tensor schema — the
// forward pass itself doesn't change.
var registry = map[string]archAdapter{
	"gemma3":              gemma3Architecture,
	"gemma3_text":         gemma3Architecture,     // the 270M/1B text checkpoints
	"gemma4":              gemma4Architecture,     // Gemma 4 (E2B/E4B + 12B dense; parity-gated)
	"gemma4_text":         gemma4Architecture,     // Gemma 4 text checkpoints (the 26B-A4B MoE tiny golden: model_type gemma4_text)
	"gemma4_unified_text": gemma4Architecture,     // real Gemma 4 unified checkpoints' text_config model_type (E2B/E4B/12B dense + 26B-A4B MoE; K=V globals, model.language_model.* prefix)
	"qwen3":               qwen3Architecture,      // Qwen3 dense (0.6B/1.7B/4B/8B/…)
	"qwen2":               qwen2Architecture,      // Qwen2/Qwen2.5 dense (llama + q/k/v bias)
	"qwen2_5_vl":          qwen2_5_vlArchitecture, // Qwen2.5-VL text decoder (qwen2 + m-RoPE; nested rope_parameters)
	"qwen2_moe":           qwen2MoeArchitecture,   // Qwen-MoE/Qwen2-MoE (qwen2 + sparse MoE + shared expert)
	"qwen3_moe":           qwen3MoeArchitecture,   // Qwen3-30B-A3B / Qwen3-Coder-30B-A3B: qwen3's attention (QK-norm, no bias) + a sparse MoE on every layer, NO shared expert
	"llama":               llamaArchitecture,      // Llama-2/3 dense (single-base RoPE, no QK-norm)
	// InternLM3 is llama-shaped to the tensor name: self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj,
	// input_layernorm/post_attention_layernorm, embed_tokens/lm_head, no biases, no QK-norm. Its only
	// config departure is rope_scaling type "dynamic", which is exactly identity within the trained
	// window (see parseRopeScaling). So it is a registry ALIAS, not an adapter — the honest expression
	// of "this is a llama". InternLM2 is NOT an alias: it renames every tensor and fuses qkv.
	"internlm3":        llamaArchitecture,        // InternLM3 (llama-shaped; dynamic-NTK rope is in-window identity)
	"internlm2":        internlm2Architecture,    // InternLM2 (llama math; renamed tensors + GROUPED fused wqkv, split at load)
	"mistral":          mistralArchitecture,      // Llama + all-layer sliding-window attention
	"mistral3":         ministral3Architecture,   // Ministral 3 (3B/8B/14B): Mistral GQA + YaRN + Llama4-style attention-temperature tuning on every layer, text_config extracted from the Mistral3ForConditionalGeneration VL wrapper
	"ministral3":       ministral3Architecture,   // the nested text_config's own model_type (a plain, unwrapped Ministral3Config save carries this directly, not "mistral3")
	"gpt2":             gpt2Architecture,         // GPT-2: LayerNorm, learned pos, non-gated GELU MLP, fused QKV
	"cohere":           cohereArchitecture,       // Cohere / Command-R (+ Aya): bias-free LayerNorm + parallel attn/MLP block + logit_scale + GPT-J interleaved RoPE
	"cohere2":          cohere2Architecture,      // Cohere2 / Command-R7B (+ Command-A): cohere1 stack + interleaved sliding-window + NoPE on the global layers, no QK-norm
	"mixtral":          mixtralArchitecture,      // Llama + sparse MoE FFN (router + top-k experts)
	"mellum":           mellumArchitecture,       // JetBrains Mellum2: MoE + sliding/full interleave + YaRN
	"qwen3_5_moe":      qwen35Architecture,       // Qwen3.5/3.6-MoE: Gated DeltaNet (linear) + softmax hybrid + MoE
	"qwen3_5_moe_text": qwen35Architecture,       // the text-only checkpoint's model_type
	"qwen3_5":          qwen35DenseArchitecture,  // Qwen3.8 dense (Gated DeltaNet + softmax hybrid, plain SwiGLU FFN)
	"qwen3_5_text":     qwen35DenseArchitecture,  // the text_config's own model_type, for text-only checkpoints
	"qwen3_next":       qwen3NextArchitecture,    // Qwen3-Next: same DeltaNet/softmax/MoE hybrid shape as qwen3_5_moe, but layer_types is COMPUTED (full_attention_interval) and partial_rotary_factor is a top-level field, not nested
	"glm4_moe":         glm4moeArchitecture,      // GLM-4.5/4.6: DeepSeek-style MoE (sigmoid routing + bias) + dense prefix + QK-norm + partial RoPE
	"laguna":           lagunaArchitecture,       // Laguna (poolside) XS-2.1 / XS.2 / M.1: sigmoid-routed MoE + shared expert + softplus attention output gating + per-layer query heads
	"granitemoehybrid": graniteArchitecture,      // Granite-4.0-H: Mamba-2 + attention hybrid + MoE-on-every-layer + Granite multipliers
	"granite":          graniteDenseArchitecture, // Granite 4.2 (3B/8B/30B) dense: llama skeleton + Granite's four scalar multipliers
	"lfm2":             lfm2Architecture,         // LFM2 / LFM2.5: gated short-conv + GQA hybrid (layer_types), tied head, per-head RMSNorm QK-norm
	"nemotron_h":       nemotronhArchitecture,    // Nemotron-H: single-op-per-block hybrid (mamba | NoPE-attention | relu² MLP)
	"deepseek_v2":      deepseekArchitecture,     // DeepSeek-V2 (MLA + DeepSeekMoE; softmax routing, V2-Lite has no q-LoRA)
	"deepseek_v3":      deepseekArchitecture,     // DeepSeek-V3 (MLA + DeepSeekMoE; sigmoid + e_score_correction_bias group-limited routing)
	"kimi_k2":          deepseekArchitecture,     // Kimi K2/K2.x (architectures=DeepseekV3ForCausalLM): MLA + DeepSeekMoE, "basically V3" — 64 heads / 384 experts, config scalars only
	"phi3":             phi3Architecture,         // Phi-3 / Phi-4 dense: llama skeleton + fused qkv_proj / gate_up_proj (split at load) + partial rotary
	"llama4_text":      llama4Architecture,       // Llama 4 (Scout/Maverick) text decoder: iRoPE (RoPE/NoPE interleave) + L2 QK-norm + attn-temp + dense/MoE interleave (top-1 sigmoid + shared)
	"gpt_oss":          gptOssArchitecture,       // gpt-oss (20b/120b): sparse MoE + per-head attention sinks + clamped interleaved-SwiGLU + alternating sliding/full + YaRN (MXFP4 experts; CPU-only)
}

// resolveArchitecture picks the adapter for cfg.ModelType and builds the
// descriptor + schema. An unknown model_type is a loud error, not a silent
// wrong load.
func resolveArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	adapter, ok := registry[cfg.ModelType]
	if !ok {
		return nil, nil, fmt.Errorf("decoder: unsupported model_type %q (have: %s)", cfg.ModelType, knownModelTypes())
	}
	// M-10(b): BOUND THE CONFIG BEFORE THE ADAPTER RUNS. Several adapters allocate
	// NumLayers-sized slices with only a `> 0` check (qwen3_next, llama4), and loadConfig has
	// no bound at all — so a 300-byte .giw or a hostile safetensors config.json declaring
	// num_hidden_layers: 68719476736 is a FATAL out-of-memory, not the typed error
	// LoadSerializedWeights' doc promises. Under Go's maxAlloc, so no recover() catches it.
	//
	// Here rather than at the two JSON chokepoints the audit names: this is the single point
	// every path reaches — .giw, safetensors, GGUF and whatever is added next — and putting it
	// at the callers would be the "one predicate, N consumers" shape that produced half the
	// findings in this audit. The GGUF paths bound some of these already; re-checking costs a
	// handful of comparisons once per load.
	if err := validateConfigBounds(cfg); err != nil {
		return nil, nil, err
	}
	arch, schema, err := adapter(cfg)
	if err != nil {
		return nil, nil, err
	}
	arch.finalizeRoPE() // precompute inv-freq tables (base + scaling + rotary dim)
	if err := arch.validateResolved(); err != nil {
		return nil, nil, err
	}
	return arch, schema, nil
}

// validateResolved catches descriptor fields an adapter left at their zero value when zero
// is not a legal setting. Every adapter builds an Architecture by hand from a struct
// literal, so a field simply omitted is a compile-clean, load-clean, silently-wrong model.
//
// Both fields here were live bugs in lfm2Architecture, found 2026-08-31 against HF:
//
//   - AttnScale 0 makes every q·k score 0, so softmax returns a UNIFORM average over the
//     context. Invisible at one token (softmax of a single element is 1.0 whatever the
//     scale) and invisible in any greedy smoke test that only reads argmax, which matched
//     HF anyway. It showed up as cosine 0.928 at five tokens.
//   - NormEps 0 divides by rsqrt(variance) with no floor. Not merely imprecise: on a small
//     first-layer variance it scaled the norm output by a uniform 1.0185x.
//
// Checked here rather than in each adapter because the point is to cover the families
// nobody has written yet. An arch that genuinely wants no attention scaling sets 1.0.
func (a *Architecture) validateResolved() error {
	switch {
	// !(x > 0) rather than x <= 0: NaN fails EVERY comparison, so `a.AttnScale <= 0` is FALSE
	// for NaN and the old guard waved it through (M-06). gemma3 with a negative
	// query_pre_attn_scalar produces exactly that, and +Inf arrives from
	// num_attention_heads: 0 → Pow(0, -0.5). Both give a forward that runs and returns
	// garbage rather than one that refuses.
	case !(a.AttnScale > 0) || math.IsInf(a.AttnScale, 0):
		return fmt.Errorf("decoder(%s): AttnScale=%v must be finite and >0 (adapter omitted it, or a config value made it NaN/Inf; 1/sqrt(head_dim) is the usual value, 1.0 means deliberately unscaled)", a.Name, a.AttnScale)
	case a.Norm == NormRMS && !(a.NormEps > 0):
		return fmt.Errorf("decoder(%s): NormEps=%v must be >0 (adapter omitted it, or read the wrong config key)", a.Name, a.NormEps)
	}

	// M-06: POSITION INFORMATION MUST COME FROM SOMEWHERE. finalizeRoPE treats
	// RoPEGlobalBase <= 0 as "no tables", and applyRoPE is a silent no-op on an empty table,
	// so an adapter that never reads rope_theta loads clean and generates fluent,
	// POSITION-BLIND text — and drops YaRN with it. gpt-oss and llama4 both read only the
	// flat rope_theta, and transformers >= 5.10 nests it under rope_parameters; for
	// llama/mistral/qwen3 that is a loud error, and for these two it was silence.
	//
	// The three legitimate ways to have no global RoPE table are named explicitly rather
	// than inferred, so a new family that simply forgot cannot look like one of them:
	// GPT-2 has learned positions, Nemotron-H encodes NoPE layers as base 0, and MLA
	// carries its own decoupled rope dims.
	if !a.LearnedPosEmbed && a.nemotron == nil && a.mla == nil && len(a.ropeInvFreqGlobal) == 0 {
		return fmt.Errorf("decoder(%s): no position information — RoPEGlobalBase=%v yields no "+
			"inv-freq table, and the arch is not learned-position, Nemotron-H or MLA. The adapter "+
			"most likely did not read rope_theta (transformers >=5.10 nests it under "+
			"rope_parameters); a model loaded this way generates fluent but position-blind text",
			a.Name, a.RoPEGlobalBase)
	}

	// M-06: ZERO DIMS. Every one of these is a divisor, a slice length or both somewhere in
	// the forward. num_key_value_heads: 0 is the sharpest — `group := nH/nKV` is an integer
	// divide by zero, a panic in the decode goroutine rather than a load error.
	for _, d := range []struct {
		name string
		v    int
	}{
		{"HiddenDim", a.HiddenDim}, {"NumLayers", a.NumLayers}, {"NumHeads", a.NumHeads},
		{"NumKVHeads", a.NumKVHeads}, {"HeadDim", a.HeadDim}, {"VocabSize", a.VocabSize},
	} {
		if d.v <= 0 {
			return fmt.Errorf("decoder(%s): %s=%d must be >0", a.Name, d.name, d.v)
		}
	}
	if a.NumHeads%a.NumKVHeads != 0 {
		return fmt.Errorf("decoder(%s): NumHeads=%d is not a multiple of NumKVHeads=%d; "+
			"grouped-query attention divides one into the other", a.Name, a.NumHeads, a.NumKVHeads)
	}
	return nil
}

// validateConfigBounds rejects config dimensions that are implausible by orders of magnitude,
// before any adapter allocates from them. The ceilings are the maxGGUF* ones — named for GGUF
// because that is where they were first needed, but they are generic magnitude limits (largest
// open models: ~120 layers, hidden ~16K, vocab ~256K) and apply to any source of config JSON.
//
// Zero and negative are left to validateResolved, which reports them per-field AFTER the
// adapter has filled in what the config omitted (HeadDim is derived per family, and several
// MoE checkpoints leave IntermediateDim at zero legitimately). This function exists only to
// stop an allocation, so it checks only the upper bound.
func validateConfigBounds(cfg *Config) error {
	for _, d := range []struct {
		name string
		v    int
		max  int
	}{
		{"num_hidden_layers", cfg.NumLayers, maxGGUFLayers},
		{"hidden_size", cfg.HiddenDim, maxGGUFHidden},
		{"num_attention_heads", cfg.NumHeads, maxGGUFHeads},
		{"num_key_value_heads", cfg.NumKVHeads, maxGGUFHeads},
		{"vocab_size", cfg.VocabSize, maxGGUFVocabSize},
		{"num_experts", cfg.NumExperts, maxGGUFExperts},
	} {
		if d.v > d.max {
			return fmt.Errorf("decoder: %s %d exceeds the %d ceiling — refusing before it is "+
				"allocated from (a hostile or corrupt config, not a real model)", d.name, d.v, d.max)
		}
	}
	return nil
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
	// transformers ≥5.10 carries the dual RoPE bases in rope_parameters
	// ({full_attention, sliding_attention}.rope_theta) instead of the flat
	// rope_theta / rope_local_base_freq that gemma-3-270m uses (and the VL configs
	// nest it under text_config, flattened by loadConfig). Backfill the flat fields
	// from rope_parameters when absent so the adapter + ValidateAssumptions see
	// populated bases either format.
	if cfg.RoPEGlobalBase == 0 && len(cfg.RopeParameters) > 0 {
		full, sliding, err := parseRopeParameters(cfg.RopeParameters)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(gemma3): rope_parameters: %w", err)
		}
		if full != nil {
			cfg.RoPEGlobalBase = full.base
		}
		if sliding != nil {
			cfg.RoPELocalBase = sliding.base
		}
	}
	// Gemma3TextConfig class defaults. A VL checkpoint's nested text_config (e.g.
	// gemma-3-4b-it) carries only the dims that differ from the transformers
	// Gemma3TextConfig defaults — hidden/intermediate/layers/sliding_window — and
	// OMITS the rest, which HF fills from the class. Mirror that here: backfill
	// each scalar only when absent, so a flat config that sets them explicitly
	// (gemma-3-270m sets heads=4) is untouched, while the minimal VL text_config
	// loads with the correct 4B/12B/27B head + rope geometry. (Sizes that differ
	// from the defaults, like 270m's 4 heads, always set them in their config.)
	defScalarI := func(p *int, def int) {
		if *p == 0 {
			*p = def
		}
	}
	defScalarF := func(p *float64, def float64) {
		if *p == 0 {
			*p = def
		}
	}
	defScalarI(&cfg.NumHeads, 8)
	defScalarI(&cfg.NumKVHeads, 4)
	defScalarI(&cfg.HeadDim, 256)
	defScalarI(&cfg.VocabSize, 262208)
	defScalarI(&cfg.SlidingWindowPattern, 6)
	defScalarF(&cfg.QueryPreAttnScalar, 256)
	defScalarF(&cfg.RMSNormEps, 1e-6)
	defScalarF(&cfg.RoPEGlobalBase, 1_000_000)
	defScalarF(&cfg.RoPELocalBase, 10_000)
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

// gemma4Architecture expresses Gemma 4 dense (HF model_type
// "gemma4_unified_text"; GGUF arch "gemma4"). It is Gemma 3's stack — RMSNorm
// (1+w), Sandwich4 norm placement, GeGLU (gelu_tanh), QK-norm, query_pre_attn
// scaling, √hidden embed scale, tied head, no soft-capping, dual-base RoPE with a
// 5:1 local:global interleave — PLUS Gemma 4's per-layer attention deltas
// (gemma4Params): global/full layers use a wider head (global_head_dim), a single
// KV head (num_global_key_value_heads), shared K=V (attention_k_eq_v), and
// partial-rotary RoPE (partial_rotary_factor). The 12B dense is PLE-free; the
// E-models (E2B/E4B) add PLE (per-layer embeddings) — there is NO AltUp/Laurel
// (those are Gemma 3n). E2B also has cross-layer KV sharing + variable FFN +
// final-logit softcap 30.
//
// The forward pass (runLayersGemma4) consumes the per-layer deltas; E2B/E4B and
// the 12B dense (K=V) variants are parity-gated against the HF bf16 oracle.
func gemma4Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	globalRotary := 0
	if prf := cfg.gemma4PartialRotary(); prf > 0 && cfg.GlobalHeadDim > 0 {
		globalRotary = int(prf * float64(cfg.GlobalHeadDim))
	}
	// The real unified 26B keeps the RoPE bases nested in rope_parameters (no top-level
	// rope_theta / rope_local_base_freq); resolve them so the frequencies aren't 0/NaN.
	localBase, globalBase := cfg.gemma4RopeBases()
	// Gemma 4 26B-A4B: enable_moe_block turns on the parallel dense+MoE FFN sub-block.
	// The dense E2B/E4B/12B variants leave it false ⇒ MoE stays nil and their forward
	// is byte-unchanged. The router semantics (softmax-all → top-k → UNCONDITIONAL
	// renorm, + a weightless-norm/learned-scale pre-projection, + a per-expert scale)
	// are pinned in docs/task-gemma4-moe.md Phase 1a. The gemma4 forward (Phase 2) reads
	// these; the generic moeMLP/swiGLUExpert are NOT reused (experts are gelu-tanh, not SiLU).
	var moe *MoEConfig
	if cfg.EnableMoeBlock {
		if err := cfg.validateGemma4MoE(); err != nil {
			return nil, nil, err
		}
		moe = &MoEConfig{
			NumExperts:      cfg.NumExperts,
			TopK:            cfg.TopKExperts,
			NormTopKProb:    true,
			IntermediateDim: cfg.MoeIntermediateSize,
			RouterPreNorm:   true,
			PerExpertScale:  true,
		}
	}
	return &Architecture{
		Name:            "gemma4",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         cfg.HeadDim,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		// HF Gemma4RMSNorm is plain `x*weight` (weight init 1), NOT Gemma 3's
		// `x*(1+weight)` (init 0). So the SPEC is no offset (false). ⚠️ verify
		// against the real GGUF: if llama.cpp's gemma4 convert pre-subtracts 1
		// (as some Gemma converts do), flip this to true. (modeling_gemma4.py L~150)
		RMSAddOne:     false,
		NormEps:       cfg.RMSNormEps,
		NormPlacement: NormSandwich4,
		Act:           ActGeluTanh,
		QKNorm:        true,
		// Gemma 4 text attention uses scale 1.0 (modeling_gemma4.py L1194): the
		// learned q/k-norm weights absorb the scaling, unlike Gemma 3's explicit
		// query_pre_attn_scalar^-0.5. A new scale-less v_norm normalizes V.
		AttnScale:      1.0,
		SlidingWindow:  cfg.SlidingWindow,
		layerIsGlobal:  cfg.IsGlobalLayer,
		RoPELocalBase:  localBase,
		RoPEGlobalBase: globalBase,
		EmbedScale:     math.Sqrt(float64(cfg.HiddenDim)),
		TiedLMHead:     true,
		// Gemma 4 re-added Gemma 2's final-logit softcap (30 in the GGUF); Gemma 3
		// had dropped it. Attn softcap stays 0.
		FinalLogitSoftcap: cfg.FinalLogitSoftcap,
		MoE:               moe, // nil for the dense E2B/E4B/12B variants (byte-unchanged)
		gemma4: &gemma4Params{
			GlobalHeadDim:           cfg.GlobalHeadDim,
			NumGlobalKVHeads:        cfg.NumGlobalKVHeads,
			GlobalRotaryDim:         globalRotary,
			KVShared:                cfg.AttentionKEqV,
			SharedKVLayers:          cfg.SharedKVLayers,
			FFNPerLayer:             cfg.FFNPerLayer,
			HiddenSizePerLayerInput: cfg.HiddenSizePerLayerInput,
			VocabSizePerLayerInput:  cfg.VocabSizePerLayerInput,
		},
	}, &gemma3TensorSchema, nil // a dedicated gemma4 tensor schema lands with the loader work
}

// qwen3Architecture expresses Qwen3 dense: RMSNorm
// without the (1+w) offset, the Pre2 norm placement (no post-sublayer norms),
// SwiGLU, QK-norm (Qwen3 keeps it), 1/√head_dim attention scale, single-base
// RoPE, no embedding scale, and a separate lm_head (untied). No QKV bias
// (Qwen3 dropped Qwen2's). The tensor schema is qwen3TensorSchema.
func qwen3Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := backfillFlatRope(cfg, "qwen3"); err != nil {
		return nil, nil, err
	}
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

// qwen3MoeArchitecture expresses Qwen3-MoE (Qwen3-30B-A3B / Qwen3-Coder-30B-A3B-
// Instruct, both model_type "qwen3_moe" — confirmed against both real released
// config.json files, config-identical apart from max_position_embeddings):
// qwen3's dense attention (per-head q_norm/k_norm, GQA, no q/k/v bias,
// 1/√head_dim scale, single-base RoPE) with the FFN replaced on every layer by a
// sparse MoE — qwen2_moe's router shape (top-k of num_experts at
// moe_intermediate_size, norm_topk_prob) but with NO always-on shared expert.
// Verified against a real GGUF file's header too (unsloth/Qwen3-30B-A3B-GGUF
// Q2_K, HTTP-Range-fetched): architecture string "qwen3moe", plain
// {arch}.attention.*/{arch}.expert_*/{arch}.rope.freq_base metadata (no
// sliding-window or YaRN keys), and a tensor set with attn_q_norm/attn_k_norm +
// ffn_gate_inp/ffn_{gate,up,down}_exps but no ffn_*_shexp — so the existing
// generic GGUF loadLayer path (gated on arch.QKNorm / arch.MoE /
// arch.MoE.SharedIntermediateDim>0) handles this family with no new loader code,
// same as the safetensors path. The tensor schema is qwen3MoeTensorSchema.
func qwen3MoeArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := backfillFlatRope(cfg, "qwen3_moe"); err != nil {
		return nil, nil, err
	}
	if err := cfg.validateQwen3Moe(); err != nil {
		return nil, nil, err
	}
	scaling, err := parseRopeScaling(cfg.RopeScaling)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(qwen3_moe): %w", err)
	}
	normTopK := true // HF Qwen3MoeConfig default (norm_topk_prob true)
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	return &Architecture{
		Name:            "qwen3_moe",
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
		MoE: &MoEConfig{
			NumExperts:      cfg.NumExperts,
			TopK:            cfg.NumExpertsPerTok,
			NormTopKProb:    normTopK,
			IntermediateDim: cfg.MoeIntermediateSize,
			// No SharedIntermediateDim: qwen3_moe has no shared expert (unlike qwen2_moe).
		},
		AttnScale:      math.Pow(float64(cfg.HeadDim), -0.5),
		SlidingWindow:  0, // full attention
		layerIsGlobal:  nil,
		RoPELocalBase:  cfg.RoPEGlobalBase,
		RoPEGlobalBase: cfg.RoPEGlobalBase,
		RotaryDim:      cfg.rotaryDim(),
		ropeScaling:    scaling,
		EmbedScale:     0,
		TiedLMHead:     false, // finalized from lm_head.weight presence at load
	}, &qwen3MoeTensorSchema, nil
}

// backfillFlatRope fills the flat rope_theta / rope_scaling fields from transformers >=5.10's
// rope_parameters object, for the SINGLE-BASE architectures (llama, mistral, qwen3).
//
// transformers moved RoPE config out of top-level rope_theta/rope_scaling and into
// rope_parameters — {"rope_theta": 1e4, "rope_type": "default"}, with linear/yarn/llama3
// scaling carried inside the same object. Archs that only read the flat fields therefore
// REJECT any checkpoint saved by a current transformers, with "rope_theta must be >0" — a
// hard load failure on freshly re-saved upstream weights, not a niche path. phi3 already
// handled this via parseRopeFlat and gemma3/mellum handle the per-layer-type nesting
// (full_attention/sliding_attention); llama and mistral did not, and each has its own
// architecture func, which is exactly how one got fixed and the other did not. Hence one
// helper rather than a third copy.
func backfillFlatRope(cfg *Config, arch string) error {
	if cfg.RoPEGlobalBase != 0 || len(cfg.RopeParameters) == 0 {
		return nil // already flat (older config, or a GGUF) — nothing to backfill
	}
	spec, _, err := parseRopeFlat(cfg.RopeParameters)
	if err != nil {
		return fmt.Errorf("decoder(%s): rope_parameters: %w", arch, err)
	}
	cfg.RoPEGlobalBase = spec.base
	// Scaling may now live inside rope_parameters; hand it to the flat field the caller's
	// parseRopeScaling already understands. Never clobber an explicit rope_scaling.
	if spec.scaling != nil && len(cfg.RopeScaling) == 0 {
		cfg.RopeScaling = cfg.RopeParameters
	}
	return nil
}

// llamaArchitecture expresses Llama-2/3 dense: like Qwen3 (RMSNorm no-offset,
// Pre2 placement, SwiGLU, 1/√head_dim scale, single-base RoPE, no embed scale)
// but WITHOUT QK-norm — Llama's attention applies RoPE to raw q/k. head_dim is
// derived (headDim()) since many Llama configs omit it. The LM head is tied on
// the small text models (1B/3B) and untied on 8B+, finalized from
// lm_head.weight presence at load. validateLlama rejects scaled RoPE (G4) and
// attention bias (a later add), so reaching here implies a plain checkpoint.
func llamaArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := backfillFlatRope(cfg, "llama"); err != nil {
		return nil, nil, err
	}
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

// cohereArchitecture expresses Cohere / Command-R (model_type "cohere":
// Command-R, Command-R+, Aya, Aya-Expanse). Two things break the Llama mold:
//
//   - Norm is bias-free LayerNorm (mean-subtract + variance), NOT RMSNorm —
//     NormLayer with a nil bias (the generic loader leaves PreAttnNormBias nil).
//   - The block is PARALLEL: one shared input_layernorm feeds both attention and
//     the MLP, whose outputs sum into a single residual add (NormParallel). There
//     is no post_attention_layernorm — the schema's PreMLPNorm is empty.
//
// Everything else rides shipped features: gated SiLU MLP, GQA (Command-R v01 is
// MHA, kv==heads), tied 256k embeddings, full-dim RoPE. RoPE is GPT-J interleaved
// (ropeInterleave). logit_scale MULTIPLIES the logits in HF; goinfer's LogitScale
// divides, so we store its reciprocal and reuse the Granite logit-scale kernel.
// use_qk_norm (Command-R+) and sliding_window (that's cohere2) are rejected in
// validateCohere, so reaching here implies a plain cohere1 checkpoint.
func cohereArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	// Older CohereLabs configs carry rope_theta top-level; transformers ≥5 nests it
	// under rope_parameters. Backfill the flat field so both load identically.
	if err := backfillFlatRope(cfg, "cohere"); err != nil {
		return nil, nil, err
	}
	if err := cfg.validateCohere(); err != nil {
		return nil, nil, err
	}
	hd := cfg.headDim()
	return &Architecture{
		Name:            "cohere",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormLayer, // bias-free LayerNorm (mean-subtract), NOT RMSNorm
		NormEps:         cfg.LayerNormEps,
		NormPlacement:   NormParallel, // one input norm → attn + mlp → single residual add
		Act:             ActSiLU,
		QKNorm:          false, // cohere1 use_qk_norm rejected in validateCohere (Phase 1)
		AttnScale:       math.Pow(float64(hd), -0.5),
		SlidingWindow:   0, // cohere1: full attention (sliding_window ⇒ cohere2, rejected)
		layerIsGlobal:   nil,
		RoPELocalBase:   cfg.RoPEGlobalBase, // single base (rope_theta)
		RoPEGlobalBase:  cfg.RoPEGlobalBase,
		RotaryDim:       cfg.rotaryDim(), // 0 = full head_dim
		ropeInterleave:  true,            // GPT-J pairwise rotation (rope_gptj), not NeoX
		EmbedScale:      0,
		LogitScale:      1.0 / cfg.LogitScale, // reciprocal: HF multiplies, goinfer divides
		TiedLMHead:      true,                 // Cohere always ties (no lm_head.weight in checkpoint)
	}, &cohereTensorSchema, nil
}

// cohere2Architecture expresses Cohere2 / Command-R7B (model_type "cohere2":
// Command-R7B, Command-A, R7B-arabic). It is cohere1's stack (bias-free LayerNorm,
// parallel block, gated SiLU, tied embeddings, GPT-J interleaved RoPE, reciprocal
// logit_scale) with two additions and one subtraction:
//
//   - Interleaved SLIDING-WINDOW / full attention: every sliding_window_pattern-th
//     layer is global (full attention), the rest are windowed at sliding_window.
//   - The GLOBAL layers are NoPE (no positional encoding); only the sliding layers
//     carry RoPE. layerNoPE == the global predicate, so isNoPELayer skips RoPE on
//     exactly the full-attention layers (the per-layer NoPE primitive Llama 4 wants).
//   - NO QK-norm at all (cohere2 dropped cohere1's config-gated q/k norm).
//
// Reuses cohereTensorSchema (identical tensor names — one shared input_layernorm
// per layer, no biases, tied head).
func cohere2Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := backfillFlatRope(cfg, "cohere2"); err != nil {
		return nil, nil, err
	}
	if err := cfg.validateCohere2(); err != nil {
		return nil, nil, err
	}
	hd := cfg.headDim()
	// Snapshot the classifier inputs so the closure doesn't retain cfg. Global =
	// full-attention layer = NoPE; local = sliding + RoPE (Config.IsGlobalLayer is
	// the authoritative layer_types-then-pattern rule).
	pattern := cfg.SlidingWindowPattern
	layerTypes := append([]string(nil), cfg.LayerTypes...)
	isGlobal := func(i int) bool {
		if i >= 0 && i < len(layerTypes) {
			return layerTypes[i] == "full_attention"
		}
		if pattern <= 0 {
			return true
		}
		return (i+1)%pattern == 0
	}
	return &Architecture{
		Name:            "cohere2",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormLayer,
		NormEps:         cfg.LayerNormEps,
		NormPlacement:   NormParallel,
		Act:             ActSiLU,
		QKNorm:          false, // cohere2 has no QK-norm
		AttnScale:       math.Pow(float64(hd), -0.5),
		SlidingWindow:   cfg.SlidingWindow, // sliding layers window; global layers full
		layerIsGlobal:   isGlobal,
		layerNoPE:       isGlobal, // global (full) layers are NoPE; sliding layers keep RoPE
		RoPELocalBase:   cfg.RoPEGlobalBase,
		RoPEGlobalBase:  cfg.RoPEGlobalBase,
		RotaryDim:       cfg.rotaryDim(),
		ropeInterleave:  true, // GPT-J pairwise on the sliding (RoPE) layers
		EmbedScale:      0,
		LogitScale:      1.0 / cfg.LogitScale,
		TiedLMHead:      true,
	}, &cohereTensorSchema, nil
}

// mistralArchitecture expresses Mistral dense: the llama descriptor (RMS
// no-offset, Pre2, SwiGLU, single-base RoPE, no QK-norm, no bias, derived
// head_dim) with sliding-window attention on EVERY layer (Gemma's window
// machinery, but all-local rather than 5:1). A checkpoint with sliding_window
// null/0 (Mistral-v0.2+) falls back to full attention. The tensor schema is
// llamaTensorSchema (Mistral and Llama share tensor names).
func mistralArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := backfillFlatRope(cfg, "mistral"); err != nil {
		return nil, nil, err
	}
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

// ministral3Architecture expresses Ministral 3 (mistralai/Ministral-3-{3b,8b,14b}, model_type
// "mistral3" — the OUTER Mistral3ForConditionalGeneration wrapper's type, which `loadConfig`'s
// generic text_config flattening re-applies LAST over whatever the nested text_config's own
// model_type ("ministral3") set; confirmed by reading that flattening code directly rather than
// assumed, since it decides which registry key this family actually resolves under): Mistral's
// GQA skeleton (reused verbatim: same tensor names, confirmed by instantiating
// Ministral3ForCausalLM directly and reading its state_dict) with two real deltas Phase 0 found,
// both checked against the released config rather than assumed from the brief's own framing:
//
//  1. `sliding_window: null` on the real release — mistralArchitecture already treats
//     SlidingWindow<=0 as full attention (its own "0 ⇒ full attention" comment), so this needs
//     no new code; the brief's own caution ("verify... whether it is every layer") was answered
//     "there is no window at all", not "yes, every layer".
//  2. `rope_parameters` is `rope_type: "yarn"` with a real, load-bearing THIRD field alongside
//     the standard YaRN ones: `llama_4_scaling_beta`. Verified against the real
//     modular_ministral3.py (not guessed): `get_llama_4_attn_scale` multiplies the QUERY by
//     `1 + beta·ln(1 + floor(pos/original_max_position_embeddings))`, AFTER RoPE, on EVERY
//     layer — the exact formula llama4Architecture's own attnTemp/floorScale primitive already
//     implements (`decoder/forward_llama4.go`), but Llama4 applies it INSTEAD of RoPE on NoPE
//     layers only, never combined with RoPE the way this family needs. Generalized to the two
//     new Architecture fields AttnTempBeta/AttnTempOrigMaxPos (see their own comment) and wired
//     into the GENERIC causalAttention/forwardN paths rather than copied into an own-forward
//     function, since every existing family leaves both fields at their zero-value no-op.
//
// Also confirmed: `mscale`/`mscale_all_dim` (both 1.0 on the release) are DeepSeek's own spelling
// of the YaRN attention_factor, not the generic `attention_factor` key parseRopeScaling reads —
// left unhandled, its own default (0.1·ln(16)+1 ≈ 1.277) would silently override the correct
// value (1.0, since mscale == mscale_all_dim here, same reasoning deepseekArchitecture's own
// comment gives for V2-Lite). Overridden the same way deepseekArchitecture already does.
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

// gpt2Architecture expresses GPT-2: the GPT-2/NeoX class
// that breaks the Llama mold on several axes — LayerNorm (mean-centered, with
// bias) instead of RMSNorm, learned absolute position embeddings instead of
// RoPE, a non-gated GELU MLP (up→gelu→down) instead of a gated one, fused q/k/v
// with bias, an attention output bias, and tied embeddings. The Conv1D weight
// layout + fused projections need a dedicated loader (buildGPT2Weights), so
// this returns the gpt2TensorSchema as a marker; the schema's field names are
// unused.
// gpt2Act maps GPT-2's activation_function to the ActKind that actually implements it.
// "gelu_new" (and the empty default, which is GPT-2's own) is the TANH approximation;
// "gelu" is the exact erf function. validateGPT2 accepts both, and before this they both
// ran geluTanh — so a checkpoint declaring the exact function silently got the
// approximation. The two differ by up to 4.73e-4, small enough to pass unnoticed and
// still wrong. Every shipping GPT-2 config declares gelu_new, so nothing in tree moves.
func gpt2Act(name string) ActKind {
	if name == "gelu" {
		return ActGelu
	}
	return ActGeluTanh
}

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
		Act:             gpt2Act(cfg.ActivationFunction),
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

// mixtralArchitecture expresses Mixtral: the llama
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

// qwen35Architecture expresses Qwen3.5/3.6-MoE (model_type qwen3_5_moe): a 3:1
// hybrid where most layers are Gated DeltaNet (linear attention with a recurrent
// matrix state — its own forward path) and the rest are QK-norm softmax attention
// with partial RoPE, over a routed + shared MoE on every layer. The descriptor
// marks the per-layer kind (layerIsLinear) and carries the DeltaNet geometry;
// the softmax layers reuse the uniform attention fields. See docs/qwen3_5_moe.md.
func qwen35Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateQwen35(); err != nil {
		return nil, nil, err
	}
	spec, partialRotary, err := parseRopeFlat(cfg.RopeParameters)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(qwen3_5_moe): %w", err)
	}
	rotaryDim := 0
	if partialRotary > 0 && partialRotary < 1 {
		rotaryDim = int(partialRotary * float64(cfg.HeadDim))
	}
	// Qwen3_5MoeTopKRouter ALWAYS renormalizes the top-k probabilities
	// (router_top_value /= sum), regardless of the (absent) norm_topk_prob field.
	normTopK := true
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	return &Architecture{
		Name:            "qwen3_5_moe",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         cfg.HeadDim,
		IntermediateDim: cfg.IntermediateDim, // dense width (vestigial; experts use the MoE width)
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       true, // Qwen3_5MoeRMSNorm: output × (1 + weight), weight zero-init
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		MoE: &MoEConfig{
			NumExperts:            cfg.NumExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          normTopK,
			IntermediateDim:       cfg.MoeIntermediateSize,
			SharedIntermediateDim: cfg.SharedExpertIntermediateSize,
		},
		QKNorm:           true, // softmax layers have per-head q_norm/k_norm
		AttnScale:        math.Pow(float64(cfg.HeadDim), -0.5),
		layerIsGlobal:    cfg.IsGlobalLayer, // softmax layers are full attention
		layerIsLinear:    cfg.IsLinearLayer, // Gated DeltaNet layers
		RoPELocalBase:    spec.base,
		RoPEGlobalBase:   spec.base, // single base; only the softmax layers use RoPE
		ropeScaling:      spec.scaling,
		ropeScalingLocal: spec.scaling,
		RotaryDim:        rotaryDim, // partial_rotary_factor · head_dim
		EmbedScale:       0,
		TiedLMHead:       false, // finalized from lm_head.weight presence at load
		qwen35: &qwen35Params{
			ConvKernel:    cfg.LinearConvKernelDim,
			KeyHeadDim:    cfg.LinearKeyHeadDim,
			ValueHeadDim:  cfg.LinearValueHeadDim,
			NumKeyHeads:   cfg.LinearNumKeyHeads,
			NumValueHeads: cfg.LinearNumValueHeads,
		},
	}, &qwen35TensorSchema, nil
}

// qwen3NextArchitecture expresses Qwen3-Next (model_type qwen3_next): the same
// Gated-DeltaNet/softmax/MoE hybrid shape as qwen3_5_moe (verified field-for-field
// against the real config — DeltaNet geometry, MoE dims, QK-norm, RMSNorm's Gemma3-
// style (1+w) convention all match, confirmed against the real
// modular_qwen3_next.py rather than assumed from the family resemblance), with
// two real config-SHAPE deltas, not just value changes:
//
//  1. No layer_types field at all — the per-layer pattern is COMPUTED from
//     full_attention_interval (normalizeQwen3NextLayerTypes, called below,
//     before validateQwen35 can require LayerTypes to already be populated).
//  2. partial_rotary_factor is a TOP-LEVEL field, not nested inside
//     rope_parameters the way qwen3_5_moe's is — confirmed against
//     configuration_qwen3_next.py's kwargs.setdefault("partial_rotary_factor", …).
//     The real released config also has no rope_parameters object at all (plain
//     top-level rope_theta + rope_scaling), so this mirrors deepseekArchitecture's
//     dual-path RoPE resolution (nested-if-present, flat-fields otherwise) rather
//     than qwen35Architecture's nested-only parseRopeFlat, which would hard-error
//     on a real Qwen3-Next config (parseRopeSpec refuses an empty rope_parameters).
func qwen3NextArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.normalizeQwen3NextLayerTypes(); err != nil {
		return nil, nil, err
	}
	if err := cfg.validateQwen3Next(); err != nil {
		return nil, nil, err
	}
	var base float64
	var scaling *ropeScaling
	var partialRotary float64
	if len(cfg.RopeParameters) > 0 {
		spec, pr, err := parseRopeFlat(cfg.RopeParameters)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(qwen3_next): %w", err)
		}
		base, scaling, partialRotary = spec.base, spec.scaling, pr
	} else {
		base = cfg.RoPEGlobalBase
		sc, err := parseRopeScaling(cfg.RopeScaling)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(qwen3_next): %w", err)
		}
		scaling = sc
		partialRotary = cfg.PartialRotaryFactor
	}
	rotaryDim := 0
	if partialRotary > 0 && partialRotary < 1 {
		rotaryDim = int(partialRotary * float64(cfg.HeadDim))
	}
	normTopK := true
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	return &Architecture{
		Name:            "qwen3_next",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         cfg.HeadDim,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       true, // Qwen3NextRMSNorm(Gemma3RMSNorm): pass — inherits Gemma's (1+w), verified against modular_qwen3_next.py
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		MoE: &MoEConfig{
			NumExperts:            cfg.NumExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          normTopK,
			IntermediateDim:       cfg.MoeIntermediateSize,
			SharedIntermediateDim: cfg.SharedExpertIntermediateSize,
		},
		QKNorm:           true,
		AttnScale:        math.Pow(float64(cfg.HeadDim), -0.5),
		layerIsGlobal:    cfg.IsGlobalLayer,
		layerIsLinear:    cfg.IsLinearLayer,
		RoPELocalBase:    base,
		RoPEGlobalBase:   base,
		ropeScaling:      scaling,
		ropeScalingLocal: scaling,
		RotaryDim:        rotaryDim,
		EmbedScale:       0,
		TiedLMHead:       false,
		qwen35: &qwen35Params{
			ConvKernel:        cfg.LinearConvKernelDim,
			KeyHeadDim:        cfg.LinearKeyHeadDim,
			ValueHeadDim:      cfg.LinearValueHeadDim,
			NumKeyHeads:       cfg.LinearNumKeyHeads,
			NumValueHeads:     cfg.LinearNumValueHeads,
			FusedDeltaNetProj: true, // in_proj_qkvz/in_proj_ba, not four separate tensors — see qwen35Params doc
		},
	}, &qwen35TensorSchema, nil
}

// qwen2Architecture expresses Qwen2/Qwen2.5 dense: identical to the llama
// descriptor (RMS no-offset, Pre2, SwiGLU, single-base RoPE, no QK-norm, derived
// head_dim, tied-or-untied head) plus QKVBias — Qwen2 carries an additive bias
// on the q/k/v projections (o_proj stays biasless). The tensor schema is
// qwen2TensorSchema.
func qwen2Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := backfillFlatRope(cfg, "qwen2"); err != nil {
		return nil, nil, err
	}
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

// qwen2_5_vlArchitecture expresses the Qwen2.5-VL TEXT decoder: a Qwen2 dense
// decoder (q/k/v bias, GQA, RMSNorm, SwiGLU, derived head_dim) whose RoPE config
// lives under transformers-5.x's nested rope_parameters {mrope_section, rope_theta}
// instead of a top-level rope_theta. For TEXT, m-RoPE degenerates to standard
// scalar RoPE (the 3 position components are equal), so the text path is exactly
// Qwen2 — the 3D-position machinery only engages on the image path (P5). The
// vision_config is ignored here, like the Gemma 3 VL text side. (P5)
func qwen2_5_vlArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	// mrope_section + rope_theta live under EITHER the transformers-5.x nested
	// rope_parameters {mrope_section, rope_theta} (the tiny synthetic) OR the older
	// top-level rope_scaling {type:mrope, mrope_section} + rope_theta (the real 3B).
	// Extract from whichever is present.
	var section []int
	if len(cfg.RopeParameters) > 0 {
		var rp struct {
			RopeTheta    float64 `json:"rope_theta"`
			MRopeSection []int   `json:"mrope_section"`
		}
		if err := json.Unmarshal(cfg.RopeParameters, &rp); err != nil {
			return nil, nil, fmt.Errorf("decoder(qwen2_5_vl): parse rope_parameters: %w", err)
		}
		section = rp.MRopeSection
		if cfg.RoPEGlobalBase == 0 {
			cfg.RoPEGlobalBase = rp.RopeTheta
		}
	}
	if len(cfg.RopeScaling) > 0 {
		var rs struct {
			Type         string `json:"type"`
			RopeType     string `json:"rope_type"`
			MRopeSection []int  `json:"mrope_section"`
		}
		if err := json.Unmarshal(cfg.RopeScaling, &rs); err != nil {
			return nil, nil, fmt.Errorf("decoder(qwen2_5_vl): parse rope_scaling: %w", err)
		}
		if rs.Type == "mrope" || rs.RopeType == "mrope" {
			if section == nil {
				section = rs.MRopeSection
			}
			// m-RoPE is not a parseRopeScaling kind (linear/llama3/yarn) — the
			// per-component split IS the "scaling". Clear it so qwen2Architecture's
			// parseRopeScaling doesn't reject rope_type "mrope".
			cfg.RopeScaling = nil
		}
	}
	cfg.MRopeSection = section
	arch, schema, err := qwen2Architecture(cfg)
	if err != nil {
		return nil, nil, err
	}
	arch.Name = "qwen2_5_vl"
	arch.MRopeSection = section
	return arch, schema, nil
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

// glm4moeArchitecture expresses GLM-4.5/4.6 (model_type glm4_moe): qwen3-like
// softmax attention (per-head QK-norm, partial RoPE, NO q/k/v bias) over a
// DeepSeek-style MoE. The MoE differs from Qwen2-MoE on three axes, all carried by
// MoEConfig: experts are scored by per-expert sigmoid (not softmax), an
// e_score_correction_bias steers the top-k SELECTION while the weights stay the
// un-biased sigmoid scores, the routed weights are scaled by routed_scaling_factor,
// and the always-on shared expert is added UNGATED. first_k_dense_replace layers at
// the top are plain dense MLPs (no router) — the only family with mixed dense/MoE
// layers, handled by the loader populating Experts only on i ≥ FirstKDense. The
// num_nextn_predict_layers MTP head is dropped (only num_hidden_layers load). The
// tensor schema is glm4moeTensorSchema. See docs/completed/task-model-families-glm-granite.md.
func glm4moeArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateGlm4Moe(); err != nil {
		return nil, nil, err
	}
	// GLM nests rope_theta + partial_rotary_factor inside rope_parameters
	// (rope_type "default"); there is no top-level rope_theta.
	spec, partialRotary, err := parseRopeFlat(cfg.RopeParameters)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(glm4_moe): %w", err)
	}
	hd := cfg.headDim()
	rotaryDim := 0
	if partialRotary > 0 && partialRotary < 1 {
		rotaryDim = int(partialRotary * float64(hd))
	}
	normTopK := true // HF Glm4MoeConfig default
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	return &Architecture{
		Name:            "glm4_moe",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim, // dense width — used by the first_k_dense_replace prefix layers
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		QKVBias:         cfg.AttentionBias, // GLM-4.5/4.6 carry q/k/v bias (o_proj biasless); the tiny synthetic turns it off
		QKNorm:          cfg.UseQKNorm,     // use_qk_norm: true on the tiny synthetic, FALSE on the real GLM-4.5-Air/4.6
		FirstKDense:     cfg.FirstKDenseReplace,
		MoE: &MoEConfig{
			NumExperts:            cfg.NRoutedExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          normTopK,
			IntermediateDim:       cfg.MoeIntermediateSize,
			SharedIntermediateDim: cfg.NSharedExperts * cfg.MoeIntermediateSize, // shared expert(s) at moe_intermediate_size each
			RouterSigmoid:         true,
			RoutedScale:           cfg.RoutedScalingFactor,
			SharedUngated:         true,
		},
		AttnScale:      math.Pow(float64(hd), -0.5),
		SlidingWindow:  0, // full attention
		layerIsGlobal:  nil,
		RoPELocalBase:  spec.base,
		RoPEGlobalBase: spec.base, // single base (rope_parameters.rope_theta)
		ropeScaling:    spec.scaling,
		RotaryDim:      rotaryDim, // partial_rotary_factor · head_dim
		EmbedScale:     0,
		TiedLMHead:     false, // finalized from lm_head.weight presence at load
	}, &glm4moeTensorSchema, nil
}

// ropeBaseFlatOrNested reads a single-base RoPE config in EITHER spelling: transformers
// >=5.10's nested rope_parameters object, or the top-level rope_theta + rope_scaling that
// every checkpoint saved before it carries. backfillFlatRope is the same accommodation for
// the archs that read only the flat fields; this is the mirror, for archs that read only
// the nested object. deepseekArchitecture and phi3Architecture inline this if/else; they
// are left alone deliberately — converting them is a no-op refactor that would re-run two
// validated families' oracles for no behaviour change.
func ropeBaseFlatOrNested(cfg *Config, arch string) (base float64, scaling *ropeScaling, err error) {
	if len(cfg.RopeParameters) > 0 {
		spec, _, perr := parseRopeFlat(cfg.RopeParameters)
		if perr != nil {
			return 0, nil, fmt.Errorf("decoder(%s): %w", arch, perr)
		}
		return spec.base, spec.scaling, nil
	}
	sc, perr := parseRopeScaling(cfg.RopeScaling)
	if perr != nil {
		return 0, nil, fmt.Errorf("decoder(%s): %w", arch, perr)
	}
	return cfg.RoPEGlobalBase, sc, nil
}

// nopePredicate turns HF's position_embedding_type into the per-layer NoPE predicate.
// "nope" ⇒ every attention layer skips RoPE; absent or "rope" ⇒ nil (every layer ropes),
// which is both HF's default and the pre-existing behaviour.
func nopePredicate(kind string) func(int) bool {
	if kind == "nope" {
		return func(int) bool { return true }
	}
	return nil
}

// lfm2Architecture expresses LFM2 / LFM2.5 (model_type lfm2): a gated-short-convolution +
// softmax-attention hybrid. Every layer has a SwiGLU FFN; layer_types decides whether its
// mixer is a conv block (22 of 30 on LFM2.5-2.6B) or GQA attention with per-head RMSNorm on
// Q and K (8 of 30, at 2/5/9/13/17/21/24/27).
//
// It is EXPERIMENTAL tier: validated against the HF reference on a real checkpoint, not
// against a full-model T3.
//
// Three facts here were checked against the released LFM2.5-2.6B rather than inherited from
// the original scoping brief, and two of them contradicted it:
//
//   - QK-norm is RMSNorm, not LayerNorm. The brief said LayerNorm; the reference uses
//     Lfm2RMSNorm(head_dim) per head, and the checkpoint carries q_layernorm.weight with NO
//     bias tensor anywhere in its 266. That is the difference between reusing the existing
//     hardcoded QK-norm path and writing a bias-carrying LayerNorm variant.
//   - vocab is 128,000 (the brief said 65,536, which is the older LFM2-2.6B tokenizer), and
//     rope_theta is 1e7 (was 1e6).
//   - intermediate_size is STATED (10752), not computed from block_multiple_of — so the
//     block_ffn_dim_multiplier / block_multiple_of machinery is inert here and is not read.
func lfm2Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateLFM2(); err != nil {
		return nil, nil, err
	}
	hd := cfg.HeadDim
	if hd == 0 {
		hd = cfg.HiddenDim / cfg.NumHeads
	}
	// rope_parameters is nested on every released LFM2.5 checkpoint (rope_type "default", no
	// scaling), but accept the flat form too — the same shape granite/deepseek/phi3 use, and
	// the tiny fixture is built by a different transformers version than the release.
	base, scaling, err := ropeBaseFlatOrNested(cfg, "lfm2")
	if err != nil {
		return nil, nil, err
	}
	types := cfg.LayerTypes
	return &Architecture{
		Name:            "lfm2",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		NormEps:         cfg.NormEps,                 // "norm_eps", NOT rms_norm_eps -- see Config.NormEps
		AttnScale:       math.Pow(float64(hd), -0.5), // Lfm2Attention: self.scaling = head_dim**-0.5
		NormPlacement:   NormPre2,                    // operator_norm before the mixer, ffn_norm before the FFN
		Act:             ActSiLU,                     // SwiGLU (block_use_swiglu true)
		QKVBias:         false,
		QKNorm:          true, // per-head RMSNorm over head_dim — see the note above
		RoPELocalBase:   base,
		RoPEGlobalBase:  base,
		ropeScaling:     scaling,
		RotaryDim:       0, // full rotary
		TiedLMHead:      true,
		// conv_dim is optional and absent from Lfm2Config-written checkpoints; hidden_size is
		// what upstream's Lfm2ShortConv actually uses. validateLFM2 has already refused any
		// non-zero conv_dim that disagrees with hidden_size, so this is a default, not a guess.
		lfm2:        &lfm2Params{ConvDim: cfg.HiddenDim, ConvLCache: cfg.ConvLCache},
		layerIsConv: func(i int) bool { return i < len(types) && types[i] == "conv" },
	}, &lfm2TensorSchema, nil
}

// graniteArchitecture expresses Granite-4.0-H (model_type granitemoehybrid): a
// Mamba-2 + softmax-attention hybrid (per-layer kind from layer_types) with a routed
// + shared MoE on EVERY layer, plus four Granite scalar multipliers. The mamba
// layers run the selective-scan mixer (mamba2.go) with recurrent state in the hybrid
// cache; the attention layers are plain GQA + RoPE (no QK-norm, no bias) scaled by
// attention_multiplier instead of 1/√d. The MoE is GraniteMoe-style (fused gate+up
// experts + an ungated shared expert) but its softmax-top-k routing is identical to
// the Mixtral path, so moeMLP serves it unchanged once the fused tensors are split
// at load. Dedicated loader (buildGraniteWeights) + forward (runLayersGranite).
func graniteArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateGranite(); err != nil {
		return nil, nil, err
	}
	// RoPE: the tiny fixture (built by instantiating GraniteMoeHybridConfig on a current
	// transformers) carries rope_parameters; the RELEASED granite-4.0-h checkpoints were
	// saved by transformers 4.56 and carry a top-level rope_theta + rope_scaling instead.
	// Accept both — the same shape deepseekArchitecture and phi3Architecture already use.
	// Reading only rope_parameters made goinfer reject every published Granite-4.0-H.
	base, scaling, err := ropeBaseFlatOrNested(cfg, "granite")
	if err != nil {
		return nil, nil, err
	}
	if cfg.NumHeads <= 0 { // hd = hidden/heads below — a hostile config.json has no validateGGUFDims gate (M16)
		return nil, nil, fmt.Errorf("decoder(granite): num_attention_heads must be > 0, got %d", cfg.NumHeads)
	}
	hd := cfg.HiddenDim / cfg.NumHeads
	types := cfg.LayerTypes
	one := func(v float64) float32 {
		if v == 0 {
			return 1
		}
		return float32(v)
	}
	attnMul := cfg.AttentionMultiplier
	if attnMul == 0 {
		attnMul = math.Pow(float64(hd), -0.5) // HF default when unset
	}
	logitScale := cfg.LogitsScaling
	if logitScale == 0 {
		logitScale = 1
	}
	return &Architecture{
		Name:            "granitemoehybrid",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim, // expert FFN width (GraniteMoe uses intermediate_size)
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		QKVBias:         false,
		QKNorm:          false,
		MoE: &MoEConfig{
			NumExperts:            cfg.NumLocalExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          true, // top-k then softmax == softmax then top-k+renorm
			IntermediateDim:       cfg.IntermediateDim,
			SharedIntermediateDim: cfg.SharedIntermediateSize,
			SharedUngated:         true, // GraniteMoe shared_mlp has no sigmoid gate
		},
		AttnScale:      attnMul, // Granite attention_multiplier (not 1/√d)
		RoPELocalBase:  base,
		RoPEGlobalBase: base,
		ropeScaling:    scaling,
		RotaryDim:      0, // full rotary
		// position_embedding_type "nope" ⇒ NO positional encoding on the attention layers
		// (HF builds no rotary_emb at all and passes position_embeddings=None). It is what
		// granite-4.0-h-tiny ships, and absent/"rope" keeps the roped path — so the flag is
		// read rather than assumed. Roping a NoPE checkpoint still generates text; it just
		// generates the wrong text, which is why this is config-driven and gated by the
		// real-checkpoint oracle rather than by the tiny fixture (which is roped).
		layerNoPE:    nopePredicate(cfg.PositionEmbeddingType),
		EmbedScale:   0, // embedding_multiplier applied in runLayersGranite, not the Gemma sqrt path
		TiedLMHead:   false,
		LogitScale:   logitScale,
		layerIsMamba: func(i int) bool { return i < len(types) && types[i] == "mamba" },
		granite: &graniteParams{
			NHeads: cfg.MambaNHeads, HeadDim: cfg.MambaDHead, DState: cfg.MambaDState,
			NGroups: cfg.MambaNGroups, DConv: cfg.MambaDConv,
			EmbMul: one(cfg.EmbeddingMultiplier), ResidMul: one(cfg.ResidualMultiplier),
		},
	}, &graniteTensorSchema, nil
}

// graniteDenseArchitecture expresses dense Granite 4.2 (ibm-granite/granite-4.2-{3b,8b,30b},
// model_type "granite", GraniteForCausalLM): a plain llama skeleton — confirmed byte-identical
// tensor names (self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj, input_layernorm/
// post_attention_layernorm, no bias, no QK-norm) by instantiating GraniteForCausalLM directly and
// reading its state_dict, not assumed from Granite-4.0-H's own tensor names — plus Granite's four
// scalar multipliers, THREE of which are already generic on Architecture (embedding_multiplier →
// EmbedScale, attention_multiplier → AttnScale in place of 1/√d, logits_scaling → LogitScale;
// granitemoehybrid's own comment on EmbedScale notes it applies "the Gemma sqrt path", but the
// mechanism itself — multiply the embedding by a constant — is generic regardless of how that
// constant is derived). residual_multiplier is the one exception: granitemoehybrid's own-forward
// (runLayersGranite) applies it via graniteParams.ResidMul, and the generic uniform-layer forward
// this family rides has no such hook. Checked against all three released sizes' real config.json
// (3b/8b/30b): every one ships residual_multiplier 1.0 (identity), so validateGraniteDense rejects
// anything else rather than silently dropping it — the same discipline validateLlama already
// applies to scaled RoPE. Verified against a real GGUF header too (bartowski/granite-4.2-3b-GGUF
// Q2_K, HTTP-Range-fetched): architecture string "granite", metadata carries the multipliers
// directly (attention.scale/embedding_scale/logit_scale/residual_scale) and the tensor set is
// exactly llama's — the tensor schema is llamaTensorSchema, reused rather than duplicated.
func graniteDenseArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateGraniteDense(); err != nil {
		return nil, nil, err
	}
	// RoPE: the released checkpoints carry both a nested rope_parameters and a redundant
	// top-level rope_theta (transformers 4.57.1); accept either, same as granitemoehybrid.
	base, scaling, err := ropeBaseFlatOrNested(cfg, "granite")
	if err != nil {
		return nil, nil, err
	}
	if base <= 0 {
		return nil, nil, fmt.Errorf("decoder(granite): rope_theta must be >0, got %v", base)
	}
	hd := cfg.headDim()
	attnMul := cfg.AttentionMultiplier
	if attnMul == 0 {
		attnMul = math.Pow(float64(hd), -0.5) // HF default when unset
	}
	logitScale := cfg.LogitsScaling
	if logitScale == 0 {
		logitScale = 1
	}
	embedScale := cfg.EmbeddingMultiplier
	if embedScale == 0 {
		embedScale = 1
	}
	return &Architecture{
		Name:            "granite",
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
		QKVBias:         false,
		QKNorm:          false,
		AttnScale:       attnMul, // Granite attention_multiplier (not 1/√d)
		RoPELocalBase:   base,
		RoPEGlobalBase:  base,
		ropeScaling:     scaling,
		RotaryDim:       0, // full rotary
		EmbedScale:      embedScale,
		LogitScale:      logitScale,
		TiedLMHead:      false, // finalized from lm_head.weight presence at load; all three released sizes are untied
	}, &llamaTensorSchema, nil
}

// nemotronhArchitecture expresses Nemotron-H (model_type nemotron_h): a SINGLE-OP-
// per-block hybrid — each layer is exactly one of mamba / attention / mlp
// (layers_block_type), pre-norm + op + residual, NOT the mixer+FFN block every other
// family uses. The mamba layers reuse the Mamba-2 mixer (mamba2.go); the attention
// layers are NoPE GQA (no RoPE — the SSM layers carry position); the mlp layers are
// non-gated relu². Plain RMSNorm, no multipliers. Dedicated loader + forward
// (buildNemotronWeights / runLayersNemotron).
func nemotronhArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	// Released checkpoints spell the layer sequence as hybrid_override_pattern; only
	// transformers-instantiated configs carry layers_block_type. Normalize to the latter
	// before anything reads it.
	if err := cfg.normalizeNemotronBlocks(); err != nil {
		return nil, nil, err
	}
	// Nemotron-H's config has no num_hidden_layers — the layer count IS the length
	// of layers_block_type.
	if cfg.NumLayers == 0 {
		cfg.NumLayers = len(cfg.LayersBlockType)
	}
	if err := cfg.validateNemotron(); err != nil {
		return nil, nil, err
	}
	hd := cfg.headDim()
	kind := make([]uint8, cfg.NumLayers)
	hasMoE := false
	for i, t := range cfg.LayersBlockType {
		switch t {
		case "attention", "full_attention":
			// "full_attention" is transformers' own canonicalized spelling
			// (configuration_nemotron_h.py's remap_legacy_layer_types /
			// _pattern_to_list) — a checkpoint saved through a current transformers
			// version carries this, NOT "attention", even when "attention" is what
			// was originally passed in (confirmed empirically generating this
			// family's own MoE test fixture: layers_block_type came back
			// ["linear_attention","moe","full_attention",...] from an input of
			// ["mamba","moe","attention",...]). The real NVIDIA-released checkpoint
			// is unaffected today (it ships hybrid_override_pattern, parsed
			// independently by normalizeNemotronBlocks, not layers_block_type
			// directly) — but any checkpoint re-saved through transformers would hit
			// this, so it's handled as a real alias, not a fixture-only workaround.
			kind[i] = nemoAttn
		case "mlp":
			kind[i] = nemoMLP
		case "moe":
			kind[i] = nemoMoE
			hasMoE = true
		case "mamba", "linear_attention": // "linear_attention" is the same canonicalization, for Mamba-2 blocks
			kind[i] = nemoMamba
		default:
			return nil, nil, fmt.Errorf("decoder(nemotron_h): layers_block_type[%d] = %q unrecognized (want mamba/linear_attention, attention/full_attention, mlp, or moe)", i, t)
		}
	}
	// Nemotron 3 Nano's MoE FFN (model_type still "nemotron_h" — only the pattern's
	// "E" blocks distinguish it from plain Nemotron-H, which never sets hasMoE).
	// Routing verified against NVIDIA's own modeling_nemotron_h.py NemotronHTopkRouter,
	// not inferred from config-field-name similarity to DeepSeek-V3 alone: sigmoid
	// scores + e_score_correction_bias + group-limited top-k (n_group/topk_group) +
	// routed_scaling_factor on the selected weights + an UNGATED additive shared
	// expert ("hidden_states = hidden_states + self.shared_experts(residuals)", no
	// scoring_func key at all — sigmoid is unconditional for this family, not
	// config-driven the way DeepSeek's is). The expert FFN itself is non-gated relu²
	// (up_proj/down_proj only, no gate_proj — confirmed from the real safetensors
	// index), which routeExperts' selection is agnostic to but moeMLP's SwiGLU-only
	// expert evaluator cannot run — see nemotronMoE in forward_nemotron.go, a small
	// LOCAL function, not a change to the shared moeMLP path other families use.
	var moe *MoEConfig
	if hasMoE {
		normTopK := true
		if cfg.NormTopKProb != nil {
			normTopK = *cfg.NormTopKProb
		}
		moe = &MoEConfig{
			NumExperts:            cfg.NRoutedExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          normTopK,
			IntermediateDim:       cfg.MoeIntermediateSize,
			SharedIntermediateDim: cfg.MoeSharedExpertIntermediateSize, // NOT NSharedExperts*MoeIntermediateSize — verified not derivable that way for this family
			RouterSigmoid:         true,                                // unconditional for nemotron_h's router; no scoring_func key exists to check
			RoutedScale:           cfg.RoutedScalingFactor,
			SharedUngated:         true,
			NGroup:                cfg.NGroup,
			TopkGroup:             cfg.TopkGroup,
		}
	}
	return &Architecture{
		Name:            "nemotron_h",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim, // mlp-block width
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.LayerNormEpsilon,
		Act:             ActReLU2,
		NonGatedMLP:     true,
		QKVBias:         false,
		QKNorm:          false,
		AttnScale:       math.Pow(float64(hd), -0.5), // NoPE GQA, standard scale
		RotaryDim:       0,                           // no RoPE
		EmbedScale:      0,
		TiedLMHead:      false,
		MoE:             moe, // nil for plain Nemotron-H; set only when the pattern has an "moe" block
		nemotron: &nemotronParams{
			NHeads: cfg.MambaNumHeads, HeadDim: cfg.MambaHeadDim, DState: cfg.SSMStateSize,
			NGroups: cfg.NGroups, DConv: cfg.ConvKernel, blockKind: kind,
		},
	}, &nemotronTensorSchema, nil
}

// deepseekArchitecture expresses DeepSeek-V2/V3 (model_type deepseek_v2 / deepseek_v3):
// Multi-head Latent Attention over a DeepSeekMoE FFN. MLA is the new coverage axis —
// latent-KV attention: K/V compress to a shared low-rank latent (kv_lora_rank, the only
// thing cached), per-head K/V are reconstructed via kv_b_proj, and a separate
// qk_rope_head_dim key (shared across heads) carries decoupled RoPE; queries optionally
// route through a q_lora_rank bottleneck. The per-head dims are asymmetric
// (qk_head_dim = qk_nope+qk_rope ≠ v_head_dim), so it runs forward_deepseek.go rather than
// the uniform causalAttention. The MoE reuses moeMLP; routing flavor is config-driven —
// V3 scores experts with sigmoid + an e_score_correction_bias and limits selection to
// topk_group of n_group groups (noaux_tc), while V2/V2-Lite use softmax + plain greedy
// top-k. first_k_dense_replace dense-MLP prefix, ungated shared expert (the GLM path).
func deepseekArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateDeepseek(); err != nil {
		return nil, nil, err
	}
	// RoPE: the tiny synthetic carries rope_parameters {rope_theta, rope_type}; the real
	// V2-Lite/V3 carry a top-level rope_theta + a rope_scaling yarn object. Accept both.
	var base float64
	var scaling *ropeScaling
	if len(cfg.RopeParameters) > 0 {
		spec, _, err := parseRopeFlat(cfg.RopeParameters)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(%s): %w", cfg.ModelType, err)
		}
		base, scaling = spec.base, spec.scaling
	} else {
		base = cfg.RoPEGlobalBase
		sc, err := parseRopeScaling(cfg.RopeScaling)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(%s): %w", cfg.ModelType, err)
		}
		scaling = sc
	}
	// DeepSeek's YaRN attention_factor (the cos/sin mscale) is NOT the generic
	// 0.1·ln(factor)+1 — transformers computes it as the RATIO
	// get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim). When the config's
	// mscale == mscale_all_dim (V2-Lite: both 0.707) the ratio is exactly 1.0, so cos/sin
	// are unscaled. (The attention softmax scale stays qk_head_dim^-0.5 — transformers
	// 5.12 does NOT fold mscale² into it, unlike the old standalone modeling_deepseek.)
	// Override the generic yarn mscale parseRopeScaling computed.
	if scaling != nil && scaling.kind == ropeScaleYarn {
		raw := cfg.RopeScaling
		if len(cfg.RopeParameters) > 0 {
			raw = cfg.RopeParameters
		}
		var y struct {
			Factor       float64  `json:"factor"`
			Mscale       *float64 `json:"mscale"`
			MscaleAllDim *float64 `json:"mscale_all_dim"`
		}
		_ = json.Unmarshal(raw, &y)
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

	// Routing flavor. V3 (noaux_tc) scores with sigmoid + e_score_correction_bias and
	// limits to topk_group groups; V2 scores with softmax. Honor explicit scoring_func/
	// topk_method when present, else default by model_type. norm_topk_prob defaults true
	// (HF DeepseekV3Config) but V2-Lite sets it false. Kimi K2 is V3-style (sigmoid +
	// noaux_tc) — its config sets scoring_func="sigmoid", but default it to sigmoid too so
	// a variant that omits the field never silently falls to the softmax (V2) branch.
	sigmoid := cfg.ModelType == "deepseek_v3" || cfg.ModelType == "kimi_k2"
	if cfg.ScoringFunc != "" {
		sigmoid = cfg.ScoringFunc == "sigmoid"
	}
	normTopK := true
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	scale := cfg.RoutedScalingFactor // 0/1 ⇒ no-op (V2-Lite is 1.0)

	qk := cfg.QKNopeHeadDim + cfg.QKRopeHeadDim
	ropeInterleave := true // DeepseekV3Config default
	if cfg.RopeInterleave != nil {
		ropeInterleave = *cfg.RopeInterleave
	}
	return &Architecture{
		Name:            cfg.ModelType,
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumHeads, // MLA reconstructs per-head K/V; the cache is the shared latent
		HeadDim:         qk,           // q·k dot-product width (≠ v_head_dim); forward_deepseek uses mlaParams
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		QKVBias:         false,
		QKNorm:          false,
		FirstKDense:     cfg.FirstKDenseReplace,
		MoE: &MoEConfig{
			NumExperts:            cfg.NRoutedExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          normTopK,
			IntermediateDim:       cfg.MoeIntermediateSize,
			SharedIntermediateDim: cfg.NSharedExperts * cfg.MoeIntermediateSize,
			RouterSigmoid:         sigmoid,
			RoutedScale:           scale,
			SharedUngated:         true,
			NGroup:                cfg.NGroup,
			TopkGroup:             cfg.TopkGroup,
		},
		// Plain qk_head_dim^-0.5. ⚠️ Phase 3: the real V2-Lite/V3 fold YaRN's
		// mscale_all_dim² into this scale (DeepSeek's dual-mscale); wire that with the
		// real-model gate. The tiny golden uses default RoPE, so no mscale.
		AttnScale:      math.Pow(float64(qk), -0.5),
		RoPELocalBase:  base,
		RoPEGlobalBase: base,
		ropeScaling:    scaling,
		RotaryDim:      cfg.QKRopeHeadDim, // RoPE rotates only the rope-carrying dims
		EmbedScale:     0,
		TiedLMHead:     false, // finalized from lm_head.weight presence at load
		mla: &mlaParams{
			QLoRARank: cfg.QLoRARank, KVLoRARank: cfg.KVLoRARank,
			QKNopeHeadDim: cfg.QKNopeHeadDim, QKRopeHeadDim: cfg.QKRopeHeadDim,
			VHeadDim: cfg.VHeadDim, ropeInterleave: ropeInterleave,
		},
	}, &deepseekTensorSchema, nil
}

// phi3Architecture expresses Phi-3 / Phi-4 (model_type phi3): the llama skeleton —
// RMSNorm no-offset, Pre2 norm placement, SwiGLU, 1/√head_dim attention scale,
// single-base RoPE, no QK-norm, no bias, untied head — with two FUSED tensors that
// buildPhi3Weights splits at load (qkv_proj → q/k/v, gate_up_proj → gate/up) so the
// generic forward runs unchanged. Partial rotary (partial_rotary_factor) is supported via
// RotaryDim; LongRoPE (the 128k variants' su/longrope scaling) is deferred — a checkpoint
// carrying it fails loudly at parseRopeScaling. Phi-4 (14B) and Phi-3-mini-4k use full
// rotary + no scaling. The tensor schema is the phi3TensorSchema marker.
func phi3Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	hd := cfg.headDim()
	// RoPE: transformers ≥5.11 saves the flat rope_parameters {rope_theta,
	// partial_rotary_factor, rope_type}; the released Phi-4/Phi-3 configs use a
	// top-level rope_theta (+ rope_scaling for the 128k longrope variants). Accept both.
	var base float64
	var scaling *ropeScaling
	rotaryDim := cfg.rotaryDim()
	if len(cfg.RopeParameters) > 0 {
		spec, partial, err := parseRopeFlat(cfg.RopeParameters)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(phi3): %w", err)
		}
		base, scaling = spec.base, spec.scaling
		if partial > 0 && partial < 1 {
			rotaryDim = int(partial * float64(hd))
		}
	} else {
		base = cfg.RoPEGlobalBase
		sc, err := parseRopeScaling(cfg.RopeScaling) // nil for Phi-4/4k; longrope errors (deferred)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(phi3): %w", err)
		}
		scaling = sc
	}
	if base <= 0 {
		return nil, nil, fmt.Errorf("decoder(phi3): rope_theta must be >0, got %v", base)
	}
	// Sliding-window attention. Phi-3-mini-4k's config.json declares sliding_window: 2047
	// (Phi-4 omits it, and the released GGUF conversions drop the key entirely ⇒ 0 = full
	// attention, which is what that file honestly declares). All layers are local when a
	// window is set, mirroring mistralArchitecture. layerIsGlobal MUST be set alongside
	// SlidingWindow: isGlobalLayer defaults to TRUE (global) when the hook is nil, which
	// would leave the window silently inert.
	var layerIsGlobal func(int) bool
	if cfg.SlidingWindow > 0 {
		layerIsGlobal = func(int) bool { return false }
	}
	return &Architecture{
		Name:            "phi3",
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
		QKVBias:         false,
		QKNorm:          false,
		AttnScale:       math.Pow(float64(hd), -0.5),
		RoPELocalBase:   base,
		RoPEGlobalBase:  base,
		RotaryDim:       rotaryDim,         // partial_rotary_factor · head_dim; 0 = full
		SlidingWindow:   cfg.SlidingWindow, // 0 ⇒ full attention (Phi-4, and the GGUFs)
		layerIsGlobal:   layerIsGlobal,     // all-local when windowed
		ropeScaling:     scaling,
		EmbedScale:      0,
		TiedLMHead:      false, // finalized from lm_head.weight presence at load
	}, &phi3TensorSchema, nil
}

// llama4Architecture expresses the Llama 4 (Scout/Maverick) TEXT decoder (model_type
// llama4_text): the iRoPE stack. Most layers are RoPE GQA with a parameter-free L2
// (RMS-over-head-dim) QK-norm; every no_rope_layers==0 layer is NoPE (no RoPE) with
// attention-temperature tuning (q scaled by log1p(floor((pos+1)/floor_scale))·attn_scale+1
// for length generalization). The FFN interleaves dense (intermediate_size_mlp) and MoE
// (moe_layers) blocks; the MoE is top-1 SIGMOID routing (no group, no norm, no scale) plus
// an always-on UNGATED shared expert, both at intermediate_size. Separate q/k/v/o (no
// fusion, no bias). Interleaved (complex) RoPE over the full head_dim. The vision tower
// (early-fusion multimodal) and any MTP heads are dropped — text decoder only. Dedicated
// loader (buildLlama4Weights) + forward (runLayersLlama4).
// gptOssArchitecture expresses the gpt-oss sparse-MoE family (model_type gpt_oss,
// GGUF arch gpt-oss). It marks the family with a gptoss escape hatch — its forward
// (forward_gptoss.go) adds a learned per-head attention SINK to each softmax and
// uses a clamped interleaved-SwiGLU expert with per-expert biases — while the layer
// skeleton stays the shared pre-norm path. Attention alternates sliding (even
// layers, window sliding_window) and full (odd), q/k/v/o all carry biases, no
// QK-norm, YaRN RoPE (one table for both attention types). The router carries a
// per-expert logit bias (LayerWeights.RouterBias). MXFP4 experts are CPU-only; CUDA
// and Metal decline via FeatAttnSink (features.go).
func gptOssArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateGptOss(); err != nil {
		return nil, nil, err
	}
	scaling, err := parseRopeScaling(cfg.RopeScaling)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(gpt-oss): %w", err)
	}
	hd := cfg.headDim()
	nExperts := cfg.NumLocalExperts
	if nExperts == 0 {
		nExperts = cfg.NumExperts
	}
	expInter := cfg.MoeIntermediateSize
	if expInter == 0 {
		expInter = cfg.IntermediateDim
	}
	return &Architecture{
		Name:             "gpt-oss",
		HiddenDim:        cfg.HiddenDim,
		NumLayers:        cfg.NumLayers,
		NumHeads:         cfg.NumHeads,
		NumKVHeads:       cfg.NumKVHeads,
		HeadDim:          hd,
		IntermediateDim:  expInter, // gpt-oss has no dense layers; experts use this width
		VocabSize:        cfg.VocabSize,
		Norm:             NormRMS,
		RMSAddOne:        false,
		NormEps:          cfg.RMSNormEps,
		NormPlacement:    NormPre2,
		Act:              ActSiLU, // the gpt-oss forward uses its own clamped activation; this is only for the matrix/logs
		QKVBias:          true,    // q/k/v projection biases
		OutBias:          true,    // attn_output bias
		QKNorm:           false,
		AttnScale:        math.Pow(float64(hd), -0.5),
		SlidingWindow:    cfg.SlidingWindow,
		layerIsGlobal:    func(i int) bool { return i%2 == 1 }, // even = sliding, odd = full
		RoPEGlobalBase:   cfg.RoPEGlobalBase,
		RoPELocalBase:    cfg.RoPEGlobalBase, // one RoPE for both attention types
		ropeScaling:      scaling,
		ropeScalingLocal: scaling, // same YaRN table on sliding + full layers
		RotaryDim:        cfg.rotaryDim(),
		MoE: &MoEConfig{
			NumExperts:      nExperts,
			TopK:            cfg.NumExpertsPerTok,
			NormTopKProb:    true, // softmax over the top-k logits == softmax-all + renormalize
			IntermediateDim: expInter,
		},
		gptoss:     &gptOssParams{SwigluAlpha: 1.702, SwigluLimit: 7.0}, // gpt-oss constants (not in metadata)
		EmbedScale: 0,
		TiedLMHead: false, // finalized from output.weight presence at load
	}, nil, nil
}

func llama4Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if cfg.NumLocalExperts <= 0 || cfg.NumExpertsPerTok <= 0 {
		return nil, nil, fmt.Errorf("decoder(llama4): bad MoE (num_local_experts=%d num_experts_per_tok=%d)", cfg.NumLocalExperts, cfg.NumExpertsPerTok)
	}
	// RoPE: safetensors carries the flat rope_parameters; the GGUF path sets a
	// top-level rope_theta (+ optional rope_scaling). Accept both.
	var base float64
	var ropeSc *ropeScaling
	if len(cfg.RopeParameters) > 0 {
		spec, _, err := parseRopeFlat(cfg.RopeParameters)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(llama4): %w", err)
		}
		base, ropeSc = spec.base, spec.scaling
	} else {
		base = cfg.RoPEGlobalBase
		sc, err := parseRopeScaling(cfg.RopeScaling)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(llama4): %w", err)
		}
		ropeSc = sc
	}
	hd := cfg.HeadDim
	if hd == 0 {
		if cfg.NumHeads <= 0 { // avoid a divide-by-zero on a hostile config (M16)
			return nil, nil, fmt.Errorf("decoder(llama4): num_attention_heads must be > 0 when head_dim is unset, got %d", cfg.NumHeads)
		}
		hd = cfg.HiddenDim / cfg.NumHeads
	}
	denseInter := cfg.IntermediateSizeMLP
	if denseInter == 0 {
		denseInter = cfg.IntermediateDim
	}
	// Per-layer kind. no_rope_layers[i]==1 ⇒ RoPE (the field's sense is "uses rope");
	// absent ⇒ all RoPE. moe_layers lists the MoE indices; absent ⇒ derive from
	// interleave_moe_layer_step (layers interleave_step-1, 2·step-1, … are MoE), else all MoE.
	useRope := make([]bool, cfg.NumLayers)
	for i := range useRope {
		useRope[i] = true
		if i < len(cfg.NoRopeLayers) {
			useRope[i] = cfg.NoRopeLayers[i] != 0
		}
	}
	isMoE := make([]bool, cfg.NumLayers)
	switch {
	case len(cfg.MoeLayers) > 0:
		for _, l := range cfg.MoeLayers {
			if l >= 0 && l < cfg.NumLayers {
				isMoE[l] = true
			}
		}
	case cfg.InterleaveMoeLayerStep > 0:
		for l := cfg.InterleaveMoeLayerStep - 1; l < cfg.NumLayers; l += cfg.InterleaveMoeLayerStep {
			isMoE[l] = true
		}
	default:
		for l := range isMoE {
			isMoE[l] = true
		}
	}
	floor := cfg.FloorScale
	if floor == 0 {
		floor = 8192
	}
	attnScale := cfg.AttnScaleL4
	if attnScale == 0 {
		attnScale = 0.1
	}
	return &Architecture{
		Name:            "llama4_text",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: denseInter, // dense layers (intermediate_size_mlp); experts use MoE.IntermediateDim
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		QKVBias:         false,
		QKNorm:          false, // Llama 4's QK-norm is parameter-free L2, applied in runLayersLlama4 (not the weighted path)
		MoE: &MoEConfig{
			NumExperts:            cfg.NumLocalExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          false,               // top-1 by raw logit; weight = sigmoid(logit), no renorm
			IntermediateDim:       cfg.IntermediateDim, // routed expert width (intermediate_size)
			SharedIntermediateDim: cfg.IntermediateDim, // shared expert uses the same width
			RouterSigmoid:         true,
			RoutedScale:           1,
			SharedUngated:         true,
		},
		AttnScale:      math.Pow(float64(hd), -0.5),
		RoPELocalBase:  base,
		RoPEGlobalBase: base,
		ropeScaling:    ropeSc,
		RotaryDim:      0, // full head_dim
		EmbedScale:     0,
		TiedLMHead:     false, // finalized from lm_head.weight presence at load
		llama4: &llama4Params{
			// M-05: attention_chunk_size was read into Config and dropped here, so the RoPE
			// layers attended full-causal past it.
			chunkSize: cfg.AttentionChunkSize,
			useRope:   useRope, isMoE: isMoE, useQKNorm: cfg.UseQKNorm,
			attnTemp: cfg.AttnTemperatureTuning, floorScale: floor, attnScale: attnScale,
		},
	}, &llama4TensorSchema, nil
}

// lagunaArchitecture expresses Laguna (poolside; model_type "laguna") across all
// three released generations — Laguna-XS-2.1, Laguna-XS.2, Laguna-M.1 — from ONE
// adapter. The vendor's modeling_laguna.py is byte-identical between generations
// (only import paths differ), so every generational difference is config, and this
// reads them all rather than branching on a version.
//
// The vendor's own summary is that Laguna attention "is identical to Qwen2MoE
// attention except": no QKV bias, an explicit head_dim, per-layer sliding window,
// and output gating before o_proj. That last one is the only genuinely new
// primitive; everything else composes shipped parts:
//
//   - Router: sigmoid scoring + e_score_correction_bias steering the SELECTION only
//   - norm_topk_prob + routed scaling — goinfer's DeepSeek/GLM MoE path exactly.
//   - Shared expert: added with NO outer sigmoid gate ⇒ SharedUngated. (It IS a
//     gated SwiGLU internally; SharedUngated refers to Qwen2-MoE's extra
//     sigmoid(shared_gate·h) scaling of the shared branch, which Laguna lacks.)
//   - Dense prefix: mlp_only_layers is a contiguous prefix ⇒ FirstKDense.
//   - RoPE keyed by layer type: full_attention is YaRN at theta 500000 with partial
//     rotary 0.5; sliding_attention is plain at theta 10000, full rotary. That is
//     RoPEGlobalBase/RoPELocalBase + ropeScaling/ropeScalingLocal (Mellum already
//     splits base and scaling this way) plus RotaryDimLocal for the width.
//   - M.1 drops layer_types, sliding_window, and num_attention_heads_per_layer
//     entirely: it is all-full-attention with uniform heads, so the local/sliding
//     half of each of those pairs simply never engages.
//
// Attention sinks are NOT enabled in any released checkpoint
// (swa_attention_sink_enabled absent), so that branch of the vendor code has no
// counterpart here. See docs/task-laguna.md for the Phase 0 config verification.
func lagunaArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateLaguna(); err != nil {
		return nil, nil, err
	}
	gateOn, gatePerHead, err := cfg.lagunaGatePerHead()
	if err != nil {
		return nil, nil, err
	}
	firstKDense, err := cfg.lagunaFirstKDense()
	if err != nil {
		return nil, nil, err
	}
	full, sliding, err := parseRopeParameters(cfg.RopeParameters)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(laguna): %w", err)
	}
	if full == nil {
		return nil, nil, fmt.Errorf("decoder(laguna): rope_parameters.full_attention is required")
	}
	hd := cfg.headDim()

	// partial_rotary_factor precedence. It appears BOTH top-level and inside each
	// rope_parameters[layer_type], and the vendor documents that HF's
	// standardize_rope_params "unconditionally overwrites
	// rope_parameters['partial_rotary_factor'] with self.partial_rotary_factor" —
	// working around it by aligning the top-level field to the SWA value on a cloned
	// config. The per-layer-type value is therefore the authoritative one; the
	// top-level field is a fallback for checkpoints that carry only it.
	partialOf := func(spec *ropeLayerSpec) float64 {
		if spec != nil && spec.partial > 0 {
			return spec.partial
		}
		return cfg.PartialRotaryFactor
	}
	rotaryOf := func(p float64) int {
		if p > 0 && p < 1 {
			return int(p * float64(hd))
		}
		return 0 // 0 ⇒ full HeadDim
	}

	localBase, localScaling := full.base, full.scaling // M.1: no sliding layers; keep the tables equal
	rotaryDimLocal := 0
	if sliding != nil {
		localBase, localScaling = sliding.base, sliding.scaling
		if rl := rotaryOf(partialOf(sliding)); rl != rotaryOf(partialOf(full)) {
			rotaryDimLocal = rl
			if rotaryDimLocal == 0 {
				rotaryDimLocal = hd // sliding rotates FULL width while global is partial — must be explicit, since 0 means "same as RotaryDim"
			}
		}
	}
	normTopK := true // HF LagunaConfig default
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	var lp *lagunaParams
	if gateOn || len(cfg.NumAttentionHeadsPerLayer) > 0 {
		lp = &lagunaParams{HeadsPerLayer: cfg.NumAttentionHeadsPerLayer, GatePerHead: gatePerHead}
	}
	return &Architecture{
		Name:            "laguna",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim, // dense width, used by the mlp_only_layers prefix
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		QKVBias:         false, // "no QKV bias" is in the vendor's own list of departures from Qwen2MoE
		// QK-norm is UNCONDITIONAL in all three released modules — q_norm/k_norm are
		// constructed with no config flag guarding them, and the real XS.2 checkpoint
		// ships model.layers.N.self_attn.{q,k}_norm.weight of shape [head_dim] on every
		// layer. The vendor's prose ("identical to Qwen2MoE attention except …") omits
		// it, so the config is silent and only the checkpoint says so.
		QKNorm:      true,
		FirstKDense: firstKDense,
		MoE: &MoEConfig{
			NumExperts:            cfg.NumExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          normTopK,
			IntermediateDim:       cfg.MoeIntermediateSize,
			SharedIntermediateDim: cfg.SharedExpertIntermediateSize,
			RouterSigmoid:         true,
			RoutedScale:           cfg.MoeRoutedScalingFactor,
			SharedUngated:         true,
		},
		AttnScale:        math.Pow(float64(hd), -0.5),
		SlidingWindow:    cfg.SlidingWindow,
		layerIsGlobal:    cfg.IsGlobalLayer, // from layer_types; absent ⇒ all-global (M.1)
		RoPEGlobalBase:   full.base,
		RoPELocalBase:    localBase,
		ropeScaling:      full.scaling,
		ropeScalingLocal: localScaling,
		RotaryDim:        rotaryOf(partialOf(full)),
		RotaryDimLocal:   rotaryDimLocal,
		EmbedScale:       0,
		TiedLMHead:       false, // finalized from lm_head.weight presence at load
		laguna:           lp,
	}, &lagunaTensorSchema, nil
}

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

// qwen35DenseArchitecture expresses Qwen3.8 (model_type qwen3_5): the SAME Gated-DeltaNet /
// softmax 3:1 hybrid as qwen3_5_moe and qwen3_next, with a plain SwiGLU where they have a
// router. It is a dense sibling, not a new family — the DeltaNet math, the double-width gated
// q_proj, the per-head QK-norm, the partial RoPE and the hybrid cache are all shared.
//
// VERIFIED AGAINST THE RELEASED CHECKPOINT AND transformers 5.12's models/qwen3_5, not
// inherited by resemblance (the house standard the qwen3_next adapter set):
//
//   - head_dim 256 with hidden 5120 and 24 heads, so nH·hd = 6144 ≠ hidden. Never derive
//     head_dim from hidden for this family.
//   - RMSAddOne TRUE: Qwen3_5RMSNorm is `output * (1.0 + weight)` with weight zero-init,
//     the same Gemma-style form its MoE siblings use.
//   - The DeltaNet projections ship as FOUR separate tensors (in_proj_qkv / in_proj_z /
//     in_proj_a / in_proj_b), i.e. qwen3_5_moe's packing — NOT qwen3_next's fused
//     in_proj_qkvz/in_proj_ba. So FusedDeltaNetProj stays false and the existing loader path
//     applies unchanged. Read from the safetensors index, not assumed.
//   - The attention output gate is torch.sigmoid(gate) despite the config's
//     output_gate_type:"swish" — the field is not consulted by the modeling code.
//   - m-RoPE: the text config carries mrope_section [11,11,10] with mrope_interleaved true,
//     a new variant. For TEXT it reduces EXACTLY to standard partial RoPE: position_ids
//     arrive 2-D and are expand()ed to three identical components, so
//     apply_interleaved_mrope overwrites interleaved indices with identical values — a no-op.
//     MRopeSection is therefore deliberately NOT set; an image path would need it (non-goal).
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
	return &Architecture{
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
		MoE:              nil, // the one structural difference; runLayersQwen35 branches on it
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
		},
	}, &qwen35DenseTensorSchema, nil
}
