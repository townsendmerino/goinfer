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
// is a new entry here plus its tensor schema — the
// forward pass itself doesn't change.
var registry = map[string]archAdapter{
	"gemma":               gemmaArchitecture,  // Gemma 1 and CodeGemma: pre-norm, √hidden embed scale, (1+w) RMSNorm, GeGLU, tied head (gemma12.go)
	"gemma2":              gemma2Architecture, // Gemma 2: gemma3's sandwich norms without QK-norm, sliding/full alternating, attention + final softcaps (gemma12.go)
	"gemma3":              gemma3Architecture,
	"gemma3_text":         gemma3Architecture,     // the 270M/1B text checkpoints
	"gemma4":              gemma4Architecture,     // Gemma 4 top-level model_type for the "gemma4" family wrapper (E2B/E4B/26B-A4B/31B — real vision+audio towers, not built here)
	"gemma4_text":         gemma4Architecture,     // that family's text_config.model_type
	"gemma4_unified_text": gemma4Architecture,     // the separate "gemma4_unified" family's text_config.model_type (12B-it: encoder-free multimodal wrapper, no PLE/MoE in its text decoder; K=V globals, model.language_model.* prefix)
	"qwen3":               qwen3Architecture,      // Qwen3 dense (0.6B/1.7B/4B/8B/…)
	"qwen2":               qwen2Architecture,      // Qwen2/Qwen2.5 dense (llama + q/k/v bias)
	"qwen2_5_vl":          qwen2_5_vlArchitecture, // Qwen2.5-VL text decoder (qwen2 + m-RoPE; nested rope_parameters)
	"voxtral":             voxtralArchitecture,    // Voxtral Mini's text decoder: exactly Llama (nested text_config, tensors under language_model.*, head_dim 128 != hidden/heads); the audio tower and projector are aikit/audio.VoxtralAudio
	"qwen3_asr":           qwen3_asrArchitecture,  // Qwen3-ASR's text decoder: exactly Qwen3 (nested thinker_config.text_config, tensors under thinker.*); the audio encoder is aikit/audio
	"qwen3_vl":            qwen3_vlArchitecture,   // Qwen3-VL TEXT decoder only (qwen3 + interleaved m-RoPE; nested text_config/rope_parameters)
	"qwen2_moe":           qwen2MoeArchitecture,   // Qwen-MoE/Qwen2-MoE (qwen2 + sparse MoE + shared expert)
	"qwen3_moe":           qwen3MoeArchitecture,   // Qwen3-30B-A3B / Qwen3-Coder-30B-A3B: qwen3's attention (QK-norm, no bias) + a sparse MoE on every layer, NO shared expert
	"qwen3_vl_moe":        qwen3VLMoeArchitecture, // Qwen3-VL-30B-A3B: qwen3_moe + qwen3_vl's interleaved m-RoPE; DeepStack images through the dense Qwen3-VL's path
	"llama":               llamaArchitecture,      // Llama-2/3 dense (single-base RoPE, no QK-norm)
	"smollm3":             smollm3Architecture,    // SmolLM3-3B: llama dense + per-layer NoPE (no_rope_layers) on every 4th layer, tied embeddings
	"olmo3":               olmo3Architecture,      // Olmo 3 (7B/32B): NormPostOnly (no pre-norm at all) + whole-vector QK-norm + sliding/full 3:1 + YaRN
	"olmo_hybrid":         olmoHybridArchitecture, // Olmo Hybrid (7B): qwen3_5's Gated DeltaNet (3:1) + olmo3's own full-attention shape, mixed NormPlacement per layer kind, NoPE (rope_theta null)
	// InternLM3 is llama-shaped to the tensor name: self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj,
	// input_layernorm/post_attention_layernorm, embed_tokens/lm_head, no biases, no QK-norm. Its only
	// config departure is rope_scaling type "dynamic", which is exactly identity within the trained
	// window (see parseRopeScaling). So it is a registry ALIAS, not an adapter — the honest expression
	// of "this is a llama". InternLM2 is NOT an alias: it renames every tensor and fuses qkv.
	"internlm3":        llamaArchitecture,         // InternLM3 (llama-shaped; dynamic-NTK rope is in-window identity)
	"internlm2":        internlm2Architecture,     // InternLM2 (llama math; renamed tensors + GROUPED fused wqkv, split at load)
	"mistral":          mistralArchitecture,       // Llama + all-layer sliding-window attention
	"mistral3":         ministral3Architecture,    // Ministral 3 (3B/8B/14B): Mistral GQA + YaRN + Llama4-style attention-temperature tuning on every layer, text_config extracted from the Mistral3ForConditionalGeneration VL wrapper
	"ministral3":       ministral3Architecture,    // the nested text_config's own model_type (a plain, unwrapped Ministral3Config save carries this directly, not "mistral3")
	"gpt2":             gpt2Architecture,          // GPT-2: LayerNorm, learned pos, non-gated GELU MLP, fused QKV
	"cohere":           cohereArchitecture,        // Cohere / Command-R (+ Aya): bias-free LayerNorm + parallel attn/MLP block + logit_scale + GPT-J interleaved RoPE
	"cohere2":          cohere2Architecture,       // Cohere2 / Command-R7B (+ Command-A): cohere1 stack + interleaved sliding-window + NoPE on the global layers, no QK-norm
	"mixtral":          mixtralArchitecture,       // Llama + sparse MoE FFN (router + top-k experts)
	"mellum":           mellumArchitecture,        // JetBrains Mellum2: MoE + sliding/full interleave + YaRN
	"qwen3_5_moe":      qwen35Architecture,        // Qwen3.5/3.6-MoE: Gated DeltaNet (linear) + softmax hybrid + MoE
	"qwen3_5_moe_text": qwen35Architecture,        // the text-only checkpoint's model_type
	"qwen3_5":          qwen35DenseArchitecture,   // Qwen3.8 dense (Gated DeltaNet + softmax hybrid, plain SwiGLU FFN)
	"qwen3_5_text":     qwen35DenseArchitecture,   // the text_config's own model_type, for text-only checkpoints
	"qwen3_next":       qwen3NextArchitecture,     // Qwen3-Next: same DeltaNet/softmax/MoE hybrid shape as qwen3_5_moe, but layer_types is COMPUTED (full_attention_interval) and partial_rotary_factor is a top-level field, not nested
	"glm4_moe":         glm4moeArchitecture,       // GLM-4.5/4.6: DeepSeek-style MoE (sigmoid routing + bias) + dense prefix + QK-norm + partial RoPE
	"laguna":           lagunaArchitecture,        // Laguna (poolside) XS-2.1 / XS.2 / M.1: sigmoid-routed MoE + shared expert + softplus attention output gating + per-layer query heads
	"spark2_5":         spark25Architecture,       // Spark-X2.5 (XHToken): fused QKV, sigmoid head-wise attention output gate, 1:3 full/sliding (512) with layer-dependent partial RoPE, gated exact-GELU MLP, plain sequential (non-parallel) block
	"granitemoehybrid": graniteArchitecture,       // Granite-4.0-H: Mamba-2 + attention hybrid + MoE-on-every-layer + Granite multipliers
	"granite":          graniteDenseArchitecture,  // Granite 4.2 (3B/8B/30B) dense: llama skeleton + Granite's four scalar multipliers
	"lfm2":             lfm2Architecture,          // LFM2 / LFM2.5: gated short-conv + GQA hybrid (layer_types), tied head, per-head RMSNorm QK-norm
	"lfm2_vl":          lfm2Architecture,          // LFM2-VL / LFM2.5-VL (S10): the lfm2 decoder of an Lfm2VlForConditionalGeneration, text_config extracted; its SigLIP2 tower is multimodal/aikit's
	"nemotron_h":       nemotronhArchitecture,     // Nemotron-H: single-op-per-block hybrid (mamba | NoPE-attention | relu² MLP)
	"deepseek_v2":      deepseekArchitecture,      // DeepSeek-V2 (MLA + DeepSeekMoE; softmax routing, V2-Lite has no q-LoRA)
	"deepseek_v3":      deepseekArchitecture,      // DeepSeek-V3 (MLA + DeepSeekMoE; sigmoid + e_score_correction_bias group-limited routing)
	"kimi_k2":          deepseekArchitecture,      // Kimi K2/K2.x (architectures=DeepseekV3ForCausalLM): MLA + DeepSeekMoE, "basically V3" — 64 heads / 384 experts, config scalars only
	"bailing_hybrid":   bailingHybridArchitecture, // Ling 3.0 (inclusionAI): MLA (DeepSeek-shaped, reused) alternating with KDA (Kimi Delta Attention, per-channel-decay delta rule) every layer_group_size-th layer, over a DeepSeekMoE FFN
	"phi3":             phi3Architecture,          // Phi-3 / Phi-4 dense: llama skeleton + fused qkv_proj / gate_up_proj (split at load) + partial rotary
	"glm_ocr":          glmOcrArchitecture,        // GLM-OCR (zai-org, 0.9B document OCR) VL wrapper: the text decoder via the flattened text_config (top-level model_type wins)
	"glm_ocr_text":     glmOcrArchitecture,        // GLM-OCR text decoder: Sandwich4 RMSNorm (names reversed vs Gemma's), fused gate_up_proj, explicit head_dim 128, pairwise m-RoPE [16,24,24], untied head, MTP layer skipped
	"llama4_text":      llama4Architecture,        // Llama 4 (Scout/Maverick) text decoder: iRoPE (RoPE/NoPE interleave) + L2 QK-norm + attn-temp + dense/MoE interleave (top-1 sigmoid + shared)
	"gpt_oss":          gptOssArchitecture,        // gpt-oss (20b/120b): sparse MoE + per-head attention sinks + clamped interleaved-SwiGLU + alternating sliding/full + YaRN (MXFP4 experts)
}

// resolveArchitecture picks the adapter for cfg.ModelType and builds the
// descriptor + schema. An unknown model_type is a loud error, not a silent
// wrong load.
func resolveArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	adapter, ok := registry[cfg.ModelType]
	if !ok {
		return nil, nil, fmt.Errorf("decoder: unsupported model_type %q (have: %s)", cfg.ModelType, knownModelTypes())
	}
	// Bound the config before the adapter runs: several adapters allocate NumLayers-sized slices with only a `> 0` check and loadConfig has no
	// bound, so a hostile config (num_hidden_layers: 68719476736) is a fatal out-of-memory that no recover() catches, not the typed error
	// LoadSerializedWeights promises. Here, because this is the single point every path reaches (.giw, safetensors, GGUF and whatever is added
	// next); at the callers it would be the "one predicate, N consumers" shape. The GGUF paths bound some of these already; re-checking costs a
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

// validateResolved catches descriptor fields an adapter left at their zero value when zero is not a legal setting. Every adapter builds an
// Architecture by hand from a struct literal, so a field simply omitted is a compile-clean, load-clean, silently wrong model. Checked here
// rather than in each adapter to cover the families nobody has written yet.
//
//   - AttnScale 0 makes every q·k score 0, so softmax returns a UNIFORM average over the context. Invisible at one token (softmax of a
//     single element is 1.0 whatever the scale) and in any greedy smoke test that only reads argmax. An arch that genuinely wants no
//     attention scaling sets 1.0.
//   - NormEps 0 divides by rsqrt(variance) with no floor: not merely imprecise, it rescales the norm output on a small variance.
func (a *Architecture) validateResolved() error {
	switch {
	// !(x > 0) rather than x <= 0: NaN fails EVERY comparison, so `a.AttnScale <= 0` is false for NaN and would wave it through. gemma3 with a
	// negative query_pre_attn_scalar produces exactly that, and +Inf arrives from num_attention_heads: 0 → Pow(0, -0.5). Both give a forward
	// that runs and returns garbage rather than one that refuses.
	case !(a.AttnScale > 0) || math.IsInf(a.AttnScale, 0):
		return fmt.Errorf("decoder(%s): AttnScale=%v must be finite and >0 (adapter omitted it, or a config value made it NaN/Inf; 1/sqrt(head_dim) is the usual value, 1.0 means deliberately unscaled)", a.Name, a.AttnScale)
	case a.Norm == NormRMS && !(a.NormEps > 0):
		return fmt.Errorf("decoder(%s): NormEps=%v must be >0 (adapter omitted it, or read the wrong config key)", a.Name, a.NormEps)
	}

	// Position information must come from somewhere. finalizeRoPE treats RoPEGlobalBase <= 0 as "no tables" and applyRoPE is a silent no-op on
	// an empty table, so an adapter that never reads rope_theta loads clean and generates fluent, position-blind text (and drops YaRN with it).
	// transformers >= 5.10 nests rope_theta under rope_parameters, so an adapter that reads only the flat key fails silently.
	//
	// The legitimate ways to have no global RoPE table are named explicitly rather than inferred, so a new family that forgot cannot look like
	// one of them: GPT-2 has learned positions, Nemotron-H encodes NoPE layers as base 0, MLA carries its own decoupled rope dims, and Olmo
	// Hybrid's released checkpoint has none at all (NoPositionEncoding; its rope_parameters is {"rope_theta": null}).
	if !a.LearnedPosEmbed && !a.NoPositionEncoding && a.nemotron == nil && a.mla == nil && len(a.ropeInvFreqGlobal) == 0 {
		return fmt.Errorf("decoder(%s): no position information — RoPEGlobalBase=%v yields no "+
			"inv-freq table, and the arch is not learned-position, Nemotron-H or MLA. The adapter "+
			"most likely did not read rope_theta (transformers >=5.10 nests it under "+
			"rope_parameters); a model loaded this way generates fluent but position-blind text",
			a.Name, a.RoPEGlobalBase)
	}

	// Zero dims. Every one of these is a divisor, a slice length or both somewhere in the forward; num_key_value_heads: 0 is the sharpest:
	// `group := nH/nKV` is an integer divide by zero, a panic in the decode goroutine rather than a load error.
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
		// num_local_experts and n_routed_experts are the same kind of value as num_experts under other families' spellings. The .giw reader's own
		// caps already stop a hostile config becoming an OOM at allocation time, so this is defense in depth: refuse here, before any allocation.
		{"num_local_experts", cfg.NumLocalExperts, maxGGUFExperts},
		{"n_routed_experts", cfg.NRoutedExperts, maxGGUFExperts},
		// The feed-forward widths: newDecodeScratch allocates IntermediateDim-sized vectors for every family, MoE ones included.
		{"intermediate_size", cfg.IntermediateDim, maxGGUFIntermediate},
		{"moe_intermediate_size", cfg.MoeIntermediateSize, maxGGUFIntermediate},
		{"shared_expert_intermediate_size", cfg.SharedExpertIntermediateSize, maxGGUFIntermediate},
		{"moe_shared_expert_intermediate_size", cfg.MoeSharedExpertIntermediateSize, maxGGUFIntermediate},
		{"intermediate_size_mlp", cfg.IntermediateSizeMLP, maxGGUFIntermediate},
		{"shared_intermediate_size", cfg.SharedIntermediateSize, maxGGUFIntermediate},
	} {
		if d.v > d.max {
			return fmt.Errorf("decoder: %s %d exceeds the %d ceiling — refusing before it is "+
				"allocated from (a hostile or corrupt config, not a real model)", d.name, d.v, d.max)
		}
	}
	for i, v := range cfg.FFNPerLayer {
		if v > maxGGUFIntermediate {
			return fmt.Errorf("decoder: per-layer feed_forward_length[%d] %d exceeds the %d ceiling — refusing before it is "+
				"allocated from (a hostile or corrupt config, not a real model)", i, v, maxGGUFIntermediate)
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

// backfillFlatRope fills the flat rope_theta / rope_scaling fields from transformers >=5.10's rope_parameters object, for the single-base
// architectures that read only the flat fields (llama, mistral, qwen3 and others).
//
// transformers moved RoPE config out of top-level rope_theta/rope_scaling into rope_parameters ({"rope_theta": 1e4, "rope_type": "default"},
// with linear/yarn/llama3 scaling inside the same object), so an arch that reads only the flat fields rejects any checkpoint saved by a
// current transformers with "rope_theta must be >0". One helper rather than a copy per architecture.
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
