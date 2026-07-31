package decoder

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
)

// Config captures the Gemma 3 architecture constants the forward pass
// depends on. Field tags follow the HF config.json schema so a checkpoint's
// config drives the loader rather than hardcoded constants — the same
// config-driven approach encoder.Config uses, which is what lets one code
// path serve both 270M and 1B (and beyond) unchanged.
//
// Values that vary per layer (the 5:1 local:global attention pattern) are
// derived from SlidingWindowPattern at load time, not stored per layer.
type Config struct {
	ModelType            string  `json:"model_type"`              // "gemma3_text" — selects the arch adapter
	VocabSize            int     `json:"vocab_size"`              // 262144
	HiddenDim            int     `json:"hidden_size"`             // 640 (270M)
	NumLayers            int     `json:"num_hidden_layers"`       // 18 (270M)
	NumHeads             int     `json:"num_attention_heads"`     // 4 (270M)
	NumKVHeads           int     `json:"num_key_value_heads"`     // 1 (270M) — GQA
	HeadDim              int     `json:"head_dim"`                // 256 (270M); note heads*headDim != hidden
	IntermediateDim      int     `json:"intermediate_size"`       // 2048 (270M) — GeGLU
	MaxPositions         int     `json:"max_position_embeddings"` // 32768
	RMSNormEps           float64 `json:"rms_norm_eps"`
	RoPELocalBase        float64 `json:"rope_local_base_freq"`   // 10000
	RoPEGlobalBase       float64 `json:"rope_theta"`             // 1000000
	SlidingWindow        int     `json:"sliding_window"`         // 512 (270M)
	SlidingWindowPattern int     `json:"sliding_window_pattern"` // 6 → 5 local : 1 global
	QueryPreAttnScalar   float64 `json:"query_pre_attn_scalar"`  // 256 (270M)
	UseQKNorm            bool    `json:"use_qk_norm"`            // true in Gemma 3
	HiddenActivation     string  `json:"hidden_activation"`      // Gemma: "gelu_pytorch_tanh"
	HiddenAct            string  `json:"hidden_act"`             // Llama/Qwen: "silu" (different JSON key)
	AttentionBias        bool    `json:"attention_bias"`         // Qwen2/GPT-2 add q/k/v/o bias; Llama-3/Qwen3 don't
	UseSlidingWindow     bool    `json:"use_sliding_window"`     // Qwen2: gate for sliding-window attention (usually false)

	// Mixture-of-experts. NormTopKProb is a *bool so an absent field can default
	// to true (HF's MixtralConfig default). Mixtral names the expert count
	// num_local_experts; Mellum names it num_experts and gives the experts their
	// own (narrower) FFN width in moe_intermediate_size.
	NumLocalExperts     int   `json:"num_local_experts"`
	NumExperts          int   `json:"num_experts"`
	NumExpertsPerTok    int   `json:"num_experts_per_tok"`
	NormTopKProb        *bool `json:"norm_topk_prob"`
	MoeIntermediateSize int   `json:"moe_intermediate_size"`
	// Gemma 4 26B-A4B MoE (model_type gemma4, text_config.enable_moe_block=true). It
	// spells the per-token expert count top_k_experts (not num_experts_per_tok), and
	// gates the whole parallel dense+MoE FFN sub-block on enable_moe_block — the dense
	// E2B/E4B/12B variants leave both zero/false. See docs/task-gemma4-moe.md §A4.
	EnableMoeBlock bool `json:"enable_moe_block"`
	TopKExperts    int  `json:"top_k_experts"`
	// SharedExpertIntermediateSize is the always-on shared expert's FFN width
	// (Qwen-MoE/Qwen2-MoE). 0/absent ⇒ no shared expert.
	SharedExpertIntermediateSize int `json:"shared_expert_intermediate_size"`

	// DeepSeek-style MoE (GLM-4.5/4.6, model_type glm4_moe). NRoutedExperts is the
	// routed expert count (the n_routed_experts spelling, vs num_experts); each
	// shared expert is moe_intermediate_size wide, so the always-on shared FFN width
	// is NSharedExperts × MoeIntermediateSize. FirstKDenseReplace layers at the top
	// of the stack are plain dense MLPs (no router); the rest are MoE.
	// RoutedScalingFactor multiplies the routed top-k weights.
	NRoutedExperts      int     `json:"n_routed_experts"`
	NSharedExperts      int     `json:"n_shared_experts"`
	FirstKDenseReplace  int     `json:"first_k_dense_replace"`
	RoutedScalingFactor float64 `json:"routed_scaling_factor"`

	// DeepSeek MLA + DeepSeekMoE (deepseek_v2 / deepseek_v3). MLA splits the
	// per-head dims: queries route through an optional q_lora_rank bottleneck
	// (q_a_proj→norm→q_b_proj; null/0 ⇒ a direct q_proj, the V2-Lite path), K/V
	// compress to a shared kv_lora_rank latent (the CACHED state) plus a
	// qk_rope_head_dim rope-carrying key shared across heads. Q/K are
	// qk_nope_head_dim (no RoPE) + qk_rope_head_dim (decoupled RoPE); V is the
	// (different) v_head_dim. The MoE routing flavor is config-driven: V3 scores
	// with sigmoid + an e_score_correction_bias and limits selection to TopkGroup
	// of NGroup expert groups (noaux_tc); V2 scores with softmax and (for V2-Lite)
	// has NGroup=1 ⇒ plain greedy top-k. ScoringFunc/TopkMethod carry those knobs.
	QLoRARank      int     `json:"q_lora_rank"`
	KVLoRARank     int     `json:"kv_lora_rank"`
	QKNopeHeadDim  int     `json:"qk_nope_head_dim"`
	QKRopeHeadDim  int     `json:"qk_rope_head_dim"`
	VHeadDim       int     `json:"v_head_dim"`
	NGroup         int     `json:"n_group"`
	TopkGroup      int     `json:"topk_group"`
	ScoringFunc    string  `json:"scoring_func"` // "sigmoid" (V3) | "softmax" (V2); absent ⇒ family default
	TopkMethod     string  `json:"topk_method"`  // "noaux_tc" (V3) | "greedy"/"group_limited_greedy" (V2)
	RopeInterleave *bool   `json:"rope_interleave"`
	AttnScaleMul   float64 `json:"-"` // yarn mscale² folded into the attention scale (0 ⇒ plain qk_head_dim^-0.5)

	// Llama 4 text decoder (llama4_text): iRoPE. NoRopeLayers[i]==1 ⇒ layer i uses
	// RoPE, ==0 ⇒ NoPE (no positional). MoeLayers lists the MoE layer indices (the
	// rest are dense at IntermediateSizeMLP; experts + shared expert use
	// intermediate_size). The NoPE layers apply attention-temperature tuning
	// (q *= log1p(floor((pos+1)/FloorScale))·AttnScaleL4 + 1); the RoPE layers apply a
	// parameter-free L2 (RMS-over-head-dim) QK-norm when use_qk_norm. AttentionChunkSize
	// bounds local attention on RoPE layers (full causal for sequences below it).
	IntermediateSizeMLP    int     `json:"intermediate_size_mlp"`
	NoRopeLayers           []int   `json:"no_rope_layers"`
	MoeLayers              []int   `json:"moe_layers"`
	InterleaveMoeLayerStep int     `json:"interleave_moe_layer_step"`
	AttnTemperatureTuning  bool    `json:"attn_temperature_tuning"`
	FloorScale             float64 `json:"floor_scale"`
	AttnScaleL4            float64 `json:"attn_scale"`
	AttentionChunkSize     int     `json:"attention_chunk_size"`

	// Granite-4.0-H (GraniteMoeHybrid): the Mamba-2 mixer dims (per-layer kind in
	// LayerTypes = "mamba" | "attention"), the shared-expert width, and the four
	// Granite scalar multipliers applied in the forward (embedding scale, attention
	// scale, residual-add scale, final-logit divisor). MoE on every layer reuses
	// NumLocalExperts / NumExpertsPerTok / IntermediateDim.
	MambaNHeads            int     `json:"mamba_n_heads"`
	MambaDHead             int     `json:"mamba_d_head"`
	MambaDState            int     `json:"mamba_d_state"`
	MambaDConv             int     `json:"mamba_d_conv"`
	MambaNGroups           int     `json:"mamba_n_groups"`
	SharedIntermediateSize int     `json:"shared_intermediate_size"`
	EmbeddingMultiplier    float64 `json:"embedding_multiplier"`
	AttentionMultiplier    float64 `json:"attention_multiplier"`
	ResidualMultiplier     float64 `json:"residual_multiplier"`
	LogitsScaling          float64 `json:"logits_scaling"`

	// Nemotron-H (NemotronH): single-op-per-block hybrid (layers_block_type entries
	// "mamba" | "attention" | "mlp"), NoPE attention, non-gated relu² MLP. Its
	// Mamba-2 uses its own key spellings (mamba_num_heads / mamba_head_dim /
	// ssm_state_size / n_groups / conv_kernel) and layer_norm_epsilon for eps.
	LayersBlockType []string `json:"layers_block_type"`
	MambaNumHeads   int      `json:"mamba_num_heads"`
	MambaHeadDim    int      `json:"mamba_head_dim"`
	SSMStateSize    int      `json:"ssm_state_size"`
	NGroups         int      `json:"n_groups"`
	ConvKernel      int      `json:"conv_kernel"`

	// Gated DeltaNet (qwen3_5_moe linear-attention layers). The linear layers
	// replace softmax attention with a gated delta-rule recurrence over a
	// fixed-size matrix state; these size its conv + per-head key/value dims.
	// GVA: LinearNumValueHeads is a multiple of LinearNumKeyHeads (q/k are
	// repeat-interleaved to the value-head count). Absent for every other family.
	LinearConvKernelDim int `json:"linear_conv_kernel_dim"`
	LinearKeyHeadDim    int `json:"linear_key_head_dim"`
	LinearValueHeadDim  int `json:"linear_value_head_dim"`
	LinearNumKeyHeads   int `json:"linear_num_key_heads"`
	LinearNumValueHeads int `json:"linear_num_value_heads"`

	// RopeScaling is HF's rope_scaling object (llama3 / linear / yarn / …).
	// Plain Llama-3.0 and Qwen3 leave it null; Llama-3.1+/3.2 set it. Kept raw
	// and decoded by parseRopeScaling (G4: linear + llama3 + yarn supported).
	// omitempty: a nil RawMessage marshals to the literal `null` (4 bytes), which
	// on a .giw round-trip (json.Marshal(Cfg) → reload) makes len()>0 fire a
	// "present" check on absent config — omitempty keeps nil truly absent.
	RopeScaling json.RawMessage `json:"rope_scaling,omitempty"`

	// QuantizationConfig is HF's quantization_config object — for safetensors
	// shipped pre-quantized (GPTQ: packed int4 qweight/qzeros/scales/g_idx).
	// Decoded by parseGPTQ; absent for full-precision checkpoints.
	QuantizationConfig json.RawMessage `json:"quantization_config,omitempty"`

	// RopeParameters is the newer per-attention-type RoPE config (Mellum):
	// {"full_attention": {...}, "sliding_attention": {...}}, each a rope_theta +
	// a rope_scaling-style object. Decoded by parseRopeParameters. omitempty: see
	// RopeScaling (nil must not round-trip through .giw as `null`).
	RopeParameters json.RawMessage `json:"rope_parameters,omitempty"`

	// MRopeSection is Qwen2.5-VL's m-RoPE head_dim/2 split across the (temporal,
	// height, width) position components. Not parsed directly from JSON — the
	// qwen2_5_vl adapter extracts it from the nested rope_parameters. nil = plain
	// scalar RoPE (every other family). (P5)
	MRopeSection []int `json:"-"`

	// PartialRotaryFactor is the fraction of head_dim RoPE rotates (Phi: 0.4);
	// 0/absent means full rotary. Consumed via Config.rotaryDim.
	PartialRotaryFactor float64 `json:"partial_rotary_factor"`

	// Gemma 4 attention deltas (HF model_type "gemma4_unified_text"; GGUF
	// general.architecture "gemma4"). The global/full-attention layers use a
	// wider head (GlobalHeadDim, e.g. 512 vs the 256 local HeadDim), a single KV
	// head (NumGlobalKVHeads), share K and V (AttentionKEqV), and apply
	// partial-rotary RoPE (PartialRotaryFactor, e.g. 0.25). Local/sliding layers
	// keep HeadDim / NumKVHeads / full rotary. Absent/zero for every other family.
	GlobalHeadDim    int  `json:"global_head_dim"`
	NumGlobalKVHeads int  `json:"num_global_key_value_heads"`
	AttentionKEqV    bool `json:"attention_k_eq_v"`

	// Gemma 4 E-model extras. SharedKVLayers: the last N layers carry no k/v
	// projection and reuse the KV of the last non-shared layer of their type
	// (sliding vs full). HiddenSizePerLayerInput/VocabSizePerLayerInput size the
	// PLE (per-layer-embedding) table. FFNPerLayer is the variable per-layer FFN
	// width from the GGUF (config.json carries a scalar intermediate_size).
	SharedKVLayers          int   `json:"num_kv_shared_layers"`
	HiddenSizePerLayerInput int   `json:"hidden_size_per_layer_input"`
	VocabSizePerLayerInput  int   `json:"vocab_size_per_layer_input"`
	FFNPerLayer             []int `json:"-"`

	// GPT-2 uses a different config vocabulary: n_embd /
	// n_head / n_layer / n_positions / n_inner / layer_norm_epsilon /
	// activation_function instead of hidden_size etc. The gpt2 adapter reads
	// these directly.
	NEmbd              int     `json:"n_embd"`
	NHead              int     `json:"n_head"`
	NLayer             int     `json:"n_layer"`
	NPositions         int     `json:"n_positions"`
	NInner             int     `json:"n_inner"` // null/0 ⇒ 4*n_embd
	LayerNormEpsilon   float64 `json:"layer_norm_epsilon"`
	ActivationFunction string  `json:"activation_function"` // GPT-2: "gelu_new"

	// LayerTypes is the per-layer attention kind ("sliding_attention" /
	// "full_attention"). Gemma 3's checkpoints carry the local:global pattern
	// here explicitly; sliding_window_pattern is often null in config.json, so
	// this is the authoritative source when present (see IsGlobalLayer).
	LayerTypes []string `json:"layer_types"`

	// EOSTokenID is the checkpoint's end-of-sequence id(s). HF stores it as
	// either a scalar or a list, so it's kept raw and decoded by EOSIDs.
	// omitempty: nil must not round-trip through .giw as the literal `null`.
	EOSTokenID json.RawMessage `json:"eos_token_id,omitempty"`

	// Gemma 2 fields that MUST be absent/zero in a Gemma 3 checkpoint.
	// ValidateAssumptions rejects a checkpoint that still sets them so we
	// fail loudly rather than silently skip soft-capping.
	FinalLogitSoftcap float64 `json:"final_logit_softcapping"`
	AttnLogitSoftcap  float64 `json:"attn_logit_softcapping"`
}

// IsGlobalLayer reports whether layer i is a global (full) attention layer
// vs a local (sliding-window) one. Gemma 3 carries this per-layer in
// LayerTypes ("full_attention" vs "sliding_attention"), which is the
// authoritative source. Only if LayerTypes is absent do we fall back to the
// SlidingWindowPattern arithmetic (a global layer as the last of each group;
// pattern=6 → layers 5, 11, 17 → 3 global of 18 in the 270M).
//
// Getting this right is load-bearing: local and global layers use different
// RoPE bases (10k vs 1e6), so a misclassified layer silently corrupts logits.
func (c *Config) IsGlobalLayer(i int) bool {
	if i >= 0 && i < len(c.LayerTypes) {
		return c.LayerTypes[i] == "full_attention"
	}
	p := c.SlidingWindowPattern
	if p <= 0 {
		return true // no pattern configured → all-global (degenerate)
	}
	return (i+1)%p == 0
}

// IsLinearLayer reports whether layer i is a Gated DeltaNet (linear-attention)
// layer rather than a softmax-attention layer — the qwen3_5_moe 3:1 hybrid, read
// from layer_types. False for every non-hybrid family.
func (c *Config) IsLinearLayer(i int) bool {
	return i >= 0 && i < len(c.LayerTypes) && c.LayerTypes[i] == "linear_attention"
}

// headDim returns the per-head dimension, falling back to hidden/heads when
// config.json omits head_dim. Gemma and Qwen3 always set it explicitly (and
// for Gemma heads*head_dim != hidden, so the field is load-bearing there);
// many Llama/Mistral configs omit it, where hidden_size/num_attention_heads
// is the definition.
func (c *Config) headDim() int {
	if c.HeadDim > 0 {
		return c.HeadDim
	}
	if c.NumHeads > 0 {
		return c.HiddenDim / c.NumHeads
	}
	return 0
}

// ValidateAssumptions fails loudly on any config the scaffolded forward pass
// is not built to honor. Mirrors encoder.Config.ValidateAssumptions: pin the
// assumptions at load time rather than produce junk logits at run time.
func (c *Config) ValidateAssumptions() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.HeadDim == 0:
		return fmt.Errorf("decoder: missing required dim (hidden=%d layers=%d heads=%d headDim=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads, c.HeadDim)
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder: num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case c.VocabSize == 0:
		return fmt.Errorf("decoder: vocab_size is zero")
	case c.HiddenActivation != "" && c.HiddenActivation != "gelu_pytorch_tanh":
		return fmt.Errorf("decoder: hidden_activation=%q unsupported (gelu_pytorch_tanh / GeGLU only)", c.HiddenActivation)
	case c.FinalLogitSoftcap != 0 || c.AttnLogitSoftcap != 0:
		return fmt.Errorf("decoder: soft-capping set (final=%v attn=%v) — that's a Gemma 2 checkpoint; this path is Gemma 3 (QK-norm) only",
			c.FinalLogitSoftcap, c.AttnLogitSoftcap)
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder: rms_norm_eps must be >0, got %v", c.RMSNormEps)
	case c.RoPELocalBase <= 0 || c.RoPEGlobalBase <= 0:
		return fmt.Errorf("decoder: rope base must be >0 (local=%v global=%v)", c.RoPELocalBase, c.RoPEGlobalBase)
	case len(c.LayerTypes) > 0 && len(c.LayerTypes) != c.NumLayers:
		return fmt.Errorf("decoder: layer_types has %d entries, want num_hidden_layers=%d", len(c.LayerTypes), c.NumLayers)
	}
	for i, lt := range c.LayerTypes {
		if lt != "sliding_attention" && lt != "full_attention" {
			return fmt.Errorf("decoder: layer_types[%d]=%q unsupported (want sliding_attention/full_attention)", i, lt)
		}
	}
	return nil
}

// validateQwen3 pins the assumptions the qwen3 forward path makes (dense,
// SwiGLU, GQA, single-base RoPE). The Qwen3-MoE model_type isn't registered, so
// reaching here already implies a dense checkpoint; this guards the rest.
func (c *Config) validateQwen3() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.HeadDim == 0:
		return fmt.Errorf("decoder(qwen3): missing required dim (hidden=%d layers=%d heads=%d headDim=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads, c.HeadDim)
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(qwen3): num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case c.VocabSize == 0:
		return fmt.Errorf("decoder(qwen3): vocab_size is zero")
	case c.IntermediateDim == 0:
		return fmt.Errorf("decoder(qwen3): intermediate_size is zero")
	case c.HiddenAct != "" && c.HiddenAct != "silu":
		return fmt.Errorf("decoder(qwen3): hidden_act=%q unsupported (silu/SwiGLU only)", c.HiddenAct)
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder(qwen3): rms_norm_eps must be >0, got %v", c.RMSNormEps)
	case c.RoPEGlobalBase <= 0:
		return fmt.Errorf("decoder(qwen3): rope_theta must be >0, got %v", c.RoPEGlobalBase)
	}
	return nil
}

// validateQwen2 pins the assumptions the qwen2 forward path makes: like llama
// (dense, SwiGLU, GQA, single-base RoPE, no QK-norm, derived head_dim) but with
// q/k/v projection bias. Sliding-window attention (use_sliding_window) is a
// follow-up — reject it rather than silently run full attention.
func (c *Config) validateQwen2() error {
	if err := c.validateLlama(); err != nil {
		return err
	}
	if c.UseSlidingWindow {
		return fmt.Errorf("decoder(qwen2): use_sliding_window=true not yet supported (full-attention checkpoints only)")
	}
	return nil
}

// validateMixtral pins the Mixtral assumptions: the llama dense constraints
// (reused) plus a valid MoE config — top-k experts of num_local_experts, both
// positive and k ≤ E.
func (c *Config) validateMixtral() error {
	if err := c.validateLlama(); err != nil {
		return err
	}
	switch {
	case c.NumLocalExperts <= 0:
		return fmt.Errorf("decoder(mixtral): num_local_experts must be >0, got %d", c.NumLocalExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumLocalExperts:
		return fmt.Errorf("decoder(mixtral): num_experts_per_tok %d out of range (1..%d)", c.NumExpertsPerTok, c.NumLocalExperts)
	}
	return nil
}

// validateQwen2Moe pins the Qwen2-MoE assumptions: the qwen2 dense constraints
// (llama + q/k/v bias) plus a valid sparse MoE on every layer (num_experts /
// num_experts_per_tok / moe_intermediate_size) and an always-on shared expert
// (shared_expert_intermediate_size). decoder_sparse_step must be 1 (every layer
// sparse) — the only configuration shipped by the Qwen-MoE family.
func (c *Config) validateQwen2Moe() error {
	if err := c.validateLlama(); err != nil {
		return err
	}
	switch {
	case c.NumExperts <= 0:
		return fmt.Errorf("decoder(qwen2_moe): num_experts must be >0, got %d", c.NumExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumExperts:
		return fmt.Errorf("decoder(qwen2_moe): num_experts_per_tok %d out of range (1..%d)", c.NumExpertsPerTok, c.NumExperts)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(qwen2_moe): moe_intermediate_size must be >0, got %d", c.MoeIntermediateSize)
	case c.SharedExpertIntermediateSize <= 0:
		return fmt.Errorf("decoder(qwen2_moe): shared_expert_intermediate_size must be >0, got %d", c.SharedExpertIntermediateSize)
	}
	return nil
}

// validateGemma4MoE pins the Gemma 4 26B-A4B MoE config, checked only when
// enable_moe_block is set (the dense E2B/E4B/12B variants leave it false and never
// reach here). A valid sparse MoE parallel to the dense mlp: num_experts > 0,
// 1 ≤ top_k_experts ≤ num_experts, moe_intermediate_size > 0. The router.scale /
// per_expert_scale tensors are checked at load. See docs/task-gemma4-moe.md §A4.
func (c *Config) validateGemma4MoE() error {
	switch {
	case c.NumExperts <= 0:
		return fmt.Errorf("decoder(gemma4-moe): num_experts must be >0, got %d", c.NumExperts)
	case c.TopKExperts <= 0 || c.TopKExperts > c.NumExperts:
		return fmt.Errorf("decoder(gemma4-moe): top_k_experts %d out of range (1..%d)", c.TopKExperts, c.NumExperts)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(gemma4-moe): moe_intermediate_size must be >0, got %d", c.MoeIntermediateSize)
	}
	return nil
}

// validateGptOss pins the gpt-oss assumptions: a valid sparse MoE on every layer
// (num_local_experts / num_experts_per_tok / expert width) and a sliding window for
// the alternating local layers. The attention (GQA + biases + per-head sinks) and
// YaRN RoPE are validated by the generic head-dim checks + parseRopeScaling at
// resolve time.
func (c *Config) validateGptOss() error {
	nExperts := c.NumLocalExperts
	if nExperts == 0 {
		nExperts = c.NumExperts
	}
	expInter := c.MoeIntermediateSize
	if expInter == 0 {
		expInter = c.IntermediateDim
	}
	switch {
	case c.NumLayers <= 0:
		return fmt.Errorf("decoder(gpt-oss): num_hidden_layers must be >0, got %d", c.NumLayers)
	case nExperts <= 0:
		return fmt.Errorf("decoder(gpt-oss): num_local_experts must be >0, got %d", nExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > nExperts:
		return fmt.Errorf("decoder(gpt-oss): num_experts_per_tok %d out of range (1..%d)", c.NumExpertsPerTok, nExperts)
	case expInter <= 0:
		return fmt.Errorf("decoder(gpt-oss): expert intermediate size must be >0, got %d", expInter)
	case c.SlidingWindow <= 0:
		return fmt.Errorf("decoder(gpt-oss): sliding_window must be >0, got %d", c.SlidingWindow)
	}
	return nil
}

// validateGlm4Moe pins the GLM-4.5/4.6 (glm4_moe) assumptions: a valid DeepSeek-style
// sparse MoE (n_routed_experts / num_experts_per_tok / moe_intermediate_size) with a
// shared expert (n_shared_experts ≥ 1), and a first_k_dense_replace in [0, num_layers)
// dense prefix. The attention is qwen3-like (QK-norm, partial RoPE, no bias) and is
// validated by the generic head-dim/layer checks at resolve time.
func (c *Config) validateGlm4Moe() error {
	switch {
	case c.NRoutedExperts <= 0:
		return fmt.Errorf("decoder(glm4_moe): n_routed_experts must be >0, got %d", c.NRoutedExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NRoutedExperts:
		return fmt.Errorf("decoder(glm4_moe): num_experts_per_tok %d out of range (1..%d)", c.NumExpertsPerTok, c.NRoutedExperts)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(glm4_moe): moe_intermediate_size must be >0, got %d", c.MoeIntermediateSize)
	case c.NSharedExperts <= 0:
		return fmt.Errorf("decoder(glm4_moe): n_shared_experts must be >0, got %d", c.NSharedExperts)
	case c.FirstKDenseReplace < 0 || c.FirstKDenseReplace >= c.NumLayers:
		return fmt.Errorf("decoder(glm4_moe): first_k_dense_replace %d out of range (0..%d)", c.FirstKDenseReplace, c.NumLayers-1)
	}
	return nil
}

// validateDeepseek pins the DeepSeek-V2/V3 (MLA) assumptions: a valid MLA geometry
// (kv_lora_rank + the split per-head qk_nope/qk_rope/v dims) and a DeepSeekMoE
// (n_routed_experts / num_experts_per_tok / moe_intermediate_size + n_shared_experts ≥ 1,
// a first_k_dense_replace dense prefix). q_lora_rank 0/absent is the V2-Lite direct-q
// path. Group-limited routing (n_group/topk_group) is optional (≤1 ⇒ plain top-k).
func (c *Config) validateDeepseek() error {
	switch {
	case c.KVLoRARank <= 0:
		return fmt.Errorf("decoder(%s): kv_lora_rank must be >0, got %d", c.ModelType, c.KVLoRARank)
	case c.QKNopeHeadDim <= 0 || c.QKRopeHeadDim <= 0 || c.VHeadDim <= 0:
		return fmt.Errorf("decoder(%s): bad MLA head dims (qk_nope=%d qk_rope=%d v=%d)", c.ModelType, c.QKNopeHeadDim, c.QKRopeHeadDim, c.VHeadDim)
	case c.QLoRARank < 0:
		return fmt.Errorf("decoder(%s): q_lora_rank must be ≥0, got %d", c.ModelType, c.QLoRARank)
	case c.NRoutedExperts <= 0:
		return fmt.Errorf("decoder(%s): n_routed_experts must be >0, got %d", c.ModelType, c.NRoutedExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NRoutedExperts:
		return fmt.Errorf("decoder(%s): num_experts_per_tok %d out of range (1..%d)", c.ModelType, c.NumExpertsPerTok, c.NRoutedExperts)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(%s): moe_intermediate_size must be >0, got %d", c.ModelType, c.MoeIntermediateSize)
	case c.NSharedExperts <= 0:
		return fmt.Errorf("decoder(%s): n_shared_experts must be >0, got %d", c.ModelType, c.NSharedExperts)
	case c.FirstKDenseReplace < 0 || c.FirstKDenseReplace >= c.NumLayers:
		return fmt.Errorf("decoder(%s): first_k_dense_replace %d out of range (0..%d)", c.ModelType, c.FirstKDenseReplace, c.NumLayers-1)
	case c.NGroup > 1 && (c.TopkGroup <= 0 || c.TopkGroup > c.NGroup):
		return fmt.Errorf("decoder(%s): topk_group %d out of range (1..%d)", c.ModelType, c.TopkGroup, c.NGroup)
	case c.NGroup > 1 && c.NRoutedExperts%c.NGroup != 0:
		return fmt.Errorf("decoder(%s): n_routed_experts %d not divisible by n_group %d", c.ModelType, c.NRoutedExperts, c.NGroup)
	}
	return nil
}

// validateGranite pins the Granite-4.0-H (granitemoehybrid) assumptions: a layer_types
// list covering every layer (mamba | attention), a valid Mamba-2 geometry, and a
// routed+shared MoE (num_local_experts / num_experts_per_tok / intermediate_size /
// shared_intermediate_size).
func (c *Config) validateGranite() error {
	switch {
	case len(c.LayerTypes) != c.NumLayers:
		return fmt.Errorf("decoder(granite): layer_types has %d entries, want %d", len(c.LayerTypes), c.NumLayers)
	case c.MambaNHeads <= 0 || c.MambaDHead <= 0 || c.MambaDState <= 0 || c.MambaDConv <= 0 || c.MambaNGroups <= 0:
		return fmt.Errorf("decoder(granite): bad mamba dims (n_heads=%d d_head=%d d_state=%d d_conv=%d n_groups=%d)", c.MambaNHeads, c.MambaDHead, c.MambaDState, c.MambaDConv, c.MambaNGroups)
	case c.MambaNHeads%c.MambaNGroups != 0:
		return fmt.Errorf("decoder(granite): mamba_n_heads %d not divisible by mamba_n_groups %d", c.MambaNHeads, c.MambaNGroups)
	case c.NumLocalExperts <= 0 || c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumLocalExperts:
		return fmt.Errorf("decoder(granite): bad MoE (experts=%d top_k=%d)", c.NumLocalExperts, c.NumExpertsPerTok)
	case c.IntermediateDim <= 0 || c.SharedIntermediateSize <= 0:
		return fmt.Errorf("decoder(granite): intermediate_size=%d shared_intermediate_size=%d must be >0", c.IntermediateDim, c.SharedIntermediateSize)
	}
	return nil
}

// validateNemotron pins the Nemotron-H (nemotron_h) assumptions: a layers_block_type
// covering every layer (mamba | attention | mlp), a valid Mamba-2 geometry, and a
// usable mlp width + layer-norm eps.
func (c *Config) validateNemotron() error {
	switch {
	case len(c.LayersBlockType) != c.NumLayers:
		return fmt.Errorf("decoder(nemotron_h): layers_block_type has %d entries, want %d", len(c.LayersBlockType), c.NumLayers)
	case c.MambaNumHeads <= 0 || c.MambaHeadDim <= 0 || c.SSMStateSize <= 0 || c.ConvKernel <= 0 || c.NGroups <= 0:
		return fmt.Errorf("decoder(nemotron_h): bad mamba dims (num_heads=%d head_dim=%d state=%d conv=%d groups=%d)", c.MambaNumHeads, c.MambaHeadDim, c.SSMStateSize, c.ConvKernel, c.NGroups)
	case c.MambaNumHeads%c.NGroups != 0:
		return fmt.Errorf("decoder(nemotron_h): mamba_num_heads %d not divisible by n_groups %d", c.MambaNumHeads, c.NGroups)
	case c.IntermediateDim <= 0:
		return fmt.Errorf("decoder(nemotron_h): intermediate_size must be >0")
	case c.LayerNormEpsilon <= 0:
		return fmt.Errorf("decoder(nemotron_h): layer_norm_epsilon must be >0")
	}
	return nil
}

// validateMellum pins the assumptions the mellum adapter makes: a sparse MoE on
// every layer (num_experts / num_experts_per_tok / moe_intermediate_size), the
// per-attention-type rope_parameters (full_attention YaRN + sliding plain), and
// the explicit layer_types interleave. The dense FFN axes (RMSNorm, SwiGLU,
// derived head_dim) are the same as llama and resolved there.
func (c *Config) validateMellum() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.NumKVHeads == 0:
		return fmt.Errorf("decoder(mellum): missing core dims (hidden/layers/heads)")
	case c.NumExperts <= 0:
		return fmt.Errorf("decoder(mellum): num_experts must be >0, got %d", c.NumExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumExperts:
		return fmt.Errorf("decoder(mellum): num_experts_per_tok %d out of range (1..%d)", c.NumExpertsPerTok, c.NumExperts)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(mellum): moe_intermediate_size must be >0, got %d", c.MoeIntermediateSize)
	case len(c.RopeParameters) == 0:
		return fmt.Errorf("decoder(mellum): rope_parameters required (full_attention + sliding_attention)")
	case len(c.LayerTypes) != c.NumLayers:
		return fmt.Errorf("decoder(mellum): layer_types has %d entries, want %d", len(c.LayerTypes), c.NumLayers)
	}
	return nil
}

// validateQwen35 pins the qwen3_5_moe assumptions: a 3:1 linear/full layer mix
// (layer_types), the Gated DeltaNet dims (GVA value-head count a multiple of the
// key-head count), per-attention-type RoPE in rope_parameters, QK-norm softmax
// layers, and a routed + shared MoE on every layer.
func (c *Config) validateQwen35() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.HeadDim == 0:
		return fmt.Errorf("decoder(qwen3_5_moe): missing core dims (hidden=%d layers=%d heads=%d headDim=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads, c.HeadDim)
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(qwen3_5_moe): num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case len(c.LayerTypes) != c.NumLayers:
		return fmt.Errorf("decoder(qwen3_5_moe): layer_types has %d entries, want %d", len(c.LayerTypes), c.NumLayers)
	case c.LinearConvKernelDim <= 0 || c.LinearKeyHeadDim <= 0 || c.LinearValueHeadDim <= 0:
		return fmt.Errorf("decoder(qwen3_5_moe): missing linear (DeltaNet) dims (conv=%d kHead=%d vHead=%d)",
			c.LinearConvKernelDim, c.LinearKeyHeadDim, c.LinearValueHeadDim)
	case c.LinearNumKeyHeads <= 0 || c.LinearNumValueHeads <= 0 || c.LinearNumValueHeads%c.LinearNumKeyHeads != 0:
		return fmt.Errorf("decoder(qwen3_5_moe): linear_num_value_heads %d not a multiple of linear_num_key_heads %d (GVA)",
			c.LinearNumValueHeads, c.LinearNumKeyHeads)
	case c.NumExperts <= 0 || c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumExperts:
		return fmt.Errorf("decoder(qwen3_5_moe): bad MoE (experts=%d top_k=%d)", c.NumExperts, c.NumExpertsPerTok)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(qwen3_5_moe): moe_intermediate_size must be >0")
	case len(c.RopeParameters) == 0:
		return fmt.Errorf("decoder(qwen3_5_moe): rope_parameters required")
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder(qwen3_5_moe): rms_norm_eps must be >0")
	}
	return nil
}

// validateGPT2 pins the assumptions the gpt2 forward path makes: LayerNorm,
// learned absolute positions (no RoPE), a non-gated GELU MLP, fused q/k/v with
// bias, and tied embeddings. GPT-2's config uses the n_embd/n_head/n_layer keys.
func (c *Config) validateGPT2() error {
	switch {
	case c.NEmbd == 0 || c.NLayer == 0 || c.NHead == 0:
		return fmt.Errorf("decoder(gpt2): missing required dim (n_embd=%d n_layer=%d n_head=%d)", c.NEmbd, c.NLayer, c.NHead)
	case c.NEmbd%c.NHead != 0:
		return fmt.Errorf("decoder(gpt2): n_embd %d not divisible by n_head %d", c.NEmbd, c.NHead)
	case c.VocabSize == 0:
		return fmt.Errorf("decoder(gpt2): vocab_size is zero")
	case c.NPositions == 0:
		return fmt.Errorf("decoder(gpt2): n_positions is zero (need the learned position table size)")
	case c.LayerNormEpsilon <= 0:
		return fmt.Errorf("decoder(gpt2): layer_norm_epsilon must be >0, got %v", c.LayerNormEpsilon)
	case c.ActivationFunction != "" && c.ActivationFunction != "gelu_new" && c.ActivationFunction != "gelu":
		return fmt.Errorf("decoder(gpt2): activation_function=%q unsupported (gelu_new/gelu)", c.ActivationFunction)
	}
	return nil
}

// validateLlama pins the assumptions the llama forward path makes (dense,
// SwiGLU, GQA, single-base RoPE, no QK-norm). It differs from validateQwen3 by
// allowing head_dim to be derived (headDim()). RoPE scaling (rope_scaling) is
// handled by the adapter via parseRopeScaling (G4: linear + llama3). Attention
// bias (Qwen2/GPT-2 q/k/v/o bias) is rejected — a later add.
// Plain Llama-2/3 / Mistral checkpoints pass.
func (c *Config) validateLlama() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.headDim() == 0:
		return fmt.Errorf("decoder(llama): missing required dim (hidden=%d layers=%d heads=%d headDim=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads, c.headDim())
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(llama): num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case c.VocabSize == 0:
		return fmt.Errorf("decoder(llama): vocab_size is zero")
	case c.IntermediateDim == 0:
		return fmt.Errorf("decoder(llama): intermediate_size is zero")
	case c.HiddenAct != "" && c.HiddenAct != "silu":
		return fmt.Errorf("decoder(llama): hidden_act=%q unsupported (silu/SwiGLU only)", c.HiddenAct)
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder(llama): rms_norm_eps must be >0, got %v", c.RMSNormEps)
	case c.RoPEGlobalBase <= 0:
		return fmt.Errorf("decoder(llama): rope_theta must be >0, got %v", c.RoPEGlobalBase)
	case c.AttentionBias:
		return fmt.Errorf("decoder(llama): attention_bias=true (q/k/v/o bias) not yet supported")
	}
	return nil
}

// rotaryDim returns the number of head dims RoPE rotates: partial_rotary_factor
// × head_dim when set (Phi), else 0 (the descriptor reads that as full head_dim).
func (c *Config) rotaryDim() int {
	if c.PartialRotaryFactor > 0 && c.PartialRotaryFactor < 1 {
		return int(c.PartialRotaryFactor * float64(c.headDim()))
	}
	return 0
}

// EOSIDs returns the configured end-of-sequence token ids, handling both the
// scalar (eos_token_id: 1) and list (eos_token_id: [1, 106]) JSON shapes HF
// emits. Empty when the field is absent.
func (c *Config) EOSIDs() []int {
	if len(c.EOSTokenID) == 0 {
		return nil
	}
	var one int
	if err := json.Unmarshal(c.EOSTokenID, &one); err == nil {
		return []int{one}
	}
	var many []int
	if err := json.Unmarshal(c.EOSTokenID, &many); err == nil {
		return many
	}
	return nil
}

// resolveEOSIDs returns the ids that end generation: config.json's
// eos_token_id, plus any extra ids from generation_config.json. The latter is
// HF's authoritative generation source and often lists more than config.json —
// Qwen3's config.json carries only <|im_end|> (151645) while its
// generation_config adds <|endoftext|> (151643), and both must stop a chat
// turn. Deduped, config.json's ids first. generation_config is best-effort
// (absent file → ignored).
func resolveEOSIDs(dir string, cfg *Config) []int {
	seen := map[int]bool{}
	var out []int
	add := func(ids []int) {
		for _, id := range ids {
			if !seen[id] {
				seen[id] = true
				out = append(out, id)
			}
		}
	}
	add(cfg.EOSIDs())
	add(eosFromGenerationConfig(os.DirFS(dir), "generation_config.json"))
	return out
}

// eosFromGenerationConfig reads eos_token_id from generation_config.json,
// reusing EOSIDs' scalar-or-list handling. Returns nil when the file is
// absent or has no eos_token_id.
func eosFromGenerationConfig(fsys fs.FS, name string) []int {
	b, err := fs.ReadFile(fsys, name)
	if err != nil {
		return nil
	}
	var g struct {
		EOSTokenID json.RawMessage `json:"eos_token_id"`
	}
	if json.Unmarshal(b, &g) != nil {
		return nil
	}
	gc := Config{EOSTokenID: g.EOSTokenID}
	return gc.EOSIDs()
}

// loadConfig reads and parses config.json from fsys.
func loadConfig(fsys fs.FS, name string) (*Config, error) {
	b, err := fs.ReadFile(fsys, name)
	if err != nil {
		return nil, fmt.Errorf("decoder: read %s: %w", name, err)
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("decoder: parse %s: %w", name, err)
	}
	// Composite/VL checkpoints (e.g. Qwen3.6-35B's qwen3_5_moe, shipped as a
	// *ForConditionalGeneration with a vision tower) nest the TEXT decoder's dims
	// under "text_config" rather than at the top level. Flatten it: decode
	// text_config into c first so its dims (hidden_size, num_hidden_layers,
	// num_experts, rope_parameters, layer_types, …) populate the otherwise-zero
	// fields, then re-apply the top-level keys so anything authoritative there
	// (model_type, tied-head signals) wins. json.Unmarshal only writes keys that
	// are present, so a flat config.json is unaffected (text_config absent).
	var nest struct {
		TextConfig json.RawMessage `json:"text_config"`
	}
	if json.Unmarshal(b, &nest) == nil && len(nest.TextConfig) > 0 {
		if err := json.Unmarshal(nest.TextConfig, &c); err != nil {
			return nil, fmt.Errorf("decoder: parse %s text_config: %w", name, err)
		}
		if err := json.Unmarshal(b, &c); err != nil {
			return nil, fmt.Errorf("decoder: parse %s: %w", name, err)
		}
	}
	return &c, nil
}
