package config

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path"
)

// Config is the flat config.json schema shared by every supported family: field tags follow the HF names, so a checkpoint's
// config drives the loader rather than hardcoded constants, and each family's adapter reads the fields it needs (the
// per-family groups below). The per-layer local:global attention pattern is derived at load time (LayerTypes, else
// SlidingWindowPattern), not stored per layer. The example values in the trailing comments are Gemma 3 270M.
type Config struct {
	ModelType       string  `json:"model_type"`              // "gemma3_text" — selects the arch adapter
	VocabSize       int     `json:"vocab_size"`              // 262144
	HiddenDim       int     `json:"hidden_size"`             // 640 (270M)
	NumLayers       int     `json:"num_hidden_layers"`       // 18 (270M)
	NumHeads        int     `json:"num_attention_heads"`     // 4 (270M)
	NumKVHeads      int     `json:"num_key_value_heads"`     // 1 (270M) — GQA
	HeadDim         int     `json:"head_dim"`                // 256 (270M); note heads*headDim != hidden
	IntermediateDim int     `json:"intermediate_size"`       // 2048 (270M) — GeGLU
	MaxPositions    int     `json:"max_position_embeddings"` // 32768
	RMSNormEps      float64 `json:"rms_norm_eps"`
	RoPELocalBase   float64 `json:"rope_local_base_freq"` // 10000
	RoPEGlobalBase  float64 `json:"rope_theta"`           // 1000000
	// Granite-4.0-H: "rope" (HF default) or "nope" (what the released checkpoints ship), which means the attention layers get no
	// positional encoding at all.
	PositionEmbeddingType string  `json:"position_embedding_type"`
	SlidingWindow         int     `json:"sliding_window"`         // 512 (270M)
	SlidingWindowPattern  int     `json:"sliding_window_pattern"` // 6 → 5 local : 1 global
	QueryPreAttnScalar    float64 `json:"query_pre_attn_scalar"`  // 256 (270M)
	UseQKNorm             bool    `json:"use_qk_norm"`            // true in Gemma 3
	HiddenActivation      string  `json:"hidden_activation"`      // Gemma: "gelu_pytorch_tanh"
	HiddenAct             string  `json:"hidden_act"`             // Llama/Qwen: "silu" (different JSON key)
	AttentionBias         bool    `json:"attention_bias"`         // Qwen2/GPT-2 add q/k/v/o bias; Llama-3/Qwen3 don't
	UseSlidingWindow      bool    `json:"use_sliding_window"`     // Qwen2: gate for sliding-window attention (usually false)

	// Mixture-of-experts. NormTopKProb is a *bool so an absent field can default
	// to true (HF's MixtralConfig default). Mixtral names the expert count
	// num_local_experts; Mellum names it num_experts and gives the experts their
	// own (narrower) FFN width in moe_intermediate_size.
	NumLocalExperts     int   `json:"num_local_experts"`
	NumExperts          int   `json:"num_experts"`
	NumExpertsPerTok    int   `json:"num_experts_per_tok"`
	NormTopKProb        *bool `json:"norm_topk_prob"`
	MoeIntermediateSize int   `json:"moe_intermediate_size"`
	// Gemma 4 26B-A4B MoE (model_type gemma4, text_config.enable_moe_block=true). It spells the per-token expert count
	// top_k_experts (not num_experts_per_tok) and gates the whole parallel dense+MoE FFN sub-block on enable_moe_block; the dense
	// E2B/E4B/12B variants leave both zero/false.
	EnableMoeBlock bool `json:"enable_moe_block"`
	TopKExperts    int  `json:"top_k_experts"`
	// SharedExpertIntermediateSize is the always-on shared expert's FFN width
	// (Qwen-MoE/Qwen2-MoE). 0/absent ⇒ no shared expert.
	SharedExpertIntermediateSize int `json:"shared_expert_intermediate_size"`
	// MoeSharedExpertIntermediateSize (Nemotron-H MoE) is its own explicit shared-expert width, not derivable as
	// NSharedExperts*MoeIntermediateSize the way DeepSeek's is (the shared expert is twice a routed expert's width on Nemotron 3
	// Nano). A distinct field, so it is never silently mis-derived.
	MoeSharedExpertIntermediateSize int `json:"moe_shared_expert_intermediate_size"`

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

	// Laguna (poolside). MoeRoutedScalingFactor is its spelling of routed_scaling_factor.
	//
	// Gating is json.RawMessage because the same field ships with three JSON types across releases of one model_type ("per-head",
	// true, "per-element"); a string or bool field would fail on the others, so it is held raw and resolved by LagunaGatePerHead.
	//
	// NumAttentionHeadsPerLayer is layer i's query head count (absent where uniform). MlpOnlyLayers lists the plain dense-MLP
	// layers, contiguous from the top in every released config, so it maps onto FirstKDense.
	MoeRoutedScalingFactor      float64         `json:"moe_routed_scaling_factor"`
	MoeApplyRouterWeightOnInput bool            `json:"moe_apply_router_weight_on_input"`
	Gating                      json.RawMessage `json:"gating,omitempty"`
	NumAttentionHeadsPerLayer   []int           `json:"num_attention_heads_per_layer"`
	MlpOnlyLayers               []int           `json:"mlp_only_layers"`
	// MlpLayerTypes is the per-layer FFN kind ("dense" | "sparse"). It is the
	// RELIABLE spelling: all three released configs carry it, whereas
	// mlp_only_layers is absent from XS.2. Preferred over MlpOnlyLayers.
	MlpLayerTypes []string `json:"mlp_layer_types"`

	// Flint (Cognichip, model_type "flint"): hybrid sliding-window + NoPE full
	// attention with per-query-head attention sinks (self_attn.softmax_offset,
	// or self_attn.sinks on older builds) and a sigmoid-routed MoE (after
	// MlpOnlyLayers dense prefix) with an always-on ungated SwiGLU shared expert.
	// NoRopeLayerIndices lists the NoPE layer indices (derived from
	// layer_types[i]=="full_attention" when absent).
	NoRopeLayerIndices  []int  `json:"no_rope_layer_indices"`
	RouterScoreFunction string `json:"router_score_function"`
	RouterExpertBias    bool   `json:"router_expert_bias"`
	AttentionSink       *bool  `json:"attention_sink"`
	SoftmaxType         string `json:"softmax_type"`

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

	// Bailing Hybrid (Ling 3.0, model_type "bailing_hybrid"): MLA (above) alternating with Kimi Delta Attention (KDA), every
	// LayerGroupSize-th layer being MLA. layer_types is not a config.json field for this family: the pattern is computed from
	// LayerGroupSize, the same shape as Qwen3-Next's FullAttentionInterval. The key spellings are the family's own
	// (num_experts/num_shared_experts throughout, not deepseek_v3's n_routed_experts/n_shared_experts).
	LayerGroupSize   int `json:"layer_group_size"`
	NumSharedExperts int `json:"num_shared_experts"`
	// GatedAttentionProjGranularity ("head_wise" | "element_wise" | absent): MLA's optional per-head or per-element sigmoid output
	// gate (g_proj) on the attention context before the output projection: Laguna's structure (FeatAttnOutputGateSoftplus) with sigmoid
	// in place of softplus.
	GatedAttentionProjGranularity string `json:"gated_attention_proj_granularity_type"`
	// HeadwiseAttnOutputGate/GateAttnActMode (Spark-X2.5): its own spelling of the same gate-before-out_proj idea: a plain bool
	// (per-head when on; no per-element variant) plus an activation-mode string. "sigmoid" is the only released value; "silu"
	// is legal upstream but not implemented, and validateResolved rejects it rather than silently mis-running.
	HeadwiseAttnOutputGate bool   `json:"headwise_attn_output_gate"`
	GateAttnActMode        string `json:"gate_attn_act_mode"`
	// KDA's wrapper geometry: a depthwise short causal conv (kernel ShortConvKernelSize, SiLU-activated, three separate q/k/v
	// convs, unlike Gated DeltaNet's one combined conv), NoKDALora selecting a single f_proj/g_proj linear per gate (true on the
	// release) over a LoRA'd a/b-split pair, and the lower-bounded safe decay gate Ling-3.0-tiny selects.
	ShortConvKernelSize int     `json:"short_conv_kernel_size"`
	NoKDALora           bool    `json:"no_kda_lora"`
	KDASafeGate         bool    `json:"kda_safe_gate"`
	KDALowerBound       float64 `json:"kda_lower_bound"`

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

	// Cohere / Command-R. LogitScale multiplies the output logits (HF: logits =
	// logit_scale * lm_head(h)); goinfer's LogitScale DIVIDES, so the adapter
	// stores its reciprocal. LayerNormEps is Cohere's LayerNorm eps (its own JSON
	// key, distinct from rms_norm_eps and gpt-2's layer_norm_epsilon).
	LogitScale   float64 `json:"logit_scale"`
	LayerNormEps float64 `json:"layer_norm_eps"`

	// Nemotron-H (NemotronH): single-op-per-block hybrid (layers_block_type entries "mamba" | "attention" | "mlp"), NoPE
	// attention, non-gated relu^2 MLP. Its Mamba-2 uses its own key spellings (mamba_num_heads / mamba_head_dim / ssm_state_size /
	// n_groups / conv_kernel) and layer_norm_epsilon for eps.
	//
	// HybridOverridePattern is the same layer sequence in NVIDIA's released spelling: one character per block ("M" mamba, "*"
	// attention, "-" mlp). Every released NemotronH config.json carries this and not layers_block_type (transformers' internal
	// spelling, which the tiny fixtures emit), so the loader must read both.
	LayersBlockType       []string `json:"layers_block_type"`
	HybridOverridePattern string   `json:"hybrid_override_pattern"`
	MambaNumHeads         int      `json:"mamba_num_heads"`
	MambaHeadDim          int      `json:"mamba_head_dim"`
	SSMStateSize          int      `json:"ssm_state_size"`
	NGroups               int      `json:"n_groups"`
	ConvKernel            int      `json:"conv_kernel"`

	// AttentionHeadDim is NVIDIA's spelling of the attention head dimension in Nemotron-H's own config.json (the 4B carries this key and no
	// head_dim). Only the nemotron_h adapter reads it.
	AttentionHeadDim int `json:"attention_head_dim"`

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
	// LinearAllowNegEigval (Olmo Hybrid only): widens the write-gate beta from sigmoid's [0,1) to [0,2) (beta * 2.0), extending
	// the delta-rule's eigenvalue range to include negative. Absent (false) for every other DeltaNet family.
	LinearAllowNegEigval bool `json:"linear_allow_neg_eigval"`
	// FullAttentionInterval (Qwen3-Next only; qwen3_5_moe states the per-layer pattern in LayerTypes): the released config has no
	// layer_types, so the pattern is computed: layer i (0-indexed) is full_attention when (i+1)%FullAttentionInterval==0, else
	// linear_attention. NormalizeQwen3NextLayerTypes turns this into the LayerTypes list every other consumer reads, so it is the
	// only place that knows this family computes rather than states.
	FullAttentionInterval int `json:"full_attention_interval"`

	// RopeScaling is HF's rope_scaling object (llama3 / linear / yarn / ...); plain Llama-3.0 and Qwen3 leave it null. Kept raw
	// and decoded by parseRopeScaling. omitempty: a nil RawMessage marshals to the literal `null`, which on a .giw round-trip
	// (json.Marshal(Cfg), then reload) would make a len()>0 "present" check fire on an absent config.
	RopeScaling json.RawMessage `json:"rope_scaling,omitempty"`

	// QuantizationConfig is HF's quantization_config object — for safetensors
	// shipped pre-quantized (GPTQ: packed int4 qweight/qzeros/scales/g_idx).
	// Decoded by parseGPTQ; absent for full-precision checkpoints.
	QuantizationConfig json.RawMessage `json:"quantization_config,omitempty"`

	// Quantization is MLX's top-level quantization object ({"group_size":64,"bits":4,"mode":"affine"}).
	// Decoded by parseCheckpointQuantConfig when QuantizationConfig is absent.
	Quantization json.RawMessage `json:"quantization,omitempty"`

	// RopeParameters is the newer per-attention-type RoPE config (Mellum):
	// {"full_attention": {...}, "sliding_attention": {...}}, each a rope_theta +
	// a rope_scaling-style object. Decoded by parseRopeParameters. omitempty: see
	// RopeScaling (nil must not round-trip through .giw as `null`).
	RopeParameters json.RawMessage `json:"rope_parameters,omitempty"`

	// MRopeSection is Qwen2.5-VL's m-RoPE head_dim/2 split across the (temporal, height, width) position components; nil means
	// plain scalar RoPE. The qwen2_5_vl and qwen3_vl adapters extract it from the nested rope_parameters or rope_scaling
	// (clearing rope_scaling), so it is serialized under its own key: with `json:"-"` a .giw dropped it and a Qwen-VL loaded from
	// one ran plain RoPE on image positions, silently (text alone cannot show it).
	MRopeSection []int `json:"mrope_section,omitempty"`

	// PartialRotaryFactor is the fraction of head_dim RoPE rotates (Phi: 0.4);
	// 0/absent means full rotary. Consumed via Config.RotaryDim.
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
	// PerLayerConfig is how transformers >= 5.16 serializes the two fields above: a map from layer
	// index to that layer's overrides ({"2": {"head_dim": 64}}), with global_head_dim and
	// num_global_key_value_heads no longer written. Read by Gemma4GlobalGeometry.
	PerLayerConfig json.RawMessage `json:"per_layer_config,omitempty"`

	// Gemma 4 E-model extras. SharedKVLayers: the last N layers carry no k/v
	// projection and reuse the KV of the last non-shared layer of their type
	// (sliding vs full). HiddenSizePerLayerInput/VocabSizePerLayerInput size the
	// PLE (per-layer-embedding) table. FFNPerLayer is the variable per-layer FFN
	// width from the GGUF (config.json carries a scalar intermediate_size).
	SharedKVLayers          int   `json:"num_kv_shared_layers"`
	HiddenSizePerLayerInput int   `json:"hidden_size_per_layer_input"`
	VocabSizePerLayerInput  int   `json:"vocab_size_per_layer_input"`
	FFNPerLayer             []int `json:"-"`

	// PadTokenID (gemma4): the real multimodal forward substitutes this id's embedding and per-layer embedding at every
	// image/video/audio position before computing PLE's token-identity term: not the placeholder token's own id, and not a
	// skipped term (docs/multimodal.md). Lives under text_config; Load's text_config merge picks it up via this tag.
	PadTokenID int `json:"pad_token_id"`

	// UseBidirectionalAttention (gemma4): "vision" on 26B-A4B/31B checkpoints enables a blockwise bidirectional attention mask
	// over image/audio blocks on sliding (local) layers only; global layers stay strictly causal (docs/multimodal.md). Empty/null
	// on E2B/E4B, where image-block attention is plain causal, which GenerateGemma4VL's sequential prefill already produces. A
	// non-empty value is served: GenerateGemma4VL dispatches it to the batched, blockwise-masked forward
	// (prefillLogitsGemma4VLBidirectional / runLayersGemma4FromEmbedN) rather than the sequential causal path. Lives under
	// text_config, picked up by Load's text_config merge like PadTokenID.
	//
	// The type is gemma4BidirectionalAttention, not string: Config is one flat struct for every family, and Gemma 3's own config
	// has a same-named field that is a bool (use_bidirectional_attention). A plain string field fails to unmarshal it and breaks
	// gemma3 loading. The tolerant type accepts a real string and resolves anything else (bool, null, absent) to "", meaning no
	// bidirectional attention.
	UseBidirectionalAttention gemma4BidirectionalAttention `json:"use_bidirectional_attention"`

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

	// NoRopeLayerInterval (SmolLM3): HF's generation formula for NoRopeLayers (above) when config.json omits the explicit list:
	// NoPE exactly when (layer_idx+1) % interval == 0. It reuses the same NoRopeLayers field and "1 = has rope, 0 = NoPE"
	// convention as Llama 4.
	NoRopeLayerInterval int `json:"no_rope_layer_interval"`

	// LFM2 (model_type lfm2): the gated short-convolution block's geometry. The per-layer pattern rides on LayerTypes ("conv" |
	// "full_attention").
	//
	// ConvLCache is the conv kernel width (upstream conv_L_cache), named for the rolling state it implies. ConvDim is the
	// channel count the block operates on; it is optional and often absent. Upstream Lfm2ShortConv builds its Conv1d and in_proj
	// on config.hidden_size and never reads conv_dim, so hidden_size is the authority: absent means hidden_size, and
	// present-and-different is refused, because the reference would ignore it and we would not, a silent divergence rather than
	// a shape error. ConvBias is false on the released weights and there is no bias tensor to load; true is refused rather than
	// silently ignored.
	//
	// NormEps is LFM2's RMSNorm epsilon and has its own JSON key: "norm_eps", where every other RMSNorm family here writes
	// "rms_norm_eps", so reading cfg.RMSNormEps for this family yields 0, not 1e-5. That is not a rounding difference: the error
	// compounds through every norm into wrong logits while argmax still matches, so a greedy smoke test would call it correct
	// (docs/code-notes/decoder.md#Config.NormEps). The checkpoint also carries "block_norm_eps"; upstream Lfm2Config reads
	// norm_eps, so that one is deliberately not used.
	ConvLCache int `json:"conv_L_cache"`
	// LFM2's FFN width rule (Lfm2MLP): with block_auto_adjust_ff_dim the width is int(2*intermediate_size/3), then, when
	// block_ffn_dim_multiplier is set, int(multiplier*width) rounded up to block_multiple_of. The LFM2.5 text releases
	// state the width and set the flag false; LFM2-VL's LFM2-1.2B text_config sets it (12288 -> 8192).
	BlockAutoAdjustFFDim  bool     `json:"block_auto_adjust_ff_dim"`
	BlockFFNDimMultiplier *float64 `json:"block_ffn_dim_multiplier"`
	BlockMultipleOf       int      `json:"block_multiple_of"`
	ConvBias              bool     `json:"conv_bias"`
	ConvDim               int      `json:"conv_dim"`
	NormEps               float64  `json:"norm_eps"`

	// EOSTokenID is the checkpoint's end-of-sequence id(s). HF stores it as
	// either a scalar or a list, so it's kept raw and decoded by EOSIDs.
	// omitempty: nil must not round-trip through .giw as the literal `null`.
	EOSTokenID json.RawMessage `json:"eos_token_id,omitempty"`

	// Sampling is the checkpoint's recommended sampling, read by Load (generation_config.json) or decoder's ggufConfig. It is
	// not a config.json key; the name only carries it through a .giw.
	Sampling *RecommendedSampling `json:"goinfer_recommended_sampling,omitempty"`

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

// EffectiveHeadDim returns the per-head dimension, falling back to hidden/heads when
// config.json omits head_dim. Gemma and Qwen3 always set it explicitly (and
// for Gemma heads*head_dim != hidden, so the field is load-bearing there);
// many Llama/Mistral configs omit it, where hidden_size/num_attention_heads
// is the definition.
func (c *Config) EffectiveHeadDim() int {
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

// RotaryDim returns the number of head dims RoPE rotates: partial_rotary_factor
// × head_dim when set (Phi), else 0 (the descriptor reads that as full head_dim).
func (c *Config) RotaryDim() int {
	if c.PartialRotaryFactor > 0 && c.PartialRotaryFactor < 1 {
		return int(c.PartialRotaryFactor * float64(c.EffectiveHeadDim()))
	}
	return 0
}

// EOSIDs returns the configured end-of-sequence token ids, handling both the
// scalar (eos_token_id: 1) and list (eos_token_id: [1, 106]) JSON shapes HF
// emits. Empty when the field is absent.
func (c *Config) EOSIDs() []int {
	// A JSON null means "no id here" (transformers writes eos_token_id: null when generation_config.json carries it).
	// Unmarshalled into an int it would read as 0, and id 0 would end generation.
	if len(c.EOSTokenID) == 0 || string(bytes.TrimSpace(c.EOSTokenID)) == "null" {
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

// ResolveEOSIDs returns the ids that end generation: config.json's eos_token_id, plus any extra ids from
// generation_config.json, HF's authoritative generation source, which often lists more (Qwen3's config.json carries only
// <|im_end|> while its generation_config adds <|endoftext|>, and both must stop a chat turn). Deduped, config.json's ids
// first. generation_config is best-effort: an absent file is ignored.
func ResolveEOSIDs(dir string, cfg *Config) []int {
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

// Load reads and parses config.json from fsys, and the recommended sampling from the generation_config.json beside it.
func Load(fsys fs.FS, name string) (*Config, error) {
	b, err := fs.ReadFile(fsys, name)
	if err != nil {
		return nil, fmt.Errorf("decoder: read %s: %w", name, err)
	}
	b = LenientJSON(b) // Python's Infinity and NaN, as released configs carry them; request bodies stay strict
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("decoder: parse %s: %w", name, err)
	}
	// Composite/VL checkpoints nest the text decoder's dims under "text_config". Flatten it: decode text_config into c first so
	// its dims populate the otherwise-zero fields, then re-apply the top-level keys so anything authoritative there (model_type,
	// tied-head signals) wins. json.Unmarshal only writes keys that are present, so a flat config.json is unaffected.
	var nest struct {
		TextConfig json.RawMessage `json:"text_config"`
		// Qwen3-ASR's original layout wraps everything in thinker_config (as Qwen3-Omni's does): the text decoder's dims are one level deeper.
		Thinker struct {
			TextConfig json.RawMessage `json:"text_config"`
		} `json:"thinker_config"`
	}
	if json.Unmarshal(b, &nest) == nil && len(nest.TextConfig) == 0 {
		nest.TextConfig = nest.Thinker.TextConfig
	}
	if len(nest.TextConfig) > 0 {
		if err := json.Unmarshal(nest.TextConfig, &c); err != nil {
			return nil, fmt.Errorf("decoder: parse %s text_config: %w", name, err)
		}
		if err := json.Unmarshal(b, &c); err != nil {
			return nil, fmt.Errorf("decoder: parse %s: %w", name, err)
		}
	}
	c.Sampling = samplingFromGenerationConfig(fsys, path.Join(path.Dir(name), "generation_config.json"))
	return &c, nil
}
