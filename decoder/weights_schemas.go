package decoder

// tensorSchema maps each weight role to its safetensors key (per-layer roles
// are suffixes for tensorName). An empty string means the tensor is ABSENT for
// this family — e.g. Pre2 families have no Post*Norm, and a tied head has no
// LMHead. The per-family adapter (registry.go) picks the schema; the loader is
// schema-driven so one buildWeightsFromSafetensors serves every family.
//
// The norm roles are by POSITION in the block, not by HF tensor name: Gemma's
// "post_attention_layernorm" is a post-attn norm (PostAttnNorm) while Qwen's
// same-named tensor is the pre-MLP norm (PreMLPNorm) — exactly why the schema
// is per-family.
type tensorSchema struct {
	Embed     string
	LMHead    string // "" = tied (use Embed)
	FinalNorm string
	// per-layer suffixes passed to tensorName(layer, suffix); "" = absent
	QProj, KProj, VProj, OProj string
	// GProj is Laguna's attention output gate (g_proj). "" = the family has none.
	GProj string
	// AttnSinks is Flint's per-query-head attention sink logit (self_attn.softmax_offset,
	// with AttnSinksAlt = self_attn.sinks for older checkpoints). "" = none.
	AttnSinks, AttnSinksAlt    string
	QBias, KBias, VBias        string // "" = no projection bias (Qwen2 sets q/k/v)
	QNorm, KNorm               string // "" = no QK-norm
	PreAttnNorm, PostAttnNorm  string
	GateProj, UpProj, DownProj string
	PreMLPNorm, PostMLPNorm    string
	// PreAttnNormLinear, PostAttnNormLinear, PreMLPNormLinear and PostMLPNormLinear override the corresponding suffix above
	// on layers where isLinearLayer(i) is true, as a group, not per field. buildWeightsFromSafetensors ORs the four into one
	// hasLinearNormOverride flag; if any is non-empty, all four replace the generic suffixes on linear layers, and a field
	// left "" then means "no norm at this position on linear layers" (Olmo Hybrid's PostAttnNormLinear and
	// PostMLPNormLinear below), not "fall back to the generic one". Setting one and expecting per-field fallback would
	// silently drop the other three's generic norms.
	//
	// Olmo Hybrid needs this: its two decoder-layer classes each define an attribute named "post_attention_layernorm" in
	// different positional roles (a post-attn norm on full-attention layers, NormPostOnly; a pre-MLP norm on linear/DeltaNet
	// layers, NormPre2), and only its linear layers carry an input_layernorm at all. One static per-family schema cannot
	// route one on-disk name to two LayerWeights fields by layer kind, so these four give linear layers their own suffix set.
	PreAttnNormLinear, PostAttnNormLinear, PreMLPNormLinear, PostMLPNormLinear string
	// MoE (Mixtral): router + per-expert gate/up/down. The Expert* templates
	// contain a single %d for the expert index. Empty ⇒ dense FFN.
	Router                           string
	RouterBias                       string // "" = none; DeepSeek/GLM e_score_correction_bias [NumExperts]
	ExpertGate, ExpertUp, ExpertDown string
	// Shared expert (Qwen2-MoE): an always-on gated MLP + a sigmoid gate. Empty ⇒
	// no shared expert.
	SharedGate, SharedUp, SharedDown string
	SharedExpertGate                 string
}

// lfm2TensorSchema: LFM2/LFM2.5. Tied head, Pre2 norms under LFM2's own names (operator_norm before the mixer, ffn_norm
// before the FFN), per-head RMSNorm on Q and K, SwiGLU under llama's w1/w2/w3 naming, and attention output as out_proj
// rather than o_proj.
//
// The attention entries apply to the attention layers only; the conv layers have none of them and carry
// conv.{in_proj,conv,out_proj}, which this schema cannot express (it has no conv roles) and buildLFM2Weights loads
// directly, the same division Granite uses for its Mamba tensors.
//
// FinalNorm is embedding_norm, not model.norm: LFM2 normalises before the tied LM head under a name no other family here
// uses, so a copy of "model.norm.weight" fails to load rather than loading the wrong thing.
var lfm2TensorSchema = tensorSchema{
	Embed:       "model.embed_tokens.weight",
	LMHead:      "", // tied (tie_word_embeddings true on every released checkpoint)
	FinalNorm:   "model.embedding_norm.weight",
	QProj:       "self_attn.q_proj.weight",
	KProj:       "self_attn.k_proj.weight",
	VProj:       "self_attn.v_proj.weight",
	OProj:       "self_attn.out_proj.weight",
	QNorm:       "self_attn.q_layernorm.weight",
	KNorm:       "self_attn.k_layernorm.weight",
	PreAttnNorm: "operator_norm.weight",
	GateProj:    "feed_forward.w1.weight",
	UpProj:      "feed_forward.w3.weight",
	DownProj:    "feed_forward.w2.weight",
	PreMLPNorm:  "ffn_norm.weight",
}

// gemma3TensorSchema: tied head, 4-norm sandwich, QK-norm.
var gemma3TensorSchema = tensorSchema{
	Embed:        "model.embed_tokens.weight",
	LMHead:       "", // tied
	FinalNorm:    "model.norm.weight",
	QProj:        "self_attn.q_proj.weight",
	KProj:        "self_attn.k_proj.weight",
	VProj:        "self_attn.v_proj.weight",
	OProj:        "self_attn.o_proj.weight",
	QNorm:        "self_attn.q_norm.weight",
	KNorm:        "self_attn.k_norm.weight",
	PreAttnNorm:  "input_layernorm.weight",
	PostAttnNorm: "post_attention_layernorm.weight",
	GateProj:     "mlp.gate_proj.weight",
	UpProj:       "mlp.up_proj.weight",
	DownProj:     "mlp.down_proj.weight",
	PreMLPNorm:   "pre_feedforward_layernorm.weight",
	PostMLPNorm:  "post_feedforward_layernorm.weight",
}

// qwen3TensorSchema: separate lm_head, 2-norm Pre2 (input_layernorm pre-attn,
// post_attention_layernorm pre-MLP), QK-norm, SwiGLU. Llama/Mistral/Qwen2 reuse
// this minus QNorm/KNorm (and Qwen2 adds q/k/v bias — a later add).
var qwen3TensorSchema = tensorSchema{
	Embed:        "model.embed_tokens.weight",
	LMHead:       "lm_head.weight",
	FinalNorm:    "model.norm.weight",
	QProj:        "self_attn.q_proj.weight",
	KProj:        "self_attn.k_proj.weight",
	VProj:        "self_attn.v_proj.weight",
	OProj:        "self_attn.o_proj.weight",
	QNorm:        "self_attn.q_norm.weight",
	KNorm:        "self_attn.k_norm.weight",
	PreAttnNorm:  "input_layernorm.weight",
	PostAttnNorm: "", // Pre2: no post-attn norm
	GateProj:     "mlp.gate_proj.weight",
	UpProj:       "mlp.up_proj.weight",
	DownProj:     "mlp.down_proj.weight",
	PreMLPNorm:   "post_attention_layernorm.weight", // HF name; positionally the pre-MLP norm
	PostMLPNorm:  "",                                // Pre2: no post-MLP norm
}

// llamaTensorSchema: Llama-2/3 dense. Identical to qwen3TensorSchema except
// Llama has no QK-norm tensors (QNorm/KNorm empty) — RoPE applies to raw q/k.
// Same Pre2 norm layout and SwiGLU MLP; LM head tied (small) or untied (8B+),
// resolved from lm_head.weight presence at load.
var llamaTensorSchema = tensorSchema{
	Embed:        "model.embed_tokens.weight",
	LMHead:       "lm_head.weight",
	FinalNorm:    "model.norm.weight",
	QProj:        "self_attn.q_proj.weight",
	KProj:        "self_attn.k_proj.weight",
	VProj:        "self_attn.v_proj.weight",
	OProj:        "self_attn.o_proj.weight",
	QNorm:        "", // no QK-norm
	KNorm:        "",
	PreAttnNorm:  "input_layernorm.weight",
	PostAttnNorm: "", // Pre2: no post-attn norm
	GateProj:     "mlp.gate_proj.weight",
	UpProj:       "mlp.up_proj.weight",
	DownProj:     "mlp.down_proj.weight",
	PreMLPNorm:   "post_attention_layernorm.weight", // HF name; positionally the pre-MLP norm
	PostMLPNorm:  "",                                // Pre2: no post-MLP norm
}

// cohereTensorSchema: Cohere / Command-R — llama attention/MLP tensor names, but
// the parallel block has ONE norm per layer (input_layernorm, shared by attn and
// MLP), so PreMLPNorm/PostAttnNorm/PostMLPNorm are all empty. No biases anywhere;
// no QK-norm (cohere1 Phase 1). Embeddings are tied (LMHead empty ⇒ use Embed).
var cohereTensorSchema = tensorSchema{
	Embed:        "model.embed_tokens.weight",
	LMHead:       "", // tied
	FinalNorm:    "model.norm.weight",
	QProj:        "self_attn.q_proj.weight",
	KProj:        "self_attn.k_proj.weight",
	VProj:        "self_attn.v_proj.weight",
	OProj:        "self_attn.o_proj.weight",
	QNorm:        "", // no QK-norm (Phase 1)
	KNorm:        "",
	PreAttnNorm:  "input_layernorm.weight", // the single shared parallel-block norm
	PostAttnNorm: "",                       // parallel: no post-attn norm
	GateProj:     "mlp.gate_proj.weight",
	UpProj:       "mlp.up_proj.weight",
	DownProj:     "mlp.down_proj.weight",
	PreMLPNorm:   "", // parallel: MLP reads the shared input norm, no separate pre-MLP norm
	PostMLPNorm:  "",
}

// mixtralTensorSchema: Mixtral — the llama attention/norm names with a sparse
// MoE FFN in place of the dense gate/up/down. Router + 8 experts (w1=gate,
// w3=up, w2=down) per layer. No QK-norm, no bias, untied head.
var mixtralTensorSchema = tensorSchema{
	Embed:       "model.embed_tokens.weight",
	LMHead:      "lm_head.weight",
	FinalNorm:   "model.norm.weight",
	QProj:       "self_attn.q_proj.weight",
	KProj:       "self_attn.k_proj.weight",
	VProj:       "self_attn.v_proj.weight",
	OProj:       "self_attn.o_proj.weight",
	PreAttnNorm: "input_layernorm.weight",
	PreMLPNorm:  "post_attention_layernorm.weight",
	Router:      "block_sparse_moe.gate.weight",
	ExpertGate:  "block_sparse_moe.experts.%d.w1.weight",
	ExpertUp:    "block_sparse_moe.experts.%d.w3.weight",
	ExpertDown:  "block_sparse_moe.experts.%d.w2.weight",
}

// mellumTensorSchema: Mellum2 — the llama attention/norm names with a sparse
// MoE FFN, but using the standard HF Qwen/Llama MoE naming (mlp.gate router,
// mlp.experts.E.{gate,up,down}_proj) rather than Mixtral's block_sparse_moe /
// w1,w3,w2. Every layer is MoE (mlp_layer_types all "sparse"); no QK-norm, no
// bias, untied head. The experts use moe_intermediate_size (MoEConfig.IntermediateDim).
var mellumTensorSchema = tensorSchema{
	Embed:       "model.embed_tokens.weight",
	LMHead:      "lm_head.weight",
	FinalNorm:   "model.norm.weight",
	QProj:       "self_attn.q_proj.weight",
	KProj:       "self_attn.k_proj.weight",
	VProj:       "self_attn.v_proj.weight",
	OProj:       "self_attn.o_proj.weight",
	QNorm:       "self_attn.q_norm.weight", // Mellum has QK-norm (per-head RMSNorm)
	KNorm:       "self_attn.k_norm.weight",
	PreAttnNorm: "input_layernorm.weight",
	PreMLPNorm:  "post_attention_layernorm.weight", // HF name; positionally the pre-MLP norm
	Router:      "mlp.gate.weight",
	ExpertGate:  "mlp.experts.%d.gate_proj.weight",
	ExpertUp:    "mlp.experts.%d.up_proj.weight",
	ExpertDown:  "mlp.experts.%d.down_proj.weight",
}

// gpt2TensorSchema is a marker — GPT-2's fused c_attn + Conv1D weight layout
// don't fit the per-suffix schema, so buildGPT2Weights handles its tensors
// directly. Kept non-empty so the adapter returns a valid (if unused) schema.
var gpt2TensorSchema = tensorSchema{Embed: "wte.weight"}

// qwen2TensorSchema: Qwen2/Qwen2.5 dense. Identical to llamaTensorSchema plus
// the q/k/v projection biases Qwen2 carries (o_proj stays biasless), and still
// no QK-norm (that arrived in Qwen3). Pre2 norms, SwiGLU; tied head on the small
// models / untied on the large, resolved from lm_head.weight presence at load.
var qwen2TensorSchema = tensorSchema{
	Embed:        "model.embed_tokens.weight",
	LMHead:       "lm_head.weight",
	FinalNorm:    "model.norm.weight",
	QProj:        "self_attn.q_proj.weight",
	KProj:        "self_attn.k_proj.weight",
	VProj:        "self_attn.v_proj.weight",
	OProj:        "self_attn.o_proj.weight",
	QBias:        "self_attn.q_proj.bias",
	KBias:        "self_attn.k_proj.bias",
	VBias:        "self_attn.v_proj.bias",
	QNorm:        "", // no QK-norm
	KNorm:        "",
	PreAttnNorm:  "input_layernorm.weight",
	PostAttnNorm: "", // Pre2: no post-attn norm
	GateProj:     "mlp.gate_proj.weight",
	UpProj:       "mlp.up_proj.weight",
	DownProj:     "mlp.down_proj.weight",
	PreMLPNorm:   "post_attention_layernorm.weight", // HF name; positionally the pre-MLP norm
	PostMLPNorm:  "",                                // Pre2: no post-MLP norm
}

// qwen35DenseTensorSchema is Qwen3.8's (model_type qwen3_5): identical to the MoE sibling's except the router/expert names
// give way to a plain SwiGLU. Kept as its own value rather than mutating the MoE schema, so the MoE families are untouched.
var qwen35DenseTensorSchema = tensorSchema{
	Embed:       "model.embed_tokens.weight",
	LMHead:      "lm_head.weight",
	FinalNorm:   "model.norm.weight",
	QProj:       "self_attn.q_proj.weight",
	KProj:       "self_attn.k_proj.weight",
	VProj:       "self_attn.v_proj.weight",
	OProj:       "self_attn.o_proj.weight",
	QNorm:       "self_attn.q_norm.weight",
	KNorm:       "self_attn.k_norm.weight",
	PreAttnNorm: "input_layernorm.weight",
	PreMLPNorm:  "post_attention_layernorm.weight",
	GateProj:    "mlp.gate_proj.weight",
	UpProj:      "mlp.up_proj.weight",
	DownProj:    "mlp.down_proj.weight",
}

// qwen35TensorSchema covers the qwen3_5_moe softmax layers' names (QK-norm, no bias) and the routed/shared MoE common to
// every layer. The Gated DeltaNet (linear) layers carry a tensor set this schema cannot express (in_proj_qkv/z/a/b,
// conv1d, A_log, dt_bias, norm, out_proj); loadQwen35Attn loads them. See docs/qwen3_5_moe.md.
var qwen35TensorSchema = tensorSchema{
	Embed:            "model.embed_tokens.weight",
	LMHead:           "lm_head.weight",
	FinalNorm:        "model.norm.weight",
	QProj:            "self_attn.q_proj.weight",
	KProj:            "self_attn.k_proj.weight",
	VProj:            "self_attn.v_proj.weight",
	OProj:            "self_attn.o_proj.weight",
	QNorm:            "self_attn.q_norm.weight",
	KNorm:            "self_attn.k_norm.weight",
	PreAttnNorm:      "input_layernorm.weight",
	PreMLPNorm:       "post_attention_layernorm.weight",
	Router:           "mlp.gate.weight",
	ExpertGate:       "mlp.experts.%d.gate_proj.weight",
	ExpertUp:         "mlp.experts.%d.up_proj.weight",
	ExpertDown:       "mlp.experts.%d.down_proj.weight",
	SharedGate:       "mlp.shared_expert.gate_proj.weight",
	SharedUp:         "mlp.shared_expert.up_proj.weight",
	SharedDown:       "mlp.shared_expert.down_proj.weight",
	SharedExpertGate: "mlp.shared_expert_gate.weight",
}

// qwen3MoeTensorSchema: qwen3's attention tensor names (per-head q_norm/k_norm,
// no q/k/v bias) with the FFN replaced on every layer by a sparse MoE (router
// mlp.gate + per-expert mlp.experts.%d.*) — no shared expert (Shared* fields
// left empty; arch.MoE.SharedIntermediateDim stays 0, which both the
// safetensors and GGUF loaders already read as "no shared expert").
var qwen3MoeTensorSchema = tensorSchema{
	Embed:       "model.embed_tokens.weight",
	LMHead:      "lm_head.weight",
	FinalNorm:   "model.norm.weight",
	QProj:       "self_attn.q_proj.weight",
	KProj:       "self_attn.k_proj.weight",
	VProj:       "self_attn.v_proj.weight",
	OProj:       "self_attn.o_proj.weight",
	QNorm:       "self_attn.q_norm.weight",
	KNorm:       "self_attn.k_norm.weight",
	PreAttnNorm: "input_layernorm.weight",
	PreMLPNorm:  "post_attention_layernorm.weight",
	Router:      "mlp.gate.weight",
	ExpertGate:  "mlp.experts.%d.gate_proj.weight",
	ExpertUp:    "mlp.experts.%d.up_proj.weight",
	ExpertDown:  "mlp.experts.%d.down_proj.weight",
}

// olmo3TensorSchema: Olmo 3 — QKNormWhole's q_norm/k_norm are whole-vector but the TENSOR NAME
// is identical to the per-head convention; only NormPostOnly's PreAttnNorm/PreMLPNorm being empty
// (no input_layernorm tensor exists at all — confirmed by instantiating Olmo3ForCausalLM
// directly) and PostAttnNorm/PostMLPNorm mapping to post_attention_layernorm/
// post_feedforward_layernorm actually differs from qwen3TensorSchema.
var olmo3TensorSchema = tensorSchema{
	Embed:        "model.embed_tokens.weight",
	LMHead:       "lm_head.weight",
	FinalNorm:    "model.norm.weight",
	QProj:        "self_attn.q_proj.weight",
	KProj:        "self_attn.k_proj.weight",
	VProj:        "self_attn.v_proj.weight",
	OProj:        "self_attn.o_proj.weight",
	QNorm:        "self_attn.q_norm.weight",
	KNorm:        "self_attn.k_norm.weight",
	PreAttnNorm:  "", // NormPostOnly: no input_layernorm tensor exists
	PostAttnNorm: "post_attention_layernorm.weight",
	GateProj:     "mlp.gate_proj.weight",
	UpProj:       "mlp.up_proj.weight",
	DownProj:     "mlp.down_proj.weight",
	PreMLPNorm:   "", // NormPostOnly: no pre-MLP norm tensor exists
	PostMLPNorm:  "post_feedforward_layernorm.weight",
}

// olmoHybridTensorSchema: Olmo Hybrid's full-attention layers only (its linear/DeltaNet layers
// are loaded entirely by loadQwen35Attn, gated on PlainFullAttn — see qwen35Params). Identical to
// olmo3TensorSchema for those layers (verified against the real modeling_olmo_hybrid.py:
// OlmoHybridAttention literally inherits Olmo3Attention's behavior), plus the *Linear norm
// overrides: linear layers carry an input_layernorm the full-attention layers lack, and reuse the
// SAME on-disk name as the full-attention layers' PostAttnNorm ("post_attention_layernorm.weight")
// for a DIFFERENT role — pre-MLP norm, not post-attn norm — because the two decoder-layer classes
// each independently define an attribute with that name (see PreAttnNormLinear's own comment).
var olmoHybridTensorSchema = tensorSchema{
	Embed:              "model.embed_tokens.weight",
	LMHead:             "lm_head.weight", // tie_word_embeddings false on the released checkpoint
	FinalNorm:          "model.norm.weight",
	QProj:              "self_attn.q_proj.weight",
	KProj:              "self_attn.k_proj.weight",
	VProj:              "self_attn.v_proj.weight",
	OProj:              "self_attn.o_proj.weight",
	QNorm:              "self_attn.q_norm.weight",
	KNorm:              "self_attn.k_norm.weight",
	PreAttnNorm:        "", // NormPostOnly (full-attention layers): no input_layernorm tensor
	PostAttnNorm:       "post_attention_layernorm.weight",
	GateProj:           "mlp.gate_proj.weight",
	UpProj:             "mlp.up_proj.weight",
	DownProj:           "mlp.down_proj.weight",
	PreMLPNorm:         "", // NormPostOnly (full-attention layers): no pre-MLP norm tensor
	PostMLPNorm:        "post_feedforward_layernorm.weight",
	PreAttnNormLinear:  "input_layernorm.weight",          // NormPre2 (linear layers)
	PostAttnNormLinear: "",                                // NormPre2: no post-attn norm
	PreMLPNormLinear:   "post_attention_layernorm.weight", // SAME name, pre-MLP role here
	PostMLPNormLinear:  "",                                // NormPre2: no post-MLP norm
}

// qwen2MoeTensorSchema: qwen2 attention (q/k/v bias) with the FFN replaced by a sparse MoE (router mlp.gate + per-expert
// mlp.experts.%d.*) plus an always-on shared expert (mlp.shared_expert.* and the mlp.shared_expert_gate sigmoid gate).
var qwen2MoeTensorSchema = tensorSchema{
	Embed:            "model.embed_tokens.weight",
	LMHead:           "lm_head.weight",
	FinalNorm:        "model.norm.weight",
	QProj:            "self_attn.q_proj.weight",
	KProj:            "self_attn.k_proj.weight",
	VProj:            "self_attn.v_proj.weight",
	OProj:            "self_attn.o_proj.weight",
	QBias:            "self_attn.q_proj.bias",
	KBias:            "self_attn.k_proj.bias",
	VBias:            "self_attn.v_proj.bias",
	PreAttnNorm:      "input_layernorm.weight",
	PreMLPNorm:       "post_attention_layernorm.weight",
	Router:           "mlp.gate.weight",
	ExpertGate:       "mlp.experts.%d.gate_proj.weight",
	ExpertUp:         "mlp.experts.%d.up_proj.weight",
	ExpertDown:       "mlp.experts.%d.down_proj.weight",
	SharedGate:       "mlp.shared_expert.gate_proj.weight",
	SharedUp:         "mlp.shared_expert.up_proj.weight",
	SharedDown:       "mlp.shared_expert.down_proj.weight",
	SharedExpertGate: "mlp.shared_expert_gate.weight",
}

// glm4moeTensorSchema: GLM-4.5/4.6 (glm4_moe). qwen3-like attention (per-head
// QK-norm, NO q/k/v bias) over a DeepSeek-style MoE — router gate + an
// e_score_correction_bias, per-expert gate/up/down, and an always-on shared expert
// under the PLURAL "shared_experts" name with NO sigmoid gate (SharedExpertGate
// empty ⇒ the ungated path). The first_k_dense_replace prefix layers carry the
// dense mlp.{gate,up,down}_proj instead; the loader uses those suffixes on
// i < FirstKDense and the MoE ones on i ≥ FirstKDense.
var glm4moeTensorSchema = tensorSchema{
	Embed:     "model.embed_tokens.weight",
	LMHead:    "lm_head.weight",
	FinalNorm: "model.norm.weight",
	QProj:     "self_attn.q_proj.weight",
	KProj:     "self_attn.k_proj.weight",
	VProj:     "self_attn.v_proj.weight",
	OProj:     "self_attn.o_proj.weight",
	// q/k/v bias (o_proj biasless), loaded only when arch.QKVBias (attention_bias).
	QBias:       "self_attn.q_proj.bias",
	KBias:       "self_attn.k_proj.bias",
	VBias:       "self_attn.v_proj.bias",
	QNorm:       "self_attn.q_norm.weight",
	KNorm:       "self_attn.k_norm.weight",
	PreAttnNorm: "input_layernorm.weight",
	PreMLPNorm:  "post_attention_layernorm.weight",
	// dense prefix (first_k_dense_replace) layers use these
	GateProj: "mlp.gate_proj.weight",
	UpProj:   "mlp.up_proj.weight",
	DownProj: "mlp.down_proj.weight",
	// MoE layers use these
	Router:     "mlp.gate.weight",
	RouterBias: "mlp.gate.e_score_correction_bias",
	ExpertGate: "mlp.experts.%d.gate_proj.weight",
	ExpertUp:   "mlp.experts.%d.up_proj.weight",
	ExpertDown: "mlp.experts.%d.down_proj.weight",
	SharedGate: "mlp.shared_experts.gate_proj.weight",
	SharedUp:   "mlp.shared_experts.up_proj.weight",
	SharedDown: "mlp.shared_experts.down_proj.weight",
	// SharedExpertGate empty: GLM adds the shared expert ungated.
}

// deepseekTensorSchema: DeepSeek-V2/V3 (MLA). The attention tensors are the MLA set
// (q_a/q_b/kv_a/kv_b + the two latent layernorms), loaded by loadDeepseekAttn rather
// than the QProj/KProj/VProj/OProj suffixes here — so those stay empty. The FFN side is
// identical to GLM: dense prefix (first_k_dense_replace), DeepSeekMoE (router gate +
// e_score_correction_bias, per-expert gate/up/down), and an ungated shared expert under
// the plural "shared_experts" name.
var deepseekTensorSchema = tensorSchema{
	Embed:       "model.embed_tokens.weight",
	LMHead:      "lm_head.weight",
	FinalNorm:   "model.norm.weight",
	PreAttnNorm: "input_layernorm.weight",
	PreMLPNorm:  "post_attention_layernorm.weight",
	// dense prefix (first_k_dense_replace) layers use these
	GateProj: "mlp.gate_proj.weight",
	UpProj:   "mlp.up_proj.weight",
	DownProj: "mlp.down_proj.weight",
	// MoE layers use these
	Router:     "mlp.gate.weight",
	RouterBias: "mlp.gate.e_score_correction_bias",
	ExpertGate: "mlp.experts.%d.gate_proj.weight",
	ExpertUp:   "mlp.experts.%d.up_proj.weight",
	ExpertDown: "mlp.experts.%d.down_proj.weight",
	SharedGate: "mlp.shared_experts.gate_proj.weight",
	SharedUp:   "mlp.shared_experts.up_proj.weight",
	SharedDown: "mlp.shared_experts.down_proj.weight",
	// SharedExpertGate empty: DeepSeek adds the shared expert ungated.
}

// bailingHybridTensorSchema (Ling 3.0): only the tensors shared by BOTH layer kinds — the FFN
// (dense prefix / MoE + shared expert, otherwise identical to DeepSeek's own) and the block norms.
// Attention tensors are NOT here: both mixers (MLA via loadDeepseekAttn, KDA via loadBailingKDA)
// use their own bespoke loaders, same as deepseek_v2/v3's own QProj/KProj/etc being unused.
// Verified against the real modeling_bailing_moe_v3.py, which departs from deepseek_v2/v3's own
// naming in three places: the embedding is word_embeddings (not embed_tokens), the router's
// expert-bias buffer is expert_bias (not e_score_correction_bias), and both MLA/KDA mixers are
// named self.attention (not self.self_attn — see mlaParams.AttnPrefix's own comment).
var bailingHybridTensorSchema = tensorSchema{
	Embed:       "model.word_embeddings.weight",
	LMHead:      "lm_head.weight",
	FinalNorm:   "model.norm.weight",
	PreAttnNorm: "input_layernorm.weight",
	PreMLPNorm:  "post_attention_layernorm.weight",
	// dense prefix (first_k_dense_replace) layers use these
	GateProj: "mlp.gate_proj.weight",
	UpProj:   "mlp.up_proj.weight",
	DownProj: "mlp.down_proj.weight",
	// MoE layers use these
	Router:     "mlp.gate.weight",
	RouterBias: "mlp.gate.expert_bias",
	ExpertGate: "mlp.experts.%d.gate_proj.weight",
	ExpertUp:   "mlp.experts.%d.up_proj.weight",
	ExpertDown: "mlp.experts.%d.down_proj.weight",
	SharedGate: "mlp.shared_experts.gate_proj.weight",
	SharedUp:   "mlp.shared_experts.up_proj.weight",
	SharedDown: "mlp.shared_experts.down_proj.weight",
	// SharedExpertGate empty: Bailing adds the shared expert ungated too (verified above).
}

// graniteTensorSchema is a marker — Granite-4.0-H's per-layer-kind tensors (Mamba-2
// mixer vs attention) and fused-stacked GraniteMoe experts don't fit the per-suffix
// schema, so buildGraniteWeights loads them directly. Kept non-empty for a valid
// (unused) schema return.
var graniteTensorSchema = tensorSchema{Embed: "model.embed_tokens.weight"}

// internlm2TensorSchema is a MARKER, like phi3's: InternLM2's fused, GROUPED wqkv cannot be
// expressed as per-tensor suffixes, so buildInternLM2Weights loads this family directly and
// the generic schema-driven path is never used. The names are recorded here anyway so the
// family's layout is discoverable from the same place every other family's is.
var internlm2TensorSchema = tensorSchema{
	Embed:       "model.tok_embeddings.weight",
	LMHead:      "output.weight",
	FinalNorm:   "model.norm.weight",
	OProj:       "attention.wo.weight",
	PreAttnNorm: "attention_norm.weight",
	PreMLPNorm:  "ffn_norm.weight",
	GateProj:    "feed_forward.w1.weight", // llama's original naming: w1 gate, w3 up, w2 down
	UpProj:      "feed_forward.w3.weight",
	DownProj:    "feed_forward.w2.weight",
	// QProj/KProj/VProj intentionally empty: they live inside attention.wqkv.
}

// phi3TensorSchema is a marker — Phi-3/Phi-4's fused qkv_proj + gate_up_proj are split by
// buildPhi3Weights into the standard fields, after which the generic llama forward runs.
var phi3TensorSchema = tensorSchema{Embed: "model.embed_tokens.weight"}

// glmOcrTensorSchema names GLM-OCR's per-layer tensors by ROLE, but is only a marker for the generic
// loader: the fused mlp.gate_up_proj is split by buildGlmOcrWeights, which reads these suffixes so the
// mapping lives in one place and decoder/glm_ocr_test.go pins it.
//
// THE NORM NAMES ARE NOT GEMMA'S, and both are [HiddenDim] plain-weight RMSNorms, so a copy of
// Gemma's map loads without error and computes something else. By position in the block (verified
// against GlmOcrTextDecoderLayer.forward, and the checkpoint's own tensor list):
//
//	input_layernorm           -> PreAttnNorm   (before attention)
//	post_self_attn_layernorm  -> PostAttnNorm  (on the attention OUTPUT, before the residual add)
//	post_attention_layernorm  -> PreMLPNorm    (before the MLP — Gemma's same-named tensor is the post-attention one)
//	post_mlp_layernorm        -> PostMLPNorm   (on the MLP OUTPUT, before the residual add)
//
// Embedding, final norm and head live under model.language_model. / the top level (glmOcrEmbed,
// glmOcrFinalNorm, glmOcrLMHead). GateProj/UpProj are empty on purpose: both come from GateUpProj.
var glmOcrTensorSchema = tensorSchema{
	Embed:        glmOcrEmbed,
	LMHead:       glmOcrLMHead,
	FinalNorm:    glmOcrFinalNorm,
	QProj:        "self_attn.q_proj.weight",
	KProj:        "self_attn.k_proj.weight",
	VProj:        "self_attn.v_proj.weight",
	OProj:        "self_attn.o_proj.weight",
	PreAttnNorm:  "input_layernorm.weight",
	PostAttnNorm: "post_self_attn_layernorm.weight",
	PreMLPNorm:   "post_attention_layernorm.weight",
	PostMLPNorm:  "post_mlp_layernorm.weight",
	DownProj:     "mlp.down_proj.weight",
}

const (
	glmOcrPrefix    = "model.language_model."
	glmOcrEmbed     = glmOcrPrefix + "embed_tokens.weight"
	glmOcrFinalNorm = glmOcrPrefix + "norm.weight"
	glmOcrLMHead    = "lm_head.weight"
	glmOcrGateUp    = "mlp.gate_up_proj.weight" // fused [2*inter, hidden]: rows [0,inter) gate, [inter,2*inter) up
)

// spark25TensorSchema is a marker — Spark-X2.5's fused q_k_v_proj is split by
// buildSpark25Weights into the standard fields, after which the generic forward runs. Note the
// non-standard embedding tensor name (model.embedding.weight, not model.embed_tokens.weight —
// verified against the real modeling_spark.py's `self.embedding = nn.Embedding(...)`).
var spark25TensorSchema = tensorSchema{Embed: "model.embedding.weight", GProj: "self_attn.g_proj.weight"}

// llama4TensorSchema is a marker — Llama 4's per-layer dense/MoE FFN, fused+transposed
// batched experts, and parameter-free L2 QK-norm are loaded directly by buildLlama4Weights.
var llama4TensorSchema = tensorSchema{Embed: "model.embed_tokens.weight"}

// nemotronTensorSchema is a marker — Nemotron-H's single-op-per-block layout
// (per-layer one of mamba/attention/mlp under a "mixer" prefix) is loaded directly
// by buildNemotronWeights.
var nemotronTensorSchema = tensorSchema{Embed: "backbone.embedding.weight"}

// lagunaTensorSchema: Laguna (poolside). The names follow the real Laguna-XS.2 checkpoint index, not modeling_laguna.py,
// because the two disagree in two places that matter:
//
//  1. The module allocates fused 3D expert parameters (LagunaExperts holds gate_up_proj [E, 2*inter, hidden] and
//     down_proj [E, hidden, inter]), but the shipped checkpoint stores per-expert 2D tensors; HF re-packs them at load
//     through its conversion mapping. The per-expert form is what goinfer already reads, so the Expert* templates apply
//     unchanged and no stacked-expert handling is needed.
//
//  2. The module names the shared expert self.shared_experts (plural, as GLM and DeepSeek do), but the checkpoint keys
//     are mlp.shared_expert.* (singular).
//
// RouterBias is likewise the shipped spelling: the bias lives under mlp.experts.* on disk and HF's
// _checkpoint_conversion_mapping rewrites it to mlp.gate.* at load, so reading the checkpoint directly means taking the
// experts spelling.
//
// The dense prefix layers (mlp_only_layers) use the plain GateProj/UpProj/DownProj names at the model's
// intermediate_size; the MoE layers use the Expert*/Shared* names at moe_intermediate_size. See docs/task-laguna.md.
var lagunaTensorSchema = tensorSchema{
	Embed:       "model.embed_tokens.weight",
	LMHead:      "lm_head.weight",
	FinalNorm:   "model.norm.weight",
	QProj:       "self_attn.q_proj.weight",
	KProj:       "self_attn.k_proj.weight",
	VProj:       "self_attn.v_proj.weight",
	OProj:       "self_attn.o_proj.weight",
	GProj:       "self_attn.g_proj.weight",
	QNorm:       "self_attn.q_norm.weight",
	KNorm:       "self_attn.k_norm.weight",
	PreAttnNorm: "input_layernorm.weight",
	PreMLPNorm:  "post_attention_layernorm.weight",
	// dense prefix (mlp_only_layers) layers
	GateProj: "mlp.gate_proj.weight",
	UpProj:   "mlp.up_proj.weight",
	DownProj: "mlp.down_proj.weight",
	// MoE layers
	Router:     "mlp.gate.weight",
	RouterBias: "mlp.experts.e_score_correction_bias",
	ExpertGate: "mlp.experts.%d.gate_proj.weight",
	ExpertUp:   "mlp.experts.%d.up_proj.weight",
	ExpertDown: "mlp.experts.%d.down_proj.weight",
	SharedGate: "mlp.shared_expert.gate_proj.weight",
	SharedUp:   "mlp.shared_expert.up_proj.weight",
	SharedDown: "mlp.shared_expert.down_proj.weight",
	// SharedExpertGate empty: Laguna adds the shared expert with no outer sigmoid gate.
}

// flintTensorSchema: Flint (RecursiveSI/flint). Qwen3-shaped attention (per-head
// q_norm/k_norm, no biases on the released checkpoint) with a learned
// per-query-head attention sink (self_attn.sinks, or self_attn.softmax_offset),
// a dense MLP prefix (mlp_only_layers), and a sigmoid-routed MoE (both per-expert
// 2D mlp.experts.%d.{gate,up,down}_proj.weight and fused 3D
// mlp.experts.{gate_up,down}_proj are handled by buildWeightsFromSafetensorsTo)
// plus an always-on ungated SwiGLU shared expert (mlp.shared_expert.*).
var flintTensorSchema = tensorSchema{
	Embed:        "model.embed_tokens.weight",
	LMHead:       "lm_head.weight",
	FinalNorm:    "model.norm.weight",
	QProj:        "self_attn.q_proj.weight",
	KProj:        "self_attn.k_proj.weight",
	VProj:        "self_attn.v_proj.weight",
	OProj:        "self_attn.o_proj.weight",
	QNorm:        "self_attn.q_norm.weight",
	KNorm:        "self_attn.k_norm.weight",
	AttnSinks:    "self_attn.sinks",
	AttnSinksAlt: "self_attn.softmax_offset",
	QBias:        "self_attn.q_proj.bias",
	KBias:        "self_attn.k_proj.bias",
	VBias:        "self_attn.v_proj.bias",
	PreAttnNorm:  "input_layernorm.weight",
	PreMLPNorm:   "post_attention_layernorm.weight",
	// dense prefix (mlp_only_layers) layers
	GateProj: "mlp.gate_proj.weight",
	UpProj:   "mlp.up_proj.weight",
	DownProj: "mlp.down_proj.weight",
	// MoE layers
	Router:     "mlp.gate.weight",
	ExpertGate: "mlp.experts.%d.gate_proj.weight",
	ExpertUp:   "mlp.experts.%d.up_proj.weight",
	ExpertDown: "mlp.experts.%d.down_proj.weight",
	SharedGate: "mlp.shared_expert.gate_proj.weight",
	SharedUp:   "mlp.shared_expert.up_proj.weight",
	SharedDown: "mlp.shared_expert.down_proj.weight",
}
