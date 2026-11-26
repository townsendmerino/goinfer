package archparts

// TensorSchema maps each weight role to its safetensors key (per-layer roles
// are suffixes for tensorName). An empty string means the tensor is ABSENT for
// this family — e.g. Pre2 families have no Post*Norm, and a tied head has no
// LMHead. The per-family adapter (registry.go) picks the schema; the loader is
// schema-driven so one buildWeightsFromSafetensors serves every family.
//
// The norm roles are by POSITION in the block, not by HF tensor name: Gemma's
// "post_attention_layernorm" is a post-attn norm (PostAttnNorm) while Qwen's
// same-named tensor is the pre-MLP norm (PreMLPNorm) — exactly why the schema
// is per-family.
type TensorSchema struct {
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
