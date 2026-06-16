package decoder

// Architecture is the resolved, family-agnostic description of a decoder LLM's
// structure. ONE generic forward pass (runLayers/forward/causalAttention/
// gatedMLP) reads it; per-family adapters (registry.go) populate it from that
// family's config.json. Every supported family — Gemma/Llama/Mistral/Qwen/
// GPT-2/Mixtral/Mellum — is expressed as a descriptor here, so adding one is
// descriptor population, not new forward code.
//
// Every field below is consumed by the forward pass, which rejects descriptor
// values it doesn't implement rather than silently mis-running.
type Architecture struct {
	Name string // family name, for logs/errors ("gemma3")

	// Dims (mirrors config.json; the loader also reads these for tensor shapes).
	HiddenDim, NumLayers, NumHeads, NumKVHeads, HeadDim int
	IntermediateDim, VocabSize                          int
	MaxPositions                                        int // learned-position table size (GPT-2 wpe); 0 for RoPE families

	// Norm.
	Norm          NormKind
	RMSAddOne     bool // Gemma's (1+w) scaling; false for Llama/Qwen
	NormEps       float64
	NormPlacement NormPlacement // Pre2 (Llama) | Sandwich4 (Gemma)

	// MLP.
	Act         ActKind
	NonGatedMLP bool       // GPT-2: up → act → down (no gate), with biases; else gated (GeGLU/SwiGLU)
	MoE         *MoEConfig // non-nil ⇒ sparse mixture-of-experts FFN (Mixtral); nil ⇒ dense
	FirstKDense int        // GLM/DeepSeek (first_k_dense_replace): layers [0,FirstKDense) are plain dense MLPs, the rest MoE. 0 ⇒ every layer MoE (Mixtral/Qwen-MoE/Mellum).

	// Attention.
	QKVBias         bool             // additive bias on the q/k/v projections (Qwen2, GPT-2)
	OutBias         bool             // additive bias on the attention output projection (GPT-2)
	QKNorm          bool             // RMSNorm on Q and K per head before RoPE (Gemma3, Qwen3)
	LearnedPosEmbed bool             // GPT-2: add a learned position embedding and SKIP RoPE
	AttnScale       float64          // explicit q·k multiplier (resolved: query_pre_attn_scalar^-0.5 or 1/sqrt(headDim))
	SlidingWindow   int              // 0 = none
	layerIsGlobal   func(i int) bool // per-layer global(full) vs local(sliding) attention

	// RoPE (dual base for Gemma's local/global layers; equal bases = single-base).
	RoPELocalBase, RoPEGlobalBase float64
	// RotaryDim is the number of head dims RoPE rotates; 0 means the full
	// HeadDim. <HeadDim is partial rotary (Phi's partial_rotary_factor), where
	// the trailing dims pass through unrotated.
	RotaryDim int
	// MRopeSection is Qwen2.5-VL's m-RoPE head_dim/2 split over the (temporal,
	// height, width) position components; nil = plain scalar RoPE. For text tokens
	// the 3 components are equal so m-RoPE ≡ scalar RoPE; it only diverges over
	// image tokens (the 3D grid positions). (P5)
	MRopeSection []int
	// ropeScaling transforms the GLOBAL (full-attention) inv-freq table (Llama-3
	// llama3 / linear / yarn); nil = none. ropeScalingLocal does the same for the
	// LOCAL (sliding) table — usually nil even when the global table is scaled
	// (Mellum: YaRN on full layers, plain RoPE on sliding layers). Set by the
	// adapter, consumed when the tables are built.
	ropeScaling      *ropeScaling
	ropeScalingLocal *ropeScaling

	// Precomputed inverse-frequency tables (base + scaling baked in), built by
	// finalizeRoPE at resolve time so the forward pass never recomputes pow/scaling
	// per token. Local serves sliding layers, global the full-attention layers
	// (equal for single-base families).
	ropeInvFreqLocal  []float64
	ropeInvFreqGlobal []float64

	// Embedding / head.
	EmbedScale float64 // 0 or 1 = none; Gemma = sqrt(hidden)
	TiedLMHead bool    // tied embeddings as the LM head vs a separate lm_head

	// Output soft-capping (Gemma 2; 0 = none, which is Gemma 3).
	FinalLogitSoftcap float64
	AttnLogitSoftcap  float64

	// gemma4, when non-nil, carries Gemma 4's per-layer attention deltas (the
	// global/full layers differ from the local/sliding ones). nil for every
	// other family — they keep the uniform HeadDim/NumKVHeads/full-rotary path.
	gemma4 *gemma4Params

	// qwen35, when non-nil, marks a qwen3_5_moe hybrid: most layers are Gated
	// DeltaNet (linear attention with a recurrent matrix state), the rest softmax.
	// layerIsLinear picks which. The full-attention layers run the normal causal
	// path; the linear ones run the DeltaNet primitive. nil for every other family.
	qwen35        *qwen35Params
	layerIsLinear func(i int) bool

	// granite, when non-nil, marks a Granite-4.0-H hybrid: per-layer mixer is
	// Mamba-2 (layerIsMamba true) or softmax attention (false), every layer a
	// routed+shared MoE. Carries the Mamba-2 geometry + the Granite multipliers.
	// LogitScale (logits_scaling) divides the final logits; 0/1 = none.
	granite      *graniteParams
	layerIsMamba func(i int) bool
	LogitScale   float64

	// nemotron, when non-nil, marks a Nemotron-H single-op-block hybrid. nil for
	// every other family.
	nemotron *nemotronParams
}

// nemotronParams marks a Nemotron-H hybrid: a SINGLE-OP-per-block stack where each
// layer is exactly one of {mamba, attention, mlp} (blockKind, from layers_block_type)
// over a pre-norm residual — NOT the mixer+FFN block every other family uses. The
// mamba layers reuse the Mamba-2 mixer (same geometry fields as graniteParams); the
// attention layers are NoPE GQA (no RoPE); the mlp layers are non-gated relu². No
// Granite-style multipliers. forward_nemotron.go consumes this.
type nemotronParams struct {
	NHeads, HeadDim, DState, NGroups, DConv int     // Mamba-2 dims
	blockKind                               []uint8 // per layer: 0 mamba · 1 attention · 2 mlp
}

const (
	nemoMamba uint8 = iota
	nemoAttn
	nemoMLP
)

// graniteParams carries Granite-4.0-H's Mamba-2 mixer geometry (for the mamba
// layers; the attention layers use the uniform Architecture fields) and the three
// in-block scalar multipliers. The fourth Granite scalar, logits_scaling, lives on
// Architecture.LogitScale (it's applied at the shared head, not per layer).
type graniteParams struct {
	NHeads, HeadDim, DState, NGroups, DConv int     // Mamba-2 dims
	EmbMul, ResidMul                        float32 // embedding scale, residual-add scale (attention scale is Architecture.AttnScale)
}

// qwen35Params carries the Gated DeltaNet geometry for a qwen3_5_moe model's
// linear-attention layers (see docs/qwen3_5_moe.md). The softmax layers use the
// uniform Architecture attention fields; only the linear layers read these.
type qwen35Params struct {
	ConvKernel    int // depthwise causal conv width over [q;k;v] (linear_conv_kernel_dim)
	KeyHeadDim    int // per-head key/query dim (linear_key_head_dim)
	ValueHeadDim  int // per-head value dim (linear_value_head_dim)
	NumKeyHeads   int // linear_num_key_heads
	NumValueHeads int // linear_num_value_heads (GVA: a multiple of NumKeyHeads)
}

// gemma4Params describes how Gemma 4's global (full-attention) layers diverge
// from its local (sliding) layers. Local layers use Architecture.HeadDim /
// NumKVHeads and full rotary; global layers use these instead. Set by
// gemma4Architecture; consumed per-layer by the forward pass (runLayersGemma4).
type gemma4Params struct {
	GlobalHeadDim    int  // global head_dim (e.g. 512); local = Architecture.HeadDim (256)
	NumGlobalKVHeads int  // global KV-head count; local = Architecture.NumKVHeads. 0 ⇒ same as local
	GlobalRotaryDim  int  // rotated dims on global layers = partial_rotary_factor * GlobalHeadDim; local = full
	KVShared         bool // attention_k_eq_v: V reuses K's projection on global layers (12B; off for E2B)

	// E-model (E2B/E4B) extras; zero/empty on the dense 12B.
	SharedKVLayers          int   // last N layers carry no k/v and reuse an earlier layer's KV (per type)
	FFNPerLayer             []int // variable per-layer FFN width (else Architecture.IntermediateDim is uniform)
	HiddenSizePerLayerInput int   // PLE per-layer dim (256); 0 ⇒ no PLE
	VocabSizePerLayerInput  int   // PLE embedding-table vocab (== main vocab)
}

// headDimAt / kvHeadsAt / ffnAt give layer i's attention head_dim, KV-head count,
// and FFN width — Gemma 4's global layers diverge from local ones, and its FFN
// width varies per layer. For every other family these collapse to the uniform
// Architecture fields.
func (a *Architecture) headDimAt(i int) int {
	if a.gemma4 != nil && a.gemma4.GlobalHeadDim > 0 && a.isGlobalLayer(i) {
		return a.gemma4.GlobalHeadDim
	}
	return a.HeadDim
}

func (a *Architecture) kvHeadsAt(i int) int {
	if a.gemma4 != nil && a.gemma4.NumGlobalKVHeads > 0 && a.isGlobalLayer(i) {
		return a.gemma4.NumGlobalKVHeads
	}
	return a.NumKVHeads
}

func (a *Architecture) ffnAt(i int) int {
	if a.gemma4 != nil && i >= 0 && i < len(a.gemma4.FFNPerLayer) {
		return a.gemma4.FFNPerLayer[i]
	}
	return a.IntermediateDim
}

// MoEConfig describes a sparse mixture-of-experts FFN.
// A router scores all experts, the top-k run as gated MLPs, and their outputs
// combine weighted by the (renormalized) router probabilities. Mixtral:
// NumExperts=8, TopK=2, NormTopKProb=true.
type MoEConfig struct {
	NumExperts   int  // experts per layer (E)
	TopK         int  // experts evaluated per token (k)
	NormTopKProb bool // renormalize the top-k router weights to sum to 1 (Mixtral)
	// IntermediateDim is the per-expert FFN width. Mixtral's experts use the
	// model's intermediate_size; Mellum gives them a narrower moe_intermediate_size
	// (896 vs the vestigial 7168), so the expert width is tracked here rather than
	// read from arch.IntermediateDim.
	IntermediateDim int

	// SharedIntermediateDim is the FFN width of the always-on shared expert
	// (Qwen-MoE / Qwen2-MoE: shared_expert_intermediate_size). 0 means no shared
	// expert (Mixtral/Mellum). When set, every token additionally runs a gated
	// SwiGLU shared expert scaled by sigmoid(shared_gate·h), added to the routed
	// sum — unless SharedUngated (GLM/DeepSeek add it with no gate).
	SharedIntermediateDim int

	// DeepSeek-style routing (GLM-4.5/4.6). Defaults (false / 0) reproduce the
	// Mixtral/Qwen2-MoE softmax-topk path exactly.
	RouterSigmoid bool    // score experts with per-expert sigmoid(logit) instead of softmax; top-k weights are the chosen sigmoid scores (then NormTopKProb). e_score_correction_bias (LayerWeights.RouterBias) shifts the top-k SELECTION only.
	RoutedScale   float64 // routed_scaling_factor applied to the top-k weights (0 or 1 = no-op).
	SharedUngated bool    // GLM/DeepSeek add the shared expert with NO sigmoid gate (out += shared(h)); else the Qwen2-MoE sigmoid(SharedGate·h) gate.
}

// NormKind selects the normalization: RMSNorm (Llama/Gemma/Qwen/…) or
// LayerNorm (GPT-2/NeoX).
type NormKind int

const (
	NormRMS NormKind = iota
	NormLayer
)

// NormPlacement selects where norms sit relative to the residual adds. Pre2 is
// the Llama/Mistral/Qwen norm-before-each-sublayer; Sandwich4 is Gemma's
// pre+post norm on both attention and MLP.
type NormPlacement int

const (
	NormPre2 NormPlacement = iota
	NormSandwich4
)

// ActKind selects the MLP activation. GeluTanh = Gemma's GeGLU; SiLU = the
// SwiGLU used by Llama/Mistral/Qwen. Gelu/ReLU2/non-gated MLPs are later G's.
type ActKind int

const (
	ActGeluTanh ActKind = iota
	ActSiLU
	ActReLU2 // ReLU-squared (relu(x)²) — Nemotron-H's non-gated MLP
)

// isGlobalLayer reports whether layer i uses full (global) attention vs local
// (sliding-window). Defaults to global when no per-layer function is set.
func (a *Architecture) isGlobalLayer(i int) bool {
	if a.layerIsGlobal != nil {
		return a.layerIsGlobal(i)
	}
	return true
}

// isLinearLayer reports whether layer i is a Gated DeltaNet (linear-attention)
// layer rather than softmax attention — the qwen3_5_moe hybrid. False when no
// per-layer function is set (every non-hybrid family).
func (a *Architecture) isLinearLayer(i int) bool {
	return a.layerIsLinear != nil && a.layerIsLinear(i)
}

// isMambaLayer reports whether layer i is a Mamba-2 mixer rather than softmax
// attention — the Granite-4.0-H hybrid. False for every non-Granite family.
func (a *Architecture) isMambaLayer(i int) bool {
	return a.layerIsMamba != nil && a.layerIsMamba(i)
}

// ropeBase returns the RoPE base for layer i (Gemma uses a smaller base on the
// local layers; single-base families set both equal).
func (a *Architecture) ropeBase(i int) float64 {
	if a.isGlobalLayer(i) {
		return a.RoPEGlobalBase
	}
	return a.RoPELocalBase
}

// ropeMscale returns the attention_factor applied to the rotated q/k of layer i
// (YaRN's mscale; 1.0 for non-YaRN layers). Picks the global or local scaling
// per the layer's attention type.
func (a *Architecture) ropeMscale(i int) float64 {
	sc := a.ropeScaling
	if !a.isGlobalLayer(i) {
		sc = a.ropeScalingLocal
	}
	if sc != nil && sc.mscale != 0 {
		return sc.mscale
	}
	return 1
}

// rotaryDim returns the number of head dims RoPE rotates, defaulting to the
// full HeadDim when RotaryDim is unset.
func (a *Architecture) rotaryDim() int {
	if a.RotaryDim > 0 {
		return a.RotaryDim
	}
	return a.HeadDim
}

// finalizeRoPE precomputes the local/global inverse-frequency tables from the
// bases, rotary dim, and scaling. Called once by resolveArchitecture after the
// adapter populates the descriptor, so the forward pass reads a ready table.
func (a *Architecture) finalizeRoPE() {
	if a.LearnedPosEmbed || a.RoPEGlobalBase <= 0 {
		return // no RoPE (GPT-2 uses learned positions); tables stay nil
	}
	rd := a.rotaryDim()
	a.ropeInvFreqGlobal = computeInvFreq(a.RoPEGlobalBase, rd, a.ropeScaling)
	// Share the table only when the local layers use the SAME base AND scaling
	// (single-base, single-scaling families). Gemma differs by base; Mellum
	// differs by scaling (YaRN global vs plain local) at the same base.
	if a.RoPELocalBase == a.RoPEGlobalBase && a.ropeScalingLocal == a.ropeScaling {
		a.ropeInvFreqLocal = a.ropeInvFreqGlobal
	} else {
		a.ropeInvFreqLocal = computeInvFreq(a.RoPELocalBase, rd, a.ropeScalingLocal)
	}
}

// ropeInvFreq returns the precomputed inverse-frequency table for layer i.
func (a *Architecture) ropeInvFreq(i int) []float64 {
	if a.isGlobalLayer(i) {
		return a.ropeInvFreqGlobal
	}
	return a.ropeInvFreqLocal
}
