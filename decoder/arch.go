package decoder

// Architecture is the resolved, family-agnostic description of a decoder LLM's
// structure. ONE generic forward pass (runLayers/forward/causalAttention/
// gatedMLP) reads it; per-family adapters (registry.go) populate it from that
// family's config.json. Gemma 3 is currently the only family — it's expressed
// as a descriptor here so adding Llama/Mistral/Qwen (multi-model-plan G2) is
// descriptor population, not new forward code.
//
// Every field below is consumed by the forward pass. Behavioural knobs that
// need new tensors (untied LM head, QKV bias, MoE) are deferred to later G-
// milestones; the forward pass rejects descriptor values it doesn't implement
// rather than silently mis-running.
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
}

// MoEConfig describes a sparse mixture-of-experts FFN (multi-model-plan G6).
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
	// sum.
	SharedIntermediateDim int
}

// NormKind selects the normalization. Only RMSNorm is implemented today;
// LayerNorm (GPT-2/NeoX) is multi-model-plan G5.
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
)

// isGlobalLayer reports whether layer i uses full (global) attention vs local
// (sliding-window). Defaults to global when no per-layer function is set.
func (a *Architecture) isGlobalLayer(i int) bool {
	if a.layerIsGlobal != nil {
		return a.layerIsGlobal(i)
	}
	return true
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
