package arch

import (
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
	"github.com/townsendmerino/goinfer/internal/runtime/rope"
)

// Architecture is the resolved, family-agnostic description of a decoder LLM's structure. The generic forward pass
// (the program walkers, program_cpu.go and program_cpu_n.go) reads it; per-family adapters (registry.go) populate it from that family's
// config.json, so adding a family is descriptor population, not new forward code. Every field is consumed by the forward
// pass, which rejects descriptor values it does not implement rather than silently mis-running.
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
	// NormPlacementLinear, when non-nil, overrides NormPlacement on layers where isLinearLayer(i) is true: Olmo Hybrid's
	// DeltaNet layers use NormPre2 while its full-attention layers use NormPostOnly, two placements in one model. nil means
	// NormPlacement applies uniformly. See normPlacementAt.
	NormPlacementLinear *NormPlacement

	// MLP.
	Act         ActKind
	NonGatedMLP bool       // GPT-2: up → act → down (no gate), with biases; else gated (GeGLU/SwiGLU)
	MoE         *MoEConfig // non-nil ⇒ sparse mixture-of-experts FFN (Mixtral); nil ⇒ dense
	FirstKDense int        // GLM/DeepSeek (first_k_dense_replace): layers [0,FirstKDense) are plain dense MLPs, the rest MoE. 0 ⇒ every layer MoE (Mixtral/Qwen-MoE/Mellum).

	// Attention.
	QKVBias bool // additive bias on the q/k/v projections (Qwen2, GPT-2)
	OutBias bool // additive bias on the attention output projection (GPT-2)
	QKNorm  bool // RMSNorm on Q and K per head before RoPE (Gemma3, Qwen3)
	// AttnGate selects the head-wise output gate applied to the attention context before the output projection: GateNone
	// (default), GateSoftplus, or GateSigmoid (Spark-X2.5's g_proj). Laguna still gates via arch.laguna != nil, not this field.
	// Distinct from mlaParams.GateGranularity, which is MLA-only; this field is the hook of the generic (non-MLA) attention
	// forward (opOutGate; applyGateRow and applySigmoidGateRow).
	AttnGate GateKind

	// QKNormWhole (Olmo 3/Olmo Hybrid): when QKNorm is also set, normalize the whole projected q/k vector as one RMSNorm
	// (rows=1, dim=nHeads*headDim, one statistic) instead of per head (rows=nHeads, dim=headDim).
	QKNormWhole     bool
	LearnedPosEmbed bool // GPT-2: add a learned position embedding and SKIP RoPE
	// NoPositionEncoding (Olmo Hybrid): the family has no positional encoding on any layer, neither RoPE nor learned (the
	// released rope_parameters is {"rope_theta": null}). It is the fourth legitimate "no RoPEGlobalBase" case in
	// validateResolved's check, alongside LearnedPosEmbed, nemotron and mla, named so a family that forgot to read rope_theta
	// cannot look like one that deliberately has none.
	NoPositionEncoding bool
	AttnScale          float64 // explicit q·k multiplier (resolved: query_pre_attn_scalar^-0.5 or 1/sqrt(headDim))
	// AttnTempBeta/AttnTempOrigMaxPos (Ministral 3): a position-dependent scale on the query, applied after RoPE on every
	// layer: scale = 1 + beta·ln(1 + floor(pos/origMaxPos)); 0 means off. The same formula as llama4's attnTemp/floorScale,
	// but Llama 4 applies it instead of RoPE on NoPE layers only, so its own forward has no path for it on top of RoPE. scale
	// is 1 for pos < origMaxPos, so a test prompt shorter than origMaxPos exercises nothing: the tiny fixture's prompt is
	// deliberately longer.
	AttnTempBeta       float64
	AttnTempOrigMaxPos float64
	SlidingWindow      int // 0 = none
	// kvRowPadAlign is the CPU KV cache's row-pad alignment in bytes (kvRowPad): a row that is a multiple of it is padded by 64 B.
	// resolveArchitecture sets kvRowPadAlignDefault; 0 never pads (a test's unpadded arm).
	KVRowPadAlign int
	LayerIsGlobal func(i int) bool // per-layer global(full) vs local(sliding) attention
	LayerNoPE     func(i int) bool // per-layer NoPE: true ⇒ skip RoPE entirely on this layer (Cohere2 global layers; Llama-4 iRoPE style). nil ⇒ every layer ropes.

	// RoPE (dual base for Gemma's local/global layers; equal bases = single-base).
	RoPELocalBase, RoPEGlobalBase float64
	// RotaryDim is the number of head dims RoPE rotates; 0 means the full
	// HeadDim. <HeadDim is partial rotary (Phi's partial_rotary_factor), where
	// the trailing dims pass through unrotated.
	RotaryDim int
	// RotaryDimLocal is the rotated width on the local (sliding) layers when it differs from the global layers' (Laguna XS keys
	// partial_rotary_factor by layer type); 0 means both use RotaryDim. With RoPELocalBase and ropeScalingLocal it completes the
	// local/global RoPE triple.
	RotaryDimLocal int
	// MRopeSection is Qwen2.5-VL's m-RoPE head_dim/2 split over the (temporal, height, width) position components; nil means
	// plain scalar RoPE. For text tokens the three components are equal, so m-RoPE equals scalar RoPE; it diverges only over
	// image tokens (their 3D grid positions).
	MRopeSection []int
	// MRopeInterleaved selects Qwen3-VL's per-frequency-index component layout (rope.MRoPEComponentInterleaved) over Qwen2.5-VL's
	// contiguous-block one (rope.MRoPEComponent). False for every other m-RoPE family, including Qwen2.5-VL itself.
	MRopeInterleaved bool
	// ropeScaling transforms the GLOBAL (full-attention) inv-freq table (Llama-3
	// llama3 / linear / yarn); nil = none. ropeScalingLocal does the same for the
	// LOCAL (sliding) table — usually nil even when the global table is scaled
	// (Mellum: YaRN on full layers, plain RoPE on sliding layers). Set by the
	// adapter, consumed when the tables are built.
	RoPEScaling      *rope.Scaling
	RoPEScalingLocal *rope.Scaling
	// ropeInterleave selects GPT-J pairwise rotation (dims 2d,2d+1) over the NeoX split-half layout (dims d,d+half) in the
	// generic scalar RoPE path. (DeepSeek's MLA carries its own ropeInterleave on mlaParams.)
	RoPEInterleave bool

	// Precomputed inverse-frequency tables (base + scaling baked in), built by
	// finalizeRoPE at resolve time so the forward pass never recomputes pow/scaling
	// per token. Local serves sliding layers, global the full-attention layers
	// (equal for single-base families).
	RoPEInvFreqLocal  []float64
	RoPEInvFreqGlobal []float64

	// Embedding / head.
	EmbedScale float64 // 0 or 1 = none; Gemma = sqrt(hidden)
	TiedLMHead bool    // tied embeddings as the LM head vs a separate lm_head

	// Output soft-capping (Gemma 2; 0 = none, which is Gemma 3).
	FinalLogitSoftcap float64
	AttnLogitSoftcap  float64

	// gemma4, when non-nil, carries Gemma 4's per-layer attention deltas (the
	// global/full layers differ from the local/sliding ones). nil for every
	// other family — they keep the uniform HeadDim/NumKVHeads/full-rotary path.
	Gemma4 *archparts.Gemma4Params

	// laguna, when non-nil, carries Laguna's two departures from a Qwen2-MoE-shaped
	// decoder: softplus output gating before o_proj, and a per-layer QUERY head
	// count. nil for every other family. See lagunaParams.
	Laguna *archparts.LagunaParams

	// qwen35, when non-nil, marks a qwen3_5_moe hybrid: most layers are Gated
	// DeltaNet (linear attention with a recurrent matrix state), the rest softmax.
	// layerIsLinear picks which. The full-attention layers run the normal causal
	// path; the linear ones run the DeltaNet primitive. nil for every other family.
	Qwen35        *archparts.Qwen35Params
	LayerIsLinear func(i int) bool

	// granite, when non-nil, marks a Granite-4.0-H hybrid: per-layer mixer is
	// Mamba-2 (layerIsMamba true) or softmax attention (false), every layer a
	// routed+shared MoE. Carries the Mamba-2 geometry + the Granite multipliers.
	// LogitScale (logits_scaling) divides the final logits; 0/1 = none.
	Granite      *archparts.GraniteParams
	LayerIsMamba func(i int) bool

	// lfm2, when non-nil, marks an LFM2/LFM2.5 hybrid: every layer has a SwiGLU FFN, and its mixer is either a gated short
	// convolution (layerIsConv true) or GQA softmax attention with per-head RMSNorm on Q and K. The conv layers carry a rolling
	// per-channel window instead of a KV cache, so this is a cache-shape fact as well as a forward one.
	LFM2        *archparts.LFM2Params
	LayerIsConv func(i int) bool
	LogitScale  float64

	// nemotron, when non-nil, marks a Nemotron-H single-op-block hybrid. nil for
	// every other family.
	Nemotron *archparts.NemotronParams

	// mla, when non-nil, marks a DeepSeek Multi-head Latent Attention family
	// (deepseek_v2 / deepseek_v3). Attention reconstructs per-head K/V from a
	// cached low-rank latent (compressed-KV by construction) with decoupled RoPE;
	// the per-head dims are asymmetric (qk_head_dim ≠ v_head_dim), so it runs its
	// own forward path (forward_deepseek.go) rather than the generic attention.
	// nil for every other family.
	MLA *archparts.MLAParams

	// kda, when non-nil, marks Bailing Hybrid's (Ling 3.0) Kimi Delta Attention linear-attention layers, alternating with MLA
	// (mla, above) every LayerGroupSize-th layer; layerIsLinear picks which, the same hook qwen35 uses. A delta-rule
	// recurrence like Gated DeltaNet, but with a per-channel decay (one value per row of the state matrix) where Gated DeltaNet
	// has one scalar per head. Its layers are opKDA (Architecture.layerOps). nil for every other family.
	KDA *archparts.KDAParams

	// llama4, when non-nil, marks a Llama 4 text decoder (llama4_text): the iRoPE
	// stack — per-layer RoPE/NoPE interleave, parameter-free L2 QK-norm on the RoPE
	// layers, attention-temperature tuning on the NoPE layers, and a dense/MoE
	// interleave (top-1 sigmoid routing + an ungated shared expert). Own forward
	// (forward_llama4.go). nil for every other family.
	Llama4 *archparts.Llama4Params

	// gptoss, when non-nil, marks a gpt-oss sparse-MoE family: GQA with a learned per-head attention SINK in the softmax
	// denominator, alternating sliding/full attention (even layers sliding, window SlidingWindow), YaRN RoPE, and a clamped
	// interleaved-SwiGLU expert (gate·sigmoid(α·gate) · (up+1), clamped, with per-expert biases) + a router-logit bias. Own
	// forward (forward_gptoss.go); a resident backend admits it by declaring FeatAttnSink. nil for every other family.
	GPTOSS *archparts.GPTOSSParams

	// AttnSinks marks a generic-path family that carries a learned per-query-head
	// attention sink logit in the softmax denominator on every layer (LayerWeights.AttnSinks,
	// shape [NumHeads]; Flint's self_attn.softmax_offset / self_attn.sinks). Separate from
	// gptoss, which also sets per-head sinks alongside its clamped interleaved-SwiGLU MoE;
	// hasAttnSinks() unifies both for loader validation, residency, and FeatAttnSink.
	AttnSinks bool

	// Inkling2, when non-nil, marks an Inkling 2 decoder (inkling2, inkling2_loop). NumLayers counts layer visits
	// (Inkling2.Visits), each one opInkling2 holding the whole layer: attention-residual mixes, the relative-bias attention
	// with per-token sinks and short convolutions, the SITU MoE, and on the loop's later passes the message controllers.
	// Weights are per physical layer; Weights.Layers holds one entry per visit, sharing the physical layer's tensors.
	Inkling2 *archparts.Inkling2Params
	// Inkling2Head is the model-level tensors the forward reads besides the embedding, the final norm and the head.
	Inkling2Head *archparts.Inkling2HeadWeights
}

// AttnChunkStart returns the first key position a query at pos may attend on this layer, or 0
// when the layer is not chunked. Llama-4's RoPE layers are the only chunked case today.
//
// max() with the caller's existing window start rather than replacing it: a layer could in
// principle be both windowed and chunked, and taking the tighter of the two is the only reading
// that cannot silently widen attention.
func (a *Architecture) AttnChunkStart(layer, pos int) int {
	l4 := a.Llama4
	if l4 == nil || l4.ChunkSize <= 0 || layer >= len(l4.UseRope) || !l4.UseRope[layer] {
		return 0
	}
	return (pos / l4.ChunkSize) * l4.ChunkSize
}

// HeadDimAt / kvHeadsAt / ffnAt give layer i's attention head_dim, KV-head count,
// and FFN width — Gemma 4's global layers diverge from local ones, and its FFN
// width varies per layer. For every other family these collapse to the uniform
// Architecture fields.
func (a *Architecture) HeadDimAt(i int) int {
	if a.Gemma4 != nil && a.Gemma4.GlobalHeadDim > 0 && a.IsGlobalLayer(i) {
		return a.Gemma4.GlobalHeadDim
	}
	return a.HeadDim
}

// HeadsAt gives layer i's QUERY head count. Laguna's XS generations vary it by
// attention type; for every other family this collapses to Architecture.NumHeads.
// Note this is the query side only — kvHeadsAt is separate and stays uniform on
// Laguna, so the GQA group size (heads/kvHeads) is also per-layer.
func (a *Architecture) HeadsAt(i int) int {
	if a.Laguna != nil && i >= 0 && i < len(a.Laguna.HeadsPerLayer) {
		if n := a.Laguna.HeadsPerLayer[i]; n > 0 {
			return n
		}
	}
	return a.NumHeads
}

// MaxHeads is the largest per-layer query-head count, which is what attention
// scratch must be sized for. Equal to NumHeads unless a family varies it.
func (a *Architecture) MaxHeads() int {
	n := a.NumHeads
	if a.Laguna != nil {
		for _, h := range a.Laguna.HeadsPerLayer {
			if h > n {
				n = h
			}
		}
	}
	return n
}

func (a *Architecture) KVHeadsAt(i int) int {
	if a.Gemma4 != nil && a.Gemma4.NumGlobalKVHeads > 0 && a.IsGlobalLayer(i) {
		return a.Gemma4.NumGlobalKVHeads
	}
	return a.NumKVHeads
}

// KVDimAt returns the f32-equivalent elements layer i's K (or V, always equal width) cache stores per position it holds:
// zero for a layer with no position-indexed K/V (see hasNoAttentionKVAt), MLA's compressed latent width
// (KVLoRARank+QKRopeHeadDim) when a.mla != nil, else kvHeadsAt(i)*headDimAt(i). This is the width only; kvPositionsAt is
// the count.
func (a *Architecture) KVDimAt(i int) int {
	if a.HasNoAttentionKVAt(i) {
		return 0
	}
	if a.MLA != nil {
		return a.MLA.KVLoRARank + a.MLA.QKRopeHeadDim
	}
	return a.KVHeadsAt(i) * a.HeadDimAt(i)
}

// HasNoAttentionKVAt reports whether layer i holds no softmax-attention K/V array: the linear/mamba/conv mixers
// (isLinearLayer, isMambaLayer, isConvLayer) plus Nemotron's per-layer block kinds. Nemotron needs its own check because
// its mixer identity is per-layer data (nemotronParams.blockKind), not a closure registered at resolve time, so
// isMambaLayer never fires for its mamba layers; its mlp and moe blocks hold no K/V either.
func (a *Architecture) HasNoAttentionKVAt(i int) bool {
	if a.IsLinearLayer(i) || a.IsMambaLayer(i) || a.IsConvLayer(i) {
		return true
	}
	if a.Nemotron != nil && i < len(a.Nemotron.BlockKind) && a.Nemotron.BlockKind[i] != archparts.NemoAttn {
		return true
	}
	return false
}

// KVPositionsAt returns how many of ctx cache positions layer i's K/V must hold resident: ctx for an ordinary layer, or
// SlidingWindow once ctx exceeds it for a local layer under a sliding-window architecture (the ring never grows past its
// window). Meaningless but harmless for a layer kvDimAt prices at zero (the caller multiplies the two).
func (a *Architecture) KVPositionsAt(i, ctx int) int {
	if a.SlidingWindow > 0 && !a.IsGlobalLayer(i) && ctx > a.SlidingWindow {
		return a.SlidingWindow
	}
	return ctx
}

func (a *Architecture) FFNAt(i int) int {
	if a.Gemma4 != nil && i >= 0 && i < len(a.Gemma4.FFNPerLayer) {
		return a.Gemma4.FFNPerLayer[i]
	}
	return a.IntermediateDim
}

// NormKind selects the normalization: RMSNorm (Llama/Gemma/Qwen/…) or
// LayerNorm (GPT-2/NeoX).
type NormKind int

const (
	NormRMS NormKind = iota
	NormLayer
)

// String renders the norm kind for the capability matrix / logs.
func (n NormKind) String() string {
	switch n {
	case NormRMS:
		return "RMSNorm"
	case NormLayer:
		return "LayerNorm"
	default:
		return "unknown"
	}
}

// NormPlacement selects where norms sit relative to the residual adds. Pre2 is
// the Llama/Mistral/Qwen norm-before-each-sublayer; Sandwich4 is Gemma's
// pre+post norm on both attention and MLP; Parallel is Cohere/GPT-J's single
// shared input norm feeding BOTH sublayers into one residual add.
type NormPlacement int

const (
	NormPre2 NormPlacement = iota
	NormSandwich4
	// NormParallel: one input norm per layer; attention and MLP both read that
	// same normed input and their outputs sum into a single residual add
	// (residual = x + attn(norm(x)) + mlp(norm(x))). Cohere/Command-R, GPT-J,
	// Falcon. No pre-MLP norm, no post-sublayer norms.
	NormParallel
	// NormPostOnly: no pre-norm at all: attention and MLP read the raw residual stream, and the sublayer's output is
	// normalized before the residual add (residual = x + post_attn_norm(attn(x)); same for MLP). Olmo 3 has only
	// post_attention_layernorm / post_feedforward_layernorm. Unlike Sandwich4 there is no input norm to skip. Olmo Hybrid's
	// full-attention layers use this scheme and its DeltaNet layers NormPre2: see NormPlacementLinear, not a second enum value.
	NormPostOnly
)

// String renders the norm placement for the capability matrix / logs.
func (p NormPlacement) String() string {
	switch p {
	case NormPre2:
		return "pre-norm"
	case NormSandwich4:
		return "sandwich"
	case NormParallel:
		return "parallel"
	case NormPostOnly:
		return "post-only"
	default:
		return "unknown"
	}
}

// GateKind selects the generic (non-MLA) attention-output gate's activation — see
// Architecture.AttnGate's doc comment for the full story.
type GateKind int

const (
	GateNone     GateKind = iota // no gate (every family but Laguna/Spark-X2.5)
	GateSoftplus                 // reserved for a future generic-path user; Laguna's own gate is still keyed on arch.laguna, not this
	GateSigmoid                  // Spark-X2.5
)

// IsGlobalLayer reports whether layer i uses full (global) attention vs local
// (sliding-window). Defaults to global when no per-layer function is set.
func (a *Architecture) IsGlobalLayer(i int) bool {
	if a.LayerIsGlobal != nil {
		return a.LayerIsGlobal(i)
	}
	return true
}

// Gemma4KVSrcAt returns the layer whose K/V Gemma 4 layer i attends over: itself for layers before firstShared, else the
// last non-shared layer of the same attention type (global vs sliding) for a cross-layer-KV-shared tail layer
// (gemma4.SharedKVLayers). It duplicates the kvSrc closure of runLayersFromEmbedPLE on purpose, so the resident bridge
// does not touch the CPU path; TestGemma4KVSrcAt_matchesCPUClosure pins the two together, so change them together.
func (a *Architecture) Gemma4KVSrcAt(i int) int {
	if a.Gemma4 == nil {
		return i
	}
	firstShared := a.NumLayers - a.Gemma4.SharedKVLayers
	if i < firstShared {
		return i
	}
	lastSliding, lastGlobal := -1, -1
	for j := range firstShared {
		if a.IsGlobalLayer(j) {
			lastGlobal = j
		} else {
			lastSliding = j
		}
	}
	if a.IsGlobalLayer(i) {
		return lastGlobal
	}
	return lastSliding
}

// IsNoPELayer reports whether layer i skips RoPE entirely (NoPE — no positional
// encoding). Cohere2's every-Nth global-attention layer is NoPE while its sliding
// layers carry RoPE. False when no per-layer function is set (every layer ropes).
func (a *Architecture) IsNoPELayer(i int) bool {
	return a.LayerNoPE != nil && a.LayerNoPE(i)
}

// IsConvLayer reports whether layer i is an LFM2 gated short-convolution layer
// rather than softmax attention. False when no per-layer function is set (every
// non-LFM2 family), so callers can ask unconditionally.
func (a *Architecture) IsConvLayer(i int) bool {
	return a.LayerIsConv != nil && a.LayerIsConv(i)
}

// IsLinearLayer reports whether layer i is a Gated DeltaNet (linear-attention)
// layer rather than softmax attention — the qwen3_5_moe hybrid. False when no
// per-layer function is set (every non-hybrid family).
func (a *Architecture) IsLinearLayer(i int) bool {
	return a.LayerIsLinear != nil && a.LayerIsLinear(i)
}

// normPlacementAt resolves the NormPlacement for layer i, honoring
// NormPlacementLinear's per-layer override (Olmo Hybrid: DeltaNet layers use
// NormPre2 while its full-attention layers use NormPostOnly) when set. Returns
// NormPlacement unchanged for every family that leaves NormPlacementLinear nil.
func (a *Architecture) normPlacementAt(i int) NormPlacement {
	if a.NormPlacementLinear != nil && a.IsLinearLayer(i) {
		return *a.NormPlacementLinear
	}
	return a.NormPlacement
}

// IsMambaLayer reports whether layer i is a Mamba-2 mixer rather than softmax
// attention — the Granite-4.0-H hybrid. False for every non-Granite family.
func (a *Architecture) IsMambaLayer(i int) bool {
	return a.LayerIsMamba != nil && a.LayerIsMamba(i)
}

// RoPEBase returns the RoPE base for layer i (Gemma uses a smaller base on the
// local layers; single-base families set both equal).
func (a *Architecture) RoPEBase(i int) float64 {
	if a.IsGlobalLayer(i) {
		return a.RoPEGlobalBase
	}
	return a.RoPELocalBase
}

// RoPEMscale returns the attention_factor applied to the rotated q/k of layer i
// (YaRN's mscale; 1.0 for non-YaRN layers). Picks the global or local scaling
// per the layer's attention type.
func (a *Architecture) RoPEMscale(i int) float64 {
	sc := a.RoPEScaling
	if !a.IsGlobalLayer(i) {
		sc = a.RoPEScalingLocal
	}
	if sc != nil && sc.Mscale != 0 {
		return sc.Mscale
	}
	return 1
}

// EffectiveRotaryDim returns the number of head dims RoPE rotates, defaulting to the
// full HeadDim when RotaryDim is unset.
func (a *Architecture) EffectiveRotaryDim() int {
	if a.RotaryDim > 0 {
		return a.RotaryDim
	}
	return a.HeadDim
}

// FinalizeRoPE precomputes the local/global inverse-frequency tables from the
// bases, rotary dim, and scaling. Called once by resolveArchitecture after the
// adapter populates the descriptor, so the forward pass reads a ready table.
func (a *Architecture) FinalizeRoPE() {
	if a.LearnedPosEmbed || a.RoPEGlobalBase <= 0 {
		return // no RoPE (GPT-2 uses learned positions); tables stay nil
	}
	rd := a.EffectiveRotaryDim()
	a.RoPEInvFreqGlobal = rope.ComputeInvFreq(a.RoPEGlobalBase, rd, a.RoPEScaling)
	// Share the table only when the local layers use the SAME base AND scaling AND
	// rotated width (single-base, single-scaling families). Gemma differs by base;
	// Mellum differs by scaling (YaRN global vs plain local) at the same base;
	// Laguna differs by all three at once. The width matters because rope.Apply reads
	// the rotated half-width as len(invFreq) — a shorter local table IS partial
	// rotary on the local layers, with no other plumbing.
	rdLocal := rd
	if a.RotaryDimLocal > 0 {
		rdLocal = a.RotaryDimLocal
	}
	if a.RoPELocalBase == a.RoPEGlobalBase && a.RoPEScalingLocal == a.RoPEScaling && rdLocal == rd {
		a.RoPEInvFreqLocal = a.RoPEInvFreqGlobal
	} else {
		a.RoPEInvFreqLocal = rope.ComputeInvFreq(a.RoPELocalBase, rdLocal, a.RoPEScalingLocal)
	}
}

// RoPEInvFreq returns the precomputed inverse-frequency table for layer i.
func (a *Architecture) RoPEInvFreq(i int) []float64 {
	if a.IsGlobalLayer(i) {
		return a.RoPEInvFreqGlobal
	}
	return a.RoPEInvFreqLocal
}

// hasAttnOutputGate reports whether this architecture gates the attention context before o_proj: Laguna's softplus gate, or
// any family with AttnGate == GateSigmoid. One predicate for the forward's two dispatch sites (attention.go, forwardn.go)
// and the resident features derived from it (FeatAttnOutputGateSoftplus, FeatAttnOutputGateSigmoid).
func (a *Architecture) hasAttnOutputGate() bool {
	return a.Laguna != nil || a.AttnGate == GateSigmoid
}

// HasAttnSinks reports whether this architecture carries a learned per-query-head attention sink logit in the softmax
// denominator on every layer (LayerWeights.AttnSinks): either gpt-oss (a.gptoss != nil) or a generic-path sink family
// (a.AttnSinks, such as Flint).
func (a *Architecture) HasAttnSinks() bool { return a.GPTOSS != nil || a.AttnSinks }

// KVInt8OK reports whether NewCache may store this architecture's K/V as int8: a family whose attention is the batched kernel
// (attnOrder) on the generic loop, which reads K/V through the int8-aware path, MoE included. A family with its own forward loop
// (ownLoopFamily; gpt-oss's runs the batched kernel) keeps f32 until that loop is checked. What a quantized cache does to expert
// routing is a published quality reading, not a refusal.
func (a *Architecture) KVInt8OK() bool {
	return !a.OwnLoopFamily() && a.AttnOrder() == attnBatchedAcc64
}

// KVF16OK reports whether NewCache may store this architecture's KV at f16 (Options.KVQuant == "f16"): a family whose
// attention is the batched kernel (attnOrder) on the generic loop, MoE included, held to the f16 exactness gate (TestKVF16_F1Exactness). A family
// with its own forward loop keeps f32 until it is checked on its own.
func (a *Architecture) KVF16OK() bool {
	return !a.OwnLoopFamily() && a.AttnOrder() == attnBatchedAcc64
}

// RingLayerAt reports whether NewCache stores layer i as a ring: a sliding-window (local) layer of a family kvRingsOK admits.
// It decides both which layers enableRings makes rings and which reserve no append-forever capacity.
func (a *Architecture) RingLayerAt(i int) bool {
	return a.KVRingsOK() && a.SlidingWindow > 0 && !a.IsGlobalLayer(i)
}

// KVRingsOK reports whether NewCache may ring-buffer this architecture's sliding-window (local) layers, keeping only the W
// most recent positions: a family whose attention is the batched kernel (attnOrder), which reads a ring through
// attendBatchedHeads.
func (a *Architecture) KVRingsOK() bool {
	return a.AttnOrder() == attnBatchedAcc64
}

// The activations (program.ActKind).
const (
	ActGeluTanh = program.ActGeluTanh
	ActSiLU     = program.ActSiLU
	ActReLU2    = program.ActReLU2
	ActGelu     = program.ActGelu
	ActSitu     = program.ActSitu
)

// KVRowPad is the row pad, in elements, a CPU cache of this architecture takes at elemBytes per element: one 64 B line when every layer
// is a global attention layer of one KV width and the row is a multiple of kvRowPadAlign bytes; else 0.
// It is the one eligibility rule; KVCache.setRowPad and the plan's price both read it. A ring, an MLA latent or hybrid recurrent
// state declines it, as does a per-layer KV width.
func (a *Architecture) KVRowPad(elemBytes int) int {
	if a.KVRowPadAlign == 0 || a.MLA != nil || a.Gemma4 != nil || a.Qwen35 != nil || a.LFM2 != nil || a.Granite != nil || a.Nemotron != nil || a.KDA != nil || a.Inkling2 != nil {
		return 0
	}
	kvDim := a.NumKVHeads * a.HeadDim
	for l := range a.NumLayers {
		if a.RingLayerAt(l) || a.KVDimAt(l) != kvDim {
			return 0
		}
	}
	return archparts.KVRowPadElems(kvDim, elemBytes, a.KVRowPadAlign)
}
