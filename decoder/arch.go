package decoder

// Architecture is the resolved, family-agnostic description of a decoder LLM's structure. The generic forward pass
// (runLayers/forward/causalAttention/gatedMLP) reads it; per-family adapters (registry.go) populate it from that family's
// config.json, so adding a family is descriptor population, not new forward code. Every field is consumed by the forward
// pass, which rejects descriptor values it does not implement rather than silently mis-running.
type Architecture struct {
	// knobs is the owning model's per-model operator-knob snapshot (knobs.go); nil = read the live environment
	// (an Architecture built by hand in a test, never Loaded).
	knobs *knobSet

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
	// forward (applyAttnGate, attention.go/forwardn.go).
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
	SlidingWindow      int              // 0 = none
	layerIsGlobal      func(i int) bool // per-layer global(full) vs local(sliding) attention
	layerNoPE          func(i int) bool // per-layer NoPE: true ⇒ skip RoPE entirely on this layer (Cohere2 global layers; Llama-4 iRoPE style). nil ⇒ every layer ropes.

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
	// MRopeInterleaved selects Qwen3-VL's per-frequency-index component layout (mropeComponentInterleaved) over Qwen2.5-VL's
	// contiguous-block one (mropeComponent). False for every other m-RoPE family, including Qwen2.5-VL itself.
	MRopeInterleaved bool
	// ropeScaling transforms the GLOBAL (full-attention) inv-freq table (Llama-3
	// llama3 / linear / yarn); nil = none. ropeScalingLocal does the same for the
	// LOCAL (sliding) table — usually nil even when the global table is scaled
	// (Mellum: YaRN on full layers, plain RoPE on sliding layers). Set by the
	// adapter, consumed when the tables are built.
	ropeScaling      *ropeScaling
	ropeScalingLocal *ropeScaling
	// ropeInterleave selects GPT-J pairwise rotation (dims 2d,2d+1) over the NeoX split-half layout (dims d,d+half) in the
	// generic scalar RoPE path. (DeepSeek's MLA carries its own ropeInterleave on mlaParams.)
	ropeInterleave bool

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

	// laguna, when non-nil, carries Laguna's two departures from a Qwen2-MoE-shaped
	// decoder: softplus output gating before o_proj, and a per-layer QUERY head
	// count. nil for every other family. See lagunaParams.
	laguna *lagunaParams

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

	// lfm2, when non-nil, marks an LFM2/LFM2.5 hybrid: every layer has a SwiGLU FFN, and its mixer is either a gated short
	// convolution (layerIsConv true) or GQA softmax attention with per-head RMSNorm on Q and K. The conv layers carry a rolling
	// per-channel window instead of a KV cache, so this is a cache-shape fact as well as a forward one.
	lfm2        *lfm2Params
	layerIsConv func(i int) bool
	LogitScale  float64

	// nemotron, when non-nil, marks a Nemotron-H single-op-block hybrid. nil for
	// every other family.
	nemotron *nemotronParams

	// mla, when non-nil, marks a DeepSeek Multi-head Latent Attention family
	// (deepseek_v2 / deepseek_v3). Attention reconstructs per-head K/V from a
	// cached low-rank latent (compressed-KV by construction) with decoupled RoPE;
	// the per-head dims are asymmetric (qk_head_dim ≠ v_head_dim), so it runs its
	// own forward path (forward_deepseek.go) rather than the uniform causalAttention.
	// nil for every other family.
	mla *mlaParams

	// kda, when non-nil, marks Bailing Hybrid's (Ling 3.0) Kimi Delta Attention linear-attention layers, alternating with MLA
	// (mla, above) every LayerGroupSize-th layer; layerIsLinear picks which, the same hook qwen35 uses. A delta-rule
	// recurrence like Gated DeltaNet, but with a per-channel decay (one value per row of the state matrix) where Gated DeltaNet
	// has one scalar per head. Own forward (forward_bailing.go). nil for every other family.
	kda *kdaParams

	// llama4, when non-nil, marks a Llama 4 text decoder (llama4_text): the iRoPE
	// stack — per-layer RoPE/NoPE interleave, parameter-free L2 QK-norm on the RoPE
	// layers, attention-temperature tuning on the NoPE layers, and a dense/MoE
	// interleave (top-1 sigmoid routing + an ungated shared expert). Own forward
	// (forward_llama4.go). nil for every other family.
	llama4 *llama4Params

	// gptoss, when non-nil, marks a gpt-oss sparse-MoE family: GQA with a learned per-head attention SINK in the softmax
	// denominator, alternating sliding/full attention (even layers sliding, window SlidingWindow), YaRN RoPE, and a clamped
	// interleaved-SwiGLU expert (gate·sigmoid(α·gate) · (up+1), clamped, with per-expert biases) + a router-logit bias. Own
	// forward (forward_gptoss.go); a resident backend admits it by declaring FeatAttnSink. nil for every other family.
	gptoss *gptOssParams
}

// gptOssParams carries gpt-oss's expert-activation knobs. The sink weights and
// per-expert biases are per-layer (LayerWeights.AttnSinks / expertWeights.*Bias);
// the sliding-window pattern is Architecture.SlidingWindow + layerIsGlobal. forward_gptoss.go reads this.
type gptOssParams struct {
	SwigluAlpha float64 // sigmoid gain in gate·sigmoid(α·gate) (1.702)
	SwigluLimit float64 // clamp bound: gate ≤ limit, up ∈ [-limit, limit] (7.0)
}

// llama4Params carries Llama 4's per-layer iRoPE deltas. useRope[i]/isMoE[i] select layer
// i's attention (RoPE+L2-QK-norm vs NoPE+attn-temperature) and FFN (dense vs MoE). The
// dense layers use Architecture.IntermediateDim (intermediate_size_mlp); the routed +
// shared experts use MoEConfig.IntermediateDim (intermediate_size). forward_llama4.go reads this.
type llama4Params struct {
	// chunkSize is attention_chunk_size (8192 on Scout/Maverick): the RoPE layers use a block-diagonal chunked mask, so a query
	// at position p attends only to keys in its own chunk, [(p/C)*C, p]. NoPE layers stay full-causal. 0 means no chunking.
	// Below C chunked equals full causal, so a short-sequence parity gate cannot see a missing chunk mask.
	chunkSize  int
	useRope    []bool  // per layer: RoPE (true) vs NoPE (false) — from no_rope_layers
	isMoE      []bool  // per layer: MoE (true) vs dense (false) — from moe_layers
	useQKNorm  bool    // parameter-free L2 (RMS-over-head-dim) QK-norm on the RoPE layers
	attnTemp   bool    // attention-temperature tuning on the NoPE layers
	floorScale float64 // attn-temp: log1p(floor((pos+1)/floorScale))·attnScale + 1
	attnScale  float64
}

// nemotronParams marks a Nemotron-H hybrid: a SINGLE-OP-per-block stack where each
// layer is exactly one of {mamba, attention, mlp} (blockKind, from layers_block_type)
// over a pre-norm residual — NOT the mixer+FFN block every other family uses. The
// mamba layers reuse the Mamba-2 mixer (same geometry fields as graniteParams); the
// attention layers are NoPE GQA (no RoPE); the mlp layers are non-gated relu². No
// Granite-style multipliers. forward_nemotron.go consumes this.
type nemotronParams struct {
	NHeads, HeadDim, DState, NGroups, DConv int     // Mamba-2 dims
	blockKind                               []uint8 // per layer: 0 mamba · 1 attention · 2 mlp · 3 moe
}

const (
	nemoMamba uint8 = iota
	nemoAttn
	nemoMLP
	nemoMoE
)

// mlaParams carries DeepSeek Multi-head Latent Attention geometry. The cached state is the compressed latent
// [KVLoRARank + QKRopeHeadDim] per position; per-head K/V are rebuilt from it each step via kv_b_proj. QLoRARank 0 means a
// direct q_proj instead of the q_a/q_b LoRA bottleneck. forward_deepseek.go consumes this.
type mlaParams struct {
	QLoRARank      int  // q_a_proj bottleneck width; 0 ⇒ direct q_proj (no q-LoRA)
	KVLoRARank     int  // compressed KV latent width (the cached payload, minus the rope key)
	QKNopeHeadDim  int  // per-head Q/K dims WITHOUT RoPE
	QKRopeHeadDim  int  // per-head Q/K dims carrying decoupled RoPE (one K shared across heads)
	VHeadDim       int  // per-head V width (≠ QKNopeHeadDim+QKRopeHeadDim)
	ropeInterleave bool // GPT-J pairwise (true, V3 default) vs NeoX half-split RoPE on the rope dims
	// AttnPrefix/DenseSuffix override the tensor-name prefix and output-projection suffix; "" means "self_attn"/"o_proj.weight".
	// Bailing Hybrid uses "attention"/"dense.weight": its MLA and KDA mixers both sit under self.attention, and its MLA names
	// its output projection dense.
	AttnPrefix, DenseSuffix string
	// GateGranularity ("" | "head_wise" | "element_wise"): Bailing Hybrid's optional per-head or per-element sigmoid output gate
	// (g_proj) on the attention context before the output projection, Laguna's gate structure with sigmoid in place of
	// softplus. "" (every DeepSeek family) means no gate.
	GateGranularity string
}

// kdaParams carries Bailing Hybrid's (Ling 3.0) Kimi Delta Attention geometry for the linear-attention layers; the MLA
// layers use mlaParams. HeadDim/NumHeads are shared by q/k/v (no GVA). NoLora selects a single f_proj/g_proj linear per
// gate (Ling-3.0-tiny) over a LoRA'd a/b-split pair; the LoRA'd path is not implemented (see kdaArchitecture).
type kdaParams struct {
	HeadDim, NumHeads, ConvKernel int
	NoLora                        bool
	SafeGate                      bool
	LowerBound                    float64
}

// attnChunkStart returns the first key position a query at pos may attend on this layer, or 0
// when the layer is not chunked. Llama-4's RoPE layers are the only chunked case today.
//
// max() with the caller's existing window start rather than replacing it: a layer could in
// principle be both windowed and chunked, and taking the tighter of the two is the only reading
// that cannot silently widen attention.
func (a *Architecture) attnChunkStart(layer, pos int) int {
	l4 := a.llama4
	if l4 == nil || l4.chunkSize <= 0 || layer >= len(l4.useRope) || !l4.useRope[layer] {
		return 0
	}
	return (pos / l4.chunkSize) * l4.chunkSize
}

// qkHeadDim is the query·key dot-product width: the no-rope dims plus the rope dims.
func (p *mlaParams) qkHeadDim() int { return p.QKNopeHeadDim + p.QKRopeHeadDim }

// lfm2Params carries the gated short-convolution geometry for an LFM2/LFM2.5 model's conv layers; the attention layers use
// the uniform Architecture fields. The block is in_proj -> split into three ConvDim gates (B, C, x) -> Bx = B*x ->
// depthwise causal conv with NO activation -> y = C*conv -> out_proj; ConvLCache taps, no bias. The missing activation
// differs from Mamba-2's SiLU conv (upstream passes activation=None), and adding one gives a plausible, wrong model.
type lfm2Params struct {
	ConvDim    int // channels the conv block operates on (hidden_size on released weights)
	ConvLCache int // kernel width / rolling-window depth (3)
}

// graniteParams carries Granite-4.0-H's Mamba-2 mixer geometry (for the mamba layers; the attention layers use the
// uniform Architecture fields) and the three in-block scalar multipliers. The fourth Granite scalar, logits_scaling, lives
// on Architecture.LogitScale (it is applied at the shared head, not per layer).
type graniteParams struct {
	NHeads, HeadDim, DState, NGroups, DConv int     // Mamba-2 dims
	EmbMul, ResidMul                        float32 // embedding scale, residual-add scale (attention scale is Architecture.AttnScale)
}

// qwen35Params carries the Gated DeltaNet geometry for a qwen3_5_moe model's linear-attention layers
// (docs/completed/qwen3_5_moe.md). The softmax layers use the uniform Architecture attention fields; only the linear layers
// read these.
type qwen35Params struct {
	ConvKernel    int // depthwise causal conv width over [q;k;v] (linear_conv_kernel_dim)
	KeyHeadDim    int // per-head key/query dim (linear_key_head_dim)
	ValueHeadDim  int // per-head value dim (linear_value_head_dim)
	NumKeyHeads   int // linear_num_key_heads
	NumValueHeads int // linear_num_value_heads (GVA: a multiple of NumKeyHeads)

	// AttnGate: this family's full-attention (non-DeltaNet) layers use qwen3.5's double-width q_proj, [query ‖ gate] per head
	// interleaved, with the attention context multiplied by sigmoid(gate) before o_proj. False for Olmo Hybrid, whose
	// full-attention layers are olmo3's plain scheme (ordinary q_proj, no gate): a resident backend must not assume every
	// qwen35Params family has a q gate.
	AttnGate bool
	// FusedDeltaNetProj: qwen3_next's checkpoint fuses in_proj_qkv/z/b/a into in_proj_qkvz/in_proj_ba (qwen3_5_moe stores four
	// separate tensors). loadQwen35Attn splits the fused tensors into the same four deltaNetWeights fields, so the rest of the
	// pipeline (forward, gguf, serialize) is untouched.
	FusedDeltaNetProj bool
	// SeparateQKVProj (Olmo Hybrid): the checkpoint stores q_proj/k_proj/v_proj as three independent tensors. loadQwen35Attn
	// concatenates them at load into the inProjQKV layout, so gatedDeltaNetStep and every downstream consumer are untouched.
	SeparateQKVProj bool
	// NegEigval (Olmo Hybrid): linear_allow_neg_eigval (default true on the release) widens the write-gate beta from sigmoid's
	// [0,1) to [0,2) after the sigmoid. Same recurrence, wider gate range.
	NegEigval bool
	// SeparateConv (Olmo Hybrid): the checkpoint stores the depthwise causal conv as three tensors, q_conv1d/k_conv1d/v_conv1d,
	// split at the q/k/v channel boundaries (keyDim, keyDim, valueDim rows). loadQwen35Attn concatenates them in q,k,v order
	// into the [convDim,1,K] layout gatedDeltaNetStep expects. The modeling code alone shows one combined conv1d; the split is
	// visible only in a real checkpoint's tensor names and shapes, so check a real header.
	SeparateConv bool
	// DeltaNetNormSuffix/DeltaNetOutProjSuffix override the DeltaNet gated-RMSNorm and out_proj tensor suffixes; "" means
	// "linear_attn.norm.weight" / "linear_attn.out_proj.weight". Olmo Hybrid names them o_norm/o_proj: same math, different
	// on-disk tensor name.
	DeltaNetNormSuffix, DeltaNetOutProjSuffix string
	// ONormEps overrides the DeltaNet output gated-RMSNorm's epsilon; 0 means the family's NormEps. Olmo Hybrid hardcodes
	// eps=1e-5 for this one norm regardless of config (FLA's FusedRMSNormGated default), unlike its rms_norm_eps=1e-6
	// elsewhere: reusing NormEps here is off by an order of magnitude on the norm most sensitive to it (the recurrence's output
	// gate).
	ONormEps float64
	// PlainFullAttn (Olmo Hybrid): the full-attention layers are a plain olmo3-shaped self-attention (single-width q/k/v/o,
	// optional whole-vector QK-norm, generic RoPE/NoPE), not qwen3_5's double-width gated attention. false routes non-linear
	// layers through loadQwen35Attn / runLayersQwen35's bespoke qwen3.5 attention; true routes them through the shared generic
	// loader path and causalAttention, reusing olmo3's adapter.
	PlainFullAttn bool
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

	// PadTokenID is the id runLayersGemma4FromEmbed's PLE token-identity lookup uses at a multimodal (image/video/audio)
	// position in place of the caller's real token id, matching the HF multimodal forward's pad substitution
	// (docs/multimodal.md).
	PadTokenID int
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

// lagunaParams describes how Laguna (poolside) departs from an otherwise Qwen2-MoE-shaped decoder; both fields are read
// straight from the released configs (docs/completed/task-laguna.md).
type lagunaParams struct {
	// HeadsPerLayer is num_attention_heads_per_layer: layer i's QUERY head count,
	// which varies with the layer's attention type on the XS generations (48 on
	// full_attention, 64 on sliding_attention). nil ⇒ uniform Architecture.NumHeads
	// (M.1, whose config omits the field). KV heads stay uniform at 8 either way, so
	// the GQA group size — not just the head count — changes per layer.
	HeadsPerLayer []int
	// GatePerHead selects the gate's granularity: true means g_proj emits one gate per head, broadcast across head_dim (config
	// gating "per-head"); false means one gate per (head, head_dim) channel (gating true or "per-element"). It mirrors the
	// vendor's gate_per_head = (gating == "per-head").
	GatePerHead bool
}

// headsAt gives layer i's QUERY head count. Laguna's XS generations vary it by
// attention type; for every other family this collapses to Architecture.NumHeads.
// Note this is the query side only — kvHeadsAt is separate and stays uniform on
// Laguna, so the GQA group size (heads/kvHeads) is also per-layer.
func (a *Architecture) headsAt(i int) int {
	if a.laguna != nil && i >= 0 && i < len(a.laguna.HeadsPerLayer) {
		if n := a.laguna.HeadsPerLayer[i]; n > 0 {
			return n
		}
	}
	return a.NumHeads
}

// maxHeads is the largest per-layer query-head count, which is what attention
// scratch must be sized for. Equal to NumHeads unless a family varies it.
func (a *Architecture) maxHeads() int {
	n := a.NumHeads
	if a.laguna != nil {
		for _, h := range a.laguna.HeadsPerLayer {
			if h > n {
				n = h
			}
		}
	}
	return n
}

func (a *Architecture) kvHeadsAt(i int) int {
	if a.gemma4 != nil && a.gemma4.NumGlobalKVHeads > 0 && a.isGlobalLayer(i) {
		return a.gemma4.NumGlobalKVHeads
	}
	return a.NumKVHeads
}

// kvDimAt returns the f32-equivalent elements layer i's K (or V, always equal width) cache stores per position it holds:
// zero for a layer with no position-indexed K/V (see hasNoAttentionKVAt), MLA's compressed latent width
// (KVLoRARank+QKRopeHeadDim) when a.mla != nil, else kvHeadsAt(i)*headDimAt(i). This is the width only; kvPositionsAt is
// the count.
func (a *Architecture) kvDimAt(i int) int {
	if a.hasNoAttentionKVAt(i) {
		return 0
	}
	if a.mla != nil {
		return a.mla.KVLoRARank + a.mla.QKRopeHeadDim
	}
	return a.kvHeadsAt(i) * a.headDimAt(i)
}

// hasNoAttentionKVAt reports whether layer i holds no softmax-attention K/V array: the linear/mamba/conv mixers
// (isLinearLayer, isMambaLayer, isConvLayer) plus Nemotron's per-layer block kinds. Nemotron needs its own check because
// its mixer identity is per-layer data (nemotronParams.blockKind), not a closure registered at resolve time, so
// isMambaLayer never fires for its mamba layers; its mlp and moe blocks hold no K/V either.
func (a *Architecture) hasNoAttentionKVAt(i int) bool {
	if a.isLinearLayer(i) || a.isMambaLayer(i) || a.isConvLayer(i) {
		return true
	}
	if a.nemotron != nil && i < len(a.nemotron.blockKind) && a.nemotron.blockKind[i] != nemoAttn {
		return true
	}
	return false
}

// kvPositionsAt returns how many of ctx cache positions layer i's K/V must hold resident: ctx for an ordinary layer, or
// SlidingWindow once ctx exceeds it for a local layer under a sliding-window architecture (the ring never grows past its
// window). Meaningless but harmless for a layer kvDimAt prices at zero (the caller multiplies the two).
func (a *Architecture) kvPositionsAt(i, ctx int) int {
	if a.SlidingWindow > 0 && !a.isGlobalLayer(i) && ctx > a.SlidingWindow {
		return a.SlidingWindow
	}
	return ctx
}

// kvBytesForCtx sums the real per-layer KV cost at ctx positions over the whole architecture: kvDimAt (width) times
// kvPositionsAt (count). Used by fitguard.go's load-time host-RAM guard and prefill_budget.go's request-time guard, both
// Config-only via resolveArchitecture. Not used by fitplan.go's device-VRAM Plan(), which needs only the width and
// deliberately skips the sliding-window count cap (see kvBytesPerPositionAllLayers there). Do not replace it with a flat
// NumLayers*NumKVHeads*headDim formula: that overprices hybrid, sliding-window and MLA models, which cache far less.
func kvBytesForCtx(arch *Architecture, ctx int, kvF16, kvI8 bool) int64 {
	if arch == nil || ctx <= 0 || arch.NumLayers <= 0 {
		return 0
	}
	perElem := 4.0
	switch {
	case kvI8:
		perElem = 1.125 // int8 payload + per-row f32 scale
	case kvF16:
		perElem = 2
	}
	var total float64
	for l := 0; l < arch.NumLayers; l++ {
		dim := arch.kvDimAt(l)
		if dim == 0 {
			continue
		}
		positions := arch.kvPositionsAt(l, ctx)
		// A local layer's f32 ring keeps a mirror copy of its window once wrapped (decode reads it in place), so it holds 2*W
		// positions, not W; the int8 ring has no mirror. Priced here so the fit guard counts what is resident.
		if !kvI8 && arch.SlidingWindow > 0 && !arch.isGlobalLayer(l) && ctx > arch.SlidingWindow {
			positions += arch.SlidingWindow
		}
		total += 2 * perElem * float64(dim) * float64(positions) // ×2 for K and V
	}
	return int64(total)
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
	// IntermediateDim is the per-expert FFN width. Mixtral's experts use the model's intermediate_size; Mellum gives them a
	// narrower moe_intermediate_size, so the expert width is tracked here, not read from arch.IntermediateDim.
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

	// Group-limited routing (DeepSeek-V3 noaux_tc). Experts are partitioned into
	// NGroup contiguous groups; each group is scored by its top-2 selection scores
	// summed, the top TopkGroup groups are kept, and the per-token top-k is taken
	// only among experts in those groups. NGroup ≤ 1 (GLM, V2-Lite) ⇒ no grouping,
	// the plain global top-k.
	NGroup    int
	TopkGroup int

	// Gemma 4 26B-A4B router extras (Gemma4TextRouter; false for every other MoE family). Selection is still softmax over all
	// experts, top-k, renorm (NormTopKProb is unconditionally true), so routeExperts applies; these two flags add the parts it
	// lacks (docs/completed/task-gemma4-moe.md).
	RouterPreNorm  bool // before the router projection, the hidden state passes a WEIGHTLESS RMSNorm, a learned [hidden] scale (LayerWeights.RouterScale), and a hidden^-0.5 constant — not a bare Linear on the raw hidden state.
	PerExpertScale bool // the renormalized top-k weights are multiplied by a learned per-expert scale (LayerWeights.PerExpertScale), indexed by the chosen experts.
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

// ActKind selects the MLP activation. GeluTanh = Gemma's GeGLU; SiLU = the
// SwiGLU used by Llama/Mistral/Qwen. Gelu/ReLU2/non-gated MLPs are later G's.
type ActKind int

const (
	ActGeluTanh ActKind = iota
	ActSiLU
	ActReLU2 // ReLU-squared (relu(x)²) — Nemotron-H's non-gated MLP
	// ActGelu is the EXACT erf GELU — HF's "gelu", a different function from
	// "gelu_new"/"gelu_pytorch_tanh" (ActGeluTanh) rather than a spelling of it.
	// Appended deliberately: GatedActResident passes this ordinal straight to a CUDA
	// kernel (0 = GELU-tanh, 1 = SiLU), so the existing values cannot be renumbered.
	// Only non-gated archs reach it today, and GatedActResident is documented as
	// meaningless for those.
	ActGelu
)

// String renders the activation for the capability matrix / logs. The gated vs
// non-gated distinction is rendered by the matrix from Architecture.NonGatedMLP.
func (a ActKind) String() string {
	switch a {
	case ActGeluTanh:
		return "GELU-tanh"
	case ActSiLU:
		return "SiLU"
	case ActReLU2:
		return "ReLU²"
	case ActGelu:
		return "GELU"
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

// isGlobalLayer reports whether layer i uses full (global) attention vs local
// (sliding-window). Defaults to global when no per-layer function is set.
func (a *Architecture) isGlobalLayer(i int) bool {
	if a.layerIsGlobal != nil {
		return a.layerIsGlobal(i)
	}
	return true
}

// gemma4KVSrcAt returns the layer whose K/V Gemma 4 layer i attends over: itself for layers before firstShared, else the
// last non-shared layer of the same attention type (global vs sliding) for a cross-layer-KV-shared tail layer
// (gemma4.SharedKVLayers). It duplicates the kvSrc closure of runLayersGemma4FromEmbed on purpose, so the resident bridge
// does not touch the CPU path; TestGemma4KVSrcAt_matchesCPUClosure pins the two together, so change them together.
func (a *Architecture) gemma4KVSrcAt(i int) int {
	if a.gemma4 == nil {
		return i
	}
	firstShared := a.NumLayers - a.gemma4.SharedKVLayers
	if i < firstShared {
		return i
	}
	lastSliding, lastGlobal := -1, -1
	for j := range firstShared {
		if a.isGlobalLayer(j) {
			lastGlobal = j
		} else {
			lastSliding = j
		}
	}
	if a.isGlobalLayer(i) {
		return lastGlobal
	}
	return lastSliding
}

// isNoPELayer reports whether layer i skips RoPE entirely (NoPE — no positional
// encoding). Cohere2's every-Nth global-attention layer is NoPE while its sliding
// layers carry RoPE. False when no per-layer function is set (every layer ropes).
func (a *Architecture) isNoPELayer(i int) bool {
	return a.layerNoPE != nil && a.layerNoPE(i)
}

// isConvLayer reports whether layer i is an LFM2 gated short-convolution layer
// rather than softmax attention. False when no per-layer function is set (every
// non-LFM2 family), so callers can ask unconditionally.
func (a *Architecture) isConvLayer(i int) bool {
	return a.layerIsConv != nil && a.layerIsConv(i)
}

// isLinearLayer reports whether layer i is a Gated DeltaNet (linear-attention)
// layer rather than softmax attention — the qwen3_5_moe hybrid. False when no
// per-layer function is set (every non-hybrid family).
func (a *Architecture) isLinearLayer(i int) bool {
	return a.layerIsLinear != nil && a.layerIsLinear(i)
}

// normPlacementAt resolves the NormPlacement for layer i, honoring
// NormPlacementLinear's per-layer override (Olmo Hybrid: DeltaNet layers use
// NormPre2 while its full-attention layers use NormPostOnly) when set. Returns
// NormPlacement unchanged for every family that leaves NormPlacementLinear nil.
func (a *Architecture) normPlacementAt(i int) NormPlacement {
	if a.NormPlacementLinear != nil && a.isLinearLayer(i) {
		return *a.NormPlacementLinear
	}
	return a.NormPlacement
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
	// Share the table only when the local layers use the SAME base AND scaling AND
	// rotated width (single-base, single-scaling families). Gemma differs by base;
	// Mellum differs by scaling (YaRN global vs plain local) at the same base;
	// Laguna differs by all three at once. The width matters because applyRoPE reads
	// the rotated half-width as len(invFreq) — a shorter local table IS partial
	// rotary on the local layers, with no other plumbing.
	rdLocal := rd
	if a.RotaryDimLocal > 0 {
		rdLocal = a.RotaryDimLocal
	}
	if a.RoPELocalBase == a.RoPEGlobalBase && a.ropeScalingLocal == a.ropeScaling && rdLocal == rd {
		a.ropeInvFreqLocal = a.ropeInvFreqGlobal
	} else {
		a.ropeInvFreqLocal = computeInvFreq(a.RoPELocalBase, rdLocal, a.ropeScalingLocal)
	}
}

// ropeInvFreq returns the precomputed inverse-frequency table for layer i.
func (a *Architecture) ropeInvFreq(i int) []float64 {
	if a.isGlobalLayer(i) {
		return a.ropeInvFreqGlobal
	}
	return a.ropeInvFreqLocal
}

// ownForwardFamily is one family whose per-token layer loop is its own, not the generic one.
type ownForwardFamily struct {
	// Name is the descriptor name, for reports and for the test that keeps this table complete.
	Name string
	// is reports whether an Architecture is this family.
	is func(*Architecture) bool
	// run is the family's layer loop. Same signature as the generic path.
	run func(*Model, int, *KVCache) ([]float32, error)
	// Captures is true when this family's loop calls cache.captureResidual, i.e. ForwardCapture's
	// hidden-state seam is wired for it. A family is wired only when BOTH its loop captures and
	// this is set, so a half-wired one fails loudly instead of returning nil rows.
	Captures bool
	// Recurrent is true when this family carries state mutated IN PLACE per token — a conv window,
	// an SSM state, a linear-attention state — that KVCache.TruncateTo cannot rewind. It is the
	// arch-side view of KVCache.hasRecurrentState(): the cache knows once it exists, this knows
	// from the descriptor, and speculative rollback has to decide before either is built.
	Recurrent bool
	// KVInt8 is true when this family's loop reads K/V through the int8-aware attention path, so NewCache may store them as int8
	// (Options.KVQuant == "i8"). KVRings is true when it can read a ring-buffered sliding-window layer. Both default to false: a
	// loop that sizes its scores from len(cache.Keys(layer)) indexes past an int8 cache's empty f32 store on the first decode
	// step (TestLFM2_kvQuantI8_generates). See kvInt8OK / kvRingsOK.
	KVInt8, KVRings bool
}

// ownForwards is the one list of families that do not use the generic layer loop, so "runLayers dispatches here" and "the
// batched path must not touch this" are the same fact. A family added here is excluded from the batched path by
// construction, and TestOwnForward_tableNamesEveryFamilyForward fails if a runLayersXxx is written that is not listed.
// Keep it a single table: two separate lists once disagreed and sent LFM2 prompts of two or more tokens through the dense
// attention stack (docs/code-notes/decoder.md#ownForwards).
var ownForwards = []ownForwardFamily{
	// Fields: Name, is, run, Captures, Recurrent, KVInt8, KVRings.
	// Gemma 4: per-layer head_dim, KV-sharing, PLE.
	{"gemma4", func(a *Architecture) bool { return a.gemma4 != nil }, (*Model).runLayersGemma4, true, false, false, false},
	// qwen3_5_moe: Gated DeltaNet / softmax hybrid.
	{"qwen3_5_moe", func(a *Architecture) bool { return a.qwen35 != nil }, (*Model).runLayersQwen35, true, true, false, false},
	// lfm2: gated short-conv / softmax hybrid.
	{"lfm2", func(a *Architecture) bool { return a.lfm2 != nil }, (*Model).runLayersLFM2, false, true, false, false},
	// granitemoehybrid: Mamba-2 / softmax hybrid.
	{"granitemoehybrid", func(a *Architecture) bool { return a.granite != nil }, (*Model).runLayersGranite, false, true, false, false},
	// nemotron_h: single-op-per-block hybrid.
	{"nemotron_h", func(a *Architecture) bool { return a.nemotron != nil }, (*Model).runLayersNemotron, false, true, false, false},
	// bailing_hybrid: MLA / KDA hybrid (Ling 3.0). MUST precede deepseek_v2/v3 below —
	// bailingHybridArchitecture sets BOTH kda and mla (its MLA layers reuse mlaAttention
	// directly), so a.mla != nil alone would also match it and misroute to runLayersDeepseek,
	// which has no KDA branch at all.
	{"bailing_hybrid", func(a *Architecture) bool { return a.kda != nil }, (*Model).runLayersBailingHybrid, false, true, false, false}, // Recurrent: KDA state mutates in place per token
	// deepseek_v2/v3: Multi-head Latent Attention.
	{"deepseek_v2/v3", func(a *Architecture) bool { return a.mla != nil }, (*Model).runLayersDeepseek, false, false, false, false},
	// llama4_text: iRoPE (per-layer RoPE/NoPE + L2 QK-norm + attn-temp).
	{"llama4_text", func(a *Architecture) bool { return a.llama4 != nil }, (*Model).runLayersLlama4, false, false, false, false},
	// gpt-oss: per-head attention sinks + clamped-SwiGLU MoE.
	{"gpt-oss", func(a *Architecture) bool { return a.gptoss != nil }, (*Model).runLayersGptOss, true, false, false, false},
}

// hasAttnOutputGate reports whether this architecture gates the attention context before o_proj: Laguna's softplus gate, or
// any family with AttnGate == GateSigmoid. One predicate for the forward's two dispatch sites (attention.go, forwardn.go)
// and the resident feature derived from it (FeatAttnOutputGate).
func (a *Architecture) hasAttnOutputGate() bool { return a.laguna != nil || a.AttnGate == GateSigmoid }

// kvInt8OK reports whether NewCache may store this architecture's K/V as int8: a family with its own layer loop only when
// its ownForwards entry says KVInt8; otherwise the generic loop, which reads K/V through the int8-aware attention path,
// except for MoE, where attention runs the acc64 kernel so expert routing stays bit-stable and a quantized cache would
// reopen that. It is one rule in the table, replacing two hand-written lists that could disagree with kvRingsOK.
func (a *Architecture) kvInt8OK() bool {
	if f, own := a.ownForward(); own {
		return f.KVInt8
	}
	return a.MoE == nil
}

// kvRingsOK reports whether NewCache may ring-buffer this architecture's sliding-window (local) layers, keeping only the W
// most recent positions. The generic loop reads rings through attendQuery / attendBatchedHeads; a family with its own loop
// only when its ownForwards entry says KVRings.
func (a *Architecture) kvRingsOK() bool {
	if f, own := a.ownForward(); own {
		return f.KVRings
	}
	return true
}

// ownForward returns this architecture's own layer loop, if it has one.
func (a *Architecture) ownForward() (ownForwardFamily, bool) {
	for _, f := range ownForwards {
		if f.is(a) {
			return f, true
		}
	}
	return ownForwardFamily{}, false
}
