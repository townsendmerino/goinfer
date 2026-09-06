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
	// NormPlacementLinear overrides NormPlacement on layers where isLinearLayer(i) is
	// true — Olmo Hybrid's real departure from every other DeltaNet hybrid in this
	// tree (qwen3_5/qwen3_next/granitemoehybrid all use ONE scheme for both their
	// linear and full-attention layers). Verified against the real
	// modeling_olmo_hybrid.py: full-attention layers use NormPostOnly (olmo3's own
	// scheme, confirmed identical), but the DeltaNet layers use plain NormPre2 — two
	// placements in ONE model, keyed by the SAME layerIsLinear hook that already
	// selects the mixer. nil (every family so far, including olmo3 itself, which has
	// no linear layers at all) ⇒ NormPlacement applies uniformly, exactly as before
	// this field existed. See normPlacementAt.
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
	// QKNormWhole (Olmo 3/Olmo Hybrid): when QKNorm is also set, normalize the WHOLE projected
	// q/k vector as one RMSNorm (num_heads*head_dim elements, one statistic) instead of per-head
	// — verified against the real modeling_olmo3.py, not the standard per-head convention every
	// other QK-norm family uses. The underlying rmsNorm(x, weight, rows, dim, ...) already
	// supports this: per-head calls it with (rows=nHeads, dim=headDim); whole-vector calls it
	// with (rows=1, dim=nHeads*headDim) — same function, different split.
	QKNormWhole     bool
	LearnedPosEmbed bool // GPT-2: add a learned position embedding and SKIP RoPE
	// NoPositionEncoding (Olmo Hybrid): true when the family genuinely has NO positional
	// encoding at all — neither RoPE nor learned — on any layer. The released checkpoint's
	// rope_parameters is {"rope_theta": null}; modeling_olmo_hybrid.py's own comment says so
	// explicitly ("Released ckpt don't use any ROPE"). A fourth legitimate "no RoPEGlobalBase"
	// reason alongside LearnedPosEmbed/nemotron/mla in validateResolved's M-06 check, named
	// explicitly for the SAME reason those three are: so a family that simply forgot to read
	// rope_theta cannot look like one that deliberately has none.
	NoPositionEncoding bool
	AttnScale          float64 // explicit q·k multiplier (resolved: query_pre_attn_scalar^-0.5 or 1/sqrt(headDim))
	// AttnTempBeta/AttnTempOrigMaxPos (Ministral 3): a position-dependent multiplicative scale on
	// the query, applied AFTER RoPE, on every layer — get_llama_4_attn_scale in Ministral3's own
	// HF source (modular_ministral3.py), literally named after Llama 4's attention-temperature
	// tuning: scale = 1 + beta·ln(1 + floor(pos/origMaxPos)). It is IDENTICAL in shape to the
	// attnTemp/floorScale primitive llama4Architecture already has (decoder/forward_llama4.go),
	// but Llama 4 applies it INSTEAD of RoPE on NoPE layers only; Ministral 3 applies it ON TOP OF
	// RoPE on every layer, which llama4's own dedicated forward has no path for. Generalized here
	// as generic Architecture fields (0 ⇒ off, so every existing family is unaffected) rather than
	// copying llama4's own-forward path, since the formula is a straightforward postfix to the
	// generic causalAttention RoPE step. scale ≡ 1 for pos < origMaxPos (floor(pos/origMaxPos)=0),
	// so a SHORT test prompt exercises nothing — this is the "minimal repro hides the bug" trap
	// this repo's own culture warns about; the tiny fixture's prompt is deliberately longer than
	// origMaxPos.
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
	// RotaryDimLocal is the same for the LOCAL (sliding) layers when they rotate a
	// DIFFERENT width than the global ones; 0 ⇒ both use RotaryDim. It completes the
	// local/global RoPE triple that RoPELocalBase and ropeScalingLocal already start:
	// Laguna's XS generations key partial_rotary_factor by layer type (0.5 on full,
	// 1.0 on sliding), so their two layer types differ in base, scaling, AND rotated
	// width at once. Every other family leaves it 0 and is unaffected.
	RotaryDimLocal int
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
	// ropeInterleave selects GPT-J pairwise rotation (dims 2d,2d+1) over the NeoX
	// split-half layout (dims d,d+half) in the generic scalar RoPE path. Cohere/
	// Command-R, Falcon, GPT-J, StableLM set it; Llama/Qwen/Gemma leave it false.
	// (DeepSeek's MLA carries its own ropeInterleave on mlaParams.)
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

	// lfm2, when non-nil, marks an LFM2/LFM2.5 hybrid: every layer has a SwiGLU FFN,
	// and its mixer is either a gated short convolution (layerIsConv true, 22 of 30 on
	// LFM2.5-2.6B) or GQA softmax attention with per-head RMSNorm on Q and K.
	//
	// The conv layers carry a rolling per-channel window instead of a KV cache, which
	// is why this is a cache-shape fact and not only a forward one.
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

	// llama4, when non-nil, marks a Llama 4 text decoder (llama4_text): the iRoPE
	// stack — per-layer RoPE/NoPE interleave, parameter-free L2 QK-norm on the RoPE
	// layers, attention-temperature tuning on the NoPE layers, and a dense/MoE
	// interleave (top-1 sigmoid routing + an ungated shared expert). Own forward
	// (forward_llama4.go). nil for every other family.
	llama4 *llama4Params

	// gptoss, when non-nil, marks a gpt-oss sparse-MoE family: GQA with a learned
	// per-head attention SINK in the softmax denominator, alternating sliding/full
	// attention (even layers sliding, window SlidingWindow), YaRN RoPE, and a
	// clamped interleaved-SwiGLU expert (gate·sigmoid(α·gate) · (up+1), clamped, with
	// per-expert biases) + a router-logit bias. Own forward (forward_gptoss.go),
	// CPU-only (CUDA/Metal decline via FeatAttnSink). nil for every other family.
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
	// chunkSize is attention_chunk_size (8192 on Scout/Maverick): the RoPE layers use a
	// BLOCK-DIAGONAL chunked mask, so a query at position p attends only to keys in its own
	// chunk, [(p/C)*C, p]. NoPE layers stay full-causal.
	//
	// M-05: this was read from config and then dropped, and the forward attended [0, pos] on
	// every layer. Below C that is identical to chunked — which is why the parity gates, which
	// use short sequences, never saw it — and from position C on, the RoPE layers saw keys HF
	// masks out. 0 means no chunking (a checkpoint that does not set the field).
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

// mlaParams carries DeepSeek Multi-head Latent Attention geometry. The cached state
// is the compressed latent [KVLoRARank + QKRopeHeadDim] per position (the KV-memory
// payoff: ~576 floats/token vs the ~41k a reconstructed full K+V would need); per-head
// K/V are rebuilt from it each step via kv_b_proj. QLoRARank 0 ⇒ a direct q_proj (the
// V2-Lite path) instead of the q_a/q_b LoRA bottleneck. forward_deepseek.go consumes this.
type mlaParams struct {
	QLoRARank      int  // q_a_proj bottleneck width; 0 ⇒ direct q_proj (no q-LoRA)
	KVLoRARank     int  // compressed KV latent width (the cached payload, minus the rope key)
	QKNopeHeadDim  int  // per-head Q/K dims WITHOUT RoPE
	QKRopeHeadDim  int  // per-head Q/K dims carrying decoupled RoPE (one K shared across heads)
	VHeadDim       int  // per-head V width (≠ QKNopeHeadDim+QKRopeHeadDim)
	ropeInterleave bool // GPT-J pairwise (true, V3 default) vs NeoX half-split RoPE on the rope dims
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

// graniteParams carries Granite-4.0-H's Mamba-2 mixer geometry (for the mamba
// layers; the attention layers use the uniform Architecture fields) and the three
// in-block scalar multipliers. The fourth Granite scalar, logits_scaling, lives on
// Architecture.LogitScale (it's applied at the shared head, not per layer).
// lfm2Params carries the gated short-convolution geometry for an LFM2/LFM2.5 model's
// conv layers. The attention layers use the uniform Architecture fields; only the conv
// layers read these.
//
// ConvDim channels, a KERNEL of ConvLCache taps (3), and no bias on any released
// checkpoint. The block is in_proj -> split into three ConvDim gates (B, C, x) ->
// Bx = B*x -> depthwise causal conv, NO activation -> y = C*conv -> out_proj. The
// missing activation is a real difference from Mamba-2's conv, which applies SiLU:
// upstream passes activation=None here, so adding one would be a plausible, wrong model.
type lfm2Params struct {
	ConvDim    int // channels the conv block operates on (hidden_size on released weights)
	ConvLCache int // kernel width / rolling-window depth (3)
}

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

	// FusedDeltaNetProj: qwen3_5_moe's checkpoint stores in_proj_qkv/in_proj_z/
	// in_proj_b/in_proj_a as four separate tensors; qwen3_next's checkpoint fuses
	// them into in_proj_qkvz/in_proj_ba instead (same math, different packing —
	// verified against modular_qwen3_next.py's Qwen3NextGatedDeltaNet.torch_forward).
	// loadQwen35Attn splits the fused tensors into the same four deltaNetWeights
	// fields so the rest of the pipeline (forward, gguf, serialize) is untouched.
	FusedDeltaNetProj bool
	// SeparateQKVProj (Olmo Hybrid): the checkpoint stores q_proj/k_proj/v_proj as
	// THREE fully independent tensors — more unfused than qwen3_5_moe's own
	// in_proj_qkv (which is already pre-concatenated on disk into one [convDim,
	// hidden] tensor). loadQwen35Attn concatenates them at load time into the same
	// internal inProjQKV layout, so gatedDeltaNetStep and every downstream consumer
	// stay untouched. Verified against the real modeling_olmo_hybrid.py
	// (OlmoHybridGatedDeltaNet.__init__: separate self.q_proj/k_proj/v_proj
	// nn.Linear modules, vs. qwen3.5's single mixed_qkv projection).
	SeparateQKVProj bool
	// NegEigval (Olmo Hybrid): doubles the write-gate beta from sigmoid's [0,1)
	// range to [0,2) after the sigmoid — config's linear_allow_neg_eigval, default
	// true on the release. Same recurrence, wider gate range; not a new primitive.
	NegEigval bool
	// SeparateConv (Olmo Hybrid): the checkpoint stores the depthwise causal conv
	// as THREE separate tensors, q_conv1d/k_conv1d/v_conv1d, split at the SAME
	// q/k/v channel boundaries the mixed_qkv activation uses (keyDim, keyDim,
	// valueDim rows respectively) — verified against a real Olmo-Hybrid-7B
	// checkpoint's safetensors header, not assumed from source (the modeling code
	// alone shows one combined self.conv1d; only the real file's actual tensor
	// names and shapes revealed the three-way split). loadQwen35Attn concatenates
	// them in q,k,v order to reconstruct the same [convDim,1,K] layout
	// gatedDeltaNetStep's conv step already expects.
	SeparateConv bool
	// DeltaNetNormSuffix/DeltaNetOutProjSuffix override the DeltaNet gated-RMSNorm
	// and out_proj tensor suffixes. "" (qwen3_5/qwen3_5_moe/qwen3_next) means
	// "linear_attn.norm.weight" / "linear_attn.out_proj.weight". Olmo Hybrid's real
	// modeling_olmo_hybrid.py names these attributes o_norm/o_proj instead — same
	// math, different attribute name and therefore different on-disk tensor name,
	// checked against real source rather than assumed identical to qwen3.5's own
	// DeltaNet.
	DeltaNetNormSuffix, DeltaNetOutProjSuffix string
	// ONormEps overrides the DeltaNet output gated-RMSNorm's epsilon; 0 ⇒ use the
	// family's own NormEps (qwen3_5/qwen3_5_moe/qwen3_next: their gated-RMSNorm
	// reads rms_norm_eps like every other norm in the model). Olmo Hybrid hardcodes
	// eps=1e-5 for this ONE norm regardless of config — "FLA's FusedRMSNormGated
	// uses eps=1e-5 by default", per modeling_olmo_hybrid.py's own comment — which
	// differs from the release's rms_norm_eps=1e-6 used everywhere else in the
	// model. A silent-wrong trap if copied: reusing NormEps here would be off by an
	// order of magnitude on exactly the one norm most sensitive to it (the
	// recurrence's own output gate).
	ONormEps float64
	// PlainFullAttn (Olmo Hybrid): this family's full-attention layers are a PLAIN
	// olmo3-shaped self-attention (single-width q/k/v/o, optional whole-vector
	// QK-norm, generic RoPE/NoPE) — NOT qwen3_5's own double-width gated softmax
	// attention. false (every other qwen35-shaped family) routes non-linear layers
	// through loadQwen35Attn's/runLayersQwen35's bespoke qwen3.5 attention; true
	// routes them through the shared generic loader path and causalAttention
	// instead, reusing olmo3's own adapter rather than inventing a second one.
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

// lagunaParams describes how Laguna (poolside) departs from an otherwise
// Qwen2-MoE-shaped decoder. Both fields are read straight from the released
// configs; see docs/task-laguna.md for the Phase 0 verification.
type lagunaParams struct {
	// HeadsPerLayer is num_attention_heads_per_layer: layer i's QUERY head count,
	// which varies with the layer's attention type on the XS generations (48 on
	// full_attention, 64 on sliding_attention). nil ⇒ uniform Architecture.NumHeads
	// (M.1, whose config omits the field). KV heads stay uniform at 8 either way, so
	// the GQA group size — not just the head count — changes per layer.
	HeadsPerLayer []int
	// GatePerHead selects the gate's granularity: true ⇒ g_proj emits one gate per
	// HEAD, broadcast across head_dim (config gating "per-head"); false ⇒ one gate
	// per (head, head_dim) CHANNEL (config gating true or "per-element").
	//
	// Three released spellings map onto this one bool exactly as the vendor code
	// does — self.gate_per_head = (gating == "per-head") — so `true` and
	// "per-element" are the same path and only "per-head" differs.
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

	// Group-limited routing (DeepSeek-V3 noaux_tc). Experts are partitioned into
	// NGroup contiguous groups; each group is scored by its top-2 selection scores
	// summed, the top TopkGroup groups are kept, and the per-token top-k is taken
	// only among experts in those groups. NGroup ≤ 1 (GLM, V2-Lite) ⇒ no grouping,
	// the plain global top-k.
	NGroup    int
	TopkGroup int

	// Gemma 4 26B-A4B router extras (Gemma4TextRouter; false for every other MoE
	// family). Its selection is still softmax-over-all → top-k → renorm (NormTopKProb
	// is UNCONDITIONALLY true), so the base routeExperts path applies — these two flags
	// add the parts it doesn't have. See docs/task-gemma4-moe.md §A2 (Phase 1a refs).
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
	// NormPostOnly: NO pre-norm at all — attention/MLP read the RAW residual
	// stream directly — and the sublayer's OUTPUT is normalized before the
	// residual add (residual = x + post_attn_norm(attn(x)); same for MLP).
	// Olmo 3, verified against the real modeling_olmo3.py: no input_layernorm
	// exists at all, only post_attention_layernorm / post_feedforward_layernorm,
	// applied to the SUBLAYER OUTPUT before the add. Genuinely different from
	// Sandwich4, which normalizes the input AND (separately) the output; here
	// there is no input norm to skip past. Olmo Hybrid's full-attention layers
	// use this SAME scheme, but its DeltaNet layers use NormPre2 instead — see
	// NormPlacementLinear, not a second value of this enum.
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

// isGlobalLayer reports whether layer i uses full (global) attention vs local
// (sliding-window). Defaults to global when no per-layer function is set.
func (a *Architecture) isGlobalLayer(i int) bool {
	if a.layerIsGlobal != nil {
		return a.layerIsGlobal(i)
	}
	return true
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
}

// ownForwards is THE list of families that do not use the generic layer loop — one table, so that
// "runLayers dispatches here" and "the batched path must not touch this" are the same fact.
//
// THEY WERE TWO FACTS, AND THEY DISAGREED. runLayers dispatched LFM2 to runLayersLFM2 while
// canBatchN's hand-written exclusion list — gemma4, qwen35, granite, nemotron, mla, llama4, gptoss
// — simply did not mention it. So every prompt of ≥2 tokens ran the DENSE ATTENTION STACK over
// LFM2's 22 conv layers, whose QProj/KProj/VProj/OProj/QNorm/KNorm are never loaded: rmsNorm
// indexed a nil weight slice at layer 0 and the process died, since the panic is in the Generate
// goroutine where net/http's handler recover cannot reach it. PrefillPath() published "batched
// shape" for the same model at startup. Reproduced on the committed testdata/lfm2-tiny fixture
// with a 4-token prompt (audit-2026-09-02 C-01; three reviewers found it independently).
//
// This is audit §0 theme 1 — "one predicate, seven consumers" — applied to the first two. A family
// added to this table is excluded from the batched path by construction, and
// TestOwnForward_tableNamesEveryFamilyForward fails if a runLayersXxx is written that is not here.
var ownForwards = []ownForwardFamily{
	// Gemma 4: per-layer head_dim, KV-sharing, PLE.
	{"gemma4", func(a *Architecture) bool { return a.gemma4 != nil }, (*Model).runLayersGemma4, true, false},
	// qwen3_5_moe: Gated DeltaNet / softmax hybrid.
	{"qwen3_5_moe", func(a *Architecture) bool { return a.qwen35 != nil }, (*Model).runLayersQwen35, true, true},
	// lfm2: gated short-conv / softmax hybrid.
	{"lfm2", func(a *Architecture) bool { return a.lfm2 != nil }, (*Model).runLayersLFM2, false, true},
	// granitemoehybrid: Mamba-2 / softmax hybrid.
	{"granitemoehybrid", func(a *Architecture) bool { return a.granite != nil }, (*Model).runLayersGranite, false, true},
	// nemotron_h: single-op-per-block hybrid.
	{"nemotron_h", func(a *Architecture) bool { return a.nemotron != nil }, (*Model).runLayersNemotron, false, true},
	// deepseek_v2/v3: Multi-head Latent Attention.
	{"deepseek_v2/v3", func(a *Architecture) bool { return a.mla != nil }, (*Model).runLayersDeepseek, false, false},
	// llama4_text: iRoPE (per-layer RoPE/NoPE + L2 QK-norm + attn-temp).
	{"llama4_text", func(a *Architecture) bool { return a.llama4 != nil }, (*Model).runLayersLlama4, false, false},
	// gpt-oss: per-head attention sinks + clamped-SwiGLU MoE.
	{"gpt-oss", func(a *Architecture) bool { return a.gptoss != nil }, (*Model).runLayersGptOss, true, false},
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
