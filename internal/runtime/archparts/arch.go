package archparts

// GPTOSSParams carries gpt-oss's expert-activation knobs. The sink weights and
// per-expert biases are per-layer (LayerWeights.AttnSinks / expertWeights.*Bias);
// the sliding-window pattern is Architecture.SlidingWindow + layerIsGlobal. forward_gptoss.go reads this.
type GPTOSSParams struct {
	SwigluAlpha float64 // sigmoid gain in gate·sigmoid(α·gate) (1.702)
	SwigluLimit float64 // clamp bound: gate ≤ limit, up ∈ [-limit, limit] (7.0)
}

// Llama4Params carries Llama 4's per-layer iRoPE deltas. useRope[i]/isMoE[i] select layer
// i's attention (RoPE+L2-QK-norm vs NoPE+attn-temperature) and FFN (dense vs MoE). The
// dense layers use Architecture.IntermediateDim (intermediate_size_mlp); the routed +
// shared experts use MoEConfig.IntermediateDim (intermediate_size). forward_llama4.go reads this.
type Llama4Params struct {
	// chunkSize is attention_chunk_size (8192 on Scout/Maverick): the RoPE layers use a block-diagonal chunked mask, so a query
	// at position p attends only to keys in its own chunk, [(p/C)*C, p]. NoPE layers stay full-causal. 0 means no chunking.
	// Below C chunked equals full causal, so a short-sequence parity gate cannot see a missing chunk mask.
	ChunkSize  int
	UseRope    []bool  // per layer: RoPE (true) vs NoPE (false) — from no_rope_layers
	IsMoE      []bool  // per layer: MoE (true) vs dense (false) — from moe_layers
	UseQKNorm  bool    // parameter-free L2 (RMS-over-head-dim) QK-norm on the RoPE layers
	AttnTemp   bool    // attention-temperature tuning on the NoPE layers
	FloorScale float64 // attn-temp: log1p(floor((pos+1)/floorScale))·attnScale + 1
	AttnScale  float64
}

// NemotronParams marks a Nemotron-H hybrid: a SINGLE-OP-per-block stack where each
// layer is exactly one of {mamba, attention, mlp} (blockKind, from layers_block_type)
// over a pre-norm residual — NOT the mixer+FFN block every other family uses. The
// mamba layers reuse the Mamba-2 mixer (same geometry fields as graniteParams); the
// attention layers are NoPE GQA (no RoPE); the mlp layers are non-gated relu². No
// Granite-style multipliers. forward_nemotron.go consumes this.
type NemotronParams struct {
	NHeads, HeadDim, DState, NGroups, DConv int     // Mamba-2 dims
	BlockKind                               []uint8 // per layer: 0 mamba · 1 attention · 2 mlp · 3 moe
}

const (
	NemoMamba uint8 = iota
	NemoAttn
	NemoMLP
	NemoMoE
)

// MLAParams carries DeepSeek Multi-head Latent Attention geometry. The cached state is the compressed latent
// [KVLoRARank + QKRopeHeadDim] per position; per-head K/V are rebuilt from it each step via kv_b_proj. QLoRARank 0 means a
// direct q_proj instead of the q_a/q_b LoRA bottleneck. forward_deepseek.go consumes this.
type MLAParams struct {
	QLoRARank      int  // q_a_proj bottleneck width; 0 ⇒ direct q_proj (no q-LoRA)
	KVLoRARank     int  // compressed KV latent width (the cached payload, minus the rope key)
	QKNopeHeadDim  int  // per-head Q/K dims WITHOUT RoPE
	QKRopeHeadDim  int  // per-head Q/K dims carrying decoupled RoPE (one K shared across heads)
	VHeadDim       int  // per-head V width (≠ QKNopeHeadDim+QKRopeHeadDim)
	RoPEInterleave bool // GPT-J pairwise (true, V3 default) vs NeoX half-split RoPE on the rope dims
	// AttnPrefix/DenseSuffix override the tensor-name prefix and output-projection suffix; "" means "self_attn"/"o_proj.weight".
	// Bailing Hybrid uses "attention"/"dense.weight": its MLA and KDA mixers both sit under self.attention, and its MLA names
	// its output projection dense.
	AttnPrefix, DenseSuffix string
	// GateGranularity ("" | "head_wise" | "element_wise"): Bailing Hybrid's optional per-head or per-element sigmoid output gate
	// (g_proj) on the attention context before the output projection, Laguna's gate structure with sigmoid in place of
	// softplus. "" (every DeepSeek family) means no gate.
	GateGranularity string
}

// KDAParams carries Bailing Hybrid's (Ling 3.0) Kimi Delta Attention geometry for the linear-attention layers; the MLA
// layers use mlaParams. HeadDim/NumHeads are shared by q/k/v (no GVA). NoLora selects a single f_proj/g_proj linear per
// gate (Ling-3.0-tiny) over a LoRA'd a/b-split pair; the LoRA'd path is not implemented (see kdaArchitecture).
type KDAParams struct {
	HeadDim, NumHeads, ConvKernel int
	NoLora                        bool
	SafeGate                      bool
	LowerBound                    float64
}

// QKHeadDim is the query·key dot-product width: the no-rope dims plus the rope dims.
func (p *MLAParams) QKHeadDim() int { return p.QKNopeHeadDim + p.QKRopeHeadDim }

// LFM2Params carries the gated short-convolution geometry for an LFM2/LFM2.5 model's conv layers; the attention layers use
// the uniform Architecture fields. The block is in_proj -> split into three ConvDim gates (B, C, x) -> Bx = B*x ->
// depthwise causal conv with NO activation -> y = C*conv -> out_proj; ConvLCache taps, no bias. The missing activation
// differs from Mamba-2's SiLU conv (upstream passes activation=None), and adding one gives a plausible, wrong model.
type LFM2Params struct {
	ConvDim    int // channels the conv block operates on (hidden_size on released weights)
	ConvLCache int // kernel width / rolling-window depth (3)
}

// GraniteParams carries Granite-4.0-H's Mamba-2 mixer geometry (for the mamba layers; the attention layers use the
// uniform Architecture fields) and the three in-block scalar multipliers. The fourth Granite scalar, logits_scaling, lives
// on Architecture.LogitScale (it is applied at the shared head, not per layer).
type GraniteParams struct {
	NHeads, HeadDim, DState, NGroups, DConv int     // Mamba-2 dims
	EmbMul, ResidMul                        float32 // embedding scale, residual-add scale (attention scale is Architecture.AttnScale)
}

// Qwen35Params carries the Gated DeltaNet geometry for a qwen3_5_moe model's linear-attention layers.
// The softmax layers use the uniform Architecture attention fields; only the linear layers
// read these.
type Qwen35Params struct {
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
	// layers through loadQwen35Attn / the bespoke qwen3.5 attention (opGatedAttn); true routes them through the shared generic
	// loader path and the generic attention, reusing olmo3's adapter.
	PlainFullAttn bool
}

// Gemma4Params describes how Gemma 4's global (full-attention) layers diverge
// from its local (sliding) layers. Local layers use Architecture.HeadDim /
// NumKVHeads and full rotary; global layers use these instead. Set by
// gemma4Architecture; consumed per-layer by the forward pass (the Gemma 4 program walk).
type Gemma4Params struct {
	GlobalHeadDim    int  // global head_dim (e.g. 512); local = Architecture.HeadDim (256)
	NumGlobalKVHeads int  // global KV-head count; local = Architecture.NumKVHeads. 0 ⇒ same as local
	GlobalRotaryDim  int  // rotated dims on global layers = partial_rotary_factor * GlobalHeadDim; local = full
	KVShared         bool // attention_k_eq_v: V reuses K's projection on global layers (12B; off for E2B)

	// E-model (E2B/E4B) extras; zero/empty on the dense 12B.
	SharedKVLayers          int   // last N layers carry no k/v and reuse an earlier layer's KV (per type)
	FFNPerLayer             []int // variable per-layer FFN width (else Architecture.IntermediateDim is uniform)
	HiddenSizePerLayerInput int   // PLE per-layer dim (256); 0 ⇒ no PLE
	VocabSizePerLayerInput  int   // PLE embedding-table vocab (== main vocab)

	// PadTokenID is the id runLayersFromEmbedPLE's PLE token-identity lookup uses at a multimodal (image/video/audio)
	// position in place of the caller's real token id, matching the HF multimodal forward's pad substitution
	// (docs/multimodal.md).
	PadTokenID int
}

// LagunaParams describes how Laguna (poolside) departs from an otherwise Qwen2-MoE-shaped decoder; both fields are read
// straight from the released configs.
type LagunaParams struct {
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
