package arch

import (
	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
)

// LayerWeights bundles one decoder block's tensors. Matrices follow
// PyTorch's [out, in] row-major layout so MatmulBT is A·Bᵀ with no
// transpose copy, matching encoder.LayerWeights.
//
// Gemma 3 specifics reflected here: separate gate/up projections (GeGLU),
// pre- AND post-norms on both the attention and MLP sub-blocks, and
// optional QK-norm weights.
type LayerWeights struct {
	// Attention projections (matmul'd → linalg.WeightMat, quantizable).
	QProj linalg.WeightMat // [NumHeads*HeadDim, HiddenDim]
	KProj linalg.WeightMat // [NumKVHeads*HeadDim, HiddenDim]
	VProj linalg.WeightMat // [NumKVHeads*HeadDim, HiddenDim]
	OProj linalg.WeightMat // [HiddenDim, NumHeads*HeadDim]
	// GProj is Laguna's attention output gate (g_proj), applied to the attention
	// context BEFORE OProj as ctx *= softplus(GProj·h). Rows are NumHeads for the
	// "per-head" granularity or NumHeads*HeadDim for "per-element". Zero-valued
	// (Rows()==0) for every other family. See applyGateRow.
	GProj linalg.WeightMat // [NumHeads | NumHeads*HeadDim, HiddenDim]
	// Projection biases ([out]; Qwen2 q/k/v only). Nil when the family/checkpoint
	// has no bias.
	QBias []float32 // [NumHeads*HeadDim]
	KBias []float32 // [NumKVHeads*HeadDim]
	VBias []float32 // [NumKVHeads*HeadDim]
	OBias []float32 // [HiddenDim] attention output-projection bias (GPT-2)
	// Norms (elementwise → stay f32). The *Bias slices are set only for
	// LayerNorm families (GPT-2); RMSNorm leaves them nil.
	QNorm           []float32 // [HeadDim] QK-norm on queries (Gemma 3)
	KNorm           []float32 // [HeadDim] QK-norm on keys (Gemma 3)
	PreAttnNorm     []float32 // [HiddenDim] input norm
	PreAttnNormBias []float32 // [HiddenDim] LayerNorm bias (GPT-2 ln_1)
	PostAttnNorm    []float32 // [HiddenDim] norm after attention, before residual add

	// MLP: projections quantizable, norms f32. UpBias/DownBias are set only for
	// the non-gated MLP (GPT-2 c_fc/c_proj); gated families leave them nil.
	GateProj       linalg.WeightMat // [IntermediateDim, HiddenDim] (unused for non-gated MLP)
	UpProj         linalg.WeightMat // [IntermediateDim, HiddenDim]
	UpBias         []float32        // [IntermediateDim] (GPT-2 c_fc bias)
	DownProj       linalg.WeightMat // [HiddenDim, IntermediateDim]
	DownBias       []float32        // [HiddenDim] (GPT-2 c_proj bias)
	PreMLPNorm     []float32        // [HiddenDim]
	PreMLPNormBias []float32        // [HiddenDim] LayerNorm bias (GPT-2 ln_2)
	PostMLPNorm    []float32        // [HiddenDim]

	// Mixture-of-experts FFN (Mixtral; set only when arch.MoE != nil). Router
	// scores experts; each expert is a gated (SwiGLU) MLP. The dense GateProj/
	// UpProj/DownProj above are unused in that case.
	Router     linalg.WeightMat          // [NumExperts, HiddenDim] router/gate logits
	RouterBias []float32                 // [NumExperts] DeepSeek/GLM e_score_correction_bias added to scores for top-k SELECTION (nil = none). gpt-oss reuses this as a plain router-LOGIT bias (added before top-k, its own forward).
	Experts    []archparts.ExpertWeights // [NumExperts] gated MLPs

	// AttnSinks is gpt-oss's per-head learned attention-sink logit ([NumHeads]):
	// an extra term in each head's softmax denominator (exp(sink_h)) with no value,
	// bleeding attention mass. nil for every other family. Consumed by attendSinksToken.
	AttnSinks []float32

	// Shared expert (Qwen-MoE / Qwen2-MoE; set when arch.MoE.SharedIntermediateDim
	// > 0). An always-on gated MLP scaled by sigmoid(SharedGate·h).
	SharedExpert archparts.ExpertWeights // gated SwiGLU MLP at SharedIntermediateDim
	SharedGate   linalg.WeightMat        // [1, HiddenDim] → sigmoid scalar gate

	// Gemma 4 per-layer extras (zero for every other family). The PLE branch runs
	// after the MLP residual: gate→gelu→×per-layer-embedding→proj→norm→+residual.
	PLEGate     linalg.WeightMat // [HiddenSizePerLayerInput, HiddenDim] (GGUF inp_gate)
	PLEProj     linalg.WeightMat // [HiddenDim, HiddenSizePerLayerInput] (GGUF proj)
	PostPLENorm []float32        // [HiddenDim] post_per_layer_input_norm (GGUF post_norm)
	LayerScalar float32          // per-layer output scalar (GGUF layer_output_scale); 1 if absent
	KVShared    bool             // carries no k/v and reuses an earlier layer's KV (Gemma 4 E-models)
	VFromK      bool             // attention_k_eq_v: no v_proj — V is v_norm(k_proj output) (Gemma 4 12B global layers)

	// qwen3_5_moe per-layer attention. Exactly one is set on the hybrid's layers:
	// delta on the Gated DeltaNet (linear) layers, qattn on the softmax layers.
	// Both nil for every other family. Stored f32 (the qwen3_5_moe forward is
	// parity-first, like Gemma 4's); the MoE FFN above is shared by both kinds.
	Delta *archparts.DeltaNetWeights
	QAttn *archparts.QwenAttnWeights

	// LFM2 gated short-convolution mixer weights, set ONLY on the conv layers (the
	// attention layers use QProj/KProj/VProj/OProj above and leave this nil). nil for
	// every other family.
	ShortConv *archparts.ShortConvWeights

	// Granite-4.0-H Mamba-2 mixer weights, set only on the mamba layers (the
	// attention layers use QProj/KProj/VProj/OProj above). nil for every other
	// family. Stored f32 (parity-first, like the qwen35 hybrid).
	Mamba *archparts.Mamba2Weights

	// DeepSeek MLA attention weights (deepseek_v2 / deepseek_v3, and Bailing Hybrid's own
	// full-attention layers). nil for every other family. Stored f32 (parity-first). The FFN
	// side (dense prefix / MoE / shared expert) reuses the generic Router/Experts/SharedExpert
	// fields above.
	MLA *archparts.MLAWeights

	// Bailing Hybrid (Ling 3.0) KDA attention weights, set on the linear layers (the
	// non-linear layers use mla above). nil for every other family.
	KDA *archparts.KDAWeights

	// Gemma 4 26B-A4B parallel dense+MoE FFN sub-block (enable_moe_block). Set only
	// on gemma4 layers when arch.MoE != nil; nil for the dense E2B/E4B/12B variants
	// and every other family. The router/experts do NOT fit the generic Router/
	// Experts fields (weightless-norm + learned-scale router, per-expert scale,
	// gelu-tanh fused-gate_up experts, three parallel-branch norms), so gemma4
	// carries its own struct consumed by gemma4MoEFFN. Stored f32 (parity-first).
	Gemma4MoE *archparts.Gemma4MoEWeights

	// Inkling holds an Inkling 2 layer's tensors beyond the shared fields (its convolutions, clips, relative bias, sinks,
	// attention-residual queries and loop controllers). nil for every other family.
	Inkling *archparts.Inkling2LayerWeights
}

// Weights is the immutable per-checkpoint bundle. Embeddings are TIED:
// Embed doubles as the LM head (logits = h · Embedᵀ), so there is no
// separate output projection tensor.
type Weights struct {
	// prof records how long each phase of this load took, when the loader instrumented it (the GGUF path does); nil otherwise.
	// It rides on Weights because that is the value the loader already returns.
	Prof *LoadProfile

	Cfg  Config
	Arch *Architecture // resolved descriptor the forward pass reads

	// bakedQuant is the resolved quant label recorded in the .giw header: "int4" | "int4mix" | "int8int8" | "int8" |
	// "native". Model.Quant() prefers it over re-inferring from tensor kinds. Empty for a direct GGUF/safetensors load
	// and for a streamed bundle (which records ""); those fall back to quantLabel() inference.
	BakedQuant    string
	Embed         linalg.WeightMat // [VocabSize, HiddenDim] — input embedding (AND tied LM head when LMHead unset)
	LMHead        linalg.WeightMat // [VocabSize, HiddenDim] — separate output head (untied families); zero value when tied
	PosEmbed      linalg.WeightMat // [MaxPositions, HiddenDim] — learned position embedding (GPT-2); zero value otherwise
	FinalNorm     []float32        // [HiddenDim] final norm before the LM head
	FinalNormBias []float32        // [HiddenDim] final LayerNorm bias (GPT-2 ln_f)
	Layers        []LayerWeights

	// Gemma 4 PLE inputs (zero for every other family). The per-layer input is
	// (token_identity + context_aware)/√2: token_identity from PerLayerTokenEmbed,
	// context_aware from PerLayerModelProj normed by PerLayerProjNorm.
	PerLayerTokenEmbed linalg.WeightMat // [VocabSize, NumLayers*HiddenSizePerLayerInput] (per_layer_token_embd)
	PerLayerModelProj  linalg.WeightMat // [NumLayers*HiddenSizePerLayerInput, HiddenDim] (per_layer_model_proj)
	PerLayerProjNorm   []float32        // [HiddenSizePerLayerInput] (per_layer_proj_norm)

	ST      *embed.SafetensorsFile // retained so alias-backed slices stay valid
	Backing []byte                 // serialized-weights blob aliased by q8/q4 arrays (LoadSerializedWeights); keeps it reachable

	Schema *archparts.TensorSchema // resolved tensor-name schema (safetensors loads); used by compute-time LoRA to map projections → adapter deltas. nil for GGUF/serialized loads.

	// kquant is where each int8 matmul of a q4k GGUF load came from when the file stores it as Q5_K or Q6_K, keyed by its int8
	// rows' first element; ggufPath is that file, and kquantFile the load's own mapping of it, held until the model's resident
	// build has run (Model.releaseKQuantFile). All are empty for any other load. KQuantReader serves them.
	KQuant     map[*int8]archparts.KQuantSrc
	GGUFPath   string
	KQuantFile *embed.GGUFFile
}

// MatmulWeights returns every quantizable matrix in the bundle (the projections,
// the embedding, and the untied head if present); norms stay f32.
func (w *Weights) MatmulWeights() []*linalg.WeightMat {
	ms := []*linalg.WeightMat{&w.Embed}
	if w.LMHead.Rows() > 0 {
		ms = append(ms, &w.LMHead)
	}
	for i := range w.Layers {
		l := &w.Layers[i]
		ms = append(ms, &l.QProj, &l.KProj, &l.VProj, &l.OProj, &l.GateProj, &l.UpProj, &l.DownProj)
		if l.Router.Rows() > 0 {
			ms = append(ms, &l.Router)
		}
		// GProj is deliberately excluded from quantization: it is tiny (one row per head, well under 1% of a layer's attention
		// weights), so quantizing buys nothing, and it feeds a softplus whose output multiplies the whole attention context, so
		// quantization error there scales every channel of every head rather than perturbing one projection additively.
		for e := range l.Experts {
			ex := &l.Experts[e]
			ms = append(ms, &ex.Gate, &ex.Up, &ex.Down)
		}
		if l.SharedExpert.Gate.Rows() > 0 {
			ms = append(ms, &l.SharedExpert.Gate, &l.SharedExpert.Up, &l.SharedExpert.Down, &l.SharedGate)
		}
		if l.PLEGate.Rows() > 0 { // Gemma 4 PLE branch
			ms = append(ms, &l.PLEGate, &l.PLEProj)
		}
	}
	if w.PerLayerTokenEmbed.Rows() > 0 { // Gemma 4 model-level PLE inputs
		ms = append(ms, &w.PerLayerTokenEmbed, &w.PerLayerModelProj)
	}
	return ms
}

// isLogitTable reports whether m is an embedding-class tensor: the token embedding, the LM head, or Gemma 4's model-level
// PLE embeddings. In int4 mode these are pinned to int8 by default (logit-critical; the EmbedInt4 knob relaxes them), so
// their precision is orthogonal to the int4-vs-int4mix distinction and must be excluded from the quant classification.
// This is the single definition of that exclusion.
func (w *Weights) isLogitTable(m *linalg.WeightMat) bool {
	return m == &w.Embed || m == &w.LMHead || m == &w.PerLayerTokenEmbed || m == &w.PerLayerModelProj
}

// BodyMatmulWeights is matmulWeights minus the logit tables: the attention/FFN projections, experts and routers, exactly
// the matmuls whose precision the chosen quant (int4|int4mix|int8|int8int8) determines. It is the one list quantLabel
// classifies over (and, through quantLabel, the value the .giw header records at bake time), so "which tensors define the
// quant" lives in one place. cuda's nonBatchableKind also excludes the logit tables but inspects resident per-layer
// projections, a different type in another module, so it cannot share this host-side function.
func (w *Weights) BodyMatmulWeights() []*linalg.WeightMat {
	all := w.MatmulWeights()
	body := make([]*linalg.WeightMat, 0, len(all))
	for _, m := range all {
		if !w.isLogitTable(m) {
			body = append(body, m)
		}
	}
	return body
}
