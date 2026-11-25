//go:build gpu

package kernels

import (
	"fmt"
	"os"
	"time"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/internal/cost"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// SsmStopLayerForTest truncates a newly built decode plan after this layer (-1 = every layer): the
// resident-SSM layer-sweep seam, set only by tests in this package (mamba_layersweep_test.go).
var SsmStopLayerForTest = -1

// DecodeRunner is the production one-command-buffer decode forward: it builds every
// scratch buffer + bind group ONCE (the per-token allocation that made
// DecodeTokenFused slow), then Run only WriteBuffers the input + the pos-dependent
// uniforms (RoPE pos, attention nKeys) and re-records the fixed dispatch plan into
// a fresh encoder — one Submit, one Poll, logits back. This is the GEMVRunner
// pattern applied to the whole token graph.
type DecodeRunner struct {
	c *Context
	// actG32 is RunModel.ActG32: every activation scale is per 32, so SetAdapter refuses an adapter.
	actG32 bool
	// scratch is this build's scratch leases by the op builder that asked for them (ScratchList).
	scratch scratchBook
	// Cost, when set, counts each submitted token's dispatches into CostPhase (cost.go); nil is off. costNames caches each
	// pipeline's name.
	Cost      *cost.Counter
	CostPhase cost.Phase
	costNames map[*wgpu.ComputePipeline]string
	// pending is the next token's command buffer, recorded by Run after it submits (KS-E3): steps, then the logits copy.
	// It is valid while steps and their bind groups are; dropPending releases it wherever they change (SetAdapter) or a call
	// records a different shape (RunNoLogits), and in release.
	pending              *wgpu.CommandBuffer
	pendingN             int // the dispatches recorded into pending, counted when it is submitted
	steps                []RunStep
	posUnis              []posUni
	xd, stag, lastLogits *wgpu.Buffer
	// hidden is the residual width and pleIn the per-layer buffers the row's tail is written to (nil unless the model has
	// per-layer embeddings): writeInputs takes a row of hidden + len(pleIn)·pleDim floats and refuses any other length.
	hidden, pleDim int
	pleIn          []*wgpu.Buffer
	lmHead         DecodeWeight // kept for LMHeadForTest; the gemv() closures already capture it too
	smp            *gumbelState // device Gumbel-max sampling resources, built on first RunSample
	vocab          int
	uniScratch     [16]uint32
	// logitsHost is the reused host-side logits buffer Run fills and returns, allocated once (vocab is fixed for the
	// runner's life). The returned slice is overwritten by the next Run, so the caller must consume it first.
	logitsHost []float32
	// geomVariants counts the distinct attention geometries in the plan (geomFor's cache): 1 for every uniform-geometry
	// family, 2 for Gemma 4's local/global interleave. Allocating one uniform per layer instead of per distinct tuple would
	// leave logits byte-identical while multiplying it by the layer count; the GeomVariantCount tests assert it stays 1.
	geomVariants int
	keep         []func()

	// §5 instrumentation: wall time of each Run phase, overwritten per call.
	// Zero overhead when ignored; the decomposition test reads them.
	TWrite, TEncode, TSync time.Duration
	// TEncodeAhead is the last Run's recording of the next token's buffer (KS-E3), done while the GPU runs and left out of
	// TSync; TEncode is then the in-line recording, 0 when Run submitted a pending buffer.
	TEncodeAhead time.Duration

	// Debug capture of the FIRST mamba layer's intermediate buffers (in_proj output,
	// conv output, SSM output y, mixer output gated), recorded at build. ReadMambaCap
	// copies the LAST run's values for the resident-vs-mamba2Step wiring diff
	// (gpu/mamba_resident_capture_test.go). nil for non-hybrid; zero production cost.
	mcapProj, mcapConv, mcapY, mcapGated *wgpu.Buffer

	// Per-layer residual-stream capture, built only when GOINFER_GPU_CAPTURE is set (nothing is allocated or recorded
	// otherwise): for each generic-attention layer, the attention context (pre-o-proj, [nH*hd] at the layer's own head width,
	// capCtxW) and the residual after the attention add and after the MLP add ([hidden] each), copied by an extra kv-store
	// dispatch recorded into the plan. They hold the LAST run's values; ReadCapture reads them back. The WebGPU twin of
	// decoder.Model.ForwardSubCapture, so a resident-vs-CPU divergence can be localised to a sublayer of a layer in one run.
	capCtx, capAttn, capMLP []*wgpu.Buffer
	capCtxW                 []int
	capHidden               int

	// Compute-time LoRA (lora_resident.go). baseSteps is the pristine adapter-free plan: SetAdapter never mutates it, only
	// rebuilds r.steps from it plus loraHooks (one per (layer, projection) the dense forward exposes, recorded in strictly
	// increasing afterIdx order), so clearing an adapter is a cheap restore. loraLayers is nil until SetAdapter binds one.
	baseSteps  []RunStep
	loraHooks  []loraHook
	loraLayers []loraRunLayer
	loraT      *wgpu.Buffer // [loraRMax]f32 scratch, shared by every projection/layer's down→up pair
	nLayers    int
}

// loraProjKind names which of the seven per-layer projections a hook targets — mirrors
// decoder.ResidentAdapterLayer's Q/K/V/O/Gate/Up/Down fields one-to-one.
type loraProjKind int

const (
	loraQ loraProjKind = iota
	loraK
	loraV
	loraO
	loraGate
	loraUp
	loraDown
)

// loraHook records where in baseSteps a projection's LoRA delta belongs: right after the base
// step at index afterIdx (the base projection's own gemv/gemvAdd), targeting dst — the SAME
// buffer the base projection just wrote (matching applyLoRA's CPU "matmul, then add" order) —
// with input aq/ascale — the SAME quantized activation the base projection consumed. k is the
// input width (In), needed to size/read the down-project reduction.
type loraHook struct {
	afterIdx   int
	layer      int
	kind       loraProjKind
	aq, ascale *wgpu.Buffer
	dst        *wgpu.Buffer
	k          int
}

type RunStep struct {
	Pl     *wgpu.ComputePipeline
	Bg     *wgpu.BindGroup
	Gx, Gy uint32
}

// Nemotron-H per-layer single-op kinds. nemoNone (0) ⇒ the layer runs the standard
// mixer+FFN path (every other resident family); the others run exactly one op + residual.
const (
	nemoNone uint8 = iota
	NemoKMamba
	NemoKAttn
	NemoKMLP
	// nemoKMoE is Nemotron-H's fourth block kind: routed NON-gated relu² experts (up_proj/down_proj only, unlike
	// moeMLP's gated SwiGLU) plus an always-on ungated shared expert of the same shape. Single-op-per-block like the other
	// three: no mixer, no separate FFN pass.
	NemoKMoE
)

// posUni is one per-token uniform buffer, regenerated every decode step by gen. ropePos is the
// rope-angle position — equal to pos for every ordinary (non-m-RoPE) family; only Qwen2.5-VL
// decode past an image block needs the two to differ (mropeDelta != 0). Most gen closures ignore
// ropePos entirely (KV-storage-offset and attention-range uniforms stay keyed on pos); only the
// rope-angle-encoding ones (ropeQUniFor/ropeKUniFor/qkvFinUniFor below) use it.
type posUni struct {
	buf  *wgpu.Buffer
	fill func(dst *[16]uint32, pos, ropePos int) int
}

// attnGeom is one distinct per-layer attention shape: head_dim (hd), KV-head count (nKV), rotary half-width
// (half = rotaryDim/2) and the attention_k_eq_v flag (kEqV). Gemma 4 interleaves two shapes (local hd=256/nKV=8; global
// hd=512/nKV=2, K=V); every other family has one. geomFor caches one attnGeom per distinct {hd, nKV, half, kEqV} tuple,
// and the per-token uniforms that carry these dims (v-store, attn, the windowed-attn variant and, additionally keyed by
// rope scale, q-rope / k-rope-store / the fused qkv-finalize) hang off it. A uniform-geometry model therefore collapses
// to a single shared *attnGeom, so byte-identity for non-Gemma models is structural, not asserted.
//
// kEqV is in the key because the geom OWNS the v-store uniforms: a K=V layer derives V from K instead of storing a
// projected V, so it must not share a geom with an otherwise equal layer. nH (query-head count) is deliberately NOT in
// the key: it is a model-level constant and the GQA ratio is recomputed as nH/nKV per geom. A family with per-layer
// QUERY-head counts would have to add nH to the key.
type attnGeom struct {
	hd, nKV, half, kvDim             int
	kEqV                             bool
	ropeQUnis, ropeKUnis, qkvFinUnis map[float32]*wgpu.Buffer
	vStoreUni, vStoreI8Uni           *wgpu.Buffer
	attnUni, attnUniLocal            *wgpu.Buffer
}

// RunLayer / runModel are the DecodeRunner's precision-agnostic view of a resident
// model: the f32 buffers (norms, RoPE freqs, KV caches) plus the projection
// weights as decodeWeight (W8A8 or W4A8). The public constructors adapt a concrete
// ModelW / ModelW4 into this; the builder below works the same for either.
type RunLayer struct {
	AttnNorm, InvFreq, KCache, VCache, MlpNorm *wgpu.Buffer
	KScale, VScale                             *wgpu.Buffer // int8-KV per-(pos,head) scales; nil unless kvI8
	Q, K, V, O, Gate, Up, Down                 DecodeWeight
	QBias, KBias, VBias                        *wgpu.Buffer // optional (Qwen2); nil ⇒ no bias
	QNorm, KNorm                               *wgpu.Buffer // optional per-head QK-norm weights [hd] (Qwen3/GLM); nil ⇒ none

	// gpt-oss's o-proj bias (FeatOutBias) and Gemma's sandwich post-sublayer norms (FeatSandwichNorm); nil ⇒ off. Both
	// defeat the fused gemvAdd residual epilogue: the o-proj/down-proj GEMV runs bare, then the bias/norm applies, THEN a
	// separate residual add via biasAdd (xd += out).
	OBias                     *wgpu.Buffer // gpt-oss o_proj bias [hidden]; nil ⇒ none
	PostAttnNorm, PostMLPNorm *wgpu.Buffer // Gemma sandwich post-sublayer norms [hidden]; nil ⇒ none

	// attnSinks/hasSink: gpt-oss's per-head learned attention sink [nH] (FeatAttnSink). Unlike every other "nil ⇒ off" field
	// here, attnSinks is ALWAYS bound (a one-element dummy for every other family) and hasSink is the on/off flag, because
	// WGSL bind groups cannot bind a null storage buffer. This is Metal's convention, not CUDA's null-pointer sentinel.
	AttnSinks *wgpu.Buffer
	HasSink   bool
	IsLocal   bool    // sliding-window (local) attention layer; false ⇒ full
	RopeScale float32 // per-layer RoPE cos/sin scale = mscale; 0 ⇒ 1.0

	// Per-layer attention geometry: this layer's head_dim / KV-heads / rotaryDim-half and attention_k_eq_v. Zero ghd ⇒ the
	// model-level nH-relative shape; gKEqV is read independently (a K=V layer always sets its full tuple). The plan loop
	// resolves these into a shared *attnGeom via geomFor; this layer's kvDim is gnKV*ghd.
	Ghd, GnKV, Ghalf int
	GKEqV            bool
	LayerScalar      float32 // Gemma 4 dense per-layer output scalar (1 if absent/unscaled)
	// vNorm: a Gemma 4 layer with its own v_proj (not K=V) applies the scale-less v_norm to the v_proj output, as the CPU
	// does on every K/V-owning layer; K=V layers always do (gKEqV). See g4DropVNormForTest.
	VNorm bool

	// Gemma 4 E-model. kvShared: the layer owns no K/V; kCache/vCache (and the int8 scales) are an earlier layer's, and
	// it projects and rotates Q only. pleGate/pleProj/postPLENorm: the per-layer-embedding branch after the FFN
	// residual (nil pleGate ⇒ none); its input is the layer's slice of the row tail (DecodeRunner.pleIn).
	KvShared         bool
	PleGate, PleProj DecodeWeight
	PostPLENorm      *wgpu.Buffer

	// MoE: when isMoE, this layer's FFN is a sparse mixture of experts instead of the dense gate/up/down above. router scores
	// all nE experts; the on-GPU top-k (moeRoute) writes the chosen indices/weights, then k indexed GEMVs per projection read
	// the right expert out of the stacked buffers (expGate/expUp/expDown) and the down-combine folds the router weight.
	IsMoE                   bool
	Router                  DecodeWeight         // [nE, hidden] router logits
	RouterBias              *wgpu.Buffer         // [nE] selection bias (DeepSeek/GLM); nil ⇒ none
	ExpGate, ExpUp, ExpDown *ResidentStackedW8A8 // nE experts stacked per projection
	// gpt-oss (FeatAttnSink): expert-major bias tables, only populated when moeRunParams.gptoss.
	// gateUpBias is [nE*2*inter] (gate then up per expert); downBias is [nE*hidden].
	GateUpBias, DownBias *wgpu.Buffer
	// g4moe, when not nil, makes this layer's feed-forward Gemma 4's parallel dense and MoE block; the fields above and
	// the dense gate/up/down are then unused.
	G4moe *G4MoELayer

	// Always-on shared expert: nil shGate ⇒ none. shGateW is the [1,hidden] sigmoid gate of the gated combine; nil ⇒ the
	// shared expert is added ungated (plain residual).
	ShGate, ShUp, ShDown, ShGateW DecodeWeight

	// MLA attention. Populated when runModel.mla != nil, in which case the runner takes the latent-attention path instead of
	// the q/k/v/o block above. mlaQA/mlaQANorm/mlaQB are the q-LoRA bottleneck (nil mlaQA ⇒ the direct mlaQ); mlaKVA
	// down-projects to the latent, mlaKVANorm normalizes it; mlaWUK/mlaWUV are the per-head absorb/lift f32 weights; mlaO is
	// the output projection; latCache is this layer's compressed-latent KV cache [ctxCap*latDim].
	MlaQA, MlaQB, MlaQ, MlaKVA, MlaO DecodeWeight
	MlaQANorm, MlaKVANorm            *wgpu.Buffer
	MlaWUK, MlaWUV                   *wgpu.Buffer
	LatCache                         *wgpu.Buffer

	// Mamba-2 SSM mixer. When isMamba, this layer's sequence-mixer is the resident SSM step (mamba.go kernels) instead of
	// attention: in/out_proj are W8A8; convW/convB/headP/normW are f32 resident; win (causal-conv ring) + ssm (selective state)
	// are build-once persistent state, updated in place per token and reset per generation. ResidMul is folded into
	// mambaOutProj's scale (the residual add). isMamba=false ⇒ attention layer (above).
	IsMamba bool
	// Nemotron-H single-op-per-block: each layer is exactly ONE op (no mixer+FFN pairing).
	// nemoKind ∈ {nemoNone, nemoKMamba, nemoKAttn, nemoKMLP}; nemoNone ⇒ standard mixer+FFN.
	NemoKind                                       uint8
	MambaInProj, MambaOutProj                      DecodeWeight
	MambaConvW, MambaConvB, MambaHeadP, MambaNormW *wgpu.Buffer
	MambaWin, MambaSSM                             *wgpu.Buffer

	// Gated-DeltaNet mixer. When isDeltaNet, this layer's mixer is the recurrent delta rule (deltanet.go kernels) instead of
	// attention. The causal conv is Mamba-2's (same shape, same SiLU, same ring window), so it reuses mambaConvW/mambaWin and
	// binds an all-zero convB (DeltaNet's conv is bias-free). dnState is the [nv*hv*hk] recurrent state, TRANSPOSED relative
	// to the CPU's [hk,hv] so each thread owns a contiguous row; build-once, updated in place, reset per generation alongside
	// mambaWin.
	IsDeltaNet                            bool
	DnQKV, DnZ, DnOut                     DecodeWeight // the three dominant projections, quantized
	DnB, DnA                              DecodeWeight // the two small gate projections
	DnDtBias, DnNegExpA, DnNormW, DnState *wgpu.Buffer

	// qGate marks a full-attention layer whose q_proj is DOUBLE WIDTH — [query ‖ gate] per
	// head — with the context scaled by sigmoid(gate) before o_proj (attn_output_gate). The
	// weight stays fused because it is quantized; the split happens on the activation.
	QGate bool
}

type RunModel struct {
	Layers        []RunLayer
	FinalNorm     *wgpu.Buffer
	LmHead        DecodeWeight
	KvF16         bool            // KCache/VCache are f16-packed (NewKVCacheF16) → use the f16 kernels
	KvI8          bool            // KCache/VCache are int8-packed (NewKVCacheI8) + scales → int8 kernels
	Moe           *MoeRunParams   // non-nil ⇒ the model has MoE layers (runLayer.isMoE picks which)
	G4moe         bool            // some layer's feed-forward is Gemma 4's parallel dense and MoE block (runLayer.g4moe)
	Mla           *MlaRunParams   // non-nil ⇒ MLA latent attention replaces the q/k/v/o block
	Mamba         *MambaRunParams // non-nil ⇒ hybrid: some layers (runLayer.isMamba) are SSM mixers
	Dnet          *DnetRunParams  // non-nil ⇒ hybrid: some layers (runLayer.isDeltaNet) are DeltaNet mixers
	RopeHalf      int             // rotated pairs per head = rotaryDim/2 (partial RoPE); 0 ⇒ HeadDim/2
	SlidingWindow int             // >0 ⇒ local layers attend only the last N positions
	// gatedGELU (FeatGatedGELU) selects gelu_tanh(gate)·up over the default SiLU in the gated MLP; model-level like
	// ropeHalf/slidingWindow, since every gated-MLP family shares one activation. It is deliberately a bool with SiLU as the
	// zero value, NOT decoder.ActKind's ordinal (where 0 = GELU-tanh): a runModel built by hand (every gpu/*_test.go that
	// does not go through BuildResident) leaves it at its Go zero value, and that must stay the SiLU behavior those tests
	// assume. BuildResident sets it to `m.BackendView().GatedAct() == 0`. Why: docs/code-notes/gpu.md#runModel.gatedGELU.
	GatedGELU bool
	// pleDim is a Gemma 4 E-model's per-layer input width; 0 for every other model. With it the resident row is
	// [hidden ‖ layers·pleDim] and each dense layer runs the per-layer-embedding branch.
	PleDim int
	// QKNormWhole (FeatQKNormWhole, Olmo 3 / Olmo Hybrid) reduces QK-norm over the whole projected Q/K vector
	// (1 block of width nH*hd for Q and 1 block of width nKV*hd for K) instead of per head.
	QKNormWhole bool
	// PostOnlyNorm (FeatPostOnlyNorm) is a model whose attention and FFN layers carry no pre-norm: AttnNorm and MlpNorm are nil there,
	// and the sublayer's input is quantized as it is.
	PostOnlyNorm bool
	// AttnTemp (decoder.FeatAttnTemp, Ministral 3) is the query's attention temperature at a token's position, applied after
	// RoPE; nil is off. Scaling q scales each score q·k by the same factor, so the runner multiplies it into the attention
	// uniform's score scale per token instead of touching q. The batched prefill applies none and declines (RunModelToPrefill).
	AttnTemp func(pos int) float32
	// ActG32 (decoder.Options.ActQuantGroup == 32) runs every decode quantize and W8A8 GEMV on the per-32 twins
	// (actgroup.go): each activation scale buffer holds np/32 floats. Dense W8A8 only; newDecodeRunner refuses the
	// paths whose kernels read one scale per vector.
	ActG32 bool
	// Pairwise (decoder.FeatPairwiseRoPE) binds the pairwise twins of the decode rope kernels (pairwise.go) in place of the
	// NeoX ones; the batched prefill declines (RunModelToPrefill).
	Pairwise bool
	// LayerNorm (decoder.FeatLayerNorm, bias-free) runs every pre-norm and the final norm as LayerNorm
	// (LayerNormQuantWGSL); Parallel (decoder.FeatParallelBlock) feeds the MLP the attention's normed input, so a layer
	// carries no MlpNorm. The batched prefill declines both (RunModelToPrefill).
	LayerNorm, Parallel bool
}

// The plants of the Cohere gates, read at newDecodeRunner: LayerNormAsRMSForTest binds RMSNorm for a LayerNorm model, and
// ParallelFromResidualForTest feeds a parallel block's MLP the post-attention residual, normed again.
var LayerNormAsRMSForTest, ParallelFromResidualForTest bool

// AttnTempOffForTest is the attention temperature's plant: a runner built while it is set ignores RunModel.AttnTemp.
var AttnTempOffForTest bool

// MlaRunParams carries the model-level MLA geometry (uniform across layers). Per-layer
// weights + the latent cache live on runLayer. latDim = kvLoRARank + qkRope is the
// cached payload width; qkHead = qkNope + qkRope is the per-head q·k width.
type MlaRunParams struct {
	QLoRARank      int     // q_a bottleneck width; 0 ⇒ direct q_proj (V2-Lite)
	KvLoRARank     int     // rank of the compressed KV latent (the score/value body)
	QkNope, QkRope int     // per-head no-rope / rope q·k dims
	VHead          int     // per-head value width (≠ qkNope+qkRope)
	Interleave     bool    // V3 GPT-J pairwise RoPE layout (vs plain NeoX)
	RopeScale      float32 // YaRN attention factor folded into cos/sin (1.0 when none)
}

// MambaRunParams carries the model-level Mamba-2 geometry (uniform across mamba layers).
// dInner=nHeads·P, convDim=dInner+2·nGroups·N, projDim=2·dInner+2·nGroups·N+nHeads,
// gSize=nGroups·N, repeat=nHeads/nGroups. normGroups is the gated-RMSNorm group count
// (1 for Granite). The dispatches are pos-independent — the {win, ssm} state carries the
// recurrence, so no posUni.
type MambaRunParams struct {
	NHeads, Hp, Dn, NGroups, DConv          int
	DInner, ConvDim, ProjDim, GSize, Repeat int
	NormGroups                              int
}

// MoeRunParams carries the model-level MoE selection knobs (uniform across layers):
// the router top-k shape + scoring flavor. Per-layer data (router/expert weights,
// selection bias) lives on runLayer. inter is the per-expert FFN width.
type MoeRunParams struct {
	NE, K, Inter      int
	Sigmoid, Norm     bool
	Scale             float32
	SharedInter       int  // shared-expert FFN width (qwen2_moe / GLM); 0 ⇒ no shared expert
	SharedUngated     bool // GLM/DeepSeek add the shared expert with no sigmoid gate
	NGroup, TopkGroup int  // DeepSeek group-limited routing; nGroup ≤ 1 ⇒ plain global top-k
	// gptoss selects a THIRD MoE dispatch shape, beside the plain top-k and DeepSeek's group-limited routing: its own router
	// contract (biased logits select AND weight), clamped-gated activation (gptossAlpha/gptossLimit, uniform across every
	// gpt-oss layer, from decoder.Model.GptOssActResident) and biased down-combine.
	Gptoss                   bool
	GptossAlpha, GptossLimit float32
}

// w8Model adapts the W8A8 ModelW into the precision-agnostic runModel.
func w8Model(m ModelW) RunModel {
	rm := RunModel{
		FinalNorm: m.FinalNorm.buf,
		LmHead:    m.LMHead,
		GatedGELU: m.GatedGELU,
		KvF16:     m.KVF16,
	}
	bufOf := func(db *DeviceBuffer) *wgpu.Buffer {
		if db == nil {
			return nil
		}
		return db.buf
	}
	for i := range m.Layers {
		lw := &m.Layers[i]
		rm.Layers = append(rm.Layers, RunLayer{
			AttnNorm: lw.Attn.Norm.buf, InvFreq: lw.Attn.InvFreq.buf,
			KCache: lw.Attn.KCache.buf, VCache: lw.Attn.VCache.buf, MlpNorm: lw.MLPNorm.buf,
			Q: lw.Attn.QProj, K: lw.Attn.KProj, V: lw.Attn.VProj, O: lw.Attn.OProj,
			Gate: lw.Gate, Up: lw.Up, Down: lw.Down,
			QGate: lw.Attn.QGate,
			QNorm: bufOf(lw.Attn.QNorm), KNorm: bufOf(lw.Attn.KNorm),
		})
	}
	return rm
}

// NewDecodeRunner builds the persistent plan for a resident W8A8 model.
func (c *Context) NewDecodeRunner(m ModelW, hidden, nH, nKV, hd, inter, start int, eps, scale float32, addOne bool) (*DecodeRunner, error) {
	return c.newDecodeRunner(w8Model(m), hidden, nH, nKV, hd, inter, start, eps, scale, addOne)
}

// attnHeadDimSupported declines a resident decode plan whose model-level or any per-layer head_dim exceeds attnMaxHeadDim,
// the widest the single-query attention kernels serve (see attnWG: one lane per dim, so a head_dim beyond the kernel's
// reach would leave tail dims un-dotted and the o-projection would consume half-zero context, which is plausible-looking
// WRONG output with no error). The caller falls back to the staged/CPU path on this error. MLAAttn guards its own rank
// limit; this covers the softmax/GQA runners, including Gemma 4's per-layer head_dim.
func attnHeadDimSupported(hd int, layers []RunLayer) error {
	if hd > attnMaxHeadDim {
		return fmt.Errorf("gpu: resident decode declines head_dim=%d > %d (attention kernel workgroup is %d-wide)", hd, attnMaxHeadDim, attnMaxHeadDim)
	}
	for i := range layers {
		if ghd := layers[i].Ghd; ghd > attnMaxHeadDim {
			return fmt.Errorf("gpu: resident decode declines layer %d head_dim=%d > %d (attention kernel workgroup)", i, ghd, attnMaxHeadDim)
		}
	}
	return nil
}

// newDecodeRunner builds the persistent decode plan for either precision.
func (c *Context) newDecodeRunner(m RunModel, hidden, nH, nKV, hd, inter, start int, eps, scale float32, addOne bool) (*DecodeRunner, error) {
	if err := validateDecodeModel(m, hd); err != nil {
		return nil, err
	}
	if err := c.ensureDecodePipelines(m, hd); err != nil {
		return nil, err
	}
	half := hd / 2
	if m.RopeHalf > 0 {
		half = m.RopeHalf
	}
	r := &DecodeRunner{
		c: c, vocab: m.LmHead.nRows(), logitsHost: make([]float32, m.LmHead.nRows()),
		nLayers: len(m.Layers), lmHead: m.LmHead, hidden: hidden, pleDim: m.PleDim, actG32: m.ActG32,
	}
	b := &decodeRunnerBuild{
		c: c, m: m, r: r,
		hidden: hidden, nH: nH, nKV: nKV, hd: hd, inter: inter, start: start,
		eps: eps, scale: scale, addOne: addOne, half: half,
		tempOff:   AttnTempOffForTest,
		geomCache: map[[4]int]*attnGeom{},
	}
	b.role = "build"
	b.noAttnSinks = b.storF(1)
	r.loraT = b.storF(loraRMax)
	b.initMamba()
	b.initDeltaNet()
	b.initRoPE()
	b.initMLA()
	b.allocResidualAndPLE()
	if os.Getenv("GOINFER_GPU_CAPTURE") != "" {
		b.captureOn = true
		r.capHidden = hidden
	}
	if m.Moe != nil {
		b.role = "moeZeroBias"
		b.moeZeroBias = b.storF(m.Moe.NE)
	}
	ssmStopLayer := SsmStopLayerForTest
	for i := range m.Layers {
		if ssmStopLayer >= 0 && i > ssmStopLayer {
			break // layer-sweep seam: logits from xd after this layer
		}
		b.emitLayer(i, &m.Layers[i])
	}
	return b.finish()
}

// GeomVariantCount reports how many distinct attention geometries (hd, nKV, rotaryDim) the resident plan built: 1 for
// every uniform-geometry family, 2 for Gemma 4's local/global interleave. Tests assert it is 1 for uniform models.
func (r *DecodeRunner) GeomVariantCount() int { return r.geomVariants }

// writeInputs uploads the per-token input embedding + pos-dependent uniforms (the
// only buffers that vary per call; the fixed dispatch plan reads them). Split out so
// the batched RunN can prime K runners before recording one command buffer. ropePos:
// see posUni's doc comment — equal to pos except for Qwen2.5-VL m-RoPE decode.
func (r *DecodeRunner) writeInputs(x []float32, pos, ropePos int) error {
	if want := r.hidden + len(r.pleIn)*r.pleDim; len(x) != want {
		return fmt.Errorf("gpu: the resident row is %d floats, want %d (hidden %d + %d layers × %d per-layer inputs)", len(x), want, r.hidden, len(r.pleIn), r.pleDim)
	}
	if err := r.c.queue.TryWriteBuffer(r.xd, 0, wgpu.ToBytes(x[:len(x)-len(r.pleIn)*r.pleDim])); err != nil {
		return err
	}
	for i, b := range r.pleIn {
		off := r.hidden + i*r.pleDim
		if err := r.c.queue.TryWriteBuffer(b, 0, wgpu.ToBytes(x[off:off+r.pleDim])); err != nil {
			return err
		}
	}
	for _, pu := range r.posUnis {
		n := pu.fill(&r.uniScratch, pos, ropePos)
		if err := r.c.queue.TryWriteBuffer(pu.buf, 0, wgpu.ToBytes(r.uniScratch[:n])); err != nil {
			return err
		}
	}
	return nil
}

// ReadMambaCap copies the first mamba layer's captured proj/conv/y/gated buffers (their values from the most recent Run)
// back to the host, the resident's actual per-token kernel I/O, for diffing against mamba2Step
// (gpu/mamba_resident_capture_test.go). projN/convN/dInner are the element counts. For tests, but exported: it allocates
// fresh staging per call and returns an error, never panics, on a failed buffer map.
func (r *DecodeRunner) ReadMambaCap(projN, convN, dInner int) (proj, conv, y, gated []float32, err error) {
	rd := func(b *wgpu.Buffer, n int) ([]float32, error) {
		stag, err := r.c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		if err != nil {
			return nil, fmt.Errorf("gpu: ReadMambaCap: stag: %w", err)
		}
		defer releaseBuf(stag)
		enc, err := r.c.device.TryCreateCommandEncoder(nil)
		if err != nil {
			return nil, fmt.Errorf("gpu: ReadMambaCap: command encoder: %w", err)
		}
		defer enc.Release()
		if err := enc.TryCopyBufferToBuffer(b, 0, stag, 0, uint64(n*4)); err != nil {
			return nil, fmt.Errorf("gpu: ReadMambaCap: copy: %w", err)
		}
		cmd, err := enc.TryFinish(nil)
		if err != nil {
			return nil, fmt.Errorf("gpu: ReadMambaCap: finish: %w", err)
		}
		r.c.queue.Submit(cmd)
		cmd.Release()
		st := wgpu.MapAsyncStatus(0)
		stag.TryMapAsync(wgpu.MapModeRead, 0, uint64(n*4), func(s wgpu.MapAsyncStatus) { st = s })
		r.c.device.Poll(true, nil)
		if st != wgpu.MapAsyncStatusSuccess {
			return nil, fmt.Errorf("gpu: ReadMambaCap: buffer map failed (status %v)", st)
		}
		out := make([]float32, n)
		copy(out, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(n*4))))
		stag.TryUnmap()
		return out, nil
	}
	if proj, err = rd(r.mcapProj, projN); err != nil {
		return nil, nil, nil, nil, err
	}
	if conv, err = rd(r.mcapConv, convN); err != nil {
		return nil, nil, nil, nil, err
	}
	if y, err = rd(r.mcapY, dInner); err != nil {
		return nil, nil, nil, nil, err
	}
	if gated, err = rd(r.mcapGated, dInner); err != nil {
		return nil, nil, nil, nil, err
	}
	return proj, conv, y, gated, nil
}

// ReadCapture copies the per-layer capture buffers (GOINFER_GPU_CAPTURE=1 at build — see the
// capCtx/capAttn/capMLP field comment) back to the host, as recorded by the most recent Run:
// ctx[l] is layer l's attention context (pre-o-proj, [nH*hd]), attn[l] the residual stream after
// its attention add, mlp[l] the residual after its MLP add ([hidden] each). Layers that take a
// non-generic mixer (Mamba/DeltaNet/MLA/Nemotron kinds) record no ctx/attn entry, so the three
// slices are indexed by generic-attention-layer order for ctx/attn and by plan-layer order for
// mlp. Returns an error when capture was not enabled or a buffer map fails.
func (r *DecodeRunner) ReadCapture() (ctx, attn, mlp [][]float32, err error) {
	if r.capHidden == 0 {
		return nil, nil, nil, fmt.Errorf("gpu: ReadCapture: runner was built without GOINFER_GPU_CAPTURE")
	}
	rd := func(b *wgpu.Buffer, n int) ([]float32, error) {
		stag, e := r.c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		if e != nil {
			return nil, e
		}
		defer releaseBuf(stag)
		enc, e := r.c.device.TryCreateCommandEncoder(nil)
		if e != nil {
			return nil, e
		}
		defer enc.Release()
		if e := enc.TryCopyBufferToBuffer(b, 0, stag, 0, uint64(n*4)); e != nil {
			return nil, e
		}
		cmd, e := enc.TryFinish(nil)
		if e != nil {
			return nil, e
		}
		defer cmd.Release()
		r.c.queue.Submit(cmd)
		st := wgpu.MapAsyncStatus(0)
		if e := stag.TryMapAsync(wgpu.MapModeRead, 0, uint64(n*4), func(s wgpu.MapAsyncStatus) { st = s }); e != nil {
			return nil, e
		}
		r.c.device.Poll(true, nil)
		if st != wgpu.MapAsyncStatusSuccess {
			return nil, fmt.Errorf("gpu: ReadCapture: buffer map failed (status %v)", st)
		}
		out := make([]float32, n)
		copy(out, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(n*4))))
		stag.TryUnmap()
		return out, nil
	}
	read := func(bufs []*wgpu.Buffer, n func(i int) int) ([][]float32, error) {
		out := make([][]float32, len(bufs))
		for i, b := range bufs {
			v, e := rd(b, n(i))
			if e != nil {
				return nil, e
			}
			out[i] = v
		}
		return out, nil
	}
	hiddenW := func(int) int { return r.capHidden }
	if ctx, err = read(r.capCtx, func(i int) int { return r.capCtxW[i] }); err != nil {
		return nil, nil, nil, err
	}
	if attn, err = read(r.capAttn, hiddenW); err != nil {
		return nil, nil, nil, err
	}
	if mlp, err = read(r.capMLP, hiddenW); err != nil {
		return nil, nil, nil, err
	}
	return ctx, attn, mlp, nil
}

// record appends this runner's dispatch plan to an existing compute pass. The plan reads r.xd and
// r.posUnis (set by writeInputs) and the resident weights + KV caches, leaving logits in r.lastLogits.
// WebGPU inserts the storage barriers between data-dependent dispatches; across batched runners sharing
// one KV cache, a row's kv-store thus correctly precedes a later row's attention read.
func (r *DecodeRunner) record(pass *wgpu.ComputePassEncoder) int {
	for _, s := range r.steps {
		pass.SetPipeline(s.Pl)
		pass.SetBindGroup(0, s.Bg, nil)
		pass.DispatchWorkgroups(s.Gx, s.Gy, 1)
	}
	n := len(r.steps)
	if CostPlantForTest == 1 && n > 0 { // the cost plant: the first step recorded twice
		s := r.steps[0]
		pass.SetPipeline(s.Pl)
		pass.SetBindGroup(0, s.Bg, nil)
		pass.DispatchWorkgroups(s.Gx, s.Gy, 1)
		n++
	}
	return n
}

// recordTrunk records all dispatches up to the final norm, omitting the last step (the LM head GEMV).
func (r *DecodeRunner) recordTrunk(pass *wgpu.ComputePassEncoder) int {
	steps := r.steps
	if len(steps) > 0 {
		steps = steps[:len(steps)-1]
	}
	for _, s := range steps {
		pass.SetPipeline(s.Pl)
		pass.SetBindGroup(0, s.Bg, nil)
		pass.DispatchWorkgroups(s.Gx, s.Gy, 1)
	}
	return len(steps)
}

// RunNoLogits executes all trunk layers and populates the KV cache for token at position pos,
// skipping the LM head GEMV dispatch, logits staging copy, MapAsync, and readback.
func (r *DecodeRunner) RunNoLogits(x []float32, pos, ropePos int) error {
	r.dropPending()
	c := r.c
	tw := time.Now()
	if err := r.writeInputs(x, pos, ropePos); err != nil {
		return err
	}
	r.TWrite = time.Since(tw)
	te := time.Now()
	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	n := r.recordTrunk(pass)
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return fmt.Errorf("gpu: end compute pass: %w", err)
	}
	pass.Release()
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		return err
	}
	defer cmd.Release()
	r.TEncode = time.Since(te)
	c.submit(cmd, n)
	if r.Cost != nil {
		r.countSteps(r.Cost, r.CostPhase, r.steps[:n])
	}
	c.device.Poll(false, nil)
	return nil
}

// Run executes the plan for one token at absolute position pos. x is the token's
// input embedding [hidden]; returns the logits [vocab]. One Submit + one Poll.
// ropePos: see posUni's doc comment — equal to pos except for Qwen2.5-VL m-RoPE decode.
func (r *DecodeRunner) Run(x []float32, pos, ropePos int) ([]float32, error) {
	c := r.c
	tw := time.Now()
	if err := r.writeInputs(x, pos, ropePos); err != nil {
		return nil, err
	}
	r.TWrite = time.Since(tw)
	te := time.Now()
	cmd, n := r.pending, r.pendingN
	r.pending = nil
	if cmd == nil || EncodeAheadOffForTest {
		if cmd != nil {
			cmd.Release()
		}
		var err error
		if cmd, n, err = r.recordToken(); err != nil {
			return nil, err
		}
	}
	defer cmd.Release()
	r.TEncode = time.Since(te)
	ts := time.Now()
	c.submit(cmd, n)
	if r.Cost != nil {
		r.countSteps(r.Cost, r.CostPhase, r.steps)
	}
	st := wgpu.MapAsyncStatus(0)
	if err := r.stag.TryMapAsync(wgpu.MapModeRead, 0, uint64(r.vocab*4), func(s wgpu.MapAsyncStatus) { st = s }); err != nil {
		return nil, err
	}
	// KS-E3: the next token's buffer is recorded while the GPU runs this one, after the staging map is requested; the next
	// Run writes its inputs and submits it. A recording that fails is retried in line by that Run.
	r.TEncodeAhead = 0
	if !EncodeAheadOffForTest {
		ta := time.Now()
		if next, nn, err := r.recordToken(); err == nil {
			r.pending, r.pendingN = next, nn
		}
		r.TEncodeAhead = time.Since(ta)
	}
	c.wait()
	r.TSync = time.Since(ts) - r.TEncodeAhead
	if st != wgpu.MapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: DecodeRunner map failed: %v", st)
	}
	copy(r.logitsHost, wgpu.FromBytes[float32](r.stag.GetMappedRange(0, uint(r.vocab*4))))
	r.stag.TryUnmap()
	return r.logitsHost, nil
}

// RunBatch executes K runners (sharing the resident weights + KV caches, distinct scratch) over inputs xs[i] at positions
// startPos+i in ONE command buffer: one Submit, one Poll, K logit rows. The runners' steps are recorded in row order into a
// single compute pass, so each row's kv-store is visible to the next row's attention (causal: row i sees positions
// [0, startPos+i]). This amortizes the encode glue and the sync over K but does not make the per-row GPU work cheaper (see
// residentDecoder.VerifyPath). len(runners) must be ≥ len(xs).
func RunBatch(c *Context, runners []*DecodeRunner, xs [][]float32, startPos int) ([][]float32, error) {
	n := len(xs)
	if n == 0 {
		return nil, nil
	}
	for i := range n {
		if err := runners[i].writeInputs(xs[i], startPos+i, startPos+i); err != nil {
			return nil, err
		}
	}
	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	recorded := 0
	for i := range n {
		recorded += runners[i].record(pass)
	}
	// Checked rather than discarded, as in Run.
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return nil, fmt.Errorf("gpu: end compute pass: %w", err)
	}
	pass.Release()
	for i := range n {
		if err := enc.TryCopyBufferToBuffer(runners[i].lastLogits, 0, runners[i].stag, 0, uint64(runners[i].vocab*4)); err != nil {
			return nil, fmt.Errorf("gpu: copy logits→stage for runner %d: %w", i, err)
		}
	}
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		return nil, err
	}
	defer cmd.Release()
	c.submit(cmd, recorded)
	if k := runners[0].Cost; k != nil { // the batch is counted on its first runner's counter, which the resident attaches
		for i := range n {
			runners[i].countSteps(k, runners[0].CostPhase, runners[i].steps)
		}
	}
	sts := make([]wgpu.MapAsyncStatus, n)
	var mapErr error
	for i := range n {
		sts[i] = wgpu.MapAsyncStatus(0)
		if err := runners[i].stag.TryMapAsync(wgpu.MapModeRead, 0, uint64(runners[i].vocab*4), func(s wgpu.MapAsyncStatus) { sts[i] = s }); err != nil {
			mapErr = fmt.Errorf("gpu: runBatch row %d MapAsync: %w", i, err)
			break // later rows are not requested; the ones already requested settle on the Poll below
		}
	}
	c.wait() // one sync settles every requested map (success or not)
	// runners[i].stag is a PERSISTENT per-runner buffer. Returning while any stag is still mapped leaves it mapped forever, and
	// every future MapAsync on it fails, poisoning the runner for the process lifetime. consumed[i] marks rows the readback
	// already Unmapped; this deferred sweep Unmaps any that mapped but were not consumed, on ALL return paths.
	consumed := make([]bool, n)
	defer func() {
		for i := range n {
			if sts[i] == wgpu.MapAsyncStatusSuccess && !consumed[i] {
				runners[i].stag.TryUnmap()
			}
		}
	}()
	if mapErr != nil {
		return nil, mapErr
	}
	out := make([][]float32, n)
	for i := range n {
		if sts[i] != wgpu.MapAsyncStatusSuccess {
			return nil, fmt.Errorf("gpu: runBatch row %d map failed: %v", i, sts[i])
		}
		row := make([]float32, runners[i].vocab)
		copy(row, wgpu.FromBytes[float32](runners[i].stag.GetMappedRange(0, uint(runners[i].vocab*4))))
		runners[i].stag.TryUnmap()
		consumed[i] = true
		out[i] = row
	}
	return out, nil
}

func (r *DecodeRunner) release() {
	r.dropPending()
	releaseLoraLayers(r.loraLayers) // an adapter still bound is the runner's too: freed with it, not left to the GC
	r.loraLayers = nil
	for _, f := range r.keep {
		f()
	}
	r.keep = nil
}

// Close frees the runner's scratch (not the resident model).
func (r *DecodeRunner) Close() error {
	r.c.releaseOwned(r.release)
	return nil
}

// EncodeAheadOffForTest makes Run record each token's command buffer in line, as before KS-E3 (the old arm of its A/B); read at
// each Run.
var EncodeAheadOffForTest bool

// recordToken records one token's command buffer: every step in one compute pass, then the logits copy to the staging buffer.
func (r *DecodeRunner) recordToken() (*wgpu.CommandBuffer, int, error) {
	enc, err := r.c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return nil, 0, err
	}
	defer enc.Release()
	// One compute pass for the whole token: WebGPU runs the dispatches in record order and the backend inserts the minimal
	// storage-buffer barriers between data-dependent dispatches. The KV appends are compute kernels (rope-store / kv-store),
	// so nothing forces a pass break.
	pass := enc.BeginComputePass(nil)
	n := r.record(pass)
	// TryEnd and the logits copy are checked, not discarded, so a validation error names its call instead of surfacing later
	// as an opaque "DecodeRunner map failed".
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return nil, 0, fmt.Errorf("gpu: end compute pass: %w", err)
	}
	pass.Release()
	if err := enc.TryCopyBufferToBuffer(r.lastLogits, 0, r.stag, 0, uint64(r.vocab*4)); err != nil {
		return nil, 0, fmt.Errorf("gpu: copy logits→stage: %w", err)
	}
	cmd, err := enc.TryFinish(nil)
	return cmd, n, err
}

// dropPending releases the pre-recorded next-token buffer, if any.
func (r *DecodeRunner) dropPending() {
	if r.pending != nil {
		r.pending.Release()
		r.pending = nil
	}
}

// DnetRunParams carries the model-level Gated-DeltaNet geometry (uniform across the linear
// layers). keyDim/valueDim/convDim are derived once here rather than at every dispatch, because
// the three of them are easy to conflate: convDim is 2*keyDim+valueDim (the conv runs over
// [q|k|v] together), and rep = nv/nk is the GVA factor mapping value heads to key heads.
type DnetRunParams struct {
	ConvK      int // depthwise causal conv width
	Hk, Hv     int // per-head key/query and value dims
	Nk, Nv     int // key-head and value-head counts
	Rep        int // nv/nk — value heads sharing one key head
	KeyDim     int // nk*hk
	ValueDim   int // nv*hv
	ConvDim    int // 2*keyDim + valueDim
	StateElems int // nv*hv*hk, the per-layer recurrent state
	Eps        float32
	BetaScale  float32 // write-gate scale in {1, 2} (2 for Olmo Hybrid's allow_neg_eigval); 0 defaults to 1
}

// G4DropVNormForTest turns off the v_norm that Gemma 4 layers with their own v_proj apply (runLayer.vNorm), so a test can
// show that v_norm is what moves the parity numbers. Set only by a test
// seam; never in production.
var G4DropVNormForTest bool

// UploadKVModelGeomForTest makes UploadKV size and group every layer's rows by the model's geometry, ignoring a layer's
// own, so a test can show that reads wrong on a model with two.
var UploadKVModelGeomForTest bool

// Planted defects for the Gemma 4 E-model gate (gpu/gemma4_emodel_test.go); each is off outside it.
var (
	G4SkipPLEForTest         bool // the per-layer-embedding branch skipped
	G4PLEOffByOneForTest     bool // a layer reads the next layer's per-layer inputs
	G4KVSrcOffForTest        bool // a shared layer reads the previous owning layer of its type, not its source
	G4OneFFNWidthForTest     bool // every dense layer's gated product covers only the model's one width
	G4DropLayerScalarForTest bool // the per-layer output scalar dropped
)
