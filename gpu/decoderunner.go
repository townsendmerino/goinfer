//go:build gpu

package gpu

import (
	"fmt"
	"math"
	"os"
	"slices"
	"time"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// ssmStopLayerForTest truncates a newly built decode plan after this layer (-1 = every layer): the
// resident-SSM layer-sweep seam, set only by tests in this package (mamba_layersweep_test.go).
var ssmStopLayerForTest = -1

// DecodeRunner is the production one-command-buffer decode forward: it builds every
// scratch buffer + bind group ONCE (the per-token allocation that made
// DecodeTokenFused slow), then Run only WriteBuffers the input + the pos-dependent
// uniforms (RoPE pos, attention nKeys) and re-records the fixed dispatch plan into
// a fresh encoder — one Submit, one Poll, logits back. This is the GEMVRunner
// pattern applied to the whole token graph.
type DecodeRunner struct {
	c                    *Context
	steps                []runStep
	posUnis              []posUni
	xd, stag, lastLogits *wgpu.Buffer
	lmHead               decodeWeight // kept for LMHeadForTest; the gemv() closures already capture it too
	smp                  *gumbelState // device Gumbel-max sampling resources, built on first RunSample
	vocab                int
	uniScratch           [16]uint32
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

	// Debug capture of the FIRST mamba layer's intermediate buffers (in_proj output,
	// conv output, SSM output y, mixer output gated), recorded at build. ReadMambaCap
	// copies the LAST run's values for the resident-vs-mamba2Step wiring diff
	// (gpu/mamba_resident_capture_test.go). nil for non-hybrid; zero production cost.
	mcapProj, mcapConv, mcapY, mcapGated *wgpu.Buffer

	// Per-layer residual-stream capture, built only when GOINFER_GPU_CAPTURE is set (nothing is allocated or recorded
	// otherwise): for each generic-attention layer, the attention context (pre-o-proj, [nH*hd]) and the residual after the
	// attention add and after the MLP add ([hidden] each), copied by an extra kv-store dispatch recorded into the plan. They
	// hold the LAST run's values; ReadCapture reads them back. The WebGPU twin of decoder.Model.ForwardSubCapture, so a
	// resident-vs-CPU divergence can be localised to a sublayer of a layer in one run
	// (docs/completed/task-webgpu-nogqa-decode-bug.md).
	capCtx, capAttn, capMLP []*wgpu.Buffer
	capCtxN, capHidden      int

	// Compute-time LoRA (lora_resident.go). baseSteps is the pristine adapter-free plan: SetAdapter never mutates it, only
	// rebuilds r.steps from it plus loraHooks (one per (layer, projection) the dense forward exposes, recorded in strictly
	// increasing afterIdx order), so clearing an adapter is a cheap restore. loraLayers is nil until SetAdapter binds one.
	baseSteps  []runStep
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

type runStep struct {
	pl     *wgpu.ComputePipeline
	bg     *wgpu.BindGroup
	gx, gy uint32
}

// Nemotron-H per-layer single-op kinds. nemoNone (0) ⇒ the layer runs the standard
// mixer+FFN path (every other resident family); the others run exactly one op + residual.
const (
	nemoNone uint8 = iota
	nemoKMamba
	nemoKAttn
	nemoKMLP
	// nemoKMoE is Nemotron-H's fourth block kind: routed NON-gated relu² experts (up_proj/down_proj only, unlike
	// moeMLP's gated SwiGLU) plus an always-on ungated shared expert of the same shape. Single-op-per-block like the other
	// three: no mixer, no separate FFN pass.
	nemoKMoE
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

// runLayer / runModel are the DecodeRunner's precision-agnostic view of a resident
// model: the f32 buffers (norms, RoPE freqs, KV caches) plus the projection
// weights as decodeWeight (W8A8 or W4A8). The public constructors adapt a concrete
// ModelW / ModelW4 into this; the builder below works the same for either.
type runLayer struct {
	attnNorm, invFreq, kCache, vCache, mlpNorm *wgpu.Buffer
	kScale, vScale                             *wgpu.Buffer // int8-KV per-(pos,head) scales; nil unless kvI8
	q, k, v, o, gate, up, down                 decodeWeight
	qBias, kBias, vBias                        *wgpu.Buffer // optional (Qwen2); nil ⇒ no bias
	qNorm, kNorm                               *wgpu.Buffer // optional per-head QK-norm weights [hd] (Qwen3/GLM); nil ⇒ none

	// gpt-oss's o-proj bias (FeatOutBias) and Gemma's sandwich post-sublayer norms (FeatSandwichNorm); nil ⇒ off. Both
	// defeat the fused gemvAdd residual epilogue: the o-proj/down-proj GEMV runs bare, then the bias/norm applies, THEN a
	// separate residual add via biasAdd (xd += out).
	oBias                     *wgpu.Buffer // gpt-oss o_proj bias [hidden]; nil ⇒ none
	postAttnNorm, postMLPNorm *wgpu.Buffer // Gemma sandwich post-sublayer norms [hidden]; nil ⇒ none

	// attnSinks/hasSink: gpt-oss's per-head learned attention sink [nH] (FeatAttnSink). Unlike every other "nil ⇒ off" field
	// here, attnSinks is ALWAYS bound (a one-element dummy for every other family) and hasSink is the on/off flag, because
	// WGSL bind groups cannot bind a null storage buffer. This is Metal's convention, not CUDA's null-pointer sentinel.
	attnSinks *wgpu.Buffer
	hasSink   bool
	isLocal   bool    // sliding-window (local) attention layer; false ⇒ full
	ropeScale float32 // per-layer RoPE cos/sin scale = mscale; 0 ⇒ 1.0

	// Per-layer attention geometry: this layer's head_dim / KV-heads / rotaryDim-half and attention_k_eq_v. Zero ghd ⇒ the
	// model-level nH-relative shape; gKEqV is read independently (a K=V layer always sets its full tuple). The plan loop
	// resolves these into a shared *attnGeom via geomFor; this layer's kvDim is gnKV*ghd.
	ghd, gnKV, ghalf int
	gKEqV            bool
	layerScalar      float32 // Gemma 4 dense per-layer output scalar (1 if absent/unscaled)
	// vNorm: a Gemma 4 layer with its own v_proj (not K=V) applies the scale-less v_norm to the v_proj output, as the CPU
	// does on every K/V-owning layer; K=V layers always do (gKEqV). See g4DropVNormForTest.
	vNorm bool

	// MoE: when isMoE, this layer's FFN is a sparse mixture of experts instead of the dense gate/up/down above. router scores
	// all nE experts; the on-GPU top-k (moeRoute) writes the chosen indices/weights, then k indexed GEMVs per projection read
	// the right expert out of the stacked buffers (expGate/expUp/expDown) and the down-combine folds the router weight.
	isMoE                   bool
	router                  decodeWeight         // [nE, hidden] router logits
	routerBias              *wgpu.Buffer         // [nE] selection bias (DeepSeek/GLM); nil ⇒ none
	expGate, expUp, expDown *ResidentStackedW8A8 // nE experts stacked per projection
	// gpt-oss (FeatAttnSink): expert-major bias tables, only populated when moeRunParams.gptoss.
	// gateUpBias is [nE*2*inter] (gate then up per expert); downBias is [nE*hidden].
	gateUpBias, downBias *wgpu.Buffer

	// Always-on shared expert: nil shGate ⇒ none. shGateW is the [1,hidden] sigmoid gate of the gated combine; nil ⇒ the
	// shared expert is added ungated (plain residual).
	shGate, shUp, shDown, shGateW decodeWeight

	// MLA attention. Populated when runModel.mla != nil, in which case the runner takes the latent-attention path instead of
	// the q/k/v/o block above. mlaQA/mlaQANorm/mlaQB are the q-LoRA bottleneck (nil mlaQA ⇒ the direct mlaQ); mlaKVA
	// down-projects to the latent, mlaKVANorm normalizes it; mlaWUK/mlaWUV are the per-head absorb/lift f32 weights; mlaO is
	// the output projection; latCache is this layer's compressed-latent KV cache [ctxCap*latDim].
	mlaQA, mlaQB, mlaQ, mlaKVA, mlaO decodeWeight
	mlaQANorm, mlaKVANorm            *wgpu.Buffer
	mlaWUK, mlaWUV                   *wgpu.Buffer
	latCache                         *wgpu.Buffer

	// Mamba-2 SSM mixer. When isMamba, this layer's sequence-mixer is the resident SSM step (mamba.go kernels) instead of
	// attention: in/out_proj are W8A8; convW/convB/headP/normW are f32 resident; win (causal-conv ring) + ssm (selective state)
	// are build-once persistent state, updated in place per token and reset per generation. ResidMul is folded into
	// mambaOutProj's scale (the residual add). isMamba=false ⇒ attention layer (above).
	isMamba bool
	// Nemotron-H single-op-per-block: each layer is exactly ONE op (no mixer+FFN pairing).
	// nemoKind ∈ {nemoNone, nemoKMamba, nemoKAttn, nemoKMLP}; nemoNone ⇒ standard mixer+FFN.
	nemoKind                                       uint8
	mambaInProj, mambaOutProj                      decodeWeight
	mambaConvW, mambaConvB, mambaHeadP, mambaNormW *wgpu.Buffer
	mambaWin, mambaSSM                             *wgpu.Buffer

	// Gated-DeltaNet mixer. When isDeltaNet, this layer's mixer is the recurrent delta rule (deltanet.go kernels) instead of
	// attention. The causal conv is Mamba-2's (same shape, same SiLU, same ring window), so it reuses mambaConvW/mambaWin and
	// binds an all-zero convB (DeltaNet's conv is bias-free). dnState is the [nv*hv*hk] recurrent state, TRANSPOSED relative
	// to the CPU's [hk,hv] so each thread owns a contiguous row; build-once, updated in place, reset per generation alongside
	// mambaWin.
	isDeltaNet                            bool
	dnQKV, dnZ, dnOut                     decodeWeight // the three dominant projections, quantized
	dnB, dnA                              decodeWeight // the two small gate projections
	dnDtBias, dnNegExpA, dnNormW, dnState *wgpu.Buffer

	// qGate marks a full-attention layer whose q_proj is DOUBLE WIDTH — [query ‖ gate] per
	// head — with the context scaled by sigmoid(gate) before o_proj (attn_output_gate). The
	// weight stays fused because it is quantized; the split happens on the activation.
	qGate bool
}

type runModel struct {
	layers        []runLayer
	finalNorm     *wgpu.Buffer
	lmHead        decodeWeight
	kvF16         bool            // KCache/VCache are f16-packed (NewKVCacheF16) → use the f16 kernels
	kvI8          bool            // KCache/VCache are int8-packed (NewKVCacheI8) + scales → int8 kernels
	moe           *moeRunParams   // non-nil ⇒ the model has MoE layers (runLayer.isMoE picks which)
	mla           *mlaRunParams   // non-nil ⇒ MLA latent attention replaces the q/k/v/o block
	mamba         *mambaRunParams // non-nil ⇒ hybrid: some layers (runLayer.isMamba) are SSM mixers
	dnet          *dnetRunParams  // non-nil ⇒ hybrid: some layers (runLayer.isDeltaNet) are DeltaNet mixers
	ropeHalf      int             // rotated pairs per head = rotaryDim/2 (partial RoPE); 0 ⇒ HeadDim/2
	slidingWindow int             // >0 ⇒ local layers attend only the last N positions
	// gatedGELU (FeatGatedGELU) selects gelu_tanh(gate)·up over the default SiLU in the gated MLP; model-level like
	// ropeHalf/slidingWindow, since every gated-MLP family shares one activation. It is deliberately a bool with SiLU as the
	// zero value, NOT decoder.ActKind's ordinal (where 0 = GELU-tanh): a runModel built by hand (every gpu/*_test.go that
	// does not go through BuildResident) leaves it at its Go zero value, and that must stay the SiLU behavior those tests
	// assume. BuildResident sets it to `m.GatedActResident() == 0`. Why: docs/code-notes/gpu.md#runModel.gatedGELU.
	gatedGELU bool
}

// mlaRunParams carries the model-level MLA geometry (uniform across layers). Per-layer
// weights + the latent cache live on runLayer. latDim = kvLoRARank + qkRope is the
// cached payload width; qkHead = qkNope + qkRope is the per-head q·k width.
type mlaRunParams struct {
	qLoRARank      int     // q_a bottleneck width; 0 ⇒ direct q_proj (V2-Lite)
	kvLoRARank     int     // rank of the compressed KV latent (the score/value body)
	qkNope, qkRope int     // per-head no-rope / rope q·k dims
	vHead          int     // per-head value width (≠ qkNope+qkRope)
	interleave     bool    // V3 GPT-J pairwise RoPE layout (vs plain NeoX)
	ropeScale      float32 // YaRN attention factor folded into cos/sin (1.0 when none)
}

// mambaRunParams carries the model-level Mamba-2 geometry (uniform across mamba layers).
// dInner=nHeads·P, convDim=dInner+2·nGroups·N, projDim=2·dInner+2·nGroups·N+nHeads,
// gSize=nGroups·N, repeat=nHeads/nGroups. normGroups is the gated-RMSNorm group count
// (1 for Granite). The dispatches are pos-independent — the {win, ssm} state carries the
// recurrence, so no posUni.
type mambaRunParams struct {
	nHeads, hp, dn, nGroups, dConv          int
	dInner, convDim, projDim, gSize, repeat int
	normGroups                              int
}

// moeRunParams carries the model-level MoE selection knobs (uniform across layers):
// the router top-k shape + scoring flavor. Per-layer data (router/expert weights,
// selection bias) lives on runLayer. inter is the per-expert FFN width.
type moeRunParams struct {
	nE, k, inter      int
	sigmoid, norm     bool
	scale             float32
	sharedInter       int  // shared-expert FFN width (qwen2_moe / GLM); 0 ⇒ no shared expert
	sharedUngated     bool // GLM/DeepSeek add the shared expert with no sigmoid gate
	nGroup, topkGroup int  // DeepSeek group-limited routing; nGroup ≤ 1 ⇒ plain global top-k
	// gptoss selects a THIRD MoE dispatch shape, beside the plain top-k and DeepSeek's group-limited routing: its own router
	// contract (biased logits select AND weight), clamped-gated activation (gptossAlpha/gptossLimit, uniform across every
	// gpt-oss layer, from decoder.Model.GptOssActResident) and biased down-combine.
	gptoss                   bool
	gptossAlpha, gptossLimit float32
}

// w8Model adapts the W8A8 ModelW into the precision-agnostic runModel.
func w8Model(m ModelW) runModel {
	rm := runModel{
		finalNorm: m.FinalNorm.buf,
		lmHead:    m.LMHead,
		gatedGELU: m.GatedGELU,
		kvF16:     m.KVF16,
	}
	bufOf := func(db *DeviceBuffer) *wgpu.Buffer {
		if db == nil {
			return nil
		}
		return db.buf
	}
	for i := range m.Layers {
		lw := &m.Layers[i]
		rm.layers = append(rm.layers, runLayer{
			attnNorm: lw.Attn.Norm.buf, invFreq: lw.Attn.InvFreq.buf,
			kCache: lw.Attn.KCache.buf, vCache: lw.Attn.VCache.buf, mlpNorm: lw.MLPNorm.buf,
			q: lw.Attn.QProj, k: lw.Attn.KProj, v: lw.Attn.VProj, o: lw.Attn.OProj,
			gate: lw.Gate, up: lw.Up, down: lw.Down,
			qGate: lw.Attn.QGate,
			qNorm: bufOf(lw.Attn.QNorm), kNorm: bufOf(lw.Attn.KNorm),
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
func attnHeadDimSupported(hd int, layers []runLayer) error {
	if hd > attnMaxHeadDim {
		return fmt.Errorf("gpu: resident decode declines head_dim=%d > %d (attention kernel workgroup is %d-wide)", hd, attnMaxHeadDim, attnMaxHeadDim)
	}
	for i := range layers {
		if ghd := layers[i].ghd; ghd > attnMaxHeadDim {
			return fmt.Errorf("gpu: resident decode declines layer %d head_dim=%d > %d (attention kernel workgroup)", i, ghd, attnMaxHeadDim)
		}
	}
	return nil
}

// newDecodeRunner builds the persistent decode plan for either precision.
func (c *Context) newDecodeRunner(m runModel, hidden, nH, nKV, hd, inter, start int, eps, scale float32, addOne bool) (*DecodeRunner, error) {
	// Decline (the caller falls back to the staged/CPU path, as for a VRAM-exhaustion decline) a plan whose head_dim the
	// single-query attention kernels cannot cover rather than emit plausible-looking wrong output; see attnHeadDimSupported.
	// MLA is exempt from that check: its attention runs the mlaAttn kernel family with its own rank-bounded accumulator, and
	// applying the GQA guard to it once regressed every MLA checkpoint off residency. It is held to the per-lane rank cap
	// MLAAttn itself checks (rank <= 1024). History: docs/code-notes/gpu.md#newDecodeRunner.guards.
	if m.mla != nil {
		if m.mla.kvLoRARank > 1024 {
			return nil, fmt.Errorf("gpu: newDecodeRunner: MLA kv-LoRA rank %d exceeds the resident per-lane cap 1024; declining to CPU", m.mla.kvLoRARank)
		}
	} else if err := attnHeadDimSupported(hd, m.layers); err != nil {
		return nil, err
	}
	// The mamba causal-conv kernel holds its window in a fixed `array<f32, 8>` indexed by conv_kernel-1, so a conv_kernel > 8
	// would overrun it (and 0 underflow): plausible-looking WRONG output, no error. Decline to CPU instead (real Mamba-2 uses
	// conv_kernel 4).
	if m.mamba != nil && (m.mamba.dConv > 8 || m.mamba.dConv < 1) {
		return nil, fmt.Errorf("gpu: newDecodeRunner: mamba conv_kernel %d out of range [1,8] for the resident conv kernel; declining to CPU", m.mamba.dConv)
	}
	ssmStopLayer := ssmStopLayerForTest                  // layer-sweep seam (tests only): truncate the plan
	ssmSkipFFN := os.Getenv("GOINFER_SSM_SKIPFFN") != "" // debug: mixer-only isolation
	ensures := []func() error{c.ensureGEMV, c.ensureGEMVBias, c.ensureQuantize, c.ensureLayer, c.ensureAttn, c.ensureFuse, c.ensureGEMVW4, c.ensureQKNorm, c.ensureLora}
	if m.moe != nil {
		ensures = append(ensures, c.ensureMoERoute, c.ensureMoEExpert, c.ensureMoEExpertW4)
		if m.moe.sharedInter > 0 && !m.moe.sharedUngated {
			ensures = append(ensures, c.ensureSharedGate)
		}
		if m.moe.gptoss { // FeatAttnSink's MoE kernels
			ensures = append(ensures, c.ensureRouteGptOss, c.ensureGptOssGluQuant, c.ensureMoEExpertGptOssDown, c.ensureMoEExpertGptOssDownW4)
		}
	}
	if m.mla != nil {
		ensures = append(ensures, c.ensureMLAStore, c.ensureMLAHeadMV, c.ensureMLAQRope, c.ensureMLAAttn)
	}
	if m.mamba != nil {
		ensures = append(ensures, c.ensureMambaConv, c.ensureMambaSSM, c.ensureMambaGNorm, c.ensureRelu2)
	}
	if hd > attnWG || slices.ContainsFunc(m.layers, func(l runLayer) bool { return l.ghd > attnWG }) {
		// Only for a plan that needs it. A device that cannot do 256-invocation workgroups errors here, which the caller turns
		// into a staged fallback.
		ensures = append(ensures, c.ensureAttnWide)
	}
	if slices.ContainsFunc(m.layers, func(l runLayer) bool { return l.gKEqV }) {
		ensures = append(ensures, c.ensureVNorm)
	}
	if slices.ContainsFunc(m.layers, func(l runLayer) bool { return l.layerScalar != 0 && l.layerScalar != 1.0 }) {
		ensures = append(ensures, c.ensureScaleVec)
	}
	if m.dnet != nil {
		// mambaConv is shared with the SSM engine (DeltaNet's causal conv is the same op); the
		// other four are DeltaNet's own. deltaQSplit/deltaAttnGate serve the family's SOFTMAX
		// layers, which are part of the same admission and so are compiled unconditionally with it.
		ensures = append(ensures, c.ensureMambaConv, c.ensureDeltaRule, c.ensureDeltaNorm,
			c.ensureDeltaGates, c.ensureDeltaGNorm, c.ensureDeltaQSplit, c.ensureDeltaAttnGate)
	}
	for _, e := range ensures {
		if err := e(); err != nil {
			return nil, err
		}
	}
	r := &DecodeRunner{c: c, vocab: m.lmHead.nRows(), logitsHost: make([]float32, m.lmHead.nRows()), nLayers: len(m.layers), lmHead: m.lmHead}
	// buildErr holds the FIRST device-allocation/bind failure: the storF/uni/bind helpers short-circuit once it is set and the
	// constructor returns it, so VRAM exhaustion is an error the caller can fall back on, never a panic in library code.
	var buildErr error
	keepBuf := func(b *wgpu.Buffer) *wgpu.Buffer {
		if b != nil {
			r.keep = append(r.keep, b.Release)
		}
		return b
	}
	keepBG := func(b *wgpu.BindGroup) *wgpu.BindGroup { r.keep = append(r.keep, b.Release); return b }
	storF := func(n int) *wgpu.Buffer {
		if buildErr != nil {
			return nil
		}
		b, e := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
		if e != nil {
			buildErr = e
			return nil
		}
		return keepBuf(b)
	}
	uni := func(v []uint32) *wgpu.Buffer {
		if buildErr != nil {
			return nil
		}
		b, e := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(v), Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst})
		if e != nil {
			buildErr = e
			return nil
		}
		return keepBuf(b)
	}
	bind := func(layout *wgpu.BindGroupLayout, bufs ...*wgpu.Buffer) *wgpu.BindGroup {
		if buildErr != nil {
			return nil
		}
		es := make([]wgpu.BindGroupEntry, len(bufs))
		for i, b := range bufs {
			if b == nil { // an upstream storF/uni failed and short-circuited to nil
				buildErr = fmt.Errorf("gpu: newDecodeRunner: nil buffer for binding %d (allocation failed)", i)
				return nil
			}
			es[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()}
		}
		bg, e := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: es})
		if e != nil {
			buildErr = e
			return nil
		}
		return keepBG(bg)
	}
	add := func(pl *wgpu.ComputePipeline, bg *wgpu.BindGroup, gx, gy uint32) {
		r.steps = append(r.steps, runStep{pl: pl, bg: bg, gx: gx, gy: gy})
	}

	// op builders (record a step against persistent buffers):
	// rmsQuant fuses RMSNorm→quantize: one dispatch, no xn round-trip, one fewer
	// link on the serialized decode spine (§2). Bit-exact with rms→quant.
	rmsQuant := func(in, w *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
		kp := padK32(K) // int8 activation for a W4A8/W8A8 gemv that reads to the weight's kPad==padK32;
		// padK (a multiple of 16) under-sizes it when K%32 is 1..16: an OOB read, and the int4 zero-pad nibbles decode to −8.
		q, s := storF(kp/4), storF(1)
		p := uni([]uint32{uint32(K), f32bits(eps), boolU32(addOne), uint32(kp)})
		add(c.rmsQuantPipeline, bind(c.rmsQuantLayout, in, w, q, s, p), 1, 1)
		return q, s
	}
	// noAttnSinks is the fallback bound for a runLayer that leaves attnSinks nil, as every hand-built runLayer literal in the
	// gpu tests does (BuildResident always populates a dummy or real sinks); bind() hard-errors on a nil buffer, so the
	// fallback lives at the one dispatch site rather than in each caller, and a new hand-built fixture cannot miss it.
	noAttnSinks := storF(1)
	r.loraT = storF(loraRMax) // compute-time LoRA scratch: always allocated, cheap
	// swigluQuant fuses SwiGLU→quantize: the inter-wide product never materializes or crosses a barrier, one fewer link and
	// the big buffer stays off the spine. With gatedGELU it uses the geglu(Quant) pipelines instead (gelu_tanh(gate)·up);
	// otherwise it is byte-identical to the SiLU path.
	gelu := m.gatedGELU
	swigluQuant := func(gate, up *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
		kp := padK32(K) // int8 activation for a W4A8/W8A8 down-proj gemv, which reads to the weight's
		// kPad == padK32 (see rmsQuant); padK32 also zeroes the tail, matching the zero-padded weight.
		q, s := storF(kp/4), storF(1)
		p := uni([]uint32{uint32(K), uint32(kp), 0, 0})
		pl, ly := c.swigluQuantPipeline, c.swigluQuantLayout
		if gelu {
			pl, ly = c.gegluQuantPipeline, c.gegluQuantLayout
		}
		add(pl, bind(ly, gate, up, q, s, p), 1, 1)
		return q, s
	}
	// relu2Quant fuses Nemotron-H's non-gated relu²(up)→int8 (the squared-ReLU MLP), the
	// unary analog of swigluQuant. Bindings: up / qout / scales / dims (4 — no gate).
	relu2Quant := func(up *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
		kp := padK32(K) // int8 activation for a W4A8/W8A8 gemv reading to padK32 — see rmsQuant
		q, s := storF(kp/4), storF(1)
		p := uni([]uint32{uint32(K), uint32(kp), 0, 0})
		add(c.relu2Pipeline, bind(c.relu2Layout, up, q, s, p), 1, 1)
		return q, s
	}
	quant := func(in *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
		kp := padK32(K) // int8 activation for a W4A8/W8A8 gemv reading to padK32 — see rmsQuant
		q, s := storF(kp/4), storF(1)
		p := uni([]uint32{1, uint32(K), uint32(kp), 0})
		add(c.quantizePipeline, bind(c.quantizeLayout, in, q, s, p), 1, 1)
		return q, s
	}
	// gemv records a projection matmul against any resident precision (W8A8 or
	// W4A8 — both expose the same 6-binding gemv + addResidual via decodeWeight).
	gemv := func(aq, as *wgpu.Buffer, w decodeWeight) *wgpu.Buffer {
		out := storF(w.nRows())
		p := uni([]uint32{1, uint32(w.kPad()), uint32(w.nRows()), 0})
		gx, gy := gemvGrid(w.nRows())
		add(w.gPipe(c), bind(w.gLayout(c), aq, w.wbuf(), as, w.sbuf(), out, p), gx, gy)
		return out
	}
	// gemvAdd is gemv with the residual fused into the epilogue: dst (the running
	// hidden state) gets dst[n] += result, deleting a standalone residual link.
	gemvAdd := func(aq, as *wgpu.Buffer, w decodeWeight, dst *wgpu.Buffer) {
		p := uni([]uint32{1, uint32(w.kPad()), uint32(w.nRows()), 1})
		gx, gy := gemvGrid(w.nRows())
		add(w.gPipe(c), bind(w.gLayout(c), aq, w.wbuf(), as, w.sbuf(), dst, p), gx, gy)
	}
	// gemvBias is gemv with a per-output bias folded into the epilogue (dst[n] =
	// r + bias[n]), deleting the standalone biasAdd link for q/k/v of bias models
	// (Qwen2). W8A8 only — the bias kernel has the 7th (bias) binding.
	gemvBias := func(aq, as *wgpu.Buffer, w decodeWeight, bias *wgpu.Buffer) *wgpu.Buffer {
		out := storF(w.nRows())
		p := uni([]uint32{1, uint32(w.kPad()), uint32(w.nRows()), 0})
		gx, gy := gemvGrid(w.nRows())
		add(c.gemvBiasPipeline, bind(c.gemvBiasLayout, aq, w.wbuf(), as, w.sbuf(), out, p, bias), gx, gy)
		return out
	}
	// Mamba-2 SSM mixer dispatches: the conv/ssm/gatedNorm kernels read slices of the in_proj output (z|xBC|dt) from the full
	// proj buffer at base offsets carried in their uniforms, so no extra split kernel is needed (bindOff can bind sub-ranges,
	// but every caller here passes 0/0). The dispatches are pos-independent: the {win, ssm} state carries the recurrence.
	type bgEnt struct {
		b         *wgpu.Buffer
		off, size uint64
	}
	bindOff := func(layout *wgpu.BindGroupLayout, es []bgEnt) *wgpu.BindGroup {
		if buildErr != nil {
			return nil
		}
		en := make([]wgpu.BindGroupEntry, len(es))
		for i, e := range es {
			if e.b == nil { // an upstream allocation failed and short-circuited to nil
				buildErr = fmt.Errorf("gpu: newDecodeRunner: nil buffer for binding %d (allocation failed)", i)
				return nil
			}
			sz := e.size
			if sz == 0 {
				sz = e.b.GetSize()
			}
			en[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: e.b, Offset: e.off, Size: sz}
		}
		bg, e := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: en})
		if e != nil {
			buildErr = e
			return nil
		}
		return keepBG(bg)
	}
	var mambaConvOp func(proj, convW, convB, win, conv *wgpu.Buffer)
	var mambaSSMOp func(conv, proj, headP, ssm, y *wgpu.Buffer)
	var mambaGNormOp func(y, proj, normW, gated *wgpu.Buffer)
	if m.mamba != nil {
		mp := m.mamba
		// in_proj output slices addressed via base offsets in the uniform (alignment-free),
		// binding the full proj buffer: z=proj[0:], xBC=proj[dInner:], dt=proj[dInner+convDim:].
		dC := uni([]uint32{uint32(mp.convDim), uint32(mp.dConv), uint32(mp.dInner), 0})
		dS := uni([]uint32{uint32(mp.nHeads), uint32(mp.hp), uint32(mp.dn), uint32(mp.nGroups), uint32(mp.repeat), uint32(mp.gSize), uint32(mp.dInner), uint32(mp.dInner + mp.convDim)})
		dG := uni([]uint32{uint32(mp.dInner), uint32(mp.normGroups), uint32(mp.dInner / mp.normGroups), f32bits(eps), 0, 0, 0, 0})
		mambaConvOp = func(proj, convW, convB, win, conv *wgpu.Buffer) {
			es := []bgEnt{{proj, 0, 0}, {convW, 0, 0}, {convB, 0, 0}, {win, 0, 0}, {conv, 0, 0}, {dC, 0, 0}}
			add(c.mambaConvPipeline, bindOff(c.mambaConvLayout, es), uint32(mp.convDim+63)/64, 1)
		}
		mambaSSMOp = func(conv, proj, headP, ssm, y *wgpu.Buffer) {
			es := []bgEnt{{conv, 0, 0}, {proj, 0, 0}, {headP, 0, 0}, {ssm, 0, 0}, {y, 0, 0}, {dS, 0, 0}}
			add(c.mambaSSMPipeline, bindOff(c.mambaSSMLayout, es), uint32(mp.nHeads*mp.hp+63)/64, 1)
		}
		mambaGNormOp = func(y, proj, normW, gated *wgpu.Buffer) {
			es := []bgEnt{{y, 0, 0}, {proj, 0, 0}, {normW, 0, 0}, {gated, 0, 0}, {dG, 0, 0}}
			add(c.mambaGNormPipeline, bindOff(c.mambaGNormLayout, es), uint32(mp.normGroups), 1)
		}
	}
	// Gated-DeltaNet op builders. The causal conv is mambaConv's — DeltaNet's conv has the same
	// shape, the same SiLU and the same ring window, differing only in being bias-free, so it
	// binds an all-zero convB and xbcBase 0. Everything after it is DeltaNet's own.
	var dnConvOp func(mixed, convW, convB, win, conv *wgpu.Buffer)
	var dnGatesOp func(bt, at, dtBias, negExpA, headP *wgpu.Buffer)
	var dnNormOp func(conv, qn, kn *wgpu.Buffer)
	var dnRuleOp func(qn, kn, v, headP, state, core *wgpu.Buffer)
	var dnGNormOp func(core, z, normW, gated *wgpu.Buffer)
	if m.dnet != nil {
		dp := m.dnet
		dC := uni([]uint32{uint32(dp.convDim), uint32(dp.convK), 0, 0})
		dGate := uni([]uint32{uint32(dp.nv), 0, 0, 0})
		dNorm := uni([]uint32{uint32(dp.nk), uint32(dp.hk), uint32(dp.keyDim), 0, f32bits(float32(1 / math.Sqrt(float64(dp.hk)))), 0, 0, 0})
		dRule := uni([]uint32{uint32(dp.nv), uint32(dp.nk), uint32(dp.hk), uint32(dp.hv), uint32(dp.rep), uint32(2 * dp.keyDim), 0, 0})
		dGN := uni([]uint32{uint32(dp.nv), uint32(dp.hv), 0, 0, f32bits(dp.eps), 0, 0, 0})
		dnConvOp = func(mixed, convW, convB, win, conv *wgpu.Buffer) {
			add(c.mambaConvPipeline, bind(c.mambaConvLayout, mixed, convW, convB, win, conv, dC), uint32(dp.convDim+63)/64, 1)
		}
		dnGatesOp = func(bt, at, dtBias, negExpA, headP *wgpu.Buffer) {
			add(c.deltaGatesPipeline, bind(c.deltaGatesLayout, bt, at, dtBias, negExpA, headP, dGate), uint32(dp.nv+63)/64, 1)
		}
		dnNormOp = func(conv, qn, kn *wgpu.Buffer) {
			add(c.deltaNormPipeline, bind(c.deltaNormLayout, conv, qn, kn, dNorm), uint32(dp.nk+31)/32, 1)
		}
		dnRuleOp = func(qn, kn, v, headP, state, core *wgpu.Buffer) {
			add(c.deltaRulePipeline, bind(c.deltaRuleLayout, qn, kn, v, headP, state, core, dRule), uint32(dp.valueDim+63)/64, 1)
		}
		dnGNormOp = func(core, z, normW, gated *wgpu.Buffer) {
			add(c.deltaGNormPipeline, bind(c.deltaGNormLayout, core, z, normW, gated, dGN), uint32(dp.nv+63)/64, 1)
		}
	}
	// qSplit unpacks a double-width [query ‖ gate]-per-head q_proj output; attnGate applies
	// ctx *= sigmoid(gate) after attention. Both are no-ops for every family without
	// attn_output_gate (runLayer.qGate false ⇒ never dispatched).
	qSplit := func(qg *wgpu.Buffer, n, headDim int) (*wgpu.Buffer, *wgpu.Buffer) {
		q, gate := storF(n), storF(n)
		p := uni([]uint32{uint32(n), uint32(headDim), 0, 0})
		add(c.deltaQSplitPipeline, bind(c.deltaQSplitLayout, qg, q, gate, p), uint32(n+63)/64, 1)
		return q, gate
	}
	attnGate := func(ctxv, gate *wgpu.Buffer, n int) {
		p := uni([]uint32{uint32(n), 0, 0, 0})
		add(c.deltaAttnGatePipeline, bind(c.deltaAttnGateLayout, ctxv, gate, p), uint32(n+63)/64, 1)
	}
	// rmsnormF32: plain f32 RMSNorm (no quantize) — the post-attention / post-MLP norms.
	rmsnormF32 := func(in, weight *wgpu.Buffer, n int) *wgpu.Buffer {
		out := storF(n)
		p := uni([]uint32{uint32(n), f32bits(eps), boolU32(addOne), 0})
		add(c.rmsnormPipeline, bind(c.rmsnormLayout, in, weight, out, p), 1, 1)
		return out
	}
	// The per-token uniforms (rope-q, rope-store-k, v-store, attn) depend on pos but not on the layer index, so one buffer per
	// distinct geometry and rope scale is shared by every layer's dispatch and Run writes a few small uniforms per token
	// instead of one set per layer. half is the rotated pairs per head, rotaryDim/2 = len(invFreq); m.ropeHalf carries it for
	// partial RoPE, 0 ⇒ HeadDim/2. The rope kernels pair vec[off+d] with vec[off+half+d] for d<half and leave the trailing
	// HeadDim-rotaryDim dims untouched, as decoder.applyRoPE's partial layout does.
	half := hd / 2
	if m.ropeHalf > 0 {
		half = m.ropeHalf
	}
	// The cos/sin are multiplied by the layer's RoPE scale (YaRN attention_factor; 1.0 otherwise). Most models use one scale;
	// families with per-layer RoPE (YaRN on the global layers, default on the local ones) use two, so the rope uniforms are
	// keyed by scale value. Per-layer attention geometry: geomFor builds one attnGeom per distinct {hd, nKV, half, kEqV} tuple
	// and caches it by value; each geom owns the v-store, attn and windowed-attn uniforms, and the rope uniforms hang off it
	// keyed by rope scale.
	geomCache := map[[4]int]*attnGeom{}
	geomFor := func(ghd, gnKV, ghalf int, kEqV bool) *attnGeom {
		kb := 0
		if kEqV {
			kb = 1
		}
		key := [4]int{ghd, gnKV, ghalf, kb}
		if g, ok := geomCache[key]; ok {
			return g
		}
		g := &attnGeom{
			hd: ghd, nKV: gnKV, half: ghalf, kvDim: gnKV * ghd, kEqV: kEqV,
			ropeQUnis:  map[float32]*wgpu.Buffer{},
			ropeKUnis:  map[float32]*wgpu.Buffer{},
			qkvFinUnis: map[float32]*wgpu.Buffer{},
		}
		g.vStoreUni = uni([]uint32{uint32(g.kvDim), 0, 0, 0})
		r.posUnis = append(r.posUnis, posUni{buf: g.vStoreUni, fill: func(dst *[16]uint32, pos, ropePos int) int {
			dst[0] = uint32(g.kvDim)
			dst[1] = uint32(pos * g.kvDim)
			dst[2] = 0
			dst[3] = 0
			return 4
		}})
		// int8 V store needs its own (differently-laid-out) per-token uniform:
		// {heads=nKV, headDim=hd, base=pos*kvDim, pos, nKV}. Only allocated for kvI8.
		if m.kvI8 {
			g.vStoreI8Uni = uni([]uint32{uint32(g.nKV), uint32(g.hd), 0, 0, uint32(g.nKV), 0, 0, 0})
			r.posUnis = append(r.posUnis, posUni{buf: g.vStoreI8Uni, fill: func(dst *[16]uint32, pos, ropePos int) int {
				dst[0] = uint32(g.nKV)
				dst[1] = uint32(g.hd)
				dst[2] = uint32(pos * g.kvDim)
				dst[3] = uint32(pos)
				dst[4] = uint32(g.nKV)
				dst[5] = 0
				dst[6] = 0
				dst[7] = 0
				return 8
			}})
		}
		g.attnUni = uni([]uint32{uint32(nH), uint32(g.nKV), uint32(g.hd), 0, uint32(start), uint32(nH / g.nKV), f32bits(scale), 0})
		r.posUnis = append(r.posUnis, posUni{buf: g.attnUni, fill: func(dst *[16]uint32, pos, ropePos int) int {
			dst[0] = uint32(nH)
			dst[1] = uint32(g.nKV)
			dst[2] = uint32(g.hd)
			dst[3] = uint32(pos + 1)
			dst[4] = uint32(start)
			dst[5] = uint32(nH / g.nKV)
			dst[6] = f32bits(scale)
			dst[7] = 0
			return 8
		}})
		// Sliding-window (local) layers attend only the last `slidingWindow` positions: the attention start advances to
		// max(0, pos+1-W) once pos reaches the window, matching decoder.KVCache.WindowStart. Full layers keep attnUni. Only built
		// when the model windows; local layers bind this instead of attnUni.
		g.attnUniLocal = g.attnUni
		if m.slidingWindow > 0 {
			w := m.slidingWindow
			g.attnUniLocal = uni([]uint32{uint32(nH), uint32(g.nKV), uint32(g.hd), 0, uint32(start), uint32(nH / g.nKV), f32bits(scale), 0})
			r.posUnis = append(r.posUnis, posUni{buf: g.attnUniLocal, fill: func(dst *[16]uint32, pos, ropePos int) int {
				ws := start
				if lo := pos + 1 - w; lo > ws {
					ws = lo
				}
				dst[0] = uint32(nH)
				dst[1] = uint32(g.nKV)
				dst[2] = uint32(g.hd)
				dst[3] = uint32(pos + 1)
				dst[4] = uint32(ws)
				dst[5] = uint32(nH / g.nKV)
				dst[6] = f32bits(scale)
				dst[7] = 0
				return 8
			}})
		}
		geomCache[key] = g
		return g
	}
	// Per-(geom, rope-scale) rope uniforms. slot 6 of the K uniform carries nKV for the
	// int8 ropeStore (it indexes scales[pos*nKV+head]); the f32/f16 ropeStore ignore it.
	// ropeQUniFor's slot 3 ("pos" in the WGSL P struct, ropeShaderWGSL) is purely the rotation
	// angle input — Q is rotated in place, never stored to a KV cache — so it takes ropePos
	// wholesale.
	ropeQUniFor := func(g *attnGeom, rs float32) *wgpu.Buffer {
		if b, ok := g.ropeQUnis[rs]; ok {
			return b
		}
		b := uni([]uint32{uint32(nH), uint32(g.hd), uint32(g.half), 0, f32bits(rs), 0, 0, 0})
		r.posUnis = append(r.posUnis, posUni{buf: b, fill: func(dst *[16]uint32, pos, ropePos int) int {
			dst[0] = uint32(nH)
			dst[1] = uint32(g.hd)
			dst[2] = uint32(g.half)
			dst[3] = uint32(ropePos)
			dst[4] = f32bits(rs)
			dst[5] = 0
			dst[6] = 0
			dst[7] = 0
			return 8
		}})
		g.ropeQUnis[rs] = b
		return b
	}
	// ropeKUniFor feeds ropeStoreShaderWGSL, whose P struct has TWO roles: slot 3 ("pos") is the rotation angle and takes
	// ropePos; slot 5 ("base") is the KV-cache write offset pos*kvDim and stays keyed on the true pos. Slot 7 ("spos") carries
	// the true pos too: the int8 variant indexes its per-(position, KV-head) scale by it (it must not read the rope-angle
	// slot), and the f32/f16 variants never read it. History: docs/code-notes/gpu.md#ropeKUniFor.
	ropeKUniFor := func(g *attnGeom, rs float32) *wgpu.Buffer {
		if b, ok := g.ropeKUnis[rs]; ok {
			return b
		}
		b := uni([]uint32{uint32(g.nKV), uint32(g.hd), uint32(g.half), 0, f32bits(rs), 0, uint32(g.nKV), 0})
		r.posUnis = append(r.posUnis, posUni{buf: b, fill: func(dst *[16]uint32, pos, ropePos int) int {
			dst[0] = uint32(g.nKV)
			dst[1] = uint32(g.hd)
			dst[2] = uint32(g.half)
			dst[3] = uint32(ropePos)
			dst[4] = f32bits(rs)
			dst[5] = uint32(pos * g.kvDim)
			dst[6] = uint32(g.nKV)
			dst[7] = uint32(pos)
			return 8
		}})
		g.ropeKUnis[rs] = b
		return b
	}
	// Fused q-rope + k-rope-store + v-store uniform (decode fusion, f32 KV), per (geom, scale): {nH, nKV, hd, half, pos,
	// base=pos*kvDim, scale, kvDim}. Same split as ropeKUniFor: slot 4 ("pos") is the rotation angle for both q and k, slot 5
	// ("base") is the KV-cache write offset.
	qkvFinUniFor := func(g *attnGeom, rs float32) *wgpu.Buffer {
		if b, ok := g.qkvFinUnis[rs]; ok {
			return b
		}
		b := uni([]uint32{uint32(nH), uint32(g.nKV), uint32(g.hd), uint32(g.half), 0, 0, f32bits(rs), uint32(g.kvDim)})
		r.posUnis = append(r.posUnis, posUni{buf: b, fill: func(dst *[16]uint32, pos, ropePos int) int {
			dst[0] = uint32(nH)
			dst[1] = uint32(g.nKV)
			dst[2] = uint32(g.hd)
			dst[3] = uint32(g.half)
			dst[4] = uint32(ropePos)
			dst[5] = uint32(pos * g.kvDim)
			dst[6] = f32bits(rs)
			dst[7] = uint32(g.kvDim)
			return 8
		}})
		g.qkvFinUnis[rs] = b
		return b
	}
	rope := func(g *attnGeom, vec, invFreq *wgpu.Buffer, ropeScale float32) {
		if ropeScale == 0 {
			ropeScale = 1
		}
		add(c.ropePipeline, bind(c.ropeLayout, vec, invFreq, ropeQUniFor(g, ropeScale)), uint32(nH*g.half+63)/64, 1)
	}
	// ropeStore rotates src (the K projection) and writes it straight into the KV
	// cache at pos*kvDim — replacing the K CopyBufferToBuffer append so the token
	// stays one compute pass. base rides the per-(geom,scale) ropeKUni. The f16 variant
	// packs 2 rotated elems/word (one thread per word = nKV*half, same dispatch count).
	ropeStore := func(g *attnGeom, src, invFreq, cache, scale *wgpu.Buffer, ropeScale float32) {
		if ropeScale == 0 {
			ropeScale = 1
		}
		ku := ropeKUniFor(g, ropeScale)
		if m.kvI8 {
			// one thread per KV head: per-head absmax → scale → quantize + pack 4/word.
			add(c.ropeStoreI8Pipeline, bind(c.ropeStoreI8Layout, src, invFreq, cache, scale, ku), uint32(g.nKV+63)/64, 1)
			return
		}
		if m.kvF16 {
			// word-based (2 f16/word): kvDim/2 = nKV·hd/2 words, covering the rotated span AND the partial-rotary pass-through tail.
			add(c.ropeStoreF16Pipeline, bind(c.ropeStoreF16Layout, src, invFreq, cache, ku), uint32(g.nKV*g.hd/2+63)/64, 1)
		} else {
			// element-based: nKV·half rotation pairs + nKV·(hd-2·half) pass-through tail = nKV·(hd-half).
			add(c.ropeStorePipeline, bind(c.ropeStoreLayout, src, invFreq, cache, ku), uint32(g.nKV*(g.hd-g.half)+63)/64, 1)
		}
	}
	// vStore copies src (the V projection) into the V cache at pos*kvDim. The f16
	// variant packs 2 elems/word, so it dispatches half as many threads (one/word);
	// the int8 variant is one thread per KV head (per-head absmax → scale → pack).
	vStore := func(g *attnGeom, src, cache, scale *wgpu.Buffer) {
		if m.kvI8 {
			add(c.kvStoreI8Pipeline, bind(c.kvStoreI8Layout, src, cache, scale, g.vStoreI8Uni), uint32(g.nKV+63)/64, 1)
			return
		}
		if m.kvF16 {
			words := g.kvDim / 2
			add(c.kvStoreF16Pipeline, bind(c.kvStoreF16Layout, src, cache, g.vStoreUni), uint32(words+63)/64, 1)
			return
		}
		add(c.kvStorePipeline, bind(c.kvStoreLayout, src, cache, g.vStoreUni), uint32(g.kvDim+63)/64, 1)
	}
	// qkvFinalize fuses rope(q) + rope-store(k) + store(v) into one dispatch (f32 KV
	// only). Threads = max(nH·half, kvDim); each does whichever of the three apply.
	qkvFinalize := func(g *attnGeom, q, k, v, invFreq, kCache, vCache *wgpu.Buffer, ropeScale float32) {
		if ropeScale == 0 {
			ropeScale = 1
		}
		n := max(g.kvDim, nH*g.half)
		add(c.qkvFinPipeline, bind(c.qkvFinLayout, q, k, v, invFreq, kCache, vCache, qkvFinUniFor(g, ropeScale)), uint32(n+63)/64, 1)
	}
	// biasAdd computes vec[i] += bias[i] by reusing the residual kernel: the Qwen2 q/k/v bias (n is the projection width), and
	// the plain `xd += out` residual add where the fused gemvAdd epilogue cannot be used.
	biasAdd := func(vec, bias *wgpu.Buffer, n int) {
		p := uni([]uint32{uint32(n), 0, 0, 0})
		add(c.residualPipeline, bind(c.residualLayout, vec, bias, p), uint32(n+63)/64, 1)
	}
	// qkNorm RMS-normalizes each of `heads` heads of vec (q or k) over headDim in place with weight[hd], before RoPE. One
	// workgroup per head; the uniform is pos-independent so it is a plain uni, not a posUni.
	qkNorm := func(vec, weight *wgpu.Buffer, heads, lhd int) {
		p := uni([]uint32{uint32(heads), uint32(lhd), f32bits(eps), boolU32(addOne)})
		add(c.qkNormPipeline, bind(c.qkNormLayout, vec, weight, p), uint32(heads), 1)
	}
	// vNorm computes scale-less per-head RMSNorm from raw K into V (Gemma 4 globals).
	vNorm := func(src, dst *wgpu.Buffer, heads, lhd int) {
		p := uni([]uint32{uint32(heads), uint32(lhd), f32bits(eps), 0})
		add(c.vNormPipeline, bind(c.vNormLayout, src, dst, p), uint32(heads), 1)
	}
	// scaleVec multiplies vec in place by a scalar (Gemma 4 dense per-layer output scalar).
	scaleVec := func(vec *wgpu.Buffer, scale float32, n int) {
		p := uni([]uint32{uint32(n), f32bits(scale), 0, 0})
		add(c.scaleVecPipeline, bind(c.scaleVecLayout, vec, p), uint32(n+255)/256, 1)
	}
	// MoE op builders. moeRoute records the on-GPU router top-k SELECTION (logits[nE] + optional bias → idx[k], wgt[k]); the p
	// uniform is pos-independent (model-level shape), so it is a plain uni. nE is tiny ⇒ one single-lane workgroup.
	var moeRoute func(logits, bias, idx, wgt *wgpu.Buffer, hasBias bool)
	var moeExpert func(aq, as *wgpu.Buffer, s *ResidentStackedW8A8, idx, wgt, dst *wgpu.Buffer, slot, mode int)
	if m.moe != nil {
		mp := m.moe
		moeRoute = func(logits, bias, idx, wgt *wgpu.Buffer, hasBias bool) {
			p := uni([]uint32{uint32(mp.nE), uint32(mp.k), boolU32(mp.sigmoid), boolU32(mp.norm), f32bits(mp.scale), boolU32(hasBias), uint32(mp.nGroup), uint32(mp.topkGroup)})
			add(c.moeRoutePipeline, bind(c.moeRouteLayout, logits, bias, idx, wgt, p), 1, 1)
		}
		// moeExpert records one indexed sparse-expert GEMV: dst[n] = expert[idx[slot]]·aq
		// (mode 0, overwrite gate/up scratch) or dst[n] += wgt[slot]·(expert[idx[slot]]·aq)
		// (mode 1, the down-projection combine into the running residual). The expert is
		// chosen at run time from idx[slot] — a fixed dispatch, no host round-trip.
		moeExpert = func(aq, as *wgpu.Buffer, s *ResidentStackedW8A8, idx, wgt, dst *wgpu.Buffer, slot, mode int) {
			d := uni([]uint32{uint32(s.kp), uint32(s.rows), uint32(slot), uint32(mode)})
			gx, gy := gemvGrid(s.rows)
			if s.w4 { // W4A8: int4 stacked expert (nibbles + f16 group scales) × int8 activation
				add(c.moeExpertW4Pipeline, bind(c.moeExpertW4Layout, aq, s.bq, as, s.bScales, dst, idx, wgt, d), gx, gy)
				return
			}
			add(c.moeExpertPipeline, bind(c.moeExpertLayout, aq, s.bq, as, s.bScales, dst, idx, wgt, d), gx, gy)
		}
	}
	// gpt-oss's three MoE dispatch builders are separate closures from moeRoute/moeExpert because the contract genuinely
	// differs (see routeGptOssWGSL/gptossGluQuantWGSL/moeExpertGptOssDownGEMVWGSL), not a parameterization of them.
	var routeGptOss func(logits, bias, idx, wgt *wgpu.Buffer, hasBias bool)
	var gptossActQuant func(gate, up, biasGU, idx *wgpu.Buffer, slot, K int) (*wgpu.Buffer, *wgpu.Buffer)
	var moeExpertGptOssDown func(aq, as *wgpu.Buffer, s *ResidentStackedW8A8, idx, wgt, dbias, dst *wgpu.Buffer, slot int)
	if m.moe != nil && m.moe.gptoss {
		mp := m.moe
		routeGptOss = func(logits, bias, idx, wgt *wgpu.Buffer, hasBias bool) {
			p := uni([]uint32{uint32(mp.nE), uint32(mp.k), boolU32(hasBias), 0})
			add(c.routeGptOssPipeline, bind(c.routeGptOssLayout, logits, bias, idx, wgt, p), 1, 1)
		}
		gptossActQuant = func(gate, up, biasGU, idx *wgpu.Buffer, slot, K int) (*wgpu.Buffer, *wgpu.Buffer) {
			kp := padK32(K)
			q, s := storF(kp/4), storF(1)
			p := uni([]uint32{uint32(K), uint32(kp), uint32(slot), 1, f32bits(mp.gptossAlpha), f32bits(mp.gptossLimit), 0, 0})
			add(c.gptossGluQuantPipeline, bind(c.gptossGluQuantLayout, gate, up, biasGU, idx, q, s, p), 1, 1)
			return q, s
		}
		moeExpertGptOssDown = func(aq, as *wgpu.Buffer, s *ResidentStackedW8A8, idx, wgt, dbias, dst *wgpu.Buffer, slot int) {
			d := uni([]uint32{uint32(s.kp), uint32(s.rows), uint32(slot), 0})
			gx, gy := gemvGrid(s.rows)
			pl, ly := c.gptOssDownPipelineFor(s)
			add(pl, bind(ly, aq, s.bq, as, s.bScales, dst, idx, wgt, dbias, d), gx, gy)
		}
	}
	// sharedGatedCombine records the qwen2_moe gated shared-expert add: dst[n] +=
	// sigmoid(gl[0])·src[n]. The GLM/DeepSeek ungated case uses gemvAdd instead.
	sharedGatedCombine := func(dst, src, gl *wgpu.Buffer, n int) {
		p := uni([]uint32{uint32(n), 0, 0, 0})
		add(c.sharedGatePipeline, bind(c.sharedGateLayout, dst, src, gl, p), uint32(n+63)/64, 1)
	}

	// MLA op builders. The latent store + attention uniforms are pos-dependent (base = pos·latDim, nKeys = pos+1), so they
	// register posUnis like the standard attention path; the absorb/lift matvecs are pos-independent shapes.
	var mlaStore func(kvDown, normW, invFreq, latCache *wgpu.Buffer)
	var mlaAbsorb func(q, wuk, qAbs *wgpu.Buffer)
	var mlaQRopeOp func(q, invFreq, qAbs *wgpu.Buffer)
	var mlaAttnOp func(qAbs, latCache, wsum *wgpu.Buffer)
	var mlaLift func(wsum, wuv, ctxv *wgpu.Buffer)
	if m.mla != nil {
		mp := m.mla
		qkHead := mp.qkNope + mp.qkRope
		latDim := mp.kvLoRARank + mp.qkRope
		rank := mp.kvLoRARank
		rhalf := mp.qkRope / 2
		// Latent store: kvA-norm the rank latent + decoupled-RoPE the key into latCache
		// at base = pos·latDim. One single-workgroup dispatch (the norm reduces in-WG).
		mlaStoreUni := uni([]uint32{uint32(rank), uint32(mp.qkRope), 0, f32bits(eps), 0, f32bits(mp.ropeScale), boolU32(mp.interleave), 0})
		r.posUnis = append(r.posUnis, posUni{buf: mlaStoreUni, fill: func(dst *[16]uint32, pos, ropePos int) int {
			dst[0] = uint32(rank)
			dst[1] = uint32(mp.qkRope)
			dst[2] = uint32(pos)
			dst[3] = f32bits(eps)
			dst[4] = uint32(pos * latDim)
			dst[5] = f32bits(mp.ropeScale)
			dst[6] = boolU32(mp.interleave)
			dst[7] = 0
			return 8
		}})
		mlaStore = func(kvDown, normW, invFreq, latCache *wgpu.Buffer) {
			add(c.mlaStorePipeline, bind(c.mlaStoreLayout, kvDown, normW, invFreq, latCache, mlaStoreUni), 1, 1)
		}
		// W_UK absorb: qNopeAbs_h = W_UKᵀ_h·q_nope_h, written strided into qAbs[h·latDim..+rank].
		mlaAbsorb = func(q, wuk, qAbs *wgpu.Buffer) {
			p := uni([]uint32{uint32(nH), uint32(rank), uint32(mp.qkNope), uint32(qkHead), uint32(latDim), 0, 0, 0})
			gx, gy := gemvGrid(nH * rank)
			add(c.mlaHeadMVPipeline, bind(c.mlaHeadMVLayout, q, wuk, qAbs, p), gx, gy)
		}
		// Query RoPE: gather + rope q's rope dims into qAbs[h·latDim+rank..]. pos-dependent.
		mlaQRopeUni := uni([]uint32{uint32(nH), uint32(qkHead), uint32(mp.qkNope), uint32(mp.qkRope), uint32(rank), uint32(latDim), 0, boolU32(mp.interleave), f32bits(mp.ropeScale), 0, 0, 0})
		r.posUnis = append(r.posUnis, posUni{buf: mlaQRopeUni, fill: func(dst *[16]uint32, pos, ropePos int) int {
			dst[0] = uint32(nH)
			dst[1] = uint32(qkHead)
			dst[2] = uint32(mp.qkNope)
			dst[3] = uint32(mp.qkRope)
			dst[4] = uint32(rank)
			dst[5] = uint32(latDim)
			dst[6] = uint32(pos)
			dst[7] = boolU32(mp.interleave)
			dst[8] = f32bits(mp.ropeScale)
			dst[9] = 0
			dst[10] = 0
			dst[11] = 0
			return 12
		}})
		mlaQRopeOp = func(q, invFreq, qAbs *wgpu.Buffer) {
			add(c.mlaQRopePipeline, bind(c.mlaQRopeLayout, q, invFreq, qAbs, mlaQRopeUni), uint32(nH*rhalf+63)/64, 1)
		}
		// Attention: rank-space online-softmax over nKeys = pos+1 latents. pos-dependent.
		mlaAttnUni := uni([]uint32{uint32(nH), uint32(latDim), uint32(rank), 0, f32bits(scale), 0, 0, 0})
		r.posUnis = append(r.posUnis, posUni{buf: mlaAttnUni, fill: func(dst *[16]uint32, pos, ropePos int) int {
			dst[0] = uint32(nH)
			dst[1] = uint32(latDim)
			dst[2] = uint32(rank)
			dst[3] = uint32(pos + 1)
			dst[4] = f32bits(scale)
			dst[5] = 0
			dst[6] = 0
			dst[7] = 0
			return 8
		}})
		mlaAttnOp = func(qAbs, latCache, wsum *wgpu.Buffer) {
			add(c.mlaAttnPipeline, bind(c.mlaAttnLayout, qAbs, latCache, wsum, mlaAttnUni), uint32(nH), 1)
		}
		// W_UV lift: ctx_h = W_UV_h·wsum_h ([vHead] per head), contiguous output.
		mlaLift = func(wsum, wuv, ctxv *wgpu.Buffer) {
			p := uni([]uint32{uint32(nH), uint32(mp.vHead), uint32(rank), uint32(rank), uint32(mp.vHead), 0, 0, 0})
			gx, gy := gemvGrid(nH * mp.vHead)
			add(c.mlaHeadMVPipeline, bind(c.mlaHeadMVLayout, wsum, wuv, ctxv, p), gx, gy)
		}
	}

	r.xd = func() *wgpu.Buffer {
		if buildErr != nil {
			return nil
		}
		b, e := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(hidden * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc})
		if e != nil {
			buildErr = e
			return nil
		}
		return keepBuf(b)
	}()

	// Per-layer capture (GOINFER_GPU_CAPTURE=1): a kv-store dispatch (dst[i] = src[i]) copying
	// a live buffer into a build-once capture buffer, recorded in plan order so it sees exactly
	// what the next dispatch sees. Reuses the kv-store kernel rather than adding one.
	captureOn := os.Getenv("GOINFER_GPU_CAPTURE") != ""
	capStep := func(src *wgpu.Buffer, n int) *wgpu.Buffer {
		dst := storF(n)
		p := uni([]uint32{uint32(n), 0, 0, 0})
		add(c.kvStorePipeline, bind(c.kvStoreLayout, src, dst, p), uint32(n+63)/64, 1)
		return dst
	}
	if captureOn {
		r.capHidden = hidden
		r.capCtxN = nH * hd
	}

	// A bound (all-zero) bias buffer for MoE layers without a selection bias
	// (Mixtral softmax routing): the route kernel binds it but hasBias=0 keeps it
	// out of the math. CreateBuffer zero-inits, so no upload needed.
	var moeZeroBias *wgpu.Buffer
	if m.moe != nil {
		moeZeroBias = storF(m.moe.nE)
	}

	for i := range m.layers {
		lw := &m.layers[i]
		if ssmStopLayer >= 0 && i > ssmStopLayer {
			break // layer-sweep seam: logits from xd after this layer
		}
		if lw.nemoKind == nemoKMLP {
			// Nemotron-H non-gated relu² MLP block (single-op-per-block, no mixer): norm → up →
			// relu²→int8 → down + residual into xd. The other kinds fall through to the mixer.
			mq, ms := rmsQuant(r.xd, lw.mlpNorm, hidden)
			up := gemv(mq, ms, lw.up)
			rq, rs := relu2Quant(up, lw.up.nRows())
			gemvAdd(rq, rs, lw.down, r.xd)
			continue
		}
		if lw.nemoKind == nemoKMoE {
			// Nemotron-H MoE block (single-op-per-block, no mixer): norm → route with the SAME moeRoute kernel DeepSeek/GLM use
			// (Nemotron's sigmoid + selection bias + group-limited top-k with nGroup=1 degenerates to plain top-k, moeRouteWGSL's
			// nGroup==1 path) → k routed experts, each a NON-gated relu² FFN (moeExpert is generic per projection; called for up/down
			// only, no expGate) weighted-summed into xd → the always-on UNGATED shared expert of the same shape, added once, not per
			// top-k slot.
			mq, ms := rmsQuant(r.xd, lw.mlpNorm, hidden)
			mp := m.moe
			logits := gemv(mq, ms, lw.router)
			idx, wgt := storF(mp.k), storF(mp.k)
			bias, hasBias := moeZeroBias, false
			if lw.routerBias != nil {
				bias, hasBias = lw.routerBias, true
			}
			moeRoute(logits, bias, idx, wgt, hasBias)
			for j := 0; j < mp.k; j++ {
				upOut := storF(mp.inter)
				moeExpert(mq, ms, lw.expUp, idx, wgt, upOut, j, 0)
				rq, rs := relu2Quant(upOut, mp.inter)
				moeExpert(rq, rs, lw.expDown, idx, wgt, r.xd, j, 1)
			}
			if lw.shUp != nil { // always present: SharedIntermediateDim > 0 gates rl.shUp's build
				su := gemv(mq, ms, lw.shUp)
				sq, ss := relu2Quant(su, mp.sharedInter)
				gemvAdd(sq, ss, lw.shDown, r.xd)
			}
			continue
		}
		if lw.isMamba {
			// Mamba-2 SSM mixer: norm → in_proj → conv(ring) → ssm(state) → gatedNorm → out_proj+residual. State {win, ssm} persists
			// in lw, updated in place per token. ResidMul is folded into the out_proj weights. The FFN sub-block below is shared.
			mp := m.mamba
			aq, as := rmsQuant(r.xd, lw.attnNorm, hidden)
			proj := gemv(aq, as, lw.mambaInProj)
			conv := storF(mp.convDim)
			mambaConvOp(proj, lw.mambaConvW, lw.mambaConvB, lw.mambaWin, conv)
			y := storF(mp.dInner)
			mambaSSMOp(conv, proj, lw.mambaHeadP, lw.mambaSSM, y)
			gated := storF(mp.dInner)
			mambaGNormOp(y, proj, lw.mambaNormW, gated)
			if r.mcapProj == nil { // debug: capture the FIRST mamba layer for the wiring diff
				r.mcapProj, r.mcapConv, r.mcapY, r.mcapGated = proj, conv, y, gated
			}
			gq, gs := quant(gated, mp.dInner)
			gemvAdd(gq, gs, lw.mambaOutProj, r.xd)
		} else if lw.isDeltaNet {
			// Gated-DeltaNet mixer: norm → in_proj_qkv → conv(ring) → l2norm(q,k) → delta rule
			// (state) → gated RMSNorm × silu(z) → out_proj + residual. The {win, dnState} pair
			// persists in lw and is updated in place per token, like Mamba's {win, ssm}.
			//
			// The two gate projections run off the SAME quantized activation as the big ones.
			// On the CPU they are f32 (deltaNetWeights keeps inProjB/inProjA unquantized because
			// they feed the write/decay gates, where the recurrence is most precision-sensitive)
			// — so this is the one place the resident path is deliberately coarser than the
			// reference, and the parity gate is what says whether that is affordable.
			dp := m.dnet
			aq, as := rmsQuant(r.xd, lw.attnNorm, hidden)
			mixed := gemv(aq, as, lw.dnQKV)
			conv := storF(dp.convDim)
			dnConvOp(mixed, lw.mambaConvW, lw.mambaConvB, lw.mambaWin, conv)
			qn, kn := storF(dp.keyDim), storF(dp.keyDim)
			dnNormOp(conv, qn, kn)
			headP := storF(dp.nv * 2)
			dnGatesOp(gemv(aq, as, lw.dnB), gemv(aq, as, lw.dnA), lw.dnDtBias, lw.dnNegExpA, headP)
			core := storF(dp.valueDim)
			// v is conv[2*keyDim:] — bound as an offset view rather than copied, the same
			// alignment-free trick mambaSSMOp uses for its in_proj slices.
			dnRuleOp(qn, kn, conv, headP, lw.dnState, core)
			gated := storF(dp.valueDim)
			dnGNormOp(core, gemv(aq, as, lw.dnZ), lw.dnNormW, gated)
			gq, gs := quant(gated, dp.valueDim)
			gemvAdd(gq, gs, lw.dnOut, r.xd)
		} else if m.mla != nil {
			// MLA latent attention: input-norm → q (LoRA/direct) + kv-down → latent store → W_UK-absorb + qRope → rank-space attend →
			// W_UV-lift → o-proj.
			mp := m.mla
			latDim := mp.kvLoRARank + mp.qkRope
			rank := mp.kvLoRARank
			aq, as := rmsQuant(r.xd, lw.attnNorm, hidden)
			var qf *wgpu.Buffer
			if mp.qLoRARank > 0 { // q_a → norm → q_b LoRA bottleneck
				qa := gemv(aq, as, lw.mlaQA)
				qaq, qas := rmsQuant(qa, lw.mlaQANorm, mp.qLoRARank)
				qf = gemv(qaq, qas, lw.mlaQB)
			} else { // direct q_proj (V2-Lite)
				qf = gemv(aq, as, lw.mlaQ)
			}
			kvDown := gemv(aq, as, lw.mlaKVA) // [latDim] = latent ‖ rope-key
			mlaStore(kvDown, lw.mlaKVANorm, lw.invFreq, lw.latCache)
			qAbs := storF(nH * latDim) // [qNopeAbs | qRope] per head
			mlaAbsorb(qf, lw.mlaWUK, qAbs)
			mlaQRopeOp(qf, lw.invFreq, qAbs)
			wsum := storF(nH * rank)
			mlaAttnOp(qAbs, lw.latCache, wsum)
			ctxv := storF(nH * mp.vHead)
			mlaLift(wsum, lw.mlaWUV, ctxv)
			cq, cs := quant(ctxv, nH*mp.vHead)
			gemvAdd(cq, cs, lw.mlaO, r.xd) // o-proj + residual into xd; FFN below is shared
		} else {
			aq, as := rmsQuant(r.xd, lw.attnNorm, hidden)
			// Resolve this layer's attention geometry: its own head_dim/KV-heads/rotaryDim, or the model-level shape when the layer
			// carries no override. geomFor dedups by value, so uniform models reuse one geom.
			ghd, gnKV, ghalf := hd, nKV, half
			if lw.ghd != 0 {
				ghd, gnKV, ghalf = lw.ghd, lw.gnKV, lw.ghalf
			}
			g := geomFor(ghd, gnKV, ghalf, lw.gKEqV)
			var q, k, v, aGate *wgpu.Buffer
			_, w8 := lw.q.(*ResidentW8A8)
			if lw.qBias != nil && w8 { // Qwen2 q/k/v bias folded into the GEMV epilogue (W8A8)
				q = gemvBias(aq, as, lw.q, lw.qBias)
				k = gemvBias(aq, as, lw.k, lw.kBias)
				if !lw.gKEqV {
					v = gemvBias(aq, as, lw.v, lw.vBias)
				}
			} else {
				q, k = gemv(aq, as, lw.q), gemv(aq, as, lw.k)
				if !lw.gKEqV {
					v = gemv(aq, as, lw.v)
				}
				if lw.qBias != nil { // bias on a non-W8A8 weight: standalone add (matches CPU)
					biasAdd(q, lw.qBias, nH*g.hd)
					biasAdd(k, lw.kBias, g.kvDim)
					if !lw.gKEqV {
						biasAdd(v, lw.vBias, g.kvDim)
					}
				}
			}
			if lw.gKEqV {
				// K=V (Gemma 4 globals): V is scale-less v_norm(raw k pre-RoPE).
				v = storF(g.kvDim)
				vNorm(k, v, g.nKV, g.hd)
			}
			// Compute-time LoRA hooks for q/k/v go right after the base projection, before qGate's split/qNorm/RoPE, so a delta
			// applies to the FULL (possibly qGate double-width) buffer the base matmul produced, in applyLoRA's CPU order. All three
			// share one afterIdx (the last of the three base dispatches just recorded, whichever branch produced them) since nothing
			// else runs between them.
			after := len(r.steps) - 1
			r.loraHooks = append(r.loraHooks,
				loraHook{afterIdx: after, layer: i, kind: loraQ, aq: aq, ascale: as, dst: q, k: hidden},
				loraHook{afterIdx: after, layer: i, kind: loraK, aq: aq, ascale: as, dst: k, k: hidden},
			)
			if !lw.gKEqV {
				r.loraHooks = append(r.loraHooks,
					loraHook{afterIdx: after, layer: i, kind: loraV, aq: aq, ascale: as, dst: v, k: hidden},
				)
			}
			if lw.vNorm && !lw.gKEqV {
				// Gemma 4, a layer with a real v_proj: scale-less v_norm of the v_proj output, after the LoRA hooks above (they splice in
				// after the base projection) and before the KV store. The shader reads src and writes dst, and one buffer cannot be bound
				// as both, so it writes a fresh buffer, as the K=V branch does.
				vn := storF(g.kvDim)
				vNorm(v, vn, g.nKV, g.hd)
				v = vn
			}
			if lw.qGate { // attn_output_gate: q_proj emitted [query ‖ gate] per head
				q, aGate = qSplit(q, nH*g.hd, g.hd)
			}
			if lw.qNorm != nil { // Qwen3/GLM per-head QK-norm, after bias, before RoPE (matches CPU)
				qkNorm(q, lw.qNorm, nH, g.hd)
				qkNorm(k, lw.kNorm, g.nKV, g.hd)
			}
			if m.kvF16 || m.kvI8 {
				rope(g, q, lw.invFreq, lw.ropeScale)
				ropeStore(g, k, lw.invFreq, lw.kCache, lw.kScale, lw.ropeScale) // rotate K + append into cache
				vStore(g, v, lw.vCache, lw.vScale)                              // append V into cache
			} else {
				// f32 KV: one fused dispatch for rope(q) + rope-store(k) + store(v).
				qkvFinalize(g, q, k, v, lw.invFreq, lw.kCache, lw.vCache, lw.ropeScale)
			}
			ctxv := storF(nH * g.hd)
			aUni := g.attnUni // local (sliding-window) layers use the windowed start
			if lw.isLocal {
				aUni = g.attnUniLocal
			}
			// head_dim > attnWG needs the 256-lane variants: the kernels put one lane per dim,
			// so the narrow ones would dot 0..127 and leave the tail zeroed. Chosen PER LAYER
			// off this layer's resolved geometry, not the model's, because Gemma 4 already
			// proves per-layer head_dim is a real thing here.
			wide := g.hd > attnWG
			// gpt-oss's per-head attention sink (FeatAttnSink) is always bound (WGSL cannot bind a null storage buffer) and gated by
			// the per-layer hasSinkUni flag, built once per layer in the fixed plan, not per token.
			hasSinkUni := uni([]uint32{boolU32(lw.hasSink), 0, 0, 0})
			attnSinks := lw.attnSinks
			if attnSinks == nil {
				attnSinks = noAttnSinks
			}
			if m.kvI8 {
				// attnI8 reads packed int8 K/V + the per-(pos,head) scale side buffers.
				pl, ly := c.attnI8Pipeline, c.attnI8Layout
				if wide {
					pl, ly = c.attnI8WidePipeline, c.attnI8WideLayout
				}
				add(pl, bind(ly, q, lw.kCache, lw.vCache, lw.kScale, lw.vScale, ctxv, attnSinks, aUni, hasSinkUni), uint32(nH), 1)
			} else {
				attnPl, attnLy := c.attnKernel(g.hd, g.kvDim, m.kvF16)
				add(attnPl, bind(attnLy, q, lw.kCache, lw.vCache, ctxv, attnSinks, aUni, hasSinkUni), uint32(nH), 1)
			}
			if aGate != nil { // ctx *= sigmoid(gate), before o_proj (matches CPU)
				attnGate(ctxv, aGate, nH*g.hd)
			}
			if captureOn { // the pre-o-proj context — decoder's subCtx[l]
				r.capCtx = append(r.capCtx, capStep(ctxv, nH*g.hd))
			}
			cq, cs := quant(ctxv, nH*g.hd)
			// FeatOutBias (gpt-oss o_proj bias) and FeatSandwichNorm (Gemma's post-attn norm) need the sublayer output BEFORE the
			// residual add, which the fused gemvAdd epilogue cannot express: bare gemv, bias/norm, then a separate residual add
			// (biasAdd).
			if lw.oBias != nil || lw.postAttnNorm != nil {
				attnOut := gemv(cq, cs, lw.o)
				r.loraHooks = append(r.loraHooks,
					loraHook{afterIdx: len(r.steps) - 1, layer: i, kind: loraO, aq: cq, ascale: cs, dst: attnOut, k: nH * g.hd})
				if lw.oBias != nil {
					biasAdd(attnOut, lw.oBias, hidden)
				}
				if lw.postAttnNorm != nil {
					attnOut = rmsnormF32(attnOut, lw.postAttnNorm, hidden)
				}
				biasAdd(r.xd, attnOut, hidden)
			} else {
				gemvAdd(cq, cs, lw.o, r.xd) // o-proj + residual into xd
				r.loraHooks = append(r.loraHooks,
					loraHook{afterIdx: len(r.steps) - 1, layer: i, kind: loraO, aq: cq, ascale: cs, dst: r.xd, k: nH * g.hd})
			}
			if captureOn { // residual after the attention sublayer's add — decoder's h after addResidual(h, subAttn)
				r.capAttn = append(r.capAttn, capStep(r.xd, hidden))
			}
		}
		if lw.nemoKind == nemoKMamba || lw.nemoKind == nemoKAttn {
			continue // Nemotron single-op-per-block: the mixer IS the layer — no FFN sub-block
		}
		if ssmSkipFFN { // GOINFER_SSM_SKIPFFN debug: isolate the mixer from the FFN
			continue
		}
		mq, ms := rmsQuant(r.xd, lw.mlpNorm, hidden)
		if lw.isMoE {
			// Sparse MoE FFN: router top-k on the GPU, then for each chosen slot run the
			// indexed gate/up GEMVs (overwrite scratch), fuse SwiGLU→quantize, and the
			// indexed down GEMV accumulates wgt[slot]·expert(h) straight into the residual
			// xd. The gate/up/dq scratch is reused across slots — WebGPU's storage
			// barriers serialize the dependent dispatches, so slot j's down read precedes
			// slot j+1's gate write. xd already holds the post-attention residual.
			mp := m.moe
			logits := gemv(mq, ms, lw.router)
			idx, wgt := storF(mp.k), storF(mp.k)
			bias, hasBias := moeZeroBias, false
			if lw.routerBias != nil {
				bias, hasBias = lw.routerBias, true
			}
			if mp.gptoss {
				// gpt-oss: own router (biased logits select AND weight), own clamped-gated activation (per-expert gate‖up bias), own
				// biased down-combine; see routeGptOssWGSL/gptossGluQuantWGSL/moeExpertGptOssDownGEMVWGSL for each contract.
				routeGptOss(logits, bias, idx, wgt, hasBias)
				gateOut, upOut := storF(mp.inter), storF(mp.inter)
				for j := 0; j < mp.k; j++ {
					moeExpert(mq, ms, lw.expGate, idx, wgt, gateOut, j, 0)
					moeExpert(mq, ms, lw.expUp, idx, wgt, upOut, j, 0)
					dq, ds := gptossActQuant(gateOut, upOut, lw.gateUpBias, idx, j, mp.inter)
					moeExpertGptOssDown(dq, ds, lw.expDown, idx, wgt, lw.downBias, r.xd, j)
				}
			} else {
				moeRoute(logits, bias, idx, wgt, hasBias)
				gateOut, upOut := storF(mp.inter), storF(mp.inter)
				for j := 0; j < mp.k; j++ {
					moeExpert(mq, ms, lw.expGate, idx, wgt, gateOut, j, 0)
					moeExpert(mq, ms, lw.expUp, idx, wgt, upOut, j, 0)
					dq, ds := swigluQuant(gateOut, upOut, mp.inter)
					moeExpert(dq, ds, lw.expDown, idx, wgt, r.xd, j, 1)
				}
			}
			// Always-on shared expert (qwen2_moe / GLM): a single gated SwiGLU MLP added
			// to the residual — sigmoid-gated (qwen2_moe) or ungated (GLM/DeepSeek).
			if lw.shGate != nil {
				sg, su := gemv(mq, ms, lw.shGate), gemv(mq, ms, lw.shUp)
				sdq, sds := swigluQuant(sg, su, mp.sharedInter)
				if lw.shGateW != nil { // qwen2_moe: scale by sigmoid(SharedGate·h)
					sdown := gemv(sdq, sds, lw.shDown)
					gl := gemv(mq, ms, lw.shGateW) // [1] gate logit
					sharedGatedCombine(r.xd, sdown, gl, hidden)
				} else { // GLM/DeepSeek: ungated residual add
					gemvAdd(sdq, sds, lw.shDown, r.xd)
				}
			}
		} else {
			gate, up := gemv(mq, ms, lw.gate), gemv(mq, ms, lw.up)
			// LoRA gate/up deltas land BEFORE the activation below, as in decoder/mlp.go.
			r.loraHooks = append(r.loraHooks,
				loraHook{afterIdx: len(r.steps) - 1, layer: i, kind: loraGate, aq: mq, ascale: ms, dst: gate, k: hidden},
				loraHook{afterIdx: len(r.steps) - 1, layer: i, kind: loraUp, aq: mq, ascale: ms, dst: up, k: hidden},
			)
			dq, ds := swigluQuant(gate, up, inter)
			if lw.postMLPNorm != nil { // FeatSandwichNorm: same defeat-the-fusion shape as o-proj above
				downOut := gemv(dq, ds, lw.down)
				r.loraHooks = append(r.loraHooks,
					loraHook{afterIdx: len(r.steps) - 1, layer: i, kind: loraDown, aq: dq, ascale: ds, dst: downOut, k: inter})
				downOut = rmsnormF32(downOut, lw.postMLPNorm, hidden)
				biasAdd(r.xd, downOut, hidden)
			} else {
				gemvAdd(dq, ds, lw.down, r.xd) // down-proj + residual into xd
				r.loraHooks = append(r.loraHooks,
					loraHook{afterIdx: len(r.steps) - 1, layer: i, kind: loraDown, aq: dq, ascale: ds, dst: r.xd, k: inter})
			}
		}
		if lw.layerScalar != 0 && lw.layerScalar != 1.0 {
			scaleVec(r.xd, lw.layerScalar, hidden)
		}
		if captureOn { // this layer's output residual — decoder's captured[l] / h after the MLP add
			r.capMLP = append(r.capMLP, capStep(r.xd, hidden))
		}
	}
	// Distinct attention geometries the plan built; see DecodeRunner.geomVariants.
	r.geomVariants = len(geomCache)
	fq, fs := rmsQuant(r.xd, m.finalNorm, hidden)
	logits := gemv(fq, fs, m.lmHead)
	r.lastLogits = logits
	if buildErr == nil {
		stag, e := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(r.vocab * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		if e != nil {
			buildErr = e
		} else {
			r.stag = keepBuf(stag)
		}
	}
	// A device-allocation or bind failure during construction (VRAM exhaustion is the common one) is returned so the caller
	// falls back to the staged/CPU path, never a panic.
	if buildErr != nil {
		r.release()
		return nil, fmt.Errorf("gpu: newDecodeRunner: device allocation failed (VRAM exhausted?): %w", buildErr)
	}
	// Snapshot the pristine (no-adapter) plan: SetAdapter rebuilds r.steps from it plus r.loraHooks rather than mutating it.
	r.baseSteps = r.steps
	return r, nil
}

// GeomVariantCount reports how many distinct attention geometries (hd, nKV, rotaryDim) the resident plan built: 1 for
// every uniform-geometry family, 2 for Gemma 4's local/global interleave. Tests assert it is 1 for uniform models.
func (r *DecodeRunner) GeomVariantCount() int { return r.geomVariants }

// writeInputs uploads the per-token input embedding + pos-dependent uniforms (the
// only buffers that vary per call; the fixed dispatch plan reads them). Split out so
// the batched RunN can prime K runners before recording one command buffer. ropePos:
// see posUni's doc comment — equal to pos except for Qwen2.5-VL m-RoPE decode.
func (r *DecodeRunner) writeInputs(x []float32, pos, ropePos int) error {
	if err := r.c.queue.TryWriteBuffer(r.xd, 0, wgpu.ToBytes(x)); err != nil {
		return err
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
		stag, _ := r.c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		defer stag.Release()
		enc, _ := r.c.device.TryCreateCommandEncoder(nil)
		enc.TryCopyBufferToBuffer(b, 0, stag, 0, uint64(n*4))
		cmd, _ := enc.TryFinish(nil)
		r.c.queue.Submit(cmd)
		cmd.Release()
		enc.Release()
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
		stag, e := r.c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		if e != nil {
			return nil, e
		}
		defer stag.Release()
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
	read := func(bufs []*wgpu.Buffer, n int) ([][]float32, error) {
		out := make([][]float32, len(bufs))
		for i, b := range bufs {
			v, e := rd(b, n)
			if e != nil {
				return nil, e
			}
			out[i] = v
		}
		return out, nil
	}
	if ctx, err = read(r.capCtx, r.capCtxN); err != nil {
		return nil, nil, nil, err
	}
	if attn, err = read(r.capAttn, r.capHidden); err != nil {
		return nil, nil, nil, err
	}
	if mlp, err = read(r.capMLP, r.capHidden); err != nil {
		return nil, nil, nil, err
	}
	return ctx, attn, mlp, nil
}

// record appends this runner's dispatch plan to an existing compute pass. The plan reads r.xd and
// r.posUnis (set by writeInputs) and the resident weights + KV caches, leaving logits in r.lastLogits.
// WebGPU inserts the storage barriers between data-dependent dispatches; across batched runners sharing
// one KV cache, a row's kv-store thus correctly precedes a later row's attention read.
func (r *DecodeRunner) record(pass *wgpu.ComputePassEncoder) {
	for _, s := range r.steps {
		pass.SetPipeline(s.pl)
		pass.SetBindGroup(0, s.bg, nil)
		pass.DispatchWorkgroups(s.gx, s.gy, 1)
	}
}

// recordTrunk records all dispatches up to the final norm, omitting the last step (the LM head GEMV).
func (r *DecodeRunner) recordTrunk(pass *wgpu.ComputePassEncoder) {
	steps := r.steps
	if len(steps) > 0 {
		steps = steps[:len(steps)-1]
	}
	for _, s := range steps {
		pass.SetPipeline(s.pl)
		pass.SetBindGroup(0, s.bg, nil)
		pass.DispatchWorkgroups(s.gx, s.gy, 1)
	}
}

// RunNoLogits executes all trunk layers and populates the KV cache for token at position pos,
// skipping the LM head GEMV dispatch, logits staging copy, MapAsync, and readback.
func (r *DecodeRunner) RunNoLogits(x []float32, pos, ropePos int) error {
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
	r.recordTrunk(pass)
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
	c.queue.Submit(cmd)
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
	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	defer enc.Release()
	// One compute pass for the whole token: WebGPU runs the dispatches in record order and the backend inserts the minimal
	// storage-buffer barriers between data-dependent dispatches. The KV appends are compute kernels (rope-store / kv-store),
	// so nothing forces a pass break.
	pass := enc.BeginComputePass(nil)
	r.record(pass)
	// TryEnd and the logits copy are checked, not discarded, so a validation error names its call instead of surfacing later
	// as an opaque "DecodeRunner map failed".
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return nil, fmt.Errorf("gpu: end compute pass: %w", err)
	}
	pass.Release()
	if err := enc.TryCopyBufferToBuffer(r.lastLogits, 0, r.stag, 0, uint64(r.vocab*4)); err != nil {
		return nil, fmt.Errorf("gpu: copy logits→stage: %w", err)
	}
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		return nil, err
	}
	defer cmd.Release()
	r.TEncode = time.Since(te)
	ts := time.Now()
	c.queue.Submit(cmd)
	st := wgpu.MapAsyncStatus(0)
	if err := r.stag.TryMapAsync(wgpu.MapModeRead, 0, uint64(r.vocab*4), func(s wgpu.MapAsyncStatus) { st = s }); err != nil {
		return nil, err
	}
	c.device.Poll(true, nil)
	r.TSync = time.Since(ts)
	if st != wgpu.MapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: DecodeRunner map failed: %v", st)
	}
	copy(r.logitsHost, wgpu.FromBytes[float32](r.stag.GetMappedRange(0, uint(r.vocab*4))))
	r.stag.TryUnmap()
	return r.logitsHost, nil
}

// runBatch executes K runners (sharing the resident weights + KV caches, distinct scratch) over inputs xs[i] at positions
// startPos+i in ONE command buffer: one Submit, one Poll, K logit rows. The runners' steps are recorded in row order into a
// single compute pass, so each row's kv-store is visible to the next row's attention (causal: row i sees positions
// [0, startPos+i]). This amortizes the encode glue and the sync over K but does not make the per-row GPU work cheaper (see
// residentDecoder.VerifyPath). len(runners) must be ≥ len(xs).
func runBatch(c *Context, runners []*DecodeRunner, xs [][]float32, startPos int) ([][]float32, error) {
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
	for i := range n {
		runners[i].record(pass)
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
	c.queue.Submit(cmd)
	sts := make([]wgpu.MapAsyncStatus, n)
	var mapErr error
	for i := range n {
		sts[i] = wgpu.MapAsyncStatus(0)
		if err := runners[i].stag.TryMapAsync(wgpu.MapModeRead, 0, uint64(runners[i].vocab*4), func(s wgpu.MapAsyncStatus) { sts[i] = s }); err != nil {
			mapErr = fmt.Errorf("gpu: runBatch row %d MapAsync: %w", i, err)
			break // later rows are not requested; the ones already requested settle on the Poll below
		}
	}
	c.device.Poll(true, nil) // one sync settles every requested map (success or not)
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
	for _, f := range r.keep {
		f()
	}
	r.keep = nil
}

// Close frees the runner's scratch (not the resident model).
func (r *DecodeRunner) Close() error { r.c.releaseOwned(r.release); return nil }

// dnetRunParams carries the model-level Gated-DeltaNet geometry (uniform across the linear
// layers). keyDim/valueDim/convDim are derived once here rather than at every dispatch, because
// the three of them are easy to conflate: convDim is 2*keyDim+valueDim (the conv runs over
// [q|k|v] together), and rep = nv/nk is the GVA factor mapping value heads to key heads.
type dnetRunParams struct {
	convK      int // depthwise causal conv width
	hk, hv     int // per-head key/query and value dims
	nk, nv     int // key-head and value-head counts
	rep        int // nv/nk — value heads sharing one key head
	keyDim     int // nk*hk
	valueDim   int // nv*hv
	convDim    int // 2*keyDim + valueDim
	stateElems int // nv*hv*hk, the per-layer recurrent state
	eps        float32
}

// g4DropVNormForTest turns off the v_norm that Gemma 4 layers with their own v_proj apply (runLayer.vNorm), so a test can
// show that v_norm is what moves the parity numbers (docs/tasks/task-multimodal-support-2026-10.md). Set only by a test
// seam; never in production.
var g4DropVNormForTest bool
