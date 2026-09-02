//go:build gpu

package gpu

import (
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"time"

	"github.com/cogentcore/webgpu/wgpu"
)

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
	vocab                int
	// geomVariants is the number of distinct attention geometries in the resident plan
	// (len of geomFor's cache): 1 for every uniform-geometry family, 2 for Gemma 4's
	// local/global interleave. A refactor that allocated one uniform per layer instead of
	// per distinct tuple would leave logits byte-identical while quietly multiplying this
	// by the layer count — the GeomVariantCount test asserts it stays 1 for uniform models.
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
)

type posUni struct {
	buf *wgpu.Buffer
	gen func(pos int) []uint32 // uniform contents for this pos
}

// attnGeom is one distinct per-layer attention shape: head_dim (hd), KV-head count
// (nKV), rotary half-width (half = rotaryDim/2), and the attention_k_eq_v flag (kEqV).
// Gemma 4 interleaves two shapes (local hd=256/nKV=8; global hd=512/nKV=2, K=V); every
// other family has exactly one. The per-token uniforms that carry these dims (v-store,
// attn, the windowed-attn variant, and — keyed additionally by rope scale — q-rope /
// k-rope-store / the fused qkv-finalize) are deduplicated by value: geomFor caches one
// attnGeom per distinct {hd, nKV, half, kEqV} tuple, so a uniform-geometry model
// collapses to a single entry with the same buffers, bind groups, and dispatch it had
// before this seam existed. Byte-identity for non-Gemma models is thus structural (a
// shared *attnGeom), not asserted.
//
// kEqV is in the key because the geom OWNS the v-store uniforms (vStoreUni/vStoreI8Uni):
// two layers with equal {hd, nKV, half} but different attention_k_eq_v want different
// v-store behaviour — a K=V layer derives V from K instead of storing a projected V — so
// they must not share a geom. The K=V forward itself (V = v_norm(k), no v_proj, the
// attention V binding aliasing the K cache) lands with Gemma-4 admission, where v_norm
// exists and it is testable; keying on it now keeps that future branch sound.
//
// nH (query-head count) is deliberately NOT in the key: it is a model-level constant, and
// GQA still tracks per-layer nKV because the group ratio is recomputed as nH/nKV per geom
// (see geomFor's attnUni). A family with per-layer QUERY-head counts would have to add nH
// to the key too; none on this seam has that (Gemma 4 is 16 query heads in both variants).
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
	isLocal                                    bool         // sliding-window (local) attention layer (Lever C6); false ⇒ full
	ropeScale                                  float32      // per-layer RoPE cos/sin scale = mscale (Lever C7); 0 ⇒ 1.0

	// Per-layer attention geometry (P1, own-forward residency bridge): this layer's
	// head_dim / KV-heads / rotaryDim-half + attention_k_eq_v. Zero ghd ⇒ use the
	// model-level nH-relative shape (every non-Gemma family leaves these unset); gKEqV is
	// read independently (a K=V layer always sets its full tuple). The plan loop resolves
	// these into a shared *attnGeom via geomFor (per-layer local, deduped by value); kvDim
	// for this layer is gnKV*ghd.
	ghd, gnKV, ghalf int
	gKEqV            bool

	// MoE (Lever C3c, Mixtral-class): when isMoE, this layer's FFN is a sparse
	// mixture of experts instead of the dense gate/up/down above. router scores all
	// nE experts; the on-GPU top-k (moeRoute) writes the chosen indices/weights,
	// then k indexed GEMVs per projection read the right expert out of the stacked
	// buffers (expGate/expUp/expDown) and the down-combine folds the router weight.
	isMoE                   bool
	router                  decodeWeight         // [nE, hidden] router logits
	routerBias              *wgpu.Buffer         // [nE] selection bias (DeepSeek/GLM); nil ⇒ none
	expGate, expUp, expDown *ResidentStackedW8A8 // nE experts stacked per projection

	// Always-on shared expert (Lever C3d, qwen2_moe / GLM). nil shGate ⇒ no shared
	// expert (Mixtral). shGateW is the [1,hidden] sigmoid gate for the qwen2_moe gated
	// combine; nil ⇒ GLM/DeepSeek add the shared expert ungated (plain residual).
	shGate, shUp, shDown, shGateW decodeWeight

	// MLA attention (Lever C4c, DeepSeek/Kimi). Populated when runModel.mla != nil, in
	// which case the runner takes the latent-attention path instead of the q/k/v/o
	// block above. mlaQA/mlaQANorm/mlaQB are the q-LoRA bottleneck (nil mlaQA ⇒ the
	// direct mlaQ, V2-Lite); mlaKVA down-projects to the latent, mlaKVANorm normalizes
	// it; mlaWUK/mlaWUV are the per-head absorb/lift f32 weights; mlaO is the output
	// projection; latCache is this layer's compressed-latent KV cache [ctxCap*latDim].
	mlaQA, mlaQB, mlaQ, mlaKVA, mlaO decodeWeight
	mlaQANorm, mlaKVANorm            *wgpu.Buffer
	mlaWUK, mlaWUV                   *wgpu.Buffer
	latCache                         *wgpu.Buffer

	// Mamba-2 SSM mixer (P5b, Granite-4.0-H/Nemotron-H hybrids). When isMamba, this
	// layer's sequence-mixer is the resident SSM step (mamba.go kernels) instead of
	// attention: in/out_proj are W8A8; convW/convB/headP/normW are f32 resident; win
	// (causal-conv ring) + ssm (selective state) are build-once persistent state,
	// updated in place per token and reset per generation. ResidMul is folded into
	// mambaOutProj's scale (the residual add). isMamba=false ⇒ attention layer (above).
	isMamba bool
	// Nemotron-H single-op-per-block: each layer is exactly ONE op (no mixer+FFN pairing).
	// nemoKind ∈ {nemoNone, nemoKMamba, nemoKAttn, nemoKMLP}; nemoNone ⇒ standard mixer+FFN.
	nemoKind                                       uint8
	mambaInProj, mambaOutProj                      decodeWeight // int8 path (nil when f16)
	mambaInProjF16, mambaOutProjF16                *wgpu.Buffer // f16 path (default); nil ⇒ int8
	mambaConvW, mambaConvB, mambaHeadP, mambaNormW *wgpu.Buffer
	mambaWin, mambaSSM                             *wgpu.Buffer

	// Gated-DeltaNet mixer (Qwen3.5/3.6-MoE, Qwen3-Next, Qwen3.8). When isDeltaNet, this
	// layer's mixer is the recurrent delta rule (deltanet.go kernels) instead of attention.
	// The causal conv is Mamba-2's — same shape, same SiLU, same ring window — so it reuses
	// mambaConvW/mambaWin and binds an all-zero convB (DeltaNet's conv is bias-free).
	// dnState is the [nv*hv*hk] recurrent state, TRANSPOSED relative to the CPU's [hk,hv]
	// so each thread owns a contiguous row; build-once, updated in place, reset per
	// generation alongside mambaWin.
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
	ropeHalf      int             // rotated pairs per head = rotaryDim/2 (Lever C5 partial RoPE); 0 ⇒ HeadDim/2
	slidingWindow int             // >0 ⇒ local layers attend only the last N positions (Lever C6)
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
}

// w8Model adapts the W8A8 ModelW into the precision-agnostic runModel.
func w8Model(m ModelW) runModel {
	rm := runModel{finalNorm: m.FinalNorm.buf, lmHead: m.LMHead}
	for i := range m.Layers {
		lw := &m.Layers[i]
		rm.layers = append(rm.layers, runLayer{
			attnNorm: lw.Attn.Norm.buf, invFreq: lw.Attn.InvFreq.buf,
			kCache: lw.Attn.KCache.buf, vCache: lw.Attn.VCache.buf, mlpNorm: lw.MLPNorm.buf,
			q: lw.Attn.QProj, k: lw.Attn.KProj, v: lw.Attn.VProj, o: lw.Attn.OProj,
			gate: lw.Gate, up: lw.Up, down: lw.Down,
		})
	}
	return rm
}

// NewDecodeRunner builds the persistent plan for a resident W8A8 model.
func (c *Context) NewDecodeRunner(m ModelW, hidden, nH, nKV, hd, inter, start int, eps, scale float32, addOne bool) (*DecodeRunner, error) {
	return c.newDecodeRunner(w8Model(m), hidden, nH, nKV, hd, inter, start, eps, scale, addOne)
}

// newDecodeRunner builds the persistent decode plan for either precision.
// attnHeadDimSupported declines a resident decode plan whose model-level or any per-layer
// head_dim exceeds what the single-query attention kernels can dot. Those kernels run at
// @workgroup_size(128) with a fixed 128-entry `red` reduction array (attention.go, one lane per
// dim), so a head_dim above attnMaxHeadDim would leave the tail dims un-dotted and the
// o-projection would consume half-zero context — plausible-looking WRONG output, no error. The
// caller falls back to the staged/CPU path on this error. MLAAttn guards its own analogous rank
// limit; this covers the softmax/GQA runners including Gemma 4's per-layer head_dim (audit M-12).
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

func (c *Context) newDecodeRunner(m runModel, hidden, nH, nKV, hd, inter, start int, eps, scale float32, addOne bool) (*DecodeRunner, error) {
	// The single-query attention kernels (attention.go) parallelize over head_dim at
	// @workgroup_size(128) with a fixed 128-wide workgroup reduction array (`red: array<f32,128>`).
	// A head_dim > 128 would dot only dims 0..127 and leave ctxv[128..hd) zeroed — the o-projection
	// then consumes half-zero context: plausible-looking WRONG output, no error. Decline here (like a
	// VRAM-exhaustion decline) so the caller falls back to the staged/CPU path; MLAAttn guards its own
	// analogous rank limit (audit M-12). Covers the model-level shape and any per-layer geometry
	// override (Gemma 4's per-layer head_dim), so an admitted arch can never silently truncate.
	// MLA (DeepSeek/Kimi) attention runs the mlaAttn kernel family (mla.go) with its own rank-bounded
	// accumulator, NOT the 128-wide GQA kernels attnHeadDimSupported protects — its qk head dim
	// (qk_nope+qk_rope) is 192 on real V2-Lite/V3/Kimi and legitimately exceeds 128. Applying the GQA
	// guard here regressed EVERY MLA checkpoint off residency (audit R-05, collateral of M-12). Exempt
	// MLA, but enforce the analogous per-lane rank cap MLAAttn itself checks (rank ≤ 1024; audit R-24).
	if m.mla != nil {
		if m.mla.kvLoRARank > 1024 {
			return nil, fmt.Errorf("gpu: newDecodeRunner: MLA kv-LoRA rank %d exceeds the resident per-lane cap 1024; declining to CPU", m.mla.kvLoRARank)
		}
	} else if err := attnHeadDimSupported(hd, m.layers); err != nil {
		return nil, err
	}
	// The mamba causal-conv kernel (mamba.go) holds the window in a fixed `array<f32, 8>` indexed
	// by conv_kernel-1, so a conv_kernel > 8 would overrun it (and == 0 underflow) — plausible-looking
	// WRONG output, no error. Decline here (like the head-dim guard above) so the caller falls back to
	// CPU rather than silently corrupting (N-08; real Mamba-2 uses conv_kernel 4).
	if m.mamba != nil && (m.mamba.dConv > 8 || m.mamba.dConv < 1) {
		return nil, fmt.Errorf("gpu: newDecodeRunner: mamba conv_kernel %d out of range [1,8] for the resident conv kernel; declining to CPU", m.mamba.dConv)
	}
	ssmStopLayer := -1 // GOINFER_SSM_STOP_LAYER debug (resident SSM bring-up): truncate the plan
	if v := os.Getenv("GOINFER_SSM_STOP_LAYER"); v != "" {
		if n, e := strconv.Atoi(v); e == nil {
			ssmStopLayer = n
		}
	}
	ssmSkipFFN := os.Getenv("GOINFER_SSM_SKIPFFN") != "" // debug: mixer-only isolation
	// W8A16 (activation-precision fix, gemv_w8a16.go): int8 weights, f32 activations — no
	// activation int8 quant, so the granite re-quant cascade can't compound. Off by default.
	w8a16 := os.Getenv("GOINFER_SSM_W8A16") != ""
	ensures := []func() error{c.ensureGEMV, c.ensureGEMVBias, c.ensureQuantize, c.ensureLayer, c.ensureAttn, c.ensureFuse, c.ensureGEMVW4, c.ensureQKNorm}
	if w8a16 {
		ensures = append(ensures, c.ensureGEMVW8A16)
	}
	if m.moe != nil {
		ensures = append(ensures, c.ensureMoERoute, c.ensureMoEExpert, c.ensureMoEExpertW4)
		if m.moe.sharedInter > 0 && !m.moe.sharedUngated {
			ensures = append(ensures, c.ensureSharedGate)
		}
	}
	if m.mla != nil {
		ensures = append(ensures, c.ensureMLAStore, c.ensureMLAHeadMV, c.ensureMLAQRope, c.ensureMLAAttn)
	}
	if m.mamba != nil {
		ensures = append(ensures, c.ensureMambaConv, c.ensureMambaSSM, c.ensureMambaGNorm, c.ensureMambaF16, c.ensureRelu2)
	}
	if hd > attnWG || slices.ContainsFunc(m.layers, func(l runLayer) bool { return l.ghd > attnWG }) {
		// Only now, and only for a plan that needs it. A device that cannot do 256-invocation
		// workgroups errors here, which the caller turns into a staged fallback — the same
		// treatment the old hard head_dim decline gave, but reached only by models that need it.
		ensures = append(ensures, c.ensureAttnWide)
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
	r := &DecodeRunner{c: c, vocab: m.lmHead.nRows()}
	// buildErr accumulates the FIRST device-allocation/bind failure (M21): the storF/uni/
	// storFZ/bind helpers short-circuit once it's set and the constructor returns it, so VRAM
	// exhaustion is an error the caller can fall back on — never a panic in library code.
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
		b, e := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
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
		b, e := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(v), Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst})
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
		bg, e := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: es})
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
	// storFZ: a zeroed storage buffer (CopyDst) — W8A16 f32 activations are kp-padded and the
	// tail must be 0 (the int8 weight is zero-padded to kp, so 0·act keeps the dot exact; a NaN
	// in an uninit pad would poison it). Zeroed once at build; producers only write [0,K).
	storFZ := func(n int) *wgpu.Buffer {
		if buildErr != nil {
			return nil
		}
		b, e := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
		if e != nil {
			buildErr = e
			return nil
		}
		c.queue.WriteBuffer(b, 0, make([]byte, n*4))
		return keepBuf(b)
	}
	rmsQuant := func(in, w *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
		if w8a16 { // W8A16: f32 normed activation (kp-padded), nil scale signals the W8A16 GEMV
			out := storFZ(padK(K))
			p := uni([]uint32{uint32(K), f32bits(eps), boolU32(addOne), 0})
			add(c.rmsnormPipeline, bind(c.rmsnormLayout, in, w, out, p), 1, 1)
			return out, nil
		}
		kp := padK32(K) // int8 activation for a W4A8/W8A8 gemv that reads to the weight's kPad==padK32;
		// padK (mult-16) under-sizes it when K%32 ∈ [1,16] → OOB read + int4 zero-pad nibbles decode
		// to −8 (audit R-18 / N-05). Latent since real dims are mult-32; matches the siblings below.
		q, s := storF(kp/4), storF(1)
		p := uni([]uint32{uint32(K), f32bits(eps), boolU32(addOne), uint32(kp)})
		add(c.rmsQuantPipeline, bind(c.rmsQuantLayout, in, w, q, s, p), 1, 1)
		return q, s
	}
	// swigluQuant fuses SwiGLU→quantize: the inter-wide product never materializes
	// or crosses a barrier — one fewer link and the big buffer stays off the spine.
	swigluQuant := func(gate, up *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
		if w8a16 { // W8A16: f32 swiglu (kp-padded), nil scale
			out := storFZ(padK(K))
			p := uni([]uint32{uint32(K), 0, 0, 0})
			add(c.swigluPipeline, bind(c.swigluLayout, gate, up, out, p), uint32(K+63)/64, 1)
			return out, nil
		}
		kp := padK32(K) // int8 activation for a W4A8/W8A8 down-proj gemv, which reads to the weight's
		// kPad == padK32; padK (mult-16) under-sizes it when K%32 != 0 (N-08→N-05: OOB read; latent
		// since real dims are mult-32). padK32 also zeroes the tail, matching the zero-padded weight.
		q, s := storF(kp/4), storF(1)
		p := uni([]uint32{uint32(K), uint32(kp), 0, 0})
		add(c.swigluQuantPipeline, bind(c.swigluQuantLayout, gate, up, q, s, p), 1, 1)
		return q, s
	}
	// relu2Quant fuses Nemotron-H's non-gated relu²(up)→int8 (the squared-ReLU MLP), the
	// unary analog of swigluQuant. Bindings: up / qout / scales / dims (4 — no gate).
	relu2Quant := func(up *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
		kp := padK32(K) // int8 activation for a W4A8/W8A8 gemv reading to padK32 — see swigluQuant (N-05)
		q, s := storF(kp/4), storF(1)
		p := uni([]uint32{uint32(K), uint32(kp), 0, 0})
		add(c.relu2Pipeline, bind(c.relu2Layout, up, q, s, p), 1, 1)
		return q, s
	}
	quant := func(in *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
		if w8a16 { // W8A16: the input is already f32 — pass it through (K is mult-16 for the
			// only granite caller, the mamba out_proj gated[dInner]); nil scale.
			return in, nil
		}
		kp := padK32(K) // int8 activation for a W4A8/W8A8 gemv reading to padK32 — see swigluQuant (N-05)
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
		if as == nil { // W8A16: aq is the f32 activation; int8 weight, no activation scale
			add(c.gemvW8A16Pipeline, bind(c.gemvW8A16Layout, aq, w.wbuf(), w.sbuf(), out, p), gx, gy)
			return out
		}
		add(w.gPipe(c), bind(w.gLayout(c), aq, w.wbuf(), as, w.sbuf(), out, p), gx, gy)
		return out
	}
	// gemvAdd is gemv with the residual fused into the epilogue: dst (the running
	// hidden state) gets dst[n] += result, deleting a standalone residual link.
	gemvAdd := func(aq, as *wgpu.Buffer, w decodeWeight, dst *wgpu.Buffer) {
		p := uni([]uint32{1, uint32(w.kPad()), uint32(w.nRows()), 1})
		gx, gy := gemvGrid(w.nRows())
		if as == nil { // W8A16 + residual
			add(c.gemvW8A16Pipeline, bind(c.gemvW8A16Layout, aq, w.wbuf(), w.sbuf(), dst, p), gx, gy)
			return
		}
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
	// Mamba-2 SSM mixer dispatches (P5b): the conv/ssm/gatedNorm kernels read slices of
	// the in_proj output (z|xBC|dt) via bind-group offsets (256-aligned for granite), so
	// no extra split kernel. The dispatches are pos-independent — {win, ssm} state carries
	// the recurrence. mamba* are nil for non-hybrid models (the closures go unused).
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
		bg, e := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: en})
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
	// f16 mamba projections (quality fix): plain f32 rmsnorm activation + f16 weight GEMV.
	rmsnormF32 := func(in, weight *wgpu.Buffer, n int) *wgpu.Buffer {
		out := storF(n)
		p := uni([]uint32{uint32(n), f32bits(eps), boolU32(addOne), 0})
		add(c.rmsnormPipeline, bind(c.rmsnormLayout, in, weight, out, p), 1, 1)
		return out
	}
	mambaF16Gemv := func(act, wf16 *wgpu.Buffer, N, K int, residual bool, dst *wgpu.Buffer) *wgpu.Buffer {
		out := dst
		if out == nil {
			out = storF(N)
		}
		res := uint32(0)
		if residual {
			res = 1
		}
		p := uni([]uint32{uint32(K), uint32(N), res, 0})
		gx, gy := gemvGrid(N)
		add(c.mambaF16Pipeline, bind(c.mambaF16Layout, act, wf16, out, p), gx, gy)
		return out
	}
	// §4: the per-token uniforms (rope-q, rope-store-k, v-store, attn) depend only
	// on pos, NOT on layer index — their contents are identical across all 28
	// layers. So allocate ONE buffer per type and let every layer's dispatch bind
	// it; Run then writes 4 small uniforms per token instead of ~112. The builders
	// below reference these shared buffers and no longer append per-call posUnis.
	// Rotated pairs per head: rotaryDim/2 = len(invFreq). m.ropeHalf carries it for
	// partial RoPE (GLM/Phi rotary_dim < HeadDim, Lever C5); 0 ⇒ full HeadDim/2. The
	// rope kernels pair vec[off+d] with vec[off+half+d] for d<half, leaving the trailing
	// HeadDim-rotaryDim dims untouched — exactly decoder.applyRoPE's partial layout.
	half := hd / 2
	if m.ropeHalf > 0 {
		half = m.ropeHalf
	}
	// Per-layer RoPE scale (Lever C7): the cos/sin are multiplied by the layer's mscale
	// (YaRN attention_factor; 1.0 for non-YaRN). Most models use one scale; the per-layer-
	// rope interleave families (Mellum: YaRN on the global/full layers, default on the local/
	// sliding ones) use two. Build one shared rope uniform per distinct scale, keyed by value.
	// slot 6 of the K uniform carries nKV for the int8 ropeStore (it indexes
	// scales[pos*nKV+head]); the f32/f16 ropeStore ignore it (their unused _b pad).
	// §4.5 per-layer attention geometry. geomFor builds one attnGeom per distinct
	// {hd, nKV, half} tuple and caches it by value — a uniform-geometry model yields
	// exactly one entry (same buffers/dispatch as before this seam), Gemma 4 two. Each
	// geom owns the per-token uniforms that carry its dims: the v-store, the attn, and the
	// windowed-attn variant here; the rope uniforms below hang off it keyed by rope scale.
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
		r.posUnis = append(r.posUnis, posUni{buf: g.vStoreUni, gen: func(pos int) []uint32 {
			return []uint32{uint32(g.kvDim), uint32(pos * g.kvDim), 0, 0}
		}})
		// int8 V store needs its own (differently-laid-out) per-token uniform:
		// {heads=nKV, headDim=hd, base=pos*kvDim, pos, nKV}. Only allocated for kvI8.
		if m.kvI8 {
			g.vStoreI8Uni = uni([]uint32{uint32(g.nKV), uint32(g.hd), 0, 0, uint32(g.nKV), 0, 0, 0})
			r.posUnis = append(r.posUnis, posUni{buf: g.vStoreI8Uni, gen: func(pos int) []uint32 {
				return []uint32{uint32(g.nKV), uint32(g.hd), uint32(pos * g.kvDim), uint32(pos), uint32(g.nKV), 0, 0, 0}
			}})
		}
		g.attnUni = uni([]uint32{uint32(nH), uint32(g.nKV), uint32(g.hd), 0, uint32(start), uint32(nH / g.nKV), f32bits(scale), 0})
		r.posUnis = append(r.posUnis, posUni{buf: g.attnUni, gen: func(pos int) []uint32 {
			return []uint32{uint32(nH), uint32(g.nKV), uint32(g.hd), uint32(pos + 1), uint32(start), uint32(nH / g.nKV), f32bits(scale), 0}
		}})
		// Sliding-window (local) layers attend only the last `slidingWindow` positions: the
		// attention start advances to max(0, pos+1-W) once pos reaches the window (Lever C6),
		// matching decoder.KVCache.WindowStart. Full layers keep attnUni (start fixed). Only
		// built when the model windows; local layers bind this instead of attnUni.
		g.attnUniLocal = g.attnUni
		if m.slidingWindow > 0 {
			w := m.slidingWindow
			g.attnUniLocal = uni([]uint32{uint32(nH), uint32(g.nKV), uint32(g.hd), 0, uint32(start), uint32(nH / g.nKV), f32bits(scale), 0})
			r.posUnis = append(r.posUnis, posUni{buf: g.attnUniLocal, gen: func(pos int) []uint32 {
				ws := start
				if lo := pos + 1 - w; lo > ws {
					ws = lo
				}
				return []uint32{uint32(nH), uint32(g.nKV), uint32(g.hd), uint32(pos + 1), uint32(ws), uint32(nH / g.nKV), f32bits(scale), 0}
			}})
		}
		geomCache[key] = g
		return g
	}
	// Per-(geom, rope-scale) rope uniforms. slot 6 of the K uniform carries nKV for the
	// int8 ropeStore (it indexes scales[pos*nKV+head]); the f32/f16 ropeStore ignore it.
	ropeQUniFor := func(g *attnGeom, rs float32) *wgpu.Buffer {
		if b, ok := g.ropeQUnis[rs]; ok {
			return b
		}
		b := uni([]uint32{uint32(nH), uint32(g.hd), uint32(g.half), 0, f32bits(rs), 0, 0, 0})
		r.posUnis = append(r.posUnis, posUni{buf: b, gen: func(pos int) []uint32 {
			return []uint32{uint32(nH), uint32(g.hd), uint32(g.half), uint32(pos), f32bits(rs), 0, 0, 0}
		}})
		g.ropeQUnis[rs] = b
		return b
	}
	ropeKUniFor := func(g *attnGeom, rs float32) *wgpu.Buffer {
		if b, ok := g.ropeKUnis[rs]; ok {
			return b
		}
		b := uni([]uint32{uint32(g.nKV), uint32(g.hd), uint32(g.half), 0, f32bits(rs), 0, uint32(g.nKV), 0})
		r.posUnis = append(r.posUnis, posUni{buf: b, gen: func(pos int) []uint32 {
			return []uint32{uint32(g.nKV), uint32(g.hd), uint32(g.half), uint32(pos), f32bits(rs), uint32(pos * g.kvDim), uint32(g.nKV), 0}
		}})
		g.ropeKUnis[rs] = b
		return b
	}
	// Fused q-rope + k-rope-store + v-store uniform (decode fusion, f32 KV), per (geom,
	// scale): {nH, nKV, hd, half, pos, base=pos*kvDim, scale, kvDim}.
	qkvFinUniFor := func(g *attnGeom, rs float32) *wgpu.Buffer {
		if b, ok := g.qkvFinUnis[rs]; ok {
			return b
		}
		b := uni([]uint32{uint32(nH), uint32(g.nKV), uint32(g.hd), uint32(g.half), 0, 0, f32bits(rs), uint32(g.kvDim)})
		r.posUnis = append(r.posUnis, posUni{buf: b, gen: func(pos int) []uint32 {
			return []uint32{uint32(nH), uint32(g.nKV), uint32(g.hd), uint32(g.half), uint32(pos), uint32(pos * g.kvDim), f32bits(rs), uint32(g.kvDim)}
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
			// word-based (2 f16/word): kvDim/2 = nKV·hd/2 words, covering the rotated span AND the
			// partial-rotary pass-through tail (C4).
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
	// biasAdd adds a per-output bias into a projection result (Qwen2 q/k/v bias),
	// reusing the residual kernel (vec[i] += bias[i]); n is the projection width.
	biasAdd := func(vec, bias *wgpu.Buffer, n int) {
		p := uni([]uint32{uint32(n), 0, 0, 0})
		add(c.residualPipeline, bind(c.residualLayout, vec, bias, p), uint32(n+63)/64, 1)
	}
	// qkNorm RMS-normalizes each of `heads` heads of vec (q or k) over headDim in place
	// with weight[hd], before RoPE (Qwen3/GLM/Mellum). One workgroup per head; the
	// uniform is pos-independent so it's a plain uni, not a posUni.
	qkNorm := func(vec, weight *wgpu.Buffer, heads int) {
		p := uni([]uint32{uint32(heads), uint32(hd), f32bits(eps), boolU32(addOne)})
		add(c.qkNormPipeline, bind(c.qkNormLayout, vec, weight, p), uint32(heads), 1)
	}
	// MoE op builders (Lever C3c). moeRoute records the on-GPU router top-k SELECTION
	// (logits[nE] + optional bias → idx[k], wgt[k]); the p uniform is pos-independent
	// (model-level shape), so it's a plain uni. nE is tiny ⇒ one single-lane workgroup.
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
			if as == nil { // W8A16: f32 activation, int8 stacked weight
				add(c.moeExpertW8A16Pipeline, bind(c.moeExpertW8A16Layout, aq, s.bq, s.bScales, dst, idx, wgt, d), gx, gy)
				return
			}
			add(c.moeExpertPipeline, bind(c.moeExpertLayout, aq, s.bq, as, s.bScales, dst, idx, wgt, d), gx, gy)
		}
	}
	// sharedGatedCombine records the qwen2_moe gated shared-expert add: dst[n] +=
	// sigmoid(gl[0])·src[n]. The GLM/DeepSeek ungated case uses gemvAdd instead.
	sharedGatedCombine := func(dst, src, gl *wgpu.Buffer, n int) {
		p := uni([]uint32{uint32(n), 0, 0, 0})
		add(c.sharedGatePipeline, bind(c.sharedGateLayout, dst, src, gl, p), uint32(n+63)/64, 1)
	}

	// MLA op builders (Lever C4c). The latent store + attention uniforms are
	// pos-dependent (base = pos·latDim, nKeys = pos+1), so they register posUnis like
	// the standard attention path; the absorb/lift matvecs are pos-independent shapes.
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
		r.posUnis = append(r.posUnis, posUni{buf: mlaStoreUni, gen: func(pos int) []uint32 {
			return []uint32{uint32(rank), uint32(mp.qkRope), uint32(pos), f32bits(eps), uint32(pos * latDim), f32bits(mp.ropeScale), boolU32(mp.interleave), 0}
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
		r.posUnis = append(r.posUnis, posUni{buf: mlaQRopeUni, gen: func(pos int) []uint32 {
			return []uint32{uint32(nH), uint32(qkHead), uint32(mp.qkNope), uint32(mp.qkRope), uint32(rank), uint32(latDim), uint32(pos), boolU32(mp.interleave), f32bits(mp.ropeScale), 0, 0, 0}
		}})
		mlaQRopeOp = func(q, invFreq, qAbs *wgpu.Buffer) {
			add(c.mlaQRopePipeline, bind(c.mlaQRopeLayout, q, invFreq, qAbs, mlaQRopeUni), uint32(nH*rhalf+63)/64, 1)
		}
		// Attention: rank-space online-softmax over nKeys = pos+1 latents. pos-dependent.
		mlaAttnUni := uni([]uint32{uint32(nH), uint32(latDim), uint32(rank), 0, f32bits(scale), 0, 0, 0})
		r.posUnis = append(r.posUnis, posUni{buf: mlaAttnUni, gen: func(pos int) []uint32 {
			return []uint32{uint32(nH), uint32(latDim), uint32(rank), uint32(pos + 1), f32bits(scale), 0, 0, 0}
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
		b, e := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(hidden * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc})
		if e != nil {
			buildErr = e
			return nil
		}
		return keepBuf(b)
	}()

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
			break // GOINFER_SSM_STOP_LAYER debug: logits from xd after this layer
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
		if lw.isMamba {
			// Mamba-2 SSM mixer (P5b): norm → in_proj → conv(ring) → ssm(state) → gatedNorm
			// → out_proj+residual. State {win, ssm} persists in lw, updated in place per token.
			// ResidMul folded into the out_proj weights. The FFN sub-block below is shared.
			mp := m.mamba
			var proj *wgpu.Buffer
			if lw.mambaInProjF16 != nil { // f16 path (quality): f32 rmsnorm activation + f16 GEMV
				normed := rmsnormF32(r.xd, lw.attnNorm, hidden)
				proj = mambaF16Gemv(normed, lw.mambaInProjF16, mp.projDim, hidden, false, nil)
			} else { // int8 path (flag)
				aq, as := rmsQuant(r.xd, lw.attnNorm, hidden)
				proj = gemv(aq, as, lw.mambaInProj)
			}
			conv := storF(mp.convDim)
			mambaConvOp(proj, lw.mambaConvW, lw.mambaConvB, lw.mambaWin, conv)
			y := storF(mp.dInner)
			mambaSSMOp(conv, proj, lw.mambaHeadP, lw.mambaSSM, y)
			gated := storF(mp.dInner)
			mambaGNormOp(y, proj, lw.mambaNormW, gated)
			if r.mcapProj == nil { // debug: capture the FIRST mamba layer for the wiring diff
				r.mcapProj, r.mcapConv, r.mcapY, r.mcapGated = proj, conv, y, gated
			}
			if lw.mambaOutProjF16 != nil { // f16 out_proj + residual (ResidMul folded into weights)
				mambaF16Gemv(gated, lw.mambaOutProjF16, hidden, mp.dInner, true, r.xd)
			} else {
				gq, gs := quant(gated, mp.dInner)
				gemvAdd(gq, gs, lw.mambaOutProj, r.xd)
			}
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
			// MLA latent attention (Lever C4c): input-norm → q (LoRA/direct) + kv-down →
			// latent store → W_UK-absorb + qRope → rank-space attend → W_UV-lift → o-proj.
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
			// Resolve this layer's attention geometry (P1): its own head_dim/KV-heads/
			// rotaryDim, or the model-level shape when the layer carries no override (every
			// non-Gemma family). geomFor dedups by value, so uniform models reuse one geom.
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
				v = gemvBias(aq, as, lw.v, lw.vBias)
			} else {
				q, k, v = gemv(aq, as, lw.q), gemv(aq, as, lw.k), gemv(aq, as, lw.v)
				if lw.qBias != nil { // bias on a non-W8A8 weight: standalone add (matches CPU)
					biasAdd(q, lw.qBias, nH*g.hd)
					biasAdd(k, lw.kBias, g.kvDim)
					biasAdd(v, lw.vBias, g.kvDim)
				}
			}
			if lw.qGate { // attn_output_gate: q_proj emitted [query ‖ gate] per head
				q, aGate = qSplit(q, nH*g.hd, g.hd)
			}
			if lw.qNorm != nil { // Qwen3/GLM per-head QK-norm, after bias, before RoPE (matches CPU)
				qkNorm(q, lw.qNorm, nH)
				qkNorm(k, lw.kNorm, g.nKV)
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
			aUni := g.attnUni // local (sliding-window) layers use the windowed start (Lever C6)
			if lw.isLocal {
				aUni = g.attnUniLocal
			}
			// head_dim > attnWG needs the 256-lane variants: the kernels put one lane per dim,
			// so the narrow ones would dot 0..127 and leave the tail zeroed. Chosen PER LAYER
			// off this layer's resolved geometry, not the model's, because Gemma 4 already
			// proves per-layer head_dim is a real thing here.
			wide := g.hd > attnWG
			if m.kvI8 {
				// attnI8 reads packed int8 K/V + the per-(pos,head) scale side buffers.
				pl, ly := c.attnI8Pipeline, c.attnI8Layout
				if wide {
					pl, ly = c.attnI8WidePipeline, c.attnI8WideLayout
				}
				add(pl, bind(ly, q, lw.kCache, lw.vCache, lw.kScale, lw.vScale, ctxv, aUni), uint32(nH), 1)
			} else {
				attnPl, attnLy := c.attnPipeline, c.attnLayout
				switch {
				case m.kvF16 && wide:
					attnPl, attnLy = c.attnF16WidePipeline, c.attnF16WideLayout
				case m.kvF16:
					attnPl, attnLy = c.attnF16Pipeline, c.attnF16Layout
				case wide:
					attnPl, attnLy = c.attnWidePipeline, c.attnWideLayout
				case !attnKeysDisabled && attnKeysEligible(g.hd, g.kvDim, m.kvF16, m.kvI8):
					// Key-split attention: one reduction per TILE instead of one per key.
					// Last case on purpose — the f16/wide paths above have their own kernels
					// and attnKeysEligible declines them anyway, so this only ever claims the
					// plain f32 narrow geometry the old kernel used to serve.
					attnPl, attnLy = c.attnKeysPipeline, c.attnKeysLayout
				}
				add(attnPl, bind(attnLy, q, lw.kCache, lw.vCache, ctxv, aUni), uint32(nH), 1)
			}
			if aGate != nil { // ctx *= sigmoid(gate), before o_proj (matches CPU)
				attnGate(ctxv, aGate, nH*g.hd)
			}
			cq, cs := quant(ctxv, nH*g.hd)
			gemvAdd(cq, cs, lw.o, r.xd) // o-proj + residual into xd
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
			moeRoute(logits, bias, idx, wgt, hasBias)
			gateOut, upOut := storF(mp.inter), storF(mp.inter)
			for j := 0; j < mp.k; j++ {
				moeExpert(mq, ms, lw.expGate, idx, wgt, gateOut, j, 0)
				moeExpert(mq, ms, lw.expUp, idx, wgt, upOut, j, 0)
				dq, ds := swigluQuant(gateOut, upOut, mp.inter)
				moeExpert(dq, ds, lw.expDown, idx, wgt, r.xd, j, 1)
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
			dq, ds := swigluQuant(gate, up, inter)
			gemvAdd(dq, ds, lw.down, r.xd) // down-proj + residual into xd
		}
	}
	// Distinct attention geometries the plan actually built (1 for uniform families, 2 for
	// Gemma 4). The GeomVariantCount test asserts this stays 1 for uniform models — a
	// regression that allocated per-layer instead of per-tuple would silently inflate it.
	r.geomVariants = len(geomCache)
	fq, fs := rmsQuant(r.xd, m.finalNorm, hidden)
	logits := gemv(fq, fs, m.lmHead)
	r.lastLogits = logits
	if buildErr == nil {
		stag, e := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(r.vocab * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		if e != nil {
			buildErr = e
		} else {
			r.stag = keepBuf(stag)
		}
	}
	// M21: any device-allocation/bind failure during construction (VRAM exhaustion is the
	// common one) is returned so the caller falls back to the staged/CPU path — never a panic.
	if buildErr != nil {
		r.release()
		return nil, fmt.Errorf("gpu: newDecodeRunner: device allocation failed (VRAM exhausted?): %w", buildErr)
	}
	return r, nil
}

// GeomVariantCount reports how many distinct attention geometries (hd, nKV, rotaryDim)
// the resident plan built: 1 for every uniform-geometry family, 2 for Gemma 4's
// local/global interleave. Tests assert it is 1 for uniform models — the value-keyed
// dedup collapsing to a single entry is what makes non-Gemma byte-identity structural.
func (r *DecodeRunner) GeomVariantCount() int { return r.geomVariants }

// writeInputs uploads the per-token input embedding + pos-dependent uniforms (the
// only buffers that vary per call; the fixed dispatch plan reads them). Split out so
// the batched RunN can prime K runners before recording one command buffer.
func (r *DecodeRunner) writeInputs(x []float32, pos int) error {
	if err := r.c.queue.WriteBuffer(r.xd, 0, wgpu.ToBytes(x)); err != nil {
		return err
	}
	for _, pu := range r.posUnis {
		if err := r.c.queue.WriteBuffer(pu.buf, 0, wgpu.ToBytes(pu.gen(pos))); err != nil {
			return err
		}
	}
	return nil
}

// record appends this runner's dispatch plan to an existing compute pass. The plan
// reads r.xd / r.posUnis (set by writeInputs) and the resident weights + KV caches,
// leaving logits in r.lastLogits. WebGPU inserts the storage barriers between
// data-dependent dispatches; across batched runners sharing one KV cache, a row's
// kv-store thus correctly precedes a later row's attention read.
// ReadMambaCap copies the first mamba layer's captured proj/conv/y/gated buffers (their
// values from the most recent Run) back to the host — the resident's actual per-token kernel
// I/O, for diffing against mamba2Step (gpu/mamba_resident_capture_test.go). projN/convN/dInner
// are the element counts. Test-only; allocates fresh staging per call.
func (r *DecodeRunner) ReadMambaCap(projN, convN, dInner int) (proj, conv, y, gated []float32) {
	rd := func(b *wgpu.Buffer, n int) []float32 {
		stag, _ := r.c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		defer stag.Release()
		enc, _ := r.c.device.CreateCommandEncoder(nil)
		enc.CopyBufferToBuffer(b, 0, stag, 0, uint64(n*4))
		cmd, _ := enc.Finish(nil)
		r.c.queue.Submit(cmd)
		cmd.Release()
		enc.Release()
		st := wgpu.BufferMapAsyncStatusUnknown
		stag.MapAsync(wgpu.MapModeRead, 0, uint64(n*4), func(s wgpu.BufferMapAsyncStatus) { st = s })
		r.c.device.Poll(true, nil)
		if st != wgpu.BufferMapAsyncStatusSuccess {
			panic("ReadMambaCap: map failed")
		}
		out := make([]float32, n)
		copy(out, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(n*4))))
		stag.Unmap()
		return out
	}
	return rd(r.mcapProj, projN), rd(r.mcapConv, convN), rd(r.mcapY, dInner), rd(r.mcapGated, dInner)
}

func (r *DecodeRunner) record(pass *wgpu.ComputePassEncoder) {
	for _, s := range r.steps {
		pass.SetPipeline(s.pl)
		pass.SetBindGroup(0, s.bg, nil)
		pass.DispatchWorkgroups(s.gx, s.gy, 1)
	}
}

// Run executes the plan for one token at absolute position pos. x is the token's
// input embedding [hidden]; returns the logits [vocab]. One Submit + one Poll.
func (r *DecodeRunner) Run(x []float32, pos int) ([]float32, error) {
	c := r.c
	tw := time.Now()
	if err := r.writeInputs(x, pos); err != nil {
		return nil, err
	}
	r.TWrite = time.Since(tw)
	te := time.Now()
	enc, err := c.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	defer enc.Release()
	// One compute pass for the whole token: WebGPU runs the dispatches in record
	// order and the backend inserts the minimal storage-buffer barriers between
	// data-dependent dispatches. The KV appends are now compute kernels (rope-store
	// / kv-store), so nothing forces a pass break.
	pass := enc.BeginComputePass(nil)
	r.record(pass)
	pass.End()
	pass.Release()
	enc.CopyBufferToBuffer(r.lastLogits, 0, r.stag, 0, uint64(r.vocab*4))
	cmd, err := enc.Finish(nil)
	if err != nil {
		return nil, err
	}
	defer cmd.Release()
	r.TEncode = time.Since(te)
	ts := time.Now()
	c.queue.Submit(cmd)
	st := wgpu.BufferMapAsyncStatusUnknown
	if err := r.stag.MapAsync(wgpu.MapModeRead, 0, uint64(r.vocab*4), func(s wgpu.BufferMapAsyncStatus) { st = s }); err != nil {
		return nil, err
	}
	c.device.Poll(true, nil)
	r.TSync = time.Since(ts)
	if st != wgpu.BufferMapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: DecodeRunner map failed: %v", st)
	}
	out := make([]float32, r.vocab)
	copy(out, wgpu.FromBytes[float32](r.stag.GetMappedRange(0, uint(r.vocab*4))))
	r.stag.Unmap()
	return out, nil
}

// runBatch executes K runners (sharing the resident weights + KV caches, distinct
// scratch) over inputs xs[i] at positions startPos+i in ONE command buffer — one
// Submit, one Poll, K logit rows. The runners' steps are recorded in row order into a
// single compute pass, so each row's kv-store is visible to the next row's attention
// (causal: row i sees positions [0, startPos+i]). This amortizes the cgo-encode glue
// + the sync over K (the dominant decode cost — see gpu-assessment.md §0.5), which is
// the speculative-decode win. len(runners) must be ≥ len(xs).
func runBatch(c *Context, runners []*DecodeRunner, xs [][]float32, startPos int) ([][]float32, error) {
	n := len(xs)
	if n == 0 {
		return nil, nil
	}
	for i := range n {
		if err := runners[i].writeInputs(xs[i], startPos+i); err != nil {
			return nil, err
		}
	}
	enc, err := c.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	for i := range n {
		runners[i].record(pass)
	}
	pass.End()
	pass.Release()
	for i := range n {
		enc.CopyBufferToBuffer(runners[i].lastLogits, 0, runners[i].stag, 0, uint64(runners[i].vocab*4))
	}
	cmd, err := enc.Finish(nil)
	if err != nil {
		return nil, err
	}
	defer cmd.Release()
	c.queue.Submit(cmd)
	sts := make([]wgpu.BufferMapAsyncStatus, n)
	var mapErr error
	for i := range n {
		sts[i] = wgpu.BufferMapAsyncStatusUnknown
		if err := runners[i].stag.MapAsync(wgpu.MapModeRead, 0, uint64(runners[i].vocab*4), func(s wgpu.BufferMapAsyncStatus) { sts[i] = s }); err != nil {
			mapErr = fmt.Errorf("gpu: runBatch row %d MapAsync: %w", i, err)
			break // later rows are not requested; the ones already requested settle on the Poll below
		}
	}
	c.device.Poll(true, nil) // one sync settles every requested map (success or not)
	// runners[i].stag is a PERSISTENT per-runner buffer. Returning while any stag is still mapped
	// leaves it mapped forever, and every future MapAsync on it fails — poisoning the runner for the
	// process lifetime (audit C-28). consumed[i] marks rows the readback already Unmapped; this
	// deferred sweep Unmaps any that mapped but weren't consumed, on ALL return paths (a MapAsync
	// error, a non-Success status, or a clean finish — where nothing is left to sweep).
	consumed := make([]bool, n)
	defer func() {
		for i := range n {
			if sts[i] == wgpu.BufferMapAsyncStatusSuccess && !consumed[i] {
				runners[i].stag.Unmap()
			}
		}
	}()
	if mapErr != nil {
		return nil, mapErr
	}
	out := make([][]float32, n)
	for i := range n {
		if sts[i] != wgpu.BufferMapAsyncStatusSuccess {
			return nil, fmt.Errorf("gpu: runBatch row %d map failed: %v", i, sts[i])
		}
		row := make([]float32, runners[i].vocab)
		copy(row, wgpu.FromBytes[float32](runners[i].stag.GetMappedRange(0, uint(runners[i].vocab*4))))
		runners[i].stag.Unmap()
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

// Release frees the runner's scratch (not the resident model).
func (r *DecodeRunner) Close() error { r.release(); return nil }

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
