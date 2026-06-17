//go:build gpu

package gpu

import (
	"fmt"
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
	kvDim                int
	keep                 []func()

	// §5 instrumentation: wall time of each Run phase, overwritten per call.
	// Zero overhead when ignored; the decomposition test reads them.
	TWrite, TEncode, TSync time.Duration
}

type runStep struct {
	pl     *wgpu.ComputePipeline
	bg     *wgpu.BindGroup
	gx, gy uint32
}

type posUni struct {
	buf *wgpu.Buffer
	gen func(pos int) []uint32 // uniform contents for this pos
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
}

type runModel struct {
	layers    []runLayer
	finalNorm *wgpu.Buffer
	lmHead    decodeWeight
	kvF16     bool          // KCache/VCache are f16-packed (NewKVCacheF16) → use the f16 kernels
	kvI8      bool          // KCache/VCache are int8-packed (NewKVCacheI8) + scales → int8 kernels
	moe       *moeRunParams // non-nil ⇒ the model has MoE layers (runLayer.isMoE picks which)
	mla       *mlaRunParams // non-nil ⇒ MLA latent attention replaces the q/k/v/o block
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
func (c *Context) newDecodeRunner(m runModel, hidden, nH, nKV, hd, inter, start int, eps, scale float32, addOne bool) (*DecodeRunner, error) {
	ensures := []func() error{c.ensureGEMV, c.ensureQuantize, c.ensureLayer, c.ensureAttn, c.ensureFuse, c.ensureGEMVW4, c.ensureQKNorm}
	if m.moe != nil {
		ensures = append(ensures, c.ensureMoERoute, c.ensureMoEExpert)
		if m.moe.sharedInter > 0 && !m.moe.sharedUngated {
			ensures = append(ensures, c.ensureSharedGate)
		}
	}
	if m.mla != nil {
		ensures = append(ensures, c.ensureMLAStore, c.ensureMLAHeadMV, c.ensureMLAQRope, c.ensureMLAAttn)
	}
	for _, e := range ensures {
		if err := e(); err != nil {
			return nil, err
		}
	}
	r := &DecodeRunner{c: c, vocab: m.lmHead.nRows(), kvDim: nKV * hd}
	keepBuf := func(b *wgpu.Buffer) *wgpu.Buffer { r.keep = append(r.keep, b.Release); return b }
	keepBG := func(b *wgpu.BindGroup) *wgpu.BindGroup { r.keep = append(r.keep, b.Release); return b }
	storF := func(n int) *wgpu.Buffer {
		b, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
		return keepBuf(b)
	}
	uni := func(v []uint32) *wgpu.Buffer {
		b, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(v), Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst})
		return keepBuf(b)
	}
	bind := func(layout *wgpu.BindGroupLayout, bufs ...*wgpu.Buffer) *wgpu.BindGroup {
		es := make([]wgpu.BindGroupEntry, len(bufs))
		for i, b := range bufs {
			es[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()}
		}
		bg, e := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: es})
		if e != nil {
			r.release()
			panic(e)
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
		kp := padK(K)
		q, s := storF(kp/4), storF(1)
		p := uni([]uint32{uint32(K), f32bits(eps), boolU32(addOne), uint32(kp)})
		add(c.rmsQuantPipeline, bind(c.rmsQuantLayout, in, w, q, s, p), 1, 1)
		return q, s
	}
	// swigluQuant fuses SwiGLU→quantize: the inter-wide product never materializes
	// or crosses a barrier — one fewer link and the big buffer stays off the spine.
	swigluQuant := func(gate, up *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
		kp := padK(K)
		q, s := storF(kp/4), storF(1)
		p := uni([]uint32{uint32(K), uint32(kp), 0, 0})
		add(c.swigluQuantPipeline, bind(c.swigluQuantLayout, gate, up, q, s, p), 1, 1)
		return q, s
	}
	quant := func(in *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
		kp := padK(K)
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
	// §4: the per-token uniforms (rope-q, rope-store-k, v-store, attn) depend only
	// on pos, NOT on layer index — their contents are identical across all 28
	// layers. So allocate ONE buffer per type and let every layer's dispatch bind
	// it; Run then writes 4 small uniforms per token instead of ~112. The builders
	// below reference these shared buffers and no longer append per-call posUnis.
	half := hd / 2
	ropeQUni := uni([]uint32{uint32(nH), uint32(hd), uint32(half), 0, f32bits(1), 0, 0, 0})
	r.posUnis = append(r.posUnis, posUni{buf: ropeQUni, gen: func(pos int) []uint32 {
		return []uint32{uint32(nH), uint32(hd), uint32(half), uint32(pos), f32bits(1), 0, 0, 0}
	}})
	// slot 6 carries nKV for the int8 ropeStore (it indexes scales[pos*nKV+head]);
	// the f32/f16 ropeStore ignore it (it's their unused _b pad), so it rides along.
	ropeKUni := uni([]uint32{uint32(nKV), uint32(hd), uint32(half), 0, f32bits(1), 0, uint32(nKV), 0})
	r.posUnis = append(r.posUnis, posUni{buf: ropeKUni, gen: func(pos int) []uint32 {
		return []uint32{uint32(nKV), uint32(hd), uint32(half), uint32(pos), f32bits(1), uint32(pos * r.kvDim), uint32(nKV), 0}
	}})
	vStoreUni := uni([]uint32{uint32(r.kvDim), 0, 0, 0})
	r.posUnis = append(r.posUnis, posUni{buf: vStoreUni, gen: func(pos int) []uint32 {
		return []uint32{uint32(r.kvDim), uint32(pos * r.kvDim), 0, 0}
	}})
	// int8 V store needs its own (differently-laid-out) per-token uniform:
	// {heads=nKV, headDim=hd, base=pos*kvDim, pos, nKV}. Only allocated for kvI8.
	var vStoreI8Uni *wgpu.Buffer
	if m.kvI8 {
		vStoreI8Uni = uni([]uint32{uint32(nKV), uint32(hd), 0, 0, uint32(nKV), 0, 0, 0})
		r.posUnis = append(r.posUnis, posUni{buf: vStoreI8Uni, gen: func(pos int) []uint32 {
			return []uint32{uint32(nKV), uint32(hd), uint32(pos * r.kvDim), uint32(pos), uint32(nKV), 0, 0, 0}
		}})
	}
	attnUni := uni([]uint32{uint32(nH), uint32(nKV), uint32(hd), 0, uint32(start), uint32(nH / nKV), f32bits(scale), 0})
	r.posUnis = append(r.posUnis, posUni{buf: attnUni, gen: func(pos int) []uint32 {
		return []uint32{uint32(nH), uint32(nKV), uint32(hd), uint32(pos + 1), uint32(start), uint32(nH / nKV), f32bits(scale), 0}
	}})
	rope := func(vec, invFreq *wgpu.Buffer) {
		add(c.ropePipeline, bind(c.ropeLayout, vec, invFreq, ropeQUni), uint32(nH*half+63)/64, 1)
	}
	// ropeStore rotates src (the K projection) and writes it straight into the KV
	// cache at pos*kvDim — replacing the K CopyBufferToBuffer append so the token
	// stays one compute pass. base rides the shared ropeKUni. The f16 variant packs
	// 2 rotated elems/word (one thread per word = nKV*half, same dispatch count).
	ropeStore := func(src, invFreq, cache, scale *wgpu.Buffer) {
		if m.kvI8 {
			// one thread per KV head: per-head absmax → scale → quantize + pack 4/word.
			add(c.ropeStoreI8Pipeline, bind(c.ropeStoreI8Layout, src, invFreq, cache, scale, ropeKUni), uint32(nKV+63)/64, 1)
			return
		}
		pl, ly := c.ropeStorePipeline, c.ropeStoreLayout
		if m.kvF16 {
			pl, ly = c.ropeStoreF16Pipeline, c.ropeStoreF16Layout
		}
		add(pl, bind(ly, src, invFreq, cache, ropeKUni), uint32(nKV*half+63)/64, 1)
	}
	// vStore copies src (the V projection) into the V cache at pos*kvDim. The f16
	// variant packs 2 elems/word, so it dispatches half as many threads (one/word);
	// the int8 variant is one thread per KV head (per-head absmax → scale → pack).
	vStore := func(src, cache, scale *wgpu.Buffer) {
		if m.kvI8 {
			add(c.kvStoreI8Pipeline, bind(c.kvStoreI8Layout, src, cache, scale, vStoreI8Uni), uint32(nKV+63)/64, 1)
			return
		}
		if m.kvF16 {
			words := r.kvDim / 2
			add(c.kvStoreF16Pipeline, bind(c.kvStoreF16Layout, src, cache, vStoreUni), uint32(words+63)/64, 1)
			return
		}
		add(c.kvStorePipeline, bind(c.kvStoreLayout, src, cache, vStoreUni), uint32(r.kvDim+63)/64, 1)
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

	r.xd = keepBuf(func() *wgpu.Buffer {
		b, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(hidden * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc})
		return b
	}())

	// A bound (all-zero) bias buffer for MoE layers without a selection bias
	// (Mixtral softmax routing): the route kernel binds it but hasBias=0 keeps it
	// out of the math. CreateBuffer zero-inits, so no upload needed.
	var moeZeroBias *wgpu.Buffer
	if m.moe != nil {
		moeZeroBias = storF(m.moe.nE)
	}

	for i := range m.layers {
		lw := &m.layers[i]
		if m.mla != nil {
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
			q, k, v := gemv(aq, as, lw.q), gemv(aq, as, lw.k), gemv(aq, as, lw.v)
			if lw.qBias != nil { // Qwen2 q/k/v bias, added before RoPE (matches CPU attention)
				biasAdd(q, lw.qBias, nH*hd)
				biasAdd(k, lw.kBias, r.kvDim)
				biasAdd(v, lw.vBias, r.kvDim)
			}
			if lw.qNorm != nil { // Qwen3/GLM per-head QK-norm, after bias, before RoPE (matches CPU)
				qkNorm(q, lw.qNorm, nH)
				qkNorm(k, lw.kNorm, nKV)
			}
			rope(q, lw.invFreq)
			ropeStore(k, lw.invFreq, lw.kCache, lw.kScale) // rotate K + append into cache
			vStore(v, lw.vCache, lw.vScale)                // append V into cache
			ctxv := storF(nH * hd)
			if m.kvI8 {
				// attnI8 reads packed int8 K/V + the per-(pos,head) scale side buffers.
				add(c.attnI8Pipeline, bind(c.attnI8Layout, q, lw.kCache, lw.vCache, lw.kScale, lw.vScale, ctxv, attnUni), uint32(nH), 1)
			} else {
				attnPl, attnLy := c.attnPipeline, c.attnLayout
				if m.kvF16 {
					attnPl, attnLy = c.attnF16Pipeline, c.attnF16Layout
				}
				add(attnPl, bind(attnLy, q, lw.kCache, lw.vCache, ctxv, attnUni), uint32(nH), 1)
			}
			cq, cs := quant(ctxv, nH*hd)
			gemvAdd(cq, cs, lw.o, r.xd) // o-proj + residual into xd
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
	fq, fs := rmsQuant(r.xd, m.finalNorm, hidden)
	logits := gemv(fq, fs, m.lmHead)
	r.lastLogits = logits
	stag, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(r.vocab * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	r.stag = keepBuf(stag)
	return r, nil
}

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
	for i := 0; i < n; i++ {
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
	for i := 0; i < n; i++ {
		runners[i].record(pass)
	}
	pass.End()
	pass.Release()
	for i := 0; i < n; i++ {
		enc.CopyBufferToBuffer(runners[i].lastLogits, 0, runners[i].stag, 0, uint64(runners[i].vocab*4))
	}
	cmd, err := enc.Finish(nil)
	if err != nil {
		return nil, err
	}
	defer cmd.Release()
	c.queue.Submit(cmd)
	sts := make([]wgpu.BufferMapAsyncStatus, n)
	for i := 0; i < n; i++ {
		i := i
		sts[i] = wgpu.BufferMapAsyncStatusUnknown
		if err := runners[i].stag.MapAsync(wgpu.MapModeRead, 0, uint64(runners[i].vocab*4), func(s wgpu.BufferMapAsyncStatus) { sts[i] = s }); err != nil {
			return nil, err
		}
	}
	c.device.Poll(true, nil) // one sync drains all K maps
	out := make([][]float32, n)
	for i := 0; i < n; i++ {
		if sts[i] != wgpu.BufferMapAsyncStatusSuccess {
			return nil, fmt.Errorf("gpu: runBatch row %d map failed: %v", i, sts[i])
		}
		row := make([]float32, runners[i].vocab)
		copy(row, wgpu.FromBytes[float32](runners[i].stag.GetMappedRange(0, uint(runners[i].vocab*4))))
		runners[i].stag.Unmap()
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
func (r *DecodeRunner) Release() { r.release() }
