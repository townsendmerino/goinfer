package decoder

import (
	"strconv"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// metalScratchBatchRows is the batch rows a Metal build allocates for ForwardBatch before any call grows them (ensureBatchCap(16)).
const metalScratchBatchRows = 16

// metalAttnFAPartialFloats is the fused attention's partial buffer a Metal build holds for a layer of head width hd with nH query and
// nKV K/V heads: nH × (hd + 2) floats for each of 32 splits, where attention_fa runs (head width 128, or 64 at a group of 7, and a
// group of at most 8), else none. It is backend/metal's attnFAPartialFloats, which TestMemGate_metal holds this price to.
func metalAttnFAPartialFloats(hd, nH, nKV int) int {
	if nKV <= 0 || nH%nKV != 0 {
		return 0
	}
	g := nH / nKV
	if !(hd == 128 || hd == 64 && g == 7) || g > 8 {
		return 0
	}
	return nH * (hd + 2) * 32
}

// MetalDecodeScratchBytes is what a Metal resident build of m leases on the scratch line for decode, the most it holds before a
// ForwardBatch grows its rows: the decode list (backend/metal's decodeShape.buffers), the 16 batch rows (batchRows) and the greedy
// chain's buffers with a sampled chain's (chainBuffers). TestMemGate_metal holds it to the build's own list on every row. 0 for a
// model with no hidden or vocabulary width.
func (m *Model) MetalDecodeScratchBytes() int64 {
	H, nL, nH, _, _, I, V := m.Dims()
	if H <= 0 || V <= 0 {
		return 0
	}
	var f32, bytes, u32s int64 // f32 elements, bytes, and 4-byte uniforms
	w := m.Weights()
	type geomKey struct {
		hd, nKV, half int
		kEqV          bool
	}
	geoms := map[geomKey]bool{}
	nHhd, kvDim, fa, attn, layerFFN := 0, 0, 0, 0, 0
	maxFFNI := I
	nE, k, inter, sharedInter, _, _, _, _, _, _, moeOK := m.MoEResidentParams()
	g4 := m.HasGemma4MoEResident()
	moeOK = moeOK && !g4
	q4k := m.Quant() == "q4k"
	for l := range nL {
		mixer, hasFFN := m.ResidentLayerOps(l)
		if mixer == MixerAttention {
			hd, kv := m.HeadDimAtResident(l), m.KVHeadsAtResident(l)
			kEqV := m.VFromKResident(l) && m.KVSrcAtResident(l) == l
			geoms[geomKey{hd, kv, len(m.RopeInvFreqLayerResident(l)), kEqV}] = true
			nHhd, kvDim = max(nHhd, nH*hd), max(kvDim, kv*hd)
			fa = max(fa, metalAttnFAPartialFloats(hd, nH, kv))
			attn++
		}
		if l >= len(w.Layers) {
			continue
		}
		lw := &w.Layers[l]
		_, isG4 := m.Gemma4MoEResidentLayer(l)
		if hasFFN && !isG4 && !(moeOK && len(lw.Experts) > 0) && !m.NonGatedMLPResident() {
			if fi := lw.GateProj.Rows(); fi != 0 {
				maxFFNI = max(maxFFNI, fi)
				if fi != I {
					layerFFN++
				}
			}
		}
		if q4k { // the q4k lane's segments: one row-count uniform a run of parts of one kind in each fused group
			for _, group := range [][]*linalg.WeightMat{{&lw.QProj, &lw.KProj, &lw.VProj}, {&lw.OProj}, {&lw.DownProj}, {&lw.GateProj, &lw.UpProj}} {
				u32s += int64(m.metalQ4KRuns(group))
			}
		}
	}
	if nHhd == 0 { // no attention layer: the attention rows are sized at 4, never dispatched
		nHhd, kvDim = 4, 4
	}
	gu := maxFFNI
	if moeOK {
		gu = max(gu, inter, sharedInter)
	}
	var g4b Gemma4MoEResidentBundle
	if g4 {
		for l := range nL {
			if b, ok := m.Gemma4MoEResidentLayer(l); ok {
				g4b = b
				break
			}
		}
		gu = max(gu, g4b.DenseInter, g4b.MoeInter)
	}
	// The core: x, aSc; the fused q | k | v; the fused gate | up; ctx, cSc; oO, mSc; dSc, dO; the logits, the argmax partials, tok;
	// the sampler's block keys and indices and its draw; and the fused attention's partial.
	f32 += int64(H + 1 + nHhd + 2*kvDim + 2*gu + nHhd + 1 + H + 1 + 1 + H + V + 2*((V+7)/8) + 1 + 2*((V+1023)/1024) + 1 + fa)
	aq := H
	if q4k {
		aq = 4 * H
		f32 += int64(H + gu) // the lane's f32 activation rows
	}
	bytes += int64(aq + nHhd + H + 2*H + 2*H + 2*nHhd + gu) // aq, cq, mq; the f16 twins of aq, mq and cq; dq
	u32s += 22                                              // the core's eleven, uAct, uAddOne, uWindow, uLNHasBias, and the sampler's seven
	if fa > 0 {
		u32s += 2
	}
	if m.QKNormWholeResident() {
		u32s += 2
	}
	u32s += int64(7*len(geoms) + 2*attn + layerFFN) // each geometry's seven; each attention layer's sink flag and window; a distinct FFN width's
	if P := m.Gemma4PLEDimResident(); P > 0 {
		f32 += int64(nL*P + P + 1)
		bytes += int64(P)
		u32s++
	}
	if _, hk, hv, nk, nv, _, ok := m.Qwen35ResidentParams(); ok { // the Gated DeltaNet mixer and gated attention's [query ‖ gate]
		keyDim, valueDim := nk*hk, nv*hv
		f32 += int64(2*(2*keyDim+valueDim) + 2*nv + 2*nv + 2*keyDim + 3*valueDim + 1 + 2*nHhd + nHhd)
		bytes += int64(valueDim)
		u32s += 12
		if eps, oEps := m.metalDeltaNetEps(); oEps != 0 && oEps != eps {
			u32s++
		}
	}
	if nh, hd, ds, ng, _, normG, _, ok := m.NemotronResidentParams(); ok && normG > 0 { // the Mamba-2 mixer
		dInner, gSize := nh*hd, ng*ds
		f32 += int64(2*dInner + 2*gSize + nh + dInner + 2*gSize + 2*dInner + 1)
		bytes += int64(dInner)
		u32s += 13
	}
	if qLoRA, rank, nope, rope, vHead, _, _, _, ok := m.MLAResidentParams(); ok { // Multi-head Latent Attention
		qkHead, latDim := nope+rope, rank+rope
		f32 += int64(max(qLoRA, 1) + 1 + nH*qkHead + latDim + nH*latDim + nH*rank + nH*vHead + 1)
		bytes += int64(max(qLoRA, 4) + nH*vHead + 4*(3+5+5+7+3)) // qaQ, ctxQ, and the five packed shapes
		u32s += 5
	}
	if moeOK { // the generic MoE's router, shared expert and k slots
		_, _, gptOss := m.GptOssActResident()
		f32 += int64(nE + k + 1 + H)
		bytes += int64(4 * k)
		u32s += int64(10 + k)
		if gptOss {
			u32s += 3
		}
		if !gptOss && (2*inter)%8 == 0 && H%8 == 0 {
			f32 += int64(k*2*inter + k + k*H)
			bytes += int64(k * inter)
		}
		if m.metalMoEPaged(nE) {
			bytes += int64(4 * k)
		}
	}
	if g4 { // Gemma 4's parallel dense ‖ MoE block
		f32 += int64(2*g4b.NE + 2*g4b.TopK + 4*H)
		bytes += int64(8 * g4b.TopK)
		u32s += int64(10 + g4b.TopK)
		if m.metalMoEPaged(g4b.NE) {
			bytes += int64(4 * g4b.TopK)
		}
	}
	f32 += int64(metalScratchBatchRows * (H + V)) // batchX, batchLogits
	u32s += 3 * metalScratchBatchRows             // each batch row's position uniforms
	u32s += 12 + 7                                // the greedy chain's two tokens and position sets; a sampled chain's temperature, keys and draws
	f32 += int64(2 * V)                           // a sampled chain's two logits rows
	return 4*f32 + bytes + 4*u32s
}

// metalQ4KRuns is how many segments Metal's q4k lane uploads a fused group as: one a run of consecutive parts of one kind (the file's
// Q4_K, Q5_K or Q6_K blocks, else int8), as backend/metal's q4kLaneProjOn splits it.
func (m *Model) metalQ4KRuns(group []*linalg.WeightMat) int {
	kind := func(wm *linalg.WeightMat) string {
		if wm.Kind() == "q4k" {
			return "q4k"
		}
		if k := m.KQuantNative("metal", wm); k != "" {
			return k
		}
		return "int8"
	}
	runs := 0
	for i := range group {
		if i == 0 || kind(group[i]) != kind(group[i-1]) {
			runs++
		}
	}
	return runs
}

// metalDeltaNetEps is the model's Gated DeltaNet operation's norm epsilons (program.DeltaNetOpParams): Metal holds a uniform of its own
// for the output norm's where that differs from the model's.
func (m *Model) metalDeltaNetEps() (eps, oEps float64) {
	for _, ops := range m.Program().Layers {
		for _, o := range ops {
			if o.Kind == program.OpDeltaNet {
				return o.Delta.Eps, o.Delta.ONormEps
			}
		}
	}
	return 0, 0
}

// metalMoEPaged reports whether a Metal build of m pages its routed experts, from the request backend/metal's metalMoESlotsRequest
// reads: an expert-slot request, else expert caching (whose slot count the build sizes from memory, priced here as paging), else the
// GOINFER_METAL_MOE_SLOTS knob; paged when the slots are under the expert count.
func (m *Model) metalMoEPaged(nE int) bool {
	n := m.MoECacheSlotsRequest()
	if n <= 0 && m.MoECacheExperts() {
		return true
	}
	if n <= 0 {
		v, _ := m.Knob("GOINFER_METAL_MOE_SLOTS")
		n, _ = strconv.Atoi(v)
	}
	return n > 0 && n < nE
}
