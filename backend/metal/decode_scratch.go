//go:build darwin

package metal

import (
	"fmt"
	"maps"
	"slices"
)

// decodeShape is what decode's own scratch is sized from (docs/tasks/task-forward-architecture-2026-10.md §4.9c): the widths, in
// elements, its buffers take, each at the widest layer that uses it, since every layer reuses one buffer a role.
type decodeShape struct {
	H, V, I, nH        int
	nHhd, kvDim, guDim int // the query and K/V widths at the widest attention layer, and the widest gate|up half
	faPartial          int // the fused attention's partial buffer; 0 where it never runs
	q4k                bool
	qkWholeHD          int // the whole-vector qk_norm's width; 0 where the family has none
	attnScale, normEps float32
	act, addOne        uint32 // the gated MLP's activation and RMSNorm's (1 + w), as their uniforms hold them
	window             uint32 // the model's sliding window, 0 for full causal attention
	lnHasBias          uint32 // 1 where the family's LayerNorm carries a bias
	gumbelNB           int    // the device sampler's blocks over the vocabulary
	nL, pleP           int    // the layer count and Gemma 4's per-layer-embedding width; pleP 0 without the branch
	dnet               *dnetParams
	mamba              *mambaParams
	mla                *mlaParams
	attnSink           uint32        // 1 where the attention layers add learned sinks
	qGate              bool          // a layer runs gated attention: its [query ‖ gate] row and split gate are held
	geoms              []*attnGeom   // each distinct attention geometry once, its uniforms set into it
	layers             []decodeLayer // one a layer, in order
	families           []stepBuf     // what the family builders declared (scratchList): the MoE kinds and the paged fence
}

// scratchList is a family builder's decode scratch, declared where the builder knows its values and allocated with the rest of the
// decode list (decodeShape.families).
type scratchList []stepBuf

func (l *scratchList) add(sb stepBuf, dst *Buffer) {
	sb.set = func(b Buffer) { *dst = b }
	*l = append(*l, sb)
}

func (l *scratchList) f32(dst *Buffer, n int)    { l.add(stepBuf{kind: stepFloats, n: n}, dst) }
func (l *scratchList) bytes(dst *Buffer, n int)  { l.add(stepBuf{kind: stepBytes, n: n}, dst) }
func (l *scratchList) u32(dst *Buffer, v uint32) { l.add(stepBuf{kind: stepU32, n: 1, u: v}, dst) }
func (l *scratchList) f32Of(dst *Buffer, v float32) {
	l.add(stepBuf{kind: stepFloatsOf, n: 1, f: v}, dst)
}
func (l *scratchList) u32s(dst *Buffer, us []uint32) {
	l.add(stepBuf{kind: stepU32s, n: len(us), us: us}, dst)
}

// decodeLayer is what one layer's own uniforms hold.
type decodeLayer struct {
	attn   bool   // it runs attention: its sink flag and its window
	window uint32 // its sliding window, 0 for full causal
	ffnI   int    // its FFN width where that differs from the model's; 0 where it reads uI
}

// decodeRowsPlantForTest, when set, edits the declared row widths buildResident checks, so a test can plant a declaration the
// decode kernels do not address. nil in every production build.
var decodeRowsPlantForTest func(rows map[string]int)

// decodeRowsCheck holds the program's declared per-row widths (decoder.Model.MetalDecodeRowWidths) to what r's decode kernels
// address: each row Metal holds, by name, at its width, none declared that Metal holds no buffer for, and a mixer's or branch's rows
// declared exactly where r runs it. The list is sized from the same widths, so a declaration and kernels that differ would read or
// write past a buffer.
func (r *resident) decodeRowsCheck(declared map[string]int) error {
	s := r.decode
	want := map[string]int{"h": s.H, "x": s.H, "a": s.H, "logits": s.V}
	if _, ok := declared["b"]; ok { // the parallel block's second output, held in the FFN's output row
		want["b"] = s.H
	}
	if len(s.geoms) > 0 {
		want["q"], want["ctx"] = s.nHhd, s.nHhd
		for _, n := range []string{"k", "v"} {
			if _, ok := declared[n]; ok {
				want[n] = s.kvDim
			}
		}
	}
	gu := max(s.I, declared["gate"], declared["up"], declared["shared-gate"], declared["shared-up"])
	if gu != s.guDim {
		return fmt.Errorf("metal: the program's widest gate and up rows are %d, and the FFN kernels address %d", gu, s.guDim)
	}
	for _, n := range []string{"gate", "up", "shared-gate", "shared-up"} {
		if w, ok := declared[n]; ok {
			want[n] = w // held in the fused gate | up row, checked at its widest above
		}
	}
	if s.qGate {
		want["q-gate"], want["attn-gate"] = 2*s.nHhd, s.nHhd
	}
	if dp := s.dnet; dp != nil {
		for n, w := range map[string]int{"dn-mixed": dp.convDim, "dn-conv": dp.convDim, "dn-beta": dp.nv, "dn-decay": dp.nv, "dn-ba": 2 * dp.nv,
			"dn-q": dp.keyDim, "dn-k": dp.keyDim, "dn-core": dp.valueDim, "dn-z": dp.valueDim, "dn-gated": dp.valueDim} {
			want[n] = w
		}
	}
	if p := s.mamba; p != nil {
		want["mb-proj"], want["mb-conv"], want["mb-y"], want["mb-gated"] = p.projDim, p.convDim, p.dInner, p.dInner
	}
	if p := s.mla; p != nil {
		for n, w := range map[string]int{"mla-qa": p.qLoRA, "mla-q": p.nH * p.qkHead, "mla-kv": p.latDim, "mla-q-abs": p.nH * p.latDim,
			"mla-wsum": p.nH * p.rank, "mla-ctx": p.nH * p.vHead} {
			want[n] = w
		}
	}
	if P := s.pleP; P > 0 {
		want["ple-gate"], want["ple-in"] = P, s.nL*P
	}
	if mo := r.moe; mo != nil {
		prefix := "moe"
		if mo.isGptOss {
			prefix = "gptoss"
		}
		want[prefix+"-router"], want[prefix+"-experts"], want[prefix+"-weights"] = mo.nE, mo.k, mo.k
	}
	if g := r.g4moe; g != nil {
		for n, w := range map[string]int{"g4moe-router": g.nE, "g4moe-experts": g.topK, "g4moe-weights": g.topK, "g4moe-dense-out": s.H,
			"g4moe-moe-out": s.H, "g4moe-router-in": s.H} {
			want[n] = w
		}
	}
	for _, n := range slices.Sorted(maps.Keys(declared)) {
		w, ok := want[n]
		switch {
		case !ok:
			return fmt.Errorf("metal: the program declares a %s row of %d, which decode holds no buffer for", n, declared[n])
		case declared[n] != w:
			return fmt.Errorf("metal: the program declares the %s row at %d, and the decode kernels address %d", n, declared[n], w)
		}
	}
	for _, n := range slices.Sorted(maps.Keys(want)) {
		if _, ok := declared[n]; !ok && want[n] != 0 {
			return fmt.Errorf("metal: the program declares no %s row, which the decode kernels address at %d", n, want[n])
		}
	}
	return nil
}

// buffers is decode's scratch, each set into r as it is allocated: the list buildResident allocates it from and the price sums. A
// new buffer goes here, and so is priced.
func (s decodeShape) buffers(r *resident) []stepBuf {
	var out []stepBuf
	buf := func(kind stepBufKind, dst *Buffer, n int) {
		out = append(out, stepBuf{kind: kind, n: n, set: func(b Buffer) { *dst = b }})
	}
	u32 := func(dst *Buffer, v uint32) {
		out = append(out, stepBuf{kind: stepU32, n: 1, u: v, set: func(b Buffer) { *dst = b }})
	}
	f32Of := func(dst *Buffer, v float32) {
		out = append(out, stepBuf{kind: stepFloatsOf, n: 1, f: v, set: func(b Buffer) { *dst = b }})
	}
	aq := s.H
	if s.q4k {
		aq = 4 * s.H // the q4k lane's head reads the f32 final norm from aq's slot 0 (gemv_w8f32_head)
	}
	buf(stepFloats, &r.x, s.H)
	buf(stepBytes, &r.aq, aq)
	buf(stepFloats, &r.aSc, 1)
	buf(stepFloats, &r.qkv, s.nHhd+2*s.kvDim) // fused [q | k | v]
	buf(stepFloats, &r.gu, 2*s.guDim)         // fused [gate | up]
	buf(stepFloats, &r.ctx, s.nHhd)
	buf(stepBytes, &r.cq, s.nHhd)
	buf(stepFloats, &r.cSc, 1)
	if s.faPartial > 0 {
		buf(stepFloats, &r.attnFAPartial, s.faPartial)
		u32(&r.uAttnFAG, 0)
		u32(&r.uAttnFANSplit, 0)
	}
	buf(stepFloats, &r.oO, s.H)
	buf(stepBytes, &r.mq, s.H)
	buf(stepFloats, &r.mSc, 1)
	// The W4F16 lane's half-typed twins of aq, mq and cq, 2 bytes an element; allocated whichever lane decodes.
	buf(stepBytes, &r.axF16, 2*s.H)
	buf(stepBytes, &r.mxF16, 2*s.H)
	buf(stepBytes, &r.cxF16, 2*s.nHhd)
	if s.q4k {
		buf(stepFloats, &r.axF32, s.H)
		buf(stepFloats, &r.swF32, s.guDim)
	}
	buf(stepBytes, &r.dq, s.guDim)
	buf(stepFloats, &r.dSc, 1)
	buf(stepFloats, &r.dO, s.H)
	buf(stepFloats, &r.logits, s.V)
	// One (max logit, row) partial per 8-row threadgroup, rounded up: ForwardArgmax dispatches ceil(V/8) of them, each writing its
	// partial, so a floor would write past the buffer and leave the last tile out of the reduce.
	tiles := (s.V + 7) / 8
	buf(stepFloats, &r.part, 2*tiles)
	buf(stepFloats, &r.tok, 1)
	u32(&r.uP, uint32(tiles))
	u32(&r.uH, uint32(s.H))
	u32(&r.uI, uint32(s.I))
	u32(&r.uNH, uint32(s.nH))
	f32Of(&r.uScale, s.attnScale)
	f32Of(&r.uEps, s.normEps)
	u32(&r.uPos, 0)
	u32(&r.uNKeys, 1)
	u32(&r.uRopePos, 0)
	f32Of(&r.uQTempScale, 1) // identity; each call writes its own beside uPos
	if s.qkWholeHD > 0 {
		u32(&r.uQKWholeOne, 1)
		u32(&r.uQKWholeHD, uint32(s.qkWholeHD))
	}
	u32(&r.uZero, 0)
	u32(&r.uAct, s.act)
	u32(&r.uAddOne, s.addOne)
	u32(&r.uWindow, s.window)
	u32(&r.uLNHasBias, s.lnHasBias)
	// The device sampler (gumbel_sample.go): each block's best key and index, the draw, and its uniforms.
	buf(stepFloats, &r.gumbelBKey, s.gumbelNB)
	buf(stepFloats, &r.gumbelBIdx, s.gumbelNB)
	buf(stepFloats, &r.gumbelOut, 1)
	u32(&r.uGumbelV, uint32(s.V))
	u32(&r.uGumbelNB, uint32(s.gumbelNB))
	f32Of(&r.uGumbelInvT, 1)
	u32(&r.uGumbelK0, 0)
	u32(&r.uGumbelK1, 0)
	u32(&r.uGumbelD0, 0)
	u32(&r.uGumbelD1, 0)
	if P := s.pleP; P > 0 { // Gemma 4's per-layer-embedding branch: every layer's input, the gate, and its int8 row
		buf(stepFloats, &r.pleIn, s.nL*P)
		buf(stepFloats, &r.pleG, P)
		buf(stepBytes, &r.pleQ, P)
		buf(stepFloats, &r.pleSc, 1)
		u32(&r.uPleP, uint32(P))
	}
	if dp := s.dnet; dp != nil { // the Gated DeltaNet mixer's rows
		buf(stepFloats, &r.dnMixed, dp.convDim)
		buf(stepFloats, &r.dnConvOut, dp.convDim)
		buf(stepFloats, &r.dnBt, dp.nv)
		buf(stepFloats, &r.dnAt, dp.nv)
		buf(stepFloats, &r.dnHeadP, 2*dp.nv)
		buf(stepFloats, &r.dnQn, dp.keyDim)
		buf(stepFloats, &r.dnKn, dp.keyDim)
		buf(stepFloats, &r.dnCore, dp.valueDim)
		buf(stepFloats, &r.dnZOut, dp.valueDim)
		buf(stepFloats, &r.dnGated, dp.valueDim)
		buf(stepBytes, &r.dnGq, dp.valueDim)
		buf(stepFloats, &r.dnGSc, 1)
		u32(&r.uDnConvDim, uint32(dp.convDim))
		u32(&r.uDnK, uint32(dp.convK))
		u32(&r.uDnNv, uint32(dp.nv))
		u32(&r.uDnNk, uint32(dp.nk))
		u32(&r.uDnHk, uint32(dp.hk))
		u32(&r.uDnHv, uint32(dp.hv))
		u32(&r.uDnKeyDim, uint32(dp.keyDim))
		f32Of(&r.uDnQScale, dp.qScale)
		u32(&r.uDnRep, uint32(dp.rep))
		u32(&r.uDnVBase, uint32(2*dp.keyDim))
		u32(&r.uDnValueDim, uint32(dp.valueDim))
		f32Of(&r.uDnBetaScale, dp.betaScale)
		if dp.oNormEps != 0 { // else the model's epsilon: buildResident binds uEps itself
			f32Of(&r.uDnONormEps, dp.oNormEps)
		}
	}
	if s.qGate { // gated attention's [query ‖ gate] q_proj output and its split gate
		buf(stepFloats, &r.dnQg, 2*s.nHhd)
		buf(stepFloats, &r.dnAGate, s.nHhd)
	}
	if p := s.mamba; p != nil { // the Mamba-2 mixer: in_proj's output, the conv's, the scan's and the gated norm's, and its uniforms
		buf(stepFloats, &r.mbProj, p.projDim)
		buf(stepFloats, &r.mbConv, p.convDim)
		buf(stepFloats, &r.mbY, p.dInner)
		buf(stepFloats, &r.mbGated, p.dInner)
		buf(stepBytes, &r.mbGq, p.dInner)
		buf(stepFloats, &r.mbGSc, 1)
		for _, u := range []struct {
			dst *Buffer
			v   int
		}{{&r.uMbConvDim, p.convDim}, {&r.uMbK, p.dConv}, {&r.uMbDInner, p.dInner}, {&r.uMbNH, p.nHeads}, {&r.uMbHD, p.headDim},
			{&r.uMbDS, p.dState}, {&r.uMbRep, p.repeat}, {&r.uMbGSize, p.gSize}, {&r.uMbDtBase, p.dInner + p.convDim},
			{&r.uMbGroupSize, p.dInner / p.normGroups}, {&r.uMbZero, 0}} {
			u32(u.dst, uint32(u.v))
		}
		f32Of(&r.uMbEps, s.normEps)
		u32(&r.uMbNormGroups, uint32(p.normGroups))
	}
	if p := s.mla; p != nil { // Multi-head Latent Attention: q (through its LoRA), the latent, the absorbed q, and the lifted context
		S, b := r.mlaS, uint32(0)
		if p.interleave {
			b = 1
		}
		u32s := func(dst *Buffer, vs ...int) {
			us := make([]uint32, len(vs))
			for i, v := range vs {
				us[i] = uint32(v)
			}
			out = append(out, stepBuf{kind: stepU32s, n: len(us), us: us, set: func(x Buffer) { *dst = x }})
		}
		buf(stepFloats, &S.qa, max(p.qLoRA, 1))
		buf(stepBytes, &S.qaQ, max(p.qLoRA, 4))
		buf(stepFloats, &S.qaSc, 1)
		buf(stepFloats, &S.q, p.nH*p.qkHead)
		buf(stepFloats, &S.kv, p.latDim)
		buf(stepFloats, &S.qAbs, p.nH*p.latDim)
		buf(stepFloats, &S.wsum, p.nH*p.rank)
		buf(stepFloats, &S.ctx, p.nH*p.vHead)
		buf(stepBytes, &S.ctxQ, p.nH*p.vHead)
		buf(stepFloats, &S.ctxSc, 1)
		u32s(&S.uStore, p.rank, p.qkRope, int(b)) // packed shapes (kernels/mla_kernels_msl.go)
		u32s(&S.uAbsorb, p.nH, p.rank, p.qkNope, p.qkHead, p.latDim)
		u32s(&S.uLift, p.nH, p.vHead, p.rank, p.rank, p.vHead)
		u32s(&S.uQRope, p.nH, p.qkHead, p.qkNope, p.qkRope, p.rank, p.latDim, int(b))
		u32s(&S.uAttn, p.nH, p.latDim, p.rank)
		u32(&S.uQLoRA, uint32(p.qLoRA))
		u32(&S.uOK, uint32(p.nH*p.vHead))
		u32(&S.uZero, 0)
		f32Of(&S.uRopeScale, p.ropeScale)
		f32Of(&S.uScale, p.attnScale)
	}
	for _, g := range s.geoms {
		u32(&g.uHd, uint32(g.hd))
		u32(&g.uKvDim, uint32(g.kvDim))
		u32(&g.uNKV, uint32(g.nKV))
		u32(&g.uNHhd, uint32(s.nH*g.hd))
		u32(&g.uHalf, uint32(g.half))
		u32(&g.uQtotal, uint32(s.nH*g.half))
		u32(&g.uKtotal, uint32(g.nKV*g.half))
	}
	for l := range r.layers { // the q4k lane's segments: each one's row count
		L := &r.layers[l]
		for _, segs := range [][]q4kSeg{L.qkvSegs, L.oSegs, L.guSegs, L.dSegs} {
			for i := range segs {
				u32(&segs[i].uN, uint32(segs[i].rows))
			}
		}
	}
	for l, ly := range s.layers {
		L := &r.layers[l]
		if ly.attn {
			u32(&L.uHasSink, s.attnSink)
			u32(&L.uWindow, ly.window)
		}
		if ly.ffnI != 0 {
			u32(&L.uFFNI, uint32(ly.ffnI))
		}
	}
	out = append(out, s.families...)
	if r.batchCap > 0 { // ensureBatchCap's rows, allocated after the list at the build's capacity and grown by ForwardBatch
		out = append(out, r.batchRows(r.batchCap, 0)...)
	}
	return append(out, r.chainBuffers(r.chainTok[0] != (Buffer{}), r.chainLogits[0] != (Buffer{}))...)
}

// batchRows is ensureBatchCap's buffers at capacity c: the batch's hidden and logits rows, and the position uniforms of rows held
// through c. The decode list counts them at the resident's capacity.
func (r *resident) batchRows(c, held int) []stepBuf {
	var l scratchList
	l.f32(&r.batchX, c*r.H)
	l.f32(&r.batchLogits, c*r.V)
	for i := held; i < c; i++ {
		l.u32(&r.batchUPos[i], 0)
		l.u32(&r.batchUNKeys[i], 1)
		l.f32Of(&r.batchUQTempScale[i], 1)
	}
	return l
}

// chainBuffers is what the greedy chain allocates when it first opens (chainOpen): each buffer's token and position uniforms, and,
// for a sampled chain, the draw's temperature, keys and logits. The decode list counts each part once it is held.
func (r *resident) chainBuffers(greedy, sampled bool) []stepBuf {
	var l scratchList
	if greedy {
		for k := range r.chainSets {
			u := &r.chainSets[k]
			l.u32(&r.chainTok[k], 0)
			l.u32(&u.uPos, 0)
			l.u32(&u.uNKeys, 1)
			l.u32(&u.uRopePos, 0)
			l.f32Of(&u.uQTempScale, 1)
			l.u32(&u.uFANSplit, 0)
		}
	}
	if sampled {
		l.f32Of(&r.chainInvT, 1)
		l.u32(&r.chainK0, 0)
		l.u32(&r.chainK1, 0)
		for k := range r.chainLogits {
			l.f32(&r.chainLogits[k], r.V)
			l.u32(&r.chainD0[k], 0)
			l.u32(&r.chainD1[k], 0)
		}
	}
	return l
}

// listBytes is what the list asks the device for on r, the resident it was built for.
func (s decodeShape) listBytes(r *resident) int64 {
	var n int64
	for _, sb := range s.buffers(r) {
		n += int64(sb.bytes())
	}
	return n
}
