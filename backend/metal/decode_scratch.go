//go:build darwin

package metal

// decodeShape is what decode's own scratch is sized from (docs/tasks/task-forward-architecture-2026-10.md §4.9c): the widths, in
// elements, its buffers take, each at the widest layer that uses it, since every layer reuses one buffer a role.
type decodeShape struct {
	H, V, I, nH        int
	nHhd, kvDim, guDim int // the query and K/V widths at the widest attention layer, and the widest gate|up half
	faPartial          int // the fused attention's partial buffer; 0 where it never runs
	q4k                bool
	qkWholeHD          int // the whole-vector qk_norm's width; 0 where the family has none
	attnScale, normEps float32
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
	return out
}

// listBytes is what the list asks the device for.
func (s decodeShape) listBytes() int64 {
	var n int64
	for _, sb := range s.buffers(&resident{}) {
		n += int64(sb.bytes())
	}
	return n
}
