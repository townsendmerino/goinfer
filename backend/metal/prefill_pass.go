//go:build darwin

package metal

import (
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// prefillPass is one batched prefill pass's state (resident.prefillLastImgSrc): its geometry, the per-call f16 scratch and
// uniforms newPrefillPass allocates, and scratch, every buffer the pass allocated, which release frees. The pass owns
// every buffer here except uH, uHd and uEpsX when it is r.uEps, which are the resident's.
type prefillPass struct {
	d *Device

	M, Mpad, H, I, V          int // the rows, padded to 8, and the model's hidden, FFN and vocabulary widths
	nHhd, kvDim               int
	qkvDim                    int  // prefillWidths' q|k|v width
	fusedSw                   bool // swigluFused: the gate|up GEMM writes the activation, and there is no guF
	attnTile                  int  // exactAttnTile: > 0 when the exact-attention layers take the tiled kernel
	xF, normF, qkvF, ctxF     Buffer
	guF, dqF                  Buffer
	posB, pos3B               Buffer
	pleInB, pleGF, uP, uLP    Buffer
	moeDst                    Buffer
	moeLogits, moeIdx, moeWgt Buffer
	expertIn, expertDown      Buffer
	rowIdxBuf, rowWgtBuf      Buffer
	kScrI8, vScrI8, uEndI8    Buffer
	idTab                     Buffer
	qgF, kvF, gateF           Buffer
	u2NHhd, u2KvDim, uGateN   Buffer
	imgB, imgN                Buffer
	dn                        *prefillDelta
	mb                        *prefillMamba
	uResidInv, uEpsX          Buffer
	uM, uH, uI, u2I, uQkv     Buffer
	uHd                       Buffer
	uKOff, uStartPos          Buffer
	uTotalQ                   Buffer
	uBase0, m0, m1, m2        Buffer
	dummyBias, uMReal, uTile  Buffer

	scratch []Buffer
}

// release frees every buffer the pass allocated, in allocation-list order. It runs after the pass's last command buffer
// has completed.
func (p *prefillPass) release() {
	for _, b := range p.scratch {
		p.d.ReleaseBuf(b)
	}
}

// newPrefillPass allocates one batched pass's scratch and uniforms, in the order the pass has always allocated them, and
// converts src's rows into the f16 residual. The caller releases them (prefillPass.release) once the pass's last command
// buffer has completed.
func (r *resident) newPrefillPass(src rowSource, startPos int, mrope [][3]int, deep *prefillDeep, blocks [][2]int) *prefillPass {
	d := r.d
	M := src.n
	Mpad := (M + 7) / 8 * 8
	H, I, V := r.H, r.I, r.V
	// Prefill runs only for uniform families (prefillOK declines Gemma's per-layer geometry), so
	// every attention layer shares one geometry.
	g0 := r.prefillGeom()
	nHhd := r.nH * g0.hd
	kvDim := g0.kvDim
	qkvDim, qDim, maxI := r.prefillWidths()

	// f16 activation scratch (per call, sized to the padded prompt).
	// xF and the five prefillScratchU16 buffers below rely on Metal's zero-fill of a new buffer for their pad rows
	// (aikit gpu.NewBufferLen* contract: Metal only; CUDA's is uninitialized) — see prefillScratchU16.
	// xF too (R-20): the embeddings are converted straight into the buffer's M rows, its pad rows left as Metal zeroed
	// them, instead of into a zeroed Go slice of Mpad rows that was then copied in whole.
	xF := prefillScratchU16(d, Mpad*H)
	residInv := 1 / r.prefillResidScale(deep)
	// The input rows, read rowChunk at a time from the source: each row's first H values into xF as f16, and a Gemma 4
	// E-model's [L·P] per-layer tail straight into the device buffer the PLE gate reads (S9 step 2), so the host holds a chunk.
	var pleInB Buffer
	if r.pleP > 0 {
		LP := r.nL * r.pleP
		pleInB = d.NewBufferLenOn(mem.LineScratch, M*LP)
		prefillScratchSeen.Add(int64(4 * M * LP))
	}
	for lo := 0; lo < M; lo += rowChunk {
		hi := min(lo+rowChunk, M)
		rows := src.rows(lo, hi)
		parallelEmbedsF32ToF16Scaled(xF.U16s()[lo*H:hi*H], rows, H, residInv)
		if r.pleP > 0 {
			LP, pv := r.nL*r.pleP, pleInB.Floats()
			for i, row := range rows {
				copy(pv[(lo+i)*LP:(lo+i+1)*LP], row[H:H+LP])
			}
		}
	}
	normF := prefillScratchU16(d, Mpad*H)
	qkvF := prefillScratchU16(d, Mpad*qkvDim)
	ctxF := prefillScratchU16(d, Mpad*qDim)
	fusedSw := r.swigluFused(Mpad)
	var guF Buffer // the gate|up rows between the GEMM and swiglu_f16; none when the pass fuses them (swigluFused)
	if !fusedSw {
		guF = prefillScratchU16(d, Mpad*2*maxI)
	}
	dqF := prefillScratchU16(d, Mpad*maxI)
	posv := make([]uint32, Mpad)
	for m := range M {
		posv[m] = uint32(startPos + m)
	}
	posB := NewBufferUint32sOn(mem.LineScratch, d, posv)
	prefillScratchSeen.Add(int64(4 * Mpad))
	var pos3B Buffer // S16: the rows' m-RoPE triples, row-major [Mpad*3]; pad rows repeat the last real row's
	if mrope != nil {
		p3 := make([]uint32, Mpad*3)
		for m := range Mpad {
			t := mrope[startPos+min(m, M-1)]
			p3[m*3], p3[m*3+1], p3[m*3+2] = uint32(t[0]), uint32(t[1]), uint32(t[2])
		}
		pos3B = NewBufferUint32sOn(mem.LineScratch, d, p3)
	}

	// uniforms
	uM := NewBufferU32On(mem.LineScratch, d, uint32(Mpad))
	uH := r.uH
	uI := NewBufferU32On(mem.LineScratch, d, uint32(I))
	u2I := NewBufferU32On(mem.LineScratch, d, uint32(2*I))
	uQkv := NewBufferU32On(mem.LineScratch, d, uint32(qkvDim))
	uQDim := NewBufferU32On(mem.LineScratch, d, uint32(qDim))
	uHd := g0.uHd
	uStride := NewBufferU32On(mem.LineScratch, d, uint32(qkvDim))
	uKOff := NewBufferU32On(mem.LineScratch, d, uint32(nHhd))
	uVOff := NewBufferU32On(mem.LineScratch, d, uint32(nHhd+kvDim))
	uStartPos := NewBufferU32On(mem.LineScratch, d, uint32(startPos))
	uTotalQ := NewBufferU32On(mem.LineScratch, d, uint32(nHhd))
	uTotalK := NewBufferU32On(mem.LineScratch, d, uint32(kvDim))
	uBase0 := NewBufferU32On(mem.LineScratch, d, 0)
	uBaseK := NewBufferU32On(mem.LineScratch, d, uint32(nHhd))
	m0, m1, m2 := NewBufferU32On(mem.LineScratch, d, 0), NewBufferU32On(mem.LineScratch, d, 1), NewBufferU32On(mem.LineScratch, d, 2)
	dummyBias := NewBufferFloatsOn(mem.LineScratch, d, make([]float32, 1))
	// L2-Metal: attention_prefill_fused's own row-count uniform — REAL M (unpadded), unlike uM
	// above which holds Mpad for the GEMM grid. attention_prefill (the exact kernel) needs no
	// such uniform because its Go-side dispatch grid is already sized off the real M.
	uMReal := NewBufferU32On(mem.LineScratch, d, uint32(M))
	// An int8 cache: each layer's rows [0, startPos+M) are dequantized into these f16 buffers before its attention (D9 (b)).
	// The attention kernels read that scratch through idTab, an identity table, as they read the cache through its own.
	var kScrI8, vScrI8, uEndI8, idTab Buffer
	if r.kvI8 {
		kScrI8, vScrI8 = prefillScratchU16(d, r.kvI8ScratchRows()*kvDim), prefillScratchU16(d, r.kvI8ScratchRows()*kvDim)
		uEndI8 = NewBufferU32On(mem.LineScratch, d, uint32(startPos+M))
		idTab = kvIdentityTab(d, r.kvI8ScratchRows())
	}
	attnTile := exactAttnTile(startPos + M) // > 0: the exact-attention layers take attention_prefill_tiled
	uTile := NewBufferU32On(mem.LineScratch, d, uint32(max(attnTile, 1)))
	// G8: the MoE row loop's F32 accumulate scratch (see the per-layer loop's own comment).
	// Allocated unconditionally (cheap — H floats) rather than gated on whether any layer is
	// MoE, avoiding a nil-buffer special case in the loop below.
	moeDst := NewBufferFloatsOn(mem.LineScratch, d, make([]float32, H))

	// MoE expert-major prefill scratch buffers (allocated only when model has MoE layers).
	var (
		moeLogits  Buffer
		moeIdx     Buffer
		moeWgt     Buffer
		expertIn   Buffer
		expertDown Buffer
		rowIdxBuf  Buffer
		rowWgtBuf  Buffer
	)
	if r.moe != nil {
		// R-20: zero-filled by Metal, not copied from zeroed Go slices; the router GEMM, the route kernels and the host
		// fill them before anything reads them (see prefillScratch).
		moeLogits = prefillScratch[float32](d, M*r.moe.nE)
		moeIdx = prefillScratch[uint32](d, M*r.moe.k)
		moeWgt = prefillScratch[float32](d, M*r.moe.k)
		// expertIn/expertDown rely on the same Metal zero-fill for their pad rows as the scratch above (see prefillScratchU16).
		expertIn = prefillScratchU16(d, Mpad*H)
		expertDown = prefillScratchU16(d, Mpad*H)
		rowIdxBuf = prefillScratch[uint32](d, Mpad*r.moe.k)
		rowWgtBuf = prefillScratch[float32](d, Mpad*r.moe.k)
		if prefillScratchPoison { // a test's proof that nothing reads these before the pass writes them
			for _, b := range []Buffer{moeLogits, moeIdx, moeWgt, rowIdxBuf, rowWgtBuf} {
				for i, u := 0, b.U32s(); i < len(u); i++ {
					u[i] = 0xFFFFFFFF
				}
			}
		}
	}

	// Every buffer above is per-call scratch or uniform allocated onto the device ledger, which ReleaseAll frees
	// only at Close, so each is released at the end of the call (the pass's last e.End() commits and waits, so the
	// GPU is finished with them by then). Unreleased they ratchet by a prompt's worth per call (guF alone is Mpad*2I*2)
	// until the mustBuf OOM panic kills serve, which is recovered only on the BuildResident path, not here. r.uH,
	// r.uKvDim and r.uHd are resident-owned and reused, so deliberately not in this list: releasing them would
	// corrupt the decode path.
	scratch := []Buffer{
		xF, normF, qkvF, ctxF, guF, dqF, posB, moeDst,
		uM, uI, u2I, uQkv, uQDim, uStride, uKOff, uVOff, uStartPos,
		uTotalQ, uTotalK, uBase0, uBaseK, m0, m1, m2, dummyBias, uMReal, uTile,
	}
	if pos3B != (Buffer{}) {
		scratch = append(scratch, pos3B)
	}
	if r.moe != nil {
		scratch = append(scratch, moeLogits, moeIdx, moeWgt, expertIn, expertDown, rowIdxBuf, rowWgtBuf)
	}
	if r.kvI8 {
		scratch = append(scratch, kScrI8, vScrI8, uEndI8, idTab)
	}
	// S17: the scaled residual's 1/s, for residual_f16_scaled, and the eps of the norms that read the residual, eps/s², so
	// that x/s / sqrt(ms/s² + eps/s²) is x / sqrt(ms + eps) exactly (s is a power of two).
	var uResidInv Buffer
	uEpsX := r.uEps
	if residInv != 1 {
		uResidInv = NewBufferFloatsOn(mem.LineScratch, d, []float32{residInv})
		uEpsX = NewBufferFloatsOn(mem.LineScratch, d, []float32{r.uEps.Floats()[0] * residInv * residInv})
		scratch = append(scratch, uResidInv, uEpsX)
	}
	// S9 step 2: a Gemma 4 E-model's per-layer inputs, each row's [L·P] tail of its [h ‖ L·P] embedding, and the PLE
	// gate's scratch.
	var pleGF, uP, uLP Buffer
	if r.pleP > 0 {
		P, LP := r.pleP, r.nL*r.pleP
		if emodelBatchDefect == 3 { // G-S9c (3): each row's PLE inputs taken from the next row
			pv := pleInB.Floats()
			for m := 0; m+1 < M; m++ {
				copy(pv[m*LP:(m+1)*LP], pv[(m+1)*LP:(m+2)*LP])
			}
		}
		pleGF, uP, uLP = prefillScratchU16(d, Mpad*P), NewBufferU32On(mem.LineScratch, d, uint32(P)), NewBufferU32On(mem.LineScratch, d, uint32(LP))
		scratch = append(scratch, pleInB, pleGF, uP, uLP)
	}
	// D-B01: a Gated-DeltaNet hybrid's mixer scratch, and its gated-attention layers' [query ‖ gate] projection, K‖V
	// projection and split gate.
	var dn *prefillDelta
	var qgF, kvF, gateF, u2NHhd, u2KvDim, uGateN Buffer
	if r.dnet != nil {
		dn = r.newPrefillDelta(M, Mpad)
		qgF = prefillScratchU16(d, Mpad*2*nHhd)
		kvF = prefillScratchU16(d, Mpad*2*kvDim)
		gateF = prefillScratchU16(d, M*nHhd)
		u2NHhd, u2KvDim, uGateN = NewBufferU32On(mem.LineScratch, d, uint32(2*nHhd)), NewBufferU32On(mem.LineScratch, d, uint32(2*kvDim)), NewBufferU32On(mem.LineScratch, d, uint32(M*nHhd))
		scratch = append(append(scratch, dn.bufs...), qgF, kvF, gateF, u2NHhd, u2KvDim, uGateN)
	}
	var mb *prefillMamba // KS-D8: a Mamba-2 hybrid's mixer scratch
	if r.mamba != nil {
		mb = r.newPrefillMamba(M, Mpad)
		scratch = append(scratch, mb.bufs...)
	}
	var imgB, imgN Buffer // the bidirectional image blocks and their count, when there are any (attention_prefill_img)
	if len(blocks) > 0 {
		flat := make([]uint32, 0, 2*len(blocks))
		for _, b := range blocks {
			flat = append(flat, uint32(b[0]), uint32(b[1]))
		}
		imgB, imgN = NewBufferUint32sOn(mem.LineScratch, d, flat), NewBufferU32On(mem.LineScratch, d, uint32(len(blocks)))
		scratch = append(scratch, imgB, imgN)
	}
	return &prefillPass{d: d, M: M, Mpad: Mpad, H: H, I: I, V: V, nHhd: nHhd, kvDim: kvDim, qkvDim: qkvDim, xF: xF, pleInB: pleInB, normF: normF, qkvF: qkvF, ctxF: ctxF, fusedSw: fusedSw, guF: guF, dqF: dqF, posB: posB, pos3B: pos3B, uM: uM, uH: uH, uI: uI, u2I: u2I, uQkv: uQkv, uHd: uHd, uKOff: uKOff, uStartPos: uStartPos, uTotalQ: uTotalQ, uBase0: uBase0, m0: m0, m1: m1, m2: m2, dummyBias: dummyBias, uMReal: uMReal, kScrI8: kScrI8, vScrI8: vScrI8, uEndI8: uEndI8, idTab: idTab, attnTile: attnTile, uTile: uTile, moeDst: moeDst, moeLogits: moeLogits, moeIdx: moeIdx, moeWgt: moeWgt, expertIn: expertIn, expertDown: expertDown, rowIdxBuf: rowIdxBuf, rowWgtBuf: rowWgtBuf, scratch: scratch, uResidInv: uResidInv, uEpsX: uEpsX, pleGF: pleGF, uP: uP, uLP: uLP, dn: dn, qgF: qgF, kvF: kvF, gateF: gateF, u2NHhd: u2NHhd, u2KvDim: u2KvDim, uGateN: uGateN, mb: mb, imgB: imgB, imgN: imgN}
}
