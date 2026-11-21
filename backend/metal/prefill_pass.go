//go:build darwin

package metal

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
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

	r        *resident
	pf       *prefillState
	e        *Encoder // the open command buffer: moeBlock commits it and begins the next
	normPipe Pipeline // the GEMM-input norm: RMSNorm, or a bias-free LayerNorm
	startPos int
	mrope    [][3]int
	deep     *prefillDeep
	deepB    []Buffer // deep's sets on the device
	zeroKV   Buffer
	fusedOn  bool // prefillFusedOn
	uConsts  map[int]Buffer
	luCache  map[luKey]*layerU

	scratch []Buffer
}

// release frees every buffer the pass allocated, in allocation-list order. It runs after the pass's last command buffer
// has completed.
func (ps *prefillPass) release() {
	for _, b := range ps.scratch {
		ps.d.ReleaseBuf(b)
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
	// xF too: the embeddings are converted straight into the buffer's M rows, its pad rows left as Metal zeroed
	// them, instead of into a zeroed Go slice of Mpad rows that was then copied in whole.
	xF := prefillScratchU16(d, Mpad*H)
	residInv := 1 / r.prefillResidScale(deep)
	// The input rows, read rowChunk at a time from the source: each row's first H values into xF as f16, and a Gemma 4
	// E-model's [L·P] per-layer tail straight into the device buffer the PLE gate reads, so the host holds a chunk.
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
	var pos3B Buffer // the rows' m-RoPE triples, row-major [Mpad*3]; pad rows repeat the last real row's
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
	// An int8 cache: each layer's rows [0, startPos+M) are dequantized into these f16 buffers before its attention.
	// The attention kernels read that scratch through idTab, an identity table, as they read the cache through its own.
	var kScrI8, vScrI8, uEndI8, idTab Buffer
	if r.kvI8 {
		kScrI8, vScrI8 = prefillScratchU16(d, r.kvI8ScratchRows()*kvDim), prefillScratchU16(d, r.kvI8ScratchRows()*kvDim)
		uEndI8 = NewBufferU32On(mem.LineScratch, d, uint32(startPos+M))
		idTab = kvIdentityTab(d, r.kvI8ScratchRows())
	}
	attnTile := exactAttnTile(startPos + M) // > 0: the exact-attention layers take attention_prefill_tiled
	uTile := NewBufferU32On(mem.LineScratch, d, uint32(max(attnTile, 1)))
	// The MoE row loop's F32 accumulate scratch (see the per-layer loop's own comment).
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
		// Zero-filled by Metal, not copied from zeroed Go slices; the router GEMM, the route kernels and the host
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
	// The scaled residual's 1/s, for residual_f16_scaled, and the eps of the norms that read the residual, eps/s², so
	// that x/s / sqrt(ms/s² + eps/s²) is x / sqrt(ms + eps) exactly (s is a power of two).
	var uResidInv Buffer
	uEpsX := r.uEps
	if residInv != 1 {
		uResidInv = NewBufferFloatsOn(mem.LineScratch, d, []float32{residInv})
		uEpsX = NewBufferFloatsOn(mem.LineScratch, d, []float32{r.uEps.Floats()[0] * residInv * residInv})
		scratch = append(scratch, uResidInv, uEpsX)
	}
	// A Gemma 4 E-model's per-layer inputs, each row's [L·P] tail of its [h ‖ L·P] embedding, and the PLE
	// gate's scratch.
	var pleGF, uP, uLP Buffer
	if r.pleP > 0 {
		P, LP := r.pleP, r.nL*r.pleP
		if emodelBatchDefect == 3 { // planted defect (3): each row's PLE inputs taken from the next row
			pv := pleInB.Floats()
			for m := 0; m+1 < M; m++ {
				copy(pv[m*LP:(m+1)*LP], pv[(m+1)*LP:(m+2)*LP])
			}
		}
		pleGF, uP, uLP = prefillScratchU16(d, Mpad*P), NewBufferU32On(mem.LineScratch, d, uint32(P)), NewBufferU32On(mem.LineScratch, d, uint32(LP))
		scratch = append(scratch, pleInB, pleGF, uP, uLP)
	}
	// A Gated-DeltaNet hybrid's mixer scratch, and its gated-attention layers' [query ‖ gate] projection, K‖V
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
	var mb *prefillMamba // a Mamba-2 hybrid's mixer scratch
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
	return &prefillPass{d: d, uConsts: map[int]Buffer{}, luCache: map[luKey]*layerU{}, M: M, Mpad: Mpad, H: H, I: I, V: V, nHhd: nHhd, kvDim: kvDim, qkvDim: qkvDim, xF: xF, pleInB: pleInB, normF: normF, qkvF: qkvF, ctxF: ctxF, fusedSw: fusedSw, guF: guF, dqF: dqF, posB: posB, pos3B: pos3B, uM: uM, uH: uH, uI: uI, u2I: u2I, uQkv: uQkv, uHd: uHd, uKOff: uKOff, uStartPos: uStartPos, uTotalQ: uTotalQ, uBase0: uBase0, m0: m0, m1: m1, m2: m2, dummyBias: dummyBias, uMReal: uMReal, kScrI8: kScrI8, vScrI8: vScrI8, uEndI8: uEndI8, idTab: idTab, attnTile: attnTile, uTile: uTile, moeDst: moeDst, moeLogits: moeLogits, moeIdx: moeIdx, moeWgt: moeWgt, expertIn: expertIn, expertDown: expertDown, rowIdxBuf: rowIdxBuf, rowWgtBuf: rowWgtBuf, scratch: scratch, uResidInv: uResidInv, uEpsX: uEpsX, pleGF: pleGF, uP: uP, uLP: uLP, dn: dn, qgF: qgF, kvF: kvF, gateF: gateF, u2NHhd: u2NHhd, u2KvDim: u2KvDim, uGateN: uGateN, mb: mb, imgB: imgB, imgN: imgN}
}

// layerU is one attention geometry's uniforms, per layer. Every family before Gemma 4's E-models has one geometry, so
// it gets one bundle with the values the single-geometry pass used; an E-model gets one per head dim, and a KV-shared
// layer's Q-only projection its own stride. The fused/steel attention kernels hold head dims up to 128; past that the
// exact kernel runs (its key limit is checked before the pass).
type layerU struct {
	g                                                            *attnGeom
	qkvDim, nHhd, kvDim                                          int
	uQkv, uStride, uKOff, uVOff, uTotalQ, uTotalK, uBaseK, uQDim Buffer
	fused, steel, split                                          bool
}
type luKey struct {
	g      *attnGeom
	shared bool
}

// attnFusedGrid is attention_prefill_fused's grid — nH×ceil(M/8) simdgroups (ATTN_SGPT=4/threadgroup,
// prefill.go's own #define, matched here). Real M (unpadded): the tail row-tile's out-of-range
// rows are masked in-kernel via buffer(11), not dropped from the grid.
func (ps *prefillPass) attnFusedGrid() (total, tg int) {
	const attnFusedSGPT = 4
	numRowTiles := (ps.M + 7) / 8
	attnFusedTotal := ps.r.nH * numRowTiles
	attnFusedTotal = (attnFusedTotal + attnFusedSGPT - 1) / attnFusedSGPT * attnFusedSGPT * 32
	return attnFusedTotal, attnFusedSGPT * 32
}

// u32 is the pass's uniform holding val, one buffer per distinct value.
func (ps *prefillPass) u32(val int) Buffer {
	d := ps.d
	if b, ok := ps.uConsts[val]; ok {
		return b
	}
	b := NewBufferU32On(mem.LineScratch, d, uint32(val))
	ps.uConsts[val] = b
	ps.scratch = append(ps.scratch, b)
	return b
}

// gemm is one GEMM dispatch over rows×N — a 2-D grid of (ceil(N/TN), ceil(rows/TM)) threadgroups of 128 threads, each
// owning a TN-feature × TM-token tile: gemm_w4f16_store's 64 × 64, or for a small pass a smaller tile of the same
// arithmetic (gemmTile).
func (ps *prefillPass) gemm(e *Encoder, rows, N int, bufs ...Buffer) {
	pf := ps.pf
	p, tm, tn := pf.gemmTile(rows, N)
	e.Dispatch2D(p, (N+tn-1)/tn, (rows+tm-1)/tm, 128, 1, bufs...)
}

// gemmAttn is gemm for the attention projections (qkv, o), whose weights are int8 on an int4mix resident too.
func (ps *prefillPass) gemmAttn(e *Encoder, rows, N int, bufs ...Buffer) {
	r, pf := ps.r, ps.pf
	p, tm, tn := pf.gemmTileFor(rows, N, r.w8 || r.w8Attn)
	e.Dispatch2D(p, (N+tn-1)/tn, (rows+tm-1)/tm, 128, 1, bufs...)
}

// gemmSegs is gemm over a q4k-lane projection: one gemm_kq_tile dispatch per segment, each writing its own columns of the
// uLdc-wide output; mode 1 reads the bias at the output column.
func (ps *prefillPass) gemmSegs(e *Encoder, rows int, segs []q4kSeg, a, c Buffer, uM, uK, uLdc, bias, mode Buffer) {
	pf := ps.pf
	for _, s := range segs {
		p, tm, tn := pf.gemmKQ(rows, s)
		ws := s.s // int8's per-row scales; the block kinds carry theirs in w
		if ws == (Buffer{}) {
			ws = s.w
		}
		e.Dispatch2D(p, (s.rows+tn-1)/tn, (rows+tm-1)/tm, 128, 1, a, s.w, ws, c, uM, ps.u32(s.rows), uK, bias, mode, uLdc, ps.u32(s.off))
	}
}

// gemmW8 is gemm on the int8 tiles: a Mamba layer's in_proj and out_proj.
func (ps *prefillPass) gemmW8(e *Encoder, rows, N int, bufs ...Buffer) {
	pf := ps.pf
	p, tm, tn := pf.gemmTileFor(rows, N, true)
	e.Dispatch2D(p, (N+tn-1)/tn, (rows+tm-1)/tm, 128, 1, bufs...)
}

// layerUFor is L's attention uniforms, built once per geometry and per KV-shared or owning layer.
func (ps *prefillPass) layerUFor(L *residLayer) *layerU {
	r, fusedOn := ps.r, ps.fusedOn
	k := luKey{L.geom, L.kvShared}
	if u, ok := ps.luCache[k]; ok {
		return u
	}
	g := L.geom
	nh := r.nH * g.hd
	q := nh + 2*g.kvDim
	if L.kvShared {
		q = nh // Q only: a shared layer projects no K or V
	}
	fused := fusedOn && g.hd%8 == 0 && g.hd <= 128
	u := &layerU{g: g, qkvDim: q, nHhd: nh, kvDim: g.kvDim,
		uQkv: ps.u32(q), uStride: ps.u32(q), uKOff: ps.u32(nh), uVOff: ps.u32(nh + g.kvDim),
		uTotalQ: ps.u32(nh), uTotalK: ps.u32(g.kvDim), uBaseK: ps.u32(nh), uQDim: ps.u32(nh),
		fused: fused, steel: fused && steelHeadDim(g.hd) && !prefillSteelAttnOff,
		split: fusedOn && splitSteelHeadDim(g.hd) && !prefillSteelAttnOff}
	ps.luCache[k] = u
	return u
}

// addDeep adds DeepStack set k to the rows of this pass that fall in the image block. Set k is added after layer k.
func (ps *prefillPass) addDeep(k int) {
	pf, e, startPos, deep, M, H, xF, deepB := ps.pf, ps.e, ps.startPos, ps.deep, ps.M, ps.H, ps.xF, ps.deepB
	if k >= len(deepB) {
		return
	}
	lo, hi := max(deep.start, startPos), min(deep.start+deep.n, startPos+M)
	if hi <= lo {
		return
	}
	e.Dispatch(pf.pResF32, (hi-lo)*H, 256, xF.At((lo-startPos)*H*2), deepB[k].At((lo-deep.start)*H*4))
}

// attnBlock, moeBlock, ffnBlock, pleBlock and layerScaleBlock are the batched pass's blocks, one for each layer step it
// implements, each encoding into ps.e. attnBlock is the attention step: the input norm, the q|k|v projection, QK-norm,
// RoPE, the K/V store, attention and the output projection with its residual.
func (ps *prefillPass) attnBlock(l int, L *residLayer) {
	r, pf, e, normPipe, startPos, mrope, M, Mpad, H, nHhd, kvDim, xF, normF, qkvF, ctxF, posB, pos3B, uM, uH, uQkv, uHd, uKOff, uStartPos, uTotalQ, uBase0, m0, m1, m2, dummyBias, uMReal, kScrI8, vScrI8, uEndI8, idTab, attnTile, uTile, uResidInv, uEpsX, qgF, kvF, gateF, u2NHhd, u2KvDim, uGateN, imgB, imgN, zeroKV := ps.r, ps.pf, ps.e, ps.normPipe, ps.startPos, ps.mrope, ps.M, ps.Mpad, ps.H, ps.nHhd, ps.kvDim, ps.xF, ps.normF, ps.qkvF, ps.ctxF, ps.posB, ps.pos3B, ps.uM, ps.uH, ps.uQkv, ps.uHd, ps.uKOff, ps.uStartPos, ps.uTotalQ, ps.uBase0, ps.m0, ps.m1, ps.m2, ps.dummyBias, ps.uMReal, ps.kScrI8, ps.vScrI8, ps.uEndI8, ps.idTab, ps.attnTile, ps.uTile, ps.uResidInv, ps.uEpsX, ps.qgF, ps.kvF, ps.gateF, ps.u2NHhd, ps.u2KvDim, ps.uGateN, ps.imgB, ps.imgN, ps.zeroKV
	attnFusedTotal, attnFusedTg := ps.attnFusedGrid()
	aw := &L.walk.Attn
	lu := ps.layerUFor(L)
	// pre-attn norm — addOne (Gemma's 1+w) matters even for a plain non-sandwich family's
	// GEMV-input norm, so it is always passed (0 for every family without RMSAddOne).
	// Post-only placement (NormCopy): no pre-norm; GEMV reads raw residual xF directly.
	inAttn := normF
	if aw.InputForm == program.NormCopy {
		inAttn = xF
	} else {
		e.Dispatch(normPipe, M*tgReduceNorm, tgReduceNorm, xF, L.preNorm, normF, uH, uEpsX, r.uAddOne)
	}
	// fused QKV (+bias)
	if aw.Gated {
		// Qwen3.5's gated softmax layer (decode's encodeAttention qGate branch): q_proj is [query ‖ gate]
		// per head at double width, split on the activation into qkv's Q slot and the gate; K‖V is its own
		// bias-free projection, copied into qkv's K and V slots.
		ps.gemm(e, Mpad, 2*nHhd, inAttn, L.dnQw, L.dnQs, qgF, uM, u2NHhd, uH, dummyBias, m0)
		e.Dispatch(pf.pQGateSplit, M*nHhd, 256, qgF, qkvF, gateF, uTotalQ, uHd, uQkv, uMReal)
		ps.gemm(e, Mpad, 2*kvDim, inAttn, L.qkvW, L.qkvS, kvF, uM, u2KvDim, uH, dummyBias, m0)
		e.Dispatch(pf.pCopyCols, M*2*kvDim, 256, kvF, qkvF, u2KvDim, uQkv, uKOff, uMReal)
	} else if r.q4kLane {
		ps.gemmSegs(e, Mpad, L.qkvSegs, inAttn, qkvF, uM, uH, lu.uQkv, L.qkvBias, m1)
	} else {
		ps.gemmAttn(e, Mpad, lu.qkvDim, inAttn, L.qkvW, L.qkvS, qkvF, uM, lu.uQkv, uH, L.qkvBias, m1)
	}
	if aw.QKNorm { // Qwen3 / Olmo 3: per-head or whole-vector Q/K RMSNorm before RoPE
		qkNH, qkNKV, qkHD, qkNHhd := r.uNH, lu.g.uNKV, lu.g.uHd, lu.g.uNHhd
		tgCount := r.nH + lu.g.nKV
		if L.kvShared {
			// Q heads only: a shared layer projects no K. The kernel maps a threadgroup to (row, head) by nH+nKV, so nKV
			// must be 0 here too, or rows shift and the "K" head normalizes the next row's Q.
			tgCount, qkNKV = r.nH, r.uZero
		}
		if aw.QKNormParams.Whole {
			qkNH, qkNKV, qkHD, qkNHhd = r.uQKWholeOne, r.uQKWholeOne, r.uQKWholeHD, r.uQKWholeHD
			tgCount = 2
		}
		e.Dispatch(pf.pQK, M*tgCount*tgReduceAttn, tgReduceAttn, qkvF, L.qNorm, L.kNorm, qkNH, qkNKV, qkHD, qkNHhd, lu.uStride, r.uEps, r.uAddOne)
	}
	// Gemma 4's scale-less v_norm on every K/V-owning layer's V slot, after Q/K norm and before RoPE, as
	// decode does (encodeAttentionResidualWith): the Q/K-norm kernel over the V slot with nH = 0, a unit weight and
	// addOne 0.
	if aw.G4VNorm && emodelBatchDefect != 5 {
		e.Dispatch(pf.pQK, M*lu.g.nKV*tgReduceAttn, tgReduceAttn, qkvF.At((lu.nHhd+lu.kvDim)*2), r.vNormUnit, r.vNormUnit,
			r.uZero, lu.g.uNKV, lu.g.uHd, r.uZero, lu.uStride, r.uEps, r.uZero)
	}
	// rope q, k (per-row positions): bind the per-layer RoPE table and window, exactly as decode does
	// (encodeTrunkInto), not the model-level r.invf/r.uWindow. For a mixed local/global-window arch the global
	// layers must see window=0, and each layer its own RoPE base; FeatPerLayerRoPE is in prefillFeatures, so the
	// per-layer binding is not behaviour-neutral.
	if mrope != nil { // each row's own (t, h, w) rotation
		e.Dispatch(pf.pRopeM, M*r.nH*lu.g.half, 128, qkvF, L.invf, lu.g.uHd, pos3B, lu.uTotalQ, lu.uStride, uBase0, lu.g.uHalf, L.mscale, r.mropeAxis)
		if !L.kvShared {
			e.Dispatch(pf.pRopeM, M*lu.g.nKV*lu.g.half, 128, qkvF, L.invf, lu.g.uHd, pos3B, lu.uTotalK, lu.uStride, lu.uBaseK, lu.g.uHalf, L.mscale, r.mropeAxis)
		}
	} else {
		e.Dispatch(pf.pRope, M*r.nH*lu.g.half, 128, qkvF, L.invf, lu.g.uHd, posB, lu.uTotalQ, lu.uStride, uBase0, lu.g.uHalf, L.mscale)
		if !L.kvShared {
			e.Dispatch(pf.pRope, M*lu.g.nKV*lu.g.half, 128, qkvF, L.invf, lu.g.uHd, posB, lu.uTotalK, lu.uStride, lu.uBaseK, lu.g.uHalf, L.mscale)
		}
	}
	// scatter K,V to cache; a KV-shared layer stores nothing (its cache is its source's). The cache is the layer's pool,
	// bound from its start (Buffer.At is absolute), and a position's row is the bound slot's table's (kv_row).
	kcPool, vcPool := r.kc[l].At(0), r.vc[l].At(0)
	if !L.kvShared && r.kvI8 {
		e.Dispatch(pf.pKvI8, M*lu.g.nKV, 32, qkvF, posB, kcPool, vcPool, r.ks[l].At(0), r.vs[l].At(0), lu.g.uNKV, lu.g.uHd, lu.uStride, lu.uKOff, lu.uVOff, uMReal, r.kvKT(l))
	} else if !L.kvShared {
		e.Dispatch(pf.pKv, M*lu.kvDim, 128, qkvF, kcPool, vcPool, posB, lu.g.uKvDim, lu.uStride, lu.uKOff, lu.uVOff, r.kvKT(l))
	}
	kc, vc, kt := kcPool, vcPool, r.kvKT(l)
	if r.kvI8 { // the attention kernels read f16: this layer's cache, dequantized into contiguous rows read through an identity table
		e.Dispatch(pf.pDeqI8, (startPos+M)*lu.kvDim, 256, kcPool, vcPool, r.ks[l].At(0), r.vs[l].At(0), kScrI8, vScrI8, lu.g.uNKV, lu.g.uHd, uEndI8, r.kvKT(l))
		kc, vc, kt = kScrI8, vScrI8, idTab
	}
	if L.kvShared && emodelBatchDefect == 1 {
		kc, vc, kt = zeroKV, zeroKV, idTab
	}
	// causal attention → ctx (per-layer window: 0 = full causal on a global layer)
	ran := ""
	if prefillAttnSkipForTest {
		// no attention: ctxF keeps what it held (the attention-share measurement)
	} else if imgB != (Buffer{}) {
		ran = "attention_prefill_img"
		e.Dispatch(pf.pAttnImg, M*r.nH*tgReduceAttn, tgReduceAttn, qkvF, kc, vc, ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, uStartPos, r.uScale, lu.uStride, L.uWindow, imgB, imgN, kt)
	} else if lu.steel {
		ran = steelName(lu.g.hd)
		e.Dispatch(pf.steelFor(lu.g.hd), r.nH*((M+31)/32)*128, 128, qkvF, kc, vc, ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, uStartPos, r.uScale, lu.uStride, L.uWindow, uMReal, kt)
	} else if lu.split { // 32 query rows a threadgroup at head dim 256, 16 at 512
		p, bq := pf.pAttnSteel256, 32
		if lu.g.hd == 512 {
			p, bq = pf.pAttnSteel512, 16
		}
		ran = fmt.Sprintf("attention_prefill_steel_hd%d", lu.g.hd)
		e.Dispatch(p, r.nH*((M+bq-1)/bq)*256, 256, qkvF, kc, vc, ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, uStartPos, r.uScale, lu.uStride, L.uWindow, uMReal, kt)
	} else if lu.fused {
		ran = "attention_prefill_fused"
		e.Dispatch(pf.pAttnFused, attnFusedTotal, attnFusedTg, qkvF, kc, vc, ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, uStartPos, r.uScale, lu.uStride, L.uWindow, uMReal, kt)
	} else if attnTile > 0 {
		ran = "attention_prefill_tiled"
		e.Dispatch(pf.pAttnTiled, M*r.nH*tgReduceAttn, tgReduceAttn, qkvF, kc, vc, ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, uStartPos, r.uScale, lu.uStride, L.uWindow, uTile, kt)
	} else {
		ran = "attention_prefill"
		e.Dispatch(pf.pAttn, M*r.nH*tgReduceAttn, tgReduceAttn, qkvF, kc, vc, ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, uStartPos, r.uScale, lu.uStride, L.uWindow, kt)
	}
	if prefillAttnRanForTest != nil && ran != "" {
		prefillAttnRanForTest[ran]++
	}
	if aw.Gated { // ctx *= sigmoid(gate), before o-proj, as decode does
		e.Dispatch(pf.pAttnGate, M*nHhd, 256, ctxF, gateF, uGateN)
	}
	// o-proj, then either the plain residual epilogue, Gemma's sandwich norm, or Olmo 3 postOnly: mode-0 write
	// into normF (free scratch here: its last use, the fused-QKV input, already ran, and its next use, the pre-MLP
	// norm's output, is below), norm the sublayer output in place (safe: rmsnorm_f16's read pass completes before
	// its write pass touches the same buffer), then a separate residual add. Other families take the fused mode-2
	// residual epilogue.
	if aw.OutMode == program.OutPostNormResidual {
		ps.gemmAttn(e, Mpad, H, ctxF, L.oW, L.oS, normF, uM, uH, lu.uQDim, dummyBias, m0)
		e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, normF, L.postAttnNorm, normF, uH, r.uEps, r.uAddOne)
		if uResidInv != (Buffer{}) {
			e.Dispatch(pf.pResScaled, M*H, 256, xF, normF, uResidInv)
		} else {
			e.Dispatch(pf.pRes, M*H, 256, xF, normF)
		}
	} else if r.q4kLane {
		ps.gemmSegs(e, Mpad, L.oSegs, ctxF, xF, uM, lu.uQDim, uH, dummyBias, m2)
	} else {
		ps.gemmAttn(e, Mpad, H, ctxF, L.oW, L.oS, xF, uM, uH, lu.uQDim, dummyBias, m2)
	}
}

// moeBlock is the MoE step. It commits ps.e to read the routing on the host and begins the next, and reports false when an
// encoder error stops the pass.
func (ps *prefillPass) moeBlock(l int, L *residLayer) bool {
	e := ps.e // the routing commit ends it and begins ps.e anew
	r, pf, M, Mpad, H, xF, normF, guF, dqF, uM, uH, m0, m2, dummyBias, uMReal, moeDst, moeLogits, moeIdx, moeWgt, expertIn, expertDown, rowIdxBuf, rowWgtBuf, uEpsX := ps.r, ps.pf, ps.M, ps.Mpad, ps.H, ps.xF, ps.normF, ps.guF, ps.dqF, ps.uM, ps.uH, ps.m0, ps.m2, ps.dummyBias, ps.uMReal, ps.moeDst, ps.moeLogits, ps.moeIdx, ps.moeWgt, ps.expertIn, ps.expertDown, ps.rowIdxBuf, ps.rowWgtBuf, ps.uEpsX
	fw := &L.walk.FFN
	normW, _ := r.ffnNormWeights(L, fw.InputNorm.Slot)
	if r.knobValue("GOINFER_MOE_EXPERT_MAJOR") == "0" {
		// Fallback to row-by-row path for A/B testing
		for m := range M {
			row := xF.At(m * H * 2)
			e.Dispatch(pf.pRmsQ, tgReduceNorm, tgReduceNorm, row, normW, r.mq, r.mSc, uH, r.uEps, r.uAddOne)
			r.encodeMoERouteForForm(e, L, fw.FFN.Form)
			if r.moeCap.idx != (Buffer{}) && m < r.moeCap.rows {
				r.encodeMoECapture(e, l*r.moeCap.rows+m)
			}
			e.Dispatch(pf.pZeroF32, r.H, 256, moeDst)
			r.encodeMoEExperts(e, L, moeDst)
			e.Dispatch(pf.pResF32, r.H, 256, row, moeDst)
		}
		return true
	}

	// --- EXPERT-MAJOR PATH ---
	// 1. Pre-norm all M rows into normF
	e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, xF, normW, normF, uH, uEpsX, r.uAddOne)

	// 2. Batched Router GEMM: normF [M, H] x routerW [nE, H] -> moeLogits [M, nE]
	totalRouterThreads := (M*r.moe.nE*32 + 255) / 256 * 256
	e.Dispatch(pf.pRouterGemm, totalRouterThreads, 256, normF, L.moe.routerW, moeLogits, uH, r.moe.uNE, uMReal)

	// 3. Batched Top-k Route: moeLogits -> moeIdx [M, k], moeWgt [M, k]
	if fw.FFN.Form == program.FfnGptOss {
		e.Dispatch(pf.pRouteGptOssBatch, M*32, 32, moeLogits, L.moe.routerBias, moeIdx, moeWgt, r.moe.uNE, r.moe.uK, uMReal)
	} else {
		e.Dispatch(pf.pMoeRouteBatch, M*32, 32, moeLogits, L.moe.routerBias, moeIdx, moeWgt, r.moe.uNE, r.moe.uK, r.moe.uSigmoid, r.moe.uNorm, r.moe.uScale, r.moe.uNGroup, r.moe.uTopkGroup, uMReal)
	}

	// 4. Commit and wait so the host can read the routing decisions
	e.End()
	if db02Trace != nil {
		db02Trace(db02Event{layer: l, gpuStart: e.GPUStart(), gpuEnd: e.GPUEnd()})
	}
	r.recordExecErr(e.Err())
	if err := r.takeExecErr(); err != nil {
		return false
	}

	// 5. Host Grouping: group (token m, rank j) by expert e
	type moeSlot struct {
		row  int
		rank int
		wgt  float32
	}
	byExpert := make([][]moeSlot, r.moe.nE)
	idxSlice := moeIdx.U32s()[:M*r.moe.k]
	wgtSlice := moeWgt.Floats()[:M*r.moe.k]
	if dg01Mutation != "" {
		dg01Mutate(idxSlice, wgtSlice, M, r.moe.k)
	}
	if r.moeCap.major != nil {
		r.moeCap.major(l, idxSlice, wgtSlice)
	}
	for m := range M {
		for j := 0; j < r.moe.k; j++ {
			eIdx := int(idxSlice[m*r.moe.k+j])
			w := wgtSlice[m*r.moe.k+j]
			if eIdx >= 0 && eIdx < r.moe.nE {
				byExpert[eIdx] = append(byExpert[eIdx], moeSlot{row: m, rank: j, wgt: w})
			}
		}
	}

	// 6. Begin new command encoder for expert dispatches
	e = r.q.Begin()
	ps.e = e

	// 7. Dispatch each active expert
	moeInter := r.moe.inter
	rowsPerExpertGu := 2 * moeInter
	rowsPerExpertD := H
	wprGu := H / 8
	gprGu := H / 32
	wprD := moeInter / 8
	gprD := moeInter / 32

	rowIdxHost := rowIdxBuf.U32s()
	rowWgtHost := rowWgtBuf.Floats()

	slotOffset := 0
	for eIdx, slots := range byExpert {
		Ce := len(slots)
		if Ce == 0 {
			continue
		}

		for i, s := range slots {
			rowIdxHost[slotOffset+i] = uint32(s.row)
			rowWgtHost[slotOffset+i] = s.wgt
		}

		CePad := (Ce + 7) / 8 * 8
		if db02Trace != nil {
			_, tmGu, _ := pf.gemmTile(CePad, rowsPerExpertGu)
			_, tmD, _ := pf.gemmTile(CePad, rowsPerExpertD)
			db02Trace(db02Event{layer: l, expert: eIdx, rows: Ce, padRows: CePad,
				tileRowsGu: (CePad + tmGu - 1) / tmGu * tmGu, tileRowsD: (CePad + tmD - 1) / tmD * tmD})
		}
		uCe := ps.u32(Ce)
		uCePad := ps.u32(CePad)
		uMoeI := ps.u32(moeInter)
		u2MoeI := ps.u32(2 * moeInter)

		expIdxBuf := rowIdxBuf.At(slotOffset * 4)
		expWgtBuf := rowWgtBuf.At(slotOffset * 4)

		// A. Gather Ce rows from normF into expertIn (with padding zeroed)
		gatherTotal := (CePad*H + 255) / 256 * 256
		e.Dispatch(pf.pGatherRows, gatherTotal, 256, normF, expertIn, expIdxBuf, uH, uCe, uCePad)

		// B. Expert Gate/Up GEMM
		guWOff := eIdx * rowsPerExpertGu * wprGu * 4
		guSOff := eIdx * rowsPerExpertGu * gprGu * 2
		if r.w8 { // int8 slice 4: H bytes of int8 codes and one f32 scale a row
			guWOff, guSOff = eIdx*rowsPerExpertGu*H, eIdx*rowsPerExpertGu*4
		}
		ps.gemm(e, CePad, rowsPerExpertGu, expertIn, L.moe.expGuW.At(guWOff), L.moe.expGuS.At(guSOff), guF, uCePad, u2MoeI, uH, dummyBias, m0)

		// C. SwiGLU
		e.Dispatch(pf.pSw, CePad*moeInter, 256, guF, dqF, uMoeI, r.uAct)

		// D. Expert Down GEMM
		dOff := eIdx * rowsPerExpertD * wprD * 4
		dSOff := eIdx * rowsPerExpertD * gprD * 2
		if r.w8 { // moeInter bytes of int8 codes and one f32 scale a row
			dOff, dSOff = eIdx*rowsPerExpertD*moeInter, eIdx*rowsPerExpertD*4
		}
		ps.gemm(e, CePad, rowsPerExpertD, dqF, L.moe.expDW.At(dOff), L.moe.expDS.At(dSOff), expertDown, uCePad, uH, uMoeI, dummyBias, m0)

		// E. Scatter-add weighted output into residual xF
		scatterTotal := (Ce*H + 255) / 256 * 256
		e.Dispatch(pf.pScatterAdd, scatterTotal, 256, expertDown, xF, expIdxBuf, expWgtBuf, uH, uCe, uMReal)

		slotOffset += Ce
	}

	// 8. Shared expert (if present)
	if shI := fw.FFN.MoE.SharedIntermediateDim; shI > 0 && dg01Mutation != "shared-off" {
		u2ShI := ps.u32(2 * shI)
		uShI := ps.u32(shI)

		ps.gemm(e, Mpad, 2*shI, normF, L.moe.shGuW, L.moe.shGuS, guF, uM, u2ShI, uH, dummyBias, m0)
		e.Dispatch(pf.pSw, Mpad*shI, 256, guF, dqF, uShI, r.uAct)

		if fw.FFN.MoE.SharedUngated {
			// dst += down directly into xF
			ps.gemm(e, Mpad, H, dqF, L.moe.shDW, L.moe.shDS, xF, uM, uH, uShI, dummyBias, m2)
		} else {
			// gated variant (Qwen2-MoE): compute sigmoid(gate)*down
			ps.gemm(e, Mpad, H, dqF, L.moe.shDW, L.moe.shDS, expertDown, uM, uH, uShI, dummyBias, m0)
			// Gate logit for all M rows
			gateLogitsTotal := (M*32 + 255) / 256 * 256
			e.Dispatch(pf.pRouterGemm, gateLogitsTotal, 256, normF, L.moe.shGateW, moeLogits, uH, ps.u32(1), uMReal)
			// Combine into xF via shared_gate_add_f16
			e.Dispatch(pf.pSharedGateAdd, (M*H+255)/256*256, 256, expertDown, xF, moeLogits, uH, uMReal)
		}
	}
	return true
}

// ffnBlock is the dense or non-gated FFN step.
func (ps *prefillPass) ffnBlock(L *residLayer) {
	r, pf, e, normPipe, M, Mpad, H, I, xF, normF, fusedSw, guF, dqF, uM, uH, uI, u2I, m0, m1, m2, dummyBias, uResidInv, uEpsX := ps.r, ps.pf, ps.e, ps.normPipe, ps.M, ps.Mpad, ps.H, ps.I, ps.xF, ps.normF, ps.fusedSw, ps.guF, ps.dqF, ps.uM, ps.uH, ps.uI, ps.u2I, ps.m0, ps.m1, ps.m2, ps.dummyBias, ps.uResidInv, ps.uEpsX
	fw := &L.walk.FFN
	// pre-MLP norm. A parallel block (Cohere) has none: its MLP reads the same shared norm of the layer's input that the
	// attention read, which normF still holds (the o-proj added into xF directly), so x ends as x + attn + mlp.
	inFFN := normF
	switch {
	case fw.InputMode == program.FFNInShared && !prefillParallelDefectForTest:
	case fw.InputMode == program.FFNInShared: // Part B's planted defect: the MLP fed from the post-attention residual
		e.Dispatch(normPipe, M*tgReduceNorm, tgReduceNorm, xF, L.preNorm, normF, uH, uEpsX, r.uAddOne)
	case fw.InputMode == program.FFNInRaw:
		inFFN = xF
	default:
		normW, _ := r.ffnNormWeights(L, fw.InputNorm.Slot)
		e.Dispatch(normPipe, M*tgReduceNorm, tgReduceNorm, xF, normW, normF, uH, uEpsX, r.uAddOne)
	}
	if fw.FFN.Form == program.FfnNonGated {
		// The non-gated MLP (Nemotron-H's relu²; a down projection with a bias never reaches the pass): up with its
		// bias, the activation in place, down accumulated into the residual.
		ps.gemm(e, Mpad, I, inFFN, L.upW, L.upS, dqF, uM, uI, uH, L.upBias, m1)
		e.Dispatch(r.pNgActRows, M*I, 256, dqF, ps.u32(M*I), r.uAct)
		ps.gemm(e, Mpad, H, dqF, L.dW, L.dS, xF, uM, uH, uI, dummyBias, m2)
		return
	}
	// This layer's FFN width (a Gemma 4 E-model's KV-shared layers are double-wide; every other family's
	// layers are the model's I).
	fI := I
	if L.ffnI != 0 {
		fI = L.ffnI
	}
	if emodelBatchDefect == 6 && r.layers[0].ffnI != 0 { // planted defect (6): every layer at the first layer's width
		fI = r.layers[0].ffnI
	}
	uFI, u2FI := uI, u2I
	if fI != I {
		uFI, u2FI = ps.u32(fI), ps.u32(2*fI)
	}
	// gate/up, then swiglu/geglu (r.uAct selects, 0 for every family without FeatGatedGELU): one fused GEMM when the pass
	// takes it (swigluFused), else the GEMM into guF and swiglu_f16.
	if r.q4kLane {
		ps.gemmSegs(e, Mpad, L.guSegs, inFFN, guF, uM, uH, u2FI, dummyBias, m0)
		e.Dispatch(pf.pSw, M*fI, 256, guF, dqF, uFI, r.uAct)
	} else if fusedSw {
		e.Dispatch2D(pf.pGemmSwiglu, (fI+31)/32, (Mpad+63)/64, 128, 1, inFFN, L.guW, L.guS, dqF, uM, uFI, uH, r.uAct)
	} else {
		ps.gemm(e, Mpad, 2*fI, inFFN, L.guW, L.guS, guF, uM, u2FI, uH, dummyBias, m0)
		e.Dispatch(pf.pSw, M*fI, 256, guF, dqF, uFI, r.uAct)
	}
	// down-proj, same plain-vs-sandwich split as o-proj above.
	if fw.OutMode == program.OutPostNormResidual {
		ps.gemm(e, Mpad, H, dqF, L.dW, L.dS, normF, uM, uH, uFI, dummyBias, m0)
		e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, normF, L.postMLPNorm, normF, uH, r.uEps, r.uAddOne)
		if uResidInv != (Buffer{}) {
			e.Dispatch(pf.pResScaled, M*H, 256, xF, normF, uResidInv)
		} else {
			e.Dispatch(pf.pRes, M*H, 256, xF, normF)
		}
	} else if r.q4kLane {
		ps.gemmSegs(e, Mpad, L.dSegs, dqF, xF, uM, uFI, uH, dummyBias, m2)
	} else {
		ps.gemm(e, Mpad, H, dqF, L.dW, L.dS, xF, uM, uH, uFI, dummyBias, m2)
	}
}

// pleBlock is a Gemma 4 E-model's per-layer-embedding step.
func (ps *prefillPass) pleBlock(l int, L *residLayer) {
	r, pf, e, M, Mpad, H, xF, pleInB, normF, uM, uH, m0, dummyBias, pleGF, uP, uLP := ps.r, ps.pf, ps.e, ps.M, ps.Mpad, ps.H, ps.xF, ps.pleInB, ps.normF, ps.uM, ps.uH, ps.m0, ps.dummyBias, ps.pleGF, ps.uP, ps.uLP
	// A Gemma 4 E-model's PLE block, in decode's order (encodeLayerResidualWith): the gate GEMM (H→P) on the raw
	// residual, gelu(gate) times the row's own per-layer input, the projection GEMM (P→H), the post-PLE norm and the
	// residual add. The layer scalar is its own step (layerScaleBlock).
	if L.pleGW != (Buffer{}) && emodelBatchDefect != 2 {
		P := r.pleP
		ps.gemm(e, Mpad, P, xF, L.pleGW, L.pleGS, pleGF, uM, uP, uH, dummyBias, m0)
		e.Dispatch(pf.pPLEGeluMulF16, M*P, 256, pleGF, pleInB, uP, uLP, ps.u32(l*P))
		ps.gemm(e, Mpad, H, pleGF, L.plePW, L.plePS, normF, uM, uH, uP, dummyBias, m0)
		e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, normF, L.postPLENorm, normF, uH, r.uEps, r.uAddOne)
		e.Dispatch(pf.pRes, M*H, 256, xF, normF)
	}
}

// layerScaleBlock is the layer scalar step.
func (ps *prefillPass) layerScaleBlock(L *residLayer) {
	pf, e, M, H, xF := ps.pf, ps.e, ps.M, ps.H, ps.xF
	if L.uLayerScalar != (Buffer{}) && emodelBatchDefect != 4 {
		e.Dispatch(pf.pLayerScaleF16, M*H, 256, xF, L.uLayerScalar)
	}
}
