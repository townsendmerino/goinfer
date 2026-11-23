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

// newPrefillPass allocates one batched pass's scratch and uniforms, in the order the pass has always allocated them,
// converts src's rows into the f16 residual, and records the pass's inputs. The caller releases them (prefillPass.release) once the pass's last command
// buffer has completed.
func (r *resident) newPrefillPass(src rowSource, startPos int, mrope [][3]int, deep *prefillDeep, blocks [][2]int) *prefillPass {
	d := r.d
	M := src.n
	Mpad := (M + 7) / 8 * 8
	// Prefill runs only for uniform families (prefillOK declines Gemma's per-layer geometry), so
	// every attention layer shares one geometry.
	g0 := r.prefillGeom()
	qkvDim, qDim, maxI := r.prefillWidths()
	normPipe := r.pf.pRms // the GEMM-input norms: RMSNorm, or a bias-free LayerNorm for Cohere (Part B)
	if r.layerNorm {
		normPipe = r.pf.pLN
	}
	ps := &prefillPass{d: d, r: r, pf: r.pf, normPipe: normPipe, startPos: startPos, mrope: mrope, deep: deep, fusedOn: r.prefillFusedOn(),
		uConsts: map[int]Buffer{}, luCache: map[luKey]*layerU{},
		M: M, Mpad: Mpad, H: r.H, I: r.I, V: r.V, nHhd: r.nH * g0.hd, kvDim: g0.kvDim, qkvDim: qkvDim,
		uH: r.uH, uHd: g0.uHd, uEpsX: r.uEps}

	residInv := 1 / r.prefillResidScale(deep)
	ps.loadRows(src, residInv)
	ps.activations(qDim, maxI)
	ps.positions(startPos, mrope)
	ps.uniforms(qDim)
	ps.int8KVScratch()
	ps.attnTile = exactAttnTile(startPos + M) // > 0: the exact-attention layers take attention_prefill_tiled
	ps.uTile = ps.scratchU32(max(ps.attnTile, 1))
	// The MoE row loop's F32 accumulate scratch (see the per-layer loop's own comment).
	// Allocated unconditionally (cheap — H floats) rather than gated on whether any layer is
	// MoE, avoiding a nil-buffer special case in the loop below.
	ps.moeDst = ps.scratchFloats(make([]float32, ps.H))
	ps.moeScratch()
	ps.residScale(residInv)
	ps.pleScratch()
	ps.hybridScratch()
	ps.imageBlocks(blocks)
	// planted defect (1): a shared layer attending over an empty cache instead of its source's (zero in production).
	if emodelBatchDefect == 1 {
		ps.zeroKV = prefillScratchU16(d, (startPos+Mpad+8)*qkvDim)
		ps.scratch = append(ps.scratch, ps.zeroKV)
		if ps.idTab == (Buffer{}) {
			ps.idTab = kvIdentityTab(d, startPos+Mpad+8)
			ps.scratch = append(ps.scratch, ps.idTab)
		}
	}
	return ps
}

// scratchU32 and scratchFloats allocate a per-call uniform onto the device ledger and record it for release.
func (ps *prefillPass) scratchU32(v int) Buffer {
	b := NewBufferU32On(mem.LineScratch, ps.d, uint32(v))
	ps.scratch = append(ps.scratch, b)
	return b
}

func (ps *prefillPass) scratchFloats(v []float32) Buffer {
	b := NewBufferFloatsOn(mem.LineScratch, ps.d, v)
	ps.scratch = append(ps.scratch, b)
	return b
}

// loadRows allocates xF and converts src's rows into it as f16, scaled by residInv. The rows are read rowChunk at a time:
// each row's first H values go into xF, and a Gemma 4 E-model's [L·P] per-layer tail straight into the device buffer the
// PLE gate reads (pleInB), so the host holds a chunk.
//
// xF and the prefillScratchU16 buffers below rely on Metal's zero-fill of a new buffer for their pad rows (aikit
// gpu.NewBufferLen* contract: Metal only; CUDA's is uninitialized) — see prefillScratchU16. The embeddings are converted
// straight into xF's M rows, its pad rows left as Metal zeroed them.
func (ps *prefillPass) loadRows(src rowSource, residInv float32) {
	r, d, M, H := ps.r, ps.d, ps.M, ps.H
	ps.xF = prefillScratchU16(d, ps.Mpad*H)
	ps.scratch = append(ps.scratch, ps.xF)
	if r.pleP > 0 {
		LP := r.nL * r.pleP
		ps.pleInB = d.NewBufferLenOn(mem.LineScratch, M*LP)
		prefillScratchSeen.Add(int64(4 * M * LP))
		ps.scratch = append(ps.scratch, ps.pleInB)
	}
	for lo := 0; lo < M; lo += rowChunk {
		hi := min(lo+rowChunk, M)
		rows := src.rows(lo, hi)
		parallelEmbedsF32ToF16Scaled(ps.xF.U16s()[lo*H:hi*H], rows, H, residInv)
		if r.pleP > 0 {
			LP, pv := r.nL*r.pleP, ps.pleInB.Floats()
			for i, row := range rows {
				copy(pv[(lo+i)*LP:(lo+i+1)*LP], row[H:H+LP])
			}
		}
	}
}

// activations allocates the f16 activation scratch, sized to the padded prompt.
func (ps *prefillPass) activations(qDim, maxI int) {
	d, Mpad := ps.d, ps.Mpad
	ps.normF = prefillScratchU16(d, Mpad*ps.H)
	ps.qkvF = prefillScratchU16(d, Mpad*ps.qkvDim)
	ps.ctxF = prefillScratchU16(d, Mpad*qDim)
	ps.fusedSw = ps.r.swigluFused(Mpad)
	if !ps.fusedSw { // the gate|up rows between the GEMM and swiglu_f16; none when the pass fuses them (swigluFused)
		ps.guF = prefillScratchU16(d, Mpad*2*maxI)
	}
	ps.dqF = prefillScratchU16(d, Mpad*maxI)
	ps.scratch = append(ps.scratch, ps.normF, ps.qkvF, ps.ctxF, ps.guF, ps.dqF)
}

// positions uploads the rows' positions, and their m-RoPE triples when mrope is set.
func (ps *prefillPass) positions(startPos int, mrope [][3]int) {
	d, M, Mpad := ps.d, ps.M, ps.Mpad
	posv := make([]uint32, Mpad)
	for m := range M {
		posv[m] = uint32(startPos + m)
	}
	ps.posB = NewBufferUint32sOn(mem.LineScratch, d, posv)
	prefillScratchSeen.Add(int64(4 * Mpad))
	ps.scratch = append(ps.scratch, ps.posB)
	if mrope != nil { // the rows' m-RoPE triples, row-major [Mpad*3]; pad rows repeat the last real row's
		p3 := make([]uint32, Mpad*3)
		for m := range Mpad {
			t := mrope[startPos+min(m, M-1)]
			p3[m*3], p3[m*3+1], p3[m*3+2] = uint32(t[0]), uint32(t[1]), uint32(t[2])
		}
		ps.pos3B = NewBufferUint32sOn(mem.LineScratch, d, p3)
		ps.scratch = append(ps.scratch, ps.pos3B)
	}
}

// uniforms allocates the pass's per-call uniforms. ps.uH and ps.uHd are the resident's, and are not released.
func (ps *prefillPass) uniforms(qDim int) {
	M, Mpad, nHhd, kvDim, qkvDim := ps.M, ps.Mpad, ps.nHhd, ps.kvDim, ps.qkvDim
	ps.uM = ps.scratchU32(Mpad)
	ps.uI = ps.scratchU32(ps.I)
	ps.u2I = ps.scratchU32(2 * ps.I)
	ps.uQkv = ps.scratchU32(qkvDim)
	ps.scratchU32(qDim)
	ps.scratchU32(qkvDim)
	ps.uKOff = ps.scratchU32(nHhd)
	ps.scratchU32(nHhd + kvDim)
	ps.uStartPos = ps.scratchU32(ps.startPos)
	ps.uTotalQ = ps.scratchU32(nHhd)
	ps.scratchU32(kvDim)
	ps.uBase0 = ps.scratchU32(0)
	ps.scratchU32(nHhd)
	ps.m0, ps.m1, ps.m2 = ps.scratchU32(0), ps.scratchU32(1), ps.scratchU32(2)
	ps.dummyBias = ps.scratchFloats(make([]float32, 1))
	// L2-Metal: attention_prefill_fused's own row-count uniform — REAL M (unpadded), unlike uM
	// above which holds Mpad for the GEMM grid. attention_prefill (the exact kernel) needs no
	// such uniform because its Go-side dispatch grid is already sized off the real M.
	ps.uMReal = ps.scratchU32(M)
}

// int8KVScratch allocates, for an int8 cache, the f16 buffers each layer's rows [0, startPos+M) are dequantized into
// before its attention. The attention kernels read that scratch through idTab, an identity table, as they read the cache
// through its own.
func (ps *prefillPass) int8KVScratch() {
	r, d := ps.r, ps.d
	if !r.kvI8 {
		return
	}
	rows := r.kvI8ScratchRows()
	ps.kScrI8, ps.vScrI8 = prefillScratchU16(d, rows*ps.kvDim), prefillScratchU16(d, rows*ps.kvDim)
	ps.uEndI8 = NewBufferU32On(mem.LineScratch, d, uint32(ps.startPos+ps.M))
	ps.idTab = kvIdentityTab(d, rows)
	ps.scratch = append(ps.scratch, ps.kScrI8, ps.vScrI8, ps.uEndI8, ps.idTab)
}

// moeScratch allocates the MoE expert-major prefill scratch, only when the model has MoE layers.
func (ps *prefillPass) moeScratch() {
	r, d, M, Mpad, H := ps.r, ps.d, ps.M, ps.Mpad, ps.H
	if r.moe == nil {
		return
	}
	// Zero-filled by Metal, not copied from zeroed Go slices; the router GEMM, the route kernels and the host
	// fill them before anything reads them (see prefillScratch).
	ps.moeLogits = prefillScratch[float32](d, M*r.moe.nE)
	ps.moeIdx = prefillScratch[uint32](d, M*r.moe.k)
	ps.moeWgt = prefillScratch[float32](d, M*r.moe.k)
	// expertIn/expertDown rely on the same Metal zero-fill for their pad rows as the scratch above (see prefillScratchU16).
	ps.expertIn = prefillScratchU16(d, Mpad*H)
	ps.expertDown = prefillScratchU16(d, Mpad*H)
	ps.rowIdxBuf = prefillScratch[uint32](d, Mpad*r.moe.k)
	ps.rowWgtBuf = prefillScratch[float32](d, Mpad*r.moe.k)
	if prefillScratchPoison { // a test's proof that nothing reads these before the pass writes them
		for _, b := range []Buffer{ps.moeLogits, ps.moeIdx, ps.moeWgt, ps.rowIdxBuf, ps.rowWgtBuf} {
			for i, u := 0, b.U32s(); i < len(u); i++ {
				u[i] = 0xFFFFFFFF
			}
		}
	}
	ps.scratch = append(ps.scratch, ps.moeLogits, ps.moeIdx, ps.moeWgt, ps.expertIn, ps.expertDown, ps.rowIdxBuf, ps.rowWgtBuf)
}

// residScale allocates the scaled residual's 1/s, for residual_f16_scaled, and the eps of the norms that read the
// residual, eps/s², so that x/s / sqrt(ms/s² + eps/s²) is x / sqrt(ms + eps) exactly (s is a power of two). ps.uEpsX is
// the resident's r.uEps when residInv is 1.
func (ps *prefillPass) residScale(residInv float32) {
	if residInv == 1 {
		return
	}
	ps.uResidInv = ps.scratchFloats([]float32{residInv})
	ps.uEpsX = ps.scratchFloats([]float32{ps.r.uEps.Floats()[0] * residInv * residInv})
}

// pleScratch allocates a Gemma 4 E-model's PLE gate scratch; the per-layer inputs are pleInB (loadRows).
func (ps *prefillPass) pleScratch() {
	r, d, M := ps.r, ps.d, ps.M
	if r.pleP <= 0 {
		return
	}
	P, LP := r.pleP, r.nL*r.pleP
	if emodelBatchDefect == 3 { // planted defect (3): each row's PLE inputs taken from the next row
		pv := ps.pleInB.Floats()
		for m := 0; m+1 < M; m++ {
			copy(pv[m*LP:(m+1)*LP], pv[(m+1)*LP:(m+2)*LP])
		}
	}
	ps.pleGF, ps.uP, ps.uLP = prefillScratchU16(d, ps.Mpad*P), NewBufferU32On(mem.LineScratch, d, uint32(P)), NewBufferU32On(mem.LineScratch, d, uint32(LP))
	ps.scratch = append(ps.scratch, ps.pleGF, ps.uP, ps.uLP)
}

// hybridScratch allocates a Gated-DeltaNet hybrid's mixer scratch and its gated-attention layers' [query ‖ gate]
// projection, K‖V projection and split gate, or a Mamba-2 hybrid's mixer scratch.
func (ps *prefillPass) hybridScratch() {
	r, d, M, Mpad, nHhd, kvDim := ps.r, ps.d, ps.M, ps.Mpad, ps.nHhd, ps.kvDim
	if r.dnet != nil {
		ps.dn = r.newPrefillDelta(M, Mpad)
		ps.qgF = prefillScratchU16(d, Mpad*2*nHhd)
		ps.kvF = prefillScratchU16(d, Mpad*2*kvDim)
		ps.gateF = prefillScratchU16(d, M*nHhd)
		ps.u2NHhd, ps.u2KvDim, ps.uGateN = NewBufferU32On(mem.LineScratch, d, uint32(2*nHhd)), NewBufferU32On(mem.LineScratch, d, uint32(2*kvDim)), NewBufferU32On(mem.LineScratch, d, uint32(M*nHhd))
		ps.scratch = append(append(ps.scratch, ps.dn.bufs...), ps.qgF, ps.kvF, ps.gateF, ps.u2NHhd, ps.u2KvDim, ps.uGateN)
	}
	if r.mamba != nil {
		ps.mb = r.newPrefillMamba(M, Mpad)
		ps.scratch = append(ps.scratch, ps.mb.bufs...)
	}
}

// imageBlocks uploads the bidirectional image blocks and their count, when there are any (attention_prefill_img).
func (ps *prefillPass) imageBlocks(blocks [][2]int) {
	if len(blocks) == 0 {
		return
	}
	flat := make([]uint32, 0, 2*len(blocks))
	for _, b := range blocks {
		flat = append(flat, uint32(b[0]), uint32(b[1]))
	}
	ps.imgB, ps.imgN = NewBufferUint32sOn(mem.LineScratch, ps.d, flat), NewBufferU32On(mem.LineScratch, ps.d, uint32(len(blocks)))
	ps.scratch = append(ps.scratch, ps.imgB, ps.imgN)
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
	pf, e := ps.pf, ps.e
	if k >= len(ps.deepB) {
		return
	}
	lo, hi := max(ps.deep.start, ps.startPos), min(ps.deep.start+ps.deep.n, ps.startPos+ps.M)
	if hi <= lo {
		return
	}
	e.Dispatch(pf.pResF32, (hi-lo)*ps.H, 256, ps.xF.At((lo-ps.startPos)*ps.H*2), ps.deepB[k].At((lo-ps.deep.start)*ps.H*4))
}

// attnBlock, moeBlock, ffnBlock, pleBlock and layerScaleBlock are the batched pass's blocks, one for each layer step it
// implements, each encoding into ps.e. attnBlock is the attention step: the input norm, the q|k|v projection, QK-norm,
// RoPE, the K/V store, attention and the output projection with its residual.
func (ps *prefillPass) attnBlock(l int, L *residLayer) {
	r, pf, e := ps.r, ps.pf, ps.e
	attnFusedTotal, attnFusedTg := ps.attnFusedGrid()
	aw := &L.walk.Attn
	lu := ps.layerUFor(L)
	// pre-attn norm — addOne (Gemma's 1+w) matters even for a plain non-sandwich family's
	// GEMV-input norm, so it is always passed (0 for every family without RMSAddOne).
	// Post-only placement (NormCopy): no pre-norm; GEMV reads raw residual xF directly.
	inAttn := ps.normF
	if aw.InputForm == program.NormCopy {
		inAttn = ps.xF
	} else {
		e.Dispatch(ps.normPipe, ps.M*tgReduceNorm, tgReduceNorm, ps.xF, L.preNorm, ps.normF, ps.uH, ps.uEpsX, r.uAddOne)
	}
	// fused QKV (+bias)
	if aw.Gated {
		// Qwen3.5's gated softmax layer (decode's encodeAttention qGate branch): q_proj is [query ‖ gate]
		// per head at double width, split on the activation into qkv's Q slot and the gate; K‖V is its own
		// bias-free projection, copied into qkv's K and V slots.
		ps.gemm(e, ps.Mpad, 2*ps.nHhd, inAttn, L.dnQw, L.dnQs, ps.qgF, ps.uM, ps.u2NHhd, ps.uH, ps.dummyBias, ps.m0)
		e.Dispatch(pf.pQGateSplit, ps.M*ps.nHhd, 256, ps.qgF, ps.qkvF, ps.gateF, ps.uTotalQ, ps.uHd, ps.uQkv, ps.uMReal)
		ps.gemm(e, ps.Mpad, 2*ps.kvDim, inAttn, L.qkvW, L.qkvS, ps.kvF, ps.uM, ps.u2KvDim, ps.uH, ps.dummyBias, ps.m0)
		e.Dispatch(pf.pCopyCols, ps.M*2*ps.kvDim, 256, ps.kvF, ps.qkvF, ps.u2KvDim, ps.uQkv, ps.uKOff, ps.uMReal)
	} else if r.q4kLane {
		ps.gemmSegs(e, ps.Mpad, L.qkvSegs, inAttn, ps.qkvF, ps.uM, ps.uH, lu.uQkv, L.qkvBias, ps.m1)
	} else {
		ps.gemmAttn(e, ps.Mpad, lu.qkvDim, inAttn, L.qkvW, L.qkvS, ps.qkvF, ps.uM, lu.uQkv, ps.uH, L.qkvBias, ps.m1)
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
		e.Dispatch(pf.pQK, ps.M*tgCount*tgReduceAttn, tgReduceAttn, ps.qkvF, L.qNorm, L.kNorm, qkNH, qkNKV, qkHD, qkNHhd, lu.uStride, r.uEps, r.uAddOne)
	}
	// Gemma 4's scale-less v_norm on every K/V-owning layer's V slot, after Q/K norm and before RoPE, as
	// decode does (encodeAttentionResidualWith): the Q/K-norm kernel over the V slot with nH = 0, a unit weight and
	// addOne 0.
	if aw.G4VNorm && emodelBatchDefect != 5 {
		e.Dispatch(pf.pQK, ps.M*lu.g.nKV*tgReduceAttn, tgReduceAttn, ps.qkvF.At((lu.nHhd+lu.kvDim)*2), r.vNormUnit, r.vNormUnit,
			r.uZero, lu.g.uNKV, lu.g.uHd, r.uZero, lu.uStride, r.uEps, r.uZero)
	}
	// rope q, k (per-row positions): bind the per-layer RoPE table and window, exactly as decode does
	// (encodeTrunkInto), not the model-level r.invf/r.uWindow. For a mixed local/global-window arch the global
	// layers must see window=0, and each layer its own RoPE base; FeatPerLayerRoPE is in prefillFeatures, so the
	// per-layer binding is not behaviour-neutral.
	if ps.mrope != nil { // each row's own (t, h, w) rotation
		e.Dispatch(pf.pRopeM, ps.M*r.nH*lu.g.half, 128, ps.qkvF, L.invf, lu.g.uHd, ps.pos3B, lu.uTotalQ, lu.uStride, ps.uBase0, lu.g.uHalf, L.mscale, r.mropeAxis)
		if !L.kvShared {
			e.Dispatch(pf.pRopeM, ps.M*lu.g.nKV*lu.g.half, 128, ps.qkvF, L.invf, lu.g.uHd, ps.pos3B, lu.uTotalK, lu.uStride, lu.uBaseK, lu.g.uHalf, L.mscale, r.mropeAxis)
		}
	} else {
		e.Dispatch(pf.pRope, ps.M*r.nH*lu.g.half, 128, ps.qkvF, L.invf, lu.g.uHd, ps.posB, lu.uTotalQ, lu.uStride, ps.uBase0, lu.g.uHalf, L.mscale)
		if !L.kvShared {
			e.Dispatch(pf.pRope, ps.M*lu.g.nKV*lu.g.half, 128, ps.qkvF, L.invf, lu.g.uHd, ps.posB, lu.uTotalK, lu.uStride, lu.uBaseK, lu.g.uHalf, L.mscale)
		}
	}
	// scatter K,V to cache; a KV-shared layer stores nothing (its cache is its source's). The cache is the layer's pool,
	// bound from its start (Buffer.At is absolute), and a position's row is the bound slot's table's (kv_row).
	kcPool, vcPool := r.kc[l].At(0), r.vc[l].At(0)
	if !L.kvShared && r.kvI8 {
		e.Dispatch(pf.pKvI8, ps.M*lu.g.nKV, 32, ps.qkvF, ps.posB, kcPool, vcPool, r.ks[l].At(0), r.vs[l].At(0), lu.g.uNKV, lu.g.uHd, lu.uStride, lu.uKOff, lu.uVOff, ps.uMReal, r.kvKT(l))
	} else if !L.kvShared {
		e.Dispatch(pf.pKv, ps.M*lu.kvDim, 128, ps.qkvF, kcPool, vcPool, ps.posB, lu.g.uKvDim, lu.uStride, lu.uKOff, lu.uVOff, r.kvKT(l))
	}
	kc, vc, kt, idTab := kcPool, vcPool, r.kvKT(l), ps.idTab
	if r.kvI8 { // the attention kernels read f16: this layer's cache, dequantized into contiguous rows read through an identity table
		e.Dispatch(pf.pDeqI8, (ps.startPos+ps.M)*lu.kvDim, 256, kcPool, vcPool, r.ks[l].At(0), r.vs[l].At(0), ps.kScrI8, ps.vScrI8, lu.g.uNKV, lu.g.uHd, ps.uEndI8, r.kvKT(l))
		kc, vc, kt = ps.kScrI8, ps.vScrI8, idTab
	}
	if L.kvShared && emodelBatchDefect == 1 {
		kc, vc, kt = ps.zeroKV, ps.zeroKV, idTab
	}
	// causal attention → ctx (per-layer window: 0 = full causal on a global layer)
	ran := ""
	if prefillAttnSkipForTest {
		// no attention: ctxF keeps what it held (the attention-share measurement)
	} else if ps.imgB != (Buffer{}) {
		ran = "attention_prefill_img"
		e.Dispatch(pf.pAttnImg, ps.M*r.nH*tgReduceAttn, tgReduceAttn, ps.qkvF, kc, vc, ps.ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, ps.uStartPos, r.uScale, lu.uStride, L.uWindow, ps.imgB, ps.imgN, kt)
	} else if lu.steel {
		ran = steelName(lu.g.hd)
		e.Dispatch(pf.steelFor(lu.g.hd), r.nH*((ps.M+31)/32)*128, 128, ps.qkvF, kc, vc, ps.ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, ps.uStartPos, r.uScale, lu.uStride, L.uWindow, ps.uMReal, kt)
	} else if lu.split { // 32 query rows a threadgroup at head dim 256, 16 at 512
		p, bq := pf.pAttnSteel256, 32
		if lu.g.hd == 512 {
			p, bq = pf.pAttnSteel512, 16
		}
		ran = fmt.Sprintf("attention_prefill_steel_hd%d", lu.g.hd)
		e.Dispatch(p, r.nH*((ps.M+bq-1)/bq)*256, 256, ps.qkvF, kc, vc, ps.ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, ps.uStartPos, r.uScale, lu.uStride, L.uWindow, ps.uMReal, kt)
	} else if lu.fused {
		ran = "attention_prefill_fused"
		e.Dispatch(pf.pAttnFused, attnFusedTotal, attnFusedTg, ps.qkvF, kc, vc, ps.ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, ps.uStartPos, r.uScale, lu.uStride, L.uWindow, ps.uMReal, kt)
	} else if ps.attnTile > 0 {
		ran = "attention_prefill_tiled"
		e.Dispatch(pf.pAttnTiled, ps.M*r.nH*tgReduceAttn, tgReduceAttn, ps.qkvF, kc, vc, ps.ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, ps.uStartPos, r.uScale, lu.uStride, L.uWindow, ps.uTile, kt)
	} else {
		ran = "attention_prefill"
		e.Dispatch(pf.pAttn, ps.M*r.nH*tgReduceAttn, tgReduceAttn, ps.qkvF, kc, vc, ps.ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, ps.uStartPos, r.uScale, lu.uStride, L.uWindow, kt)
	}
	if prefillAttnRanForTest != nil && ran != "" {
		prefillAttnRanForTest[ran]++
	}
	if aw.Gated { // ctx *= sigmoid(gate), before o-proj, as decode does
		e.Dispatch(pf.pAttnGate, ps.M*ps.nHhd, 256, ps.ctxF, ps.gateF, ps.uGateN)
	}
	// o-proj, then either the plain residual epilogue, Gemma's sandwich norm, or Olmo 3 postOnly: mode-0 write
	// into normF (free scratch here: its last use, the fused-QKV input, already ran, and its next use, the pre-MLP
	// norm's output, is below), norm the sublayer output in place (safe: rmsnorm_f16's read pass completes before
	// its write pass touches the same buffer), then a separate residual add. Other families take the fused mode-2
	// residual epilogue.
	if aw.OutMode == program.OutPostNormResidual {
		ps.gemmAttn(e, ps.Mpad, ps.H, ps.ctxF, L.oW, L.oS, ps.normF, ps.uM, ps.uH, lu.uQDim, ps.dummyBias, ps.m0)
		e.Dispatch(pf.pRms, ps.M*tgReduceNorm, tgReduceNorm, ps.normF, L.postAttnNorm, ps.normF, ps.uH, r.uEps, r.uAddOne)
		if ps.uResidInv != (Buffer{}) {
			e.Dispatch(pf.pResScaled, ps.M*ps.H, 256, ps.xF, ps.normF, ps.uResidInv)
		} else {
			e.Dispatch(pf.pRes, ps.M*ps.H, 256, ps.xF, ps.normF)
		}
	} else if r.q4kLane {
		ps.gemmSegs(e, ps.Mpad, L.oSegs, ps.ctxF, ps.xF, ps.uM, lu.uQDim, ps.uH, ps.dummyBias, ps.m2)
	} else {
		ps.gemmAttn(e, ps.Mpad, ps.H, ps.ctxF, L.oW, L.oS, ps.xF, ps.uM, ps.uH, lu.uQDim, ps.dummyBias, ps.m2)
	}
}

// moeBlock is the MoE step. It commits ps.e to read the routing on the host and begins the next, and reports false when an
// encoder error stops the pass.
func (ps *prefillPass) moeBlock(l int, L *residLayer) bool {
	e := ps.e // the routing commit ends it and begins ps.e anew
	r, pf := ps.r, ps.pf
	fw := &L.walk.FFN
	normW, _ := r.ffnNormWeights(L, fw.InputNorm.Slot)
	if r.knobValue("GOINFER_MOE_EXPERT_MAJOR") == "0" {
		// Fallback to row-by-row path for A/B testing
		for m := range ps.M {
			row := ps.xF.At(m * ps.H * 2)
			e.Dispatch(pf.pRmsQ, tgReduceNorm, tgReduceNorm, row, normW, r.mq, r.mSc, ps.uH, r.uEps, r.uAddOne)
			r.encodeMoERouteForForm(e, L, fw.FFN.Form)
			if r.moeCap.idx != (Buffer{}) && m < r.moeCap.rows {
				r.encodeMoECapture(e, l*r.moeCap.rows+m)
			}
			e.Dispatch(pf.pZeroF32, r.H, 256, ps.moeDst)
			r.encodeMoEExperts(e, L, ps.moeDst)
			e.Dispatch(pf.pResF32, r.H, 256, row, ps.moeDst)
		}
		return true
	}

	// --- EXPERT-MAJOR PATH ---
	// 1. Pre-norm all M rows into normF
	e.Dispatch(pf.pRms, ps.M*tgReduceNorm, tgReduceNorm, ps.xF, normW, ps.normF, ps.uH, ps.uEpsX, r.uAddOne)

	// 2. Batched Router GEMM: normF [M, H] x routerW [nE, H] -> moeLogits [M, nE]
	totalRouterThreads := (ps.M*r.moe.nE*32 + 255) / 256 * 256
	e.Dispatch(pf.pRouterGemm, totalRouterThreads, 256, ps.normF, L.moe.routerW, ps.moeLogits, ps.uH, r.moe.uNE, ps.uMReal)

	// 3. Batched Top-k Route: moeLogits -> moeIdx [M, k], moeWgt [M, k]
	if fw.FFN.Form == program.FfnGptOss {
		e.Dispatch(pf.pRouteGptOssBatch, ps.M*32, 32, ps.moeLogits, L.moe.routerBias, ps.moeIdx, ps.moeWgt, r.moe.uNE, r.moe.uK, ps.uMReal)
	} else {
		e.Dispatch(pf.pMoeRouteBatch, ps.M*32, 32, ps.moeLogits, L.moe.routerBias, ps.moeIdx, ps.moeWgt, r.moe.uNE, r.moe.uK, r.moe.uSigmoid, r.moe.uNorm, r.moe.uScale, r.moe.uNGroup, r.moe.uTopkGroup, ps.uMReal)
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
	idxSlice := ps.moeIdx.U32s()[:ps.M*r.moe.k]
	wgtSlice := ps.moeWgt.Floats()[:ps.M*r.moe.k]
	if dg01Mutation != "" {
		dg01Mutate(idxSlice, wgtSlice, ps.M, r.moe.k)
	}
	if r.moeCap.major != nil {
		r.moeCap.major(l, idxSlice, wgtSlice)
	}
	for m := range ps.M {
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
	rowsPerExpertD := ps.H
	wprGu := ps.H / 8
	gprGu := ps.H / 32
	wprD := moeInter / 8
	gprD := moeInter / 32

	rowIdxHost := ps.rowIdxBuf.U32s()
	rowWgtHost := ps.rowWgtBuf.Floats()

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

		expIdxBuf := ps.rowIdxBuf.At(slotOffset * 4)
		expWgtBuf := ps.rowWgtBuf.At(slotOffset * 4)

		// A. Gather Ce rows from normF into expertIn (with padding zeroed)
		gatherTotal := (CePad*ps.H + 255) / 256 * 256
		e.Dispatch(pf.pGatherRows, gatherTotal, 256, ps.normF, ps.expertIn, expIdxBuf, ps.uH, uCe, uCePad)

		// B. Expert Gate/Up GEMM
		guWOff := eIdx * rowsPerExpertGu * wprGu * 4
		guSOff := eIdx * rowsPerExpertGu * gprGu * 2
		if r.w8 { // int8 slice 4: H bytes of int8 codes and one f32 scale a row
			guWOff, guSOff = eIdx*rowsPerExpertGu*ps.H, eIdx*rowsPerExpertGu*4
		}
		ps.gemm(e, CePad, rowsPerExpertGu, ps.expertIn, L.moe.expGuW.At(guWOff), L.moe.expGuS.At(guSOff), ps.guF, uCePad, u2MoeI, ps.uH, ps.dummyBias, ps.m0)

		// C. SwiGLU
		e.Dispatch(pf.pSw, CePad*moeInter, 256, ps.guF, ps.dqF, uMoeI, r.uAct)

		// D. Expert Down GEMM
		dOff := eIdx * rowsPerExpertD * wprD * 4
		dSOff := eIdx * rowsPerExpertD * gprD * 2
		if r.w8 { // moeInter bytes of int8 codes and one f32 scale a row
			dOff, dSOff = eIdx*rowsPerExpertD*moeInter, eIdx*rowsPerExpertD*4
		}
		ps.gemm(e, CePad, rowsPerExpertD, ps.dqF, L.moe.expDW.At(dOff), L.moe.expDS.At(dSOff), ps.expertDown, uCePad, ps.uH, uMoeI, ps.dummyBias, ps.m0)

		// E. Scatter-add weighted output into residual xF
		scatterTotal := (Ce*ps.H + 255) / 256 * 256
		e.Dispatch(pf.pScatterAdd, scatterTotal, 256, ps.expertDown, ps.xF, expIdxBuf, expWgtBuf, ps.uH, uCe, ps.uMReal)

		slotOffset += Ce
	}

	// 8. Shared expert (if present)
	if shI := fw.FFN.MoE.SharedIntermediateDim; shI > 0 && dg01Mutation != "shared-off" {
		u2ShI := ps.u32(2 * shI)
		uShI := ps.u32(shI)

		ps.gemm(e, ps.Mpad, 2*shI, ps.normF, L.moe.shGuW, L.moe.shGuS, ps.guF, ps.uM, u2ShI, ps.uH, ps.dummyBias, ps.m0)
		e.Dispatch(pf.pSw, ps.Mpad*shI, 256, ps.guF, ps.dqF, uShI, r.uAct)

		if fw.FFN.MoE.SharedUngated {
			// dst += down directly into xF
			ps.gemm(e, ps.Mpad, ps.H, ps.dqF, L.moe.shDW, L.moe.shDS, ps.xF, ps.uM, ps.uH, uShI, ps.dummyBias, ps.m2)
		} else {
			// gated variant (Qwen2-MoE): compute sigmoid(gate)*down
			ps.gemm(e, ps.Mpad, ps.H, ps.dqF, L.moe.shDW, L.moe.shDS, ps.expertDown, ps.uM, ps.uH, uShI, ps.dummyBias, ps.m0)
			// Gate logit for all M rows
			gateLogitsTotal := (ps.M*32 + 255) / 256 * 256
			e.Dispatch(pf.pRouterGemm, gateLogitsTotal, 256, ps.normF, L.moe.shGateW, ps.moeLogits, ps.uH, ps.u32(1), ps.uMReal)
			// Combine into xF via shared_gate_add_f16
			e.Dispatch(pf.pSharedGateAdd, (ps.M*ps.H+255)/256*256, 256, ps.expertDown, ps.xF, ps.moeLogits, ps.uH, ps.uMReal)
		}
	}
	return true
}

// ffnBlock is the dense or non-gated FFN step.
func (ps *prefillPass) ffnBlock(L *residLayer) {
	r, pf, e := ps.r, ps.pf, ps.e
	fw := &L.walk.FFN
	// pre-MLP norm. A parallel block (Cohere) has none: its MLP reads the same shared norm of the layer's input that the
	// attention read, which normF still holds (the o-proj added into xF directly), so x ends as x + attn + mlp.
	inFFN := ps.normF
	switch {
	case fw.InputMode == program.FFNInShared && !prefillParallelDefectForTest:
	case fw.InputMode == program.FFNInShared: // Part B's planted defect: the MLP fed from the post-attention residual
		e.Dispatch(ps.normPipe, ps.M*tgReduceNorm, tgReduceNorm, ps.xF, L.preNorm, ps.normF, ps.uH, ps.uEpsX, r.uAddOne)
	case fw.InputMode == program.FFNInRaw:
		inFFN = ps.xF
	default:
		normW, _ := r.ffnNormWeights(L, fw.InputNorm.Slot)
		e.Dispatch(ps.normPipe, ps.M*tgReduceNorm, tgReduceNorm, ps.xF, normW, ps.normF, ps.uH, ps.uEpsX, r.uAddOne)
	}
	if fw.FFN.Form == program.FfnNonGated {
		// The non-gated MLP (Nemotron-H's relu²; a down projection with a bias never reaches the pass): up with its
		// bias, the activation in place, down accumulated into the residual.
		ps.gemm(e, ps.Mpad, ps.I, inFFN, L.upW, L.upS, ps.dqF, ps.uM, ps.uI, ps.uH, L.upBias, ps.m1)
		e.Dispatch(r.pNgActRows, ps.M*ps.I, 256, ps.dqF, ps.u32(ps.M*ps.I), r.uAct)
		ps.gemm(e, ps.Mpad, ps.H, ps.dqF, L.dW, L.dS, ps.xF, ps.uM, ps.uH, ps.uI, ps.dummyBias, ps.m2)
		return
	}
	// This layer's FFN width (a Gemma 4 E-model's KV-shared layers are double-wide; every other family's
	// layers are the model's I).
	fI := ps.I
	if L.ffnI != 0 {
		fI = L.ffnI
	}
	if emodelBatchDefect == 6 && r.layers[0].ffnI != 0 { // planted defect (6): every layer at the first layer's width
		fI = r.layers[0].ffnI
	}
	uFI, u2FI := ps.uI, ps.u2I
	if fI != ps.I {
		uFI, u2FI = ps.u32(fI), ps.u32(2*fI)
	}
	// gate/up, then swiglu/geglu (r.uAct selects, 0 for every family without FeatGatedGELU): one fused GEMM when the pass
	// takes it (swigluFused), else the GEMM into guF and swiglu_f16.
	if r.q4kLane {
		ps.gemmSegs(e, ps.Mpad, L.guSegs, inFFN, ps.guF, ps.uM, ps.uH, u2FI, ps.dummyBias, ps.m0)
		e.Dispatch(pf.pSw, ps.M*fI, 256, ps.guF, ps.dqF, uFI, r.uAct)
	} else if ps.fusedSw {
		e.Dispatch2D(pf.pGemmSwiglu, (fI+31)/32, (ps.Mpad+63)/64, 128, 1, inFFN, L.guW, L.guS, ps.dqF, ps.uM, uFI, ps.uH, r.uAct)
	} else {
		ps.gemm(e, ps.Mpad, 2*fI, inFFN, L.guW, L.guS, ps.guF, ps.uM, u2FI, ps.uH, ps.dummyBias, ps.m0)
		e.Dispatch(pf.pSw, ps.M*fI, 256, ps.guF, ps.dqF, uFI, r.uAct)
	}
	// down-proj, same plain-vs-sandwich split as o-proj above.
	if fw.OutMode == program.OutPostNormResidual {
		ps.gemm(e, ps.Mpad, ps.H, ps.dqF, L.dW, L.dS, ps.normF, ps.uM, ps.uH, uFI, ps.dummyBias, ps.m0)
		e.Dispatch(pf.pRms, ps.M*tgReduceNorm, tgReduceNorm, ps.normF, L.postMLPNorm, ps.normF, ps.uH, r.uEps, r.uAddOne)
		if ps.uResidInv != (Buffer{}) {
			e.Dispatch(pf.pResScaled, ps.M*ps.H, 256, ps.xF, ps.normF, ps.uResidInv)
		} else {
			e.Dispatch(pf.pRes, ps.M*ps.H, 256, ps.xF, ps.normF)
		}
	} else if r.q4kLane {
		ps.gemmSegs(e, ps.Mpad, L.dSegs, ps.dqF, ps.xF, ps.uM, uFI, ps.uH, ps.dummyBias, ps.m2)
	} else {
		ps.gemm(e, ps.Mpad, ps.H, ps.dqF, L.dW, L.dS, ps.xF, ps.uM, ps.uH, uFI, ps.dummyBias, ps.m2)
	}
}

// pleBlock is a Gemma 4 E-model's per-layer-embedding step.
func (ps *prefillPass) pleBlock(l int, L *residLayer) {
	r, pf, e := ps.r, ps.pf, ps.e
	// A Gemma 4 E-model's PLE block, in decode's order (encodeLayerResidualWith): the gate GEMM (H→P) on the raw
	// residual, gelu(gate) times the row's own per-layer input, the projection GEMM (P→H), the post-PLE norm and the
	// residual add. The layer scalar is its own step (layerScaleBlock).
	if L.pleGW != (Buffer{}) && emodelBatchDefect != 2 {
		P := r.pleP
		ps.gemm(e, ps.Mpad, P, ps.xF, L.pleGW, L.pleGS, ps.pleGF, ps.uM, ps.uP, ps.uH, ps.dummyBias, ps.m0)
		e.Dispatch(pf.pPLEGeluMulF16, ps.M*P, 256, ps.pleGF, ps.pleInB, ps.uP, ps.uLP, ps.u32(l*P))
		ps.gemm(e, ps.Mpad, ps.H, ps.pleGF, L.plePW, L.plePS, ps.normF, ps.uM, ps.uH, ps.uP, ps.dummyBias, ps.m0)
		e.Dispatch(pf.pRms, ps.M*tgReduceNorm, tgReduceNorm, ps.normF, L.postPLENorm, ps.normF, ps.uH, r.uEps, r.uAddOne)
		e.Dispatch(pf.pRes, ps.M*ps.H, 256, ps.xF, ps.normF)
	}
}

// layerScaleBlock is the layer scalar step.
func (ps *prefillPass) layerScaleBlock(L *residLayer) {
	pf, e, M, H, xF := ps.pf, ps.e, ps.M, ps.H, ps.xF
	if L.uLayerScalar != (Buffer{}) && emodelBatchDefect != 4 {
		e.Dispatch(pf.pLayerScaleF16, M*H, 256, xF, L.uLayerScalar)
	}
}

// begin opens the pass's command buffer and uploads deep's sets, which addDeep adds after each layer.
func (ps *prefillPass) begin() {
	r, d, deep := ps.r, ps.d, ps.deep
	ps.e = r.q.Begin()
	if deep != nil {
		for _, set := range deep.sets {
			ps.deepB = append(ps.deepB, NewBufferFloatsOn(mem.LineScratch, d, set))
		}
		ps.scratch = append(ps.scratch, ps.deepB...) // the deferred release reads scratch when the call returns
	}
}

// layers encodes every layer's steps, then the last DeepStack set. It reports false when an encoder error stops the pass.
func (ps *prefillPass) layers() bool {
	r := ps.r
	for l := 0; l < r.nL; l++ {
		if l > 0 {
			ps.addDeep(l - 1)
		}
		L := &r.layers[l]
		for _, st := range L.steps {
			switch st {
			case stepAttention:
				ps.attnBlock(l, L)
			case stepDeltaNet:
				r.encodePrefillDeltaMixer(ps.e, L, ps.xF, ps.normF, ps.dn, ps.M, ps.Mpad, ps.gemm, ps.uM, ps.m0, ps.m2, ps.dummyBias)
			case stepMamba:
				r.encodePrefillMambaMixer(ps.e, L, ps.xF, ps.normF, ps.mb, ps.M, ps.Mpad, ps.gemmW8, ps.uM, ps.m0, ps.m2, ps.dummyBias)
			case stepMoE:
				if !ps.moeBlock(l, L) {
					return false
				}
			case stepDenseFFN, stepNonGatedFFN:
				ps.ffnBlock(L)
			case stepPLE:
				ps.pleBlock(l, L)
			case stepLayerScale:
				ps.layerScaleBlock(L)
			default:
				panic(fmt.Sprintf("metal: the batched prefill has no block for the %v step, and prefillStepsDecline admits none", st))
			}
		}
	}
	ps.addDeep(r.nL - 1)
	return true
}

// lastRowLogits encodes the final norm and the LM head on the last row, commits and waits for the pass, records whether
// the residual's last row is non-finite (r.prefillResidNonFinite), and returns the last row's logits.
func (ps *prefillPass) lastRowLogits() []float32 {
	r, pf := ps.r, ps.pf
	e := ps.e
	// Final norm and LM head for the last token only, through the same int8-pinned head the decode path runs
	// (rmsnorm to int8, then gemv_w8a8): the head weights are int8 (logit-critical), and the int4 gemm_w4f16 would
	// misread them as packed nibbles and f16 scales, producing NaN logits. Norm-quant the last token's f16
	// residual row to int8, then run the decode head.
	finalNormQ := pf.pRmsQ
	if r.layerNorm {
		finalNormQ = pf.pLNQ
	}
	if r.q4kLane { // the lane's head reads an f32 input (r.pGemvW8 is gemv_w8f32_head there)
		e.Dispatch(pf.pRmsF32Out, tgReduceNorm, tgReduceNorm, ps.xF.At((ps.M-1)*ps.H*2), r.finalNorm, r.aq, ps.uH, ps.uEpsX, r.uAddOne)
	} else {
		e.Dispatch(finalNormQ, tgReduceNorm, tgReduceNorm, ps.xF.At((ps.M-1)*ps.H*2), r.finalNorm, r.aq, r.aSc, ps.uH, ps.uEpsX, r.uAddOne)
	}
	e.Dispatch(r.pGemvW8, ps.V*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	e.End()
	// The int8 head turns a NaN row into finite zeros, which the caller's non-finite-logit check passes, so the
	// residual itself is checked. Its last row is enough: a row that overflows before the final layer writes NaN K/V,
	// which reaches the last row through attention; a row that overflows in the final layer writes no further K/V.
	last := ps.xF.U16s()[(ps.M-1)*ps.H : ps.M*ps.H]
	if r.poisonPrefillResidForTest {
		last[0] = 0x7c00 // +inf
	}
	r.prefillResidNonFinite = false
	for _, h := range last {
		if h&0x7c00 == 0x7c00 {
			r.prefillResidNonFinite = true
			break
		}
	}
	if db02Trace != nil {
		db02Trace(db02Event{layer: r.nL, gpuStart: e.GPUStart(), gpuEnd: e.GPUEnd()})
	}
	r.recordExecErr(e.Err())

	out := make([]float32, ps.V)
	copy(out, r.logits.Floats()[:ps.V])
	// Gemma's final-logit softcap: finalizeLogits (the decode path's entry point) applies it to r.logitsHost, but
	// this pass copies straight out of r.logits into a fresh slice and returns before finalizeLogits runs, so it
	// is applied here too. softcapParallel is a no-op for a non-softcapped family (0).
	if r.finalSoftcap > 0 {
		softcapParallel(out, r.finalSoftcap)
	}
	// Cohere's logit scale (FeatLogitScale), as finalizeLogits applies it for decode.
	if r.logitScale != 0 && r.logitScale != 1 {
		for i, v := range out {
			out[i] = v * r.logitScale
		}
	}
	return out
}
