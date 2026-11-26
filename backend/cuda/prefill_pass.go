//go:build cuda

package cuda

import (
	"context"
	"fmt"
	"slices"
	"sync/atomic"
	"time"

	"github.com/townsendmerino/aikit/linalg"
	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/program"

	"github.com/townsendmerino/goinfer/backend/cuda/internal/launchgeom"
)

// prefillPass is one prefillCore pass's state, shared by its phases: the call's inputs, the scratch it allocates and the
// inputs it uploads. It is cudaResident.pass, reset at the start of each pass and cleared at its end, so a pass allocates
// nothing for it; it is read and written only inside the pass's executor job (prefill is serialized there by doOn), and
// prefillCore copies the outputs out before the job returns.
//
// Every phase is a move of prefillCore's code: the launches, uploads and allocations keep their order
// (TestPrefillLaunch_sequencesHoldTheGolden).
type prefillPass struct {
	r          *cudaResident
	ctx        context.Context
	embeddings [][]float32
	startPos   int
	tail       int
	blocks     [][2]int
	mropePos   [][3]int
	rows       []stepRow
	M          int
	img        bool // the pass's attention is the image-block kernel's (prefillCore's img)
	psFused    func(hd int) bool

	maxQDim, maxKvDim, hidden, inter int

	// scratch is every buffer the pass allocated, released when it ends (release), however it ends.
	scratch []Buffer

	xB, aqB, aScB       Buffer
	qBb, kBb, vBb       Buffer
	cctxB               Buffer
	rag                 *raggedRows // a step's decode and verify rows for the ragged launches
	psScB, psInvB       Buffer      // the row-batched exact attention's scores and inverses (none: no buffer)
	psStride, psRows    int
	cqB, cScB           Buffer
	mqB, mScB           Buffer
	gOb, uOb            Buffer
	dqB, dScB           Buffer
	dnB                 dnPrefillBufs
	sbB                 Buffer
	residMN             uint32
	dsBufs              []Buffer // DeepStack sets of this pass's image rows, one per layer that gets one
	dsOff, dsN          int
	pleB, pleGB         Buffer
	xhost               []float32
	imgBlk              Buffer
	mropeWin            [][3]int // this pass's rows' m-RoPE triples (mropePosWindow), set by prefillCore
	rposT, rposH, rposW Buffer

	outs [][]float32
	ids  []int
}

// reset starts a pass, keeping the free list's and the DeepStack list's backing arrays.
func (p *prefillPass) reset(r *cudaResident, ctx context.Context, embeddings [][]float32, startPos, tail int, blocks [][2]int, mropePos [][3]int, rows []stepRow, img bool, maxQDim, maxKvDim, hidden, inter int) {
	scratch, ds := p.scratch[:0], p.dsBufs[:0]
	*p = prefillPass{r: r, ctx: ctx, embeddings: embeddings, startPos: startPos, tail: tail, blocks: blocks, mropePos: mropePos,
		rows: rows, M: len(embeddings), img: img, maxQDim: maxQDim, maxKvDim: maxKvDim, hidden: hidden, inter: inter,
		scratch: scratch, dsBufs: ds}
}

// release frees the pass's scratch and drops every reference the pass held, so a finished pass keeps no buffer or row alive.
func (p *prefillPass) release() {
	for _, b := range p.scratch {
		p.r.dev.ReleaseBuf(b)
	}
	clear(p.scratch)
	clear(p.dsBufs)
	scratch, ds := p.scratch[:0], p.dsBufs[:0]
	*p = prefillPass{scratch: scratch, dsBufs: ds}
}

// af and ai allocate M-sized scratch of n float32s or int32s, at least one element: a pure-MoE family (qwen3_5_moe's
// intermediate_size=0) has no dense FFN, so its gate/up/down scratch is zero-sized and unused, and a zero-length device
// allocation is refused. Each buffer joins the free list as it is created: allocation panics on OOM (gpu.NewBufferLenOf's
// contract), runJob recovers the panic into a decline, and at M=3000 this is hundreds of MB, so a partial allocation is an
// expected failure on a nearly full card and must not leak.
func (p *prefillPass) af(n int) Buffer {
	if prefillScratchSeenForTest != nil {
		prefillScratchSeenForTest(int64(max(n, 1)) * 4)
	}
	b := p.r.af(max(n, 1))
	p.scratch = append(p.scratch, b)
	return b
}

func (p *prefillPass) ai(n int) Buffer {
	if prefillScratchSeenForTest != nil {
		prefillScratchSeenForTest(int64(max(n, 1)) * 4)
	}
	b := p.r.ai(max(n, 1))
	p.scratch = append(p.scratch, b)
	return b
}

// allocScratch allocates the pass's working buffers: the residual, the attention and FFN operands, a step's ragged rows,
// the row-batched attention's scores, the Gated-DeltaNet rows (starting a fresh recurrence for a prompt from position 0) and
// the sandwich temp.
func (p *prefillPass) allocScratch() error {
	r, M, hidden, inter, maxQDim, maxKvDim := p.r, p.M, p.hidden, p.inter, p.maxQDim, p.maxKvDim
	p.xB = p.af(M * hidden)
	p.aqB, p.aScB = p.ai(M*hidden/4), p.af(M*r.actScaleLen(hidden))
	p.qBb, p.kBb, p.vBb = p.af(M*maxQDim), p.af(M*maxKvDim), p.af(M*maxKvDim)
	p.cctxB = p.af(M * maxQDim)
	if p.rows != nil {
		var e error
		if p.rag, e = r.raggedRowsFor(p.rows[r.stepChunk.rows:], M, p.ai, p.af); e != nil {
			return e
		}
	}
	// The row-batched exact attention's scores and inverses, for a pass with a layer past attn_batched's bound.
	if p.rows == nil {
		if p.psStride, p.psRows = r.prefillSplitKVPlan(M, p.startPos+M, p.psFused, p.blocks); p.psStride > 0 {
			p.psScB, p.psInvB = p.af(p.psRows*r.nH*p.psStride), p.af(p.psRows*r.nH)
		}
	}
	p.cqB, p.cScB = p.ai(M*maxQDim/4), p.af(M*r.actScaleLen(maxQDim))
	p.mqB, p.mScB = p.ai(M*hidden/4), p.af(M*r.actScaleLen(hidden))
	p.gOb, p.uOb = p.af(M*inter), p.af(M*inter)
	p.dqB, p.dScB = p.ai(M*inter/4), p.af(M*r.actScaleLen(inter))
	// Gated-DeltaNet families (qwen3_5 / qwen3_5_moe / qwen3_next): the M-row buffers the recurrence reads row
	// by row, and the softmax layers' double-width q projection and output gate.
	if r.dnet != nil {
		dp := r.dnet
		p.dnB = dnPrefillBufs{
			mix: p.af(M * dp.convDim), bt: p.af(M * dp.nv), at: p.af(M * dp.nv), z: p.af(M * dp.valueDim),
			conv: p.af(M * dp.convDim), qn: p.af(M * dp.keyDim), kn: p.af(M * dp.keyDim), headP: p.af(M * dp.nv * 2),
			core: p.af(M * dp.valueDim), g: p.af(M * dp.valueDim), gq: p.ai(M * dp.valueDim / 4), gsc: p.af(M),
			qg: p.af(M * 2 * maxQDim), gate: p.af(M * maxQDim),
		}
		if p.startPos == 0 {
			// A prompt from position 0 starts a fresh recurrence (not positional: no rewind restores it),
			// as ForwardN's sequential path and decode's Reset do.
			if e := r.resetState(); e != nil {
				return e
			}
		}
	}
	// Sandwich families (Gemma) norm the attention / MLP sublayer output BEFORE adding it to the
	// residual, so the o-proj and down GEMVs write a temp instead of accumulating in place. One
	// [M, hidden] buffer, reused for both (the two uses are sequential).
	if r.sandwich || r.postOnly || r.parallelBlock {
		p.sbB = p.af(M * hidden)
	}
	p.residMN = uint32((M*hidden + 255) / 256)
	return nil
}

// uploadInputs allocates and uploads the pass's inputs: the DeepStack sets of its image rows, the E-model's per-layer
// embedding tails, the embeddings themselves, the image blocks and the m-RoPE positions.
func (p *prefillPass) uploadInputs() error {
	r, M, hidden, startPos := p.r, p.M, p.hidden, p.startPos
	// Qwen3-VL's DeepStack sets for the rows of THIS pass that fall in the image run: uploaded once, added to the residual
	// after each layer (deepstackAdd). A pass holding none of the image run uploads nothing.
	if ds := r.deepPlan; ds != nil {
		if a, b := max(ds.start, startPos), min(ds.start+ds.n, startPos+M); a < b {
			imgOff, imgN := (a-startPos)*hidden, (b-a)*hidden
			p.dsOff, p.dsN = imgOff, imgN
			if deepDefectForTest == deepDefectTextRows {
				p.dsOff, p.dsN = 0, M*hidden
			}
			for _, set := range ds.sets {
				rows := set[(a-ds.start)*hidden : (b-ds.start)*hidden]
				if deepDefectForTest == deepDefectTextRows { // planted defect: the set added to the text rows too (the image rows' values cycled over the whole pass)
					rows = make([]float32, M*hidden)
					for i := range rows {
						rows[i] = set[(a-ds.start)*hidden+i%imgN]
					}
				}
				buf := p.af(p.dsN)
				if e := gpu.Upload(buf, rows); e != nil {
					return e
				}
				p.dsBufs = append(p.dsBufs, buf)
			}
		}
	}
	// Gemma 4 E-model per-layer embedding inputs: pleB holds every row's tail REORDERED to [layer][row][P], so layer l's slice is one contiguous
	// [M, P] operand (glu_quant_batched's `up`, row stride P), and pleGB is the gate GEMV's [M, P] output.
	if r.pleP > 0 {
		P := r.pleP
		pleHost := make([]float32, r.nLayers*M*P)
		for m, e := range p.embeddings {
			for l := 0; l < r.nLayers; l++ {
				copy(pleHost[(l*M+m)*P:(l*M+m+1)*P], e[hidden+l*P:hidden+(l+1)*P])
			}
		}
		p.pleB, p.pleGB = p.af(len(pleHost)), p.af(M*P)
		if e := gpu.Upload(p.pleB, pleHost); e != nil {
			return e
		}
	}

	// Upload the M embeddings contiguously as xB[M, hidden] (already FeatEmbedScale-scaled by the
	// caller, exactly as Forward/ForwardNoLogits receive them).
	if cap(r.xhostBuf) < M*hidden {
		r.xhostBuf = make([]float32, M*hidden) // grown to the widest pass and kept: a fresh block per pass left the prompt's size in garbage
	}
	p.xhost = r.xhostBuf[:M*hidden]
	for m, e := range p.embeddings {
		n := copy(p.xhost[m*hidden:(m+1)*hidden], e)
		clear(p.xhost[m*hidden+n : (m+1)*hidden])
	}
	if e := gpu.Upload(p.xB, p.xhost); e != nil {
		return e
	}

	// Every image block's [start, end), uploaded once for every layer's attn_img_batched launch.
	if p.img {
		flat := make([]int32, 0, 2*len(p.blocks))
		for _, b := range p.blocks {
			flat = append(flat, int32(b[0]), int32(b[1]))
		}
		p.imgBlk = p.ai(len(flat))
		if len(flat) > 0 {
			if e := gpu.Upload(p.imgBlk, flat); e != nil {
				return e
			}
		}
	}
	// m-RoPE: this pass's per-row (t,h,w) rotation triples, uploaded once and reused by every layer's
	// rope_kv_mrope_batched launch.
	if p.mropePos != nil {
		pt, ph, pw := make([]int32, M), make([]int32, M), make([]int32, M)
		for i, q := range p.mropeWin {
			pt[i], ph[i], pw[i] = int32(q[0]), int32(q[1]), int32(q[2])
		}
		p.rposT, p.rposH, p.rposW = p.ai(M), p.ai(M), p.ai(M)
		if e := gpu.Upload(p.rposT, pt); e != nil {
			return e
		}
		if e := gpu.Upload(p.rposH, ph); e != nil {
			return e
		}
		if e := gpu.Upload(p.rposW, pw); e != nil {
			return e
		}
	}
	return nil
}

// layer runs layer l of the pass: the layer's batched steps (cudaPrefillStepsOf, OP4cu-d1), the DeltaNet rows or the attention
// half, then the MoE or the dense FFN, each the method its kind chose; then the DeepStack add and the batched hidden-state capture.
func (p *prefillPass) layer(l int) error {
	r, inter := p.r, p.inter
	Ly := &r.layers[l]
	// PER-LAYER, read here and bound below — never L0's hoisted and assumed uniform.
	hd, nKV, qDim, rhalf := Ly.hd, Ly.nKV, Ly.qDim, Ly.rhalf
	// A KV-shared layer rotates Q only (nKV=0 leaves rope_kv's K and V work empty, so nothing is stored through the alias into its source's cache), as
	// decodeAttnGap does. The attention launch keeps the real nKV, which is how it reads the source's cache.
	ropeNKV := nKV
	if Ly.kvShared && !g4KeepSharedKVStoreForTest {
		ropeNKV = 0
	}
	ropeN := r.nH*rhalf + ropeNKV*rhalf + ropeNKV*(hd-2*rhalf)
	ffn := inter
	if r.eModel {
		ffn = Ly.ffnI
	}
	qb, kb, vb := ArgNull(), ArgNull(), ArgNull()
	if Ly.hasBias {
		qb, kb, vb = Arg(Ly.qb), Arg(Ly.kb), Arg(Ly.vb)
	}
	g := &prefillLayerGeom{hd: hd, nKV: nKV, qDim: qDim, rhalf: rhalf, ropeNKV: ropeNKV, ropeN: ropeN, ffn: ffn, qb: qb, kb: kb, vb: vb}
	for _, pst := range r.prefillLayerSteps(Ly) {
		var e error
		switch pst {
		case pstepDeltaNet:
			e = p.stepDeltaNet(l, Ly)
		case pstepAttn:
			e = p.stepAttn(l, Ly, g)
		case pstepMoE:
			e = p.stepMoE(l, Ly)
		case pstepDenseFFN:
			e = p.stepDenseFFN(l, Ly, g)
		}
		if e != nil {
			return e
		}
	}
	if e := p.deepstackAdd(l); e != nil {
		return e
	}
	return p.captureTaps(l)
}

// prefillLayerGeom is one layer's geometry as the pass's steps read it: per layer, never layer 0's hoisted and assumed uniform.
type prefillLayerGeom struct {
	hd, nKV, qDim, rhalf int
	ropeNKV, ropeN       int // the rotated K/V heads (0 on a KV-shared layer) and rope_kv's element count
	ffn                  int // the dense FFN width
	qb, kb, vb           KernelArg
}

// stepDeltaNet is the Gated-DeltaNet mixer's rows.
func (p *prefillPass) stepDeltaNet(l int, Ly *cudaLayer) error {
	r, M, xB, aqB, aScB, dnB := p.r, p.M, p.xB, p.aqB, p.aScB, p.dnB
	// Gated-DeltaNet mixer: replaces segA + rope +
	// attention + o-proj, as deltaNetMixer does in decode — the projections batched over the M rows,
	// the recurrence row by row in order. The FFN below is the shared one.
	pend, pM := r.specPending(l)
	if e := r.prefillDeltaNetRows(Ly, xB, aqB, aScB, M, r.specBufs(l, dnB), pend, pM); e != nil { // profiles its own categories
		return e
	}
	return nil
}

// stepAttn is the attention half: segA (input norm and quant, the Q/K/V GEMVs, the per-head norms), the rope, KV store and
// attention (ordAttn, or a step's rows through decode's gap), then segB (ctx quant and the o-projection into the residual).
func (p *prefillPass) stepAttn(l int, Ly *cudaLayer, g *prefillLayerGeom) error {
	r, M, startPos, tail, rows, hidden, xB, aqB, aScB, qBb, kBb, vBb, cctxB, rag, cqB, cScB, dnB, sbB, residMN := p.r, p.M, p.startPos, p.tail, p.rows, p.hidden, p.xB, p.aqB, p.aScB, p.qBb, p.kBb, p.vBb, p.cctxB, p.rag, p.cqB, p.cScB, p.dnB, p.sbB, p.residMN
	hd, nKV, qDim, qb, kb, vb := g.hd, g.nKV, g.qDim, g.qb, g.kb, g.vb
	var t time.Time // the category timer (profTic): each step sets it before it reads it
	// segA: input norm+quant (glue), then Q/K/V GEMVs. Category timers (r.prof) sync r.stream at
	// each group boundary; nil in production, so the launch sequence is otherwise unchanged.
	aw := &Ly.walk.Attn
	t = r.profTic()
	if e := r.emitInputNormB(aw.InputForm, xB, Ly.preNorm, hidden, aqB, aScB, M); e != nil {
		return e
	}
	r.profToc(glueCat, t)
	t = r.profTic()
	if aw.Gated {
		// attn_output_gate (qwen3_5 softmax layers): q_proj emits [query ‖ gate] per head at double
		// width. Decode's split, one launch over all M rows: M rows are M·nH contiguous heads.
		if e := r.bGemvB(Ly.q, aqB, aScB, qb, dnB.qg, M, 0); e != nil {
			return e
		}
		if e := r.launch(r.dnQSplit, g1cfg(M*qDim, 256),
			Arg(dnB.qg), Arg(qBb), Arg(dnB.gate),
			gpu.ArgValue(int32(M*qDim)), gpu.ArgValue(int32(hd))); e != nil {
			return e
		}
	} else if e := r.bGemvB(Ly.q, aqB, aScB, qb, qBb, M, 0); e != nil {
		return e
	}
	if Ly.kvShared {
		// No k_proj and no v_proj: K/V are the source layer's, already in the aliased cache.
	} else if e := r.bGemvB(Ly.k, aqB, aScB, kb, kBb, M, 0); e != nil {
		return e
	} else if Ly.kEqV {
		// K=V (Gemma 4 global layers): no v_proj. V is v_norm(the RAW pre-RoPE k_proj output), so copy the k projection into the
		// V buffer and normalize it below, before rope_kv_batched rotates k, as segA's decode path does. kBb and vBb are [M,
		// kvDim] row-major, so one contiguous copy of M*kvDim floats moves every row.
		if e := r.copyF32(kBb, vBb, M*Ly.kvDim); e != nil {
			return e
		}
	} else if e := r.bGemvB(Ly.v, aqB, aScB, vb, vBb, M, 0); e != nil {
		return e
	}
	r.profToc(gemvCat, t)
	// per-head Q/K RMSNorm BEFORE rope (Qwen3): in place on qBb/kBb, one block per (head,token).
	// Bit-identical to the decode qk_norm applied per token (same f64 reduction, same addOne).
	if aw.QKNorm {
		t = r.profTic()
		addOne := int32(0)
		if aw.QKNormParams.AddOne {
			addOne = 1
		}
		qkNH, qkNKV, qkHD := r.nH, nKV, hd
		if aw.QKNormParams.Whole {
			qkNH, qkNKV, qkHD = 1, 1, r.nH*hd
		}
		kNormW := Ly.kNorm
		if Ly.kvShared {
			// Q only: nKV=0 launches no K blocks, and a shared layer has no k_norm weight to bind (an empty Buffer is a nil deref).
			qkNKV, kNormW = 0, Ly.qNorm
		}
		if e := r.launch(r.bQKN, LaunchConfig{GridX: uint32(qkNH + qkNKV), GridY: uint32(M), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
			Arg(qBb), Arg(kBb), Arg(Ly.qNorm), Arg(kNormW),
			gpu.ArgValue(int32(qkNH)), gpu.ArgValue(int32(qkNKV)), gpu.ArgValue(int32(qkHD)),
			gpu.ArgValue(r.eps), gpu.ArgValue(addOne), gpu.ArgValue(int32(M))); e != nil {
			return e
		}
		r.profToc(glueCat, t)
	}
	if aw.G4VNorm && (Ly.kEqV || !g4DropVNormForTest) {
		// Scale-less v_norm over vBb (the raw k on a K=V layer, the v_proj output otherwise), BEFORE rope rotates k — segA's
		// decode launch with an M dimension added. nH=0 makes qk_norm_batched treat every
		// block as a K-head (base = v + m*kvDim + h*hd), and vNormUnit is a unit weight so
		// addOne=0 gives a pure RMS scale with no learned gain.
		t = r.profTic()
		if e := r.launch(r.bQKN, LaunchConfig{GridX: uint32(nKV), GridY: uint32(M), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
			Arg(vBb), Arg(vBb), Arg(r.vNormUnit), Arg(r.vNormUnit),
			gpu.ArgValue(int32(0)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(hd)),
			gpu.ArgValue(r.eps), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(M))); e != nil {
			return e
		}
		r.profToc(glueCat, t)
	}
	if rows != nil {
		// Multi-sequence step: rope, the KV store and attention per sequence through decode's own gap, each row on its own slot
		// at its own position; a prefill chunk's rows (the first) through its own pass's attention, in its slot and class.
		// The batched stack resumes at the ctx-quant below. The chunk's attention times itself (glue, then attnCat).
		if c := r.stepChunk; c.rows > 0 {
			exact, tab := r.forceExactKernels, r.kvTabSlot
			r.forceExactKernels, r.kvTabSlot = c.exact, c.slot
			e := p.ordAttn(l, Ly, g, c.pos, c.rows, tailKVOnly)
			r.forceExactKernels, r.kvTabSlot = exact, tab
			if e != nil {
				return e
			}
		}
		t = r.profTic()
		if e := r.stepAttnRows(Ly, l, rows, r.stepChunk.rows, qBb, kBb, vBb, cctxB, rag); e != nil {
			return e
		}
		r.profToc(attnCat, t)
	} else if e := p.ordAttn(l, Ly, g, startPos, M, tail); e != nil {
		return e
	}
	// segB: ctx-quant (glue), o-proj (gemv, accum into residual), MLP.
	t = r.profTic()
	if aw.Gated { // ctx *= sigmoid(gate), before o_proj — segB's launch over all M rows
		if e := r.launch(r.dnAttnGate, g1cfg(M*qDim, 256),
			Arg(cctxB), Arg(dnB.gate), gpu.ArgValue(int32(M*qDim))); e != nil {
			return e
		}
	}
	if e := r.launch(r.bQuant, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
		Arg(cctxB), gpu.ArgValue(int32(qDim)), Arg(cqB), Arg(cScB), gpu.ArgValue(int32(M))); e != nil {
		return e
	}
	r.profToc(glueCat, t)
	t = r.profTic()
	if Ly.attnOutUnfused() {
		// o-proj → temp (accum=0), optional post-attn RMSNorm per row, then add to residual.
		if e := r.bGemvB(Ly.o, cqB, cScB, r.oBiasArg(Ly), sbB, M, 0); e != nil {
			return e
		}
		if aw.OutMode == program.OutPostNormResidual {
			if e := r.bNormF32B(sbB, Ly.postAttnNorm, hidden, M); e != nil {
				return e
			}
		}
		if e := r.launch(r.bRes, LaunchConfig{GridX: residMN, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(xB), Arg(sbB), gpu.ArgValue(int32(M*hidden))); e != nil {
			return e
		}
	} else if e := r.bGemvB(Ly.o, cqB, cScB, r.oBiasArg(Ly), xB, M, 1); e != nil {
		return e
	}
	r.profToc(gemvCat, t)
	return nil
}

// ordAttn is an ordinary pass's rope, KV store and attention over rows [0, M) at startPos in the bound slot's tables.
func (p *prefillPass) ordAttn(l int, Ly *cudaLayer, g *prefillLayerGeom, startPos, M, tail int) error {
	r, blocks, mropePos, img, qBb, kBb, vBb, cctxB, psScB, psInvB, psStride, psRows, imgBlk, rposT, rposH, rposW := p.r, p.blocks, p.mropePos, p.img, p.qBb, p.kBb, p.vBb, p.cctxB, p.psScB, p.psInvB, p.psStride, p.psRows, p.imgBlk, p.rposT, p.rposH, p.rposW
	hd, nKV, rhalf, ropeNKV, ropeN := g.hd, g.nKV, g.rhalf, g.ropeNKV, g.ropeN
	// rope + kv-store (glue): token m at absolute position startPos+m; rotates q/k, writes K/V. Its own timer: a
	// step times the chunk's attention inside its own attnCat span.
	t := r.profTic()
	ropeCfg := LaunchConfig{GridX: uint32((ropeN + 255) / 256), GridY: uint32(M), GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
	ropeArgs := []gpu.KernelArg{
		Arg(qBb), Arg(kBb), Arg(vBb), Arg(Ly.invF), Arg(r.kvK(l)), Arg(r.kvV(l)), Arg(r.kvT(l)),
		gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(ropeNKV)), gpu.ArgValue(int32(hd)),
		gpu.ArgValue(int32(startPos)), gpu.ArgValue(int32(rhalf)), gpu.ArgValue(int32(M)),
		gpu.ArgValue(Ly.mscale),
	}
	var ropeErr error
	if mropePos != nil {
		mropeArgs := append(append([]gpu.KernelArg{}, ropeArgs...),
			Arg(rposT), Arg(rposH), Arg(rposW), gpu.ArgValue(r.mropeSec0), gpu.ArgValue(r.mropeSec1))
		if r.mropeTakesMode { // rope_kv_mrope_batched takes the layout mode; the pairwise twin has no such argument
			mropeArgs = append(mropeArgs, gpu.ArgValue(r.mropeMode))
		}
		ropeErr = r.launch(r.bRopeKVMRoPE, ropeCfg, mropeArgs...)
	} else {
		// qTempRows (Ministral 3, FeatAttnTemp): this launch covers M rows at different positions, so each row's post-RoPE query
		// scale comes from a host-built table using launchToken's own float64 expression, which keeps a batched row
		// bit-identical to decode. Not carried into the m-RoPE branch above: rope_kv_mrope_batched has no attention temperature,
		// and no family needs both.
		qTemp, qe := r.attnTempRows(startPos, M)
		if qe != nil {
			return qe
		}
		ropeErr = r.launch(r.bRopeKV, ropeCfg, append(append([]gpu.KernelArg{}, ropeArgs...), qTemp)...)
	}
	if ropeErr == nil {
		ropeErr = r.kvStore8(l, kBb, vBb, ropeNKV, hd, startPos, M)
	}
	if ropeErr != nil {
		return ropeErr
	}
	r.profToc(glueCat, t)
	// causal + per-row sliding-window attention; block 128 matches the M=1 attention reduce.
	maxNWin := startPos + M
	if Ly.window > 0 && int(Ly.window) < maxNWin {
		maxNWin = int(Ly.window)
	}
	if len(blocks) > 0 {
		// Widened by launchgeom.ImgBlocksMaxNWin, the formula checkPrefillShmemImgBlocks uses; the two must not drift, or this allocation
		// under-sizes the launch it covers.
		maxNWin = launchgeom.ImgBlocksMaxNWin(maxNWin, int(Ly.window), blocks)
	}
	t = r.profTic()
	// L2: the fused kernel when it serves this (hd, M), else attn_batched (see useAttnFused for the selection rule). The
	// argument list is identical for all of them, so it is built once; the launches differ only in pipeline, grid and shared
	// memory. Each names its pipeline field directly (see useAttnFused for why a local variable will not do).
	attnArgs := []gpu.KernelArg{
		Arg(qBb), Arg(r.kvK(l)), Arg(r.kvV(l)), Arg(r.kvT(l)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(nKV)),
		gpu.ArgValue(int32(hd)), gpu.ArgValue(int32(startPos)), gpu.ArgValue(r.attnScale),
		// r.sinkArg(l), not ArgNull(): decode threads the gpt-oss learned sink through, and this launch must not depend on a
		// different check declining gpt-oss.
		gpu.ArgValue(Ly.window), gpu.ArgValue(int32(M)), Arg(cctxB), r.sinkArg(l),
	}
	var attnErr error
	// Flash-decode lane for speculative verify (attn-decode-fa-verify-PREREGISTERED.md): rows [laneFrom, M) of an all-rows
	// verify batch are served by the multi-row lane, bit-identical to the M=1 lane at each row's position
	// (TestFlashDecodeRowsBitIdentical), so a verified position scores as plain lane decode scores it. Rows below laneFrom
	// are under the attended-span floor and take the exact path, as their M=1 decode would. laneFrom == M means no lane rows
	// (the common case); laneFrom == 0 skips the exact launch.
	laneFrom := r.verifyLaneFrom(l, startPos, M, tail, img)
	// Image block first, unconditionally, before useAttnFused is consulted: attn_fused's tile-level aggregates assume
	// monotonic per-row nKeys across a 64-row tile, which an image block breaks, so the exact attn_img_batched is used
	// instead. Checking the block after useAttnFused would silently route a long image prompt through the incompatible fused
	// kernel.
	if laneFrom == 0 {
		// every row is a lane row: the exact attention launch is skipped and the lane below writes the whole context buffer
	} else if img {
		if psStride > 0 && r.prefillSplitKVServes(maxNWin) {
			if r.spanPass && Ly.window > 0 {
				// the row-batched kernels keep Gemma 3's window rule, which drops a span row's reach-back to its span start
				return fmt.Errorf("cuda prefill: layer %d (sliding) span attention at %d keys needs the row-batched kernels: %w", l, maxNWin, errPrefillDeclined)
			}
			// past attn_img_batched's shared-memory bound: the row-batched kernels, with the same block table
			attnErr = r.prefillSplitKVAttn(l, startPos, M, qBb, cctxB, psScB, psInvB, psStride, psRows, imgBlk, len(blocks))
		} else {
			imgArgs := append(append([]gpu.KernelArg{}, attnArgs...),
				Arg(imgBlk), gpu.ArgValue(int32(len(blocks))))
			imgCfg := LaunchConfig{GridX: uint32(r.nH), GridY: uint32(M), GridZ: 1,
				BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((maxNWin + 128) * 4)}
			if r.spanPass && spanDefectForTest != spanDefectNoReachBack {
				attnErr = r.launch(r.bAttnSpan, imgCfg, imgArgs...)
			} else {
				attnErr = r.launch(r.bAttnImg, imgCfg, imgArgs...)
			}
		}
	} else if fsh, use := r.useAttnFused(hd, M); use {
		fcfg := LaunchConfig{GridX: uint32(r.nH),
			GridY: uint32((M + attnFusedBM - 1) / attnFusedBM), GridZ: 1,
			BlockX: attnFusedThreads, BlockY: 1, BlockZ: 1, SharedMemBytes: fsh}
		// Default tile (attn-fused-tile128-default-PREREGISTERED.md): hd128 layers with no sliding window run the 128-row-tile
		// kernel, bit-identical to the 64x64 one for window == 0. hd64 stays 64x64, and so do windowed layers: their key-tile
		// grouping starts at the block's first row, so a taller block is not bit-identical there.
		// GOINFER_CUDA_ATTN_FUSED_TILE=64x64 forces 64x64.
		tile := r.attnTile
		if tile == 0 && hd == 128 && Ly.window <= 0 && r.bAttnBM128hd128 != (Pipeline{}) {
			tile = 3
		}
		if tile > 0 {
			// Diagnostic query-tile arms (attn-fused-tile-PREREGISTERED.md): 32 / 32 / 128 rows. Name each pipeline at its launch:
			// TestPipelineLint_boundKernelsAreLaunched keys on that.
			bm, bn := 32, 64
			if tile == 2 {
				bn = 32
			} else if tile == 3 {
				bm = 128
			}
			fcfg.GridY = uint32((M + bm - 1) / bm)
			fcfg.BlockX = uint32(bm / 16 * 32)
			fcfg.SharedMemBytes = uint32(2 * (bn*(hd+attnFusedKPAD) + hd*(bn+attnFusedKPAD)))
			switch {
			case tile == 1 && hd == 64:
				attnErr = r.launch(r.bAttnBM32x64hd64, fcfg, r.kvScaled(l, attnArgs...)...)
			case tile == 1:
				attnErr = r.launch(r.bAttnBM32x64hd128, fcfg, r.kvScaled(l, attnArgs...)...)
			case tile == 2 && hd == 64:
				attnErr = r.launch(r.bAttnBM32x32hd64, fcfg, r.kvScaled(l, attnArgs...)...)
			case tile == 2:
				attnErr = r.launch(r.bAttnBM32x32hd128, fcfg, r.kvScaled(l, attnArgs...)...)
			case hd == 64:
				attnErr = r.launch(r.bAttnBM128hd64, fcfg, r.kvScaled(l, attnArgs...)...)
			default:
				attnErr = r.launch(r.bAttnBM128hd128, fcfg, r.kvScaled(l, attnArgs...)...)
			}
			r.fastAttnLaunches++
			if tile == 3 {
				r.tile128Launches++
			}
		} else if hd == 64 {
			attnErr = r.launch(r.bAttnFused64, fcfg, r.kvScaled(l, attnArgs...)...)
			r.fastAttnLaunches++
		} else {
			attnErr = r.launch(r.bAttnFused128, fcfg, r.kvScaled(l, attnArgs...)...)
			r.fastAttnLaunches++
		}
	} else if psStride > 0 && r.prefillSplitKVServes(maxNWin) {
		attnErr = r.prefillSplitKVAttn(l, startPos, M, qBb, cctxB, psScB, psInvB, psStride, psRows, Buffer{}, 0)
	} else {
		attnErr = r.launch(r.bAttn, LaunchConfig{GridX: uint32(r.nH), GridY: uint32(M), GridZ: 1,
			BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((maxNWin + 128) * 4)}, r.kvScaled(l, r.withCap(attnArgs...)...)...)
		r.exactAttnLaunches++
	}
	if attnErr == nil && laneFrom < M {
		attnErr = r.flashVerifyAttn(l, startPos, laneFrom, M, qBb, cctxB)
	}
	if attnErr != nil {
		return attnErr
	}
	r.profToc(attnCat, t)
	return nil
}

// stepMoE is the routed FFN. FFN: dense batches; MoE runs row by row off the batched residual. The routed-expert GEMVs are indexed by a device-side
// routing decision that differs per token, so there is no M-wide form without an expert-major gather (the expert-major
// path below); the batched attention half still reads its weights once per pass. gpu.Buffer.At gives a zero-copy
// sub-view, so row m of the batched residual is a valid single-row residual for the per-token FFN chain (segBFFN,
// layerTail, segC: decode's calls in decode's order, including the g4x2 accumulator clear and the C' routed-expert DMA).
//
// gC=false: prefill never replays captured graphs (a graph bakes r.x, and these rows are not r.x). layerTail also
// carries per-token debug seams (hidCapTaps, layerCap) that would fire M times per layer here; prefillStaticDecline
// refuses a model with either armed.
func (p *prefillPass) stepMoE(l int, Ly *cudaLayer) error {
	r, ctx, M, hidden, xB := p.r, p.ctx, p.M, p.hidden, p.xB
	t := r.profTic()
	g4moe := Ly.walk.FFN.Kind == program.SublayerGemma4MoE
	// Expert-major MoE (moe_expert_major.go, moe_expert_major_gemma4.go;
	// docs/measurements/p20-expert-locality-2026-09-21.md): route, bucket and admit once per distinct expert instead of once
	// per (row, rank), when eligible. Checked once per layer, so ineligible layers pay one field-and-map read.
	if !g4moe && r.prefillMoEExpertMajorEligible(Ly) {
		if e := r.prefillMoEExpertMajorRun(ctx, Ly, xB, M, hidden); e != nil {
			return e
		}
	} else if g4moe && r.prefillGemma4ExpertMajorEligible(Ly) {
		if e := r.prefillGemma4ExpertMajorRun(ctx, Ly, xB, M, hidden); e != nil {
			return e
		}
	} else {
		for m := range M {
			// Between rows: a MoE chunk is M sequential per-token FFNs, so without this check a cancelled chunk still runs all of
			// them. Checking per row keeps cancellation at roughly one token's granularity.
			if e := ctx.Err(); e != nil {
				return e
			}
			xm := xB.At(m * hidden * 4)
			if e := r.segBFFN(Ly, l, xm); e != nil {
				return e
			}
			if e := r.layerTail(Ly, l, false, xm); e != nil {
				return e
			}
		}
	}
	r.profToc(moeCat, t)
	return nil
}

// stepDenseFFN is the dense FFN, with the E-model's per-layer embedding branch and the per-layer output scalar.
func (p *prefillPass) stepDenseFFN(l int, Ly *cudaLayer, g *prefillLayerGeom) error {
	r, M, hidden, xB, aqB, aScB, mqB, mScB, gOb, uOb, dqB, dScB, sbB, residMN, pleB, pleGB := p.r, p.M, p.hidden, p.xB, p.aqB, p.aScB, p.mqB, p.mScB, p.gOb, p.uOb, p.dqB, p.dScB, p.sbB, p.residMN, p.pleB, p.pleGB
	ffn := g.ffn
	var t time.Time // the category timer (profTic): each step sets it before it reads it
	fw := &Ly.walk.FFN
	t = r.profTic()
	gqB, gScB := mqB, mScB
	switch fw.InputMode {
	case program.FFNInShared:
		// Parallel block (Cohere/Command-R): reuse segA's shared input norm (aqB/aScB) — the MLP consumes
		// the same normed activation already computed, with zero norm dispatch.
		gqB, gScB = aqB, aScB
	default:
		if e := r.emitInputNormB(fw.InputNorm.Form, xB, Ly.postNorm, hidden, mqB, mScB, M); e != nil {
			return e
		}
	}
	r.profToc(glueCat, t)
	t = r.profTic()
	if e := r.bGemvB(Ly.g, gqB, gScB, ArgNull(), gOb, M, 0); e != nil {
		return e
	}
	if e := r.bGemvB(Ly.u, gqB, gScB, ArgNull(), uOb, M, 0); e != nil {
		return e
	}
	r.profToc(gemvCat, t)
	t = r.profTic()
	if e := r.launch(r.bSw, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
		Arg(gOb), Arg(uOb), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(ffn)),
		gpu.ArgValue(int32(fw.FFN.Act)), Arg(dqB), Arg(dScB), gpu.ArgValue(int32(M))); e != nil {
		return e
	}
	r.profToc(glueCat, t)
	t = r.profTic()
	if fw.OutMode != program.OutFusedResidual {
		// down → temp (accum=0), optional post-MLP RMSNorm per row, then add to residual.
		if e := r.bGemvB(Ly.d, dqB, dScB, ArgNull(), sbB, M, 0); e != nil {
			return e
		}
		if fw.OutMode == program.OutPostNormResidual {
			if e := r.bNormF32B(sbB, Ly.postMLPNorm, hidden, M); e != nil {
				return e
			}
		}
		if e := r.launch(r.bRes, LaunchConfig{GridX: residMN, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(xB), Arg(sbB), gpu.ArgValue(int32(M*hidden))); e != nil {
			return e
		}
	} else if e := r.bGemvB(Ly.d, dqB, dScB, ArgNull(), xB, M, 1); e != nil {
		return e
	}
	if Ly.walk.HasPLE && !g4SkipPLEForTest {
		// Gemma 4 E-model per-layer embedding branch, segBFFN's own order over M rows (cuda/resident.go): quantize the RAW residual, the gate GEMV,
		// geluTanh(gate) * this layer's slice of the row tails (glu_quant_batched act 0 with pleB's contiguous [M, P] slice as `up`), the projection
		// GEMV, the post-norm, the residual add. All existing batched kernels. A pleLayerShiftForTest defect reads another layer's slice (modulo
		// the layer count here: decode reads past the row, which has no batched twin).
		P, pl := r.pleP, ((l+pleLayerShiftForTest)%r.nLayers+r.nLayers)%r.nLayers
		if e := r.launch(r.bQuant, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
			Arg(xB), gpu.ArgValue(int32(hidden)), Arg(mqB), Arg(mScB), gpu.ArgValue(int32(M))); e != nil {
			return e
		}
		if e := r.bGemvB(Ly.pleGate, mqB, mScB, ArgNull(), pleGB, M, 0); e != nil {
			return e
		}
		if e := r.launch(r.bSw, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
			Arg(pleGB), Arg(pleB.At(pl*M*P*4)), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(P)),
			gpu.ArgValue(int32(0)), Arg(dqB), Arg(dScB), gpu.ArgValue(int32(M))); e != nil {
			return e
		}
		if e := r.bGemvB(Ly.pleProj, dqB, dScB, ArgNull(), sbB, M, 0); e != nil {
			return e
		}
		if e := r.bNormF32B(sbB, Ly.postPLENorm, hidden, M); e != nil {
			return e
		}
		if e := r.launch(r.bRes, LaunchConfig{GridX: residMN, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(xB), Arg(sbB), gpu.ArgValue(int32(M*hidden))); e != nil {
			return e
		}
	}
	// Per-layer output scalar (Gemma 4 dense), segB's decode twin. fScaleVec is a pure elementwise dst[i]*=s, so it batches
	// over the flattened M*hidden buffer in one launch, like bRes above. TestPrefillNonUniform_bitIdentical exists to catch
	// batched prefill omitting it while decode has it.
	if Ly.walk.HasLayerScale && Ly.layerScalar != 0 && !g4DropLayerScalarForTest {
		if e := r.launch(r.fScaleVec, LaunchConfig{GridX: residMN, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(xB), gpu.ArgValue(Ly.layerScalar), gpu.ArgValue(int32(M*hidden))); e != nil {
			return e
		}
	}
	r.profToc(gemvCat, t)
	return nil
}

// deepstackAdd adds DeepStack set l to the image rows after decoder layer l.
func (p *prefillPass) deepstackAdd(l int) error {
	r, xB, dsBufs, dsOff, dsN := p.r, p.xB, p.dsBufs, p.dsOff, p.dsN
	// DeepStack: after decoder layer l, set l is added to the image rows, as the CPU prefill does (addDeepstack; HF's
	// hidden_states[visual_pos_masks] += deepstack_visual_embeds[l]).
	if dsN > 0 && deepDefectForTest != deepDefectNotAdded {
		k := l
		if deepDefectForTest == deepDefectOneLayerLate {
			k = l - 1 // set l-1 added after layer l: every set one layer late, set 0 never early
		}
		if k >= 0 && k < len(dsBufs) {
			if e := r.launch(r.bRes, LaunchConfig{GridX: uint32((dsN + 255) / 256), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
				Arg(xB.At(dsOff*4)), Arg(dsBufs[k]), gpu.ArgValue(int32(dsN))); e != nil {
				return e
			}
		}
	}
	return nil
}

// captureTaps downloads the residual of all M rows for each block-drafter tap at layer l.
func (p *prefillPass) captureTaps(l int) error {
	r, M, hidden, xB := p.r, p.M, p.hidden, p.xB
	// Batched hidden-state capture (block drafter): the per-token seam (capVec) syncs and downloads once per tap per token.
	// Here the residual for all M rows is already in xB, so one download per tap covers the whole block.
	if len(r.capBTaps) > 0 {
		for slot, tap := range r.capBTaps {
			if tap != l {
				continue
			}
			if e := r.stream.Sync(); e != nil {
				return e
			}
			buf := make([]float32, M*hidden)
			if e := gpu.Download(xB, buf); e != nil {
				return e
			}
			r.capBOut[slot] = buf
			break
		}
	}
	return nil
}

// tailResidualAll ends a tailResidualAll pass: every row's residual, downloaded.
func (p *prefillPass) tailResidualAll() error {
	r, M, hidden, xB, xhost := p.r, p.M, p.hidden, p.xB, p.xhost
	// The only tail that reads the whole residual on the host: the argmax and all-logits heads read xB on the device, and
	// the last-row tails copy one row device-to-device. Downloading all M rows for them was pageable D2H after every pass
	// for nothing.
	if e := gpu.Download(xB, xhost); e != nil {
		return e
	}
	p.outs = make([][]float32, M)
	for m := range p.outs {
		p.outs[m] = slices.Clone(xhost[m*hidden : (m+1)*hidden])
	}
	return r.launchErr
}

// tailAllArgmax ends a tailAllArgmax pass: the batched head's argmax of every row (batchedHeadArgmax).
func (p *prefillPass) tailAllArgmax() error {
	return p.r.batchedHeadArgmax(p.xB, p.aqB, p.aScB, p.M, &p.ids)
}

// tailAllLogits ends a tailAllLogits pass: the logits of every row whose logits are read.
func (p *prefillPass) tailAllLogits() error {
	r, M, rows, xB, aqB, aScB := p.r, p.M, p.rows, p.xB, p.aqB, p.aScB
	// Every row whose logits are read needs the head (PrefillLastN, speculative verify, a step's rows; not a ragged step's
	// prefill chunk but its final pass's last row, raggedChunk.headFrom): batch it, ONE norm+quant+GEMV over those rows,
	// bit-identical to the per-row GEMV by construction (bGemvB). batchedHeadFull says what stays per row.
	var e error
	if p.outs, p.ids, e = r.batchedHeadFull(xB, aqB, aScB, M, rows, r.stepChunk.headFrom()); e != nil {
		return e
	}
	return r.launchErr
}

// tailLastRow ends a tailLastLogits or tailHiddenLast pass: the last row's logits, or its post-final-norm hidden state.
func (p *prefillPass) tailLastRow() error {
	r, M, hidden, tail, rows, xB := p.r, p.M, p.hidden, p.tail, p.rows, p.xB
	p.outs = make([][]float32, M)
	// tailLastLogits / tailHiddenLast: only the last row's output is used, so batching the head would compute M-1 rows
	// nobody reads.
	for m := M - 1; m < M; m++ {
		// The last row goes xB -> r.x on the device, the same bytes the host round trip (Download all, Upload one row) produced. A launch on r.stream
		// like every other op here, so it is ordered after the layer stack with no extra sync.
		if e := r.copyF32(xB.At(m*hidden*4), r.x, hidden); e != nil {
			return e
		}
		if e := r.norm(r.x, r.finalNorm, r.aq, r.aSc); e != nil {
			return e
		}
		if tail == tailHiddenLast {
			// No head dispatch: r.aq/r.aSc (just written by the norm above) already hold what the head would read, the quantized
			// post-final-norm hidden state. Download and dequantize it instead.
			if e := r.stream.Sync(); e != nil {
				return e
			}
			q := make([]int8, hidden)
			if e := gpu.Download(r.aq, q); e != nil {
				return e
			}
			sc := make([]float32, r.actScaleLen(hidden))
			if e := gpu.Download(r.aSc, sc); e != nil {
				return e
			}
			row := make([]float32, hidden)
			if r.actG32 {
				// One scale a group of 32 (the scale row's first half; the group sums follow).
				for g := range hidden / 32 {
					linalg.DequantizeRowInt8(q[32*g:32*g+32], sc[g], row[32*g:32*g+32])
				}
			} else {
				linalg.DequantizeRowInt8(q, sc[0], row)
			}
			p.outs[m] = row
			continue
		}
		if e := r.doG(r.lmW, r.aq, r.aSc, ArgNull(), r.logits, 0); e != nil {
			return e
		}
		if rows != nil && rows[m].draw != nil {
			// A step row whose token is drawn on-device: ForwardSample's own pick over this row's logits.
			if p.ids == nil {
				p.ids = make([]int, M)
			}
			id, e := r.stepDraw(rows[m].draw)
			if e != nil {
				return e
			}
			p.ids[m] = id
			continue
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		if e := gpu.ReadToHost(r.logits, r.logitsPinned); e != nil {
			return e
		}
		p.outs[m] = slices.Clone(r.logitsHost)
	}
	return r.launchErr
}

// mapKV maps the pass's KV rows before anything is allocated: a step's rows in their slots (and its prefill chunk's shared-memory
// check), or the ordinary pass's positions in the bound slot.
func (p *prefillPass) mapKV() error {
	r, startPos, M, rows := p.r, p.startPos, p.M, p.rows
	if rows != nil {
		r.kvStepAbort = false
		for _, rw := range rows {
			if e := r.kvEnsureSlot(rw.slot, rw.pos, 1); e != nil {
				return e
			}
		}
		if r.kvStepAbort {
			// a row took back blocks from a borrower that may be another row of this step: run nothing, so each row can go alone
			return fmt.Errorf("cuda step: a borrower was stopped while the rows were mapped: %w", decoder.ErrKVPoolFull)
		}
		if c := r.stepChunk; c.rows > 0 {
			// The chunk's attention is an ordinary pass's without the row-batched exact kernels: a chunk that would need them declines.
			fused := func(hd int) bool {
				exact := r.forceExactKernels
				r.forceExactKernels = c.exact
				_, ok := r.useAttnFused(hd, c.rows)
				r.forceExactKernels = exact
				return ok
			}
			if e := r.checkPrefillShmem(c.pos, c.rows, fused); e != nil {
				return fmt.Errorf("cuda ragged step: the prefill chunk at %d: %w", c.pos, e)
			}
			r.stepChunk.split = true
		}
	} else if e := r.kvEnsure(startPos, M); e != nil {
		return e
	}
	return nil
}

// prefillRun is cudaResident's per-pass prefill state: the pass and its staging block, the learned and default pass widths, the
// prompt lengths the fast-prefill floor reads, the exact-kernel switches, and the launch counters a test reads.
type prefillRun struct {
	// xhostBuf is the contiguous host block a batched pass stages its embeddings in before the upload (prefillCore), grown to the widest pass and reused.
	xhostBuf []float32
	// pass is the batched prefill pass in progress (prefillPass), reset by each pass and cleared at its end. Executor goroutine only.
	pass prefillPass
	// prefillChunkCap is the LEARNED row budget shared by every batched-prefill caller that has a row-count knob to
	// shrink: prefillChunked (retries the SAME pass smaller and stores whatever width worked) and PrefillImageLast
	// (cannot retry a bidirectional image block smaller, since an image OOM is a function of that block's own M, but
	// halves and stores the budget on OOM so the NEXT image call is caught by its cheap pre-check). 0 until a pass
	// OOMs, then the width that worked (for the image path, the width to try next). A card that cannot hold the
	// default chunk is thereby discovered ONCE, not on every prompt, which matters beyond waste: a context taken to
	// refusal and kept in use can afterwards launch kernels that "return SUCCESS and execute NOTHING"
	// (docs/code-notes/cuda.md#cudaResident.prefillChunkCap). Atomic because both callers run on their own
	// CALLER's goroutine (only the per-pass job is serialized through the executor).
	prefillChunkCap atomic.Int64
	// prefillWidth is the default pass width the build chose (prefillWidthFor; 0 before the choice, which reads as
	// prefillDefaultChunk). GOINFER_PREFILL_CHUNK overrides it and prefillChunkCap lowers it (prefillBaseRows).
	prefillWidth int
	prof         *prefillProf // non-nil ⇒ PrefillLast times each kernel category (test-only; adds stream syncs)
	// passPromptLen is the TOTAL prompt length this batched pass belongs to (startPos+M), set once at the top of
	// prefillCore. The fast-prefill floor is a property of the PROMPT, not of the chunk: prefillChunked splits a long
	// prompt into passes of <=512 rows, so gating on M alone would judge a 3900-token prompt by its 512-row chunk.
	// Per-pass mutable state on the resident, the same shape as prof above.
	//
	// It is NOT cleared after prefillCore returns: residentDrafter.DraftBlock (cuda/drafter.go) reads it too, via
	// aboveFastPrefillFloor/bGemvB, and inherits whatever the TARGET's last prefill set. DraftBlock's doc says why
	// that costs acceptance rate at worst, never correctness.
	passPromptLen int
	// forceExactKernels disables useAttnFused/useGemmMMA (the L2/L3 fast levers) for the DURATION of one prefillCore
	// pass: per-pass mutable state like passPromptLen, set and cleared at prefillCore's own r.do(...) boundary. Those
	// levers are chosen purely on shape, never on WHO is asking, and they are cosine-close to
	// gemv_w4a8_rn/attn_batched, not bit-identical: HiddenLast's bit-identity contract and speculative verify's
	// tailAllLogits/tailAllArgmax tails need decode-identical numerics. So it is true for every prefillCore tail
	// except tailLastLogits (ordinary single-row-output prefill, where the cosine-gated tolerance applies). The
	// drafter's own forward (cuda/drafter.go) never touches it: a proposal need not be bit-identical to anything,
	// only the target's verify of it.
	forceExactKernels bool
	// stepChunk is a ragged step's prefill chunk for the duration of its pass (StepRagged); zero otherwise. Executor thread.
	stepChunk raggedChunk
	// chunkOrdinary is set by prefillChunked for the duration of a multi-chunk prefill whose FINAL tail is the ordinary tailLastLogits: its non-final passes (tailKVOnly) are part of that
	// same prompt prefill and must get the same kernels as the final pass. Without it forceExactKernels (tail != tailLastLogits) demoted EVERY non-final chunk to the slow exact path
	// (docs/measurements/prefill-chunk-demotion-2026-09-21.md). A HiddenLast prefill (tailHiddenLast) leaves it false, so every one of its chunks stays exact.
	chunkOrdinary bool
	// chunkPromptLen is the WHOLE prompt's length (startPos + all rows), set by prefillChunked for the duration of a multi-chunk prefill so the fast-prefill floor is judged on the prompt and not on
	// the prompt-so-far: passPromptLen is startPos+M of THIS pass, so a first chunk narrower than the floor (the OOM-halving path, or GOINFER_PREFILL_CHUNK) would run the exact kernels and
	// its later chunks the fast ones, mixing numerics inside one prompt (TestPrefillChunked_bitIdentical, chunk=300). Zero outside a chunked prefill.
	chunkPromptLen int
	// fastAttnLaunches / fastGemmLaunches count attn_fused and gemm_w4a8_mma launches, so a test can prove WHICH kernels a prefill pass used rather than infer it from timing.
	fastAttnLaunches, fastGemmLaunches int
	// raggedAttnLaunches counts attn_ragged launches, so a test can prove a step's rows took the one launch; headRows counts the
	// rows batchedHeadFull heads, so it can prove a prefill chunk's unread rows took none.
	raggedAttnLaunches, headRows int
	// raggedFALaunches and raggedSKLaunches count fa_partial_ragged and splitkv_scores_ragged launches, so a test can prove a step's
	// flash-lane and split-KV rows took one launch.
	raggedFALaunches, raggedSKLaunches int
	exactAttnLaunches                  int // attn_batched launches in a prefill pass, so a test can prove a tile-served layer took none
	tile128Launches                    int // attn_fused launches that used the 128-row-tile kernel (default for hd128 layers without a window)
}

// prefillKernels is cudaResident's batched prefill: the row-batched kernels and whether they loaded, the exact attention past
// attn_batched's bound, the fast levers (L2's fused attention with its tile arms, L3's tensor-core GEMM) and the ragged twins.
type prefillKernels struct {
	// qTempRowsB is rope_kv_batched's per-row query-scale table (attnTempRows), sized to the largest batched pass seen; qTempRowsKey is the (startPos, M) it currently holds.
	qTempRowsB   Buffer
	qTempRowsCap int
	qTempRowsKey [2]int
	// Batched (M=len) prefill pipelines (prefill_batched.ptx), the weight-stationary path. bGemv is the batched W4A8
	// GEMV; the rest are the M=1 glue kernels with an M dimension, each bit-identical per row. Loaded once at build
	// (small module).
	bRN, bRms, bRopeKV, bAttn, bQuant, bSw, bRes Pipeline
	bW8                                          Pipeline // batched W8A8 GEMV (int8 bundles); nil ⇒ int8 prefill declines
	// The per-32 rows GEMVs (prefill_g32.cu), bound with the per-32 quantizers into bRms/bQuant/bSw under actG32;
	// prefillG32Ready says all of them loaded, and without it a per-32 load's prefill declines.
	bQ4KRows, bQ5KRows, bQ6KRows, bW8G32Rows Pipeline
	prefillG32Ready                          bool
	bQKN                                     Pipeline // batched per-head Q/K RMSNorm (qwen3 etc.); loaded with the batched set
	bNormF32                                 Pipeline // batched plain f32 RMSNorm for Gemma sandwich post-norms; loaded with the batched set
	bLN                                      Pipeline // batched mean-centered LayerNorm+quant (layernorm_quant.ptx) for Cohere/Command-R
	zeroBias                                 Buffer   // [hidden] zeros for bias-free layernorm_quant_batched
	psScores, psSoftmax, psVsum              Pipeline // the exact batched-prefill attention past attn_batched's shared-memory bound (prefill_splitkv.cu)
	psLaunches                               int      // launch groups of it, for tests
	// L2: the fused prefill attention, one instantiation per supported head
	// dim. Zero-valued unless GOINFER_CUDA_FAST_PREFILL selected it AND the module loaded; every selection site
	// treats the zero Pipeline as "use attn_batched".
	bAttnFused64, bAttnFused128 Pipeline
	// Tile-shape arms of attn_fused (attn_fused_bm.cu): 32x64 and 32x32 (query rows x keys), per head dim.
	// attnTile: 0 = default (128x64 for hd128 layers without a window, else 64x64), -1 = 64x64 everywhere (GOINFER_CUDA_ATTN_FUSED_TILE=64x64), 1/2/3 = force 32x64 / 32x32 / 128x64 (experiment arms).
	bAttnBM32x64hd64, bAttnBM32x64hd128, bAttnBM32x32hd64, bAttnBM32x32hd128 Pipeline
	bAttnBM128hd64, bAttnBM128hd128                                          Pipeline
	attnTile                                                                 int
	// L3: the tensor-core int4 GEMM. Zero-valued unless selected AND loaded; bGemvB
	// treats the zero Pipeline as "use gemv_w4a8_rn", which is the exact path.
	bGemmMMA Pipeline
	// Per-lever selection: fastAttn drives L2, fastGemm drives L3.
	fastAttn, fastGemm bool
	prefillReady       bool // batched kernels loaded; PrefillLast usable
	// bRopeKVRagged and bAttnRagged are rope_kv_batched's and attn_batched's ragged twins (prefill_batched.cu); zero when not
	// loaded, when a step's rows take decode's own gap.
	bRopeKVRagged, bAttnRagged Pipeline
}

// visionPrefill is cudaResident's image-block, span and m-RoPE batched prefill: their kernels, whether each loaded, the m-RoPE
// layout, and the DeepStack sets of the call in progress.
type visionPrefill struct {
	bAttnImg             Pipeline      // attn_img_batched (attn_img_prefill.ptx) — Gemma 3's bidirectional image-block prefill attention; own module, see cuda/attn_img_prefill.cu
	imgPrefillReady      bool          // bAttnImg loaded; PrefillImageLast usable. A load failure is not fatal: it stays false and the caller falls back to CPU prefill + UploadKV
	bAttnSpan            Pipeline      // attn_span_batched (attn_img_prefill.ptx) — Gemma 4's bidirectional span attention (gemma4AttendRange)
	spanPrefillReady     bool          // bAttnSpan loaded; PrefillSpansLast usable, else it declines to the CPU prefill + UploadKV
	spanPass             bool          // set by PrefillSpansLast for its passes: every pass's attention is attn_span_batched, with or without a span in it
	bRopeKVMRoPE         Pipeline      // rope_kv_mrope_batched (rope_mrope_prefill.ptx) — Qwen2.5-VL's m-RoPE batched-prefill rotation; own module, see cuda/rope_mrope_prefill.cu
	mropePrefillReady    bool          // bRopeKVMRoPE loaded AND this model has MRopeSection; PrefillMRoPELast usable. A load failure (or a non-m-RoPE model) is not fatal: it stays false and the caller falls back to CPU prefill + UploadKV
	mropeTakesMode       bool          // the bound bRopeKVMRoPE is rope_kv_mrope_batched (takes the layout mode as a final argument), not the pairwise twin (no such argument)
	mropeMode            int32         // rope_kv_mrope_batched's layout: 0 contiguous sections (Qwen2.5-VL), 1 interleaved per index (Qwen3-VL, Qwen3.5+); mropeSec0/1 mean what rope_mrope_prefill.cu's mrope_pos says for that mode
	mropeSec0, mropeSec1 int32         // cumulative MRopeSection boundaries (sec0=section[0], sec1=section[0]+section[1]), computed once at build time — see rope_kv_mrope_batched's own doc comment for the (d<sec0)?t:(d<sec1?h:w) rule this feeds
	deepPlan             *cudaDeepPlan // the DeepStack sets of the PrefillMRoPEDeepstackLast call in progress; nil otherwise
}
