//go:build cuda

package cuda

import (
	"context"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
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
	p.aqB, p.aScB = p.ai(M*hidden/4), p.af(M)
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
	p.cqB, p.cScB = p.ai(M*maxQDim/4), p.af(M)
	p.mqB, p.mScB = p.ai(M*hidden/4), p.af(M)
	p.gOb, p.uOb = p.af(M*inter), p.af(M*inter)
	p.dqB, p.dScB = p.ai(M*inter/4), p.af(M)
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
