//go:build darwin

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"runtime"

	"github.com/townsendmerino/goinfer/decoder"
)

// MC3 (docs/tasks/task-concurrency-2026-09.md): batched decode, one decode token for up to batchMaxSeqs sequences in ONE
// command buffer, each sequence on its own resident KV slot at its own position, and every logit bit-identical to
// production's single-token forward on that sequence (TestMC3Step_bitIdentical). Per-row work runs production's own
// kernels, per sequence: norm + int8 quantisation, RoPE, the KV store into that sequence's slot, attention over that slot
// (attention_fa at or above attnFADepthFloor, as a single-token step would plan it), ctx quantisation, SwiGLU +
// quantisation, the final norm. The five matmuls run once for all sequences on the matrix units and must reproduce
// production's arithmetic, so do not reorder their accumulation: the W4A8 projections (mc3_bt, mc3_btd) sum each 32-k
// group exactly in the f32 fragment, accumulate per lane residue in the decode GEMV's lane order and reduce by simd_sum's
// own xor tree 1, 2, 4, 8, 16 (batchSimdSumTreeOK checks that tree at build); the int8 LM head (mc3_lm) sums in int32.
// The decoder calls this at B >= batchMinSeqs only. Measurements: docs/code-notes/metal.md#batchMaxSeqs.

// batchMaxSeqs is the fragment's token width: every batched matmul computes 8 token columns.
const batchMaxSeqs = 8

// batchMinSeqs is the smallest batch the decoder should send here: the fragment's cost does not shrink with the batch,
// so at B = 1 a step costs more than a production token, and B = 2 is about even but spares the per-token KV-slot
// rebinding two interleaved single-token decodes would pay (docs/code-notes/metal.md#batchMinSeqs).
const batchMinSeqs = 2

// batchState is a resident's MC3 batched-step machinery: pipelines, per-row scratch for batchMaxSeqs sequences, and
// per-sequence uniforms. nil when the resident cannot batch (batchIneligible).
type batchState struct {
	pack, packLM, bt, btd, lm Pipeline
	xB, qkvB, guB, logitsB    Buffer // f32 rows
	aqB, cqB, mqB, dqB        Buffer // int8 rows
	aScB, cScB, mScB, dScB    Buffer // one f32 per row
	aT, aTp                   Buffer // half [K][8]
	uM, uQKV, uGU, uV         Buffer
	uMode                     [3]Buffer
	uPos, uNKeys, uQTemp      []Buffer
	uFANSplit                 []Buffer
	// per-row uniforms and outputs of a device draw (ForwardSample's gumbel dispatches, run on a row of logitsB)
	uGInvT, uGK0, uGK1, uGD0, uGD1 []Buffer
	gOut                           Buffer // [batchMaxSeqs] int32 ids
	amaxPart, amaxTok, uAmaxP      Buffer // E-P08: each greedy row's argmax partials, its id, and the partial count
	greedyDevRows                  int    // E-P08: greedy-draw rows a step served from the device argmax (a test reads it)
	noHeadSteps                    int    // R-19: steps that ran no head (prefillByStep's pieces before the last; a test reads it)
	// per-row positions and Q temperature scales for mc3_rope2_rows, and per-row attention outputs, so the ctx quantisation
	// runs over every row at once
	posB, qtB, ctxB           Buffer
	nKeysB, rowmapB, slotOffB Buffer // per-row key counts; the rows mc3_attention_rows serves; each row's slot offset
	// per-row split counts, the rows the multi-row flash attention serves, one partial region per row, and the uniforms for
	// that region's length and the step's largest split count
	nSplitB, rowmapFA, faPartB Buffer
	uPartStride, uMaxS         Buffer
	uCount                     []Buffer // uCount[k] holds k (0..batchMaxSeqs): a row count for a dispatch
	noBias                     Buffer
	qkvRows, nHhd, kOff, vOff  int
	// qkv / gate|up run as per-row production GEMVs at B <= rowsQKV / rowsGU, and as the fragment above that (0: always the
	// fragment). Set by calibrateRows at build; tests set them to force either path.
	rowsQKV, rowsGU int
	// The q4k lane's f32 rows (the normed residual, SwiGLU's output) and its row strides as uniforms, by width.
	axB, swB Buffer
	uLd      map[int]Buffer
	// steps / seqs / maxSeqs count completed batched steps, the sequences they served, and the largest batch — read by
	// tests to confirm a concurrent run really batched. Written only inside forwardMulti, which the decoder serialises.
	steps, seqs, maxSeqs int
}

// batchSeq is one sequence of a batched step: its resident KV slot, the position it decodes, and its input embedding.
type batchSeq struct {
	slot, pos int
	emb       []float32
	draw      *decoder.ResidentBatchDraw // non-nil: draw this row's next token on-device (ForwardSample's draw)
}

// batchIneligible is why this resident cannot run a batched step ("" if it can): MC3 covers the plain dense W4A8 decode
// path — the family flags every per-row kernel in forwardMulti assumes — with each layer's KV slots in one f16
// allocation. One slot is enough to prefill a prompt on the step (one sequence's consecutive rows); batched decode and the
// step verify need two, which BatchStepRange, BatchGreedyDraw and VerifyCost check.
func (r *resident) batchIneligible() string {
	switch {
	case len(r.kvSlotBufs) == 0 || !r.kvContig:
		return "KV slots that are not one allocation per layer"
	case r.moe != nil || r.g4moe != nil:
		return "MoE"
	case r.pleP > 0:
		return "a Gemma 4 E-model (per-row PLE inputs, KV-shared layers, per-layer FFN widths)"
	case r.pairwiseRoPE:
		return "GPT-J pairwise rope (the step's mc3_rope2_rows is the NeoX half-split; docs/tasks/task-metal-pairwise-rope-2026-10.md)"
	case r.attnSoftcap != 0:
		return "an attention softcap (gated on the single-token and exact prefill kernels, not the step's rows forms)"
	case r.sandwich || r.postOnly || r.parallelBlock || r.layerNorm || r.decodeLaneW4F16 || r.nonGatedMLP ||
		r.outBias || r.qkNormWhole || r.learnedPos || r.attnSink: // per-head QK-norm (Qwen3) is in the step since E-P07
		return "a family variant the batched step does not reproduce"
	}
	g0 := r.layers[0].geom
	if g0 == nil {
		return "no attention geometry"
	}
	for _, L := range r.layers {
		if L.qGate || L.recurrent() || L.geom == nil || L.geom.kEqV || L.window != 0 ||
			L.geom.hd != g0.hd || L.geom.kvDim != g0.kvDim || L.geom.nKV != g0.nKV {
			return "a layer the batched step does not reproduce"
		}
	}
	qkvRows := r.nH*g0.hd + 2*g0.kvDim
	if r.q4kLane { // its rows GEMVs take any width its decode GEMVs take, and its head runs decode's kernel per row
		return ""
	}
	for _, c := range []struct{ n, by int }{{qkvRows, 16}, {r.H, 16}, {2 * r.I, 16}, {r.V, 128}, {r.H, 32},
		{r.nH * g0.hd, 32}, {r.I, 32}} {
		if c.n%c.by != 0 {
			return fmt.Sprintf("a width (%d) the batched kernels do not tile (%% %d)", c.n, c.by)
		}
	}
	return ""
}

// buildBatch compiles and allocates the batched step when the resident is eligible and this GPU's simd_sum is the tree
// the kernels reproduce; otherwise it leaves r.batch nil (the decoder then never batches). It compiles with the build's
// own fast-math setting (r.preciseMath), so the batched kernels round like the kernels they must match.
func (r *resident) buildBatch() {
	if why := r.batchIneligible(); why != "" {
		return
	}
	if len(r.kvSlotBufs) == 1 { // the one-slot build is the memory-tight one: its scratch is priced, and a miss declines the step, not the load
		if need := r.stepScratchBytes(); need > r.stepHeadroom {
			if r.stepHeadroom < 0 {
				r.stepDecline = "this machine's RAM size is unreadable, so the batched step's scratch cannot be priced"
			} else {
				r.stepDecline = fmt.Sprintf("the batched step's %.1f MB of scratch does not fit beside one KV slot (the memory guard leaves %.1f MB)",
					float64(need)/(1<<20), float64(r.stepHeadroom)/(1<<20))
			}
			fmt.Fprintf(os.Stderr, "metal: batched step off: %s; prompts prefill one token at a time\n", r.stepDecline)
			return
		}
	}
	compile := r.d.CompileLibrary
	if r.preciseMath {
		compile = r.d.CompileLibraryPrecise
	}
	lib, err := compile(batchKernelsHeader+batchKernels, MSL3_1) // its own small library: allKernels is compiled once, above
	if err != nil {
		fmt.Fprintf(os.Stderr, "metal: batched decode off (compile: %v)\n", err)
		return
	}
	p := func(n string) Pipeline {
		pp, e := r.d.NewComputePipeline(lib, n)
		if e != nil {
			panic(fmt.Sprintf("metal pipeline %s: %v", n, e))
		}
		return pp
	}
	if !batchSimdSumTreeOK(r.d, p("mc3_sstree")) {
		fmt.Fprintf(os.Stderr, "metal: batched decode off — this GPU's simd_sum is not the xor tree the batched kernels reproduce, so their output would not be bit-identical\n")
		return
	}
	d, g0 := r.d, r.layers[0].geom
	b := &batchState{nHhd: r.nH * g0.hd}
	b.qkvRows = b.nHhd + 2*g0.kvDim
	b.kOff, b.vOff = b.nHhd*4, (b.nHhd+g0.kvDim)*4
	b.pack, b.packLM, b.bt, b.btd, b.lm = p("mc3_pack"), p("mc3_pack_lm"), p("mc3_bt_fb2"), p("mc3_btd_fb2"), p("mc3_lm_fb4")
	const B = batchMaxSeqs
	maxK := max(r.H, b.nHhd, r.I)
	b.xB, b.qkvB, b.guB, b.logitsB = d.NewBufferLen(B*r.H), d.NewBufferLen(B*b.qkvRows), d.NewBufferLen(B*2*r.I), d.NewBufferLen(B*r.V)
	b.aqB, b.cqB, b.mqB, b.dqB = d.NewBufferBytes(B*r.H), d.NewBufferBytes(B*b.nHhd), d.NewBufferBytes(B*r.H), d.NewBufferBytes(B*r.I)
	b.aScB, b.cScB, b.mScB, b.dScB = d.NewBufferLen(B), d.NewBufferLen(B), d.NewBufferLen(B), d.NewBufferLen(B)
	b.aT, b.aTp = NewBufferU16s(d, make([]uint16, maxK*8)), NewBufferU16s(d, make([]uint16, r.H*8))
	b.uM = NewBufferU32(d, 1)
	b.uQKV, b.uGU, b.uV = NewBufferU32(d, uint32(b.qkvRows)), NewBufferU32(d, uint32(2*r.I)), NewBufferU32(d, uint32(r.V))
	b.uMode = [3]Buffer{NewBufferU32(d, 0), NewBufferU32(d, 1), NewBufferU32(d, 2)}
	b.noBias = d.NewBufferLen(max(r.H, 2*r.I))
	for range B {
		b.uPos = append(b.uPos, NewBufferU32(d, 0))
		b.uNKeys = append(b.uNKeys, NewBufferU32(d, 1))
		b.uQTemp = append(b.uQTemp, NewBufferFloats(d, []float32{1}))
		b.uFANSplit = append(b.uFANSplit, NewBufferU32(d, 1))
		b.uGInvT = append(b.uGInvT, NewBufferFloats(d, []float32{1}))
		b.uGK0, b.uGK1 = append(b.uGK0, NewBufferU32(d, 0)), append(b.uGK1, NewBufferU32(d, 0))
		b.uGD0, b.uGD1 = append(b.uGD0, NewBufferU32(d, 0)), append(b.uGD1, NewBufferU32(d, 0))
	}
	b.gOut = d.NewBufferLen(B)
	b.amaxPart, b.amaxTok, b.uAmaxP = d.NewBufferLen(2*B*batchArgmaxParts), d.NewBufferLen(B), NewBufferU32(d, batchArgmaxParts)
	b.posB, b.qtB, b.ctxB = d.NewBufferLen(B), NewBufferFloats(d, make([]float32, B)), d.NewBufferLen(B*b.nHhd)
	b.nKeysB, b.rowmapB, b.slotOffB = d.NewBufferLen(B), d.NewBufferLen(B), d.NewBufferLen(B)
	b.nSplitB, b.rowmapFA, b.uMaxS = d.NewBufferLen(B), d.NewBufferLen(B), NewBufferU32(d, 1)
	if r.attnFAPartialLen > 0 {
		b.faPartB, b.uPartStride = d.NewBufferLen(B*r.attnFAPartialLen), NewBufferU32(d, uint32(r.attnFAPartialLen))
	}
	for k := 0; k <= B; k++ {
		b.uCount = append(b.uCount, NewBufferU32(d, uint32(k)))
	}
	if r.q4kLane {
		b.axB, b.swB = d.NewBufferLen(B*r.H), d.NewBufferLen(B*r.I)
		b.uLd = map[int]Buffer{}
		for _, n := range []int{r.H, b.nHhd, r.I, 2 * r.I, b.qkvRows} {
			b.uLd[n] = NewBufferU32(d, uint32(n))
		}
		r.batch = b // calibrateRows picks between int4 kernels the lane does not run
		return
	}
	r.batch = b
	r.calibrateRows()
}

// stepScratchBytes is what buildBatch allocates, each buffer rounded up to the page Metal allocates in, so the one-slot price
// does not under-count (TestP29_oneSlotStepScratchPriced holds it against the driver's own count). A buffer added to
// buildBatch is added here.
func (r *resident) stepScratchBytes() int64 {
	const B = batchMaxSeqs
	page := os.Getpagesize()
	g0 := r.layers[0].geom
	nHhd := r.nH * g0.hd
	qkvRows := nHhd + 2*g0.kvDim
	var n int64
	add := func(bytes, count int) { n += int64(count) * int64((bytes+page-1)/page*page) }
	f := func(floats, count int) { add(4*floats, count) }
	f(B*r.H, 1)                     // xB
	f(B*qkvRows, 1)                 // qkvB
	f(B*2*r.I, 1)                   // guB
	f(B*r.V, 1)                     // logitsB
	add(B*r.H, 2)                   // aqB, mqB
	add(B*nHhd, 1)                  // cqB
	add(B*r.I, 1)                   // dqB
	f(B, 4)                         // aScB, cScB, mScB, dScB
	add(2*max(r.H, nHhd, r.I)*8, 1) // aT
	add(2*r.H*8, 1)                 // aTp
	f(1, 7)                         // uM, uQKV, uGU, uV, uMode
	f(max(r.H, 2*r.I), 1)           // noBias
	f(1, 9*B)                       // the per-row uniforms
	f(B, 1)                         // gOut
	f(2*B*batchArgmaxParts, 1)      // amaxPart
	f(B, 1)                         // amaxTok
	f(1, 1)                         // uAmaxP
	f(B, 2)                         // posB, qtB
	f(B*nHhd, 1)                    // ctxB
	f(B, 5)                         // nKeysB, rowmapB, slotOffB, nSplitB, rowmapFA
	f(1, 1)                         // uMaxS
	if r.attnFAPartialLen > 0 {
		f(B*r.attnFAPartialLen, 1) // faPartB
		f(1, 1)                    // uPartStride
	}
	f(1, B+1) // uCount
	if r.q4kLane {
		f(B*r.H, 1) // axB
		f(B*r.I, 1) // swB
		f(1, 5)     // uLd, at most one per distinct width
	}
	return n
}

// mc3W8RowsOn runs a native int8 resident's step projections as mc3_gemv_w8a8_rows (each weight row read once for all
// B rows; int8 slice 3b) rather than production's int8 GEMV once per row; both are decode's bits. Tests turn it off
// for the per-row arm.
var mc3W8RowsOn = true

// mc3FARowsOn runs the batched step's rows at attention_fa depth through one multi-row dispatch pair
// (mc3_attention_fa*_rows), bit-identical to their per-row pairs. On by default; tests turn it off for the per-row arm.
// Evidence: docs/code-notes/metal.md#mc3FARowsOn.
var mc3FARowsOn = true

// mc3DeviceArgmaxOn takes a batched step's greedy rows' argmax on the GPU (E-P08): argmax_rows_part then argmax_finish
// over each row of logitsB, so the host reads 4 bytes a row instead of scanning V floats after the command buffer. The id
// is the host argmaxF32's (the first maximum; a row that starts with NaN is 0, as there), so output is unchanged.
var mc3DeviceArgmaxOn = true

// batchArgmaxParts is the partial count argmax_rows_part writes a row: 64 threadgroups of 256 threads.
const batchArgmaxParts = 64

// mc3AdjRowsOn runs the batched step's per-row qkv and gate|up GEMVs (the sizes calibrateRows gives to per-row) as one
// dispatch with each weight tile's B threadgroups adjacent (mc3_gemv_w4a8_sa_{bias_}rows_adj), bit-identical to the B
// per-row dispatches. On by default; evidence: docs/code-notes/metal.md#mc3AdjRowsOn.
var mc3AdjRowsOn = true

// batchTGBytes is mc3_bt's / mc3_btd's threadgroup memory at FB = 2: the Q exchange.
const batchTGBytes = 4 * 2 * 32 * 8

// calibrateRows decides, once, at which batch sizes qkv and gate|up are cheaper as B per-row production GEMVs than as one
// fragment dispatch (docs/tasks/task-concurrency-2026-09.md, MC3 S4). The fragment's cost is fixed for any B up to 8 and a
// GEMV's is per row, so on large shapes (a bandwidth-bound GEMV against an ALU-bound fragment) the GEMVs win at small B.
// Both arms run on the real weights of the first layers, one layer's weights per dispatch so the cache does not serve
// them, and the fastest of 7 command buffers after 3 warm-ups decides. The choice changes speed only: the per-row path is
// production's own kernel on each row.
// History: docs/code-notes/metal.md#calibrateRows.
func (r *resident) calibrateRows() {
	b := r.batch
	if r.w8 { // int8 slice 3: the fragment kernels read int4, so every projection runs production's int8 GEMV per row
		b.rowsQKV, b.rowsGU = batchMaxSeqs, batchMaxSeqs
		return
	}
	attnInt8 := r.w8Attn // int4mix: qkv is int8 (never the int4 fragment); gate|up is int4 and calibrates as usual
	n := min(8, r.nL)
	b.uM.SetU32(batchMaxSeqs)
	pq, nq := saRowsPick(r.pSABias, r.pSABiasRows, b.qkvRows, r.gemvRows.qkv)
	pg, ng := saRowsPick(r.pSA, r.pSARows, 2*r.I, r.gemvRows.gu)
	arms := []func(e *Encoder, L *residLayer){
		func(e *Encoder, L *residLayer) { // qkv, fragment
			e.DispatchTG(b.bt, b.qkvRows/16*128, 128, batchTGBytes, L.qkvW, L.qkvS, b.aT, b.aScB, b.qkvB, r.uH, b.uQKV, b.uM, L.qkvBias, b.uMode[1])
		},
		func(e *Encoder, L *residLayer) { // qkv, one production GEMV
			e.DispatchTG(pq, nq, 256, r.H*2, L.qkvW, L.qkvS, b.aqB, b.aScB, b.qkvB, L.qkvBias, r.uH)
		},
		func(e *Encoder, L *residLayer) { // gate|up, fragment
			e.DispatchTG(b.bt, 2*r.I/16*128, 128, batchTGBytes, L.guW, L.guS, b.aT, b.mScB, b.guB, r.uH, b.uGU, b.uM, b.noBias, b.uMode[0])
		},
		func(e *Encoder, L *residLayer) { // gate|up, one production GEMV
			e.DispatchTG(pg, ng, 256, r.H*2, L.guW, L.guS, b.mqB, b.mScB, b.guB, r.uH)
		},
	}
	// The arms interleave rep by rep, so both of a pair see the same GPU state: a first use of a pipeline, or a clock still
	// ramping, only ever makes a buffer slower.
	best := make([]float64, len(arms))
	for rep := range 10 {
		for a, enc := range arms {
			if attnInt8 && a < 2 { // the qkv fragment reads int4 weights and f16 group scales, which int4mix's qkv is not
				continue
			}
			e := r.q.Begin()
			for l := range n {
				enc(e, &r.layers[l])
			}
			e.End()
			if e.Err() != nil {
				return // keep the fragment everywhere (rowsQKV = rowsGU = 0)
			}
			if t := (e.GPUEnd() - e.GPUStart()) / float64(n); rep >= 3 && (best[a] == 0 || t < best[a]) {
				best[a] = t
			}
		}
	}
	rows := func(frag, gemv float64) int {
		k := 0
		for B := batchMinSeqs; B <= batchMaxSeqs && frag > 0 && gemv > 0 && float64(B)*gemv < frag; B++ {
			k = B
		}
		return k
	}
	b.rowsQKV, b.rowsGU = rows(best[0], best[1]), rows(best[2], best[3])
	if attnInt8 {
		b.rowsQKV = batchMaxSeqs
	}
	fmt.Fprintf(os.Stderr, "metal: batched step: qkv as per-row GEMVs at B <= %d (GEMV %.3f ms, fragment %.3f ms), gate|up at B <= %d (%.3f, %.3f); 0 = always the fragment\n",
		b.rowsQKV, best[1]*1e3, best[0]*1e3, b.rowsGU, best[3]*1e3, best[2]*1e3)
}

// batchSimdSumTreeOK runs mc3_sstree over random vectors spanning 24 binades (where different addition trees round
// differently) and reports whether simd_sum equalled the xor tree 1, 2, 4, 8, 16 on every one.
func batchSimdSumTreeOK(d *Device, p Pipeline) bool {
	const W = 4096 // simdgroups
	in := make([]float32, W*32)
	seed := uint32(12345)
	rnd := func() uint32 { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed }
	for i := range in {
		m := float64(rnd())/float64(1<<32) - 0.5
		in[i] = float32(math.Ldexp(m, int(rnd()%24)-12))
	}
	ib, ob := NewBufferFloats(d, in), d.NewBufferLen(W*2)
	q := d.NewCommandQueue()
	e := q.Begin()
	e.DispatchTG(p, W*32, 256, 0, ib, ob)
	e.End()
	if e.Err() != nil {
		return false
	}
	o := ob.Floats()[:W*2]
	for w := range W {
		if math.Float32bits(o[w*2]) != math.Float32bits(o[w*2+1]) {
			return false
		}
	}
	return true
}

// canUseAttnFAAt is canUseAttnFA for a given key count rather than the one planNKeys reports: the batched step plans
// each sequence's attention for its own depth, exactly as that sequence's single-token step would.
func (r *resident) canUseAttnFAAt(l, nKeys int) bool {
	if !r.decodeAttnFA || r.attnFAPartial == (Buffer{}) {
		return false
	}
	if r.sandwich || r.postOnly || r.parallelBlock || r.attnSink || r.kvI8 || r.kvF32 {
		return false
	}
	if r.loraLayers != nil {
		return false
	}
	L := &r.layers[l]
	if L.moe != nil || L.g4moe != nil || L.recurrent() || L.qGate || L.window != 0 {
		return false
	}
	g := L.geom
	return g != nil && attnFAHeadDimOK(g.hd, r.nH, g.nKV) && attnFAGroupOK(r.nH, g.nKV) && nKeys >= r.attnFAFloor()
}

// forwardMulti runs one decode token for each of seqs — 1..batchMaxSeqs sequences, each on its own resident KV slot at
// its own position — in one command buffer. It returns each row's logits (fresh slices), or, for a row that carries a
// draw, the id ForwardSample would have drawn (ids[m] >= 0, logits[m] nil); ids[m] is -1 for a logits row. It never
// touches the bound slot (r.kc/r.vc); every sequence's KV is addressed through kvSlotBufs. The pipelined executor is
// stopped first: a command buffer it pre-encoded would otherwise run after this one's writes with its own stale plan.
func (r *resident) forwardMulti(seqs []batchSeq) (logits [][]float32, ids []int, err error) {
	return r.forwardMultiInto(seqs, false)
}

// forwardMultiInto is forwardMulti; with argmaxOnly it returns no logits, only each row's argmax, read in place from
// the step's logits buffer (no per-row copy). That is the same id the copied row's argmax gives when no logit transform
// applies, and the caller falls back to the full rows when one does (softcap, a logit scale).
func (r *resident) forwardMultiInto(seqs []batchSeq, argmaxOnly bool) (logits [][]float32, ids []int, err error) {
	if argmaxOnly {
		return r.forwardMultiOut(seqs, multiArgmax)
	}
	return r.forwardMultiOut(seqs, multiFull)
}

// multiOut is what a step returns. multiNoHead and multiLastRow serve a prompt run on the step kernels (prefillByStep, R-19,
// docs/tasks/task-recompute-audit.md): a piece before the last needs only its K/V, so the step encodes no final norm, LM head
// or argmax and returns nothing; the last piece needs only its last row's logits, so only that row is copied out (the head
// still runs for every row: it reads the whole LM head's weights once at any row count).
type multiOut int

// stepPrefillAllHeads restores prefillByStep's pre-R-19 modes (every piece runs the head; the last converts every row),
// for the test that compares the two and the A/B that times them. Tests only.
var stepPrefillAllHeads = false

// stepSlotOffsetPlantForTest shifts every step row's KV slot offset by this many rows: the planted defect that the one-slot
// step-prefill identity check must catch. Zero in production.
var stepSlotOffsetPlantForTest = 0

const (
	multiFull    multiOut = iota // every row's logits, or a draw row's id
	multiArgmax                  // every row's argmax, read in place
	multiNoHead                  // the trunk only: K/V written, no head, nothing returned
	multiLastRow                 // multiFull for the last row only; the other rows' logits are nil
)

// forwardMultiOut is forwardMultiInto with an explicit output mode (multiOut).
func (r *resident) forwardMultiOut(seqs []batchSeq, out multiOut) (logits [][]float32, ids []int, err error) {
	argmaxOnly := out == multiArgmax
	b := r.batch
	if b == nil {
		return nil, nil, fmt.Errorf("metal: batched decode unavailable")
	}
	B := len(seqs)
	if B < 1 || B > batchMaxSeqs {
		return nil, nil, fmt.Errorf("metal: batched step of %d sequences (1..%d)", B, batchMaxSeqs)
	}
	if r.loraLayers != nil {
		return nil, nil, fmt.Errorf("metal: batched step with an adapter bound")
	}
	// A slot may carry several rows only as ONE sequence's consecutive positions, in row order (a speculative verify:
	// every row's K/V is stored before its attention runs, layer by layer, so each row sees the rows below it as
	// sequential decode would). Two sequences on one slot would overwrite each other.
	last := map[int]int{}
	for _, q := range seqs {
		if q.slot < 0 || q.slot >= len(r.kvSlotBufs) {
			return nil, nil, fmt.Errorf("metal: batched step: slot %d out of range", q.slot)
		}
		if p, ok := last[q.slot]; ok && q.pos != p+1 {
			return nil, nil, fmt.Errorf("metal: batched step: slot %d repeated at position %d after %d (rows on one slot must be consecutive positions of one sequence)", q.slot, q.pos, p)
		}
		last[q.slot] = q.pos
		if q.pos < 0 || q.pos >= r.ctxCap {
			return nil, nil, fmt.Errorf("metal: batched step: position %d outside the resident context %d", q.pos, r.ctxCap)
		}
		if len(q.emb) != r.H {
			return nil, nil, fmt.Errorf("metal: batched step: embedding length %d != hidden %d", len(q.emb), r.H)
		}
		if q.draw != nil && !r.SampleAvailable() {
			return nil, nil, fmt.Errorf("metal: batched step: a device draw on a resident that cannot sample on-device")
		}
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r.stopExec()
	if r.attnFANKV > 0 {
		r.uAttnFAG.SetU32(uint32(r.nH / r.attnFANKV))
	}
	b.uM.SetU32(uint32(B))
	for m, q := range seqs {
		copy(b.xB.Floats()[m*r.H:(m+1)*r.H], q.emb)
		r.addLearnedPosTo(b.xB.Floats()[m*r.H:(m+1)*r.H], q.pos)
		b.uPos[m].SetU32(uint32(q.pos))
		b.uNKeys[m].SetU32(uint32(q.pos + 1))
		scale := float32(1)
		if r.attnTempBeta != 0 {
			scale = float32(1 + r.attnTempBeta*math.Log1p(math.Floor(float64(q.pos)/r.attnTempOrigMaxPos)))
		}
		b.uQTemp[m].Floats()[0] = scale
		b.posB.U32s()[m] = uint32(q.pos)
		b.nKeysB.U32s()[m] = uint32(q.pos + 1)
		b.slotOffB.U32s()[m] = uint32(q.slot*r.kvSlotBytes[0]/r.kvElemBytes() + stepSlotOffsetPlantForTest*r.layers[0].geom.kvDim) // cache elements; batchIneligible: uniform layers
		b.qtB.Floats()[m] = scale
		if r.attnFANKV > 0 {
			ns := r.attnFASplitFor(q.pos+1, r.attnFANKV)
			b.uFANSplit[m].SetU32(uint32(ns))
			b.nSplitB.U32s()[m] = uint32(ns)
		}
	}
	H, I, nHhd := r.H, r.I, b.nHhd
	// Buffer.At SETS the byte offset from the allocation's start — it does not add to one a view already carries — so a
	// sub-row offset is always computed whole (row*n + off), never as row.At(off).
	f32 := func(buf Buffer, m, n int) Buffer { return buf.At(4 * m * n) }
	pack := func(e *Encoder, aq Buffer, K int, uK Buffer) { e.Dispatch(b.pack, K*8, 256, aq, b.aT, b.uM, uK) }
	const tgb = batchTGBytes
	// S4: below the calibrated sizes, qkv and gate|up run production's GEMV once per row (the row's int8 activations
	// and scale in, its output row out), which is production's arithmetic by definition.
	rowsQKV, rowsGU := B <= b.rowsQKV, B <= b.rowsGU
	w8Rows := r.w8 && mc3W8RowsOn && B >= 2 // int8 slice 3b: one weight read for all B rows
	// int4mix (slice 4): the attention projections are int8 on r.w8Attn too, its FFN int4 on the step's int4 kernels
	w8AttnRows := (r.w8 || r.w8Attn) && mc3W8RowsOn && B >= 2
	w8Attn := r.w8 || r.w8Attn
	pq, nq := saRowsPick(r.pSABias, r.pSABiasRows, b.qkvRows, r.gemvRows.qkv)
	pg, ng := saRowsPick(r.pSA, r.pSARows, 2*I, r.gemvRows.gu)
	// Each row's attention plan, once per step: batchIneligible admits only uniform layers without a window, so a row's
	// plan depends on its depth alone. Rows on the per-head kernel run in one dispatch (rowmapB); a row at attention_fa
	// depth keeps its own two dispatches.
	faRow := make([]bool, B)
	nPlain, nFA, maxS := 0, 0, 0
	for m, q := range seqs {
		faRow[m] = r.canUseAttnFAAt(0, q.pos+1)
		if !faRow[m] {
			b.rowmapB.U32s()[nPlain] = uint32(m)
			nPlain++
		} else {
			b.rowmapFA.U32s()[nFA] = uint32(m)
			nFA++
			maxS = max(maxS, int(b.nSplitB.U32s()[m]))
		}
	}
	faRows := mc3FARowsOn && nFA > 0 && b.faPartB != (Buffer{})
	if faRows {
		b.uMaxS.SetU32(uint32(maxS))
	}
	e := r.q.Begin()
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		g := L.geom
		// the per-row kernels run as one dispatch over all B rows (S3; batch_rows.go): norm+quant, RoPE, ctx quant,
		// SwiGLU+quant. The KV store and attention address each sequence's own slot, so they stay per sequence.
		if r.q4kLane { // decode's lane (encodeAttentionResidualWith): f32 rows, no activation quantization
			e.Dispatch(r.pRmsF32OutRows, B*tgReduceNorm, tgReduceNorm, b.xB, L.preNorm, b.axB, r.uH, r.uEps, r.uAddOne)
			r.q4kStepGemv(e, B, L.qkvSegs, L.qkvBias, b.axB, H, b.qkvB, b.qkvRows, r.uH)
		} else {
			e.Dispatch(r.pRmsRows, B*tgReduceNorm, tgReduceNorm, b.xB, L.preNorm, b.aqB, b.aScB, r.uH, r.uEps, r.uAddOne)
			if w8AttnRows {
				e.Dispatch(r.pW8Rows[B], b.qkvRows*32, 32, L.qkvW, L.qkvS, b.aqB, b.aScB, b.qkvB, r.uH, b.uQKV, b.uCount[B], L.qkvBias, b.uMode[1])
			} else if rowsQKV && mc3AdjRowsOn && B >= 2 && r.gemvRows.qkv > 0 {
				e.DispatchTG(r.pSABiasRowsAdj, B*nq, 256, H*2, L.qkvW, L.qkvS, b.aqB, b.aScB, b.qkvB, L.qkvBias, r.uH, b.uQKV, b.uCount[B])
			} else if rowsQKV {
				for m := range B {
					e.DispatchTG(pq, nq, 256, H*2, L.qkvW, L.qkvS, b.aqB.At(m*H), b.aScB.At(4*m), f32(b.qkvB, m, b.qkvRows), L.qkvBias, r.uH)
				}
			} else {
				pack(e, b.aqB, H, r.uH)
				e.DispatchTG(b.bt, b.qkvRows/16*128, 128, tgb, L.qkvW, L.qkvS, b.aT, b.aScB, b.qkvB, r.uH, b.uQKV, b.uM, L.qkvBias, b.uMode[1])
			}
		}
		if r.qkNorm { // E-P07: Qwen3's per-head Q/K RMSNorm before RoPE, decode's qk_norm over every row, at its width
			e.Dispatch(r.pQKNormRows, B*(r.nH+g.nKV)*tgReduceAttn, tgReduceAttn, b.qkvB, L.qNorm, L.kNorm, r.uNH, g.uNKV, g.uHd, g.uNHhd, r.uEps, r.uAddOne, b.uQKV)
		}
		e.Dispatch(r.pRope2Rows, B*(r.nH*g.half+g.nKV*g.half), 64, b.qkvB, L.invf, g.uHd, b.posB, g.uQtotal, g.uKtotal, g.uHalf, L.mscale, g.uNHhd, b.qtB, b.uQKV, b.uM)
		// KV store and attention, each row over its OWN slot: a layer's slots are one allocation (kvContig), reached
		// as the slot-0 base (kvSlotBufs[0], offset 0) plus the row's element offset (slotOffB). One dispatch stores
		// every row's K/V; one runs every per-head-kernel row's attention.
		kcAll, vcAll := r.kvSlotBufs[0].kc[l], r.kvSlotBufs[0].vc[l]
		if r.kvI8 { // one thread per (row, KV head), as kv_store_i8 is one per head; the scales from slot 0's allocation too
			ksAll, vsAll := r.kvSlotBufs[0].ks[l], r.kvSlotBufs[0].vs[l]
			e.Dispatch(r.pKvI8Rows, B*g.nKV, 32, b.qkvB.At(b.kOff), b.qkvB.At(b.vOff), kcAll, vcAll, ksAll, vsAll, g.uNKV, g.uHd, b.posB, b.uQKV, b.uM, b.slotOffB)
		} else {
			e.Dispatch(r.pKvRows, B*g.kvDim, 64, b.qkvB.At(b.kOff), b.qkvB.At(b.vOff), kcAll, vcAll, g.uKvDim, b.posB, b.uQKV, b.uM, b.slotOffB)
		}
		if faRows {
			shmBytes := 128 * 6 * (r.nH / g.nKV) * 4
			e.DispatchTG(r.pAttnFARows, nFA*g.nKV*maxS*128, 128, shmBytes, b.qkvB, kcAll, vcAll, b.faPartB, g.uNKV, r.uAttnFAG,
				b.nKeysB, r.uScale, L.uWindow, b.nSplitB, b.uCount[nFA], b.rowmapFA, b.slotOffB, b.uQKV, b.uPartStride, b.uMaxS)
			e.Dispatch(r.pAttnFACombineRows, nFA*r.nH*g.hd, g.hd, b.faPartB, b.ctxB, r.uAttnFAG, g.uHd, b.nSplitB, r.uNH,
				b.uCount[nFA], b.rowmapFA, b.uPartStride, g.uNHhd)
		}
		for m, q := range seqs {
			if !faRow[m] || faRows {
				continue
			}
			sb := r.kvSlotBufs[q.slot]
			nSplit := r.attnFASplitFor(q.pos+1, g.nKV)
			shmBytes := 128 * 6 * (r.nH / g.nKV) * 4
			e.DispatchTG(r.pAttnFA, g.nKV*nSplit*128, 128, shmBytes, b.qkvB.At(4*m*b.qkvRows), sb.kc[l], sb.vc[l], r.attnFAPartial,
				g.uNKV, r.uAttnFAG, b.uNKeys[m], r.uScale, L.uWindow, b.uFANSplit[m])
			e.Dispatch(r.pAttnFACombine, r.nH*g.hd, g.hd, r.attnFAPartial, f32(b.ctxB, m, nHhd), r.uAttnFAG, g.uHd, b.uFANSplit[m])
		}
		if nPlain > 0 && r.kvI8 {
			e.Dispatch(r.pAttnI8Rows, nPlain*r.nH*tgReduceAttn, tgReduceAttn, b.qkvB, kcAll, vcAll, r.kvSlotBufs[0].ks[l], r.kvSlotBufs[0].vs[l],
				b.ctxB, r.uNH, g.uNKV, g.uHd, b.nKeysB, r.uScale, L.uWindow, b.uCount[nPlain], b.rowmapB, b.slotOffB)
		} else if nPlain > 0 {
			e.Dispatch(r.pAttnRows, nPlain*r.nH*tgReduceAttn, tgReduceAttn, b.qkvB, kcAll, vcAll, b.ctxB, r.uNH, g.uNKV, g.uHd,
				b.nKeysB, r.uScale, L.uWindow, L.attnSinks, L.uHasSink, b.uCount[nPlain], b.rowmapB, b.slotOffB)
		}
		if r.q4kLane { // o-proj + residual reads the f32 ctx rows directly
			r.q4kStepGemv(e, B, L.oSegs, Buffer{}, b.ctxB, nHhd, b.xB, H, g.uNHhd, true)
		} else {
			e.Dispatch(r.pQvRows, B*256, 256, b.ctxB, b.cqB, b.cScB, g.uNHhd)
			if w8AttnRows {
				e.Dispatch(r.pW8Rows[B], H*32, 32, L.oW, L.oS, b.cqB, b.cScB, b.xB, g.uNHhd, r.uH, b.uCount[B], b.noBias, b.uMode[2])
			} else if w8Attn { // int8 slice 3: decode's o-proj + residual (gemv_w8a8_sa_resid) once per row
				for m := range B {
					e.DispatchTG(r.pSAResid, H*32, 256, nHhd*2, L.oW, L.oS, b.cqB.At(m*nHhd), b.cScB.At(4*m), f32(b.xB, m, H), g.uNHhd)
				}
			} else {
				pack(e, b.cqB, nHhd, g.uNHhd)
				e.DispatchTG(b.bt, H/16*128, 128, tgb, L.oW, L.oS, b.aT, b.cScB, b.xB, g.uNHhd, r.uH, b.uM, b.noBias, b.uMode[2])
			}
		}
		if r.q4kLane { // norm, fused gate|up, SwiGLU, down + residual, all f32
			e.Dispatch(r.pRmsF32OutRows, B*tgReduceNorm, tgReduceNorm, b.xB, L.postNorm, b.axB, r.uH, r.uEps, r.uAddOne)
			r.q4kStepGemv(e, B, L.guSegs, Buffer{}, b.axB, H, b.guB, 2*I, r.uH)
			e.Dispatch(r.pSwF32Rows, B*I, 256, b.guB, b.guB, b.swB, r.uI, r.uAct, b.uM)
			r.q4kStepGemv(e, B, L.dSegs, Buffer{}, b.swB, I, b.xB, H, r.uI, true)
			continue
		}
		e.Dispatch(r.pRmsRows, B*tgReduceNorm, tgReduceNorm, b.xB, L.postNorm, b.mqB, b.mScB, r.uH, r.uEps, r.uAddOne)
		if w8Rows {
			e.Dispatch(r.pW8Rows[B], 2*I*32, 32, L.guW, L.guS, b.mqB, b.mScB, b.guB, r.uH, b.uGU, b.uCount[B], b.noBias, b.uMode[0])
		} else if rowsGU && mc3AdjRowsOn && B >= 2 && r.gemvRows.gu > 0 {
			e.DispatchTG(r.pSARowsAdj, B*ng, 256, H*2, L.guW, L.guS, b.mqB, b.mScB, b.guB, r.uH, b.uGU, b.uCount[B])
		} else if rowsGU {
			for m := range B {
				e.DispatchTG(pg, ng, 256, H*2, L.guW, L.guS, b.mqB.At(m*H), b.mScB.At(4*m), f32(b.guB, m, 2*I), r.uH)
			}
		} else {
			pack(e, b.mqB, H, r.uH)
			e.DispatchTG(b.bt, 2*I/16*128, 128, tgb, L.guW, L.guS, b.aT, b.mScB, b.guB, r.uH, b.uGU, b.uM, b.noBias, b.uMode[0])
		}
		e.Dispatch(r.pSwRows, B*256, 256, b.guB, b.guB, b.dqB, b.dScB, r.uI, r.uAct)
		if w8Rows {
			e.Dispatch(r.pW8Rows[B], H*32, 32, L.dW, L.dS, b.dqB, b.dScB, b.xB, r.uI, r.uH, b.uCount[B], b.noBias, b.uMode[2])
		} else if r.w8 { // int8 slice 3: decode's down-proj + residual (gemv_w8a8_body_resid) once per row
			for m := range B {
				e.Dispatch(r.pGemvResid, H*32, 32, L.dW, L.dS, b.dqB.At(m*I), b.dScB.At(4*m), f32(b.xB, m, H), r.uI)
			}
		} else {
			pack(e, b.dqB, I, r.uI)
			e.DispatchTG(b.btd, H/16*128, 128, tgb, L.dW, L.dS, b.aT, b.dScB, b.xB, r.uI, r.uH, b.uM)
		}
	}
	if out == multiNoHead { // R-19: the caller reads nothing from this step but the K/V it wrote
		e.End()
		r.recordExecErr(e.Err())
		if err := r.takeExecErr(); err != nil {
			return nil, nil, err
		}
		r.gpuStart, r.gpuEnd, r.kernStart, r.kernEnd = e.GPUStart(), e.GPUEnd(), e.KernStart(), e.KernEnd()
		b.steps, b.seqs, b.maxSeqs = b.steps+1, b.seqs+B, max(b.maxSeqs, B)
		b.noHeadSteps++
		return nil, nil, nil
	}
	if r.q4kLane { // decode's head (gemv_w8f32_head) on each row's f32 final norm
		e.Dispatch(r.pRmsF32OutRows, B*tgReduceNorm, tgReduceNorm, b.xB, r.finalNorm, b.axB, r.uH, r.uEps, r.uAddOne)
		for m, q := range seqs {
			if out == multiLastRow && m != B-1 && q.draw == nil {
				continue // R-19: only the last row is read
			}
			e.Dispatch(r.pGemvW8, r.V*32, 32, f32(b.axB, m, H), r.aSc, r.lmW, r.lmS, f32(b.logitsB, m, r.V), r.uH)
		}
	} else {
		e.Dispatch(r.pRmsRows, B*tgReduceNorm, tgReduceNorm, b.xB, r.finalNorm, b.aqB, b.aScB, r.uH, r.uEps, r.uAddOne)
		e.Dispatch(b.packLM, H*8, 256, b.aqB, b.aTp, b.uM, r.uH)
		e.DispatchTG(b.lm, r.V/128*128, 128, 0, r.lmW, r.lmS, b.aTp, b.aScB, b.logitsB, r.uH, b.uV, b.uM)
	}
	// Device draws: ForwardSample's two gumbel dispatches, exactly, on each drawing row of logitsB (the same bits as
	// r.logits would hold) with that row's uniforms. The partial buffers are reused row after row — dispatches in one
	// encoder run in order.
	greedyDraw := make([]bool, B)
	for m, q := range seqs {
		if q.draw == nil {
			continue
		}
		invT := float32(1 / q.draw.Temperature)
		if math.IsInf(float64(invT), 0) { // ForwardSample's own shortcut: the argmax, as greedy
			greedyDraw[m] = true
			continue
		}
		b.uGInvT[m].Floats()[0] = invT
		b.uGK0[m].SetU32(uint32(q.draw.Seed))
		b.uGK1[m].SetU32(uint32(q.draw.Seed >> 32))
		b.uGD0[m].SetU32(uint32(q.draw.Draw))
		b.uGD1[m].SetU32(uint32(q.draw.Draw >> 32))
		const gbThreads = 256
		const gbShmBytes = gbThreads * 2 * 4
		e.DispatchTG(r.pGumbel1, r.gumbelNB*gbThreads, gbThreads, gbShmBytes,
			b.logitsB.At(4*m*r.V), r.uGumbelV, b.uGInvT[m], b.uGK0[m], b.uGK1[m], b.uGD0[m], b.uGD1[m],
			r.gumbelBKey, r.gumbelBIdx)
		e.DispatchTG(r.pGumbel2, gbThreads, gbThreads, gbShmBytes,
			r.gumbelBKey, r.gumbelBIdx, r.uGumbelNB, b.gOut.At(4*m))
	}
	devArg := make([]bool, B) // E-P08: rows whose id is the argmax of the raw row, taken on the GPU
	if mc3DeviceArgmaxOn {
		for m, q := range seqs {
			if !argmaxOnly && !greedyDraw[m] {
				continue // a full-logits row (the caller reads the row) or a sampled one (the gumbel draw gives its id)
			}
			if !argmaxOnly && q.draw.Greedy && r.logitTransform() {
				continue // a greedy generation's id is the argmax of the transformed row: taken on the host below
			}
			devArg[m] = true
			e.DispatchTG(r.pArgRowsPart, batchArgmaxParts*256, 256, 0, b.logitsB.At(4*m*r.V), b.amaxPart.At(8*m*batchArgmaxParts), b.uV, b.uAmaxP)
			e.DispatchTG(r.pArgFinish, 256, 256, 0, b.amaxPart.At(8*m*batchArgmaxParts), b.amaxTok.At(4*m), b.uAmaxP)
		}
	}
	e.End()
	r.recordExecErr(e.Err())
	if err := r.takeExecErr(); err != nil {
		return nil, nil, err
	}
	r.gpuStart, r.gpuEnd, r.kernStart, r.kernEnd = e.GPUStart(), e.GPUEnd(), e.KernStart(), e.KernEnd()
	b.steps, b.seqs, b.maxSeqs = b.steps+1, b.seqs+B, max(b.maxSeqs, B)
	logits, ids = make([][]float32, B), make([]int, B)
	lg := b.logitsB.Floats()
	gid := b.gOut.U32s()
	devID := func(m int) int { // the GPU's argmax, or argmaxF32's answer for a row that starts with NaN (0)
		if math.IsNaN(float64(lg[m*r.V])) {
			return 0
		}
		return int(b.amaxTok.U32s()[m])
	}
	if argmaxOnly {
		for m := range seqs {
			if devArg[m] {
				ids[m] = devID(m)
			} else {
				ids[m] = argmaxF32(lg[m*r.V : (m+1)*r.V])
			}
		}
		return nil, ids, nil
	}
	for m, q := range seqs {
		ids[m] = -1
		if q.draw != nil {
			id := int32(gid[m])
			if greedyDraw[m] && devArg[m] {
				id = int32(devID(m))
				if q.draw.Greedy {
					b.greedyDevRows++
				}
			} else if greedyDraw[m] && q.draw.Greedy { // E-P08 with a softcap or logit scale: the transformed row's argmax
				id = int32(argmaxF32(r.transformedRow(lg[m*r.V : (m+1)*r.V])))
			} else if greedyDraw[m] || id < 0 { // ForwardSample's fallbacks: the argmax of the raw row
				id = int32(argmaxF32(lg[m*r.V : (m+1)*r.V]))
			}
			ids[m] = int(id)
			continue
		}
		if out == multiLastRow && m != B-1 {
			continue // R-19: only the last row is read
		}
		logits[m] = r.transformedRow(lg[m*r.V : (m+1)*r.V])
	}
	return logits, ids, nil
}

// q4kStepGemv is q4kGemv for the B rows of a batched step: activation row m at ax + m*lda floats, output row m at
// out + m*ldo, each segment into its own rows of every output row, each weight row read once for all B rows. A non-empty
// bias is added per row, as decode's bias variant does; resid accumulates. Every row is decode's arithmetic
// (batch_rows.go, q4kRowsSrc).
func (r *resident) q4kStepGemv(e *Encoder, B int, segs []q4kSeg, bias, ax Buffer, lda int, out Buffer, ldo int, uK Buffer, resid ...bool) {
	b := r.batch
	mode, bs := b.uMode[0], b.noBias
	switch {
	case len(resid) > 0 && resid[0]:
		mode = b.uMode[2]
	case bias != (Buffer{}):
		mode = b.uMode[1]
	}
	uLda, uLdo := b.uLd[lda], b.uLd[ldo]
	if uLda == (Buffer{}) || uLdo == (Buffer{}) {
		panic(fmt.Sprintf("metal: q4k step: no stride uniform for %d or %d", lda, ldo))
	}
	for _, sg := range segs {
		o := out.At(sg.off * 4)
		if bias != (Buffer{}) {
			bs = bias.At(sg.off * 4)
		}
		if sg.q4k {
			e.Dispatch(r.pQ4KRows[B], sg.rows*32, 256, sg.w, ax, o, uK, sg.uN, uLda, uLdo, bs, mode)
		} else {
			e.Dispatch(r.pW8F32Rows[B], sg.rows*32, 256, sg.w, sg.s, ax, o, uK, sg.uN, uLda, uLdo, bs, mode)
		}
	}
}

// logitTransform reports whether a step's logits row is transformed on the host before it is returned (a final softcap
// or a logit scale), which the device argmax over the raw row would not see.
func (r *resident) logitTransform() bool {
	return r.finalSoftcap > 0 || (r.logitScale != 0 && r.logitScale != 1)
}

// transformedRow is a copy of one raw logits row of a step with the host's transforms applied: the row forwardMulti
// returns.
func (r *resident) transformedRow(raw []float32) []float32 {
	row := make([]float32, r.V)
	copy(row, raw)
	if r.finalSoftcap > 0 {
		softcapParallel(row, r.finalSoftcap)
	}
	if r.logitScale != 0 && r.logitScale != 1 {
		for i := range row {
			row[i] *= r.logitScale
		}
	}
	return row
}

// BatchGreedyDraw is decoder.ResidentBatchGreedy's (E-P08): a step returns a Greedy draw row's argmax id, the argmax of
// the row forwardMulti would have returned, first maximum on a tie.
func (a *metalResident) BatchGreedyDraw() bool { return a.r.batch != nil && len(a.r.kvSlotBufs) >= 2 }

// BatchStepRange is decoder.ResidentBatchStepper's: the batch sizes StepBatch serves well (batchMinSeqs..the smaller
// of batchMaxSeqs and the slot count), or (0, 0) when this resident cannot batch.
func (a *metalResident) BatchStepRange() (lo, hi int) {
	if a.r.batch == nil || len(a.r.kvSlotBufs) < 2 {
		return 0, 0
	}
	return batchMinSeqs, min(batchMaxSeqs, len(a.r.kvSlotBufs))
}

// StepBatch is decoder.ResidentBatchStepper's: one decode token for each sequence, each on its own KV slot.
func (a *metalResident) StepBatch(seqs []decoder.ResidentBatchSeq) ([]decoder.ResidentBatchOut, error) {
	bs := make([]batchSeq, len(seqs))
	for i, q := range seqs {
		bs[i] = batchSeq{slot: q.Slot, pos: q.Pos, emb: q.Emb, draw: q.Draw}
	}
	logits, ids, err := a.r.forwardMulti(bs)
	if err != nil {
		return nil, err
	}
	out := make([]decoder.ResidentBatchOut, len(seqs))
	for i := range out {
		out[i] = decoder.ResidentBatchOut{Logits: logits[i], ID: ids[i]}
	}
	return out, nil
}

// stepVerifyCost is the step-kernel verify's cost in single-token steps, indexed by rows verified: the conservative end of
// TestMC3Verify_rowCost (the dearest cell measured). 1 row is production's own Forward; 3 and 5-7 are interpolated; 9 rows
// are an 8-row step plus one Forward. Derivation: docs/code-notes/metal.md#stepVerifyCost.
var stepVerifyCost = []float64{0, 1, 1.79, 1.95, 2.11, 2.25, 2.38, 2.52, 2.65, 3.65}

var (
	_ decoder.VerifyCostReporter = (*metalResident)(nil)
)

// stepPrefill reports whether a prompt can prefill on the step kernels: the step exists and no logit transform separates
// its rows from Forward's. One KV slot is enough, since a prompt is one sequence's consecutive rows.
func (a *metalResident) stepPrefill() bool {
	return a.r != nil && a.r.batch != nil && len(a.r.kvSlotBufs) > 0 && a.r.finalSoftcap == 0 && (a.r.logitScale == 0 || a.r.logitScale == 1)
}

// VerifyCost (decoder.VerifyCostReporter): the argmax-only verify's cost curve when this resident can run it on the
// step kernels, nil otherwise. It needs two KV slots: a one-slot resident prefills on the step but gives speculation no
// verify cost, and enabling one there is its own decision with its own gate.
func (a *metalResident) VerifyCost() []float64 {
	if !a.stepPrefill() || len(a.r.kvSlotBufs) < 2 {
		return nil
	}
	return stepVerifyCost
}

// PrefillLastNArgmax is the n-gram loop's argmax-only verify (decoder spec_ngram.go, idsVerify) on the step kernels:
// the rows are consecutive positions of the bound slot's sequence from startPos, run by forwardMulti, which is
// bit-identical to production decode for such rows (TestMC3Verify_sameSlotRowsBitIdentical), in pieces of at most
// batchMaxSeqs, the fragment's width. A one-row piece runs production's own Forward, so a round with no draft costs
// what plain decode does. Each id is the row's first maximum, as the decoder's own argmax.
func (a *metalResident) PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error) {
	r := a.r
	if a.VerifyCost() == nil {
		return nil, fmt.Errorf("metal: no step-kernel verify on this resident")
	}
	if e := a.checkCap(startPos, len(embeddings)); e != nil {
		return nil, e
	}
	ids := make([]int, 0, len(embeddings))
	for from := 0; from < len(embeddings); {
		n := min(batchMaxSeqs, len(embeddings)-from)
		if n == 1 {
			lg, err := a.Forward(embeddings[from], startPos+from)
			if err != nil {
				return nil, err
			}
			ids = append(ids, argmaxF32(lg))
		} else {
			seqs := make([]batchSeq, n)
			for i := range n {
				seqs[i] = batchSeq{slot: r.kvSlot, pos: startPos + from + i, emb: embeddings[from+i]}
			}
			_, got, err := r.forwardMultiInto(seqs, true)
			if err != nil {
				return nil, err
			}
			ids = append(ids, got...)
		}
		from += n
	}
	return ids, nil
}

// promptStepMaxAboveFloor is the longest suffix the above-floor half of the step route takes (promptStepAboveFloor): the
// step is faster than the pass only up to this length, so it stops at the last K measured faster
// (docs/code-notes/metal.md#promptStepMaxAboveFloor).
const promptStepMaxAboveFloor = 32

// promptStepAboveFloor turns on E-P01's above-floor half: a suffix of up to promptStepMaxAboveFloor tokens on a prompt
// that reaches the floor also runs on the step. OFF, and test-only, until the owner decides: unlike the below-floor half
// it changes numerics. The pass computes a suffix's K/V from f16 activations and the step from decode's int8 ones, so
// a reuse turn would no longer equal a cold prefill of the same prompt (TestPromptStepAboveFloor_reuseVsCold measures
// by how much), and a chunked prefill's short tail would differ from the whole pass unless the decoder keeps tails
// longer than promptStepMaxAboveFloor.
var promptStepAboveFloor = false

// promptStepOK reports whether PrefillLast takes the step route for n tokens at startPos: the resident prefills on the step
// (stepPrefill: the batched step exists, and no logit transform separates its rows from Forward's), and the prompt ends
// below the fast-prefill floor, where the batched pass declines and the decoder would run the sequential loop. The step's
// rows are that loop's bits, so this route changes no output. Above the floor the route is promptStepAboveFloor, off: it
// would replace the pass's numerics with decode's, and a chunked prefill's short tail must equal the whole pass
// (TestMC5_prefillChunkInvariance; docs/code-notes/metal.md#promptStepOK).
func (a *metalResident) promptStepOK(n, startPos, floor int) bool {
	if a.r.promptStepOff || n < 2 || !a.stepPrefill() {
		return false
	}
	if !a.r.prefillOK || a.r.kvF32 { // the pass never runs here (the q4k lane; an f32 cache), so every prompt would otherwise take the sequential loop
		return true
	}
	if floor > 0 && startPos+n < max(floor, metalStepPrefillCeiling) { // A-P02: the step keeps prompts where it beats the pass
		return true
	}
	return promptStepAboveFloor && n <= promptStepMaxAboveFloor
}

// prefillByStep is PrefillLast on the step kernels: the prompt's positions run as consecutive rows of the bound slot's
// sequence, batchMaxSeqs at a time, exactly as PrefillLastNArgmax runs a verify, and the last row's logits come back.
// forwardMulti is bit-identical to production decode for such rows (TestMC3Verify_sameSlotRowsBitIdentical,
// TestMC3Step_promptInRowsBitIdentical), so this is the sequential path's numerics in a fraction of its time. Pieces before
// the last run no head at all (multiNoHead), and the last piece copies out only its last row (multiLastRow); a one-row last
// piece runs production's own Forward. Cancellation is checked between pieces.
// History: docs/code-notes/metal.md#prefillByStep.
func (a *metalResident) prefillByStep(ctx context.Context, embeddings [][]float32, startPos int) (logits []float32, err error) {
	return a.prefillByStepSrc(ctx, sliceRows(embeddings), startPos)
}

// prefillByStepSrc is prefillByStep over a row source (rows.go), read one piece at a time.
func (a *metalResident) prefillByStepSrc(ctx context.Context, src rowSource, startPos int) (logits []float32, err error) {
	r := a.r
	defer func() {
		if p := recover(); p != nil {
			logits, err = nil, fmt.Errorf("metal: step prefill aborted: %v", p)
		}
	}()
	for from := 0; from < src.n; {
		if e := ctx.Err(); e != nil {
			return nil, e
		}
		n := min(batchMaxSeqs, src.n-from)
		last := from+n == src.n
		piece := src.rows(from, from+n)
		if n == 1 {
			lg, err := a.Forward(piece[0], startPos+from)
			if err != nil {
				return nil, err
			}
			logits = lg
		} else {
			seqs := make([]batchSeq, n)
			for i := range n {
				seqs[i] = batchSeq{slot: r.kvSlot, pos: startPos + from + i, emb: piece[i]}
			}
			mode := multiNoHead // R-19: a piece before the last needs only its K/V
			if last {
				mode = multiLastRow
			}
			if stepPrefillAllHeads { // the route as it was, for the test that compares the two
				mode = multiArgmax
				if last {
					mode = multiFull
				}
			}
			rows, _, err := r.forwardMultiOut(seqs, mode)
			if err != nil {
				return nil, err
			}
			if last {
				logits = rows[n-1]
			}
		}
		from += n
	}
	if err := r.takeExecErr(); err != nil {
		return nil, err
	}
	return logits, nil
}
