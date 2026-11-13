//go:build cuda

package cuda

import (
	"context"
	"fmt"
	"math"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// One decode token for each of several sequences, each on its own resident KV slot at its own position, in one step
// (docs/tasks/parked/task-concurrency-2026-09.md). The step is prefillCore's batched layer stack in its multi-sequence mode
// (rows != nil): rms+quant, the q/k/v, o, gate/up and down GEMVs, the batched qk-norm, SwiGLU and residuals run over the
// B rows through the exact kernels (forceExactKernels holds for tailAllLogits), bit-identical to decode per row. Rope,
// the KV store and attention run per sequence through decode's own gap (decodeAttnGap), so every row gets exactly the
// kernel and position-dependent scales its own decode would, the flash-decode lane past 2048 keys included. The head is
// per row, decode's, with Forward's host tail; a row whose token is drawn on-device ends in ForwardSample's pick.

var _ decoder.ResidentBatchStepper = (*cudaResident)(nil)

// stepRow is one sequence of a step: its KV slot, the position it decodes, and its on-device draw (nil: logits).
type stepRow struct {
	slot, pos int
	draw      *decoder.ResidentBatchDraw
}

// batchStepStaticDeclineOffForTest skips BatchStepRange's batched-prefill decline, so a test can see the recurrent check above it on its own: the prefill
// decline also refuses a Mamba-2 model, and would hide a guard left DeltaNet-only. Set by tests only.
var batchStepStaticDeclineOffForTest bool

// BatchStepRange reports the sequences a step serves: 2 up to the slot count, or none (hi = 0) for a model the step's
// table (program.CUDAStep: no MoE, Gemma 4 MoE block, MLA or recurrent mixer) does not admit, or that has no batched prefill,
// streams its experts (C′) or holds one slot. Those keep one generation at a time.
func (r *cudaResident) BatchStepRange() (lo, hi int) {
	if !r.prefillReady || r.cacheExperts || r.KVSlots() < 2 || program.CUDAStep.Check(r.prog) != nil {
		return 0, 0
	}
	if !batchStepStaticDeclineOffForTest && r.prefillStaticDecline() != nil {
		return 0, 0
	}
	return 2, r.KVSlots()
}

// StepBatch runs one decode token for every sequence in seqs (decoder.ResidentBatchStepper). Each row equals that
// sequence's own Forward on its slot bit for bit, or ForwardSample's id when it carries a Draw. The bound slot's binding
// is restored before it returns.
func (r *cudaResident) StepBatch(seqs []decoder.ResidentBatchSeq) ([]decoder.ResidentBatchOut, error) {
	if len(seqs) == 0 {
		return nil, nil
	}
	if lo, hi := r.BatchStepRange(); hi == 0 || len(seqs) < lo || len(seqs) > hi {
		return nil, fmt.Errorf("cuda step: %d sequences outside the step range", len(seqs))
	}
	embs := make([][]float32, len(seqs))
	rows := make([]stepRow, len(seqs))
	for i, s := range seqs {
		if s.Slot < 0 || s.Slot >= r.KVSlots() {
			return nil, fmt.Errorf("cuda step: sequence %d names KV slot %d of %d", i, s.Slot, r.KVSlots())
		}
		embs[i] = s.Emb
		rows[i] = stepRow{slot: s.Slot, pos: s.Pos, draw: s.Draw}
	}
	outs, ids, err := r.prefillCore(context.Background(), embs, 0, tailAllLogits, nil, nil, rows)
	if err != nil {
		return nil, err
	}
	res := make([]decoder.ResidentBatchOut, len(seqs))
	for i := range seqs {
		if rows[i].draw != nil {
			res[i] = decoder.ResidentBatchOut{ID: ids[i]}
		} else {
			res[i] = decoder.ResidentBatchOut{Logits: outs[i], ID: -1}
		}
	}
	return res, nil
}

var _ decoder.ResidentRaggedStepper = (*cudaResident)(nil)

// raggedPlantForTest plants a defect in StepRagged's rows, for its identity test's controls: 1 binds a sequence's second row
// to the next sequence's slot (a row reading another sequence's tables), 2 gives each row of a sequence its first row's
// position (a verify row run as a decode row), 3 runs a prefill chunk on the exact kernels (a chunk out of its pass's class),
// 4 reads the ragged rows' tables with a stride one entry short, so slot s's row reads its table s entries back: a row's table
// offset off by blocks, inside the tables the device holds. 0 in production. Set by tests only.
var raggedPlantForTest int

// raggedChunk is a ragged step's prefill chunk while its pass runs (StepRagged sets it, prefillCore reads it): the pass's
// first rows rows, at pos in slot, of a prompt promptLen long; last when it is the prompt's final pass, whose last row alone
// is headed; exact when its own pass takes the exact kernels. split holds while the layer stack runs, when bGemvB and the
// attention give the chunk's rows its class.
type raggedChunk struct {
	rows, pos, slot, promptLen int
	last, exact, split         bool
}

// headFrom is the first row of a step's pass whose logits are read: the chunk's rows are not, but for a final pass's last.
func (c raggedChunk) headFrom() int {
	if c.last {
		return c.rows - 1
	}
	return c.rows
}

// StepRagged runs one step over seqs (decoder.ResidentRaggedStepper): every sequence's rows in one pass of prefillCore's
// multi-sequence mode, a sequence's rows in order, so each row's attention reads the rows before it. Each row runs the exact
// kernels and decode's attention at its own slot and position, as a decode token does and as a batched verify does
// (forceExactKernels under tailAllLogits, TestSpecDecodeCurve's lossless check). The model and slot count are StepBatch's;
// a windowed model takes a sequence's rows only where they fit its window's slack (kvRoomFor).
func (r *cudaResident) StepRagged(seqs []decoder.ResidentRaggedSeq) ([]decoder.ResidentRaggedOut, error) {
	if len(seqs) == 0 {
		return nil, fmt.Errorf("cuda ragged step: no sequences")
	}
	// The range's upper bound is the KV slots; its lower bound is MC3's StepBatch, and a ragged step of one sequence is a step.
	if _, hi := r.BatchStepRange(); hi == 0 || len(seqs) > hi {
		return nil, fmt.Errorf("cuda ragged step: %d sequences outside the step range", len(seqs))
	}
	order, err := raggedOrder(seqs)
	if err != nil {
		return nil, err
	}
	if seqs[order[0]].Prefill != nil {
		if e := r.raggedChunkDecline(); e != nil {
			return nil, e
		}
	}
	if s := seqs[order[0]]; s.Prefill != nil {
		r.stepChunk = raggedChunk{rows: len(s.Embs), pos: s.Pos, slot: s.Slot, promptLen: s.Prefill.PromptLen, last: s.Prefill.Last, exact: raggedPlantForTest == 3}
		defer func() { r.stepChunk = raggedChunk{} }()
	}
	var embs [][]float32
	var rows []stepRow
	seen := make(map[int]bool, len(seqs))
	for _, i := range order {
		s := seqs[i]
		switch {
		case s.Slot < 0 || s.Slot >= r.KVSlots():
			return nil, fmt.Errorf("cuda ragged step: sequence %d names KV slot %d of %d", i, s.Slot, r.KVSlots())
		case seen[s.Slot]:
			return nil, fmt.Errorf("cuda ragged step: KV slot %d is named by two sequences", s.Slot)
		case len(s.Embs) == 0:
			return nil, fmt.Errorf("cuda ragged step: sequence %d has no rows", i)
		case !r.kvRoomFor(s.Pos, len(s.Embs)):
			return nil, fmt.Errorf("cuda ragged step: sequence %d's %d rows at position %d do not fit the windowed KV (window %d + %d slack)",
				i, len(s.Embs), s.Pos, r.kvWindow, r.kvSlack)
		}
		seen[s.Slot] = true
		for j, e := range s.Embs {
			rw := stepRow{slot: s.Slot, pos: s.Pos + j}
			switch {
			case raggedPlantForTest == 1 && j == 1:
				rw.slot = seqs[(i+1)%len(seqs)].Slot
			case raggedPlantForTest == 2:
				rw.pos = s.Pos
			}
			if j == len(s.Embs)-1 {
				rw.draw = s.Draw
			}
			embs = append(embs, e)
			rows = append(rows, rw)
		}
	}
	outs, ids, err := r.prefillCore(context.Background(), embs, 0, tailAllLogits, nil, nil, rows)
	if err != nil {
		return nil, err
	}
	res := make([]decoder.ResidentRaggedOut, len(seqs))
	m := 0
	for _, i := range order {
		s := seqs[i]
		res[i] = decoder.ResidentRaggedOut{Logits: make([][]float32, len(s.Embs)), ID: -1}
		for j := range s.Embs {
			switch {
			case rows[m].draw != nil:
				res[i].ID = ids[m]
			case s.Prefill == nil || s.Prefill.Last && j == len(s.Embs)-1:
				res[i].Logits[j] = outs[m]
			}
			m++
		}
	}
	return res, nil
}

// raggedOrder is StepRagged's sequence order and the checks it makes on the sequences alone: a prefill chunk first (at most one,
// with no draw, its rows inside its prompt and ending it exactly when it is the prompt's Last pass, so its class and its head are
// its pass's), then the others in their order.
func raggedOrder(seqs []decoder.ResidentRaggedSeq) ([]int, error) {
	order := make([]int, 0, len(seqs))
	for i, s := range seqs {
		if s.Prefill == nil {
			continue
		}
		switch {
		case len(order) > 0:
			return nil, fmt.Errorf("cuda ragged step: sequences %d and %d are both prefill chunks; a step takes one", order[0], i)
		case s.Draw != nil:
			return nil, fmt.Errorf("cuda ragged step: sequence %d is a prefill chunk with a draw", i)
		case s.Prefill.PromptLen < s.Pos+len(s.Embs):
			return nil, fmt.Errorf("cuda ragged step: sequence %d's chunk ends at %d, past its prompt's %d", i, s.Pos+len(s.Embs), s.Prefill.PromptLen)
		case s.Prefill.Last != (s.Pos+len(s.Embs) == s.Prefill.PromptLen):
			return nil, fmt.Errorf("cuda ragged step: sequence %d's chunk ends at %d of a %d-row prompt with Last %v", i, s.Pos+len(s.Embs), s.Prefill.PromptLen, s.Prefill.Last)
		}
		order = append(order, i)
	}
	for i, s := range seqs {
		if s.Prefill == nil {
			order = append(order, i)
		}
	}
	return order, nil
}

// raggedChunkDecline is why this model's step cannot take a prefill chunk, or nil: the chunk runs prefillCore's attention and
// dense GEMVs over its rows, so a recurrent mixer, a routed-expert layer and a windowed KV (whose slot base the step's rows
// do not share) stay with their own passes.
func (r *cudaResident) raggedChunkDecline() error {
	if r.recurrent() || r.kvWin {
		return fmt.Errorf("cuda ragged step: a prefill chunk on a recurrent or windowed-KV model: %w", errPrefillDeclined)
	}
	for l := range r.layers {
		if r.layers[l].isMoE || r.layers[l].g4moe {
			return fmt.Errorf("cuda ragged step: a prefill chunk on a routed-expert model (layer %d): %w", l, errPrefillDeclined)
		}
	}
	return nil
}

// raggedRows is a step's decode and verify rows on the device for the ragged launches (stepAttnRows): each row's KV slot and
// position, its post-RoPE query scale (nil Buffer: 1 on every row), and, per class of layers (raggedLayerClasses), which rows take
// which attention launch. Built once per pass; hostPos is the host copy of the positions.
type raggedRows struct {
	slot, pos, qTemp Buffer
	hostPos          []int
	classOf          []int // layer -> index into classes
	classes          []raggedClass
	// skScores and skInv are the split-KV rows' scratch: skRows rows of nH x skStride scores and nH inverse denominators per
	// launch, so a class's split rows go in chunks of skRows.
	skScores, skInv  Buffer
	skRows, skStride int
}

// raggedClass is one class of layers' split of a step's rows by decode's attention launch there (decodeAttnArm): the rows taking
// attn_batched (one attn_ragged launch), the flash-decode lane (fa_partial_ragged and fa_combine_ragged) and split-KV (its three
// ragged launches), as device lists of row indices, and the rest (the glue attention, a refused span, split-KV's opt-in split
// V-sum, or rows the margin has no room for), which take decode's own launches row by row.
type raggedClass struct {
	batched, flash, split Buffer
	nB, nF, nS            int
	maxWin, maxWinS       int // the widest span among the batched rows, and among the split-KV rows
	other                 []int
}

// raggedLayerClasses groups the layers by what decodeAttnArm reads of a layer (its window, KV heads, head dim and flash-lane
// eligibility), so a step's rows split once per class instead of once per layer: classOf maps each layer to its class's first layer.
func (r *cudaResident) raggedLayerClasses() (reps, classOf []int) {
	type key struct {
		window     int32
		nKV, hd    int
		faEligible bool
	}
	seen := map[key]int{}
	classOf = make([]int, len(r.layers))
	for l := range r.layers {
		Ly := &r.layers[l]
		k := key{Ly.window, Ly.nKV, Ly.hd, r.faEligible(l)}
		c, ok := seen[k]
		if !ok {
			c = len(reps)
			seen[k] = c
			reps = append(reps, l)
		}
		classOf[l] = c
	}
	return reps, classOf
}

// raggedRowsWords is the 4-byte words raggedRowsFor allocates for n rows: each row's slot and position, its query scale when the
// family has one, and three n-entry row lists per layer class; 0 when the ragged kernels are not loaded. Split-KV rows' score
// scratch comes beside it, only as much as the margin's prefill pass leaves (raggedRowsFor).
func (r *cudaResident) raggedRowsWords(n int) int {
	if n == 0 || r.bRopeKVRagged == (Pipeline{}) || r.bAttnRagged == (Pipeline{}) {
		return 0
	}
	w := 2 * n
	if r.attnTempBeta != 0 {
		w += n
	}
	reps, _ := r.raggedLayerClasses()
	return w + 3*n*len(reps)
}

// raggedSKReady reports whether split-KV's ragged twins are loaded and decode's split-KV is the bit-identical V-sum (not the
// opt-in split one), so a step's split-KV rows can take one launch of each.
func (r *cudaResident) raggedSKReady() bool {
	return r.skScoresRagged != (Pipeline{}) && r.skSoftmaxRagged != (Pipeline{}) && r.skVsumRagged != (Pipeline{}) && r.skVsumSplit < 1
}

// raggedFAReady reports whether the flash-decode lane's ragged twins are loaded, so a step's lane rows can take one launch.
func (r *cudaResident) raggedFAReady() bool {
	return r.faCombineRagged != (Pipeline{}) && r.faRagged != [3]Pipeline{}
}

// raggedRowsFor uploads rows into one buffer from ai (prefillCore's scratch, so StepScratchBytes prices it), or returns nil when
// the ragged kernels are not loaded, so every row takes decode's own gap. passRows is the pass's whole width (a prefill chunk's rows
// included), whose scratch is already allocated: the split-KV scores take only what the margin's 512-row pass leaves beside it.
func (r *cudaResident) raggedRowsFor(rows []stepRow, passRows int, ai, af func(int) Buffer) (*raggedRows, error) {
	n := len(rows)
	words := r.raggedRowsWords(n)
	if words == 0 {
		return nil, nil
	}
	b := ai(words)
	host := make([]int32, words)
	rg := &raggedRows{slot: b, pos: b.At(4 * n), hostPos: make([]int, n)}
	off := 2 * n
	if r.attnTempBeta != 0 {
		rg.qTemp = b.At(4 * off)
		off += n
	}
	for i, rw := range rows {
		host[i], host[n+i], rg.hostPos[i] = int32(rw.slot), int32(rw.pos), rw.pos
		if rg.qTemp != (Buffer{}) {
			host[2*n+i] = int32(math.Float32bits(r.qTempScaleAt(rw.pos)))
		}
	}
	reps, classOf := r.raggedLayerClasses()
	rg.classOf = classOf
	lists := off // the classes' row lists start here: three n-entry lists per class
	for _, l := range reps {
		Ly := &r.layers[l]
		c := raggedClass{batched: b.At(4 * off), flash: b.At(4 * (off + n)), split: b.At(4 * (off + 2*n))}
		for i, p := range rg.hostPos {
			w := decodeNWin(Ly, p)
			switch arm := r.decodeAttnArm(Ly, l, w); {
			case arm == armBatched:
				host[off+c.nB] = int32(i)
				c.nB++
				c.maxWin = max(c.maxWin, w)
			case arm == armFlash && r.raggedFAReady():
				host[off+n+c.nF] = int32(i)
				c.nF++
			case arm == armSplitKV && r.raggedSKReady():
				host[off+2*n+c.nS] = int32(i)
				c.nS++
				c.maxWinS = max(c.maxWinS, w)
			default:
				c.other = append(c.other, i)
			}
		}
		rg.classes = append(rg.classes, c)
		off += 3 * n
	}
	// The split-KV rows' scores: chunks of as many rows as the margin's 512-row pass leaves room for beside the step's own scratch,
	// so a step never holds more than the margin reserves; with no room for one row, those rows take decode's own launches.
	for _, c := range rg.classes {
		if c.nS > 0 {
			rg.skStride = max(rg.skStride, c.maxWinS)
		}
	}
	if rg.skStride > 0 {
		used := r.PrefillScratchBytes(passRows) + scratchBufBytes(words)
		left := r.PrefillScratchBytes(prefillDefaultChunk) - used - 2*allocRoundSlack(1)
		perRow := int64(r.nH) * int64(rg.skStride+1) * 4
		rg.skRows = int(min(int64(n), max(0, left/perRow)))
		for ci := range rg.classes {
			if c := &rg.classes[ci]; rg.skRows == 0 && c.nS > 0 {
				split := lists + 3*n*ci + 2*n
				for k := range c.nS {
					c.other = append(c.other, int(host[split+k]))
				}
				c.nS = 0
			}
		}
		if rg.skRows > 0 {
			rg.skScores, rg.skInv = af(rg.skRows*r.nH*rg.skStride), af(rg.skRows*r.nH)
		}
	}
	if e := gpu.Upload(b, host); e != nil {
		return nil, e
	}
	return rg, nil
}

// stepAttnRows is prefillCore's per-sequence gap in a step (executor thread) for the rows from from on (those before are a prefill
// chunk's, which the pass attends as its own). With rag, rope and the KV store are one rope_kv_ragged launch over the rows, each into
// its own slot at its own position (an int8 cache's store then per row, as decode's). Then, by the layer's class, the rows whose
// decode takes attn_batched take one attn_ragged launch and those taking the flash-decode lane one fa_partial_ragged and one
// fa_combine_ragged launch per faMaxRows rows: each the arithmetic of the launch decode makes for that row, so each row is its
// decode's. The other rows, and every row without rag, run decode's own launches, decode's single-row buffers pointing at the row's
// views of the batched ones and the row's slot's tables bound; the previous buffers and binding are restored afterwards. Row strides
// are this layer's own (qDim for q and the context, kvDim for k and v), as bGemvB wrote them and bQuant reads the context.
func (r *cudaResident) stepAttnRows(Ly *cudaLayer, l int, rows []stepRow, from int, qBb, kBb, vBb, cctxB Buffer, rag *raggedRows) error {
	qB, kB, vB, cctx, tabSlot := r.qB, r.kB, r.vB, r.cctx, r.kvTabSlot
	defer func() { r.qB, r.kB, r.vB, r.cctx, r.kvTabSlot = qB, kB, vB, cctx, tabSlot }()
	n := len(rows) - from
	if n <= 0 {
		return nil
	}
	view := func(m int) {
		r.qB, r.cctx = qBb.At(m*Ly.qDim*4), cctxB.At(m*Ly.qDim*4)
		r.kB, r.vB = kBb.At(m*Ly.kvDim*4), vBb.At(m*Ly.kvDim*4)
		r.kvTabSlot = rows[m].slot // each row reads and writes through its own slot's tables
	}
	if rag == nil {
		for m := from; m < len(rows); m++ {
			view(m)
			if e := r.decodeAttnGap(Ly, l, rows[m].pos, rows[m].pos, r.qTempScaleAt(rows[m].pos)); e != nil {
				return e
			}
		}
		return nil
	}
	ropeNKV := Ly.nKV
	if Ly.kvShared && !g4KeepSharedKVStoreForTest {
		ropeNKV = 0
	}
	g := r.kvGroupOf(l)
	tabs, tabStride := r.kvTabDev.At(g*r.kvMaxBlocks()*4), kvGroups*r.kvMaxBlocks()
	qTemp := ArgNull()
	if rag.qTemp != (Buffer{}) {
		qTemp = Arg(rag.qTemp)
	}
	ropeN := r.nH*Ly.rhalf + ropeNKV*Ly.rhalf + ropeNKV*(Ly.hd-2*Ly.rhalf)
	q0, k0, v0, c0 := qBb.At(from*Ly.qDim*4), kBb.At(from*Ly.kvDim*4), vBb.At(from*Ly.kvDim*4), cctxB.At(from*Ly.qDim*4)
	if e := r.launch(r.bRopeKVRagged, LaunchConfig{GridX: uint32((ropeN + 255) / 256), GridY: uint32(n), GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
		Arg(q0), Arg(k0), Arg(v0), Arg(Ly.invF), Arg(r.kvK(l)), Arg(r.kvV(l)), Arg(tabs), gpu.ArgValue(int32(tabStride)),
		Arg(rag.slot), Arg(rag.pos), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(ropeNKV)), gpu.ArgValue(int32(Ly.hd)),
		gpu.ArgValue(int32(Ly.rhalf)), gpu.ArgValue(int32(n)), gpu.ArgValue(Ly.mscale), qTemp); e != nil {
		return e
	}
	if r.kvI8 {
		for m := from; m < len(rows); m++ {
			view(m)
			if e := r.kvStore8(l, r.kB, r.vB, ropeNKV, Ly.hd, rows[m].pos, 1); e != nil {
				return e
			}
		}
	}
	c := rag.classes[rag.classOf[l]]
	attnStride := tabStride
	if raggedPlantForTest == 4 {
		attnStride--
	}
	if c.nB > 0 {
		r.raggedAttnLaunches++
		if e := r.launch(r.bAttnRagged, LaunchConfig{GridX: uint32(r.nH), GridY: uint32(c.nB), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((c.maxWin + 128) * 4)},
			r.kvScaled(l, r.withCap(Arg(q0), Arg(r.kvK(l)), Arg(r.kvV(l)), Arg(tabs), gpu.ArgValue(int32(attnStride)), Arg(rag.slot), Arg(rag.pos),
				Arg(c.batched), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)), gpu.ArgValue(r.attnScale),
				gpu.ArgValue(Ly.window), gpu.ArgValue(int32(c.nB)), Arg(c0), r.sinkArg(l))...)...); e != nil {
			return e
		}
	}
	if c.nF > 0 {
		which := map[int]int{64: 0, 128: 1, 256: 2}[Ly.hd]
		G, U := r.nH/Ly.nKV, r.faSplit*faWarps
		for a := 0; a < c.nF; a += faMaxRows {
			k := min(faMaxRows, c.nF-a)
			r.raggedFALaunches++
			if e := r.launch(r.faRagged[which], LaunchConfig{GridX: uint32(Ly.nKV), GridY: uint32(r.faSplit), GridZ: uint32(k), BlockX: faWarps * 32, BlockY: 1, BlockZ: 1,
				SharedMemBytes: uint32((G*Ly.hd + faWarps*64) * 4)},
				r.withCap(append([]gpu.KernelArg{Arg(q0), Arg(r.kvK(l)), Arg(r.kvV(l)), Arg(tabs), gpu.ArgValue(int32(attnStride)), Arg(rag.slot), Arg(rag.pos),
					Arg(c.flash), gpu.ArgValue(int32(a)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(Ly.window), gpu.ArgValue(r.attnScale),
					gpu.ArgValue(int32(r.faSplit)), Arg(r.faBuf)}, r.kvScaled(l)...)...)...); e != nil {
				return e
			}
			if e := r.launch(r.faCombineRagged, LaunchConfig{GridX: uint32(r.nH), GridY: uint32(k), GridZ: 1, BlockX: uint32(Ly.hd), BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(3 * U * 4)},
				Arg(r.faBuf), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.hd)), gpu.ArgValue(int32(U)), gpu.ArgValue(int32(a)), Arg(c.flash), Arg(c0)); e != nil {
				return e
			}
		}
	}
	if c.nS > 0 {
		const dTile = 32 // decode's split-KV V-sum tile (splitKVAttnDecode)
		for a := 0; a < c.nS; a += rag.skRows {
			k := min(rag.skRows, c.nS-a)
			r.raggedSKLaunches++
			if e := r.launch(r.skScoresRagged, LaunchConfig{GridX: uint32(r.nH), GridY: uint32((c.maxWinS + 127) / 128), GridZ: uint32(k), BlockX: 128, BlockY: 1, BlockZ: 1,
				SharedMemBytes: uint32(Ly.hd * 4)},
				r.kvScaled(l, r.withCap(Arg(q0), Arg(r.kvK(l)), Arg(tabs), gpu.ArgValue(int32(attnStride)), Arg(rag.slot), Arg(rag.pos), Arg(c.split),
					gpu.ArgValue(int32(a)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)), gpu.ArgValue(Ly.window),
					gpu.ArgValue(r.attnScale), Arg(rag.skScores), gpu.ArgValue(int32(rag.skStride)))...)...); e != nil {
				return e
			}
			if e := r.launch(r.skSoftmaxRagged, LaunchConfig{GridX: uint32(r.nH), GridY: uint32(k), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 4},
				Arg(rag.skScores), Arg(rag.pos), Arg(c.split), gpu.ArgValue(int32(a)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(Ly.window),
				gpu.ArgValue(int32(rag.skStride)), Arg(rag.skInv), r.sinkArg(l)); e != nil {
				return e
			}
			if e := r.launch(r.skVsumRagged, LaunchConfig{GridX: uint32(r.nH), GridY: uint32((Ly.hd + dTile - 1) / dTile), GridZ: uint32(k), BlockX: dTile, BlockY: 1, BlockZ: 1},
				r.kvScaled(l, Arg(rag.skScores), Arg(r.kvV(l)), Arg(tabs), gpu.ArgValue(int32(attnStride)), Arg(rag.slot), Arg(rag.pos), Arg(c.split),
					gpu.ArgValue(int32(a)), Arg(rag.skInv), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
					gpu.ArgValue(Ly.window), gpu.ArgValue(int32(rag.skStride)), Arg(c0))...); e != nil {
				return e
			}
		}
	}
	for _, i := range c.other {
		view(from + i)
		if e := r.decodeAttn(Ly, l, rows[from+i].pos); e != nil {
			return e
		}
	}
	return nil
}

// stepDraw is ForwardSample's pick over the row's logits in r.logits (executor thread): Gumbel-max for a finite
// temperature, ForwardArgmax's argmax for one too small to be finite, exactly as ForwardSample chooses.
func (r *cudaResident) stepDraw(d *decoder.ResidentBatchDraw) (int, error) {
	invT := float32(1 / d.Temperature)
	if !math.IsInf(float64(invT), 0) {
		return r.gumbelPick(r.vocab, invT, d.Seed, d.Draw)
	}
	return r.argmaxPick(r.vocab)
}

// qTempScaleAt is the post-RoPE query scale at pos (Ministral 3, FeatAttnTemp; 1 for every other family) — launchToken's
// own expression, shared so a step row computes exactly its decode's.
func (r *cudaResident) qTempScaleAt(pos int) float32 {
	if r.attnTempBeta == 0 {
		return 1
	}
	return float32(1 + r.attnTempBeta*math.Log1p(math.Floor(float64(pos)/r.attnTempOrigMaxPos)))
}
