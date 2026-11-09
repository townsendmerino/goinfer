//go:build cuda

package cuda

import (
	"context"
	"fmt"
	"math"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// One decode token for each of several sequences, each on its own resident KV slot at its own position, in one step
// (docs/tasks/task-concurrency-2026-09.md). The step is prefillCore's batched layer stack in its multi-sequence mode
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
// position (a verify row run as a decode row), 3 runs a prefill chunk on the exact kernels (a chunk out of its pass's class).
// 0 in production. Set by tests only.
var raggedPlantForTest int

// raggedChunk is a ragged step's prefill chunk while its pass runs (StepRagged sets it, prefillCore reads it): the pass's
// first rows rows, at pos in slot, of a prompt promptLen long; exact when its own pass takes the exact kernels. split holds
// while the layer stack runs, when bGemvB and the attention give the chunk's rows its class.
type raggedChunk struct {
	rows, pos, slot, promptLen int
	exact, split               bool
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
		r.stepChunk = raggedChunk{rows: len(s.Embs), pos: s.Pos, slot: s.Slot, promptLen: s.Prefill.PromptLen, exact: raggedPlantForTest == 3}
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

// stepAttnRows is prefillCore's per-sequence gap in a step (executor thread): for each row, decode's single-row q/k/v
// and context buffers point at that row's views of the batched ones, the row's KV slot's tables are the ones read, and decodeAttnGap runs
// at the row's position — rope_kv into that slot, then decode's attention choice. The previous buffers and binding are
// restored afterwards. Row strides are this layer's own (qDim for q and the context, kvDim for k and v), as bGemvB wrote
// them and bQuant reads the context. Rows before from are a prefill chunk's, which the pass attends as its own.
func (r *cudaResident) stepAttnRows(Ly *cudaLayer, l int, rows []stepRow, from int, qBb, kBb, vBb, cctxB Buffer) error {
	qB, kB, vB, cctx, tabSlot := r.qB, r.kB, r.vB, r.cctx, r.kvTabSlot
	defer func() { r.qB, r.kB, r.vB, r.cctx, r.kvTabSlot = qB, kB, vB, cctx, tabSlot }()
	for m := from; m < len(rows); m++ {
		rw := rows[m]
		r.qB, r.cctx = qBb.At(m*Ly.qDim*4), cctxB.At(m*Ly.qDim*4)
		r.kB, r.vB = kBb.At(m*Ly.kvDim*4), vBb.At(m*Ly.kvDim*4)
		r.kvTabSlot = rw.slot // each row reads and writes through its own slot's tables
		if e := r.decodeAttnGap(Ly, l, rw.pos, rw.pos, r.qTempScaleAt(rw.pos)); e != nil {
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
