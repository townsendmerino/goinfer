//go:build cuda

package cuda

import (
	"context"
	"fmt"
	"math"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
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

// BatchStepRange reports the sequences a step serves: 2 up to the slot count, or none (hi = 0) for a model the step's
// scope excludes — MoE (C′ expert streaming included), MLA, the recurrent families, a model with no batched prefill,
// one slot. Those keep one generation at a time.
func (r *cudaResident) BatchStepRange() (lo, hi int) {
	if !r.prefillReady || r.moe || r.gemma4Moe || r.cacheExperts || r.isMLA || r.dnet != nil || r.KVSlots() < 2 {
		return 0, 0
	}
	if r.prefillStaticDecline() != nil {
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

// stepAttnRows is prefillCore's per-sequence gap in a step (executor thread): for each row, decode's single-row q/k/v
// and context buffers point at that row's views of the batched ones, the row's KV slot is bound, and decodeAttnGap runs
// at the row's position — rope_kv into that slot, then decode's attention choice. The previous buffers and binding are
// restored afterwards. Row strides are this layer's own (qDim for q and the context, kvDim for k and v), as bGemvB wrote
// them and bQuant reads the context.
func (r *cudaResident) stepAttnRows(Ly *cudaLayer, l int, rows []stepRow, qBb, kBb, vBb, cctxB Buffer) error {
	qB, kB, vB, cctx := r.qB, r.kB, r.vB, r.cctx
	kc, vc := r.kc, r.vc
	defer func() { r.qB, r.kB, r.vB, r.cctx, r.kc, r.vc = qB, kB, vB, cctx, kc, vc }()
	for m, rw := range rows {
		r.qB, r.cctx = qBb.At(m*Ly.qDim*4), cctxB.At(m*Ly.qDim*4)
		r.kB, r.vB = kBb.At(m*Ly.kvDim*4), vBb.At(m*Ly.kvDim*4)
		if len(r.kvSlotBufs) > 0 {
			r.kc, r.vc = r.kvSlotBufs[rw.slot].kc, r.kvSlotBufs[rw.slot].vc
		}
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
	if e := r.launch(r.fArg, onecfg(256, 256*4+256*4), Arg(r.logits),
		gpu.ArgValue(int32(r.vocab)), Arg(r.argIdx), Arg(r.argVal)); e != nil {
		return 0, e
	}
	if e := r.stream.Sync(); e != nil {
		return 0, e
	}
	out := make([]int32, 1)
	if e := gpu.Download(r.argIdx, out); e != nil {
		return 0, e
	}
	return int(out[0]), nil
}

// qTempScaleAt is the post-RoPE query scale at pos (Ministral 3, FeatAttnTemp; 1 for every other family) — launchToken's
// own expression, shared so a step row computes exactly its decode's.
func (r *cudaResident) qTempScaleAt(pos int) float32 {
	if r.attnTempBeta == 0 {
		return 1
	}
	return float32(1 + r.attnTempBeta*math.Log1p(math.Floor(float64(pos)/r.attnTempOrigMaxPos)))
}
