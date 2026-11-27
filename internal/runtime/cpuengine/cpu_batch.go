package cpuengine

import (
	"fmt"

	"github.com/townsendmerino/aikit/linalg"
)

// CPUBatchCacheEligible is decodeMultiStep's per-cache scope: append-forever KV at any precision (no sliding-window ring) with no
// adapter, tree mask or manual position. A token on any other cache runs production's own forward.
func CPUBatchCacheEligible(c *Cache) error {
	if c == nil || c.Scr == nil || c.LocalAny || len(c.LoRALayers) > 0 || c.TreeMask != nil || c.ManualPos {
		return fmt.Errorf("cpu batch: not a plain append-forever cache")
	}
	return nil
}

// CPUBatchFusedW4A8 lets decodeMultiStep run an int4 layer's q‖k‖v, and its gate‖up, each as one batched W4A8 call over
// the B rows instead of one Matmul per projection. The kernel (linalg.MatmulBTBatchInto) is numerically identical to
// calling MatmulBTW4A8Into once per op, so this changes only how many fork/joins a step pays. A package variable, not
// an environment read: the measurement flips it in-process to
// interleave its arms.
var CPUBatchFusedW4A8 = true

// W4A8FusedOps fills ops with one linalg.BatchOp per weight for a fused batched W4A8 call and reports whether the group
// qualifies: fusion on, every weight int4 with its canonical bytes present (WMW4A8Op passes no split-half layout), one
// weight-group size, one K, and one activation-quant group. It returns that group size and activation-quant group.
// Anything else keeps the per-projection Matmul calls.
func W4A8FusedOps(ops []linalg.BatchOp, ws []*linalg.WeightMat, dsts [][]float32) (group, actGroup int, ok bool) {
	if !CPUBatchFusedW4A8 {
		return 0, 0, false
	}
	for i, w := range ws {
		_, _, g, canon := w.Int4F16()
		if !IsW4A8(w) || !canon || w.Cols() != ws[0].Cols() || w.ActQuantGroup() != ws[0].ActQuantGroup() {
			return 0, 0, false
		}
		var gi int
		ops[i], gi = WMW4A8Op(w, dsts[i])
		if gi != g || (i > 0 && g != group) {
			return 0, 0, false
		}
		group = g
	}
	return group, ws[0].ActQuantGroup(), true
}

// DecodeMultiStep is the batched CPU decode's step: one forward carrying B independent sequences, each with its own KV
// cache at its own position, one token each. The projections, o-proj, MLP and LM head run as M = B matmuls through the
// kernels forwardN uses (bit-identical to M = 1 row for row, by forwardN's own contract). Attention runs per sequence
// over its own cache, exactly as attendToken's default (f32 KV, append-forever) case does at decode. Every row's
// logits are bit-identical to that sequence's own m.Forward (TestMC2_decodeMultiStepBitIdentical,
// TestCPUBatch_everyEligibleFixtureBitIdentical).
func (m *Engine) DecodeMultiStep(ids []int, caches []*Cache) ([][]float32, error) {
	for i, c := range caches {
		if err := CPUBatchCacheEligible(c); err != nil {
			return nil, fmt.Errorf("cache %d: %w", i, err)
		}
	}
	prog := m.Program()
	hidden, vocab := m.W.Arch.HiddenDim, m.W.Arch.VocabSize
	h := make([]float32, len(ids)*hidden)
	for b, id := range ids {
		hb := h[b*hidden : (b+1)*hidden]
		m.W.Embed.Row(id, hb)
		if s := prog.Embed.Scale; s != 0 && s != 1 {
			for j := range hb {
				hb[j] *= float32(s)
			}
		}
	}
	h, err := m.WalkMulti(prog, h, caches)
	if err != nil {
		return nil, err
	}
	flat := m.LMHeadN(h, len(ids))
	out := make([][]float32, len(ids))
	for b := range out {
		out[b] = flat[b*vocab : (b+1)*vocab]
	}
	return out, nil
}
