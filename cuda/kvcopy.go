//go:build cuda

package cuda

import (
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/decoder"
)

var (
	_ decoder.ResidentKVCopier       = (*cudaResident)(nil)
	_ decoder.ResidentReusableKVSlot = (*cudaResident)(nil)
)

// kvCopyF16 is whether CopyKV serves a load whose KV cache is f16. False declines every such copy, so an f16 load shares
// nothing across conversations and an f32 load is unaffected: the one switch for holding cross-conversation sharing to
// f32 (docs/measurements/kv2-cuda-2026-10-15.md, G0-C).
var kvCopyF16 = true

// CopyKV implements decoder.ResidentKVCopier: rows [start, end) of every layer's KV, from slot src to slot dst, so dst
// continues at position end. It is one executor job, so it is ordered on the one stream after everything posted before
// it (a busy source's prefill included) and runs between MC3 steps, never inside one. A busy source's decode writes only
// rows past its prompt, which the decoder never asks for. The job synchronises the stream before it returns, so the
// caller's timing covers the copy.
//
//   - A full-attention or MLA latent layer's blocks for rows [start, end) are shared: dst's table names src's blocks, and
//     whichever slot writes into one later takes a copy first (kvShare, kvMap).
//   - A layer that reads another layer's KV copies nothing: it aliases its source.
//   - A windowed layer copies the rows dst attends to from end and, behind them, as much of the rollback (kvRollback) as
//     src still holds: src's rows [max(src's base, end-(window-1)-rollback), end), and dst's base becomes that first row.
//     dst then holds what a slot that wrote the rows itself holds after its base moved. Rows below start that dst already
//     held are rewritten with src's bytes of the same published blocks.
//
// Rows are read and written through each slot's tables (kvpool.go).
//
// A *decoder.KVCopyDecline, before anything is written, refuses a range the copy cannot serve; its check of src's base runs
// inside the job, so a compaction of a running source cannot fall between the check and the read. Each copy and each
// windowed decline is logged with the slots' bases and the rollback budget, so a served decline can be traced to it.
func (r *cudaResident) CopyKV(src, dst, start, end int) error {
	n := r.KVSlots()
	switch {
	case src < 0 || src >= n || dst < 0 || dst >= n || src == dst:
		return fmt.Errorf("cuda: CopyKV slots %d -> %d of %d", src, dst, n)
	case start < 0 || start > end || start%decoder.KVBlockRows != 0 || end%decoder.KVBlockRows != 0:
		return fmt.Errorf("cuda: CopyKV rows [%d, %d) are not whole %d-row blocks", start, end, decoder.KVBlockRows)
	case end > r.ctxCap:
		return &decoder.KVCopyDecline{Reason: fmt.Sprintf("rows up to %d are past the %d-position cache", end, r.ctxCap)}
	case r.recurrent():
		return &decoder.KVCopyDecline{Reason: "recurrent state is not part of a KV slot"}
	case r.kvF16 && !kvCopyF16:
		return &decoder.KVCopyDecline{Reason: "copies of an f16 KV cache are off (kvCopyF16)"}
	}
	if start == end {
		return nil
	}
	return r.do(func() error { return r.copyKVSlots(src, dst, start, end) })
}

// copyKVSlots is CopyKV's job. Executor thread only.
func (r *cudaResident) copyKVSlots(src, dst, start, end int) error {
	if r.kvReleased(src) {
		return &decoder.KVCopyDecline{Reason: fmt.Sprintf("slot %d's blocks were released to make room", src)}
	}
	if r.kvReleased(dst) {
		start, r.kvEvicted[dst] = 0, false // dst holds nothing below start any more: take the whole range
	}
	lo := 0 // the first row a windowed layer of dst holds afterwards
	if r.kvWin {
		if !kvWindowReusable(r.kvBases[src], end, r.kvWindow) {
			return &decoder.KVCopyDecline{Reason: fmt.Sprintf("slot %d's windowed layers hold rows from %d; continuing at %d needs them from %d; %s",
				src, r.kvBases[src], end, end-(r.kvWindow-1), r.kvBudgetText())}
		}
		lo = max(0, r.kvBases[src], end-(r.kvWindow-1)-r.kvRollback())
	}
	if e := r.kvShare(src, dst, start, end); e != nil {
		return e
	}
	if r.kvWin {
		r.kvBases[dst] = lo
		if e := r.kvMap(dst, kvGroupWin, lo, end); e != nil {
			return e
		}
	}
	if e := r.kvFlushTabs(dst); e != nil {
		return e
	}
	for l := range r.layers {
		Ly := &r.layers[l]
		if Ly.kvShared || Ly.noKV() || !Ly.kvWin {
			continue // a full-group layer's rows are shared by kvShare above
		}
		g, from := kvGroupWin, lo
		row := Ly.kvDim * r.kvElem(l) // bytes
		srcIDs := r.kvTabs[src].ids[g]
		err := r.kvRowRuns(dst, g, from, end, func(p, n, dstRow int) error {
			// dst's run may span several blocks; src's rows are contiguous only within one of its blocks. A source block never
			// mapped reads the null block's zeros: the rows were never written, as a caller copying them should know.
			for q := p; q < p+n; {
				m := min(p+n, (q/kvBlock+1)*kvBlock) - q
				srcRow := int(srcIDs[q/kvBlock])*kvBlock + q%kvBlock
				for _, b := range []Buffer{r.kc[l], r.vc[l]} {
					if b.Len() == 0 {
						continue // an MLA latent layer has no V
					}
					if e := r.copyF32(b.At(srcRow*row), b.At((dstRow+q-p)*row), m*row/4); e != nil {
						return e
					}
				}
				q += m
			}
			return nil
		})
		if err != nil {
			return err
		}
	}
	if err := r.stream.Sync(); err != nil {
		return err
	}
	if r.kvWin {
		fmt.Fprintf(os.Stderr, "cuda: kv copy: slot %d -> %d, rows [%d, %d); windowed layers: slot %d's base %d, slot %d's base %d after the copy; %s\n",
			src, dst, start, end, src, r.kvBases[src], dst, lo, r.kvBudgetText())
	}
	fmt.Fprintf(os.Stderr, "cuda: kv copy: slot %d -> %d shares rows [%d, %d), %d blocks; %d of %d full-pool blocks in use\n",
		src, dst, start, end, (end-start)/kvBlock, r.kvPools[kvGroupFull].inUse(), r.kvPools[kvGroupFull].blocks-1)
	return nil
}

// kvBudgetText states the windowed rollback budget: the planned figure and what kvGrantRollback added from free VRAM. It
// has no parentheses, so it survives inside the decline reason serve prints in parentheses.
func (r *cudaResident) kvBudgetText() string {
	planned := kvWindowSlack - kvWindowStep
	return fmt.Sprintf("rollback budget %d = %d planned + %d from free VRAM", r.kvRollback(), planned, r.kvRollback()-planned)
}

// ReusableKVSlot implements decoder.ResidentReusableKVSlot: ReusableKV's answer for any slot, bound or not.
func (r *cudaResident) ReusableKVSlot(slot, prefix int) int {
	if prefix <= 0 {
		return prefix
	}
	if slot < 0 || slot >= r.KVSlots() {
		return 0
	}
	out := prefix
	// the bases and the released marks are executor state: read them where they are written
	if err := r.do(func() error {
		if r.kvReleased(slot) {
			out = 0
			return nil
		}
		if r.kvWin && !kvWindowReusable(r.kvBases[slot], prefix, r.kvWindow) {
			out = 0
			fmt.Fprintf(os.Stderr, "cuda: kv copy: slot %d's windowed layers hold rows from %d; continuing at %d needs them from %d; %s\n",
				slot, r.kvBases[slot], prefix, prefix-(r.kvWindow-1), r.kvBudgetText())
		}
		return nil
	}); err != nil {
		return 0
	}
	return out
}
